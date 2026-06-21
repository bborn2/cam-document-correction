"""ONNX Runtime detector for a YOLOv8-Seg model, with vertex extraction.

A segmentation model outputs a pixel mask per object rather than a box, so the
A4 *vertices* are computed from the mask outline:
    mask -> largest contour -> approxPolyDP -> 4 corner points.

Model layout (exported YOLOv8-seg):
    input  : images   [1, 3, 640, 640]
    output0: [1, 4+nc+32, 8400]  -> [cx,cy,w,h, <nc scores>, <32 mask coeffs>]
    output1: [1, 32, 160, 160]   -> 32 prototype masks
A detection's final mask = sigmoid(coeffs @ protos), cropped to its box.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection, Detector

INPUT_SIZE = 640
NUM_MASKS = 32


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise from the top-left (same convention as cv_detector)."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left
    return ordered


def _quad_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (4 ordered corners, full contour) from a binary mask, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1:
        return None

    peri = cv2.arcLength(cnt, True)
    quad = None
    for eps in (0.02, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            quad = approx
            break
    if quad is None:
        # Fall back to the rotated min-area rectangle's 4 corners.
        quad = cv2.boxPoints(cv2.minAreaRect(cnt))

    return _order_corners(quad), cnt.reshape(-1, 2).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class OnnxSegDetector(Detector):
    """Run a YOLOv8-Seg ONNX model via onnxruntime and extract A4 vertices."""

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        label: str = "A4",
        mask_thresh: float = 0.5,
        imgsz: int | None = None,
    ) -> None:
        try:
            import onnxruntime as ort  # noqa: PLC0415 (lazy, optional dep)
        except ImportError as exc:  # pragma: no cover
            raise ImportError("onnxruntime is not installed. Run `uv add onnxruntime`.") from exc

        self.session = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        # Decide the inference input size, in priority order:
        #   1. explicit imgsz argument (e.g. --imgsz 320)
        #   2. fixed size baked into the model's input shape
        #   3. the model's exported `imgsz` metadata
        #   4. fallback INPUT_SIZE
        shape = self.session.get_inputs()[0].shape
        fixed = shape[2] if isinstance(shape[2], int) else None
        if imgsz is not None:
            self.input_size = int(imgsz)
        elif fixed is not None:
            self.input_size = int(fixed)
        else:
            meta = self.session.get_modelmeta().custom_metadata_map
            try:
                # metadata imgsz looks like "[1024, 1024]"
                self.input_size = int(meta.get("imgsz", "").strip("[] ").split(",")[0])
            except (ValueError, IndexError):
                self.input_size = INPUT_SIZE

        # Models with a dynamic input must receive a multiple of the stride (32).
        self.input_size = max(32, (self.input_size // 32) * 32)

        self.conf = conf
        self.iou = iou
        self.label = label
        self.mask_thresh = mask_thresh

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = frame.shape[:2]
        sz = self.input_size
        scale = min(sz / w, sz / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        padx, pady = (sz - nw) // 2, (sz - nh) // 2
        canvas[pady:pady + nh, padx:padx + nw] = resized
        return canvas, scale, padx, pady

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h0, w0 = frame.shape[:2]
        img, scale, padx, pady = self._letterbox(frame)

        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]

        out0, out1 = self.session.run(None, {self.input_name: blob})
        preds = np.squeeze(out0, 0).T          # (8400, 4+nc+32)
        protos = np.squeeze(out1, 0)           # (32, 160, 160)

        nc = preds.shape[1] - 4 - NUM_MASKS
        boxes_xywh = preds[:, :4]
        cls_scores = preds[:, 4:4 + nc]
        coeffs = preds[:, 4 + nc:]             # (8400, 32)

        scores = cls_scores.max(axis=1)
        keep = scores > self.conf
        if not np.any(keep):
            return []
        boxes_xywh, scores, coeffs = boxes_xywh[keep], scores[keep], coeffs[keep]

        # xywh -> xyxy in the 640 letterboxed space for NMS.
        xy = boxes_xywh[:, :2]
        wh = boxes_xywh[:, 2:4]
        x1y1 = xy - wh / 2
        boxes_xyxy = np.concatenate([x1y1, wh], axis=1)  # x,y,w,h for NMSBoxes
        idxs = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), self.conf, self.iou)
        if len(idxs) == 0:
            return []
        idxs = np.asarray(idxs).flatten()

        mh, mw = protos.shape[1], protos.shape[2]       # 160, 160
        protos_flat = protos.reshape(NUM_MASKS, -1)     # (32, 160*160)

        detections: list[Detection] = []
        sz = self.input_size
        for i in idxs:
            # Build the instance mask at proto resolution, then upscale to input.
            m = _sigmoid(coeffs[i] @ protos_flat).reshape(mh, mw)
            m = cv2.resize(m, (sz, sz), interpolation=cv2.INTER_LINEAR)
            bin_mask = (m > self.mask_thresh).astype(np.uint8)

            # Restrict to this detection's box to avoid leaking onto other objects.
            cx, cy, bw, bh = boxes_xywh[i]
            x1 = int(max(0, cx - bw / 2)); y1 = int(max(0, cy - bh / 2))
            x2 = int(min(sz, cx + bw / 2)); y2 = int(min(sz, cy + bh / 2))
            box_only = np.zeros_like(bin_mask)
            box_only[y1:y2, x1:x2] = bin_mask[y1:y2, x1:x2]

            res = _quad_from_mask(box_only)
            if res is None:
                continue
            quad, contour = res

            # Map 640-letterbox coords back to the original frame.
            def unpad(pts: np.ndarray) -> np.ndarray:
                pts = (pts - np.array([padx, pady], np.float32)) / scale
                pts[:, 0] = np.clip(pts[:, 0], 0, w0 - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h0 - 1)
                return pts

            quad = unpad(quad)
            polygon = unpad(contour)
            detections.append(
                Detection(quad=quad, score=float(scores[i]), label=self.label, polygon=polygon)
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
