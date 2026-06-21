"""ONNX Runtime detector for a YOLOv8-Pose model (document corner keypoints).

The model detects a document and predicts 4 keypoint corners (TL, TR, BR, BL).
From those corners we can:
  - draw the detected quad (same as other detectors)
  - compute a perspective warp to "flatten" the document

Model layout (exported YOLOv8-pose):
    input  : images   [1, 3, 1024, 1024]  (RGB, 0-1, letterboxed)
    output : output0  [1, 4+1+4*3, N]  ->  [cx, cy, w, h, score, kpt*12...]
    kpt_shape = [4, 3]  -> 4 keypoints, each with (x, y, visibility/conf)
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection, Detector

INPUT_SIZE = 1024  # this pose model uses 1024x1024


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [TL, TR, BR, BL]."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # TL (min x+y)
    ordered[1] = pts[np.argmin(d)]   # TR (min y-x)
    ordered[2] = pts[np.argmax(s)]   # BR (max x+y)
    ordered[3] = pts[np.argmax(d)]   # BL (max y-x)
    return ordered


class OnnxPoseDetector(Detector):
    """Detect document corners via a YOLOv8-Pose ONNX model."""

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        label: str = "A4",
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is not installed. Run `uv add onnxruntime`.") from exc

        self.session = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        # Determine input size from model (some are 640, this one is 1024).
        shape = self.session.get_inputs()[0].shape
        self.input_size = int(shape[2]) if shape[2] is not None else INPUT_SIZE
        self.conf = conf
        self.iou = iou
        self.label = label

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
        img, scale, padx, pady = self._letterbox(frame)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]

        out = self.session.run(None, {self.input_name: blob})[0]  # (1, 17, N)
        pred = np.squeeze(out, 0).T  # (N, 17)

        # Decode: [cx, cy, w, h, score, kpt0_x, kpt0_y, kpt0_conf, ... x4]
        scores = pred[:, 4]
        keep = scores > self.conf
        if not np.any(keep):
            return []
        pred = pred[keep]
        scores = pred[:, 4]
        boxes_xywh = pred[:, :4]

        # Standard NMS on axis-aligned boxes.
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_for_nms = np.stack([x1, y1, boxes_xywh[:, 2], boxes_xywh[:, 3]], axis=1)
        idxs = cv2.dnn.NMSBoxes(boxes_for_nms.tolist(), scores.tolist(), self.conf, self.iou)
        if len(idxs) == 0:
            return []
        idxs = np.asarray(idxs).flatten()

        detections: list[Detection] = []
        for i in idxs:
            kpts_raw = pred[i, 5:].reshape(-1, 3)  # (4, 3) -> x, y, conf per kpt
            kpts_xy = kpts_raw[:, :2]  # in letterbox space

            # Undo letterbox.
            kpts_xy = (kpts_xy - np.array([padx, pady], np.float32)) / scale

            corners = _order_corners(kpts_xy)
            detections.append(
                Detection(quad=corners, score=float(scores[i]), label=self.label)
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections


def warp_perspective(frame: np.ndarray, quad: np.ndarray,
                     out_w: int = 800, out_h: int = 1100) -> np.ndarray:
    """Warp the region defined by quad (TL, TR, BR, BL) to a flat rectangle."""
    src = quad.astype(np.float32)
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (out_w, out_h))
