"""ONNX Runtime detector for a YOLOv8-OBB model (no torch needed).

The exported model has:
    input  : images  [1, 3, 640, 640]  (RGB, 0-1, letterboxed)
    output : output0 [1, 4+nc+1, 8400]  -> [cx, cy, w, h, <nc class scores>, angle]

For this A4 model nc=1 (class 'document'), so each of the 8400 candidates is
[cx, cy, w, h, score, angle] in the 640x640 letterboxed space. We decode, run
rotated-box NMS (cv2.dnn.NMSBoxesRotated), map boxes back to the original frame,
and emit unified Detection objects with rotated quads.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection, Detector

INPUT_SIZE = 640


class OnnxObbDetector(Detector):
    """Run a YOLOv8-OBB ONNX model via onnxruntime on CPU."""

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        label: str = "A4",
    ) -> None:
        try:
            import onnxruntime as ort  # noqa: PLC0415 (lazy, optional dep)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "onnxruntime is not installed. Run `uv add onnxruntime`."
            ) from exc

        self.session = ort.InferenceSession(
            weights_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.iou = iou
        self.label = label

    # --- pre/post processing ------------------------------------------------
    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Resize keeping aspect ratio, pad to INPUT_SIZE. Returns (img, scale, padx, pady)."""
        h, w = frame.shape[:2]
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padx, pady = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
        canvas[pady:pady + nh, padx:padx + nw] = resized
        return canvas, scale, padx, pady

    def detect(self, frame: np.ndarray) -> list[Detection]:
        img, scale, padx, pady = self._letterbox(frame)

        # BGR->RGB, HWC->CHW, normalize, add batch dim.
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]  # (1,3,640,640)

        out = self.session.run(None, {self.input_name: blob})[0]  # (1, 6, 8400)
        pred = np.squeeze(out, 0).T  # (8400, 6) -> [cx, cy, w, h, score, angle]

        boxes_xywh = pred[:, :4]
        scores = pred[:, 4]
        angles = pred[:, 5]  # radians

        keep = scores > self.conf
        if not np.any(keep):
            return []
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        angles = angles[keep]

        # Rotated-box NMS. cv2 wants ((cx,cy),(w,h),angle_degrees).
        rot_rects = [
            ((float(cx), float(cy)), (float(bw), float(bh)), float(np.degrees(a)))
            for (cx, cy, bw, bh), a in zip(boxes_xywh, angles)
        ]
        indices = cv2.dnn.NMSBoxesRotated(
            rot_rects, scores.tolist(), self.conf, self.iou
        )
        if len(indices) == 0:
            return []
        indices = np.asarray(indices).flatten()

        detections: list[Detection] = []
        for i in indices:
            quad = cv2.boxPoints(rot_rects[i])  # (4,2) in letterboxed space
            # Undo letterbox: subtract pad, divide by scale -> original frame coords.
            quad = (quad - np.array([padx, pady], dtype=np.float32)) / scale
            detections.append(
                Detection(quad=quad, score=float(scores[i]), label=self.label)
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
