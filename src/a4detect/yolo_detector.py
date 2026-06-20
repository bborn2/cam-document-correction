"""Placeholder deep-learning detector (YOLOv8-Seg / OBB).

This is NOT used by default. Official YOLOv8 weights cannot detect "A4 paper"
(OBB weights are DOTA aerial classes; Seg weights are COCO 80 classes with no
"paper" class). To use this path you must first train your own model on an A4
dataset, then run:

    uv sync --extra yolo
    uv run a4detect --detector yolo --weights path/to/best.pt

Because the ``ultralytics`` import is done lazily inside ``__init__``, the rest
of the project runs with zero torch dependency until you opt in.
"""

from __future__ import annotations

import numpy as np

from .detector import Detection, Detector


class YoloSegDetector(Detector):
    """Wrap a trained YOLOv8 segmentation/OBB model behind the Detector API."""

    def __init__(self, weights_path: str, conf: float = 0.25) -> None:
        try:
            from ultralytics import YOLO  # noqa: PLC0415 (lazy, optional dep)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is not installed. Run `uv sync --extra yolo` to use "
                "the YOLO detector."
            ) from exc

        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections: list[Detection] = []

        for result in results:
            names = result.names

            # --- OBB models -------------------------------------------------
            obb = getattr(result, "obb", None)
            if obb is not None and obb.xyxyxyxy is not None:
                polys = obb.xyxyxyxy.cpu().numpy()        # (N, 4, 2)
                confs = obb.conf.cpu().numpy()
                clss = obb.cls.cpu().numpy().astype(int)
                for quad, score, cls in zip(polys, confs, clss):
                    detections.append(
                        Detection(quad=quad, score=float(score), label=names.get(cls, str(cls)))
                    )
                continue

            # --- Segmentation models ---------------------------------------
            masks = getattr(result, "masks", None)
            boxes = getattr(result, "boxes", None)
            if masks is not None and boxes is not None:
                polys = masks.xy  # list of (P, 2) polygon point arrays
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                for poly, score, cls in zip(polys, confs, clss):
                    poly = np.asarray(poly, dtype=np.float32)
                    if poly.shape[0] < 3:
                        continue
                    # Reduce the polygon to a 4-point rotated box for highlighting.
                    import cv2  # local import keeps top-level torch-free

                    rect = cv2.minAreaRect(poly)
                    quad = cv2.boxPoints(rect)
                    detections.append(
                        Detection(quad=quad, score=float(score), label=names.get(cls, str(cls)))
                    )

        return detections
