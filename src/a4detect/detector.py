"""Detector interface and shared data structures.

Both the traditional-CV detector and the (future) YOLO detector return the same
``Detection`` objects, so the main loop and drawing code never depend on which
detector is actually running. To plug in a trained model later, just implement a
new ``Detector`` subclass that returns ``Detection`` objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """A single detected object.

    Attributes:
        quad: (4, 2) float array of the four corner vertices, in pixel
            coordinates, ordered clockwise from top-left. This is the OBB-style
            box / detected vertices used for highlighting.
        score: Confidence / shape-match score in [0, 1].
        label: Class label, e.g. "A4".
        bbox: Axis-aligned (x, y, w, h) bounding box, derived from ``quad``.
        mask: Optional (H, W) uint8 mask (used by segmentation models). May be None.
        polygon: Optional (N, 2) float array of the full segmentation outline
            (more than 4 points). Used to fill the exact paper shape; ``quad`` is
            still the 4 corner vertices extracted from it.
    """

    quad: np.ndarray
    score: float = 1.0
    label: str = "A4"
    bbox: tuple[int, int, int, int] = field(default=(0, 0, 0, 0))
    mask: np.ndarray | None = None
    polygon: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.quad = np.asarray(self.quad, dtype=np.float32).reshape(4, 2)
        if self.bbox == (0, 0, 0, 0):
            xs = self.quad[:, 0]
            ys = self.quad[:, 1]
            x, y = float(xs.min()), float(ys.min())
            w, h = float(xs.max()) - x, float(ys.max()) - y
            self.bbox = (int(x), int(y), int(w), int(h))


class Detector(ABC):
    """Abstract base class for all A4 detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect A4 sheets in a BGR frame and return a list of detections."""
        raise NotImplementedError
