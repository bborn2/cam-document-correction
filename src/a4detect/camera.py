"""Webcam capture wrapper (Windows-friendly)."""

from __future__ import annotations

import cv2
import numpy as np


class Camera:
    """Thin wrapper around ``cv2.VideoCapture`` usable as a context manager.

    Uses the DirectShow backend on Windows, which opens faster and is more
    reliable than the default MSMF backend for many webcams.
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 max_res: bool = False) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.max_res = max_res
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> "Camera":
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fall back to the default backend if DSHOW fails.
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.index}. "
                "Check the device is connected and not in use."
            )

        if self.max_res:
            # Request an absurdly large frame; the driver clamps to the highest
            # resolution the camera actually supports.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Read back what the camera actually gave us.
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap = cap
        return self

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            raise RuntimeError("Camera is not opened. Call open() first.")
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.release()
