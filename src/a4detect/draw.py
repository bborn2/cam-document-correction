"""Drawing utilities: highlight detections on a frame."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

HIGHLIGHT_COLOR = (0, 255, 0)   # BGR green
VERTEX_COLOR = (0, 0, 255)      # BGR red
TEXT_COLOR = (0, 0, 0)
FILL_ALPHA = 0.25               # transparency of the filled overlay


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw the filled shape, the outline, the corner vertices, and labels (in place)."""
    overlay = frame.copy()

    for det in detections:
        # Fill the exact segmentation polygon if available, else the quad.
        if det.polygon is not None and len(det.polygon) >= 3:
            cv2.fillPoly(overlay, [det.polygon.astype(np.int32)], HIGHLIGHT_COLOR)
        else:
            cv2.fillPoly(overlay, [det.quad.astype(np.int32)], HIGHLIGHT_COLOR)

        # Optional raw mask (if a detector set one).
        if det.mask is not None:
            overlay[det.mask.astype(bool)] = HIGHLIGHT_COLOR

    cv2.addWeighted(overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0, frame)

    for det in detections:
        quad = det.quad.astype(np.int32)

        # Outline through the 4 corner vertices.
        cv2.polylines(frame, [quad], isClosed=True, color=HIGHLIGHT_COLOR, thickness=3)

        # Draw + number the vertices so the detected corners are explicit.
        for idx, (vx, vy) in enumerate(quad):
            cv2.circle(frame, (int(vx), int(vy)), 6, VERTEX_COLOR, -1)
            cv2.circle(frame, (int(vx), int(vy)), 6, (255, 255, 255), 1)
            cv2.putText(
                frame, str(idx + 1), (int(vx) + 8, int(vy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, VERTEX_COLOR, 2, cv2.LINE_AA,
            )

        label = f"{det.label} {det.score:.2f}"
        x, y = quad[0]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), HIGHLIGHT_COLOR, -1)
        cv2.putText(
            frame, label, (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA,
        )

    return frame
