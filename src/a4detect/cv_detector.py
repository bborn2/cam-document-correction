"""Traditional computer-vision detector for A4 paper.

Robust multi-strategy pipeline (per frame):
  A. Brightness segmentation  -- white paper is usually brighter than the desk,
     so Otsu + adaptive thresholding segments the sheet directly. This is the
     most reliable cue and works even when edges are weak.
  B. Edge detection (Canny)   -- catches paper whose brightness is close to the
     background but whose border is crisp.
  Candidates from both are merged, approximated to quads, and scored by how
  rectangular and how A4-like (aspect ratio ~ 1.414) they are. Scoring is soft:
  a slightly-off ratio lowers the score instead of rejecting the candidate,
  which matters because perspective makes a tilted A4 look non-1.414.

Needs no training, runs in real time, and produces a rotated quad (OBB).
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection, Detector

# --- Tunable parameters (adjust on site if needed) -------------------------
BLUR_KSIZE = 5
CANNY_LOW = 30
CANNY_HIGH = 120
CLOSE_KSIZE = 7          # bigger close kernel bridges more edge gaps
MIN_AREA_RATIO = 0.01    # contour area must be >= 1% of the frame
MAX_AREA_RATIO = 0.98    # ... and <= 98% (reject the whole-frame border)
APPROX_EPS_RATIOS = (0.02, 0.04, 0.06)  # try several epsilons, accept first quad
A4_RATIO = 1.4142        # long / short side of A4
RATIO_TOLERANCE = 0.6    # accept aspect ratios within +/- 60% of A4_RATIO (loose for tilt)
MIN_FILL = 0.55          # quad must cover >= 55% of its minAreaRect (reject blobs)
MIN_SCORE = 0.30         # final score gate
# ---------------------------------------------------------------------------


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise starting from the top-left corner."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # top-left  (min x+y)
    ordered[2] = pts[np.argmax(s)]      # bottom-right (max x+y)
    ordered[1] = pts[np.argmin(diff)]   # top-right (min y-x)
    ordered[3] = pts[np.argmax(diff)]   # bottom-left (max y-x)
    return ordered


def _approx_quad(cnt: np.ndarray) -> np.ndarray | None:
    """Approximate a contour to a convex 4-gon, trying several epsilons."""
    peri = cv2.arcLength(cnt, True)
    for eps in APPROX_EPS_RATIOS:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx
    return None


def _build_masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary mask from brightness, edge map) for candidate extraction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

    # A. Brightness segmentation: Otsu picks a global bright/dark split.
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Adaptive handles uneven lighting; combine both.
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5
    )
    mask = cv2.bitwise_or(otsu, adapt)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KSIZE, CLOSE_KSIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    # B. Edge map.
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edges = cv2.dilate(edges, k, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
    return mask, edges


class CvA4Detector(Detector):
    """Detect A4 sheets via brightness + edge contour analysis."""

    def __init__(self) -> None:
        # Holds the most recent intermediate images for the debug view.
        self.debug_mask: np.ndarray | None = None
        self.debug_edges: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        frame_area = float(h * w)

        mask, edges = _build_masks(frame)
        self.debug_mask, self.debug_edges = mask, edges

        candidates: list[np.ndarray] = []
        for src in (mask, edges):
            contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            candidates.extend(contours)

        detections: list[Detection] = []
        for cnt in candidates:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_RATIO * frame_area or area > MAX_AREA_RATIO * frame_area:
                continue

            approx = _approx_quad(cnt)
            if approx is None:
                continue

            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            if rw < 1 or rh < 1:
                continue
            long_side, short_side = max(rw, rh), min(rw, rh)
            ratio = long_side / short_side

            rect_area = rw * rh
            fill = float(area / rect_area) if rect_area > 0 else 0.0
            if fill < MIN_FILL:
                continue

            # Soft aspect-ratio score: 1.0 at exactly A4, decaying with deviation.
            ratio_dev = abs(ratio - A4_RATIO) / (RATIO_TOLERANCE * A4_RATIO)
            if ratio_dev > 1.0:
                continue
            ratio_score = 1.0 - ratio_dev
            score = float(np.clip(0.5 * ratio_score + 0.5 * fill, 0.0, 1.0))
            if score < MIN_SCORE:
                continue

            quad = _order_corners(approx)
            detections.append(Detection(quad=quad, score=score, label="A4"))

        detections = _dedupe(detections)
        detections.sort(key=lambda d: (d.score, d.bbox[2] * d.bbox[3]), reverse=True)
        return detections


def _dedupe(dets: list[Detection], iou_thresh: float = 0.6) -> list[Detection]:
    """Drop near-duplicate boxes (the two masks often find the same sheet)."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: x.score, reverse=True):
        if all(_bbox_iou(d.bbox, k.bbox) < iou_thresh for k in kept):
            kept.append(d)
    return kept


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
