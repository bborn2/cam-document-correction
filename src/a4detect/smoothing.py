"""Temporal smoothing of detections to stop the highlight from jittering.

Per-frame detection is noisy: the same static sheet yields corner coordinates
that wobble by a few pixels every frame, the OBB angle occasionally flips, and
the odd frame misses entirely (causing a flicker). This module matches each new
detection to a track from the previous frame (by bounding-box IoU), exponentially
smooths the corner positions (EMA), and keeps a track alive for a few frames when
a detection is briefly missed.

It is detector-agnostic: it operates on ``Detection`` objects, so the cv, onnx,
and yolo backends all benefit.
"""

from __future__ import annotations

import numpy as np

from .detector import Detection


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _align_corners(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Reorder ``cur``'s 4 corners to best match ``prev`` (handles OBB flips).

    Tries all 4 cyclic rotations of the corner order and picks the one with the
    smallest total corner displacement, so EMA blends matching corners instead
    of averaging a corner against the wrong one (which would yank the box).
    """
    best, best_cost = cur, float("inf")
    for k in range(4):
        cand = np.roll(cur, k, axis=0)
        cost = float(np.sum((cand - prev) ** 2))
        if cost < best_cost:
            best, best_cost = cand, cost
    return best


class _Track:
    __slots__ = ("quad", "score", "label", "misses")

    def __init__(self, det: Detection) -> None:
        self.quad = det.quad.astype(np.float32)
        self.score = det.score
        self.label = det.label
        self.misses = 0


class DetectionSmoother:
    """Match detections across frames and EMA-smooth them.

    Args:
        alpha: EMA weight for the new measurement (0..1). Lower = smoother but
            laggier. 0.5 is a good balance for ~30 fps.
        iou_match: minimum IoU to consider a new detection the same object.
        max_misses: how many consecutive missed frames a track survives before
            it is dropped (prevents flicker on the odd dropped detection).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        iou_match: float = 0.3,
        max_misses: int = 5,
        hold_frames: int = 2,
    ) -> None:
        self.alpha = alpha
        self.iou_match = iou_match
        self.max_misses = max_misses
        self.hold_frames = hold_frames
        self._tracks: list[_Track] = []

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched = list(detections)

        # 1. Match each existing track to its best new detection by IoU.
        for track in self._tracks:
            t_bbox = Detection(quad=track.quad).bbox
            best_i, best_iou = -1, self.iou_match
            for i, det in enumerate(unmatched):
                iou = _bbox_iou(t_bbox, det.bbox)
                if iou >= best_iou:
                    best_i, best_iou = i, iou

            if best_i >= 0:
                det = unmatched.pop(best_i)
                aligned = _align_corners(track.quad, det.quad.astype(np.float32))
                track.quad = self.alpha * aligned + (1 - self.alpha) * track.quad
                track.score = self.alpha * det.score + (1 - self.alpha) * track.score
                track.label = det.label
                track.misses = 0
            else:
                track.misses += 1

        # 2. Drop stale tracks.
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]

        # 3. Start tracks for any new, unmatched detections.
        for det in unmatched:
            self._tracks.append(_Track(det))

        # 4. Emit smoothed detections. Briefly missed tracks (within hold_frames)
        #    keep showing their last box so a dropped frame does not flicker.
        out: list[Detection] = []
        for t in self._tracks:
            if t.misses <= self.hold_frames:
                out.append(Detection(quad=t.quad.copy(), score=float(t.score), label=t.label))
        out.sort(key=lambda d: d.score, reverse=True)
        return out
