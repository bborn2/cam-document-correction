"""Entry point: open the webcam, detect A4 sheets, and highlight them live."""

from __future__ import annotations

import argparse
import time

import cv2

import numpy as np

from .camera import Camera
from .cv_detector import CvA4Detector
from .detector import Detector
from .draw import draw_detections
from .smoothing import DetectionSmoother

WINDOW_NAME = "A4 Detect (press q or ESC to quit)"
DEBUG_WINDOW = "A4 Detect - debug (mask | edges)"
WARP_WINDOW = "Perspective Corrected (A4)"


def build_detector(args: argparse.Namespace) -> Detector:
    if args.detector == "pose":
        if not args.weights:
            raise SystemExit("--detector pose requires --weights path/to/best-pose.onnx")
        from .onnx_pose_detector import OnnxPoseDetector  # lazy import (optional dep)

        return OnnxPoseDetector(args.weights, conf=args.conf, iou=args.iou)
    if args.detector == "seg":
        if not args.weights:
            raise SystemExit("--detector seg requires --weights path/to/best.onnx")
        from .onnx_seg_detector import OnnxSegDetector  # lazy import (optional dep)

        return OnnxSegDetector(args.weights, conf=args.conf, iou=args.iou)
    if args.detector == "obb":
        if not args.weights:
            raise SystemExit("--detector obb requires --weights path/to/best.onnx")
        from .onnx_detector import OnnxObbDetector  # lazy import (optional dep)

        return OnnxObbDetector(args.weights, conf=args.conf, iou=args.iou)
    if args.detector == "yolo":
        if not args.weights:
            raise SystemExit("--detector yolo requires --weights path/to/best.pt")
        from .yolo_detector import YoloSegDetector  # lazy import (optional dep)

        return YoloSegDetector(args.weights, conf=args.conf)
    return CvA4Detector()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time A4 paper detection from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument(
        "--detector", choices=["cv", "obb", "seg", "pose", "yolo"], default="cv",
        help="Detector backend: 'cv' (traditional), 'obb' (rotated box), "
             "'seg' (segmentation vertices), 'pose' (corner keypoints + perspective "
             "correction), or 'yolo' (ultralytics .pt).",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Path to model weights (.onnx for onnx, .pt for yolo).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (onnx).")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument(
        "--no-smooth", dest="smooth", action="store_false",
        help="Disable temporal smoothing (smoothing is ON by default).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Smoothing strength: lower = steadier but laggier (0..1, default 0.5).",
    )
    parser.set_defaults(smooth=True)
    parser.add_argument(
        "--debug", action="store_true",
        help="Show an extra window with the brightness mask and edge map (cv detector).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    detector = build_detector(args)
    smoother = DetectionSmoother(alpha=args.alpha) if args.smooth else None

    prev = time.perf_counter()
    fps = 0.0

    with Camera(args.camera, args.width, args.height) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                print("Failed to read frame; stopping.")
                break

            detections = detector.detect(frame)
            if smoother is not None:
                detections = smoother.update(detections)

            # Keep a clean copy (before overlays) for perspective correction.
            clean = frame.copy() if args.detector == "pose" else None
            draw_detections(frame, detections)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(
                frame, f"FPS: {fps:4.1f}  | {len(detections)} A4",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame)

            # Pose detector: show the perspective-corrected ("flattened") document.
            if args.detector == "pose" and detections and clean is not None:
                from .onnx_pose_detector import warp_perspective

                warped = warp_perspective(clean, detections[0].quad)
                cv2.imshow(WARP_WINDOW, warped)

            # Debug window: show internal masks/edges side by side.
            if args.debug and isinstance(detector, CvA4Detector):
                mask = detector.debug_mask
                edges = detector.debug_edges
                if mask is not None and edges is not None:
                    dbg = np.hstack([
                        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
                    ])
                    cv2.imshow(DEBUG_WINDOW, dbg)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
