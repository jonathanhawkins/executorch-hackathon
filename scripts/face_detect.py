#!/usr/bin/env python3
"""Find the face in a frame, so the parser gets the square crop it needs.

    python scripts/face_detect.py frame.jpg                 # prints JSON
    python scripts/face_detect.py frame.jpg --draw out.png  # and a debug overlay

BiSeNet falls apart on anything that is not a face-filling square (see
docs/faceparse-npu.md), so something has to say where the face is. This is that
something: YuNet, OpenCV's bundled detector — 233 kB, runs in a few
milliseconds on CPU, and returns five landmarks along with the box.

The landmarks are not decoration. Compositing a head onto a product photo needs
scale and rotation, and inter-eye distance plus the eye line give both. They
are carried through to crop.json so the compositing step does not have to
detect all over again.

Largest face wins: at a billboard, that is the person standing in front of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MODEL_REPO = "opencv/face_detection_yunet"
MODEL_FILE = "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# YuNet's landmark order, for whoever reads crop.json next.
LANDMARKS = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")


def model_path() -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        sys.exit(f"YuNet digest mismatch: {digest}")
    return path


_DETECTORS: dict = {}


def _detector(width: int, height: int, score_threshold: float):
    """Cached per input size — constructing one costs ~100x running one."""
    import cv2

    key = (width, height, score_threshold)
    if key not in _DETECTORS:
        _DETECTORS[key] = cv2.FaceDetectorYN.create(
            model_path(), "", (width, height), score_threshold, 0.3, 5000
        )
    return _DETECTORS[key]


def detect_frame(frame, score_threshold: float = 0.6, scale: int = 1):
    """Largest face in an already-decoded BGR frame, or None.

    This is the entry point for a live loop: it takes the array the camera
    already gave you, and reuses the detector across calls.

    `scale` divides the frame before detection and multiplies the coordinates
    back, which is the cheapest speed lever there is. Measured on the board
    against a 1280x720 frame with a 121 px face:

        scale  input      time     box drift
          1    1280x720   70.1 ms  —
          2     640x360   20.4 ms  1 px
          3     426x240   11.5 ms  2 px
          4     320x180    7.1 ms  7 px

    Default is 1 so nothing changes under anyone tuning placement against
    full-resolution boxes. A live loop should pass 3.
    """
    import cv2

    scale = max(1, int(scale))
    if scale > 1:
        height, width = frame.shape[:2]
        frame = cv2.resize(
            frame, (width // scale, height // scale), interpolation=cv2.INTER_AREA
        )

    height, width = frame.shape[:2]
    _, faces = _detector(width, height, score_threshold).detect(frame)
    if faces is None or len(faces) == 0:
        return None

    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = (int(round(v * scale)) for v in face[:4])
    points = {
        name: [int(round(face[4 + 2 * i] * scale)),
               int(round(face[5 + 2 * i] * scale))]
        for i, name in enumerate(LANDMARKS)
    }
    return {
        "bbox": [x, y, w, h],
        "landmarks": points,
        "score": float(face[14]),
        "frame": [width * scale, height * scale],
    }


def detect(image_path: Path, score_threshold: float = 0.6, scale: int = 1):
    """Largest face in an image file, or None. Coordinates are in image pixels."""
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        sys.exit(f"{image_path}: could not be read as an image")
    return detect_frame(frame, score_threshold, scale)


def main() -> int:
    ap = argparse.ArgumentParser(description="Find the largest face in an image")
    ap.add_argument("image", type=Path)
    ap.add_argument("--draw", type=Path, help="write an overlay for eyeballing it")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--scale", type=int, default=1,
                    help="divide the frame before detecting; 3 is ~6x faster")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"{args.image}: not found", file=sys.stderr)
        return 1

    found = detect(args.image, args.threshold, args.scale)
    if found is None:
        print(json.dumps({"face": None}))
        return 2

    print(json.dumps(found, indent=2))

    if args.draw:
        import cv2

        frame = cv2.imread(str(args.image))
        x, y, w, h = found["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for point in found["landmarks"].values():
            cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)
        cv2.imwrite(str(args.draw), frame)
        print(f"overlay -> {args.draw}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
