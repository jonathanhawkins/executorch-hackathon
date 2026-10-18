#!/usr/bin/env python3
"""Turn an image into the raw tensor InternVL3's vision encoder expects.

The QNN multimodal runner takes raw float32 tensors, not JPEGs — it calls
load_raw_files() on whatever --image_path points at. This produces one.

    python scripts/make_vision_raw.py photo.jpg -o photo.raw

Implements InternVL's documented build_transform: convert to RGB, resize to
448x448 bicubic, scale to [0,1], normalize with ImageNet statistics, emit
NCHW float32. That matches what the export pipeline does via the HF processor
with crop_to_patches=False and size={"height":448,"width":448} — reimplemented
here because the HF route needs trust_remote_code, and running a repo's
arbitrary code to resize a picture is a poor trade.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# InternVL3Encoder in the bundle's encoder_config.py.
INPUT_SIZE = 448
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path: Path, size: int = INPUT_SIZE) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC in [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD               # normalize
    arr = np.transpose(arr, (2, 0, 1))                       # CHW
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)  # NCHW


def main() -> int:
    ap = argparse.ArgumentParser(description="Image -> InternVL3 raw tensor")
    ap.add_argument("image", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="defaults to <image>.raw")
    ap.add_argument("--size", type=int, default=INPUT_SIZE)
    args = ap.parse_args()

    if not args.image.exists():
        print(f"{args.image}: not found", file=sys.stderr)
        return 1

    tensor = preprocess(args.image, args.size)
    out = args.output or args.image.with_suffix(".raw")
    tensor.tofile(out)

    print(f"  in    {args.image.name}")
    print(f"  shape {tuple(tensor.shape)} float32")
    print(f"  range [{tensor.min():.3f}, {tensor.max():.3f}]")
    print(f"  out   {out}  ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
