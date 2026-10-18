#!/usr/bin/env python3
"""Turn face images into the raw tensors the BiSeNet face parser expects.

    python scripts/make_face_raw.py portrait.jpg -o out/          # one image
    python scripts/make_face_raw.py --calib 40 -o calib/          # CelebA-HQ set

The QNN executor runner takes raw little-endian float32 and nothing else, so
this does what the model's own preprocessing does: resize to 512x512, scale to
[0,1], normalise with ImageNet statistics, emit NCHW. The same function serves
inference and quantizer calibration, which matters — a calibration set that
went through different preprocessing would set the activation ranges wrong.

`--calib` pulls faces from CelebA-HQ through the HF datasets-server rows API
rather than downloading a 460 MB parquet shard for forty pictures. They arrive
at 256x256 and are upsampled, so they are slightly softer than a webcam frame;
for setting 8-bit activation ranges that is close enough.

BiSeNet was trained on CelebAMask-HQ, which is square and face-filling, and it
falls apart on anything else — a 16:9 webcam frame squeezed into 512x512 comes
back as a blocky rectangle with invented hats. So the crop is not an
afterthought: by default this runs the YuNet detector from face_detect.py and
crops around the face it finds. `--bbox` overrides it, `--no-detect` falls back
to a centre square crop, and either way the crop (and the landmarks, when a
detector ran) land in crop.json so face_matte.py maps the result back to the
same pixels and the compositing step does not have to detect again.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

INPUT_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CALIB_DATASET = "korexyz/celeba-hq-256x256"
ROWS_API = "https://datasets-server.huggingface.co/rows"


def square_crop(img: Image.Image, bbox: tuple[int, int, int, int] | None = None):
    """Crop to a square. Returns the image and the (x, y, side) actually used."""
    if bbox is not None:
        x, y, w, h = bbox
        # Square it off around the box centre, with room for hair and neck.
        side = int(max(w, h) * 1.6)
        cx, cy = x + w // 2, y + h // 2
    else:
        side = min(img.width, img.height)
        cx, cy = img.width // 2, img.height // 2

    left = max(0, min(cx - side // 2, img.width - side))
    top = max(0, min(cy - side // 2, img.height - side))
    side = min(side, img.width - left, img.height - top)
    return img.crop((left, top, left + side, top + side)), (left, top, side)


def preprocess(img: Image.Image, size: int = INPUT_SIZE, bbox=None) -> np.ndarray:
    img, _ = square_crop(img.convert("RGB"), bbox)
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def _get_json(url: str, attempts: int = 4):
    """GET with retries — datasets-server returns a 502 often enough to matter."""
    import json
    import time

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_calibration_images(count: int):
    """CelebA-HQ faces, without pulling a parquet shard."""
    query = urllib.parse.urlencode(
        {
            "dataset": CALIB_DATASET,
            "config": "default",
            "split": "train",
            "offset": 0,
            "length": min(count, 100),
        }
    )
    payload = _get_json(f"{ROWS_API}?{query}")
    if "rows" not in payload:
        sys.exit(f"datasets-server returned no rows: {str(payload)[:200]}")

    for row in payload["rows"][:count]:
        src = row["row"]["image"]["src"]
        with urllib.request.urlopen(src, timeout=60) as resp:
            yield Image.open(io.BytesIO(resp.read()))


def write_set(images, out: Path, bbox=None, detection=None) -> list[str]:
    import json

    out.mkdir(parents=True, exist_ok=True)
    names = []
    for idx, img in enumerate(images):
        if idx == 0:
            # Record the crop so the matte lands back on the right pixels, and
            # the landmarks so whatever composites next can skip its own detect.
            _, crop = square_crop(img.convert("RGB"), bbox)
            record = {"left": crop[0], "top": crop[1], "side": crop[2]}
            if detection:
                record["landmarks"] = detection["landmarks"]
                record["score"] = detection["score"]
            (out / "crop.json").write_text(json.dumps(record, indent=2))
        tensor = preprocess(img, bbox=bbox)
        assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE), tensor.shape
        name = f"input_{idx}.raw"
        tensor.tofile(out / name)
        names.append(name)
    (out / "input_list.txt").write_text("\n".join(names) + "\n")
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description="Face image -> BiSeNet raw tensor")
    ap.add_argument("image", nargs="?", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("face_input"))
    ap.add_argument("--calib", type=int, metavar="N",
                    help="instead of an image, fetch N CelebA-HQ faces")
    ap.add_argument("--bbox", type=str, metavar="X,Y,W,H",
                    help="face box to crop around, instead of detecting one")
    ap.add_argument("--no-detect", action="store_true",
                    help="skip detection and take a centre square crop")
    args = ap.parse_args()

    bbox = tuple(int(v) for v in args.bbox.split(",")) if args.bbox else None
    if bbox is not None and len(bbox) != 4:
        return ap.error("--bbox takes X,Y,W,H")

    if args.calib:
        names = write_set(fetch_calibration_images(args.calib), args.output)
        print(f"  calib  {len(names)} faces from {CALIB_DATASET}")
    else:
        if not args.image or not args.image.exists():
            return ap.error("give an image, or --calib N")

        detection = None
        if bbox is None and not args.no_detect:
            from face_detect import detect

            detection = detect(args.image)
            if detection is None:
                print("  detect no face found — falling back to a centre crop")
            else:
                bbox = tuple(detection["bbox"])
                print(f"  detect face at {bbox}, score {detection['score']:.2f}")

        names = write_set([Image.open(args.image)], args.output, bbox, detection)
        print(f"  in     {args.image.name}")

    print(f"  shape  (1, 3, {INPUT_SIZE}, {INPUT_SIZE}) float32")
    print(f"  out    {args.output}/  (+ input_list.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
