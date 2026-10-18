#!/usr/bin/env python3
"""Turn frames into the raw tensors the person segmenter expects.

    python scripts/make_person_raw.py frame.jpg -o out/        # one frame
    python scripts/make_person_raw.py --calib 48 -o calib/     # ATR people

Same contract as scripts/make_face_raw.py: the QNN executor runner takes raw
little-endian float32 NCHW and nothing else, and the calibration set goes
through this exact function so the quantizer sees the activation ranges that
inference will actually produce.

Where this differs from the face path is the crop. BiSeNet wants a
face-filling square; a human-parsing model wants a *person*. Measured on a
billboard-framed frame, a crop 3.5 face-widths across loses the arms
altogether — they are cut off at the frame edge, so the model has nothing to
label — while 5 face-widths recovers them. So the default crop is sized in
face widths and placed to put the head near its top, which is where a standing
subject's head is in the training data.

`--full-frame` skips the crop and squashes the whole frame into the input
instead. It is the fallback when no face is found, and it is what to use if the
subject is turned away from the camera.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from person_seg import MODELS, DEFAULT_MODEL, SegModel  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# How much of the person to take, in units of the detected face box.
COVER = 5.0          # square side, in face widths
ABOVE = 0.8          # how far above the face box the crop starts, in face heights

CALIB_DATASET = "mattmdjaga/human_parsing_dataset"
ROWS_API = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True)
class BodyCrop:
    """The region of the frame the model saw, so the mask can be put back."""

    left: int
    top: int
    width: int
    height: int
    face: dict | None = None

    def to_json(self, model: SegModel) -> str:
        record = {
            "left": self.left, "top": self.top,
            "width": self.width, "height": self.height,
            "model": model.key,
            "logits": list(model.logit_shape),
        }
        if self.face:
            record["bbox"] = list(self.face["bbox"])
            record["landmarks"] = self.face["landmarks"]
            record["score"] = self.face["score"]
        return json.dumps(record, indent=2)


def body_crop(img: Image.Image, face: dict | None, cover: float = COVER) -> BodyCrop:
    """A square around the person, inferred from the face box.

    Clamped into the frame rather than padded: a padded border is a texture the
    model has never seen, and on a 16:9 frame the clamp costs at most a little
    of the crop's reach. If the subject is so close that the square does not
    fit, it shrinks to the frame — which is the whole-frame case anyway.
    """
    if face is None:
        return BodyCrop(0, 0, img.width, img.height, None)

    x, y, w, h = face["bbox"]
    # Size first, then place. Doing it the other way round means a face big
    # enough to want a crop wider than the frame gets the square pinned to the
    # top-left corner, which on a close-up portrait leaves the subject hanging
    # off the right-hand edge of the region the model sees.
    side = min(int(w * cover), img.width, img.height)
    left = max(0, min(int(round(x + w / 2.0 - side / 2.0)), img.width - side))
    top = max(0, min(int(round(y - ABOVE * h)), img.height - side))
    return BodyCrop(left, top, side, side, face)


# (x/255 - mean) / std, folded into one multiply and one subtract. Same
# arithmetic, two fewer full-size temporaries — this runs once a frame in the
# live loop and the temporaries were most of its cost.
_SCALE = (1.0 / 255.0) / IMAGENET_STD
_SHIFT = IMAGENET_MEAN / IMAGENET_STD


def preprocess(img: Image.Image, model: SegModel, crop: BodyCrop) -> np.ndarray:
    """Crop, resize, ImageNet-normalise, NCHW float32."""
    region = img.convert("RGB").crop(
        (crop.left, crop.top, crop.left + crop.width, crop.top + crop.height)
    )
    height, width = model.size
    # PIL's BILINEAR, not cv2's: the calibration set was built with this one,
    # and a different downsampler would quietly shift every activation range.
    region = region.resize((width, height), Image.BILINEAR)
    arr = np.asarray(region, dtype=np.float32)
    arr *= _SCALE
    arr -= _SHIFT
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])


def prepare(img: Image.Image, model: SegModel, *, cover: float | None = None,
            full_frame: bool = False, face: dict | None = None,
            detect_image=None) -> tuple[np.ndarray, BodyCrop]:
    """Detect (unless told not to), crop, preprocess. The one entry point."""
    if not full_frame and face is None:
        from face_detect import detect_frame

        frame = np.asarray(img.convert("RGB"))[:, :, ::-1]
        face = detect_frame(np.ascontiguousarray(frame))

    crop = BodyCrop(0, 0, img.width, img.height, face) if full_frame \
        else body_crop(img, face, COVER if cover is None else cover)
    return preprocess(img, model, crop), crop


def _get_json(url: str, attempts: int = 4):
    """GET with retries — datasets-server returns a 502 often enough to matter."""
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
    """People from ATR — the dataset the chosen checkpoint was trained on.

    Through the rows API rather than the parquet shard, as in make_face_raw.py.
    Whole people on textured backgrounds, which is what the billboard sees; the
    body crop below turns each one into roughly billboard framing.
    """
    fetched = 0
    for offset in range(0, 10_000, 100):
        if fetched >= count:
            return
        query = urllib.parse.urlencode({
            "dataset": CALIB_DATASET, "config": "default", "split": "train",
            "offset": offset, "length": min(count - fetched, 100),
        })
        payload = _get_json(f"{ROWS_API}?{query}")
        rows = payload.get("rows")
        if not rows:
            sys.exit(f"datasets-server returned no rows: {str(payload)[:200]}")
        for row in rows:
            if fetched >= count:
                return
            with urllib.request.urlopen(row["row"]["image"]["src"], timeout=60) as resp:
                yield Image.open(io.BytesIO(resp.read()))
            fetched += 1


def write_set(images, out: Path, model: SegModel, **kwargs) -> tuple[int, int]:
    out.mkdir(parents=True, exist_ok=True)
    names, detected = [], 0
    for idx, img in enumerate(images):
        tensor, crop = prepare(img, model, **kwargs)
        if crop.face:
            detected += 1
        assert tensor.shape == model.input_shape, tensor.shape
        if idx == 0:
            (out / "person_crop.json").write_text(crop.to_json(model))
        name = f"input_{idx}.raw"
        tensor.tofile(out / name)
        names.append(name)
    (out / "input_list.txt").write_text("\n".join(names) + "\n")
    return len(names), detected


def main() -> int:
    ap = argparse.ArgumentParser(description="Frame -> person-segmenter raw tensor")
    ap.add_argument("image", nargs="?", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("person_input"))
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    ap.add_argument("--calib", type=int, metavar="N",
                    help="instead of an image, fetch N people from ATR")
    ap.add_argument("--cover", type=float, default=COVER,
                    help="crop side, in face widths (default %(default)s)")
    ap.add_argument("--full-frame", action="store_true",
                    help="skip the body crop and squash the whole frame")
    args = ap.parse_args()

    model = MODELS[args.model]
    kwargs = {"cover": args.cover, "full_frame": args.full_frame}

    if args.calib:
        count, detected = write_set(
            fetch_calibration_images(args.calib), args.output, model, **kwargs
        )
        print(f"  calib  {count} people from {CALIB_DATASET}"
              f"  ({detected} body-cropped, {count - detected} whole-frame)")
    else:
        if not args.image or not args.image.exists():
            return ap.error("give an image, or --calib N")
        count, detected = write_set([Image.open(args.image)], args.output, model, **kwargs)
        print(f"  in     {args.image.name}"
              f"  ({'body crop' if detected else 'whole frame, no face found'})")

    height, width = model.size
    print(f"  model  {model.key}")
    print(f"  shape  (1, 3, {height}, {width}) float32")
    print(f"  out    {args.output}/  (+ input_list.txt, person_crop.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
