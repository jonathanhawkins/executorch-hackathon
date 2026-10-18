#!/usr/bin/env python3
"""Record what a correctly-sized head measures on a billboard template.

    python scripts/calibrate_template.py assets/templates/model-navy-tee.png

Feeds the model's own photo back through the compositor and stores the width
his head ends up at, as `headSpanRef` in the template JSON. `head_match()`
compares shoppers against that, so a head genuinely the right size reads 1.00.

Why this is needed at all: the stored head mask is not comparable to a
shopper's matte. It was grown so no hair fringe survived the inpaint, and it
never went through the feather, resample and alpha cut that a shopper's matte
does. Measured against it, feeding the model his own head back reads **0.87** —
and anyone trusting that number would scale the head up by 15% to correct a
discrepancy that is entirely an artefact of measuring two different things.

Run once per template, after prep_template.py. It is also the sharpest
end-to-end test there is: the landmark fit, the scale and the placement all
have a known right answer here, because the answer is the original photograph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import face_detect  # noqa: E402
import face_parse  # noqa: E402
from billboard_photo import PhotoBillboard, Smoother  # noqa: E402
from make_face_raw import preprocess, square_crop  # noqa: E402
from vendor.face_parsing import CLASSES, load_face_parser  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate a billboard template")
    ap.add_argument("image", type=Path, help="the original product photo")
    args = ap.parse_args()

    stem = args.image.with_suffix("")
    meta_path = Path(f"{stem}.json")
    if not meta_path.exists():
        print(f"{meta_path}: run prep_template.py first", file=sys.stderr)
        return 1

    img = Image.open(args.image).convert("RGB")
    found = face_detect.detect(args.image, score_threshold=0.5)
    if not found:
        print("  no face in the template photo", file=sys.stderr)
        return 1
    bbox = found["bbox"]

    _, (left, top, side) = square_crop(img, bbox)
    net = load_face_parser(face_parse.checkpoint_path())
    with torch.no_grad():
        logits = net(torch.from_numpy(preprocess(img, bbox=bbox)))[0]
    parse = logits.numpy().astype(np.float32).reshape(len(CLASSES), 512, 512).argmax(0).astype(np.uint8)

    head_idx = [i for i, n in enumerate(CLASSES) if n not in ("background", "cloth")]
    neck_idx = [i for i, n in enumerate(CLASSES) if n in ("neck", "neck_l")]
    skin_idx = CLASSES.index("skin")

    k = 512.0 / side
    to_crop = lambda p: ((p[0] - left) * k, (p[1] - top) * k)  # noqa: E731
    lm = {name: to_crop(pt) for name, pt in found["landmarks"].items()}
    face_box = [(bbox[0] - left) * k, (bbox[1] - top) * k, bbox[2] * k, bbox[3] * k]
    crop512 = img.crop((left, top, left + side, top + side)).resize((512, 512), Image.BILINEAR)

    board = PhotoBillboard(
        Path(f"{stem}.template.png"), Path(f"{stem}.skin.png"), meta_path
    )
    smoother = Smoother(alpha=0.3)
    smoother.reset()
    composed, info = board.compose(
        crop512, parse, head_idx, neck_idx,
        lm["right_eye"], lm["left_eye"], smoother,
        offsets={"headScale": 1.0, "dx": 0.0, "dy": 0.0,
                 "skin_idx": skin_idx, "neckBlend": True},
        face_box=face_box, landmarks=lm,
    )
    if composed is None:
        print(f"  compose failed: {info.get('reason')}", file=sys.stderr)
        return 1

    if info.get("align") != "landmarks":
        # Refuse rather than warn. The face-box path measured 4 degrees of
        # tilt on this model's demonstrably untilted photo, so a reference
        # derived from it is wrong — and it would be written into the template
        # JSON permanently, silently skewing every later head_match. A warning
        # on stderr during a one-off setup step is far too easy to miss.
        print(f"  alignment fell back to {info.get('align')!r} on the template's "
              f"own photo — the landmark fit is not working, so this "
              f"calibration would be wrong. Nothing written.", file=sys.stderr)
        return 1

    # Take the measurement from the compositor itself, not from the pixels.
    # head_match already measures the placed head the same way it will measure
    # shoppers, and reports it as headSpanPx; reading that makes the reference
    # like-for-like by construction. Deriving it here from colour instead
    # would hardcode this template's pale backdrop — on a dark or saturated
    # one the "not backdrop" test matches the whole row, the span becomes the
    # full image width, and every later headMatch reads about 0.23 while the
    # script still exits 0 with a plausible-looking number.
    span = info.get("headSpanPx")
    if not span:
        print("  compose did not report a head span — is the head mask "
              "(.head.png) present?", file=sys.stderr)
        return 1

    meta = json.loads(meta_path.read_text())
    eyes = [meta["landmarks"][n] for n in ("right_eye", "left_eye") if n in meta["landmarks"]]
    row = int(round(sum(e[1] for e in eyes) / len(eyes)))

    meta["headSpanRef"] = round(span, 1)
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"  align      {info.get('align')}")
    # The crop is resized up to 512, so placing it back at original size means
    # scaling by the inverse of that factor. Anything else is a real error.
    print(f"  scale      {info.get('scale')}  (expected {side / 512:.3f} — undoes the 512 crop)")
    print(f"  eye row    {row}")
    print(f"  headSpanRef {span:.1f}px  -> written to {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
