#!/usr/bin/env python3
"""One-time prep of a product photo into a billboard template.

    python scripts/prep_template.py assets/templates/model-navy-tee.png

Takes a photo of a model wearing the garment and produces everything the
per-frame compositor needs:

    <name>.template.png   the model with his head removed and inpainted
    <name>.skin.png       mask of his neck and arms, for skin-tone matching
    <name>.json           jawline anchor, jaw width, image size

Run once per product. Nothing here happens per frame.

The model's **neck is deliberately kept**, including the shadow under his jaw.
That shadow is most of what sells the composite: a shopper's head dropped onto
a neck that is already lit and shaded reads as one photograph, where a head
over a drawn column never quite does.
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
from make_face_raw import preprocess, square_crop  # noqa: E402
from vendor.face_parsing import CLASSES, load_face_parser  # noqa: E402

# Everything that is the model's HEAD, which we remove. The neck classes stay:
# we want his neck and its shading. Derived from the label map by name rather
# than hardcoded, because the indices are only meaningful through CLASSES.
REMOVE_NAMES = {
    "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g",
    "l_ear", "r_ear", "ear_r", "nose", "mouth", "u_lip", "l_lip",
    "hair", "hat",
}
HEADROOM = 150


def parse_face(img: Image.Image, bbox):
    """BiSeNet class map for the face crop, fp32 on the Mac. One-time, so the
    quantized NPU path is not worth the round trip."""
    net = load_face_parser(face_parse.checkpoint_path())
    tensor = torch.from_numpy(preprocess(img, bbox=bbox))
    with torch.no_grad():
        logits = net(tensor)[0]
    # The output still carries its batch axis; reshape to (classes, H, W) the
    # same way face_matte.parse_map does, or argmax collapses the batch and
    # every pixel comes back as class 0.
    arr = logits.numpy().astype(np.float32).reshape(len(CLASSES), 512, 512)
    return arr.argmax(0).astype(np.uint8)


def build_skin_mask(rgb: np.ndarray) -> np.ndarray:
    """The model's visible skin: neither the grey backdrop nor the navy shirt.

    Defining it by exclusion is more robust than trying to describe skin: the
    two things to exclude are a flat desaturated backdrop and a single dark
    saturated garment colour, both easy to characterise.
    """
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    backdrop = (sat < 42) & (val > 140)
    # OpenCV hue is 0-179; navy sits around 105-135.
    shirt = (h > 95) & (h < 140) & (sat > 55) & (val < 165)

    skin = ~(backdrop | shirt)
    skin = skin.astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k)
    return skin


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser(description="Prepare a billboard template")
    ap.add_argument("image", type=Path)
    ap.add_argument("--dilate", type=int, default=7, help="grow the head mask, px")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"{args.image}: not found", file=sys.stderr)
        return 1

    img = Image.open(args.image).convert("RGB")
    rgb = np.asarray(img)
    print(f"  template   {args.image.name}  {img.width}x{img.height}")

    found = face_detect.detect(args.image, score_threshold=0.5)
    if not found:
        print("  no face found in the template", file=sys.stderr)
        return 1
    bbox = found["bbox"]
    print(f"  model face {bbox[2]}x{bbox[3]} at {bbox[0]},{bbox[1]}")

    parse = parse_face(img, bbox)
    present = sorted({CLASSES[i] for i in np.unique(parse)})
    print(f"  classes    {present}")

    remove_idx = [i for i, name in enumerate(CLASSES) if name in REMOVE_NAMES]
    mask512 = np.isin(parse, remove_idx).astype(np.uint8) * 255

    # Back to full-image coordinates through the same crop the parse used.
    _, (left, top, side) = square_crop(img, bbox)
    mask_full = np.zeros(rgb.shape[:2], dtype=np.uint8)
    resized = cv2.resize(mask512, (side, side), interpolation=cv2.INTER_NEAREST)
    mask_full[top:top + side, left:left + side] = resized

    # Grow it so no hair fringe survives to halo the inpaint.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate * 2 + 1,) * 2)
    mask_full = cv2.dilate(mask_full, k)

    ys, xs = np.nonzero(mask_full)
    if ys.size == 0:
        print("  head mask came back empty", file=sys.stderr)
        return 1
    jaw_y = int(ys.max())
    # Width measured across the bottom band of the removed area — that is the
    # jaw, not the widest part of the hair.
    band = mask_full[max(0, jaw_y - 24):jaw_y + 1]
    band_xs = np.nonzero(band.any(axis=0))[0]
    jaw_w = int(band_xs.max() - band_xs.min())
    jaw_x = int((band_xs.min() + band_xs.max()) / 2)
    print(f"  jawline    ({jaw_x}, {jaw_y})  jaw width {jaw_w}px")

    inpainted = cv2.inpaint(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask_full, 7, cv2.INPAINT_TELEA
    )
    inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    # Headroom: replicate the top row upward. The backdrop is a smooth
    # gradient, so replication matches it where a flat fill would band.
    pad = np.repeat(inpainted[:1], HEADROOM, axis=0)
    out_rgb = np.vstack([pad, inpainted])
    jaw_y += HEADROOM

    skin = build_skin_mask(out_rgb)
    # Never let the mask reach into the padded strip.
    skin[:HEADROOM] = 0

    # Where the garment starts, measured rather than eyeballed: the neck blend
    # fades the shopper's neck out by this row, so it has to be the real
    # neckline. Topmost row below the jaw that is neither skin nor backdrop,
    # in a band under the chin.
    hsv = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2HSV)
    backdrop = (hsv[..., 1] < 42) & (hsv[..., 2] > 140)
    garment = ~backdrop & (skin == 0)
    band = garment[:, max(0, jaw_x - 60):jaw_x + 60]
    below = np.nonzero(band[jaw_y:].any(axis=1))[0]
    collar_y = int(jaw_y + below.min()) if below.size else int(jaw_y + 125)
    print(f"  collar     y={collar_y}  ({collar_y - jaw_y}px below the jaw)")

    stem = args.image.with_suffix("")
    Image.fromarray(out_rgb).save(f"{stem}.template.png")
    Image.fromarray(skin).save(f"{stem}.skin.png")
    # Keep the head we removed. It is the one piece of ground truth about
    # where a head belongs on this body and how big it is — measured from the
    # original photo rather than reconstructed from a few anchor numbers — and
    # it is exactly the region the inpaint replaced with backdrop, so anything
    # it covers that the shopper's head does not is a visible halo.
    head_pad = np.vstack([np.zeros((HEADROOM, mask_full.shape[1]), np.uint8), mask_full])
    Image.fromarray(head_pad).save(f"{stem}.head.png")
    # Anchor on the model's own face box rather than the mask's bottom band.
    # Both sides then measure the same thing with the same detector, which the
    # mask bands do not: beard and hair widen a shopper's band but not the
    # model's, so scaling by them shrinks the head.
    fx, fy, fw, fh = bbox
    meta = {
        "source": args.image.name,
        "size": [int(out_rgb.shape[1]), int(out_rgb.shape[0])],
        "jawline": [jaw_x, jaw_y],
        "jawWidth": jaw_w,
        "faceWidth": int(fw),
        "chin": [int(fx + fw / 2), int(fy + fh) + HEADROOM],
        "collar": collar_y,
        # How much the removal mask was grown. The mask is deliberately wider
        # than the head so no hair fringe survives the inpaint, which means
        # anything measuring against it has to take the growth back off first.
        "dilate": int(args.dilate),
        # What the model's own head measures at his eye line once it has been
        # through the compositor — feather, resample, alpha cut and all. This
        # is the reference head_match compares a shopper against, so that a
        # head genuinely the right size reads 1.00 rather than 0.95.
        # Written by calibrate_template.py; absent until that has been run.
        # The model's own landmarks, in the padded template's coordinates.
        # The compositor fits the shopper's five onto these, which places,
        # scales and rotates the head in one estimate.
        "landmarks": {k: [float(v[0]), float(v[1]) + HEADROOM]
                      for k, v in found["landmarks"].items()},
        "headroom": HEADROOM,
    }
    Path(f"{stem}.json").write_text(json.dumps(meta, indent=2))

    print(f"  skin mask  {int((skin > 0).sum() / skin.size * 100)}% of frame")
    print(f"  head mask  {int((head_pad > 0).sum() / head_pad.size * 100)}% of frame")
    print(f"  wrote      {Path(stem).name}.template.png, .skin.png, .head.png, .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
