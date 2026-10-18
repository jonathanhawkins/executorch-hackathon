#!/usr/bin/env python3
"""Put the garment on the person, rather than the person's head on the garment.

    python scripts/garment_overlay.py frame.jpg -o tryon.png
    python scripts/garment_overlay.py frame.jpg --matte run/output_0.raw -o tryon.png

The billboard effect reads better this way round. Pasting a live head onto a
studio product photo means a frozen body at one angle and a live head at
another, with a seam and a lighting mismatch between them; pasting the garment
onto the live frame means the head is already right, because it is really
their head, and the whole thing moves with them.

Placement comes from the face: YuNet's box gives scale and its eye landmarks
give roll, and the shoulders are estimated from those by proportion. Those
proportions are the weak part — they assume someone facing the camera roughly
square on, and they will not survive a turn or a lean. Real shoulder keypoints
from a pose model are the fix; the transform here is written to take a pair of
shoulder points from anywhere, so swapping the estimate for a measurement is a
one-function change.

Pass `--matte` and the head matte from the NPU face parser is drawn back over
the garment, so hair falling across a shoulder stays in front of the shirt
instead of vanishing under it. That is what the face segmentation buys in this
arrangement, and it is the part a plain overlay cannot fake.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Proportions, in units of the detected face box. Rough by construction — see
# the module docstring. Exposed as flags so they can be dialled in against a
# real camera rather than argued about.
SHOULDER_HALF_WIDTH = 1.15   # x face width, centre to shoulder point
SHOULDER_DROP = 0.45         # x face height, below the chin


def estimate_shoulders(face: dict, half_width: float, drop: float):
    """Shoulder points in frame pixels, inferred from the face box and eyes.

    Returned as (left, right) in IMAGE coordinates — left meaning smaller x on
    screen, which is the subject's right. The garment's anchors use the same
    convention; mixing the two up rotates the shirt 180 degrees.
    """
    x, y, w, h = face["bbox"]
    marks = face["landmarks"]
    left_eye = np.array(marks["left_eye"], dtype=float)
    right_eye = np.array(marks["right_eye"], dtype=float)

    # The eye line gives the head's roll; shoulders run parallel to it, and
    # "down the body" is its perpendicular. right_eye is the subject's right,
    # so it sits at smaller x — this vector points screen-left to screen-right.
    along = left_eye - right_eye
    norm = np.linalg.norm(along)
    along = along / norm if norm > 1e-6 else np.array([1.0, 0.0])
    down = np.array([-along[1], along[0]])

    chin = np.array([x + w / 2.0, y + h], dtype=float)
    centre = chin + down * (drop * h)
    offset = along * (half_width * w)
    return centre - offset, centre + offset


def warp_garment(garment: Image.Image, meta: dict, shoulder_left, shoulder_right, size):
    """Scale, rotate and translate the garment onto a pair of shoulder points.

    Both pairs are in image coordinates, left meaning smaller x.
    """
    import cv2

    src_l = np.array(meta["shoulder_left"], dtype=float)
    src_r = np.array(meta["shoulder_right"], dtype=float)
    dst_l = np.asarray(shoulder_left, dtype=float)
    dst_r = np.asarray(shoulder_right, dtype=float)

    src_vec, dst_vec = src_r - src_l, dst_r - dst_l
    src_len = np.linalg.norm(src_vec)
    if src_len < 1e-6:
        sys.exit("garment metadata has coincident shoulder anchors")

    scale = np.linalg.norm(dst_vec) / src_len
    angle = math.atan2(dst_vec[1], dst_vec[0]) - math.atan2(src_vec[1], src_vec[0])
    cos_a, sin_a = math.cos(angle) * scale, math.sin(angle) * scale

    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    translation = dst_l - rotation @ src_l
    matrix = np.hstack([rotation, translation.reshape(2, 1)])

    source = np.array(garment.convert("RGBA"))
    warped = cv2.warpAffine(
        source, matrix, size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped, mode="RGBA")


def head_alpha(frame: Image.Image, logits: Path, crop_file: Path, feather, erode):
    """The BiSeNet head matte, as a frame-sized alpha."""
    from face_matte import alpha_from_parse, parse_map

    crop = json.loads(crop_file.read_text())
    parse = parse_map(logits)

    seed = None
    nose = crop.get("landmarks", {}).get("nose")
    if nose:
        scale = 512 / crop["side"]
        seed = (int((nose[0] - crop["left"]) * scale),
                int((nose[1] - crop["top"]) * scale))

    alpha = alpha_from_parse(parse, feather, erode, seed).resize(
        (crop["side"], crop["side"]), Image.BILINEAR
    )
    full = Image.new("L", frame.size, 0)
    full.paste(alpha, (crop["left"], crop["top"]))
    return full


def head_layer(frame: Image.Image, logits: Path, crop_file: Path, feather, erode):
    """The head, cut out of the frame, back at full frame size — the occluder."""
    layer = frame.convert("RGBA").copy()
    layer.putalpha(head_alpha(frame, logits, crop_file, feather, erode))
    return layer


def main() -> int:
    ap = argparse.ArgumentParser(description="Overlay a garment on the person")
    ap.add_argument("image", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("tryon.png"))
    ap.add_argument("--garment", type=Path,
                    default=Path("assets/garments/placeholder-tee.png"))
    ap.add_argument("--matte", type=Path,
                    help="NPU logits (output_0.raw); draws the head over the garment")
    ap.add_argument("--crop", type=Path,
                    help="crop.json for --matte; defaults to beside it")
    ap.add_argument("--body", type=Path,
                    help="NPU person-segmentation logits; clips the garment to "
                         "the silhouette and puts arms back over it")
    ap.add_argument("--body-crop", type=Path,
                    help="person_crop.json for --body; defaults to beside it")
    ap.add_argument("--clip-grow", type=int, default=0,
                    help="pixels to push the clip boundary out, past the two "
                         "it already gets")
    ap.add_argument("--body-feather", type=float, default=2.0)
    ap.add_argument("--half-width", type=float, default=SHOULDER_HALF_WIDTH)
    ap.add_argument("--drop", type=float, default=SHOULDER_DROP)
    ap.add_argument("--feather", type=float, default=1.5)
    ap.add_argument("--erode", type=int, default=1)
    ap.add_argument("--debug", action="store_true", help="mark the shoulder points")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"{args.image}: not found", file=sys.stderr)
        return 1

    from face_detect import detect

    face = detect(args.image)
    if face is None:
        print("no face found — nothing to dress", file=sys.stderr)
        return 2

    frame = Image.open(args.image).convert("RGBA")
    meta_path = args.garment.with_suffix(".json")
    if not meta_path.exists():
        sys.exit(f"{meta_path}: garment needs its anchor metadata beside it")
    meta = json.loads(meta_path.read_text())

    shoulder_left, shoulder_right = estimate_shoulders(
        face, args.half_width, args.drop
    )
    garment = warp_garment(
        Image.open(args.garment), meta, shoulder_left, shoulder_right, frame.size
    )

    crop_file = None
    if args.matte:
        crop_file = args.crop or args.matte.parent / "crop.json"
        if not crop_file.exists():
            sys.exit(f"{crop_file}: needed to place the matte; run make_face_raw first")

    clip = None
    if args.body:
        # Clipping and occlusion live in garment_clip.py; placement above is
        # untouched by them, which is the point of the split.
        import garment_clip
        from person_matte import body_masks

        body_crop = args.body_crop or args.body.parent / "person_crop.json"
        if not body_crop.exists():
            sys.exit(f"{body_crop}: needed to place the body mask; "
                     "run make_person_raw first")
        masks = body_masks(args.body, body_crop, frame.size,
                           feather=args.body_feather, grow=args.clip_grow)
        head = head_alpha(frame, args.matte, crop_file, args.feather, args.erode) \
            if crop_file else None
        composed, clip = garment_clip.compose(frame, garment, masks, head)
    else:
        composed = Image.alpha_composite(frame, garment)
        if crop_file:
            composed = Image.alpha_composite(
                composed,
                head_layer(frame, args.matte, crop_file, args.feather, args.erode),
            )

    if args.debug:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(composed)
        for point in (shoulder_left, shoulder_right):
            x, y = point
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 0, 0, 255), width=3)
        draw.line([tuple(shoulder_left), tuple(shoulder_right)],
                  fill=(255, 0, 0, 255), width=2)

    composed.convert("RGB").save(args.output)

    width = float(np.linalg.norm(np.asarray(shoulder_right) - np.asarray(shoulder_left)))
    print(f"  face       {face['bbox']}  score {face['score']:.2f}")
    print(f"  shoulders  {width:.0f} px apart")
    occluders = [n for n, on in (("head", args.matte),
                                 ("arms", clip and clip.arms_available)) if on]
    print(f"  occlusion  {', '.join(occluders) if occluders else 'none'}")
    if clip:
        print(f"  clip       {clip.kept:.0f}% of the garment is on the body"
              f"  ({clip.occluded_px:,} px behind an occluder)")
        if clip.extrapolated > 0.02:
            print(f"  note       {clip.extrapolated:.0%} of the garment lies outside "
                  "the body crop, clipped against a replicated edge")
    print(f"  out        {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
