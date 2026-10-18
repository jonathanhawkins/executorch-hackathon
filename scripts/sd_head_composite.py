#!/usr/bin/env python3
"""Put the real person's head back onto a ControlNet-generated body.

ControlNet follows the source photo's Canny edges, so the body, pose and
garment come out right — but the face does not, because edges carry shape and
not identity. The head is the one part we already have perfect pixels for.

What makes this easy, and what does not work without it: because the
generation was conditioned on *this* photo, the generated head is already in
the same place and at the same scale as the real one. Nothing has to be warped
or matched by eye separation, unlike pasting a head onto a stock product shot
(scripts/garment_overlay.py does the hard version).

Inputs are the BiSeNet matte and crop geometry that
`scripts/board_faceparse_run.sh` already produces:

    make board-cn IMAGE=frame.jpg PROMPT="..." OUT=body.png
    make board-faceparse IMAGE=<the 512 crop make_canny_raw.py saved> OUT=head.png
    python scripts/sd_head_composite.py body.png head.png \\
        --crop .faceparse_stage/run/crop.json -o final.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def feather(alpha, radius):
    """Soften the matte edge so the seam does not read as a cut-out."""
    if radius <= 0:
        return alpha
    kernel = radius * 2 + 1
    return cv2.GaussianBlur(alpha, (kernel, kernel), 0)


def match_illumination(head, generated, alpha, strength):
    """Pull the head's per-channel mean toward the generated head's.

    The generated body carries the prompt's lighting; the real head carries
    the camera's. Matching the means closes most of the gap without touching
    the face's structure. Full strength also drags skin tone toward the
    generated person's, which is usually not wanted — hence a dial.
    """
    if strength <= 0:
        return head
    mask = alpha > 0.5
    if mask.sum() < 100:
        return head
    out = head.astype(np.float32)
    for channel in range(3):
        source_mean = out[:, :, channel][mask].mean()
        target_mean = generated[:, :, channel][mask].mean()
        out[:, :, channel] += strength * (target_mean - source_mean)
    return np.clip(out, 0, 255)


def skin_mask(image):
    """A crude YCrCb skin detector — enough to find arms and neck.

    Its limitation is why --match-body defaults to off: skin and a red shirt
    occupy the same Cr range, so on warm garments this selects the clothing
    too and the correction ruins it. Masking the garment out properly would
    mean BiSeNet's `cloth` class or the person segmenter.
    """
    ycrcb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    return ((cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)).astype(np.uint8)


def segformer_skin_mask(logits, crop, frame_size):
    """Skin from the ATR person parser: arms only, garment explicitly excluded.

    This is what makes --match-body safe. The chrominance detector below
    cannot tell skin from a red shirt; ATR SegFormer was trained to separate
    `Upper-clothes` from `Left-arm`/`Right-arm`, which is the same property
    the billboard relies on to put a forearm back in front of a garment.
    Reuses person_matte.body_masks rather than reimplementing the decode.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from person_matte import body_masks

    masks = body_masks(Path(logits), Path(crop), frame_size, feather=1.0)
    arms = np.asarray(masks.arms, dtype=np.float32) / 255.0
    worn = np.asarray(masks.worn, dtype=np.float32) / 255.0
    return ((arms > 0.5) & (worn < 0.5)).astype(np.uint8)


def match_body_to_head(generated, head_rgb, head_alpha, box, strength, body_skin=None):
    """Shift the generated body's skin toward the real head's skin tone.

    This is the direction that preserves identity: the face is the part we
    have correct pixels for, so the arms and neck should move to meet it
    rather than the other way round. Only pixels the skin detector picks out
    are touched, and the head's own box is excluded so it is not counted as
    its own reference.
    """
    if strength <= 0:
        return generated
    left, top, right, bottom = box
    body = skin_mask(generated) if body_skin is None else body_skin.copy()
    body[top:bottom, left:right] = 0
    face = (head_alpha > 0.5) & (skin_mask(head_rgb) > 0)
    if body.sum() < 500 or face.sum() < 500:
        return generated

    out = generated.astype(np.float32)
    selected = body > 0
    for channel in range(3):
        shift = head_rgb[:, :, channel][face].mean() - out[:, :, channel][selected].mean()
        out[:, :, channel][selected] += strength * shift
    return np.clip(out, 0, 255)


def main(args):
    generated = cv2.imread(args.generated, cv2.IMREAD_COLOR)
    if generated is None:
        raise SystemExit(f"could not read {args.generated}")
    matte = cv2.imread(args.head, cv2.IMREAD_UNCHANGED)
    if matte is None or matte.shape[2] != 4:
        raise SystemExit(f"{args.head} must be an RGBA cut-out from face_matte.py")

    with open(args.crop) as f:
        crop = json.load(f)
    left, top, side = crop["left"], crop["top"], crop["side"]

    # The matte is the face crop blown up to 512; put it back at its real size.
    resized = cv2.resize(matte, (side, side), interpolation=cv2.INTER_AREA)
    head_rgb = resized[:, :, :3].astype(np.float32)
    alpha = resized[:, :, 3].astype(np.float32) / 255.0

    height, width = generated.shape[:2]
    right, bottom = min(left + side, width), min(top + side, height)
    if right <= left or bottom <= top:
        raise SystemExit("the crop does not overlap the generated image")
    head_rgb = head_rgb[: bottom - top, : right - left]
    alpha = alpha[: bottom - top, : right - left]

    alpha = feather(alpha, args.feather)

    body_skin, strength = None, args.match_body
    if args.person_logits and args.person_crop:
        body_skin = segformer_skin_mask(
            args.person_logits, args.person_crop, (width, height)
        )
        # With a real per-pixel label the correction is safe, so it is on
        # unless the caller has explicitly asked for something else.
        if strength == 0.0:
            strength = 0.8
        print(f"skin from ATR SegFormer: {int(body_skin.sum())} px, garment excluded")

    generated = match_body_to_head(
        generated, head_rgb, alpha, (left, top, right, bottom), strength, body_skin
    ).astype(np.uint8)

    region = generated[top:bottom, left:right].astype(np.float32)
    head_rgb = match_illumination(head_rgb, region, alpha, args.match_lighting)

    blended = head_rgb * alpha[:, :, None] + region * (1.0 - alpha[:, :, None])
    out = generated.copy()
    out[top:bottom, left:right] = np.clip(blended, 0, 255).astype(np.uint8)
    cv2.imwrite(args.out, out)

    print(f"head placed at ({left}, {top}) {side}x{side} in a {width}x{height} frame")
    print(f"covering {float((alpha > 0.5).sum()) / (width * height) * 100:.1f}% of it")
    print(f"out {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", help="The ControlNet output")
    parser.add_argument("head", help="RGBA head cut-out from face_matte.py")
    parser.add_argument("--crop", required=True, help="crop.json from make_face_raw.py")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument(
        "--feather", type=int, default=3, help="Matte blur radius in pixels, 0 for none"
    )
    parser.add_argument(
        "--person-logits",
        help="output_0.raw from board_personseg_run.sh on the GENERATED image. "
        "Gives a real skin/garment separation, which makes --match-body safe.",
    )
    parser.add_argument(
        "--person-crop",
        help="person_crop.json beside those logits",
    )
    parser.add_argument(
        "--match-body",
        type=float,
        default=0.0,
        help="Shift the generated body's skin toward the real head's tone — "
        "the direction that keeps the face exact. OFF by default because the "
        "skin detector below is chrominance-based and cannot tell skin from a "
        "red or pink garment, so it wrecks warm-coloured clothes. Worth "
        "turning up (0.8 works) for cool-coloured garments.",
    )
    parser.add_argument(
        "--match-lighting",
        type=float,
        default=0.2,
        help="0 keeps the head's own colour, 1 fully matches the generated "
        "body's. Partial is usually right: it closes the lighting gap without "
        "dragging skin tone to the generated person's.",
    )
    main(parser.parse_args())
