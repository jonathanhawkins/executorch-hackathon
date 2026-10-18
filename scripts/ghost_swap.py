#!/usr/bin/env python3
"""fp32 reference for the GHOST face swap — what the NPU is compared against.

    python scripts/ghost_swap.py --source me.jpg --target tee.png -o out.png
    python scripts/ghost_swap.py --source me.jpg --bench

Runs both models eager on the CPU and writes the swapped photo, plus
`output_0.raw` — the same (1, 3, 256, 256) float32 the QNN runner produces on
the board. Both paths then go through the same paste-back, so the only
difference between them is where the arithmetic happened. That is the point:
it makes the quantization cost measurable rather than assumed.

Two models, in order:

    ArcFace iresnet100   112x112 -> 512-d      the person's identity
    AEI-Net (GHOST)      256x256 + 512-d       the swapped face

Only the second is on the clock at a billboard: the identity vector describes
whoever is standing there, not the instant the shutter fires, so it can be
computed while they are walking up. See docs/faceswap-npu.md.

This is the Apache-2.0 alternative to scripts/face_swap.py, which is the same
idea with inswapper_128 — better looking, but with non-commercial weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "ghost"

RELEASE = "https://github.com/sberbank-ai/sber-swap/releases/download"
# Release assets of the Apache-2.0 repository. The identity encoder is the one
# worth being careful about — see vendor/ghost/__init__.py for why it is not
# InsightFace's non-commercial r100, and how that was established.
WEIGHTS = {
    "G_unet_2blocks.pth": (
        f"{RELEASE}/sber-swap-v2.0/G_unet_2blocks.pth",
        "85f130421c65722927c47502ea1b28b62be4893ca9180fec735b67be1e5edd8d",
    ),
    "backbone.pth": (
        f"{RELEASE}/arcface/backbone.pth",
        "a566a62357f0c55b679d9ff2f022a294486568be0c00665d39029d0e46a8109b",
    ),
}

DEFAULT_TARGET = ROOT / "assets/templates/model-navy-tee.png"


def weight_path(name: str) -> Path:
    """Fetch on first use, then cache. Digest checked every time."""
    url, want = WEIGHTS[name]
    path = MODEL_DIR / name
    if not path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"fetching {name} (once)", file=sys.stderr)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=300) as resp:
            tmp.write_bytes(resp.read())
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != want:
        sys.exit(f"{name} digest mismatch: {digest}")
    return path


# The blend ellipse, as fractions of the aligned crop. Centre + axis must stay
# below 1.0 or the mask runs off the generated image; see paste_back.
ELLIPSE_CENTRE_Y = 0.53
ELLIPSE_AXIS_X = 0.38
ELLIPSE_AXIS_Y = 0.44
assert ELLIPSE_CENTRE_Y + ELLIPSE_AXIS_Y < 0.98, "blend ellipse runs off the crop"


def _border_falloff(crop_size: int):
    """Force the mask to zero before the crop's edge, whatever the ellipse does.

    Belt and braces for the assert above: the ellipse is checked, but the
    warp back into frame coordinates can still clip, and a hard edge here is
    the single most visible failure this function has. Cheap enough to just
    guarantee.
    """
    import cv2

    inset = max(2, int(crop_size * 0.04))
    border = np.zeros((crop_size, crop_size), dtype=np.float32)
    border[inset:-inset, inset:-inset] = 1.0
    return cv2.GaussianBlur(border, (0, 0), crop_size * 0.02)


def paste_back(frame, swapped_bgr, matrix, crop_size: int):
    """Warp the swapped face home and feather it into the photo.

    The mask is an ellipse, not the crop's square, for the same two reasons as
    the inswapper path: the generator repaints the whole aligned box — hair,
    ear, backdrop — and its version of those is close but not identical, so a
    square blend leaves a rectangular halo against a flat studio background.
    The ellipse also encodes the choice the billboard wants: their face, the
    model's hair.

    **The ellipse must stay inside the generated crop.** GHOST renders only
    the ArcFace-aligned box, so a mask that reaches the edge blends in that
    box's own border — a hard straight line across the face, which is far
    worse than anything it was widened to capture. The constants below are
    checked against that at import time.

    This was got wrong once, chasing a bearded subject's jaw: centre 0.56 with
    a 0.49 vertical axis puts the ellipse's bottom at 1.05 of the crop, 11 px
    past the edge at 224. It looked fine on the subject it was tuned on, whose
    alignment happened to rotate the overflow out of frame, and produced an
    obvious diagonal seam on the next one.

    The beard is not reachable this way regardless. ArcFace's template puts
    the mouth at 0.82 of the crop, so the chin and the beard below it sit at
    the very edge of what the generator ever draws. Widening the mask to chase
    them necessarily runs off it. That is a property of the crop, not a
    tuning problem — docs/faceswap-npu.md has what to do instead.
    """
    import cv2

    swapped = cv2.resize(swapped_bgr, (crop_size, crop_size),
                         interpolation=cv2.INTER_LINEAR)

    mask = np.zeros((crop_size, crop_size), dtype=np.float32)
    cv2.ellipse(
        mask,
        center=(crop_size // 2, int(crop_size * ELLIPSE_CENTRE_Y)),
        axes=(int(crop_size * ELLIPSE_AXIS_X), int(crop_size * ELLIPSE_AXIS_Y)),
        angle=0, startAngle=0, endAngle=360, color=1.0, thickness=-1,
    )
    mask = cv2.GaussianBlur(mask, (0, 0), crop_size * 0.06)
    mask *= _border_falloff(crop_size)

    h, w = frame.shape[:2]
    matrix = np.asarray(matrix, dtype=np.float32)
    inverse = cv2.invertAffineTransform(matrix)

    # Match the generated face to the photo's own colour before blending.
    # GHOST renders in the target's lighting but not in its exact skin tone,
    # and the mismatch shows up as a line where the rendered box ends — the
    # one seam a feathered ellipse cannot hide, because it is a difference in
    # content rather than a hard edge. Matching against the crop the generator
    # was *given* costs one warp and no arguments.
    original = cv2.warpAffine(frame, matrix, (crop_size, crop_size))
    swapped = _match_tone(swapped, original, mask > 0.5)

    warped = cv2.warpAffine(swapped, inverse, (w, h), borderValue=0.0)
    alpha = cv2.warpAffine(mask, inverse, (w, h), borderValue=0.0)[:, :, None]

    return (warped * alpha + frame.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def _match_tone(swapped, original, where):
    """Shift `swapped` toward `original`'s colour, over the blended pixels.

    Mean in full, spread only halfway: matching the standard deviation
    outright flattens the generated skin texture, which costs more than the
    tone mismatch it fixes. LAB rather than BGR so lightness moves
    independently of hue.
    """
    import cv2

    if where.sum() < 500:  # too little overlap to estimate anything from
        return swapped

    out = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    for channel in range(3):
        a, b = out[:, :, channel], ref[:, :, channel]
        mean_a, mean_b = a[where].mean(), b[where].mean()
        std_a, std_b = a[where].std() + 1e-6, b[where].std() + 1e-6
        out[:, :, channel] = (a - mean_a) * (1 + 0.5 * (std_b / std_a - 1)) + mean_b
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def to_bgr(y: np.ndarray) -> np.ndarray:
    """AEI-Net output (1, 3, 256, 256) in [-1, 1] -> BGR uint8."""
    img = y[0].transpose(1, 2, 0)          # CHW -> HWC, RGB
    img = (img * 0.5 + 0.5) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)[:, :, ::-1]


def load_models():
    """The identity encoder and the generator, both eager and both in eval.

    The generator comes back wrapped in `GeneratorForExport`, which drops the
    eight attribute tensors `AEI_Net.forward` also returns. That is not just
    tidiness: it makes this reference the *same module* that gets exported, so
    a difference between the two paths is quantization and nothing else.
    """
    import torch
    from vendor.ghost import GeneratorForExport, load_generator, load_identity_encoder

    torch.set_grad_enabled(False)
    arc = load_identity_encoder(weight_path("backbone.pth"))
    gen = GeneratorForExport(load_generator(weight_path("G_unet_2blocks.pth"))).eval()
    return arc, gen


def main() -> int:
    import cv2
    import torch

    import make_ghost_raw as prep

    ap = argparse.ArgumentParser(description="GHOST face swap, fp32 on CPU")
    ap.add_argument("--source", type=Path, required=True,
                    help="the live frame: whoever is standing there")
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                    help="the product photo their face goes into")
    ap.add_argument("-o", "--out", type=Path, default=Path("ghost_swapped.png"))
    ap.add_argument("--raw", type=Path, default=None,
                    help="also write the raw (1,3,256,256) float32 reference here")
    ap.add_argument("--bench", action="store_true", help="per-stage timings")
    args = ap.parse_args()

    source = cv2.imread(str(args.source))
    target = cv2.imread(str(args.target))
    if source is None:
        return ap.error(f"could not read {args.source}")
    if target is None:
        return ap.error(f"could not read {args.target}")

    timings: dict = {}

    def clock(name, fn):
        start = time.time()
        value = fn()
        timings[name] = (time.time() - start) * 1000
        return value

    arc, gen = clock("load", load_models)

    source_crop, _, _ = clock(
        "detect_source", lambda: prep.crop_for(source, "source frame"))
    target_crop, matrix, _ = clock(
        "detect_target", lambda: prep.crop_for(target, "target photo"))

    identity_in = torch.from_numpy(prep.identity_tensor(source_crop))
    target_in = torch.from_numpy(prep.target_tensor(target_crop))

    arc(identity_in)                                        # warm
    embedding = clock("embed", lambda: arc(identity_in))
    gen(target_in, embedding)                               # warm
    y = clock("swap", lambda: gen(target_in, embedding))

    y_np = y.numpy().astype(np.float32)
    result = clock("paste", lambda: paste_back(
        target, to_bgr(y_np), matrix, prep.CROP_SIZE))

    cv2.imwrite(str(args.out), result)
    print(f"  out    {args.out}")

    if args.raw:
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        y_np.tofile(args.raw)
        embedding.numpy().astype(np.float32).tofile(
            args.raw.with_name(args.raw.stem + "_embedding.raw"))
        print(f"  raw    {args.raw}  {tuple(y_np.shape)}")

    if args.bench:
        print(json.dumps({k: round(v, 1) for k, v in timings.items()}, indent=2))
        shutter = timings["swap"] + timings["paste"]
        print(f"  shutter to picture: {shutter / 1000:.2f} s "
              f"(identity is precomputed while they walk up)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
