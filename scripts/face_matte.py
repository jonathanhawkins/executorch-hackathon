#!/usr/bin/env python3
"""Turn BiSeNet logits into a head/neck cutout.

    python scripts/face_matte.py out/output_0.raw portrait.jpg -o head.png

Takes the (1, 19, 512, 512) float32 logits that either path produces — the NPU
runner on the board or scripts/face_parse.py on the Mac — and writes an RGBA
PNG of the head with everything else transparent, plus a `_classes.png`
visualisation of the raw parse when asked.

The cutout is every class except background and `cloth`: skin, hair, ears,
neck, and anything worn on the head. `cloth` is excluded deliberately — it is
the shirt being replaced.

Two post-processing steps earn their place:

* **Keep one connected region.** Removes stray specks, and would remove a
  detached mislabelled blob. Note what it does *not* fix: BiSeNet labels flat,
  uniform background next to the crop as `hair`, and on a plain grey backdrop
  that slab comes out touching the hair — one 234k-pixel component, so this
  pass keeps all of it. Whether a real, textured scene behaves the same way is
  untested; see docs/faceparse-npu.md.
* **Feather the edge.** An argmax gives a hard, aliased boundary, and hair
  against a bright store wall is exactly where that reads as a cut-out-and-
  pasted head. Eroding a pixel and blurring the alpha costs nothing and hides
  most of it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vendor.face_parsing import CLASSES, HEAD_CLASSES  # noqa: E402

SIZE = 512
N_CLASSES = 19


def parse_map(logits_path: Path) -> np.ndarray:
    logits = np.fromfile(logits_path, dtype=np.float32)
    expected = N_CLASSES * SIZE * SIZE
    if logits.size != expected:
        sys.exit(f"{logits_path}: expected {expected} floats, got {logits.size}")
    return logits.reshape(N_CLASSES, SIZE, SIZE).argmax(0).astype(np.uint8)


def largest_component(mask: np.ndarray, seed: tuple[int, int] | None) -> np.ndarray:
    """Keep the region the face is in, dropping detached mislabelled blobs."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:                                   # background plus one region
        return mask

    keep = None
    if seed is not None:
        x, y = seed
        if 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]:
            candidate = int(labels[y, x])
            keep = candidate if candidate != 0 else None
    if keep is None:
        # No usable seed: the biggest region is the best guess available.
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    return np.where(labels == keep, 255, 0).astype(np.uint8)


def alpha_from_parse(parse, feather: float, erode: int, seed=None) -> Image.Image:
    mask = np.isin(parse, HEAD_CLASSES).astype(np.uint8) * 255
    mask = largest_component(mask, seed)
    alpha = Image.fromarray(mask, mode="L")
    if erode:
        alpha = alpha.filter(ImageFilter.MinFilter(2 * erode + 1))
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    return alpha


def main() -> int:
    ap = argparse.ArgumentParser(description="BiSeNet logits -> head cutout")
    ap.add_argument("logits", type=Path, help="output_0.raw")
    ap.add_argument("image", type=Path, help="the source image")
    ap.add_argument("-o", "--output", type=Path, default=Path("head.png"))
    ap.add_argument("--feather", type=float, default=1.5, help="alpha blur radius")
    ap.add_argument("--erode", type=int, default=1, help="pixels to pull the edge in")
    ap.add_argument("--classes", action="store_true", help="also write a parse map")
    ap.add_argument("--keep-all", action="store_true",
                    help="skip the connected-component pass")
    args = ap.parse_args()

    parse = parse_map(args.logits)

    source = Image.open(args.image).convert("RGB")
    crop_file = args.logits.parent / "crop.json"
    seed = None
    if crop_file.exists():
        # Crop exactly as make_face_raw.py did, or the matte lands on the
        # wrong pixels — which looks like a model failure and is not one.
        import json

        crop = json.loads(crop_file.read_text())
        source = source.crop(
            (crop["left"], crop["top"],
             crop["left"] + crop["side"], crop["top"] + crop["side"])
        )
        # The nose, in the 512x512 frame the model saw — the one point that is
        # certainly inside the head.
        nose = crop.get("landmarks", {}).get("nose")
        if nose:
            scale = SIZE / crop["side"]
            seed = (int((nose[0] - crop["left"]) * scale),
                    int((nose[1] - crop["top"]) * scale))
    source = source.resize((SIZE, SIZE), Image.BILINEAR)

    alpha = alpha_from_parse(
        parse, args.feather, args.erode, None if args.keep_all else seed
    )
    cutout = source.copy()
    cutout.putalpha(alpha)
    cutout.save(args.output)

    present = sorted(np.unique(parse).tolist())
    covered = 100.0 * np.asarray(alpha).astype(np.float32).mean() / 255.0

    if args.classes:
        palette = (np.random.default_rng(0).integers(40, 255, (N_CLASSES, 3))).astype(np.uint8)
        palette[0] = 0
        vis = Image.fromarray(palette[parse])
        vis.save(args.output.with_name(args.output.stem + "_classes.png"))

    print(f"  classes found  {[CLASSES[i] for i in present]}")
    print(f"  head coverage  {covered:.1f}% of frame")
    print(f"  out            {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
