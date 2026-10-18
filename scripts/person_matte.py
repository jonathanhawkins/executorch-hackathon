#!/usr/bin/env python3
"""Turn person-segmentation logits into frame-sized masks.

    python scripts/person_matte.py out/output_0.raw frame.jpg -o body.png
    python scripts/person_matte.py out/output_0.raw frame.jpg --classes

Mirrors scripts/face_matte.py, and takes logits from either path — the QNN
runner on the board or scripts/person_seg.py on the Mac.

The one idea worth knowing is how the mask is enlarged. These logits are small:
SegFormer's are 128x128 for a 512x512 input, and over a 700 px body crop in a
1280x720 frame that is one mask pixel per 5.5 frame pixels. Taking the argmax
first and resizing the resulting binary mask gives a staircase: measured along
one shoulder, treads averaging 13.8 px and running up to 44. That is visible,
and on a billboard it is the thing that gives the compositing away.

So the argmax is done *last*. The class logits are reduced to one signed field
per mask — "person minus background", "arm minus everything else" — that field
is resized bilinearly, and the threshold is applied at zero on the enlarged
field. The boundary then lands wherever the two scores cross, which is at
sub-mask-pixel precision, and the edge comes out smooth. It costs one resize of
a single-channel float image and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from person_seg import MODELS, SegModel  # noqa: E402


@dataclass(frozen=True)
class BodyMasks:
    """Everything the compositor needs, as frame-sized 8-bit alphas."""

    silhouette: Image.Image        # the whole person
    arms: Image.Image              # forearms and hands, the occluders
    worn: Image.Image              # the garment already being worn
    head: Image.Image              # hair, face, hat — for seam bookkeeping
    crop: dict                     # the region the model saw
    model: SegModel

    @property
    def has_parts(self) -> bool:
        """False for a single-blob model: no arms to put back over anything."""
        return bool(self.model.arms)


def load_logits(path: Path, model: SegModel) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(model.logit_shape))
    if raw.size != expected:
        sys.exit(
            f"{path}: expected {expected} floats for {model.key} "
            f"{model.logit_shape}, got {raw.size}"
        )
    return raw.reshape(model.logit_shape)


def _field(logits: np.ndarray, wanted: tuple[int, ...]) -> np.ndarray:
    """Signed score: how much more this pixel looks like `wanted` than not.

    Positive inside, negative outside, and — unlike a binary mask — it varies
    smoothly across the boundary, which is what makes the resize below work.
    """
    if not wanted:
        return np.full(logits.shape[1:], -1.0, dtype=np.float32)
    rest = tuple(i for i in range(logits.shape[0]) if i not in wanted)
    # np.maximum.reduce over per-channel views rather than logits[list(wanted)],
    # which would copy 17 of the 18 channels just to reduce them. At 128x128
    # the time is the same either way; the allocation is not.
    inside = np.maximum.reduce([logits[i] for i in wanted])
    outside = np.maximum.reduce([logits[i] for i in rest]) if rest \
        else np.zeros_like(inside)
    return inside - outside


def _to_frame(field: np.ndarray, crop: dict, frame_size: tuple[int, int],
              feather: float, grow: int) -> Image.Image:
    """Resize the signed field onto the frame, threshold, feather.

    Outside the crop the model said nothing, and only one direction deserves a
    guess. Below the crop the body genuinely continues — cut it off there and a
    ruler-straight line appears across the garment halfway down the chest,
    which is worse than any mask error. So the bottom edge is replicated
    downwards. Left, right and above are filled with background instead: a
    crop sized in face widths is wider than the garment, so there is nothing
    out there to keep, and replicating sideways smears the edge column across
    a third of the frame.
    """
    import cv2

    width, height = crop["width"], crop["height"]
    frame_w, frame_h = frame_size
    left, top = crop["left"], crop["top"]

    enlarged = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)
    below = max(0, frame_h - top - height)
    if below:
        enlarged = cv2.copyMakeBorder(enlarged, 0, below, 0, 0, cv2.BORDER_REPLICATE)

    mask = (enlarged > 0).astype(np.uint8) * 255
    if grow:
        kernel = np.ones((2 * abs(grow) + 1,) * 2, np.uint8)
        mask = cv2.dilate(mask, kernel) if grow > 0 else cv2.erode(mask, kernel)

    full = np.zeros((frame_h, frame_w), np.uint8)
    rows = min(mask.shape[0], frame_h - top)
    cols = min(width, frame_w - left)
    full[top:top + rows, left:left + cols] = mask[:rows, :cols]

    if feather:
        # Blur only the band the mask can reach, not the whole frame, and with
        # cv2 rather than PIL — together that is most of this function's cost,
        # and it runs once per mask per frame.
        pad = int(np.ceil(3 * feather)) + 1
        y0, y1 = max(0, top - pad), min(frame_h, top + rows + pad)
        x0, x1 = max(0, left - pad), min(frame_w, left + cols + pad)
        full[y0:y1, x0:x1] = cv2.GaussianBlur(full[y0:y1, x0:x1], (0, 0), feather)
    return Image.fromarray(full, mode="L")


def body_masks(logits_path: Path, crop_file: Path, frame_size: tuple[int, int],
               *, feather: float = 2.0, grow: int = 0,
               with_head: bool = False) -> BodyMasks:
    """The whole decode, from a .raw on disk to frame-sized alphas.

    `with_head` is off by default. The head class is only ever used for
    reporting — the BiSeNet matte is the one that composites — and each mask
    costs a full resize and blur, which in a live loop is not free.
    """
    crop = json.loads(crop_file.read_text())
    model = MODELS[crop.get("model", "segformer-b0-atr")]
    logits = load_logits(logits_path, model)
    empty = Image.new("L", frame_size, 0)

    def alpha(wanted, extra_grow=0):
        if not wanted:
            return empty
        return _to_frame(_field(logits, wanted), crop, frame_size,
                         feather, grow + extra_grow)

    return BodyMasks(
        # The silhouette is grown a little on purpose. It is the clip boundary,
        # and the failure modes are not symmetric: a mask a pixel too wide
        # leaves a sliver of garment over the background, while a mask a pixel
        # too narrow eats the garment's own edge and puts a halo of the real
        # shirt around the new one.
        silhouette=alpha(model.silhouette, extra_grow=2),
        arms=alpha(model.arms),
        worn=alpha(model.worn),
        head=alpha(model.head) if with_head else empty,
        crop=crop,
        model=model,
    )


def parse_map(logits: np.ndarray) -> np.ndarray:
    return logits.argmax(0).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description="Person logits -> body masks")
    ap.add_argument("logits", type=Path, help="output_0.raw")
    ap.add_argument("image", type=Path, help="the source frame")
    ap.add_argument("-o", "--output", type=Path, default=Path("body.png"))
    ap.add_argument("--crop", type=Path, help="person_crop.json; defaults to beside it")
    ap.add_argument("--feather", type=float, default=2.0)
    ap.add_argument("--grow", type=int, default=0)
    ap.add_argument("--classes", action="store_true", help="also write a parse map")
    args = ap.parse_args()

    crop_file = args.crop or args.logits.parent / "person_crop.json"
    if not crop_file.exists():
        sys.exit(f"{crop_file}: needed to place the mask; run make_person_raw first")

    frame = Image.open(args.image).convert("RGB")
    masks = body_masks(args.logits, crop_file, frame.size,
                       feather=args.feather, grow=args.grow, with_head=True)

    cutout = frame.convert("RGBA")
    cutout.putalpha(masks.silhouette)
    cutout.save(args.output)

    crop = masks.crop
    logits = load_logits(args.logits, masks.model)
    parse = parse_map(logits)
    present = [masks.model.classes[i] for i in sorted(np.unique(parse).tolist())]

    if args.classes:
        palette = np.random.default_rng(1).integers(
            40, 255, (masks.model.n_classes, 3)
        ).astype(np.uint8)
        palette[0] = 0
        vis = Image.fromarray(palette[parse]).resize(
            (crop["width"], crop["height"]), Image.NEAREST
        )
        full = Image.new("RGB", frame.size, (0, 0, 0))
        full.paste(vis, (crop["left"], crop["top"]))
        full.save(args.output.with_name(args.output.stem + "_classes.png"))

    def coverage(mask):
        return 100.0 * float(np.asarray(mask, np.float32).mean()) / 255.0

    scale = crop["width"] / masks.model.logit_shape[2]
    print(f"  model      {masks.model.key}")
    print(f"  classes    {present}")
    print(f"  crop       {crop['width']}x{crop['height']} at "
          f"({crop['left']}, {crop['top']})  "
          f"-> {scale:.1f} frame px per mask px")
    print(f"  silhouette {coverage(masks.silhouette):.1f}% of frame")
    print(f"  arms       {coverage(masks.arms):.1f}%"
          f"{'' if masks.has_parts else '  (model has no arm class)'}")
    print(f"  out        {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
