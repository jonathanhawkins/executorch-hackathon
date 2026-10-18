#!/usr/bin/env python3
"""Clip a warped garment to the body, and put arms and hands back in front of it.

    from garment_clip import clip_and_occlude
    clipped, occluder, stats = clip_and_occlude(frame, garment, masks, head)
    composed = Image.alpha_composite(Image.alpha_composite(frame, clipped), occluder)

Deliberately separate from placement. `garment_overlay.py` decides *where* the
shirt goes — scale, rotation, shoulder points — and this decides *which of its
pixels survive*. The two fail independently: placement that is a few degrees
off still reads as worn once the garment is cut to the real outline, and that
is the point of doing this at all. Nothing here touches the transform.

Two operations, in this order:

1. **Clip.** The garment's alpha is multiplied by the person silhouette, so
   nothing of it lands on the background. This is what stops a slightly
   mis-scaled shirt from hanging in the air beside a shoulder.
2. **Occlude.** Arms, hands and the head are drawn back over the garment from
   the original frame, so a forearm crossing the chest stays in front of the
   shirt instead of vanishing under it.

## Where the body mask and the head matte disagree

They overlap across the neck and jaw and they will not agree at the edges, so
one of them has to win:

* **The head matte wins, everywhere except one strip.** It is far the finer
  measurement. BiSeNet runs 512x512 over a face crop 1.6x the face box — about
  2.3 of its pixels per frame pixel on a 141 px face — while the body mask is
  128x128 over a ~700 px body crop, about 5.5 frame pixels per mask pixel.
  Along an edge that is about 13x the resolution, at the hairline, which is
  exactly where an alpha edge gets judged. The body mask has no opinion worth hearing
  about individual hairs.
* **The garment wins inside `Upper-clothes`.** The head matte's classes run
  down to wherever the subject's existing shirt starts — `neck` is a head
  class — so drawing it over the garment punches a hole through the new
  collar and shows the old one through it. But that strip is precisely the
  region the demo is replacing, and the body mask is the only thing that knows
  where it is. So the head matte's alpha is suppressed wherever the body model
  says `Upper-clothes`, and the new collar survives.

The second rule is the one that needed the new model; without it the two
mattes argue over the neck and the old collar wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ClipStats:
    """What the clip actually did, for printing and for judging it."""

    garment_px: int
    clipped_px: int
    occluded_px: int
    extrapolated: float     # fraction of garment alpha outside the analysed crop
    arms_available: bool

    @property
    def kept(self) -> float:
        return 100.0 * self.clipped_px / self.garment_px if self.garment_px else 0.0


def _sub(img: Image.Image, box) -> np.ndarray:
    """A float32 view of one channel inside `box`, scaled to 0..1."""
    x0, y0, x1, y1 = box
    return np.asarray(img)[y0:y1, x0:x1].astype(np.float32) * (1.0 / 255.0)


def clip_and_occlude(frame: Image.Image, garment: Image.Image, masks,
                     head: Image.Image | None = None, *,
                     hold_collar: bool = True, stats: bool = True):
    """Clip `garment` to the body and build the layer that goes back on top.

    `masks` is a `person_matte.BodyMasks`; `head` is the BiSeNet head alpha as
    an "L" image at frame size, or None when the face parser was not run.

    Returns `(clipped_garment, occluder, stats)`. Both layers are RGBA at frame
    size and are meant to be composited in that order. `stats` is None when the
    caller asked for it to be skipped — it is four reductions over the garment
    box, which a live loop does not want every frame.

    All of the per-pixel work happens inside the garment's own alpha bounding
    box, and that is the shape of the problem rather than an optimisation: the
    clip can only remove garment pixels, and the occluder only matters where
    there is garment under it — drawing the frame back over itself anywhere
    else is a no-op that still costs a full-frame pass.
    """
    garment = garment.convert("RGBA")
    alpha_img = garment.getchannel("A")
    box = alpha_img.getbbox()
    if box is None:                                   # nothing to clip
        empty = Image.new("L", frame.size, 0)
        occluder = frame.convert("RGBA")
        occluder.putalpha(empty)
        return garment, occluder, (ClipStats(0, 0, 0, 0.0, masks.has_parts)
                                   if stats else None)

    x0, y0, x1, y1 = box
    garment_alpha = _sub(alpha_img, box)
    clipped_alpha = garment_alpha * _sub(masks.silhouette, box)

    occluder_alpha = _sub(masks.arms, box)
    if head is not None:
        head_alpha = _sub(head, box)
        if hold_collar and masks.model.worn:
            # The tie-break described in the module docstring: inside the
            # subject's existing shirt, the new garment wins over the head
            # matte.
            head_alpha = head_alpha * (1.0 - _sub(masks.worn, box))
        occluder_alpha = np.maximum(head_alpha, occluder_alpha)

    new_alpha = np.zeros((frame.height, frame.width), np.uint8)
    new_alpha[y0:y1, x0:x1] = clipped_alpha * 255.0
    clipped = garment
    clipped.putalpha(Image.fromarray(new_alpha, mode="L"))

    occ = np.zeros((frame.height, frame.width), np.uint8)
    occ[y0:y1, x0:x1] = occluder_alpha * 255.0
    occluder = frame.convert("RGBA")
    occluder.putalpha(Image.fromarray(occ, mode="L"))

    return clipped, occluder, (_stats(garment_alpha, clipped_alpha, occluder_alpha,
                                      box, masks) if stats else None)


def _stats(garment_alpha, clipped_alpha, occluder_alpha, box, masks) -> ClipStats:
    # How much of the garment sits outside the region the model actually saw.
    # Those pixels are clipped against a replicated border rather than a
    # measurement (see person_matte._to_frame), so it is worth reporting.
    x0, y0, x1, y1 = box
    crop = masks.crop
    inside = np.zeros_like(garment_alpha)
    iy0 = max(0, crop["top"] - y0)
    ix0 = max(0, crop["left"] - x0)
    inside[iy0:crop["top"] + crop["height"] - y0,
           ix0:crop["left"] + crop["width"] - x0] = 1.0
    total = float(garment_alpha.sum())
    return ClipStats(
        garment_px=int(round(total)),
        clipped_px=int(round(float(clipped_alpha.sum()))),
        occluded_px=int(round(float((clipped_alpha * occluder_alpha).sum()))),
        extrapolated=float((garment_alpha * (1.0 - inside)).sum()) / max(total, 1.0),
        arms_available=masks.has_parts,
    )


def compose(frame: Image.Image, garment: Image.Image, masks,
            head: Image.Image | None = None, **kwargs):
    """clip_and_occlude, composited. The one-call form for a live loop."""
    clipped, occluder, stats = clip_and_occlude(frame, garment, masks, head, **kwargs)
    out = Image.alpha_composite(frame.convert("RGBA"), clipped)
    return Image.alpha_composite(out, occluder), stats
