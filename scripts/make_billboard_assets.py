#!/usr/bin/env python3
"""Generate the virtual-billboard backdrop, garment and collar.

    python scripts/make_billboard_assets.py -o web/public/billboard

Drawn programmatically rather than shipped as photos: no licensing questions,
and the anchor points are exact instead of eyeballed. The store is fictional on
purpose — rendering a real retailer's branding onto a stranger's face is
impersonation, whatever the demo.

Three layers, composited in this order so the cut edge at the neck is hidden:

    backdrop  ->  body (shirt below the neckline)  ->  HEAD  ->  collar

Putting the collar last is the whole trick. The head cutout ends in a ragged
neck, and tucking that under a collar drawn on top of it reads as someone
wearing the shirt rather than a head pasted onto one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1350

# Where the head belongs, and how big. The daemon scales the cutout so the
# subject's eye line lands on EYE_Y with EYE_SPAN pixels between the pupils,
# which keeps a near face and a far face the same size on the billboard.
#
# Strictly anatomical scaling (shoulder width ~7x pupil distance) gives a head
# that reads as tiny here, because the drawn tee is deliberately oversized and
# fills the frame. A billboard is a portrait, not a figure study, so the head is
# scaled up past life-size and the garment trimmed to match. Raise EYE_SPAN for
# a bigger head; the chin lands ~1.2 eye-spans below EYE_Y, so keep
# NECK_Y - EYE_Y a little over that or the neck stops short of the collar.
NECK_X, NECK_Y = W // 2, 648
EYE_Y = 428
EYE_SPAN = 128

BRAND = "NORTHWIND"
TAGLINE = "THE EVERYDAY TEE"
PRICE = "$28"

SHIRT = (38, 70, 102)
SHIRT_SHADOW = (26, 50, 76)
COLLAR = (30, 58, 86)


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Futura.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_backdrop() -> Image.Image:
    """Store wall: a warm vertical gradient with branding and a floor line."""
    img = Image.new("RGB", (W, H), (236, 232, 226))
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        d.line(
            [(0, y), (W, y)],
            fill=(
                int(242 - 34 * t),
                int(238 - 32 * t),
                int(231 - 28 * t),
            ),
        )
    # Floor
    d.rectangle([0, int(H * 0.86), W, H], fill=(206, 199, 190))
    d.line([(0, int(H * 0.86)), (W, int(H * 0.86))], fill=(188, 180, 170), width=3)

    d.text((60, 56), BRAND, font=font(64, bold=True), fill=(28, 32, 38))
    d.text((62, 132), TAGLINE, font=font(26), fill=(104, 110, 118))

    price_f = font(40, bold=True)
    d.text((W - 60 - d.textlength(PRICE, font=price_f), 60), PRICE,
           font=price_f, fill=(28, 32, 38))
    return img


def _torso_outline(d: ImageDraw.ImageDraw, fill) -> None:
    """A tee silhouette: sloping shoulders, angled sleeves, slight waist taper.

    The first version used a wide box with round ellipse sleeve caps, which
    read as a sack. Real tees drop from the shoulder at an angle and the sleeve
    hem sits lower on the outside than the armpit — those two facts do most of
    the work.
    """
    hem_y = int(H * 0.93)
    nx, ny = NECK_X, NECK_Y

    def side(sign: int) -> list[tuple[int, int]]:
        return [
            (nx + sign * 112, ny - 8),        # neckline
            (nx + sign * 236, ny + 34),       # shoulder point
            (nx + sign * 338, ny + 206),      # sleeve hem, outer
            (nx + sign * 258, ny + 250),      # armpit
            (nx + sign * 276, hem_y),         # hem
        ]

    right = side(+1)
    left = list(reversed(side(-1)))
    d.polygon(left + right, fill=fill)


def make_body() -> Image.Image:
    """The shirt, with nothing where the head and neck go."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    _torso_outline(d, SHIRT + (255,))

    # Soft shading down the sides and under the collar so it is not flat.
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    hem_y = int(H * 0.93)
    for sign in (-1, 1):
        sd.polygon(
            [
                (NECK_X + sign * 236, NECK_Y + 40),
                (NECK_X + sign * 150, NECK_Y + 90),
                (NECK_X + sign * 182, hem_y),
                (NECK_X + sign * 272, hem_y),
            ],
            fill=SHIRT_SHADOW + (110,),
        )
    sd.ellipse([NECK_X - 150, NECK_Y - 30, NECK_X + 150, NECK_Y + 150],
               fill=SHIRT_SHADOW + (90,))
    shade = shade.filter(ImageFilter.GaussianBlur(30))
    img = Image.alpha_composite(img, shade)

    # Punch out the neck hole so the head sits *in* the shirt.
    hole = Image.new("L", (W, H), 255)
    hd = ImageDraw.Draw(hole)
    hd.ellipse([NECK_X - 124, NECK_Y - 84, NECK_X + 124, NECK_Y + 60], fill=0)
    alpha = img.getchannel("A")
    img.putalpha(Image.composite(alpha, Image.new("L", (W, H), 0), hole))
    return img


def make_shoulders() -> Image.Image:
    """The yoke: shoulders and sleeve tops, drawn *after* the head.

    Splitting this off the body is what makes the neck sit *in* the shirt. The
    body goes down first, the head on top of it, then this band over the head's
    lower edge — so the sides of the neck are occluded by fabric the way a real
    collar area occludes them, instead of the neck ending in a straight cut.
    """
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    _torso_outline(d, SHIRT + (255,))

    # Keep only the band around the shoulders.
    band = Image.new("L", (W, H), 0)
    ImageDraw.Draw(band).rectangle([0, NECK_Y - 20, W, NECK_Y + 170], fill=255)
    band = band.filter(ImageFilter.GaussianBlur(10))
    img.putalpha(ImageChops.multiply(img.getchannel("A"), band))

    # Same neck hole as the body, so the neck still shows through the opening.
    hole = Image.new("L", (W, H), 255)
    ImageDraw.Draw(hole).ellipse(
        [NECK_X - 124, NECK_Y - 84, NECK_X + 124, NECK_Y + 60], fill=0
    )
    img.putalpha(ImageChops.multiply(img.getchannel("A"), hole))
    return img


def make_collar() -> Image.Image:
    """The ribbed collar, drawn last so the neck seam disappears under it."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([NECK_X - 138, NECK_Y - 92, NECK_X + 138, NECK_Y + 74],
              outline=COLLAR + (255,), width=34)
    # Hide the top half: only the front of the collar should cross the neck.
    cut = Image.new("L", (W, H), 255)
    ImageDraw.Draw(cut).rectangle([0, 0, W, NECK_Y - 34], fill=0)
    img.putalpha(Image.composite(img.getchannel("A"), Image.new("L", (W, H), 0), cut))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate billboard layers")
    ap.add_argument("-o", "--output", type=Path, default=Path("web/public/billboard"))
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    make_backdrop().save(out / "backdrop.png")
    make_body().save(out / "body.png")
    make_shoulders().save(out / "shoulders.png")
    make_collar().save(out / "collar.png")
    (out / "billboard.json").write_text(json.dumps({
        "width": W, "height": H,
        "neck": [NECK_X, NECK_Y],
        "eyeY": EYE_Y,
        "eyeSpan": EYE_SPAN,
        "brand": BRAND,
    }, indent=2))

    print(f"  wrote {out}/backdrop.png, body.png, shoulders.png, collar.png, billboard.json")
    print(f"  canvas {W}x{H}  neck {NECK_X},{NECK_Y}  eyeY {EYE_Y}  eyeSpan {EYE_SPAN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
