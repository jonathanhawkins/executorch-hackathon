#!/usr/bin/env python3
"""Turn a photo into the Canny edge tensor ControlNet conditions on.

Writes image_cond.raw, a [1, 512, 512, 3] uint8 buffer the ControlNet graph
takes directly — pixel space, not latent space, which is why this path needs
no VAE encoder. Runs on the Mac, the same split as make_sd_tokens.py and
make_face_raw.py.

The thresholds are the whole game. Too few edges and the model is barely
constrained and wanders like plain text-to-image; too many and it traces
sensor noise and fabric weave, and the output comes out scribbly. Defaults
here are the usual ControlNet 100/200, which are a reasonable start for a
studio portrait on a plain background.
"""

import argparse
import os

import cv2
import numpy as np

SIZE = 512  # the context binaries are built for 512x512 and nothing else


def center_crop_square(image):
    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return image[top : top + side, left : left + side]


def main(args):
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read {args.image}")

    if args.no_crop:
        resized = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(
            center_crop_square(image), (SIZE, SIZE), interpolation=cv2.INTER_AREA
        )

    if args.blur:
        # Canny on a slightly blurred frame finds garment and body outlines
        # rather than fabric texture, which is what we want it following.
        resized_for_edges = cv2.GaussianBlur(resized, (args.blur | 1, args.blur | 1), 0)
    else:
        resized_for_edges = resized

    edges = cv2.Canny(resized_for_edges, args.low, args.high)
    # ControlNet-Canny wants the edge map as a 3-channel image, white on black.
    conditioning = np.repeat(edges[:, :, None], 3, axis=2).astype(np.uint8)

    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "image_cond.raw")
    conditioning.reshape(1, SIZE, SIZE, 3).tofile(raw_path)

    preview_path = os.path.join(args.out, "canny_preview.png")
    cv2.imwrite(preview_path, conditioning)
    cv2.imwrite(os.path.join(args.out, "source_512.png"), resized)

    density = float(np.count_nonzero(edges)) / edges.size
    print(f"{os.path.basename(raw_path)}  {conditioning.shape}  uint8")
    print(f"edge density {density * 100:.1f}%  (thresholds {args.low}/{args.high})")
    if density < 0.01:
        print("  warning: very few edges — the model will be barely constrained")
    elif density > 0.12:
        print("  warning: very dense edges — output tends to come out scribbly")
    print(f"preview {preview_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("-o", "--out", required=True, help="Directory to write the raw into")
    parser.add_argument("--low", type=int, default=100, help="Canny lower threshold")
    parser.add_argument("--high", type=int, default=200, help="Canny upper threshold")
    parser.add_argument(
        "--blur",
        type=int,
        default=3,
        help="Gaussian blur kernel before Canny, 0 to disable. Suppresses fabric weave.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Squash to 512x512 instead of centre-cropping to square first",
    )
    main(parser.parse_args())
