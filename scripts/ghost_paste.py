#!/usr/bin/env python3
"""Put the NPU's swapped face back into the product photo.

    python scripts/ghost_paste.py output_0.raw ghost_input/crop.json -o out.png
    python scripts/ghost_paste.py output_0.raw ghost_input/crop.json \
        --reference ref/ffhq45.raw --identity

The board writes a (1, 3, 256, 256) float32 tensor in [-1, 1]; everything that
turns that into a picture happens here, on the host, exactly as it does for
the fp32 path in ghost_swap.py. Same paste-back function for both, so a
difference between the two images is the quantization and nothing else — the
same discipline scripts/face_matte.py applies to the face parser.

Two comparisons, because one is not enough for a generator:

  --reference  mean / p99 / max absolute difference against an fp32
               output_0.raw, in 0-255 units. How far the pixels moved.
  --identity   runs the fp32 ArcFace over the source frame and over each
               swapped face and reports the cosine between them. Whether it
               is still the right person.

They disagree more often than you would think. A swap can be a long way off
per-pixel and unmistakably the right face, or pixel-close and nobody in
particular. `--identity` also prints an `unswapped_target` control — the
model's own face scored against the shopper — without which the other numbers
have no scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for vendor.ghost
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ghost_swap import paste_back, to_bgr  # noqa: E402

OUTPUT_SHAPE = (1, 3, 256, 256)


def load_output(path: Path) -> np.ndarray:
    array = np.fromfile(path, dtype=np.float32)
    want = int(np.prod(OUTPUT_SHAPE))
    if array.size != want:
        sys.exit(f"{path}: expected {want} floats, got {array.size}")
    return array.reshape(OUTPUT_SHAPE)


def compare(npu: np.ndarray, reference: Path) -> dict:
    """How far the quantized face drifted from the fp32 one."""
    ref = load_output(reference)
    # [-1, 1] spans 0-255, so a difference of d maps to d * 127.5 levels.
    diff = np.abs(npu - ref) * 127.5
    a, b = npu.ravel(), ref.ravel()
    return {
        "mean_abs_diff_255": float(diff.mean()),
        "max_abs_diff_255": float(diff.max()),
        "p99_abs_diff_255": float(np.percentile(diff, 99)),
        "pixel_cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
    }


def identity_similarity(swaps: dict[str, np.ndarray], record: dict) -> dict:
    """Does the swapped face still read as the person who walked up?

    Per-pixel error is the wrong headline for a generator. A swap can be a
    long way off pixel-for-pixel and still be unmistakably the right person,
    and it can be pixel-close and lose the likeness entirely. So this runs the
    fp32 ArcFace over the source frame and over each swapped face, and reports
    the cosine between them — the same recogniser the swap was conditioned on,
    used here as a judge rather than as an input.

    Treat it as relative, not absolute. The useful reading is the NPU's number
    against the fp32 one; a face swap is not identity-preserving in the
    face-recognition sense, so neither will approach 1.0.
    """
    import cv2
    import torch

    from ghost_swap import weight_path
    from vendor.ghost import load_identity_encoder
    import make_ghost_raw as prep

    source = cv2.imread(record["source"])
    if source is None:
        return {"error": f"could not re-read the source frame {record['source']}"}
    source_crop, _, _ = prep.crop_for(source, "source frame")

    # Only the encoder. `ghost_swap.load_models` would also read the 703 MB
    # generator, which this scoring pass has no use for.
    torch.set_grad_enabled(False)
    arc = load_identity_encoder(weight_path("backbone.pth"))

    def embed(crop_bgr):
        tensor = torch.from_numpy(prep.identity_tensor(crop_bgr))
        vector = arc(tensor).numpy().ravel()
        return vector / np.linalg.norm(vector)

    reference = embed(source_crop)
    out = {}
    for name, tensor in swaps.items():
        # The 256 output is the same aligned framing, so it needs no re-detect.
        face = to_bgr(tensor)
        vector = embed(face)
        out[name] = float(reference @ vector)

    # The control, without which the numbers above mean nothing: the model
    # whose photo this is, scored against the shopper by the same recogniser.
    # If a swap does not land far above this, it did not move the identity.
    target = cv2.imread(record["target"])
    if target is not None:
        target_crop, _, _ = prep.crop_for(target, "target photo")
        out["unswapped_target"] = float(reference @ embed(target_crop))
    return out


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser(description="NPU swap output -> finished photo")
    ap.add_argument("output", type=Path, help="output_0.raw from the board")
    ap.add_argument("crop", type=Path, help="crop.json from make_ghost_raw.py")
    ap.add_argument("-o", "--out", type=Path, default=Path("ghost_swapped.png"))
    ap.add_argument("--reference", type=Path,
                    help="fp32 output_0.raw to compare against")
    ap.add_argument("--identity", action="store_true",
                    help="also score both swaps against the source face")
    args = ap.parse_args()

    record = json.loads(args.crop.read_text())
    target = cv2.imread(record["target"])
    if target is None:
        sys.exit(f"could not re-read the target photo {record['target']}")

    y = load_output(args.output)
    result = paste_back(target, to_bgr(y), record["target_matrix"],
                        record["crop_size"])
    cv2.imwrite(str(args.out), result)
    print(f"  out    {args.out}")

    if args.reference:
        stats = compare(y, args.reference)
        print(f"  vs fp32  mean {stats['mean_abs_diff_255']:.2f}/255, "
              f"p99 {stats['p99_abs_diff_255']:.2f}, "
              f"max {stats['max_abs_diff_255']:.2f}, "
              f"cosine {stats['pixel_cosine']:.5f}")
        if args.identity:
            scores = identity_similarity(
                {"npu": y, "fp32": load_output(args.reference)}, record)
            stats.update({f"identity_{k}": v for k, v in scores.items()})
            if "error" not in scores:
                print(f"  identity vs the source face: "
                      f"fp32 {scores['fp32']:.4f}, NPU {scores['npu']:.4f}")
        print(json.dumps(stats, indent=2))
    elif args.identity:
        print(json.dumps(identity_similarity({"npu": y}, record), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
