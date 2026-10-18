#!/usr/bin/env python3
"""Turn a camera frame and a product photo into the raw tensors GHOST expects.

    python scripts/make_ghost_raw.py --source me.jpg --target tee.png -o out/
    python scripts/make_ghost_raw.py --calib 40 -o calib/

Two models, so two tensors with different jobs:

    identity_0.raw   (1, 3, 112, 112)   the SOURCE face -> ArcFace -> 512-d
    target_0.raw     (1, 3, 256, 256)   the TARGET face -> AEI-Net, with that 512-d

Both are RGB in [-1, 1], NCHW, little-endian float32 — which is the one format
the QNN executor runner reads.

## The alignment has to match how GHOST was trained

Upstream crops with `insightface.utils.face_align.estimate_norm(kps, 224,
mode='None')`: a 5-point similarity transform onto ArcFace's canonical
template scaled by 224/112 = 2. The generator then sees that 224 crop resized
to 256, and ArcFace sees it at 112 (upstream gets there with
`F.interpolate(scale_factor=0.5)`, which is a bilinear halving of the 224).

Getting this wrong does not throw — it quietly produces a face that belongs to
nobody. So the template here is the same five points in the same order, and
the 224 crop is the single source of both tensors.

No insightface dependency: YuNet already returns exactly these five landmarks
(scripts/face_detect.py), in exactly this order. Mind the crossed left/right
that CLAUDE.md warns about — "right_eye" is the subject's right, which is the
*left* side of the image.

## Consent, in the shape of the interface

The entry point is a frame and a target, not a photo picker. The source face
is whoever is standing in front of the board right now; there is deliberately
no path here for uploading someone else's face. See docs/faceswap-npu.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for vendor.ghost
sys.path.insert(0, str(Path(__file__).resolve().parent))
import face_detect  # noqa: E402
from make_face_raw import fetch_calibration_images  # noqa: E402

# What the two graphs are compiled for.
IDENTITY_SIZE = 112
TARGET_SIZE = 256
# Upstream's crop size, and its own comment on the flag is "Don't change this".
CROP_SIZE = 224

# ArcFace's canonical 112x112 landmarks, in YuNet's order. Same table as
# scripts/face_swap.py — the inswapper path needs the identical transform, and
# two copies that must agree is worse than one that is stated twice, so if you
# change one, change both.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # right_eye  (image left)
        [73.5318, 51.5014],  # left_eye   (image right)
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # right_mouth
        [70.7299, 92.2041],  # left_mouth
    ],
    dtype=np.float32,
)

LANDMARK_ORDER = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")


def landmarks_array(face: dict) -> np.ndarray:
    return np.array([face["landmarks"][k] for k in LANDMARK_ORDER], dtype=np.float32)


def align(image, landmarks: np.ndarray, size: int = CROP_SIZE):
    """Similarity transform onto the ArcFace template, scaled to `size`."""
    import cv2

    dst = ARCFACE_TEMPLATE * (size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks, dst, method=cv2.LMEDS, refineIters=20
    )
    if matrix is None:
        sys.exit("could not solve the alignment transform")
    return cv2.warpAffine(image, matrix, (size, size), borderValue=0.0), matrix


def _to_tensor(crop_bgr, size: int) -> np.ndarray:
    """BGR uint8 crop -> (1, 3, size, size) RGB float32 in [-1, 1]."""
    import cv2

    if crop_bgr.shape[0] != size:
        crop_bgr = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
    arr = (arr - 0.5) / 0.5  # -> [-1, 1]
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


def identity_tensor(crop_bgr) -> np.ndarray:
    """The ArcFace input: the 224 crop at 112.

    Upstream gets to 112 with `F.interpolate(scale_factor=0.5,
    mode='bilinear', align_corners=True)` on the normalised tensor; this uses
    `cv2.INTER_LINEAR` on the uint8 crop, which is the same idea with
    different corner handling. Measured on a real face, the two tensors differ
    by up to 0.30 on a [-1, 1] scale — and the embeddings they produce agree
    to **cosine 0.9989**, which is far inside what the generator notices. Not
    worth a torch dependency in the preprocessing path to close.
    """
    return _to_tensor(crop_bgr, IDENTITY_SIZE)


def target_tensor(crop_bgr) -> np.ndarray:
    """The AEI-Net input: the 224 crop at 256."""
    return _to_tensor(crop_bgr, TARGET_SIZE)


def crop_for(image, where: str, scale: int = 1):
    """Detect, then align. Returns the 224 crop and its transform.

    No `--bbox` escape hatch, unlike make_face_raw.py: a box is enough to crop
    a square for BiSeNet, but the alignment here is solved from the five
    landmarks, and there is no way to invent those from a rectangle.
    """
    face = face_detect.detect_frame(image, scale=scale)
    if face is None:
        sys.exit(f"no face found in the {where}")
    crop, matrix = align(image, landmarks_array(face), CROP_SIZE)
    return crop, matrix, face


def write_run(source_path: Path, target_path: Path, out: Path, scale: int = 1,
              embed: bool = False) -> dict:
    """One swap's worth of inputs, plus what the paste-back will need."""
    import cv2

    source = cv2.imread(str(source_path))
    target = cv2.imread(str(target_path))
    if source is None:
        sys.exit(f"could not read {source_path}")
    if target is None:
        sys.exit(f"could not read {target_path}")

    source_crop, _, source_face = crop_for(source, "source frame", scale=scale)
    target_crop, target_matrix, target_face = crop_for(target, "target photo")

    out.mkdir(parents=True, exist_ok=True)
    identity = identity_tensor(source_crop)
    identity.tofile(out / "identity_0.raw")
    target_tensor(target_crop).tofile(out / "target_0.raw")
    (out / "identity_list.txt").write_text("identity_0.raw\n")

    if embed:
        # The identity encoder in fp32, here on the host, because quantizing
        # it destroys the embedding (cosine 0.118 against fp32 — a different
        # person) and it costs nothing to avoid: it describes whoever is
        # standing there, not the instant the shutter fires, so it is off the
        # critical path entirely. docs/faceswap-npu.md has the measurements.
        import torch

        from vendor.ghost import load_identity_encoder
        from ghost_swap import weight_path

        torch.set_grad_enabled(False)
        encoder = load_identity_encoder(weight_path("backbone.pth"))
        vector = encoder(torch.from_numpy(identity)).numpy().astype(np.float32)
        vector.tofile(out / "embedding_0.raw")

    # The AEI-Net takes two inputs, so its list line is two filenames. The
    # embedding lands here only after the ArcFace stage has run on the board.
    (out / "target_list.txt").write_text("target_0.raw embedding_0.raw\n")

    record = {
        "target_matrix": target_matrix.tolist(),
        "target_size": list(target.shape[:2]),
        "crop_size": CROP_SIZE,
        "target_landmarks": target_face["landmarks"],
        "source_landmarks": source_face["landmarks"],
        "source": str(source_path),
        "target": str(target_path),
    }
    (out / "crop.json").write_text(json.dumps(record, indent=2))
    return record


def write_calibration(count: int, out: Path) -> int:
    """Calibration tensors for both graphs, from the same faces.

    CelebA-HQ through the datasets-server rows API, the same source and the
    same code path as the face parser's calibration — see make_face_raw.py.
    These images are already face-filling and roughly centred, so they are
    aligned here with their own detected landmarks, exactly as an inference
    frame would be. Calibration that skipped the alignment would set the
    activation ranges for a framing the model never sees.

    The AEI-Net's *second* input is a 512-d identity vector, which cannot be
    manufactured from noise — its range is set by what ArcFace actually emits.
    So those are left to the export script, which has the fp32 ArcFace loaded
    and writes one embedding per calibration face.
    """
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    identity_names, target_names, kept = [], [], 0

    for idx, img in enumerate(fetch_calibration_images(count)):
        frame = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        face = face_detect.detect_frame(frame)
        if face is None:
            continue  # a face the detector misses is not a face GHOST will see
        crop, _ = align(frame, landmarks_array(face), CROP_SIZE)

        identity_tensor(crop).tofile(out / f"identity_{kept}.raw")
        target_tensor(crop).tofile(out / f"target_{kept}.raw")
        identity_names.append(f"identity_{kept}.raw")
        target_names.append(f"target_{kept}.raw")
        kept += 1

    if kept == 0:
        sys.exit("no usable calibration faces")
    (out / "identity_list.txt").write_text("\n".join(identity_names) + "\n")
    (out / "target_list.txt").write_text("\n".join(target_names) + "\n")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description="Frame + product photo -> GHOST tensors")
    ap.add_argument("--source", type=Path, help="the live frame: whoever is standing there")
    ap.add_argument("--target", type=Path, help="the product photo their face goes into")
    ap.add_argument("-o", "--output", type=Path, default=Path("ghost_input"))
    ap.add_argument("--calib", type=int, metavar="N",
                    help="instead of a swap, fetch N CelebA-HQ faces")
    ap.add_argument("--scale", type=int, default=1,
                    help="detect on a downscaled frame (see CLAUDE.md)")
    ap.add_argument("--embed", action="store_true",
                    help="also write embedding_0.raw, the fp32 identity vector")
    args = ap.parse_args()

    if args.calib:
        kept = write_calibration(args.calib, args.output)
        print(f"  calib  {kept} faces, aligned, both graphs")
    else:
        if not args.source or not args.target:
            return ap.error("give --source and --target, or --calib N")
        write_run(args.source, args.target, args.output, scale=args.scale,
                  embed=args.embed)
        print(f"  source {args.source.name}")
        print(f"  target {args.target.name}")

    print(f"  shape  identity (1, 3, {IDENTITY_SIZE}, {IDENTITY_SIZE}), "
          f"target (1, 3, {TARGET_SIZE}, {TARGET_SIZE}) float32")
    print(f"  out    {args.output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
