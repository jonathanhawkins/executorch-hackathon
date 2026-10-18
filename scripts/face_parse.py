#!/usr/bin/env python3
"""fp32 reference for the BiSeNet face parser — the thing the NPU is compared against.

    python scripts/face_parse.py portrait.jpg -o out/

Runs the eager PyTorch model on the Mac and writes `output_0.raw`, the same
(1, 19, 512, 512) float32 logits the QNN runner produces on the board. Both go
through scripts/face_matte.py afterwards, so the only difference between the
two paths is where the convolutions ran — which is the point: it makes the
quantization cost visible instead of assumed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vendor.face_parsing import load_face_parser  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_face_raw import preprocess  # noqa: E402

CHECKPOINT_REPO = "vivym/face-parsing-bisenet"
CHECKPOINT_FILE = "79999_iter.pth"
# All three HF mirrors of this file agree on this digest; it is the checkpoint
# from zllrunning/face-parsing.PyTorch.
CHECKPOINT_SHA256 = "468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567"


def checkpoint_path() -> str:
    import hashlib

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        sys.exit(f"checkpoint digest mismatch: {digest}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="BiSeNet face parsing, fp32 on CPU")
    ap.add_argument("image", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("face_reference"))
    args = ap.parse_args()

    if not args.image.exists():
        print(f"{args.image}: not found", file=sys.stderr)
        return 1

    net = load_face_parser(checkpoint_path())
    tensor = torch.from_numpy(preprocess(Image.open(args.image)))

    with torch.no_grad():
        net(tensor)                                   # warm
        start = time.time()
        logits = net(tensor)[0]
        elapsed = time.time() - start

    args.output.mkdir(parents=True, exist_ok=True)
    logits.numpy().astype(np.float32).tofile(args.output / "output_0.raw")

    print(f"  in     {args.image.name}")
    print(f"  infer  {elapsed:.3f} s (fp32, eager, CPU)")
    print(f"  out    {args.output}/output_0.raw  {tuple(logits.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
