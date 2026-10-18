#!/usr/bin/env python3
"""Export GHOST to an XNNPACK .pte — the CPU fallback, and the honest baseline.

    python scripts/ghost_xnnpack_export.py -o models/ghost
    python scripts/ghost_xnnpack_export.py --stage identity -o models/ghost

Runs on the **Mac**, unlike board_faceswap_export.py: XNNPACK's lowering is
host-portable, and only QNN's ahead-of-time compiler is board-only.

## Why this exists

Two reasons, and the second is the one that matters.

1. **It is a fallback that ships without QNN.** Numerically it is the fp32
   model: 0.0006/255 against the eager reference, identity cosine 0.8768
   against 0.8768. Slow, but exact and dependency-free.

2. **It is the only fair CPU baseline.** The board's torch is PyPI's
   `linux-aarch64` wheel, which is the sbsa/CUDA build (`2.14.1+cu130`), and
   it reports `CPU capability usage: DEFAULT` — scalar fallback kernels, no
   vectorised ARM dispatch. Timing eager PyTorch there measures that build,
   not the hardware. Quoting it as "the CPU number" overstates the NPU's win
   by about 2x:

       eager fp32, 8 threads      4379 ms
       XNNPACK fp32, 8 threads    2064 ms
       NPU, fp16                    82 ms

   CLAUDE.md already warned the board's torch is the CUDA build; this is what
   that costs when you benchmark against it.

The generator partitions into ~33 XNNPACK subgraphs rather than one, because
`InstanceNorm2d` is not an XNNPACK op — those 21 layers fall back to portable
kernels and split the graph around them. That fragmentation is most of why
this is 2 s and not several hundred milliseconds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGES = ("generator", "identity")


def build(stage: str):
    import torch

    from vendor.ghost import (
        GeneratorForExport,
        load_generator,
        load_identity_encoder,
    )
    from ghost_swap import weight_path

    if stage == "generator":
        model = GeneratorForExport(load_generator(weight_path("G_unet_2blocks.pth")))
        args = (torch.randn(1, 3, 256, 256), torch.randn(1, 512))
    else:
        model = load_identity_encoder(weight_path("backbone.pth"))
        args = (torch.randn(1, 3, 112, 112),)
    return model.eval(), args


def main() -> int:
    import torch

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    ap = argparse.ArgumentParser(description="GHOST -> XNNPACK .pte (CPU)")
    ap.add_argument("--stage", default="generator", choices=STAGES)
    ap.add_argument("-o", "--output", type=Path, default=Path("models/ghost"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model, sample = build(args.stage)

    start = time.time()
    exported = torch.export.export(model, sample)
    program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    args.output.mkdir(parents=True, exist_ok=True)
    name = "ghost_aei_xnnpack" if args.stage == "generator" else "ghost_arcface_xnnpack"
    path = args.output / f"{name}.pte"
    path.write_bytes(program.buffer)

    delegates = program.buffer.count(b"XnnpackBackend")
    print(f"  out        {path}  ({path.stat().st_size / 1e6:.0f} MB)")
    print(f"  export     {time.time() - start:.0f} s")
    print(f"  delegates  XnnpackBackend x{delegates}")
    if args.stage == "generator" and delegates > 1:
        print("             (InstanceNorm2d is not an XNNPACK op, so the graph "
              "splits around those 21 layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
