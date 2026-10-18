#!/usr/bin/env python3
"""Say whether a .pte runs on CPU or the NPU.

Worth having because the two are indistinguishable from the outside — same
extension, same loading code — and running an NPU-targeted model on a CPU
runtime (or vice versa) fails in confusing ways. A model's filename is not
evidence; the delegate records inside it are.

    python scripts/pte_backend.py models/llama3.2-1b-open/model.pte
    python scripts/pte_backend.py ~/Downloads/*.pte

Works on a partial download too — delegate records live near the start of the
file, so the first few MB of a huge .pte is enough to classify it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Backend ids as ExecuTorch writes them into the program flatbuffer.
BACKENDS = {
    "QnnBackend": ("NPU", "Qualcomm Hexagon via QNN/HTP"),
    "XnnpackBackend": ("CPU", "XNNPACK"),
    "CoreMLBackend": ("ANE/GPU", "Apple Core ML"),
    "MPSBackend": ("GPU", "Apple Metal"),
    "VulkanBackend": ("GPU", "Vulkan"),
    "CudaBackend": ("GPU", "CUDA"),
}

# Enough to cover the program section of a large model without reading it all.
SCAN_BYTES = 16 * 1024 * 1024


def classify(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            head = fh.read(SCAN_BYTES)
    except OSError as exc:
        print(f"{path}: cannot read ({exc})", file=sys.stderr)
        return 1

    counts = {
        name: len(re.findall(name.encode(), head))
        for name in BACKENDS
    }
    found = {n: c for n, c in counts.items() if c}

    size_mb = path.stat().st_size / 1024 / 1024
    partial = " (scanned first 16MB)" if path.stat().st_size > SCAN_BYTES else ""
    print(f"\n{path.name}  —  {size_mb:.1f} MB{partial}")

    if not found:
        print("  no delegate records found — likely a portable/CPU-only program,")
        print("  or the file is truncated before the program section")
        return 0

    for name, count in sorted(found.items(), key=lambda kv: -kv[1]):
        where, desc = BACKENDS[name]
        print(f"  {where:<8} {desc}  ({name} ×{count})")

    # One fused blob vs many small partitions is a real signal: QNN tends to
    # lower the whole graph at once, XNNPACK to carve out many subgraphs.
    if "QnnBackend" in found:
        print("  -> NPU build. Needs QNN runtime libraries and a matching SoC target.")
    elif "XnnpackBackend" in found:
        print("  -> CPU build. Runs anywhere; an NPU will NOT accelerate it.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Identify a .pte's execution backend")
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()

    rc = 0
    for path in args.paths:
        if not path.exists():
            print(f"{path}: not found", file=sys.stderr)
            rc = 1
            continue
        rc |= classify(path)
    print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
