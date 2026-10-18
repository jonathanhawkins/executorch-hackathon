#!/usr/bin/env python3
"""Prove that board_sd_driver.py's fast residual remaps change no bits.

ControlNet emits 13 residual tensors under its own output encodings and the
UNet wants them under 13 different input encodings — no two of the pairs share
a scale and an offset, so all 6.7 M elements are remapped twice per denoising
step. `Requantizer` has four ways to do that, and the only acceptable
difference between them in the finished image is none.

This checks that claim the way it is actually made: uint16 in, uint16 out, so
the remap is a function of 65536 possible values and all 65536 can be tried.
Nothing here is sampling or tolerance.

    adb pull /home/arduino/models/controlnet-npu/encodings.json /tmp/enc.json
    .venv/bin/python scripts/check_requant.py /tmp/enc.json

Runs anywhere — it needs numpy and an encodings.json, not a board or a .pte,
which is why the example pulls the one the board exported rather than running
there. `lut` and `inplace` must come out bit-exact; `affine` is expected not
to, and this prints how far off it is so the number in the docs stays honest.
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

EXACT = ("lut", "inplace", "pairs")
APPROXIMATE = ("affine",)


def load_driver():
    """board_sd_driver.py as a module, without needing it importable."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_sd_driver.py")
    spec = importlib.util.spec_from_file_location("board_sd_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def residual_pairs(encodings):
    produced = sorted(encodings["controlnet"]["outputs"], key=lambda e: e["index"])
    consumed = [
        entry
        for entry in sorted(encodings["unet"]["inputs"], key=lambda e: e["index"])
        if entry["name"].startswith("controlnet_")
    ]
    if len(produced) != len(consumed):
        raise SystemExit(
            f"controlnet emits {len(produced)} residuals but the UNet takes "
            f"{len(consumed)} — these encodings are not a matched set"
        )
    return list(zip(produced, consumed))


def main(args):
    driver = load_driver()
    with open(args.encodings) as f:
        encodings = json.load(f)
    if "controlnet" not in encodings:
        raise SystemExit(f"{args.encodings} has no controlnet stage — nothing to check")
    pairs = residual_pairs(encodings)

    every_uint16 = np.arange(1 + driver.UINT16_MAX, dtype=np.uint16)
    rng = np.random.default_rng(args.seed)
    work = tempfile.mkdtemp(prefix="check_requant_")
    failures, approximate = 0, {}

    for conditioning_scale in args.conditioning_scale:
        modes = {
            mode: driver.Requantizer(pairs, conditioning_scale, mode)
            for mode in EXACT + APPROXIMATE
        }
        for slot, (produced, consumed) in enumerate(pairs):
            size = int(np.prod(produced["shape"]))
            # Every value a uint16 can hold, tiled out to the real tensor
            # length so the shapes, the file sizes and the mappings are the
            # ones the driver really uses.
            for run, data in enumerate(
                (
                    np.resize(every_uint16, size),
                    rng.integers(0, 1 + driver.UINT16_MAX, size=size, dtype=np.uint16),
                )
            ):
                reference = driver.requantize(
                    data, produced, consumed, conditioning_scale
                )
                for mode, requantizer in modes.items():
                    path = os.path.join(work, f"{mode}_{slot}.raw")
                    written = requantizer.into(0, slot, data, path)
                    if not isinstance(written, driver.Written) or written.size != size:
                        raise SystemExit(f"{mode}: into() returned {written!r}")
                    got = np.fromfile(path, dtype=np.uint16)
                    off = np.abs(got.astype(np.int32) - reference.astype(np.int32))
                    differing = int(np.count_nonzero(off))
                    if mode in EXACT:
                        if differing:
                            print(
                                f"  FAIL  {mode} residual {slot} "
                                f"({produced['name']}) at scale {conditioning_scale}, "
                                f"run {run}: {differing} of {size} elements differ, "
                                f"by up to {int(off.max())}"
                            )
                            failures += 1
                    else:
                        tally = approximate.setdefault(mode, [0, 0, 0])
                        tally[0] += differing
                        tally[1] += size
                        tally[2] = max(tally[2], int(off.max()) if differing else 0)

    print(
        f"  {len(pairs)} residuals x {len(args.conditioning_scale)} conditioning "
        f"scales x 2 runs, every uint16 value covered"
    )
    for mode in EXACT:
        if mode != "pairs":
            print(f"  {mode:8s} bit-exact against the reference")
    for mode, (differing, total, worst) in approximate.items():
        print(
            f"  {mode:8s} {differing} of {total} elements differ "
            f"({100 * differing / total:.3f} %), none by more than {worst} LSB"
        )
    if failures:
        print(f"\n  {failures} exact mode(s) are not exact — do not ship this")
        return 1
    print("\n  ok")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "encodings",
        help="A ControlNet model directory's encodings.json, as written by "
        "scripts/board_sd_export.py",
    )
    parser.add_argument(
        "--conditioning_scale",
        type=float,
        nargs="+",
        default=[1.0, 0.8, 1.25],
        help="ControlNet strengths to check. It folds into the table and the "
        "constants, so it is part of what has to come out bit-exact.",
    )
    parser.add_argument("--seed", type=int, default=11)
    sys.exit(main(parser.parse_args()))
