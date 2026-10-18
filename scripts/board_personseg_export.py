#!/usr/bin/env python3
"""Export a person-segmentation model to a QNN .pte. Runs ON THE BOARD.

Driven by board_personseg_export.sh. Same shape as board_faceparse_export.py:
the lowering is the bundle's own `build_executorch_binary` — export, quantize,
calibrate, lower and write in one call — and what this adds is the model and a
calibration set read from raw tensors prepared on the Mac.

`--model` selects between the candidates in person_seg.py, so the comparison in
docs/personseg-npu.md is one flag rather than three scripts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    QnnConfig,
    setup_common_args_and_variables,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

sys.path.insert(0, str(Path(__file__).resolve().parent))
from person_seg import DEFAULT_MODEL, MODELS, use_local_checkpoint  # noqa: E402

QUANT_DTYPES = {"8a8w": QuantDtype.use_8a8w, "16a8w": QuantDtype.use_16a8w}


def calibration_dataset(directory: Path, limit: int, shape: tuple[int, ...]):
    raws = sorted(directory.glob("input_*.raw"),
                  key=lambda p: int(p.stem.split("_")[1]))
    if not raws:
        sys.exit(f"no input_*.raw in {directory} — run make_person_raw.py --calib first")

    want = int(np.prod(shape))
    dataset = []
    for path in raws[:limit]:
        array = np.fromfile(path, dtype=np.float32)
        if array.size != want:
            sys.exit(f"{path}: expected {want} floats, got {array.size}")
        dataset.append((torch.from_numpy(array.reshape(shape)),))
    logging.info("calibration: %d tensors from %s", len(dataset), directory)
    return dataset


def main() -> int:
    parser = setup_common_args_and_variables()
    parser.add_argument("-a", "--artifact", default="./personseg", type=str)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    parser.add_argument("--checkpoint", type=Path,
                        help="a pushed HF snapshot directory, instead of the Hub")
    parser.add_argument("--calib_dir", required=True, type=Path)
    parser.add_argument("--calib_samples", default=48, type=int)
    parser.add_argument("--quant", default="8a8w", choices=sorted(QUANT_DTYPES))
    args = parser.parse_args()

    model = MODELS[args.model]
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)
    Path(args.artifact).mkdir(parents=True, exist_ok=True)

    use_local_checkpoint(args.checkpoint)
    net = model.load()
    dataset = calibration_dataset(args.calib_dir, args.calib_samples, model.input_shape)

    name = f"personseg_{args.model}_{args.quant}"
    build_executorch_binary(
        model=net,
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{name}",
        dataset=dataset,
        quant_dtype=QUANT_DTYPES[args.quant],
    )
    print(f"\nWrote {args.artifact}/{name}.pte")
    print(f"  input  {model.input_shape}")
    print(f"  output (1, {model.logit_shape[0]}, "
          f"{model.logit_shape[1]}, {model.logit_shape[2]})")
    return 0


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    sys.exit(main())
