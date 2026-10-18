#!/usr/bin/env python3
"""Export BiSeNet face parsing to a QNN .pte. Runs ON THE BOARD.

Driven by board_faceparse_export.sh. The lowering itself is the bundle's own
`build_executorch_binary`, the same entry point its deeplab_v3.py example uses
— export, quantize, calibrate, lower and write, in one call. What this adds is
the model (vendored, since BiSeNet is not one of the bundle's examples) and a
calibration set read from raw tensors prepared on the Mac.

Calibration comes in as .raw files rather than images on purpose: the board has
no image stack worth trusting for this, and preprocessing the calibration set
through a different code path from the inference set is a good way to set every
activation range slightly wrong.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vendor.face_parsing import FaceParserForExport, load_face_parser  # noqa: E402

INPUT_SHAPE = (1, 3, 512, 512)
PTE_NAME = "faceparse_qnn_8a8w"


def calibration_dataset(directory: Path, limit: int):
    raws = sorted(directory.glob("input_*.raw"))
    if not raws:
        sys.exit(f"no input_*.raw in {directory} — run make_face_raw.py --calib first")

    dataset = []
    for path in raws[:limit]:
        array = np.fromfile(path, dtype=np.float32)
        if array.size != int(np.prod(INPUT_SHAPE)):
            sys.exit(f"{path}: expected {int(np.prod(INPUT_SHAPE))} floats, got {array.size}")
        dataset.append((torch.from_numpy(array.reshape(INPUT_SHAPE)),))
    logging.info("calibration: %d tensors from %s", len(dataset), directory)
    return dataset


def main() -> int:
    parser = setup_common_args_and_variables()
    parser.add_argument("-a", "--artifact", default="./faceparse", type=str)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calib_dir", required=True, type=Path)
    parser.add_argument("--calib_samples", default=40, type=int)
    args = parser.parse_args()

    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)
    Path(args.artifact).mkdir(parents=True, exist_ok=True)

    model = FaceParserForExport(load_face_parser(args.checkpoint))
    dataset = calibration_dataset(args.calib_dir, args.calib_samples)

    build_executorch_binary(
        model=model,
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{PTE_NAME}",
        dataset=dataset,
        quant_dtype=QuantDtype.use_8a8w,
    )
    print(f"\nWrote {args.artifact}/{PTE_NAME}.pte")
    return 0


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    sys.exit(main())
