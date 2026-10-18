#!/usr/bin/env python3
"""Export GHOST face swap to QNN .pte files. Runs ON THE BOARD.

Driven by board_faceswap_export.sh. Same shape as board_faceparse_export.py:
the lowering is the bundle's own `build_executorch_binary` — export, quantize,
calibrate, lower and write in one call.

Two graphs, because a swap is two models:

    --stage identity    ArcFace iresnet100,  (1,3,112,112) -> (1,512)
    --stage generator   GHOST AEI-Net,       (1,3,256,256) + (1,512) -> (1,3,256,256)

**One stage per process.** Two `build_executorch_binary` calls in one
interpreter is the same shape of mistake as calling `from_context_binary()`
twice (see CLAUDE.md) — the QNN manager is process-global. The shell script
invokes this script twice.

## The generator's second input cannot be faked

`build_executorch_binary` calibrates by running the model over a dataset, and
the AEI-Net takes a 512-d identity vector alongside the image. Feeding that
input noise would set its quantization range to the range of noise, and the
AAD layers scale every feature map by it — so the vectors here are real ones,
produced by running the fp32 ArcFace over the same calibration faces. That is
also why `--stage generator` loads both checkpoints.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    make_quantizer,
    QnnConfig,
    setup_common_args_and_variables,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vendor.ghost import (  # noqa: E402
    GeneratorForExport,
    load_generator,
    load_identity_encoder,
)

QUANT_DTYPES = {
    "8a8w": QuantDtype.use_8a8w,
    "16a8w": QuantDtype.use_16a8w,
    "16a16w": QuantDtype.use_16a16w,
    # fp16 activations, 8-bit per-channel weights. The one that matters for
    # the generator: its activations span ±1031 with a median of 5.6, and a
    # float format does not need a calibrated range to carry that. Its weights
    # quantize to 8 bits for about 1 level of error. See docs/faceswap-npu.md.
    "fp16a8w": QuantDtype.use_fp16a8w,
}

def _observers():
    """Activation observers, by the name the CLI uses.

    Imported lazily and defensively: these live in torch.ao and move between
    releases, and a missing one should cost that choice, not the whole run.
    """
    from torch.ao.quantization.observer import (
        HistogramObserver,
        MinMaxObserver,
        MovingAverageMinMaxObserver,
    )

    return {
        "moving_minmax": MovingAverageMinMaxObserver,  # the bundle's default
        "minmax": MinMaxObserver,
        "histogram": HistogramObserver,
    }


ACT_OBSERVERS = ("histogram", "minmax", "moving_minmax")

IDENTITY_SHAPE = (1, 3, 112, 112)
TARGET_SHAPE = (1, 3, 256, 256)
EMBEDDING_SHAPE = (1, 512)


def _raws(directory: Path, prefix: str, limit: int, shape: tuple[int, ...]):
    paths = sorted(directory.glob(f"{prefix}_*.raw"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        sys.exit(f"no {prefix}_*.raw in {directory} — run make_ghost_raw.py --calib")

    want = int(np.prod(shape))
    tensors = []
    for path in paths[:limit]:
        array = np.fromfile(path, dtype=np.float32)
        if array.size != want:
            sys.exit(f"{path}: expected {want} floats, got {array.size}")
        tensors.append(torch.from_numpy(array.reshape(shape)))
    return tensors


def main() -> int:
    parser = setup_common_args_and_variables()
    parser.add_argument("-a", "--artifact", default="./faceswap", type=str)
    parser.add_argument("--stage", required=True, choices=("identity", "generator"))
    parser.add_argument("--generator_ckpt", type=Path)
    parser.add_argument("--identity_ckpt", type=Path, required=True)
    parser.add_argument("--calib_dir", required=True, type=Path)
    parser.add_argument("--calib_samples", default=40, type=int)
    # "none" skips quantization entirely and hands QNN the float graph. The
    # HTP's own validation messages list FP16 and BF16 among its supported
    # I/O datatype sets, so this is not obviously hopeless — and for a
    # generator whose whole problem is fixed-point normalisation statistics,
    # it is the only configuration that removes the cause rather than
    # widening it. Expect a bigger .pte and slower inference than 16a8w.
    parser.add_argument("--quant", default="16a8w",
                        choices=sorted(QUANT_DTYPES) + ["none"])
    # The two knobs that decide whether this model survives quantization at
    # all. Both go through `make_quantizer`, whose defaults are wrong here:
    #
    #   per_channel_conv=True   breaks `fp16a8w` outright — with float
    #     activations the weights need an explicit Dequantize node, and this
    #     QAIRT's only takes per-tensor ("Op Dequantize does not support
    #     per-channel quant tensor"). Per-tensor 8-bit weights were measured
    #     to cost 1.89/255 against 1.08 for per-channel, so this is cheap.
    #
    #   act_observer=MovingAverageMinMaxObserver tracks the *extremes*, and
    #     this generator's activations run to ±1031 against a median of 5.6,
    #     with max/std up to 344. A min-max range spends almost all its
    #     levels on outliers. `histogram` picks the range that minimises
    #     quantization error instead, which is the standard answer for a
    #     distribution like this.
    parser.add_argument("--per_channel", default="true",
                        choices=("true", "false"),
                        help="per-channel conv weights (must be false for fp16a8w)")
    parser.add_argument("--act_observer", default="moving_minmax",
                        choices=sorted(ACT_OBSERVERS),
                        help="activation range observer")
    # The surgical option. All 21 InstanceNorm2d layers — the thing most
    # likely responsible for the generator's speckle — live inside
    # `AAD_ResBlk`, so those blocks can be given a wider config while the rest
    # of the graph stays cheap. Off by default; it costs latency wherever it
    # is applied.
    parser.add_argument("--aad_quant", default=None, choices=sorted(QUANT_DTYPES),
                        help="separate quant config for the AAD_ResBlk submodules")
    args = parser.parse_args()

    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)
    artifact = Path(args.artifact)
    artifact.mkdir(parents=True, exist_ok=True)
    quant_dtype = None if args.quant == "none" else QUANT_DTYPES[args.quant]

    torch.set_grad_enabled(False)

    if args.stage == "identity":
        model = load_identity_encoder(args.identity_ckpt)
        dataset = [(t,) for t in _raws(args.calib_dir, "identity",
                                       args.calib_samples, IDENTITY_SHAPE)]
        name = f"ghost_arcface_qnn_{args.quant}"
    else:
        if not args.generator_ckpt:
            sys.exit("--generator_ckpt is required for --stage generator")
        model = GeneratorForExport(load_generator(args.generator_ckpt)).eval()

        targets = _raws(args.calib_dir, "target", args.calib_samples, TARGET_SHAPE)
        identities = _raws(args.calib_dir, "identity", args.calib_samples,
                           IDENTITY_SHAPE)
        if len(identities) < len(targets):
            sys.exit("fewer identity_*.raw than target_*.raw in the calibration set")

        # Real embeddings, fp32, from the same faces. See the module docstring.
        arcface = load_identity_encoder(args.identity_ckpt)
        logging.info("embedding %d calibration faces for the generator", len(targets))
        dataset = [(target, arcface(identity))
                   for target, identity in zip(targets, identities)]
        del arcface
        name = f"ghost_aei_qnn_{args.quant}"

    per_channel = args.per_channel == "true"
    if quant_dtype is None:
        # No quantizer, no calibration ranges, no dataset pass. Everything
        # below about observers and per-channel weights is moot.
        float_name = name.replace("_none", "_float")
        logging.info("no quantization: handing QNN the float graph")
        build_executorch_binary(
            model=model,
            qnn_config=qnn_config,
            file_name=str(artifact / float_name),
            dataset=dataset,
            quant_dtype=None,
        )
        print(f"\nWrote {artifact / float_name}.pte")
        return 0

    if quant_dtype is QuantDtype.use_fp16a8w and per_channel:
        sys.exit("fp16a8w needs --per_channel false; see the note in this file")

    # Built here rather than left to `make_quantizer`'s defaults, because
    # those defaults are what broke this model. Passing `custom_quantizer`
    # makes `build_executorch_binary` use it verbatim.
    submodule_qconfig_list = []
    if args.aad_quant:
        if args.stage != "generator":
            sys.exit("--aad_quant only applies to --stage generator")
        from executorch.backends.qualcomm.quantizer.quantizer import (
            get_submodule_type_predicate,
            ModuleQConfig,
        )

        submodule_qconfig_list = [(
            get_submodule_type_predicate("AAD_ResBlk"),
            ModuleQConfig(QUANT_DTYPES[args.aad_quant]),
        )]
        logging.info("AAD_ResBlk submodules get %s", args.aad_quant)

    quantizer = make_quantizer(
        quant_dtype=quant_dtype,
        per_channel_conv=per_channel,
        per_channel_linear=False,
        act_observer=_observers()[args.act_observer],
        submodule_qconfig_list=submodule_qconfig_list,
        backend=qnn_config.backend,
        soc_model=qnn_config.soc_model,
    )

    suffix = "" if (per_channel and args.act_observer == "moving_minmax") else (
        f"_{'pc' if per_channel else 'pt'}_{args.act_observer}")
    if args.aad_quant:
        suffix += f"_aad{args.aad_quant}"
    name += suffix

    logging.info("calibration: %d samples", len(dataset))
    logging.info("quant: %s, per-channel conv: %s, act observer: %s",
                 args.quant, per_channel, args.act_observer)
    build_executorch_binary(
        model=model,
        qnn_config=qnn_config,
        file_name=str(artifact / name),
        dataset=dataset,
        quant_dtype=quant_dtype,
        custom_quantizer=quantizer,
    )
    print(f"\nWrote {artifact / name}.pte")
    return 0


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    sys.exit(main())
