#!/usr/bin/env python3
"""Wrap Qualcomm AI Hub Stable Diffusion context binaries into ExecuTorch .pte.

Runs ON THE BOARD — the ahead-of-time QNN path only exists as the bundle's
aarch64 PyQnnManagerAdaptor. Driven by scripts/board_sd_export.sh.

The AI Hub release ships three QNN context binaries (text_encoder, unet, vae)
already compiled for the QCS8275's Hexagon v75. Nothing is quantized or lowered
here: each binary is handed to from_context_binary(), which wraps it in a
one-op ExecuTorch graph, and written out as a .pte the prebuilt
qnn_executor_runner can load.

The quantization encodings are dumped alongside as encodings.json, keyed by
tensor NAME rather than index. The index matters: this release's UNet takes
(timestep, latent, text_emb), while the bundle's own qaihub_stable_diffusion.py
assumes (latent, time_embedding, text_emb). See docs/stable-diffusion-npu.md.
"""

import argparse
import gc
import json
import os
import time

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManagerAdaptor
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    from_context_binary,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    generate_qnn_executorch_option,
    get_soc_to_chipset_map,
)
from executorch.examples.qualcomm.qaihub_scripts.utils.utils import preprocess_binary
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

# Stable Diffusion v2.1 ships three graphs. ControlNet-Canny ships four — the
# extra one produces the 13 residuals its (16-input) UNet takes. Nothing below
# is specific to either: the stage list is an argument and every encoding is
# keyed by tensor name.
DEFAULT_STAGES = ("text_encoder", "unet", "vae")


def compiler_specs(soc_model: str):
    # use_multi_contexts is upstream's choice because it packs several graphs
    # into one .pte and wants a shared spill-fill buffer. Here each stage is a
    # separate .pte loaded by its own runner process, so there is nothing to
    # share — and asking for a persistent shared weights buffer is what makes
    # the 834 MB UNet fail to map onto the cDSP.
    backend_options = generate_htp_compiler_spec(use_fp16=False)
    return generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, soc_model),
        backend_options=backend_options,
        is_from_context_binary=True,
    )


def dump_encodings(binary: str, specs) -> dict:
    """Read every graph input/output's shape, dtype, scale and offset."""
    with open(binary, "rb") as f:
        ctx = preprocess_binary(f.read(), specs)
    mgr = PyQnnManagerAdaptor.QnnManager(generate_qnn_executorch_option(specs), ctx)
    assert mgr.Init().value == 0, f"failed to load context binary: {binary}"

    graph_name = mgr.GetGraphNames()[0]
    mgr.AllocateTensor(graph_name)
    info = {"graph": graph_name, "inputs": [], "outputs": []}
    for key, getter in (("inputs", mgr.GetGraphInputs), ("outputs", mgr.GetGraphOutputs)):
        for index, tensor in enumerate(getter(graph_name)):
            entry = {
                "index": index,
                "name": tensor.GetName(),
                "shape": [int(d) for d in tensor.GetDims()],
                "dtype": str(tensor.GetDataType()).rsplit(".", 1)[-1],
            }
            try:
                encoding = tensor.GetEncodings()
                entry["scale"] = encoding.data["scale"].item()
                entry["offset"] = encoding.data["offset"].item()
            except Exception:
                # int32 token input carries no encoding; that is expected.
                entry["scale"], entry["offset"] = None, None
            info[key].append(entry)
    mgr.Destroy()
    return info


def write_encodings(path: str, encodings: dict) -> None:
    with open(path, "w") as f:
        json.dump(encodings, f, indent=2)


def main(args):
    os.makedirs(args.artifact, exist_ok=True)
    specs = compiler_specs(args.soc_model)
    soc = get_soc_to_chipset_map()[args.soc_model]

    encodings_path = os.path.join(args.artifact, "encodings.json")
    encodings = {}
    if os.path.exists(encodings_path):
        with open(encodings_path) as f:
            encodings = json.load(f)

    all_stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.only and args.only not in all_stages:
        raise SystemExit(f"--only {args.only!r} is not in --stages {all_stages}")
    stages = [args.only] if args.only else all_stages

    for stage in stages:
        index = all_stages.index(stage)
        binary = os.path.join(args.context_binaries, f"{stage}.bin")
        pte = os.path.join(args.artifact, f"{args.prefix}_{stage}.pte")
        size_mb = os.path.getsize(binary) / 1e6
        print(f"==> {stage}  ({size_mb:.0f} MB context binary)", flush=True)

        encodings[stage] = dump_encodings(binary, specs)
        for kind in ("inputs", "outputs"):
            for entry in encodings[stage][kind]:
                print(
                    f"    {kind[:-1]}[{entry['index']}] {entry['name']} "
                    f"{entry['shape']} {entry['dtype']} "
                    f"scale={entry['scale']} offset={entry['offset']}",
                    flush=True,
                )

        if os.path.exists(pte) and not args.force:
            print(f"    {os.path.basename(pte)} exists, skipping (use --force)", flush=True)
            write_encodings(encodings_path, encodings)
            continue

        # One context binary in flight at a time. Holding all three open at
        # once means 1.3 GB of weights mapped onto the cDSP simultaneously,
        # which is where this falls over.
        start = time.time()
        bundle = from_context_binary(binary, f"ctx_loader_{index}", soc)
        edge_program = bundle["edge_program_manager"]
        backend_config = ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            )
        )
        with open(pte, "wb") as f:
            edge_program.to_executorch(config=backend_config).write_to_file(f)
        del bundle, edge_program
        gc.collect()
        print(
            f"    wrote {os.path.basename(pte)} "
            f"({os.path.getsize(pte) / 1e6:.0f} MB) in {time.time() - start:.1f} s",
            flush=True,
        )

    write_encodings(encodings_path, encodings)
    print(f"==> wrote {encodings_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context_binaries",
        required=True,
        help="Directory holding text_encoder.bin, unet.bin and vae.bin",
    )
    parser.add_argument("-a", "--artifact", required=True, help="Where to write the .pte files")
    parser.add_argument("--soc_model", default="QCS8275")
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help="Comma-separated graphs to wrap, named after their .bin files. "
        "ControlNet-Canny is text_encoder,controlnet,unet,vae.",
    )
    parser.add_argument(
        "--prefix",
        default="sd21",
        help="Filename prefix for the .pte files, so model sets can coexist.",
    )
    parser.add_argument(
        "--only",
        help="Wrap a single stage. The bundle's preprocess_multimethod leaves "
        "state behind that breaks the second call in one process, so "
        "board_sd_export.sh invokes this once per stage.",
    )
    parser.add_argument("--force", action="store_true", help="Re-wrap .pte files that exist")
    main(parser.parse_args())
