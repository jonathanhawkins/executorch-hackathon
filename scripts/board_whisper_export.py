#!/usr/bin/env python3
"""Export Whisper to a QNN .pte. Runs ON THE BOARD, driven by board_whisper_export.sh.

A thin driver around the bundle's own
`examples/qualcomm/oss_scripts/whisper/whisper.py`, which is imported rather
than copied — the lowering, the 16a8w quantizer config and the encoder/decoder
wrappers all come from there.

The one thing it replaces is calibration data loading. Upstream's `get_dataset`
hands the audio column to `datasets`, which decodes it through torchcodec,
which wants ffmpeg's shared libraries; the board's image has none and the
decode fails even with them unpacked alongside. Reading the parquet with
pyarrow and decoding the FLAC with soundfile (whose wheel bundles libsndfile)
gets the same LibriSpeech samples with nothing to install.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys

import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, list_repo_files

import executorch.examples.qualcomm.oss_scripts.whisper.whisper as W

CALIB_REPO = "hf-internal-testing/librispeech_asr_dummy"
SAMPLE_RATE = 16_000


def calibration_inputs(model_id: str, count: int):
    """LibriSpeech validation clips as (1, 80, 3000) mel tensors."""
    from transformers import AutoProcessor

    parquet = next(
        f
        for f in list_repo_files(CALIB_REPO, repo_type="dataset")
        if f.endswith(".parquet")
    )
    table = pq.read_table(hf_hub_download(CALIB_REPO, parquet, repo_type="dataset"))
    processor = AutoProcessor.from_pretrained(model_id)

    inputs, target = [], []
    for row in table.slice(0, count).to_pylist():
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        feature = processor(
            audio, sampling_rate=sr, return_tensors="pt", truncation=False
        ).input_features
        inputs.append((feature,))
        target.append(row["text"])
    logging.info("calibration: %d clips from %s", len(inputs), CALIB_REPO)
    return inputs, target


def main() -> int:
    parser = W.setup_common_args_and_variables()
    parser.add_argument("-a", "--artifact", default="./whisper")
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--calib_samples", default=20, type=int)
    # Only the calibration feature extractor: the bundle's compile_whisper
    # pins openai/whisper-tiny for the weights themselves.
    parser.add_argument("--model_id", default="openai/whisper-tiny")
    args = parser.parse_args()

    qnn_config = W.QnnConfig.load_config(
        args.config_file if args.config_file else args
    )
    # compile_whisper reaches for this as a module global, not through its
    # argument — setting it here is what makes the import-and-reuse work.
    W.qnn_config = qnn_config

    inputs, _ = calibration_inputs(args.model_id, args.calib_samples)
    W.compile_whisper(args, qnn_config, inputs)
    print(f"\nWrote {args.artifact}/{W.WHISPER_PTE_FILENAME}")
    return 0


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    sys.exit(main())
