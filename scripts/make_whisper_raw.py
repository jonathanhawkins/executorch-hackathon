#!/usr/bin/env python3
"""Turn an audio file into the raw mel tensors the QNN Whisper runner expects.

    python scripts/make_whisper_raw.py clip.wav -o out/

`qnn_whisper_runner` does no audio handling at all: it reads whatever files
input_list.txt names as little-endian float32 and feeds them straight in as a
(1, 80, 3000) tensor. That shape is hardcoded in runner.cpp, which is Whisper's
fixed 30-second window — so anything longer has to be cut into 30s chunks, one
raw file each. This writes those files plus the input_list.txt beside them.

Decoding goes through ffmpeg (16 kHz mono f32le) rather than librosa/soundfile,
so it takes mp3/m4a/video as happily as wav and adds no Python dependency.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


def decode(path: Path) -> np.ndarray:
    """Any audio/video file -> mono float32 at 16 kHz."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed on {path}:\n{proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audio -> Whisper raw mel tensors")
    ap.add_argument("audio", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("whisper_input"),
                    help="output directory (default: ./whisper_input)")
    ap.add_argument("--model", default="openai/whisper-tiny",
                    help="which feature extractor to use; must match the export")
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"{args.audio}: not found", file=sys.stderr)
        return 1

    from transformers import AutoProcessor

    audio = decode(args.audio)
    duration = len(audio) / SAMPLE_RATE
    processor = AutoProcessor.from_pretrained(args.model)

    args.output.mkdir(parents=True, exist_ok=True)
    chunks = [audio[i:i + CHUNK_SAMPLES] for i in range(0, max(len(audio), 1), CHUNK_SAMPLES)]

    names = []
    for idx, chunk in enumerate(chunks):
        # truncation+padding take every chunk to exactly 3000 frames, including
        # a short final one.
        feature = processor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="np", truncation=True
        ).input_features.astype(np.float32)
        assert feature.shape == (1, 80, 3000), feature.shape
        name = f"input_{idx}.raw"
        feature.tofile(args.output / name)
        names.append(name)

    (args.output / "input_list.txt").write_text("\n".join(names) + "\n")

    print(f"  in     {args.audio.name}  ({duration:.1f}s)")
    print(f"  chunks {len(names)} x 30s, (1, 80, 3000) float32")
    print(f"  out    {args.output}/  (+ input_list.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
