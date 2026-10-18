# Whisper on the Hexagon NPU

**Working.** `openai/whisper-tiny`, quantized 16a8w and lowered wholesale to
the QCS8275's Hexagon v75, transcribes a 30-second window in **0.22 s**:

```bash
make board-whisper-export                 # one-off, ~20 min, runs on the board
make board-whisper AUDIO=clip.wav
```

```
--- transcript ---
<|en|><|transcribe|><|notimestamps|> The hexagon neural processing unit on this
board runs whisper entirely offline, no audio ever leaves the device.<|endoftext|>
```

## Measured, same board, same 30-second window

| | Encode | Decode | Per token |
|---|---|---|---|
| **NPU** (16a8w, QNN) | **0.028 s** | **0.190 s** / 25 tokens | **7.6 ms** |
| CPU (eager PyTorch, fp32) | 1.299 s | 5.692 s / 21 tokens | 271 ms |

That is **46× on the encoder** and **~36× per decoded token**. Model load is
0.44 s.

Read the CPU column as a reference point, not a tuned baseline: it is HF
`generate()` in fp32 eager mode, not an XNNPACK `.pte`. The encoder comparison
is the honest one — a single fixed-shape graph, run both ways.

## How it fits together

Three pieces, and the only one we had to write is the middle one:

1. **`qnn_whisper_runner`** — already built, aarch64, in the bundle at
   `build-x86/examples/qualcomm/oss_scripts/whisper/`. Nothing to compile.
2. **`whisper_qnn_16a8w.pte`** (89 MB) — exported by
   [scripts/board_whisper_export.sh](../scripts/board_whisper_export.sh), which
   drives [board_whisper_export.py](../scripts/board_whisper_export.py) on the
   board. Encoder and decoder are two methods in one file; both are
   `QnnBackend`, confirmed with `scripts/pte_backend.py`.
3. **Mel features** — [scripts/make_whisper_raw.py](../scripts/make_whisper_raw.py).
   The runner does no audio handling at all.

### The runner takes raw tensors, not audio

`runner.cpp` hardcodes `{1, 80, 3000}` float32 and reads whatever files
`input_list.txt` names. That shape is Whisper's fixed 30-second window, so
longer audio has to be cut into 30 s chunks, one raw file each — which
`make_whisper_raw.py` does, writing the `input_list.txt` alongside. It decodes
through `ffmpeg`, so mp3/m4a/video work as well as wav, with no extra Python
dependency on the Mac side.

## Four things that had to be worked around

The export only runs on the board — the AOT QNN compiler exists solely as the
bundle's aarch64 `PyQnnManagerAdaptor.cpython-312-aarch64-linux-gnu.so`. On top
of that:

- **`PYTHONPATH=/home/arduino/git`.** The board's `et-venv` has the pip
  `executorch==1.5.1` wheel installed, which has no QNN AOT support and would
  otherwise shadow the bundle's tree. Putting the tree ahead of site-packages
  is what makes `--soc_model QCS8275` resolvable at all.
- **`QNN_SDK_ROOT` has to lie about its architecture.**
  `check_qnn_version.py` hardcodes the host as `x86_64-linux-clang` and looks
  for `$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so`. The fix is a shim
  root at `~/qnn-sdk` whose `lib/x86_64-linux-clang` is a symlink to the
  unpacked aarch64 libraries. The loader only cares what it finds there.
- **Calibration does not go through `datasets`.** Upstream's `get_dataset`
  decodes the audio column with torchcodec, which needs ffmpeg's shared
  libraries; the image has none, and unpacking them with `apt download` was not
  enough — the decode still failed once `datasets` was in the process. Reading
  the parquet with pyarrow and decoding the FLAC with `soundfile` (whose wheel
  bundles libsndfile) gets the same 20 LibriSpeech clips with nothing to
  install. That is the one function `board_whisper_export.py` replaces; the
  lowering, the quantizer config and the model wrappers are all imported from
  the bundle's own `whisper.py`.
- **`--compile_only` never saves the tokenizer.** It exits before the inference
  path that would, and the runner will not start without `tokenizer.json`. The
  export script saves it afterwards.

## Context length is compiled in

`MAX_SEQ_LEN` (default 256) sizes the decoder's static KV cache at export time;
it is not a runtime flag. 256 tokens is generous for one 30 s window. Lowering
it shrinks the decoder graph, raising it costs memory per step.

The audio window is not adjustable at all — 30 s is baked into both the
feature extractor and the runner.

## What this is not, yet

- **whisper-tiny only.** The bundle's `compile_whisper` pins
  `openai/whisper-tiny` for the weights. Base/small would need that hardcoding
  lifted, and the accuracy/latency trade is worth measuring before assuming
  tiny is the right pick.
- **No streaming.** It transcribes complete 30 s chunks from a file. Live
  microphone capture would mean a ring buffer feeding chunks, plus something to
  handle the seams between them.
- **No WER number.** Upstream's eval path needs `torchmetrics` and the dataset
  loader we removed; the calibration transcripts in the export log read
  correctly, which is as far as this was taken.

## Sources

- Bundle: `examples/qualcomm/oss_scripts/whisper/` (`whisper.py`, `runner/`)
- Board runs, 2026-10-17 — see [qualcomm-npu.md](qualcomm-npu.md) for the
  bring-up this builds on
