# The Hexagon NPU path (QNN backend)

**Working.** Gemma 4 E2B runs on the Hexagon NPU today:

```bash
make board-npu-run                                # default prompt
PROMPT="Explain edge AI." make board-npu-run
```

## Measured: NPU vs CPU, same board

| | Model | Prompt eval | Generation | Resident |
|---|---|---|---|---|
| **NPU** | Gemma 4 E2B (~2B) | **188 tok/s** | **16.3 tok/s** | **494 MiB** |
| CPU | Llama 3.2 1B | 15.3 tok/s | 10.1 tok/s | 789 MB mmap'd |
| CPU | Qwen2.5 0.5B | 22.3 tok/s | 20.5 tok/s | 396 MB mmap'd |

Three things worth drawing out:

- **Prompt eval is ~12× faster** (188 vs 15.3 tok/s). This is the headline
  number — it's where the NPU wins outright, and it's what makes a long system
  prompt or a RAG context feel instant rather than sluggish.
- **Generation is 1.6× faster on a roughly 2× larger model**, so comparing E2B
  against Llama 1B understates the gap.
- **494 MiB resident for a 2.8 GB model** — the weights live in the NPU context
  rather than host RAM, which is a real win on a constrained board.

Model load is 3–8 s depending on page cache. Warm it with a throwaway
generation before demoing.

## Vision: InternVL3-**1B** works, 2B does not

```bash
make board-npu-vision                          # sample image
make board-npu-vision IMAGE=~/Desktop/shot.png
```

Asked *"What landmark is shown, and where is it?"* about the bundled Statue of
Liberty photo, it answers **"The image shows the Statue of Liberty"** — running
entirely on the Hexagon.

| | InternVL3-1B on NPU |
|---|---|
| Model load | 2.1 s |
| Vision encoder | ~0.5 s per image |
| Prompt eval | 34.3 tok/s (270 tokens, of which 256 are the image) |
| Generation | ~33 tok/s |

### Why not the 2B

**InternVL3-2B does not fit this chip.** Its vision encoder runs fine alone
(466 ms, 256 × 1536 features), and the decoder and token-embedding graphs load,
but all three together exhaust the cDSP's protection-domain memory:

```
Could not map shared weights buffer!
Failed to find available PD for contextId 3 ... context size estimate 486802688
```

Two signs it was never the intended target: the bundle's `decoder_constants.py`
maps **only** `"internvl3_1b"`, and upstream's README lists InternVL3 **1B**
under Vision-Language models. A sizing mismatch, not a bug to chase.

### Two settings that are not optional

- **`seq_len` must exceed 256.** One image costs `img_seq_len = 256` tokens
  before any text, so a text-sized default like 128 aborts with "sequence
  length exceeded" before generating anything. The script defaults to 640;
  the compiled ceiling is 1000.
- **`temperature` must be above 0.** The runner defaults to 0.0 (greedy
  argmax), which sends this quantized 1B into hard repetition loops — "New
  York, New York, New York…" forever. 0.7 is not enough; **1.0–1.2** produces
  answers that terminate properly. Short specific questions beat long
  open-ended ones; "describe this image in detail" still rambles.

**Download `InternVL3-1B_QCS8275_ctxt_1000` from the Drive folder** — all six
files, including the three `.pte` — push it to `~/models/internvl3-1b`, then:

```bash
make board-npu-vision                       # uses the bundle's sample image
make board-npu-vision IMAGE=~/Desktop/shot.png
```

With no `IMAGE`, it falls back to the Statue of Liberty sample shipped in the
bundle's assets, so the command works with no arguments.

### Two traps in the vision path

- **`--image_path` is an input *list* file, not the tensor.**
  `load_raw_files()` opens it and reads newline-separated paths. Point it
  straight at a `.raw` and it tries to open the tensor's bytes as filenames,
  failing with a screenful of binary.
- **The runner wants a preprocessed raw tensor, not a JPEG.**
  `scripts/make_vision_raw.py` produces it: RGB, 448×448 bicubic, ImageNet
  normalization, NCHW float32 (2,408,448 bytes). That reimplements InternVL's
  documented transform, because the HF processor route needs
  `trust_remote_code` and the `-hf` repo errors on transformers 5.0.0rc1.

`--eval_mode 0` (KV-only) uses less DSP memory than the default hybrid mode and
gets further; it's the default in the script.

## How it was unblocked

The track team's bundle supplies both missing pieces: a QCS8275-capable
ExecuTorch tree *and* pre-built NPU models. Confirmed in the extracted checkout:

```
backends/qualcomm/serialization/qc_schema.py:
    QCS8275 = 82  # v75
    QcomChipset.QCS8275: SocInfo(QcomChipset.QCS8275, HtpInfo(HtpArch.V75, 8))
```

That `V75` matches what the QAIRT libraries independently say about this chip.

It also ships **prebuilt aarch64 binaries**, despite the misleading
`build-x86/` directory name — `file` reports `ELF 64-bit LSB pie executable,
ARM aarch64`. So there was no build step at all:

```
build-x86/examples/qualcomm/oss_scripts/llama/qnn_llama_runner
build-x86/backends/qualcomm/libqnn_executorch_backend.so
build-x86/backends/qualcomm/PyQnnManagerAdaptor.cpython-312-aarch64-linux-gnu.so
```

That last one is the AOT module absent from every pip wheel, built for aarch64
cp312 — so this tree can also *export* new QNN models natively on the board.

## No sudo was needed

The bring-up PDF's `apt install` steps turned out to be avoidable. Everything
resolves from unpacked `.deb`s under `~/qairt-probe` plus the bundle's own libs:

```
LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:
                $PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp
```

One gap had to be filled: `libQnnSystem.so` needs **`libatomic.so.1`**, which
this image lacks. `apt download libatomic1` + `dpkg-deb -x` solves it without
root. If you'd rather install properly, the sudo block is still available via
`scripts/board_npu_setup.sh --print`.

A harmless warning appears on every run: *"Qnn API version 2.38.0 is used. The
version is tested against 2.35.0."* It works regardless.

## The models in the bundle are NPU builds — verified

Not inferred from their filenames. Verified by reading the delegate records
inside the `.pte`:

```
gemma3-1b_qcs8275_ctxt-5000.pte   QnnBackend      -> NPU
models/llama3.2-1b-open/model.pte XnnpackBackend  -> CPU (ours)
```

Check any `.pte` yourself:

```bash
python scripts/pte_backend.py path/to/model.pte
```

It works on a partial download, since delegate records sit near the start of
the file — handy for a 1 GB model.

Note the shape of the result: our CPU exports show **84–154** `XnnpackBackend`
records (many small partitions carved out of the graph), while the QNN build
shows **2** (the graph lowered essentially wholesale to the NPU). That's the
difference between "accelerate the bits we can" and "run this on the Hexagon".

## What's in the Drive folder

| Item | What it is |
|---|---|
| `gemma3-1b_qcs8275_ctxt-5000` | Gemma 3 1B, 5000-token context — **the one to start with** |
| `gemma4_e2b_QCS8275_ctxt_2000` | Gemma 4 E2B, 2000-token context |
| `InternVL3-1B_QCS8275_ctxt_1000` | Vision-language, 1B |
| `InternVL3-2B_QCS8275_ctxt_1000` | Vision-language, 2B |
| `depthanything_v2_small_QCS8275` | Monocular depth estimation |
| `source/ventuno_executorch_source.tar.gz` | **2.96 GB** — the QCS8275-capable ExecuTorch |
| `source/Ventuno_Q_ExecuTorch_Bringup.pdf` | Bring-up instructions |

Each model folder has the same layout as our own exports — `.pte`,
`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` — so
`scripts/board_run.sh`'s push logic applies, though these need the QNN runner
rather than our Python path.

Context length is **compiled in** (`ctxt-5000`), not a runtime flag. That's
normal for QNN: the graph is static, so sequence length is fixed at export.

## Why this solves what upstream couldn't

Two blockers, both fixed by the prepared source tree:

1. **Upstream has no QCS8275 target.** The bundled checkout adds it.
2. **Upstream gates QNN Python bindings on x86-64.** The bring-up guide
   compiles `add.pte` *on the board* (aarch64), so the prepared tree lifts that
   gate too.

That second point matters — it means you do **not** need an x86 Linux host.
Everything happens on the VENTUNO Q.

## Bring-up

Two make targets do the parts that can be automated:

```bash
make board-npu          # prove the NPU executes code (no sudo, installs nothing)
make board-npu-setup    # extract the bundle on the board, then print the sudo steps
```

`board-npu-setup` extracts the archive, checks the checkout really contains a
`QCS8275` target, looks for an already-built runner, and prints the commands
that need your password. Everything after that is the block below.

### The manual steps, from the bundle's own PDF

On the board (`make board-shell`), after copying the archive across:

```bash
cd ~ && tar -xzvf ventuno_executorch_source.tar.gz     # lands in ~/git/executorch

sudo apt install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-qcom-iot/qcom-ppa
sudo apt update
sudo apt install -y libqnn1 libqnn-dev qnn-tools
sudo apt install -y python3.12-venv

cd ~/git/executorch
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd ~/git/executorch/examples/qualcomm/scripts
python export_example.py --soc_model add -g      # produces add.pte
```

All of that needs your `sudo` password — `sudo` on the board is not
passwordless.

Two notes from our own investigation that the PDF doesn't mention:

- `python3.12-venv` is genuinely missing from the image (that's why our CPU
  setup uses `uv` instead). Step 6 installs it.
- The `qcom-ppa` is in fact *already* configured on this board's image, so
  `add-apt-repository` may report it as present. Harmless.

## The runtime half was already proven

Before any of this, `make board-npu` confirmed the NPU executes code —
it installs nothing and needs no `sudo`:

```
Core Version of the backend DSP: Hexagon Architecture V75
Unit Test on the backend DSP: Passed.
```

So the hardware was never in doubt; only the compiler was.

## Recommendation

**Grab `gemma3-1b_qcs8275_ctxt-5000` first** and run it with the bundled QNN
runner. It's a 1 GB download against a model class we already have a CPU
baseline for — Llama 3.2 1B at **10.1 tok/s** on CPU — so you get a direct
NPU-vs-CPU comparison, which is exactly the kind of number worth putting on a
slide.

Keep the CPU path working as the fallback. An NPU `.pte` will not run without
the QNN runtime and the matching runner, and our CPU `.pte` files will not use
the NPU no matter what's installed. They are separate artifacts.

## Speech

Whisper also runs on the Hexagon, but unlike the models above it is not in
the bundle — it is exported on the board from the bundle's own `whisper.py`.
See [whisper-npu.md](whisper-npu.md).

## Sources

- Track team Drive bundle, incl. `Ventuno_Q_ExecuTorch_Bringup.pdf`
- [Official hackathon track guide](https://github.com/aaishikasb/qc-executorch-guide/tree/main/IoT)
- Board probes and `.pte` delegate inspection, 2026-10-17
