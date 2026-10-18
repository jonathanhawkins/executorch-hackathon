# ExecuTorch Hackathon

On-device LLM inference with [ExecuTorch](https://executorch.ai/).
Founders, Inc. — October 17–18, 2026. Part of Open Source AI Week.

The pipeline works end to end today: a PyTorch model from Hugging Face gets
exported to a quantized `.pte`, and that same file runs on the Mac now and on
the device later.

## Quick start

```bash
make setup    # venv + ExecuTorch + optimum-executorch (~2GB of torch)
make doctor   # confirm the toolchain is sane
make smoke    # export a 135M model and run it, end to end
```

`make smoke` takes about three minutes and proves the whole chain works before
you depend on it.

## What's verified

All three models export on the Mac **and run on the Arduino VENTUNO Q board**.
XNNPACK CPU backend, `8da4w` quantization.

| Model | `.pte` | Export (Mac) | Mac gen | **Board gen** |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | 100 MB | 166 s | 341 tok/s | **41.1 tok/s** |
| Qwen2.5-0.5B-Instruct | 396 MB | 173 s | 224 tok/s | **20.5 tok/s** |
| Llama-3.2-1B-Instruct | 789 MB | 224 s | 141 tok/s | **10.1 tok/s** |

Comfortable reading speed is about 10 tok/s, so **Llama 3.2 1B is genuinely
usable on the board** and Qwen 0.5B feels snappy. Full board detail, including
what the hardware really is, in [docs/board.md](docs/board.md).

Mac numbers are for comparing exports, not predicting hardware.

## Models

```bash
make list                        # show presets
make export MODEL=llama3.2-1b    # export one
make run    MODEL=llama3.2-1b    # run it, with the chat template applied
```

| Preset | Model | Status |
|---|---|---|
| `smollm2-135m` | HuggingFaceTB/SmolLM2-135M-Instruct | exported, verified |
| `qwen2.5-0.5b` | Qwen/Qwen2.5-0.5B-Instruct | exported, verified |
| `llama3.2-1b-open` | unsloth/Llama-3.2-1B-Instruct | exported, verified |
| `gemma3-1b-open` | unsloth/gemma-3-1b-it | ungated, untested |
| `llama3.2-1b` | meta-llama/Llama-3.2-1B-Instruct | awaiting approval |
| `gemma3-1b` | google/gemma-3-1b-it | awaiting approval |

### About the gated repos

`meta-llama/Llama-3.2-1B-Instruct` and `google/gemma-3-1b-it` are both
**`gated: manual`**. Accepting the license doesn't grant access — it files a
request that a human has to approve, which can take hours or days. Until then
every download 403s with "you are not in the authorized list."

So the `-open` presets point at community mirrors of the **same weights**
without the gate. Use those now; switch to the official repos once the requests
clear. Nothing else changes — same architecture, same tokenizer, same export.

If a request seems stuck, check that the HF account you accepted the license
with is the same one your local token belongs to:

```bash
.venv/bin/hf auth whoami
```

`export_model.py` prechecks access, so a gated model fails in two seconds with
a link rather than a 30-line traceback.

Add your own models in [models.toml](models.toml) — quote any name containing
a dot, or TOML will read it as a nested table and silently drop the preset.

Anything `optimum-executorch` supports works via `--model`:

```bash
python scripts/export_model.py --model Qwen/Qwen2.5-1.5B-Instruct
```

## Running on the board (IoT track) — working

The attached Arduino VENTUNO Q is **Ubuntu aarch64, not Android**, so it runs
the same `.pte` through Python rather than an APK:

```bash
make board-info                        # what's actually plugged in
make board-setup                       # one-off runtime install on the board
make board-run MODEL=llama3.2-1b-open  # push if needed, run on board
```

See [docs/board.md](docs/board.md) — including why the hardware is better than
its spec sheet (QCS8275, Cortex-A78C, 14 GB RAM).

## Running on Android (Mobile + XR track) — builds, untested

There's a Compose demo app in [`android/`](android) that loads a `.pte` through
the `executorch-android` AAR and streams tokens. It compiles and bundles the
native runtime for `arm64-v8a`, but **has not run on hardware** — no Android
device has been attached.

```bash
make apk                               # build it
make deploy MODEL=llama3.2-1b-open     # push model + install + launch
make logcat                            # watch for load errors
```

Detail in [docs/android.md](docs/android.md).

## Picking a track

Per-track detail in [docs/tracks.md](docs/tracks.md).

### On the Hexagon NPU — working

```bash
make board-npu-run    # Gemma 4 E2B on the Hexagon
```

Measured on the same board, NPU against our CPU baseline:

| | Model | Prompt eval | Generation | Resident |
|---|---|---|---|---|
| **NPU** | Gemma 4 E2B (~2B) | **188 tok/s** | **16.3 tok/s** | **494 MiB** |
| CPU | Llama 3.2 1B | 15.3 tok/s | 10.1 tok/s | 789 MB mmap'd |

**Prompt eval is ~12× faster**, generation 1.6× faster on a ~2× larger model,
and resident memory is a fraction — the weights live in the NPU context.

This needs the track team's bundle (upstream has no QCS8275 target). It ships
prebuilt aarch64 runners, so there's no build step and no x86 host. Details,
including why no `sudo` was needed, in
[docs/qualcomm-npu.md](docs/qualcomm-npu.md).

```bash
python scripts/pte_backend.py <any>.pte   # QnnBackend -> NPU, XnnpackBackend -> CPU
```

## The virtual billboard

```bash
make web      # then press "Start billboard"
```

Stand in front of the camera and you appear wearing the shirt. BiSeNet
segments head/hair/neck on the Hexagon NPU (**12 ms** of compute) and the
cutout drops into the neck of a product photo, scaled by eye separation so it
sits right whether you lean in or stand back. ~1 fps end to end.
See [docs/billboard.md](docs/billboard.md).

## The walkthrough deck

```bash
make web      # then open http://<board>:3000/deck
```

The presentation, served from the board like everything else — 15 slides on
the store setup, the hardware, the techniques, why the ControlNet path won,
and what we would do next. Arrow keys to
navigate, `G` for the overview grid, `F` for full screen, and every slide has
a hash (`/deck#p-winner`) so a mid-talk reload lands where you were.

## The camera demo

```bash
make web      # camera + Next.js app, all on the board
```

A USB camera on the board, InternVL3-1B on the Hexagon, and the web page served
from the same device — nothing leaves it. ~12 s per frame end to end. See
[docs/web-demo.md](docs/web-demo.md).

### Live speech — working

Speak into the camera's mic; the transcript and a translation appear on the
page. Transcription alone lands **~1.8–2.6 s** after you speak; with
translation on it is ~9 s, because Gemma's 4.5 s dominates. Capture is
continuous, so nothing is lost while the NPU is busy.

Whisper's own `translate` task isn't reachable through this runner, so
translation is a second NPU pass through Gemma — which also means any target
language, not just English. See [docs/live-speech.md](docs/live-speech.md).

### Vision on the NPU — working

```bash
make board-npu-vision                          # bundled sample image
make board-npu-vision IMAGE=~/Desktop/shot.png
```

InternVL3-**1B** runs the full three-graph pipeline (vision encoder → token
embedding → decoder) on the Hexagon: ~0.5 s to encode an image, 34 tok/s prompt
eval, ~33 tok/s generation. Shown the Statue of Liberty, it identifies it.

InternVL3-2B does **not** fit — it exhausts the cDSP's protection-domain
memory. Use the 1B.

### Speech on the NPU — working

```bash
make board-whisper-export            # one-off, ~20 min, runs on the board
make board-whisper AUDIO=clip.wav
```

Whisper-tiny, 16a8w, lowered wholesale to the Hexagon: a 30-second window takes
**0.22 s** — 0.028 s to encode, 7.6 ms per decoded token. The same clip through
eager fp32 PyTorch on the board's CPU takes 1.3 s to encode and 271 ms a token.

Unlike the LLM and vision models, this one is **not** in the bundle — it is
exported on the board from the bundle's own `whisper.py`. The workarounds that
took (a lying `QNN_SDK_ROOT`, calibration that avoids `datasets`) are in
[docs/whisper-npu.md](docs/whisper-npu.md).

### Face parsing on the NPU — working

```bash
make board-faceparse-export                   # one-off, ~10 min, on the board
make board-faceparse IMAGE=frame.jpg          # finds the face itself
```

BiSeNet face parsing, 8a8w, lowered wholesale to the Hexagon: a head/face/neck
cutout in **12.3 ms**, against **978 ms** for the same model in eager fp32 on
the board's own CPU — **80×**. Quantization costs almost nothing here: against
the fp32 model, per-pixel class agreement is 99.6% and head-mask IoU is 0.9990.

This is the matte for the virtual-billboard idea — everything except background
and `cloth`, so the garment is already excluded at the neckline. YuNet finds the
face first and the crop follows it, which is not optional: without it, a 16:9
frame makes the model invent hats. Detail, including a flat-background failure
that is *not* yet solved, in [docs/faceparse-npu.md](docs/faceparse-npu.md).

### Person segmentation on the NPU — working

```bash
make board-personseg-export                     # one-off, ~8 min, on the board
make board-personseg IMAGE=frame.jpg            # the silhouette
```

A human-parsing SegFormer-B0 (ATR, 18 classes), 16a8w, lowered wholesale to the
Hexagon: the person's silhouette **and their arms** in **24.9 ms**, against
**1668 ms** for the same model in eager fp32 on the board's own CPU — **67×**.
A transformer, not a convnet, and it still lowers as a single `QnnBackend`
delegate.

This is what clips the garment to the wearer's outline and puts a forearm back
in front of it. The bundle's own DeepLabV3-ResNet101 was exported as the
baseline and is faster (8.4 ms at 224×224), but VOC's `person` is a single
blob — it cannot tell an arm from a chest — and its real mask resolution is
28×28, against SegFormer's 128×128.

Quantization is not free here, unlike face parsing: at 8a8w the silhouette
holds up (IoU 0.96–0.99 against fp32) but the arms do not (0.48–0.92), which is
why the default is 16a8w at 24.9 ms rather than 8a8w at 11.2. Detail, including
the frame-budget cost and a resize trick that is the difference between a
usable clip and a staircase, in
[docs/personseg-npu.md](docs/personseg-npu.md).

### Image generation on the NPU — working

```bash
make board-sd-export                     # one-off, ~15 s, on the board
make board-sd PROMPT="studio product photo of a folded white cotton t-shirt on a grey background, soft lighting, high detail"
```

Stable Diffusion v2.1 at 512×512, w8a16, straight from Qualcomm AI Hub's
prebuilt QCS8275 context binaries — **0.30 s per denoising step**, so a
20-step image is **5.9 s of NPU compute and 8.8 s of wall clock**. One UNet
execution is 148 ms, the text encoder 10 ms, the VAE decode 272 ms. Sampling
is DPM-Solver++(2M) on Karras sigmas with a default negative prompt, which
costs no extra NPU time and is worth more than doubling the step count.

![a folded white t-shirt, generated on the board](assets/sd21-tshirt.png)

That is six times inside the 2 s/step line we set for viability — but speed
turned out not to be the binding constraint. **About one generation in six is
usable**: six seeds of one garment prompt gave one clean t-shirt and four
images that were not a t-shirt at all, and raising guidance did not help
because the same seeds fail the same way. Good for generating a catalogue
offline and picking; not good for a passerby pressing a button once.
At **8 steps a product shot is as good and takes 2.4 s** of NPU time, 5.1 s
wall — though people need the full 20; flat-lays converge early, figures do
not. All three graphs (1.3 GB) are resident on the cDSP at once while an image
generates, alongside the billboard daemon's two.

It used to take 28–45 s rather than 8.8, because the prebuilt
`qnn_executor_runner` predates its own source's persistent-stdin mode and the
881 MB UNet was reloaded once per step. That runner has been rebuilt, the
graphs are held open for the length of an image, and the output is
byte-identical — [docs/persistent-runner.md](docs/persistent-runner.md).

Two surprises worth knowing before touching this: the UNet is
**v-prediction**, not epsilon as the bundle's own runner assumes — run it as
epsilon and every image is RGB static; and its timestep input is a scalar
quantized over `[0, 951]`, so the usual t=999 start silently saturates.
Detail, including why no `diffusers` download is needed at all, in
[docs/stable-diffusion-npu.md](docs/stable-diffusion-npu.md).

### Conditioning on a real photo — working

```bash
make board-cn-export                        # once, ~20 s
make board-cn IMAGE=frame.jpg PROMPT="studio photo of a man wearing a bright red t-shirt"
```

AI Hub's **ControlNet-Canny** build for the QCS8275 takes a photo as input,
and the pose, framing, body and garment silhouette of the result follow it
exactly. **0.37 s per step** — four graph runs instead of two — so 20 steps is
**7.4 s** of NPU time. ControlNet is 77 ms, the UNet 108 ms.

It needs **no VAE encoder**: `image_cond` is `[1, 512, 512, 3]` in pixel
space, so the edge map goes straight in. What it does not carry is identity —
the face comes out as someone else — so it composes with the head matte or the
GHOST swap rather than replacing them.

**It also fixes the reliability problem.** The same six-seed test that gave
1-of-6 usable images from pure noise gives **6 of 6** with conditioning, and
zero wrong subjects — no fabric swatches, no yellow tins. Every seed is a man
facing the camera in a t-shirt, framed like the source. That is the difference
between "generate four and pick" and something a stranger can press once.

It is a separate SD **1.5** stack, so epsilon-prediction and EOS padding, not
2.1's v-prediction and `"!"` padding. Same doc.

### Generated garment, real face — working

```bash
make board-cn-tryon IMAGE=frame.jpg GARMENT="a bright red crew neck t-shirt"
```

![source, Canny, ControlNet output, head restored](assets/cn15-pipeline.png)

ControlNet gets the clothes right and the face wrong, so the face is not
generated at all: BiSeNet cuts the real head out (11.9 ms) and it goes back
on. Because the generation was conditioned on that same photo, the head is
already in the right place at the right scale — **a paste, not a warp**,
unlike pasting onto a stock product shot. Identity is exact by construction.

This is wired into the billboard as a **`generate` mode**: pick it in the web
UI, choose a garment, press the shutter, **~7.5 s** at 8 steps or ~14 s at 20
— it was 21–26 s before the QNN runner was held open across the denoising
loop ([docs/persistent-runner.md](docs/persistent-runner.md)). The
conditioning is masked to the person's silhouette first — without that it
generates the room, not the shopper. Still to confirm: the output with someone
standing at the distance the framing gate asks for.

### Face swap on the NPU — working

```bash
make board-faceswap-export                   # one-off, ~12 min, on the board
make board-faceswap IMAGE=frame.jpg          # onto the default product photo
make ghost-swap SOURCE=frame.jpg             # the same thing, fp32, on the Mac
```

The other way round from the garment overlay: instead of moving a shirt onto a
live frame, move the shopper's face into a studio photo that already has the
shirt, the lighting and the pose. GHOST (`ai-forever/ghost`, Apache-2.0),
184 M parameters, vendored and exported on the board like BiSeNet before it.

**82 ms a swap** on the Hexagon against **2064 ms** for the same model through
XNNPACK on the board's own CPU — **25×** — matching the fp32 reference to
**1.78/255** and preserving identity (ArcFace cosine 0.870 to the shopper,
against 0.877 for fp32 and −0.088 for the unswapped photo). That 82 ms is the compute, and the
shutter is now **~120 ms** end to end: the Whisper session's persistent-stdin
`qnn_executor_runner` keeps the 357 MB `.pte` loaded instead of reloading it
every shot, which took the shutter from ~960 ms with output byte-identical.
For comparison, [docs/faceswap.md](docs/faceswap.md) measured ~4.4 s
of *pure compute* for the CPU-only `inswapper_128` swap. There is also a CPU fallback that needs no
QNN at all: `BACKEND=cpu make board-faceswap`, slow but numerically exact.

Whole-graph delegation and no shims at all — unlike `inswapper_128`, which is
ONNX-only and needed 40 baked shims to survive `torch.export`.

(Eager PyTorch on the board is 4379 ms, but that measures the board's torch
build — PyPI's sbsa/CUDA wheel, `CPU capability usage: DEFAULT`, i.e. scalar
fallback kernels — not its cores. Quoting it would have claimed 53×.)

**The interesting part is that it only works unquantized.** At 16a8w the
generator ran 4 % faster and was visibly broken — 47.5/255, colour speckle
across the face — and the cause is not bit width: faithful PyTorch simulation
of the same weight and activation quantization predicts 0.2/255, and clipping
the ranges or changing the observer measurably makes it *worse*. The one
mechanism that reproduces the right order of magnitude is the 21
`InstanceNorm2d` layers' internal statistics in fixed point. Lowering the float
graph instead lets the HTP run it in fp16 — which this IoT-tier SoC turns out
to support — for 3 ms more and 27× the accuracy. **Worth re-checking on the
other NPU models here: quantization bought 4 % on this one.**

The identity encoder is the exception that proves it: ArcFace does not survive
quantization at any config (cosine 0.118 at 16a8w), so it runs fp32 on the CPU,
where it is exact and free — it describes the person, not the shutter, so it is
computed while they walk up.

Whether this should replace the head-cutout billboard is a product call, not a
technical one — and there are now three paths, not two. The cutout is literally
them and reads as pasted on; this swap reads as one photograph but gives them
the model's hair; `make board-cn-tryon` generates the body from their own photo
and keeps their real head, beard included. A swap structurally cannot keep hair
or a beard, so for a distinctive one that third path is the right answer. The licence check that
motivated the whole exercise — whether Apache-2.0 really covers the weights —
came out better than feared but not conclusively. All of it, with the
measurements, in [docs/faceswap-npu.md](docs/faceswap-npu.md).

### Garment overlay — working

```bash
make board-tryon IMAGE=frame.jpg                     # CPU only
OCCLUDE=1 make board-tryon IMAGE=frame.jpg           # + head matte from the NPU
OCCLUDE=1 CLIP=1 make board-tryon IMAGE=frame.jpg    # + body clip and arms
```

The billboard effect, the right way round: the garment goes onto the live
person rather than the person's head onto a product photo, so the head is
already correct and the whole thing moves with them. On the board: **~37 ms a
frame**, about 27 fps — 11.5 ms to detect the face at `scale=3`, 12–16 ms to
parse it on the NPU, 8.4 ms to warp and composite.

`CLIP=1` adds the second NPU model and about **56 ms**, so ~93 ms a frame and
11 fps. What it buys is that the garment stops floating: clipped to the real
outline it reads as worn even when the placement is a few degrees out, which
makes it worth the budget.

Placement is still estimated from face proportions, so it assumes someone
facing the camera square on. Detail in [docs/tryon.md](docs/tryon.md).

## Layout

```
models.toml           model presets
Makefile              every task worth repeating
scripts/
  setup.sh            venv + deps
  doctor.py           toolchain check, incl. a real XNNPACK round-trip
  export_model.py     HF model -> quantized .pte
  run_model.py        run a .pte, report tok/s
  push_android.sh     adb push to device
  board_whisper_export.sh/.py   export Whisper to a QNN .pte, on the board
  board_whisper_run.sh          transcribe an audio file on the NPU
  make_whisper_raw.py           audio -> the raw mel tensors the runner wants
  board_faceparse_export.sh/.py export BiSeNet face parsing to a QNN .pte
  board_faceparse_run.sh        cut out a head on the NPU
  make_face_raw.py              image -> the raw tensor the face parser wants
  face_detect.py                YuNet: where is the face, and its landmarks
  board_personseg_export.sh/.py export person segmentation to a QNN .pte
  board_personseg_run.sh        segment the person on the NPU
  make_person_raw.py            frame -> the raw tensor the segmenter wants
  person_seg.py                 the candidate models, and an fp32 reference
  person_matte.py               logits -> silhouette, arms, existing garment
  garment_clip.py               clip the garment to the body, arms back on top
  garment_overlay.py            warp a garment onto the person
  board_tryon.sh                detect + parse + segment + composite, end to end
  board_sd_export.sh/.py        wrap the AI Hub SD context binaries into .pte
  board_sd_run.sh               generate an image on the NPU
  board_sd_driver.py            the SD graphs + the denoising loop, on the board
  make_sd_tokens.py             prompt -> the raw CLIP token tensors
  make_canny_raw.py             photo -> the Canny edge tensor ControlNet takes
  board_cn_tryon.sh             generate the garment, keep the real face
  sd_head_composite.py          put the real head back on a generated body
  board_tryon_gen.py            the generate mode: frame -> garment -> real head
  make_tryon_prompts.py         pre-tokenize the garment catalogue
  face_matte.py                 logits -> a feathered head cutout
  face_parse.py                 fp32 reference, for comparing against the NPU
  board_faceswap_export.sh/.py  export GHOST face swap to QNN .pte files
  board_faceswap_run.sh         swap a face onto a product photo on the NPU
  make_ghost_raw.py             frame + photo -> the raw tensors GHOST wants
  ghost_swap.py                 fp32 reference, and the paste-back both paths use
  ghost_paste.py                NPU output -> finished photo, and the fp32 diff
  ghost_xnnpack_export.py       GHOST -> XNNPACK .pte: CPU fallback + fair baseline
  face_swap.py                  the older inswapper_128 swap, CPU-only
docs/
  tracks.md           the three tracks and what each costs
  android.md          device deploy detail
  qualcomm-npu.md     the NPU path
  whisper-npu.md      speech-to-text on the NPU
  tryon.md            the garment overlay
  faceparse-npu.md    head/face/neck segmentation on the NPU
  personseg-npu.md    whole-body segmentation on the NPU, and the garment clip
  stable-diffusion-npu.md  image generation on the NPU, and what it costs
  faceswap.md         the inswapper_128 face swap, on the CPU
  faceswap-npu.md     GHOST face swap on the NPU, and whether it beats the cutout
  ideas.md            project ideas scratchpad
assets/garments/      garment PNGs plus their shoulder anchors
assets/sd21-*.png     images generated on the board
assets/cn15-*.png     ControlNet-conditioned generations
vendor/
  face_parsing/       BiSeNet, vendored from zllrunning (MIT)
  ghost/              GHOST AEI-Net + ArcFace, vendored from ai-forever (Apache-2.0)
models/               exported .pte files (gitignored — they're big)
```

## Notes on the toolchain

- Python **3.12**. ExecuTorch allows 3.10–3.14, but the HF stack is happiest here.
- `optimum-executorch` pins `transformers==5.0.0rc1`, so installs need
  `--prerelease=allow`. `scripts/setup.sh` already does this.
- Quantization defaults to `8da4w` (int8 dynamic activations, int4 weights) for
  linear layers. That's the documented default for on-device decoder models.
- `.pte` files are gitignored. Don't commit a 100 MB artifact — re-export instead.
