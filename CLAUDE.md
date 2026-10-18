# Repo conventions

ExecuTorch hackathon project. On-device LLM inference.

## Environment

- Python **3.12** in `.venv` at the repo root. Never install into system Python.
- `optimum-executorch` pins `transformers==5.0.0rc1`, so any `uv pip install`
  here needs `--prerelease=allow` or the resolver rejects the whole set.
- Prefer the `make` targets over raw commands — they put `.venv/bin` on PATH,
  which `optimum-cli` needs.

## Pipeline

HF model → `optimum-cli export executorch` → quantized `.pte` → ExecuTorch
runtime (Mac now, device later). The same `.pte` runs in both places; that's
the whole point of the setup.

- `scripts/export_model.py` — presets from `models.toml`, wraps `optimum-cli`
- `scripts/run_model.py` — runs a `.pte`, applies the chat template
- `scripts/doctor.py` — toolchain check incl. a real XNNPACK lowering round-trip
- `scripts/push_android.sh` — `adb push` to `/data/local/tmp/llama`

The billboard try-on runs two NPU models over one frame: BiSeNet face parsing
for the head matte (`board_faceparse_*`) and a human-parsing SegFormer for the
body (`board_personseg_*`). Placement lives in `garment_overlay.py`; clipping
and occlusion live in `garment_clip.py`, and they are kept apart on purpose —
a garment clipped to the real outline reads as worn even when the placement is
a little off. Both are opt-in: `OCCLUDE=1` and `CLIP=1` on `board_tryon.sh`.

There is a second, independent way to put a shopper into a product photo:
**face swap** (`board_faceswap_*`, GHOST on the NPU, Apache-2.0 weights;
`face_swap.py` is the older inswapper_128 version, which is faster to set up
and non-commercial). It goes the other way round from the overlay — instead of
moving a garment onto a live frame, it moves a face into a studio shot that
already has the garment, the lighting and the pose. Standalone and composable;
it does not touch the overlay path. See [docs/faceswap-npu.md](docs/faceswap-npu.md).

## Deployment targets

Two completely different paths. Check which device is attached before working:
`make board-info`, or `make doctor` (it names the device).

- **Arduino VENTUNO Q** — Ubuntu 24.04 aarch64 over adb. **Not Android.** Runs
  the `.pte` through Python with the same ExecuTorch wheels as the Mac. Use the
  `board-*` make targets. This is what's attached and working.
- **Android phone / XR** — the Compose app in `android/`, via the
  `executorch-android` AAR. Use `apk` / `deploy` / `logcat`. Builds, untested.

Keep the AAR version equal to the `executorch` pip version that exported the
`.pte` (both 1.5.1). Mismatches fail at load with an opaque error.

## Gotchas that have already bitten us

- **Quote preset names containing dots in `models.toml`.** A bare dot is a TOML
  table separator, so `[presets.llama3.2-1b]` silently becomes nested tables
  and the preset vanishes. Use `[presets."llama3.2-1b"]`.
- **`huggingface_hub.model_info()` succeeds on gated repos you can't download.**
  The only honest access test is fetching a real file; `check_access()` in
  `export_model.py` does this.
- **Instruct models ramble without their chat template.** `run_model.py`
  applies it by default; `--raw` opts out.
- **`optimum-cli` writes only the `.pte`**, not the tokenizer. `export_model.py`
  saves the tokenizer into the export dir afterwards, because every on-device
  runner needs it alongside the model.
- **Garment anchors and face landmarks use opposite left/right.** The
  garment JSON is in image coordinates (left = smaller x); YuNet's
  landmarks are the subject's left and right, mirrored on screen. Cross
  them and the shirt lands upside down over the face.
- **Detect faces on a downscaled frame.** On the board YuNet costs 70 ms at
  1280x720 and 11.5 ms at a third of that, for 2 px of box drift — it is
  otherwise three quarters of the frame budget, five times the NPU
  segmentation. `detect_frame(frame, scale=3)`; the default stays 1.
- **Build `cv2.FaceDetectorYN` once, not per frame.** Constructing one
  costs ~100x running it — 127 ms a frame became 9.2 ms. Use
  `face_detect.detect_frame()` in a loop; it caches per input size.
- **BiSeNet face parsing needs a face-filling square.** A whole 16:9 webcam
  frame comes back as a blocky rectangle with hallucinated hats.
  `make_face_raw.py` runs YuNet and crops to the face by default; the crop and
  landmarks go in `crop.json` so `face_matte.py` maps the result back.
- **The board's OpenCV wheel has no V4L2.** `cv2.VideoCapture` cannot open
  `/dev/video0` even though a raw ioctl opens it fine. Use `linuxpy` (pure
  Python, no system packages) or GStreamer, which is installed. Check nothing
  else holds the camera first — a stray `gst-launch-1.0` makes every open
  return EBUSY.
- **Keep the runner warm, and keep scratch on tmpfs.** Launching
  `qnn_executor_runner` per inference costs 468 ms for 12.5 ms of face-parse
  compute. `scripts/warm_runner.py` drives the patched binary's
  `--input_list_path -` mode: 26 ms. Both halves matter — the warm process
  gets it to 138 ms, and moving the 19.9 MB of tensor files from eMMC
  (130 MB/s) to /dev/shm (1.3 GB/s) does the rest.
- **`strings` is not installed on the board.** `strings | grep` on a binary
  returns a silent zero that looks like a failed build. Use `grep -a`.
- **The board has a compiler now, but not in /usr.** `~/toolchain`, installed
  without root by `scripts/board_toolchain.sh`. Source `~/toolchain/env.sh`
  before building anything there. Do not expect `gcc` on the default PATH,
  and see docs/board-toolchain.md for why `cc1`, `ld` and `libc.so` each
  need help when a toolchain lives outside /usr.
- **`apt download` is unreliable on the board.** ports.ubuntu.com gives it
  unroutable IPv6 and timing-out IPv4 mirrors. Resolve the closure on the
  board, fetch on the Mac, `adb push` the debs across.
- **The board reboots under heavy load.** It dropped off USB entirely
  mid-install and came back with 13 minutes of uptime while running the
  billboard, camera, web server and a large download. Make long board-side
  jobs idempotent and resumable.
- **The board's AOT QNN compiler insists it is running on x86.**
  `check_qnn_version.py` hardcodes `x86_64-linux-clang` and looks for
  `$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so`. Point that path at the
  aarch64 libraries with a symlink; see `scripts/board_whisper_export.sh`.
- **Exporting on the board needs `PYTHONPATH=/home/arduino/git`.** Otherwise
  the pip `executorch` wheel in `et-venv` shadows the bundle's tree, and the
  QCS8275 target vanishes.
- **AI Hub's Stable Diffusion v2.1 UNet is v-prediction, not epsilon.** The
  bundle's own `qaihub_stable_diffusion.py` forces
  `prediction_type = "epsilon"`; run it that way and every image is saturated
  RGB static, with no error anywhere. Nothing in `metadata.json` says which it
  is — the tell is that the output correlates ~0.99 with a pure-noise input at
  low timesteps and collapses at high ones.
- **That UNet's `timestep` input is a scalar, quantized over [0, 951].** Not a
  `[1,1280]` time embedding like the bundle's runner expects, so no `diffusers`
  download is needed for `UNet2DConditionModel.time_embedding` at all. But the
  standard schedule starts at t=999, which saturates uint16 silently. Cap the
  schedule at `floor(65535 * scale)`.
- **Read AI Hub graph encodings by tensor NAME, not index.** Release 0.62.2's
  UNet is `(timestep, latent, text_emb)`; the bundle's driver hardcodes
  `(latent, time_embedding, text_emb)`. `board_sd_export.py` writes an
  `encodings.json` keyed by name.
- **An AI Hub release built with a newer QAIRT than the board's 2.49 is
  rejected.** `metadata.json` names it. Release 0.62.2 is QAIRT 2.45 and works;
  0.64.0 is 2.50 and should not. The runtime's `Qnn API version 2.38.0` warning
  is the ExecuTorch shim's own number — unrelated, ignore it.
- **The prebuilt `qnn_executor_runner` is older than its own source, so use
  the one we built.** The source supports `--input_list_path -` (load once,
  one inference per stdin line, `FRAME_READY` on stdout); the Aug 13 binary
  has no such string. `scripts/board_build_runner.sh` rebuilds it with
  `~/toolchain` as `qnn_executor_runner_persistent` beside the original, and
  `warm_runner.py` picks it up. Never launch per inference: a Stable
  Diffusion step costs 0.30 s of NPU and 1.34 s of wall clock that way,
  0.31 s held open. Anything new that runs a `.pte` in a loop should go
  through `WarmRunner`, and `--persistent_runner off` / `WARM_RUNNER=0` are
  there to get the old behaviour back for a comparison.
- **`from_context_binary()` twice in one interpreter dies** in
  `qnn_preprocess.preprocess_multimethod` with a bogus `range()` TypeError,
  after `No QnnManager active ... in the current QnnManagerContext`. Run one
  stage per process.
- **The AI Hub SD 2.1 and ControlNet-Canny sets are different models, not
  variants.** ControlNet-Canny is SD **1.5**: epsilon-prediction (2.1 is v),
  768-wide text embeddings (2.1 is 1024), and its tokenizer pads with EOS
  (49407) where SD 2.x pads with `"!"` (0). Mixing a binary between the two
  fails in ways that look like quality problems rather than errors. Use
  `--base sd15` / `--prefix cn15` / `--prediction_type epsilon` together.
- **ControlNet conditions in pixel space, so it needs no VAE encoder.** Its
  `image_cond` is `[1,512,512,3]` quantized over [0, 1]. That is why image
  conditioning works today while img2img on the SD 2.1 set does not.
- **Canny edges include a garment's seams, and ControlNet treats them as
  colour boundaries.** Collar bands and sleeve hems become colour-block edges,
  so "a red t-shirt" over a navy tee reliably yields a *ringer* tee. Fix the
  edge map (thresholds, blur), not the sampler.
- **A release's `metadata.json` can be read without downloading the zip.** The
  AI Hub S3 bucket honours `Accept-Ranges`, so an `io.RawIOBase` whose
  `readinto` issues HTTP Range requests can be handed to `zipfile.ZipFile` —
  kilobytes instead of ~1 GB. The URLs are in each model's
  `release_assets.json` on Hugging Face.
- **`pgrep -f board_billboard_daemon.py` matches any shell that mentions the
  name**, including a `tail -f` or an editor's argv — and the web API's
  `start` short-circuits on it, returning `{"ok":true,"running":true}` while
  spawning nothing. Two daemons sharing the NPU scratch dir give
  `the NPU runner produced no output for faceparse_qnn_8a8w.pte`, which looks
  like a broken export and is not. Count daemons from a `ps -eo pid,args`
  dump filtered elsewhere, and restart through the page's stop/start API
  rather than pkill. The route now matches
  `[e]t-venv/bin/python .*board_billboard_daemon` — the `[e]` stops pgrep
  matching its own argv, the interpreter path stops it matching an editor.
  Both halves are needed. Corollary: **hand-start the daemon with that same
  interpreter** (`/home/arduino/et-venv/bin/python`) or the page cannot see
  it, and Start will spawn a second one.
- **ControlNet conditioned on a raw camera frame generates the room.** Canny
  takes every edge, so a ceiling light and picture frames become conditioning.
  Mask the edge map to the person's SegFormer silhouette first — it is the
  difference between unusable and usable on a real frame.
- **The billboard's `check_alignment` is tuned the opposite way from the
  generate mode.** It wants a big head to cut out (`FACE_MIN` 0.28); a
  generated garment needs a half-body shot. `board_tryon_gen.framing_guidance`
  is the gate for that mode; do not reuse the billboard's.
- **QNN/NPU ahead-of-time compilation does not work on macOS.** Less of an
  obstacle for the board, which is already Linux. See
  [docs/qualcomm-npu.md](docs/qualcomm-npu.md).
- **An adb device is not necessarily Android.** The VENTUNO Q answers adb but
  runs Ubuntu; `getprop` returns empty and `uname -a` says so. Building an APK
  for it is wasted work.
- **The board's published spec is wrong.** It's a QCS8275 with Cortex-A78C big
  cores and 14 GB RAM, not a quad-A53 with 4 GB. Don't size models down on the
  spec sheet's say-so.
- **On the board, torch pulls ~4 GB of unusable CUDA packages.** PyPI's
  `linux-aarch64` torch wheel is the sbsa/CUDA build. It works on CPU; it's
  just fat. `scripts/board_setup.sh` documents the CPU-only trim.
- **Ubuntu 24.04 has no pip and is PEP 668 externally-managed**, so the board
  uses `uv` rather than `python3 -m pip`.
- **Take the argmax last when enlarging a segmentation mask.** A 128x128 mask
  blown up to a 705 px crop staircases in 5-6 px treads, and smoothing the
  binary mask afterwards cannot recover what the argmax threw away. Reduce the
  class logits to one signed field, resize *that*, threshold at zero.
  `person_matte._to_frame`; it is the difference between a usable clip and a
  visible one.
- **Size a crop before you place it.** `body_crop` clamps the square to the
  frame first and positions it second. The other order pins the crop to the
  top-left corner as soon as the face is big enough to want a crop wider than
  the frame, and the subject hangs off the edge of what the model sees.
- **Outside the body crop, replicate downward only.** The torso continues below
  the crop, so cutting there slices the garment off square. Nothing continues
  sideways — replicating there smears the edge column across a third of the
  frame.
- **8a8w costs the thin classes, not the big ones.** The person silhouette
  quantizes to IoU 0.96-0.99 against fp32; the arm classes to 0.48-0.92. Pure
  convnets (BiSeNet) do not behave this way; transformers do, which is why
  whisper needed 16a8w and why person segmentation defaults to it — 24.9 ms
  against 11.2, which this board can afford because the CPU, not the NPU, sets
  the frame rate.
- **`torchvision` does install on the board after all** — one wheel, no CUDA
  packages, because `et-venv`'s torch is the `+cu130` build and already dragged
  them in. The warning above about gigabytes of wheels was written about a venv
  with no torch. Only the DeepLab baseline needs it.
- **A CSS edit can reach the board and still not reach the browser.** Next
  serves `/_next/static/chunks/*.css` as `max-age=31536000, immutable`, which
  is only safe if the filename changes whenever the content does. It usually
  does, but not always — one edit here changed the file's content while
  Turbopack kept the name `2n1o9ig_f0m06.css`, so an ordinary reload kept
  showing the old styles even though `curl` on the board returned the new
  ones. Confirm with the served bytes, not the source:
  `curl -s localhost:3000/_next/static/chunks/<name>.css | grep <new-class>`,
  and compare the chunk name before and after a build. A hard reload fixes
  the browser; nothing on the board needs restarting.
- **An overlay on a `contain`-fitted image is positioned against the wrong
  box.** The billboard stage is 4:5 (the composite's shape) and the live
  preview is a 4:3 camera frame, so `object-fit: contain` letterboxes it and
  percentages against the stage are 1.67x the picture's. The framing oval was
  correct on both sides — daemon `GUIDE` and CSS agreed — and still drawn too
  tall, because they were agreeing about different boxes. Put the overlay in a
  child that reproduces the fitted content (`.guide-box`, `height: 75cqw`).
- **A metric needs its reference measured the same way as its subject.**
  `head_match` compared a shopper's feathered, resampled, alpha-cut matte
  against the template's raw dilated head mask, and so read **0.87** when fed
  the model's own head — a 13% bias inviting exactly the wrong correction.
  `scripts/calibrate_template.py` records what that head measures after going
  through the compositor (`headSpanRef`), which is the only like-for-like
  reference. Run it after `prep_template.py` for every new template.
- **Pick the newest camera frame by mtime, never by name.** `multifilesink`
  writes `f%05d.jpg`, but the counter does not stop at five digits, so after
  99999 the names are variable width and a lexicographic sort ranks `f451354`
  above `f15177` — older, but later in the sort. `max-files` compounds it: it
  only reaps files the *current* run wrote, so restarting gst orphans the old
  ones and they win that sort forever. The symptom is a preview frozen on a
  frame from a previous session that survives restarting the server, the
  daemon and the camera, because nothing in that chain is actually wrong.
  `latestFrame()` sorts by mtime and `ensureCamera()` clears the directory
  before spawning gst.
- **An HF snapshot directory is symlinks into the blob cache.** `cp -RL` before
  `adb push`, or the board gets the links instead of the weights.
- **`adb push vendor $BOARD/vendor` nests as `vendor/vendor`** once the
  destination directory exists, which it does as soon as any other export has
  run. It works the first time and silently nests afterwards, and surfaces as
  a `ModuleNotFoundError` that looks like a `PYTHONPATH` problem. Push the
  contents: `adb push vendor/. $BOARD/vendor/`.
- **Restart the billboard daemon through the web API, never `pkill` + `nohup`.**
  Doing it by hand races the API's own lifecycle management and leaves **two
  daemons running**. Both poll the camera and both invoke
  `qnn_executor_runner`, and the Hexagon does not take two callers — it
  surfaces as `compose failed: the NPU runner produced no output for
  faceparse_qnn_8a8w.pte`, which looks like a broken export and is not. Use
  `curl -s -X POST http://127.0.0.1:3000/api/billboard -d '{"action":"stop"}'`
  then `{"action":"start"}`. And note `pgrep -f board_billboard_daemon.py`
  **matches its own shell command line**, so it reports a daemon that is only
  your own pgrep; filter a `ps -eo pid,args` dump instead.
- **Never benchmark the board's CPU with eager PyTorch.** Its torch is PyPI's
  `linux-aarch64` wheel — the sbsa/CUDA build, `2.14.1+cu130` — and it reports
  `CPU capability usage: DEFAULT`, i.e. scalar fallback kernels with no
  vectorised ARM dispatch. GHOST's generator is 4379 ms eager and **2064 ms
  through XNNPACK on the same cores**, so an eager baseline overstates an NPU
  win by ~2x. The tell is the Mac ratio: 33x for identical eager code, where
  A78C vs M-series should be 3-5x. `scripts/ghost_xnnpack_export.py` is the
  pattern for a fair CPU number on a plain `nn.Module`.
- **Quantization is not free on this board, and `quant_dtype=None` is a real
  option.** The GHOST face-swap generator is **4 % faster** at 16a8w than
  lowered as float — and visibly broken (47.5/255 against fp32, colour
  speckle). Unquantized, the compile spec sets `use_fp16=True` and the HTP
  runs it in fp16 at 82 ms and 1.78/255. **The QCS8275 supports HTP fp16**,
  despite QNN's docs saying only "premium tier SoC like Snapdragon 8 Gen 1 or
  newer" do. The `.pte` roughly doubles. Suspect anything with runtime
  normalisation statistics (`InstanceNorm`/`GroupNorm`/`LayerNorm`) or an
  image/embedding output; see [docs/faceswap-npu.md](docs/faceswap-npu.md)
  for what was ruled out first — a histogram observer and tighter calibration
  ranges both make it *worse*. **This does not apply to the AI Hub models.**
  Stable Diffusion and ControlNet arrive as context binaries already compiled
  at w8a16, and `from_context_binary()` only wraps a finished binary, so
  `use_fp16` has nothing to act on. Their precision is baked in; see
  docs/stable-diffusion-npu.md rather than trying the flag.
- **`make_quantizer` defaults to `per_channel_conv=True`**, which silently
  breaks `fp16a8w`: with float activations the weights need an explicit
  `Dequantize`, and this QAIRT's is per-tensor only. Per-tensor weights get
  past that, and then the HTP refuses to mix int8 weights with float
  activations at all. `fp16a8w` is simply not available here.
- **A multi-input model cannot be calibrated with noise on its side inputs.**
  GHOST's generator takes a 512-d identity vector alongside the image, and
  every AAD layer scales feature maps by it, so random values fit the
  quantization range to the range of noise. `board_faceswap_export.py` runs
  the fp32 ArcFace over the same calibration faces and feeds the real
  vectors.
- **GHOST's ArcFace is not InsightFace's, despite shipping its `iresnet.py`.**
  Worth knowing because InsightFace's pretrained weights are non-commercial
  and GHOST's Apache-2.0 licence is the only reason it was chosen over the
  better-looking `inswapper_128`. Checked numerically against
  `glint360k_cosface_r100`: identical keys, shapes and file length, but mean
  weight cosine 0.0008 — an independent training run, not a fine-tune. See
  [docs/faceswap-npu.md](docs/faceswap-npu.md).

## Conventions

- Never commit `.pte` / `.ptd` files — they're 100 MB+ and gitignored. Re-export.
- New models go in `models.toml` as a preset, not as hardcoded flags. NPU
  segmentation models are the exception: they live in `scripts/person_seg.py`'s
  `MODELS`, because the class bookkeeping travels with them.
- Calibration goes through the same preprocessing function as inference, and
  the candidates that lost a comparison stay in the registry so the comparison
  can be re-run.
- Exports land in `models/<preset-name>/`.
- Mac tok/s numbers are for relative comparison only. Don't quote them as
  device performance.
