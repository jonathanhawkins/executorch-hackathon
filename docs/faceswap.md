# Face swap instead of garment overlay — feasibility

**Question:** for the "step into the oval, line up, say take photo" flow, can we
drop the segment-and-composite pipeline and just swap the user's face onto a
finished product photo, on the board, fast enough?

**Short answer:** yes, with `inswapper_128`, **on the CPU, with no export work
at all**. Measured on the board the whole chain is **~4.1 s**, and the only
part that happens after the shutter is the 3.9 s swap. It also converts
cleanly to a `torch.export` program, so the Hexagon stays available later —
but for a posed one-shot photo it is not on the critical path.

## Why face swap, not head swap

A face swap replaces the inner face of a *fixed* target photo and keeps that
photo's hair, head shape, lighting and body. For a billboard that is the right
trade: the studio shot stays a studio shot, which is what makes it read as an
ad rather than a collage, and none of [docs/tryon.md](tryon.md)'s per-frame
placement problems exist, because the target is a still we control.

Head swap (user's hair too) is the better likeness but not a candidate here:
GHOST 2.0 — the only credible open one — is an Aligner plus a Blender plus
DECA, EMOCA, BlazeFace, stylematte and SegFormer-B5, i.e. seven networks and a
PyTorch3D dependency. HeSer has no official code. HS-Diffusion is a latent
diffusion model. None of these fit a photo-booth budget on this board.

## The candidates

| Model | Format | Res | Available | License |
|---|---|---|---|---|
| **inswapper_128** | ONNX, 554 MB fp32, 138M params | 128 | yes, HF mirrors | **non-commercial** |
| HyperSwap 1a/1b/1c | ONNX | 256 | yes (FaceFusion) | ResearchRAIL |
| SimSwap | PyTorch | 224 | yes | non-commercial, unmaintained since 2020 |
| MobileFaceSwap | — | 224 | **no code released** | — (0.50M params, 0.33 GFLOPs) |
| GHOST 2.0 | PyTorch, 7 nets | — | yes, Apache-2.0 code | weights unclear |

MobileFaceSwap is the one that would have been perfect — Baidu's AAAI'22
identity-aware dynamic network, 0.5M parameters against inswapper's 138M — and
there is no public implementation, only the paper.

The deepinsight HF repo for inswapper is disabled; mirrors
(`ezioruan/inswapper_128.onnx`) still serve it. Weights are research/
non-commercial either way — fine for a hackathon demo, say so on the slide.

## Measured on the board

`inswapper_128.onnx`, onnxruntime 1.31.0 (installs straight from PyPI for
aarch64, CPU EP only), random inputs, 10 runs after warm-up:

| threads | mean | min |
|---|---|---|
| 8 | 5224 ms | 4505 ms |
| **4** | **3917 ms** | 3664 ms |

Four threads beat eight — the same big/little split that shows up elsewhere.
Session load is under a second. For comparison the same graph is 352 ms eager
on the Mac, so the board CPU is ~11x slower; this is a 138M-parameter fp32
convnet, not a small model, and there is no NEON win to be had at fp32.

Four seconds is usable for a posed photo and nowhere near usable for preview.
Since the flow is step up → "take a photo" → see yourself, posed is all it has
to be.

ArcFace (`w600k_r50`, 174 MB, 112x112) on the same board:

| threads | mean |
|---|---|
| **4** | **175 ms** |
| 8 | 495 ms |

Four threads again, by a factor of three. 175 ms is noise next to the swap —
and it does not even have to be on the clock. The embedding depends only on
the person, not on the shutter, so it can be computed while they are lining up
in the oval. **Shutter to picture is the 3.9 s swap alone.**

## The NPU port is optional

At a ~5 s budget the CPU chain already fits, so none of the QNN work below is
required to ship the demo. It is worth knowing the door is open — a sub-second
swap would let the booth re-swap on the fly, or spend the saved time on a face
restorer — but it is an optimisation, not a dependency.

## It exports — the NPU path is open

`inswapper_128` is ONNX-only, with no PyTorch source, which normally rules out
the board's AOT route (torch.export → quantize → QNN partitioner, the one
whisper and BiSeNet went through). `onnx2torch` converts it, and after baking
three families of shim that read their shape arguments from tensors at runtime
— **14 Pad, 24 Slice, 2 Resize** — `torch.export` succeeds:

```
torch.export OK, nodes: 421
max |export - eager|: 0.0
```

The bake pass is [scripts/onnx_bake.py](../scripts/onnx_bake.py). Without it,
export dies on `unhashable type: non-nested SymInt` in onnx2torch's Pad, then
`.numpy() is not supported for tensor subclasses` in Slice, then a
data-dependent guard in Resize — all the same root cause.

What is still unverified: whether QNN quantizes a 138M-parameter generator
without wrecking it, and what it costs on the Hexagon. The 8a8w-vs-16a8w
lesson from person segmentation applies — a generator's output is a lot less
forgiving than a silhouette, so expect to need 16a8w.

## What a photo-booth pipeline needs beyond the swapper

1. **Face detection + 5 landmarks on the user** — already in the repo,
   `face_detect.detect_frame()`. ArcFace alignment *is* a 5-point similarity
   transform, so YuNet's landmarks are exactly the input needed.
2. **ArcFace recogniser** for the 512-d source embedding
   (`arcface_w600k_r50`, ~166 MB ONNX). Not yet on the board.
3. **The `emap` matrix** out of the inswapper ONNX initializers — the latent is
   `normalize(emap @ embedding)`, not the raw embedding. Getting this wrong
   produces a plausible face that is not the user's.
4. **The target side is free.** The billboard photo is fixed, so its face box,
   landmarks and inverse warp are computed once at build time, not per photo.
5. **Upscaling — now the binding constraint.** YuNet puts the face in
   `assets/templates/model-navy-tee.png` at **221x306 px** in a 1122x1402
   frame, and the swapper's aligned crop covers rather more than the box, so
   a 128x128 output lands back at roughly 2x upscale. Soft, but not a
   different league from the rest of the frame. If it reads badly, a GPEN-256
   restorer or pixel-boost is the fix, and at a 5 s budget with the swap at
   3.9 s there is room for one of them.

So the first cut is: embedding while they line up, ~4 s of swap on the
shutter, everything target-side precomputed at build time.

## It is built

[scripts/face_swap.py](../scripts/face_swap.py), `make faceswap SOURCE=me.jpg`.
Runs the same on the Mac and the board; the two outputs agree to within one
LSB.

Measured on the board, swapping the template's own face back onto itself:

| stage | board | Mac | on the clock? |
|---|---|---|---|
| detect source (YuNet) | 693 ms | 409 ms | no — they are still lining up |
| ArcFace session + embed | 720 ms | 922 ms | no — same |
| detect target | 162 ms | 13 ms | no — the billboard is fixed |
| **swap** | **4306 ms** | 922 ms | **yes** |
| paste back | 65 ms | 23 ms | yes |

So **~4.4 s from shutter to picture**, and everything else hides behind the
person walking up. The Mac is 4.7x faster on the swap and *slower* on ArcFace,
which is session construction, not compute.

Two things the build turned up that the measurements had not:

- **The emap is not reachable without the `onnx` package**, which is a heavy
  dependency and a pass over 554 MB to read one 512x512 matrix. `load_emap`
  caches it as a 1 MB `.npy` beside the model, so the booth needs only numpy.
- **Blend an ellipse, not the crop.** The swapper regenerates the whole 128
  box — hair, ear, backdrop — and its version is close but not identical, so
  pasting the square leaves a visible rectangular halo against a flat studio
  background. An ellipse also encodes the choice the billboard wants: their
  face, the model's hair.

A self-swap is the smoke test that costs nothing: feed the template as both
source and target and the result should be the photo you started with. It is
— mean absolute difference 0.28/255 over the frame, confined to a box around
the face — which clears alignment, landmark order, the emap latent and the
paste in one shot. What it cannot tell you is whether a *different* face at
128 px reads as that person. That still needs one real portrait.

## Wired in

`make web-push` ships it; pick **Face swap — your face on the model** from the
mode dropdown. The daemon gained a fourth `mode` beside `billboard`, `tryon`
and `segment`, and reuses the oval guidance, the voice shutter and the
live/shot phases unchanged.

Measured through the running daemon, with the camera fed a frame that passes
the alignment check:

| step | cost |
|---|---|
| first loop in the mode (load both models, warm both graphs) | 8.1 s, once |
| latent refresh while lining up | 232 ms, at most every 2 s |
| latent inside the TTL | 0 ms |
| **shutter → picture** | **5.2 s** |

5.2 s against 4.15 s for the same swap measured on its own: the daemon is
sharing the board with the camera, the Next.js server and whatever is polling
it. Worth knowing before promising a number on stage.

The state machine goes `live` → `working` → `shot`, and the UI shows
"Making your billboard…" over the frozen preview for the whole 5 s rather
than looking hung. `Retake` returns it to `live`.

Three things the integration needed that were not obvious:

- **ArcFace caches beside the swapper**, not at a fixed path. It was resolving
  relative to the script, so the board re-downloaded 289 MB of buffalo_l on
  first use instead of finding the copy already staged next to the model.
- **Announce the load before doing it.** The first loop in the mode spends 8 s
  with the preview frozen; a state written afterwards arrives too late to say
  anything useful.
- **`pkill -f board_billboard_daemon` kills the shell running it**, because
  that pattern matches its own command line. `pkill -f "[b]oard_..."`, the
  same trick `start_camera` already uses for gst.

## What the integration had to deal with

Most of the interaction the booth needs already existed in
[scripts/board_billboard_daemon.py](../scripts/board_billboard_daemon.py):
`check_alignment` draws the oval guidance, `voice_triggered` watches the
Whisper transcript for "take a photo", `phase` already flips `live` → `shot`
and holds the result, and the UI already has a mode dropdown. A face-swap mode
is a fourth `mode` beside `billboard`, `tryon` and `segment`, not a new daemon.

Five things that are genuinely different, though:

1. **Sessions must live in the daemon, not the call.** `face_swap.py` builds
   its onnxruntime sessions lazily; at 0.9 s for the swapper that belongs at
   startup. Resident cost is 554 MB + 174 MB of fp32 weights on top of
   whatever the NPU contexts hold — fine against 14 GB, but it is not free.
2. **The embedding has to move off the shutter**, or the 4 s becomes 5 s. It
   should run during the `live` phase — but only while `aligned` is true, and
   not every frame: the live branch today deliberately skips the parse to stay
   responsive, and 175 ms a frame would undo that.
3. **The loop blocks for ~4.3 s.** Today's shutter stalls for a few hundred
   milliseconds and nobody notices. Four seconds of a frozen preview needs a
   `working` phase in `state.json` and something in the UI that admits it is
   thinking.
4. **Use `model-navy-tee.png`, not `model-navy-tee.template.png`.** The
   template is the headless, inpainted one `prep_template.py` produced for the
   compositing path. Face swap needs the original with the model's head still
   on it — it replaces the face, it does not fill a hole.
5. **Every product needs its own photo of a person wearing it.** The garment
   overlay takes any PNG with shoulder anchors; this takes a studio shot with
   a face in it. That is a content constraint on how many products the booth
   can offer, and it is worth deciding before building the UI around it.

And the standing trade: the shopper gets the model's hair and head shape with
their own face. On a billboard that reads as an advertisement, which is the
point — but it is not a mirror, and someone whose hair is nothing like the
model's will notice.
