# Person segmentation on the Hexagon NPU

**Working.** A human-parsing SegFormer, quantized 16a8w and lowered wholesale
to the QCS8275's Hexagon v75, segments the person in a frame in **24.9 ms**:

```bash
make board-personseg-export                     # one-off, ~8 min, on the board
make board-personseg IMAGE=frame.jpg            # the silhouette
CLIP=1 make board-tryon IMAGE=frame.jpg         # use it: clip the garment
OCCLUDE=1 CLIP=1 make board-tryon IMAGE=frame.jpg
```

This is the second NPU model in the try-on path and it does two things the
face parser cannot: it **clips the garment to the wearer's outline**, so a
shirt stops hanging in the air beside a shoulder when the placement is a
little off, and it **puts arms and hands back in front of the garment**, so a
forearm across the chest does not vanish under it.

`CLIP=1` is opt-in. `make board-tryon` and `make board-faceparse` do exactly
what they did before.

## Which model, and why

Three were exported to the board and measured. `scripts/person_seg.py` keeps
all three, so this table can be re-run rather than taken on trust.

| | Params | `.pte` | Input | **NPU** | Arms? |
|---|---|---|---|---|---|
| **SegFormer-B0, ATR, 16a8w** | **3.7 M** | **4.4 MB** | 512×512 | **24.9 ms** | **yes** |
| SegFormer-B0, ATR, 8a8w | 3.7 M | 4.3 MB | 512×512 | 11.2 ms | unreliably |
| DeepLabV3-ResNet101, VOC, 8a8w | 61 M | 57 MB | 224×224 | 8.4 ms | no |
| SegFormer-B2, ATR | 27 M | — | 512×512 | not exported | yes |

The bundle's own `examples/qualcomm/scripts/deeplab_v3.py` model is the
baseline, and it is a perfectly good *silhouette* — against SegFormer on the
same frame the two person masks agree at **IoU 0.874**, and it is genuinely
faster, because 224×224 is a fifth of the pixels.

It lost on two counts.

**It has no body parts.** VOC's `person` is one blob: an arm in front of a
torso carries the same label as the torso, so no amount of post-processing
separates them. Arm occlusion is half of what this work is for, and the
baseline cannot do it at any speed. ATR labels `Left-arm` and `Right-arm`
separately, and `Upper-clothes` as well — which turns out to settle the seam
argument with the head matte too (below).

**Its mask is much coarser than its output suggests.** DeepLabV3 emits logits
at the input size, but torchvision's ResNet101 variant runs at output stride 8,
so a 224×224 output is a 28×28 feature map bilinearly enlarged. Measured by
round-tripping the logits through a smaller grid and back:

| Round-trip through | DeepLabV3 (224² logits) | SegFormer-B0 (128² logits) |
|---|---|---|
| half resolution | IoU 0.9963 | IoU 0.9741 |
| quarter | IoU 0.9950 | IoU 0.9550 |
| eighth (28² / 16²) | **IoU 0.9820** | IoU 0.9298 |

DeepLabV3 loses almost nothing going down to 28×28 — there is nothing there
below it. SegFormer loses measurably at every step, so its 128×128 is real.
Over the same ~705 px body crop that is **25 frame pixels per genuine mask
pixel against 5.5**.

SegFormer-B2 was left unexported: B0 already quantizes to within a few percent
of its own fp32 on the silhouette, and 7× the parameters buys accuracy the
clip cannot use.

**PPHumanSeg was rejected without exporting it.** `opencv/human_segmentation_pphumanseg`
ships as ONNX only, and the QNN ahead-of-time path takes a `torch.nn.Module`.
Converting ONNX back to PyTorch to re-lower it is a detour with its own failure
modes, for a model that would still only give a silhouette.

## Measured, same board

| | Per inference |
|---|---|
| **NPU** (16a8w, QNN) | **24.9 ms** |
| NPU (8a8w, QNN) | 11.2 ms |
| CPU (eager PyTorch, fp32, 8 threads) | 1668 ms |

**67× faster** at 16a8w and 150× at 8a8w, over 100 warm iterations at 512×512
against 5 CPU iterations.
`scripts/pte_backend.py` reports `QnnBackend ×1`: the whole graph lowered as a
single delegate, attention and all.

That it lowers at all is the surprise. Everything on the Hexagon in this
project so far has been convolutional; this is a transformer, and it went
through `build_executorch_binary` with no partitioning and no model surgery
beyond unwrapping the `ModelOutput` dataclass that `torch.export` will not
trace.

## What quantization costs — and why the default is 16a8w

Against the fp32 model on identical input, on three **held-out** frames (the
calibration set is rows 0–47 of the same dataset; these are rows 3000+):

| | class agreement | silhouette IoU | **arms IoU** | `Upper-clothes` IoU |
|---|---|---|---|---|
| **8a8w** frame A | 96.6 % | 0.958 | **0.476** | 0.927 |
| **8a8w** frame B | 96.6 % | 0.989 | **0.924** | 0.889 |
| **8a8w** frame C | 94.7 % | 0.968 | **0.678** | 0.739 |
| **16a8w** frame A | 98.2 % | 0.972 | **0.944** | 0.965 |
| **16a8w** frame B | 98.1 % | 0.996 | **0.969** | 0.931 |
| **16a8w** frame C | 96.5 % | 0.977 | **0.918** | 0.829 |

**At 8a8w the silhouette survives and the arms do not, reliably.** 0.96–0.99
against fp32 is a clip boundary you can use; 0.48–0.92 for the arms is not the
same thing. On frame A the 8-bit model found half the arm the fp32 model found,
and the arm there is thin and partly behind a held hat. The pattern is that
small, thin classes lose most.

This is the opposite of what the face parser found — BiSeNet at 8a8w agreed
with its fp32 self on 99.6 % of pixels — and the difference is the model class.
BiSeNet is pure convolution, which is what 8-bit is good at. This is a
transformer, like whisper, which needed 16a8w for the same reason.

**16a8w fixes it**, taking the arms to 0.92–0.97 and improving everything else
as well, for **24.9 ms against 11.2**. That is the default, and the trade is
the right way round on this board: the CPU is the scarce resource here and the
Hexagon was idle two thirds of every frame, so 13.7 ms of idle accelerator buys
an arm that is occluded every time rather than sometimes. `QUANT=8a8w` on
either script gets the faster build back.

## Resolution: is a 128×128 mask good enough at 720p?

Over a 705 px body crop that is **5.5 frame pixels per mask pixel**. Taken
naively it is not good enough — and the fix is free.

The trap is doing the argmax first. Take the argmax at 128×128 and enlarge the
resulting binary mask and the boundary is quantized to a 5.5 px grid: along one
shoulder, the staircase treads average **13.8 px and run up to 44**. Smoothing
that binary mask afterwards does not help, because the information is already
gone.

So the argmax is done **last**. The class logits are reduced to one signed
field per mask — "person minus background", "arm minus everything else" — that
field is enlarged bilinearly, and the threshold is applied at zero on the
enlarged field. The boundary then lands wherever the two scores cross, at
sub-mask-pixel precision. Same shoulder: treads of **2.8 px, longest 16**.

| Enlarging 128² → 705² | mean tread | longest |
|---|---|---|
| argmax, then nearest | 13.8 px | 44 px |
| argmax, then bilinear + threshold | 3.3 px | 40 px |
| **signed field, then threshold** | **2.8 px** | **16 px** |

It costs one resize of a single-channel float image. `person_matte._to_frame`.

With that, the answer is **yes, good enough for clipping at billboard scale**.
What it is not good enough for is a *matte* — individual hairs, the gap between
two fingers. That is why the head still comes from BiSeNet at 512×512 over the
face box, and why the body mask is never asked about the hairline.

## How the pieces fit

1. **Detect** — [scripts/face_detect.py](../scripts/face_detect.py), unchanged.
   YuNet's box is what sizes the body crop.
2. **Crop and preprocess** — [scripts/make_person_raw.py](../scripts/make_person_raw.py)
   takes a square **5 face-widths across**, starting 0.8 face-heights above the
   box, and emits the `(1, 3, 512, 512)` float32 NCHW tensor. The crop and the
   face box go into `person_crop.json` beside it.
3. **Run** — the bundle's prebuilt generic `qnn_executor_runner`, same as the
   face parser.
4. **Decode** — [scripts/person_matte.py](../scripts/person_matte.py) turns the
   `(18, 128, 128)` logits into frame-sized alphas: silhouette, arms, and the
   subject's existing `Upper-clothes`.
5. **Clip and occlude** — [scripts/garment_clip.py](../scripts/garment_clip.py)
   multiplies the garment's alpha by the silhouette and builds the layer that
   goes back on top.

Placement is untouched by all of this.
[scripts/garment_overlay.py](../scripts/garment_overlay.py) still decides
*where* the shirt goes; this decides *which of its pixels survive*. The two
fail independently, which is the point: placement that is a few degrees off
still reads as worn once the garment is cut to the real outline.

### Why the crop is 5 face-widths

Measured on a billboard-framed test frame: at 3.5 face-widths the arms
disappear from the output entirely — they are cut off at the crop edge, so the
model has nothing to label — and at 5.0 they come back. Seven is no better
than five and on a 16:9 frame it just clamps to the frame height.

The crop is **sized before it is placed**. Doing it the other way round means a
face big enough to want a crop wider than the frame gets the square pinned to
the top-left corner, leaving the subject hanging off the right-hand edge of the
region the model sees. That bug was visible as a mask smeared across a third of
the frame before it was found.

Outside the crop the model said nothing, and only one direction gets a guess:
**the bottom edge is replicated downwards**, because the body genuinely
continues below the crop and cutting it off there draws a ruler-straight line
across the garment halfway down the chest. Left, right and above are filled
with background — a crop sized in face widths is wider than the garment, so
there is nothing out there to keep, and replicating sideways smears the edge
column across the frame.

## Where the body mask and the head matte disagree

They overlap across the neck and jaw, and the precedence is:

**The head matte wins, everywhere except one strip.** It is far the finer
measurement: BiSeNet runs 512×512 over a crop 1.6× the face box — about 2.3 of
its pixels per frame pixel on a 141 px face — against the body mask's 5.5 frame
pixels per mask pixel. Along a boundary that is roughly **13× the resolution**,
at the hairline, which is exactly where an alpha edge gets judged. The body
mask has no opinion worth hearing about individual hairs.

**The garment wins inside `Upper-clothes`.** BiSeNet's head classes run down to
wherever the subject's existing shirt starts — `neck` is a head class — so
drawing that matte over the garment punches a hole through the new collar and
shows the old one through it. But that strip is precisely what the demo is
replacing, and the body model is the only thing that knows where it is. So the
head matte's alpha is suppressed wherever the body model says `Upper-clothes`,
and the new collar survives.

The second rule is the one that needed this model. Without it the two mattes
argue over the neck and the old collar wins. It is `hold_collar` in
`garment_clip.clip_and_occlude`, on by default, and it does nothing for a model
with no `Upper-clothes` class.

## The new frame budget

Measured on the board, 1280×720 frame, 705 px body crop:

| Step | Where | Cost |
|---|---|---|
| Body crop + preprocess to 512² | CPU | 9.0 ms |
| **Person segmentation, 16a8w** | **Hexagon NPU** | **24.9 ms** |
| Decode 3 masks to frame alphas | CPU | 12.5 ms |
| Clip + build occluder | CPU | 8.6 ms |
| Composite (×2 at 1280×720) | CPU | 2.9 ms |

About **56 ms added** — one of those composites was already counted in the
try-on budget's "warp + composite". On top of [docs/tryon.md](tryon.md)'s
~37 ms that is **~93 ms a frame, about 11 fps**, down from 27. At `QUANT=8a8w`
it is ~80 ms and 12 fps, which is why the slower build is affordable: the NPU
is not what sets the frame rate.

That is an honest cost and it is worth saying plainly: **the clip is not free,
and most of the cost is on the CPU.** Of those 93 ms the Hexagon accounts for
37 and the CPU for 56, on a chip whose CPU was already the bottleneck. Three
things would get most of it back, in order of how much they buy:

- **Decode only inside the garment's bounding box.** The clip can only remove
  garment pixels, and the occluder only matters where there is garment under
  it. `garment_clip` already works this way — it is why the clip step is 8.6 ms
  rather than 19 — but `person_matte` still enlarges each mask over the whole
  705 px crop. Passing the box down should take the 12.5 ms to roughly a fifth.
- **Fold the argmax, or the whole decode, into the graph.** The runner hands
  back 18 channels of float32 — 1.2 MB a frame — and the host immediately
  reduces them.
- **Overlap the two NPU models with the CPU work.** Face parsing and person
  segmentation are independent given the frame, and nothing today runs them
  concurrently.

None of these were done, because the thing to check first is whether 12 fps
reads badly on a billboard, and that needs the camera.

## What had to be worked around

The same four as whisper and the face parser, all in
[docs/whisper-npu.md](whisper-npu.md): `PYTHONPATH=/home/arduino/git` so
`executorch` resolves to the bundle's QCS8275-capable tree rather than the pip
wheel, the `QNN_SDK_ROOT` shim that claims to be x86, calibration supplied by
us rather than a dataset loader, and artifacts saved explicitly. Plus:

- **The bundle's `deeplab_v3.py` wants a ~2 GB VOC download** through
  torchvision's dataset loader. Calibration here is **48 people from the ATR
  human-parsing dataset**, fetched through the HF datasets-server rows API,
  preprocessed on the Mac through the same `prepare()` that inference uses, and
  pushed as raw tensors. Each one goes through YuNet and the same body crop, so
  the calibration set is billboard-framed rather than full-body — which is the
  distribution that matters, since calibration should match inference, not
  training.
- **`torchvision` does install on the board**, which contradicts the note in
  CLAUDE.md about it dragging gigabytes of CUDA wheels. That note was written
  about a venv without torch; `et-venv`'s torch is the `+cu130` build, so the
  CUDA packages are already there and torchvision resolves to a single wheel.
  It is installed only for the DeepLab baseline, by
  `board_personseg_export.sh`, and nothing in the chosen path needs it.
- **A HF snapshot directory is symlinks into the blob cache**, so it is
  dereferenced with `cp -RL` before `adb push` or the board receives the links.
- **`HF_HUB_OFFLINE=1` on the board**, so `from_pretrained` reads the pushed
  directory instead of trying the Hub mid-export.

## What does not work yet

- **Arm occlusion is good, not certain.** 16a8w gets the arm classes to
  0.92–0.97 IoU against fp32, but that is agreement with fp32, not with the
  truth — the underlying model still misses an arm against a same-coloured
  background, and nothing here measures that.
- **Fixed 512×512.** Compiled in, like every QNN graph.
- **The whole-frame fallback is coarse, not broken.** With no face found,
  `--full-frame` squashes the 16:9 frame into the square input. Measured on the
  same test frame it finds the same seven classes and puts the silhouette at
  24.7 % of the frame against the body crop's 23.7 %, so it degrades gracefully
  — unlike BiSeNet, which invents hats when given a wide frame. What it loses
  is resolution: 10.0 frame pixels per mask pixel against 5.5.
- **The silhouette includes held objects.** `Bag` and `Scarf` are in it
  deliberately — a tote against the body is part of the outline — but so is a
  hat held out at arm's length, and the garment gets clipped around it. On the
  test frame that happens to look right; it will not always.
- **Nothing is clipped outside the body crop.** 4 % of the garment fell outside
  it on one test frame, and those pixels are clipped against a replicated edge
  rather than a measurement. `garment_overlay.py` prints the fraction when it
  is over 2 %.
- **The clip is hard.** A loose tee is cut exactly to the body, which reads a
  little shrink-wrapped. The silhouette is grown 2 px before clipping to err
  the safe way, and `--clip-grow` pushes it further, but a garment that should
  drape past the body is not something a silhouette clip can express.
- **Still no real camera frame.** Test frames here are held-out ATR photos
  cropped to 16:9 head-and-torso — real people on textured backgrounds, which
  is a step up from the synthetic frames the rest of this work used, but not
  the board's own camera. The camera notes in
  [faceparse-npu.md](faceparse-npu.md) still apply.
- **The shoulder estimate breaks on close framing, and the clip makes it
  obvious.** On three held-out frames with a face box over ~250 px wide, the
  `0.45 × face height` drop put the shoulder line below the bottom of a 720-row
  frame and almost the whole garment fell off the picture. That is placement,
  not clipping — see [tryon.md](tryon.md) — but it is what a clipped garment
  shows up.

## Sources

- Model: [mattmdjaga/segformer_b0_clothes](https://huggingface.co/mattmdjaga/segformer_b0_clothes), SegFormer-B0 fine-tuned on ATR
- Baseline: the bundle's `examples/qualcomm/scripts/deeplab_v3.py`, torchvision `deeplabv3_resnet101` COCO/VOC weights
- Calibration: [mattmdjaga/human_parsing_dataset](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset)
- Detector: YuNet, [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet)
- Board runs, 2026-10-17 — builds on [qualcomm-npu.md](qualcomm-npu.md), [faceparse-npu.md](faceparse-npu.md) and [tryon.md](tryon.md)
