# GHOST face swap on the Hexagon NPU

The question [docs/faceswap.md](faceswap.md) left open: can a face swap run on
the Hexagon rather than spending four seconds on the board's CPU, and can it
be done with weights a shop could actually deploy?

```bash
make board-faceswap-export                      # one-off, ~20 min, on the board
make board-faceswap IMAGE=frame.jpg             # onto the default template
make board-faceswap IMAGE=frame.jpg TARGET=assets/templates/model-navy-tee.png
make ghost-swap SOURCE=frame.jpg                # the same thing, fp32, on the Mac
```

**Both, and the answer was to stop quantizing.** GHOST exports clean, lowers
whole to the Hexagon and runs the swap in **82 ms**, against **2064 ms** for
the same model through XNNPACK on the board's own CPU — **25×** — with output
that matches the fp32 reference to **1.78 / 255**. End to end in the booth,
shutter to picture is **~120 ms**.

The route there is the interesting part. Quantized to 16a8w it ran at 79 ms
and was *visibly broken*: 47.5 / 255, colour speckle across the face. The fix
is not more bits or a cleverer calibration — it is **no quantization at all**.
Lowered as float, the HTP runs this graph in fp16 for **4 % more time than
int8** and 27× the accuracy. On this model the quantization was buying
essentially nothing and costing everything.

The identity encoder is a separate story and stays on the CPU in fp32; it is
not on the critical path. Details below.

## Measured, same board

Per inference, 16a8w, after warm-up. The CPU column is the same models eager
in PyTorch on the board's own Cortex-A78C cores; the Mac column is there only
for relative comparison, as CLAUDE.md asks.

The generator, every way it runs on this board:

| GHOST AEI-Net | Per inference | vs fp32 | `.pte` |
|---|---|---|---|
| **Hexagon NPU, float (fp16)** | **82.2 ms** | 1.78 / 255 | 357.7 MB |
| Hexagon NPU, 16a8w | 79.2 ms | 47.5 / 255 — broken | 181.5 MB |
| **Board CPU, XNNPACK fp32** | **2064 ms** | **0.0006 / 255** | 737.6 MB |
| Board CPU, eager PyTorch fp32 | 4379 ms | (the reference) | — |
| Mac, eager PyTorch fp32 | 134 ms | (the reference) | — |

NPU figures are averages over 25 consecutive inferences; XNNPACK over 6, at 8
threads. **The NPU is 25× the best CPU implementation**, not the 53× you get
by comparing against eager PyTorch — see the next section for why that
distinction matters.

**Use the float row.** The 16a8w generator is 4 % faster and 27× less accurate,
which is not a trade worth making for 3 ms.

### The eager-PyTorch number is not a CPU number

Comparing 82 ms against eager PyTorch's 4379 ms would have claimed 53×, and
that would have been measuring the board's torch build rather than its cores.
The tell was in the ratio: 4379 ms against 134 ms on the Mac for identical
eager code is **33×**, and a Cortex-A78C against an M-series should be 3–5× on
an fp32 convnet. Two more symptoms pointed the same way — 4 threads and 8
threads were indistinguishable (4394 vs 4379 ms), so it was not thread-bound;
and CLAUDE.md already records that the board's torch is PyPI's
`linux-aarch64` wheel, which is the sbsa/CUDA build. Asked directly, it says
so:

```
torch 2.14.1+cu130 ... CPU capability usage: DEFAULT
```

`DEFAULT` means scalar fallback kernels — no vectorised ARM dispatch. Lowered
to XNNPACK instead, the same model on the same cores is **2064 ms**, 2.1×
faster, and numerically exact (0.0006/255, identity cosine 0.8768 against
0.8768). That is the number the NPU should be measured against, and it is the
one quoted above.

Credit where due: this correction came from the Stable Diffusion session,
which spotted the implausible ratio in an earlier draft of this table.

### ArcFace, for completeness

| ArcFace iresnet100 | Per inference | Embedding cosine vs fp32 |
|---|---|---|
| NPU, 16a8w | 4.5 ms | 0.118 — a different person |
| NPU, 16a16w | 422 ms | 0.301 — still wrong |
| Board CPU, eager fp32 | 1514 ms | 1.0 |
| Mac, eager fp32 | 33 ms | 1.0 |

The NPU rows are timings, not recommendations: the quantized encoder is wrong
at every config tried, so that 4.5 ms buys nothing. It runs in fp32 on the
host instead, where it is exact — and it is off the critical path, so its cost
does not reach the shutter.

Four threads instead of eight changes almost nothing here (ArcFace 1372 ms,
generator 4394 ms), unlike `inswapper_128`, where four beat eight by a third.
The board's billboard daemon was running throughout, so the CPU column is
mildly pessimistic.

Both graphs lower as a **single `QnnBackend` delegate** — no partitioning, the
same clean result `scripts/pte_backend.py` reports for BiSeNet and SegFormer.

### The shutter: 120 ms, once the runner stays warm

The model reload used to dominate this completely. It no longer does, because
the Whisper session built a persistent-stdin `qnn_executor_runner` and
`scripts/warm_runner.py` to drive it. Measured on the board, same inputs, five
consecutive swaps:

| Generator path | Per shutter |
|---|---|
| One-shot runner (reloads the 357 MB `.pte` every time) | ~960 ms |
| **Persistent runner, `.pte` stays loaded** | **~120 ms** |

First call after `warm()` is 251 ms, then it settles at 119–121 ms. Of that,
82 ms is the NPU compute measured in isolation, so the remaining ~38 ms is the
tensor write, the read back and the paste — an **8x cut**, and close to the
floor for this model.

**The output is unchanged.** Warm and one-shot were run on the same frame and
compared: byte-identical, max absolute difference 0. Speed here costs nothing
in quality, which is worth stating because it would be easy to assume a
persistent process reuses stale state.

`GhostSwapper.resident` reports which path is in use. `WarmRunner` falls back
to one-shot launches when the patched binary is absent, so this degrades
rather than breaks.

**Keep the scratch files on tmpfs.** The runner exchanges tensors as files,
and the board's eMMC does 130 MB/s against tmpfs's 1.3 GB/s. On the face
parser that was the difference between 138 ms and 26 ms warm — most of the
win. The daemon passes `/dev/shm/billboard/work/ghost`, and `WarmRunner`
defaults to `/dev/shm` anyway, so this is only a trap if someone points the
workdir at the filesystem.

### What the whole command costs

`make board-faceswap IMAGE=frame.jpg` is a different measurement from the
shutter, and still ~2.5 s, because the CLI path pays costs a booth does not:

| Stage | Where | Time |
|---|---|---|
| Preprocess, detect, fp32 ArcFace | Mac | 1450 ms |
| Push tensors | adb | 40 ms |
| Swap | board NPU | 120 ms warm / 960 ms cold |
| Pull, paste back, write PNG | Mac | 220 ms |

The 1.45 s host step is cold Python — interpreter start plus loading the
261 MB ArcFace — which a resident daemon pays once. In the daemon the identity
is computed while the shopper lines up and the shutter is the 120 ms alone.


### Does it look like them?

A per-pixel difference is the wrong headline for a generator, so
`ghost_paste.py --identity` scores the swapped face against the source frame
with the fp32 ArcFace, and prints a control alongside it:

| | ArcFace cosine vs the shopper |
|---|---|
| fp32 GHOST swap | **0.877** |
| **NPU float (fp16) swap** | **0.870** |
| The unswapped product photo (control) | **−0.088** |

The swap moves the identity essentially all the way, and the NPU keeps it. The
control is what makes those numbers mean something: the model whose photo it
is scores −0.088 against the shopper, so 0.877 is not an artifact of both
being faces.



## Why GHOST and not inswapper_128

[docs/faceswap.md](faceswap.md) measured `inswapper_128` at **3.9 s** per swap
on the board's CPU and noted the problem: its weights are research /
non-commercial. For a hackathon slide that is a footnote. For a retail
billboard it is the whole question, so the point of this work was a swapper
whose weights can ship.

GHOST (`ai-forever/ghost`, the sber-swap paper) is Apache-2.0, PyTorch-native
and one-shot. Two corrections to the assumptions it was picked on:

- **It is not smaller than inswapper_128. It is bigger** — 184.3 M parameters
  against 138 M, a 703 MB fp32 checkpoint against 554 MB. The speed below does
  not come from a smaller model; it comes from the model being on the NPU.
- **GHOST 2.0 is a different thing** — head swap, seven networks and
  PyTorch3D, as [docs/faceswap.md](faceswap.md) already found. This is GHOST
  v1, which is one generator plus a recogniser.

## The licence, since it was the reason for the whole exercise

The repository is Apache-2.0 and both checkpoints are release assets of that
same repository, with no separate licence notice. The part worth checking was
the identity encoder: GHOST ships InsightFace's `iresnet.py`, and
InsightFace's *pretrained models* are stated to be "available for
non-commercial research purposes only". If GHOST's encoder were one of those,
the licence advantage would evaporate, because the generator was trained
against that specific embedding space and cannot be given a different one.

It is not. Measured against InsightFace's `glint360k_cosface_r100_fp16_0.1`,
the most widely redistributed r100:

| | |
|---|---|
| Same state-dict keys | 925 / 925 |
| Same shapes | all |
| Same file length | 261,223,796 bytes, both |
| **Mean cosine similarity of the weights** | **0.0008** |
| Max cosine over 104 weight tensors | 0.065 |
| Tensors above 0.5 | **0 of 104** |

Same *architecture*, unrelated *weights*. It is not that checkpoint and not a
fine-tune of it — two independently trained networks are what orthogonal
weights look like. InsightFace's architecture code is MIT, so the file GHOST
ships is fine on its own terms.

**What this does not establish** is that it is not some *other* r100 from
InsightFace's zoo that was never compared; only that it is not the one
everybody mirrors, and that the authors publish it as their own under
Apache-2.0. That is materially better footing than inswapper_128, which is
openly non-commercial. It is not a lawyer's sign-off, and this doc is not one.

## It exports with no shims at all

This is the sharpest contrast with the inswapper path. `inswapper_128` exists
only as ONNX, and getting it through `torch.export` needed 40 baked shims
across three op families ([scripts/onnx_bake.py](../scripts/onnx_bake.py)).
GHOST is PyTorch source, so:

```
arcface   torch.export OK, nodes 1286, max|export-eager| 0.00e+00
aei_net   torch.export OK, nodes  882, max|export-eager| 0.00e+00
```

Nothing to work around. The 21 `InstanceNorm2d` layers in the AAD blocks, the
7 `ConvTranspose2d` and the 9 `align_corners=True` bilinear upsamples — the
three things that looked most likely to break a QNN lowering — all went
through. Both graphs lower as a single `QnnBackend` delegate, no partitioning,
the same clean result `scripts/pte_backend.py` reports for BiSeNet.

Three changes were needed to vendor the model, all recorded in
[vendor/ghost/\_\_init\_\_.py](../vendor/ghost/__init__.py): an import, dropping
the unused `resnet` attribute-encoder backbone, and removing a
`torch.cuda.amp.autocast` wrapper that is a no-op on this board but would
still have been traced.

## How it fits together

1. **Detect** — [scripts/face_detect.py](../scripts/face_detect.py), YuNet,
   unchanged. Its five landmarks are *exactly* ArcFace's five, in ArcFace's
   order, so no InsightFace dependency is needed for alignment.
2. **Align** — [scripts/make_ghost_raw.py](../scripts/make_ghost_raw.py)
   solves the same 5-point similarity transform GHOST trains with
   (`estimate_norm(kps, 224, mode='None')`, which is the canonical template
   scaled by 224/112) and emits two tensors: the source face at 112 for
   ArcFace, the target face at 256 for the generator.
3. **Identity** — ArcFace turns the source face into a 512-d vector. **In
   fp32, on the host**, because the quantized version is wrong (below) and
   this step is not on the clock. `IDENTITY=npu` runs it on the Hexagon
   instead, for comparison. Note the vector is the **raw** ArcFace output, not
   an L2-normalised one; normalising it changes the scale the AAD layers were
   trained against.
4. **Swap** — `ghost_aei_qnn_*.pte` takes the target face and that vector and
   returns a 256×256 face in the target photo's own lighting. This is the part
   the NPU earns its keep on.
5. **Paste** — [scripts/ghost_paste.py](../scripts/ghost_paste.py) warps it
   home through the inverse transform and feathers it under an ellipse.

**Only step 4 is on the clock.** The identity vector describes the person, not
the instant the shutter fires, so a booth computes it while they walk up — the
same observation [docs/faceswap.md](faceswap.md) made about ArcFace there.
That is what makes step 3's fp32 CPU cost (1.4 s on the board) free in
practice, and it is why the split below is not a compromise.

### The ellipse, not the square — and it must stay inside the crop

Same lesson as the inswapper path: the generator repaints the whole aligned
box, hair and backdrop included, and its version of those is close but not
identical. Blending the square leaves a rectangular halo against a flat studio
background.

The fractions transfer across resolutions because the ArcFace template is
scaled by `size/112`, so the landmarks sit at the same relative positions at
any crop size. **The one rule is that the ellipse must stay inside the
generated crop.** GHOST renders only the aligned box, so a mask reaching its
edge blends in that box's own border — a hard straight line across the face,
much worse than whatever the widening was chasing.

I got this wrong. Chasing a bearded subject's jaw, I moved the ellipse to
centre 0.56 with a 0.49 vertical axis. That puts its bottom at **1.05 of the
crop** — 11 px past the edge at 224. On the frame it was tuned against, the
alignment rotated the overflow out of shot and it looked like an improvement;
on the next subject it drew an obvious diagonal seam across the jaw, and that
is what reached the live demo before it was caught there.

| Centre | Axis Y | Bottom | |
|---|---|---|---|
| 0.52 | 0.45 | 0.97 | the original, safe |
| **0.53** | **0.44** | **0.97** | current |
| 0.56 | 0.49 | **1.05** | **off the crop — the seam** |

Two guards now, in [ghost_swap.py](../scripts/ghost_swap.py): the fractions
are module constants with an `assert` that their sum stays under 0.98, and
`_border_falloff` drives the mask to zero before the crop edge regardless, so
a clipped warp cannot reintroduce a hard edge either.

**The beard is not reachable by widening the mask, and that was the real
mistake.** ArcFace's template puts the mouth at 0.82 of the crop, so the chin
sits at the very edge of what the generator ever draws and the beard below it
largely outside — there is nothing there to blend in. Compounding it, the
embedding is trained for *recognition*, which is deliberately somewhat
invariant to facial hair, so the 512-d vector barely encodes a beard and the
generator renders stubble rather than one.

For a heavily bearded shopper the answer is not this page at all — it is the
head cutout or `make board-cn-tryon`, both of which keep the real head. See
the comparison below.


## Quantization: what went wrong, and what it took to find out

This section is the investigation, kept because the conclusion —
*don't quantize this* — is not where anyone would start, and because
most of the obvious fixes are measurably wrong. If you only want the
answer, it is [below](#the-fix-stop-quantizing).

Face parsing was fine at 8a8w and person segmentation needed 16a8w for the
arms, so the plan was to measure rather than assume. Measured, at 16a8w —
against the fp32 reference, on byte-identical input (verified):

| | |
|---|---|
| Generator, mean absolute difference | **45.4 / 255** |
| Generator, p99 / max | 177.8 / 225.8 |
| ArcFace embedding, cosine vs fp32 | **0.118** |

Both are failures, and they are independent failures. The picture the
generator produces is recognisably a face with the right structure, shot
through with high-frequency colour speckle.

### The identity encoder does not survive quantization at all

| ArcFace iresnet100 | Embedding cosine vs fp32 | Embedding norm | Per inference |
|---|---|---|---|
| fp32 | 1.0 | 21.694 | — |
| **16a8w** | **0.118** | 4.250 | **4.5 ms** |
| **16a16w** | **0.301** | 11.035 | **422 ms** |

An embedding at cosine 0.118 is not a degraded identity, it is a different
person. Three things narrow down why:

- **It is not weight precision.** Simulating per-tensor 8-bit weights on the
  whole network, in PyTorch, gives cosine **0.9991**. The weights are fine.
- **It is not the activation range.** ArcFace's activations are tame — widest
  2.6, median 0.479, a 5× spread across 307 tensors.
- **The output magnitude matches the wrong stage.** The NPU's embedding has
  norm 4.250. The fp32 network's *pre*-`BatchNorm1d` output — the raw `fc`
  result, one op earlier — has norm **4.563**. The final `features`
  `BatchNorm1d` has `running_var ≈ 0.04`, so it scales by `1/sqrt(0.04) ≈ 5`,
  which is exactly the factor missing.

So the most likely mechanism is that the final `BatchNorm1d`'s variance
scaling is lost in lowering. That is a hypothesis, not a proof: the direction
is wrong too (cosine against the pre-BN output is only 0.079), so something
else is also off. Worth more digging by whoever picks this up.

**The practical answer is that it does not matter.** The identity encoder is
not on the critical path — it describes the person, not the moment, so a booth
computes it while they walk up. Run it in fp32 on the CPU, where it is exact,
and give the NPU the generator, which *is* on the clock. 16-bit weights make
it 94× slower anyway (422 ms against 4.5), so there is no version of this
worth having.

### The generator's failure is not explained by precision either

The generator's error is its own, not a knock-on from the bad embedding.
Feeding the NPU generator the *correct* fp32 embedding changes nothing:

| Generator input | Mean abs diff vs fp32 |
|---|---|
| NPU embedding (cosine 0.118) | 45.4 / 255 |
| **fp32 embedding (exact)** | **47.5 / 255** |

And simulating the quantization faithfully in PyTorch says it should be nearly
free. Each row quantizes only what it names, with ranges taken either from the
test input itself ("oracle") or from the same 40 calibration faces the real
export used:

| Simulated on the Mac | Mean abs diff vs fp32 |
|---|---|
| Weights 8-bit, per-tensor | 1.89 / 255 |
| Weights 8-bit, per-channel | 1.08 / 255 |
| Weights 16-bit, per-tensor | 0.01 / 255 |
| Activations 16-bit, oracle ranges | 0.20 / 255 |
| **Activations 16-bit, calibrated ranges** | **0.23 / 255** |
| Activations 8-bit, calibrated ranges | 20.95 / 255 |
| **Actually measured on the NPU at 16a8w** | **47.5 / 255** |

The real 16a8w result is **worse than simulated 8-bit activations** and about
200× worse than simulated 16-bit. Per-tensor weight and activation precision
does not explain it.

Three further things were tried and ruled out, each of which looked like the
answer at the time:

| Hypothesis | Test | Result |
|---|---|---|
| The calibration ranges are too wide (outliers to ±1031) | re-run with ranges clipped to the p99.99 / p99.9 / p99 of the calibration set | **Much worse** — 70.6 / 78.6 / 71.5 per 255 at 16-bit. The outliers carry signal; clipping them is catastrophic. A histogram or percentile observer would *hurt*. |
| The bundle's default `MovingAverageMinMaxObserver` under-reports the range | simulate its EMA (`averaging_constant=0.01`) over the 40 calibration faces | It does narrow ranges — to **78 % of the true max on average, 15 % at worst** — but costs only **0.60 / 255**. Not the cause. |
| **The 21 `InstanceNorm2d` layers' internal statistics** | quantize the per-channel mean, variance and `rsqrt` *inside* the op, which per-tensor simulation of its output cannot model | **16-bit internals: 0.09 / 255. 8-bit internals: 21.95 / 255.** |

Only the last lands in the right order of magnitude. Everything else is 200×
too small; 8-bit normalisation statistics are the one simulated mechanism that
produces errors like the ones actually measured.

That is still a hypothesis — it gets to ~22/255 and the board gives 47.5 — but
it is the only one left standing, and it makes physical sense. QNN has to
compute each `InstanceNorm`'s per-channel mean and variance internally and
carry them in fixed point, and a reciprocal square root of a small variance is
about the worst thing to put through one. The generator's activations make it
harder still: **±1031 against a median of 5.6**, a 185× spread, with max/std up
to 344 in the late AAD blocks. Speckle is what a per-pixel normalisation going
wrong looks like.

It also says what the fix has to be. Not more bits on the weights, not a
cleverer range — **no fixed-point activations at all.**

### The fix: stop quantizing

The diagnosis says the damage is done by *fixed-point activations*, so the
only configuration that removes the cause rather than widening it is one with
no fixed-point activations. Two attempts, and the second is the answer.

**`fp16a8w` — float activations, int8 weights — does not lower.** Twice, for
two different reasons. The bundle's `make_quantizer` defaults to
`per_channel_conv=True`, and with float activations the weights need an
explicit `Dequantize`, which this QAIRT only implements per-tensor:

```
[ERROR] <E> Op Dequantize does not support per-channel quant tensor
```

Forcing per-tensor weights (`PER_CHANNEL=false`, now a flag) gets past that
and into a second, harder wall — the HTP wants a *homogeneous* datatype set
and will not mix int8 weights with float activations:

```
[ERROR] <E> Supported I/O datatype sets for the configuration: INT8 / INT16 / FP16 / BF16
[ERROR] <E> Requested I/O datatype set:
[ERROR] <E> QnnBackend_validateOpConfig failed 3110
```

119 convolutions and linears rejected, plus 21 `instance_norm` validation
errors — **one per `InstanceNorm2d` layer**, which is its own small
confirmation of where the trouble lives.

**Lowering with no quantizer at all works.** `build_executorch_binary` with
`quant_dtype=None` hands QNN the float graph, and the compile spec then sets
`use_fp16=True` automatically. That is a homogeneous FP16 datatype set, which
the HTP accepts:

| Generator on the NPU | Per inference | vs fp32 | Identity vs shopper | `.pte` |
|---|---|---|---|---|
| 16a8w | 79.2 ms | 47.5 / 255 | — | 181.5 MB |
| **float (fp16)** | **82.2 ms** | **1.78 / 255** | **0.870** (fp32: 0.877) | 357.7 MB |

**fp16 costs 3 ms — 4 % — and is 27× more accurate.** p99 is 9.9/255 and the
worst pixel 23.4, against 184 and 229 for the quantized version. Visually the
speckle is simply gone.

The QNN docs warn that "not all SoC support QNN HTP fp16 — only premium tier
SoC like Snapdragon 8 Gen 1 or newer". The QCS8275 is an IoT part, so this was
not a given; it works, and it is worth knowing for every other model in this
repo. The `.pte` is about twice the size, which is the real cost.

### What this means for the other models here

The reflex on this board has been to quantize everything, because that is what
the examples do and because [faceparse](faceparse-npu.md) and
[personseg](personseg-npu.md) both quantize well. This model says the reflex
is not free. Quantization is worth it when the arithmetic is the bottleneck;
here the graph is memory-bound enough that int8 bought 4 %, and the accuracy
it cost was the entire product.

**Worth re-measuring unquantized:** anything with normalisation layers whose
statistics are computed at runtime — `InstanceNorm`, `GroupNorm`, `LayerNorm`
— and anything whose output is an image or an embedding rather than a class
label. The two segmentation models are probably fine as they are: an argmax
hides a lot, and they were measured against fp32 and found good.

**Not the AI Hub models, though.** Stable Diffusion and ControlNet arrive as
context binaries already compiled at w8a16, and `from_context_binary()` only
wraps a finished binary — `use_fp16` has nothing to act on, and the per-tensor
scale/offset readable out of them is proof the precision is baked in. Getting
fp16 there would mean exporting an 865 M-parameter UNet from PyTorch on the
board. [docs/stable-diffusion-npu.md](stable-diffusion-npu.md) records that so
nobody burns an afternoon on the flag.

### What it costs on disk

The float `.pte` is 357.7 MB against 181.5 MB quantized, and the board runs
at ~92 % full. Current footprint:

| | |
|---|---|
| `ghost_aei_qnn_float.pte` (the generator) | 358 MB — **required** |
| `ghost_arcface_xnnpack.pte` (the encoder) | 261 MB — **required** |
| `ghost_aei_xnnpack.pte` (the `BACKEND=cpu` fallback) | 738 MB — droppable |
| `/home/arduino/ghost/*.pth` (fp32 checkpoints) | 953 MB — only for re-export |

If the board runs short, the CPU fallback is the first thing to delete:
`make ghost-xnnpack-export` plus a push rebuilds it in under a minute. The
quantized and 16a16w artifacts were deleted once measured — they are in the
tables above and nothing needs the files.

## How it would slot into the billboard

Nothing here touches the existing pipeline — not `garment_overlay.py`, not
`board_billboard_daemon.py`, not `face_detect.py`. It is a parallel path that
starts from the same two things the daemon already has in hand.

The daemon's loop today is *detect → parse → matte → composite*, and it
already computes the first step. A swap replaces only the last three:

```
            detect (YuNet)  ──┬── parse → matte → composite   (today)
                              │
                              └── align → swap  → paste       (this)
```

Concretely, the pieces to reuse are:

- `make_ghost_raw.align()` + `landmarks_array()` take the detection the daemon
  already has — no second detect, and `detect_frame(frame, scale=3)` as
  CLAUDE.md advises.
- `ghost_paste.paste_back()` takes the generator's output back to frame
  coordinates.
- The **target side is entirely precomputable**. The product photo never
  changes, so its face box, landmarks, 256 crop and inverse transform are
  build-time constants, exactly as [docs/faceswap.md](faceswap.md) noted. At
  run time only the *source* changes.

Two things a daemon would want that this does not yet do:

1. **Keep the model loaded.** `qnn_executor_runner` is a one-shot CLI, so
   every swap pays the load of a 181 MB `.pte`. The same problem
   [docs/billboard.md](billboard.md) measures as 360 ms for a 12.7 MB model,
   and it will be much worse here. The runner's own source supports
   `--input_list_path -`, but the prebuilt binary predates it (CLAUDE.md) —
   the same blocker as Stable Diffusion's.
2. **Compute the identity once per person, not per frame.** It is the whole
   reason the split works. Recompute it when the detected face box jumps, not
   on every frame.

## Consent is in the shape of the interface

This is a retail try-on, so the source face is the person standing in front of
the board, and the entry point is deliberately **frame and target**:

```bash
./scripts/board_faceswap_run.sh frame.jpg [target.png]
```

There is no photo-picker and no upload path, because the difference between a
try-on and a deepfake tool is entirely whose face goes in. GHOST's own README
asks the same thing of anyone using it ("not for changing faces without
consent"). Anything built on top of this should keep that shape: the camera is
the only source of source faces, and nothing is retained after the swap.

Note also what the booth does *not* need to store: the identity vector is
computed, used and dropped. Nothing about the shopper has to persist past the
shutter.

## Does it beat the head cutout?

`assets/templates/model-navy-tee.png` as the target, an FFHQ portrait as the
shopper, both paths running on the board. The comparison image is in the
hand-over note; the short version:

**The head cutout wins on identity and loses on believability.** It moves the
shopper's actual pixels, so it is unarguably them — including their hair, and
including being bald if they are. But the head arrives with a seam at the jaw,
its own lighting, and a size that has to be guessed from landmarks, so it
reads as a head placed on a body. Everything
[docs/billboard.md](billboard.md) does — cutting at the jawline, scaling by
eye span, shifting the template's skin tone — is work to hide that.

**GHOST wins on believability and loses on identity.** It renders the face
*into* the photograph, so the lighting, the grain and the skin tone are the
photo's own and there is no seam to hide. It reads as one picture. But a face
swap keeps the **target's** hair, head shape and pose, so a bald shopper sees
themselves with the model's hair. It is their face on someone else's head.

Neither retires the other, and which is right depends on what the billboard is
selling. For a garment, GHOST is the better picture: the product is the shirt,
the pose is the model's, and "how do I look in this" is answered better by a
believable photo than by a literal one. For anything where the shopper has to
recognise *themselves* — a mirror, a photo they take away — the cutout is
honest and the swap is not.

The two also fail differently, which matters for an unattended booth. The
cutout's failure mode is a visible seam or a chunk of background read as hair
([docs/faceparse-npu.md](faceparse-npu.md) documents that one). GHOST's
failure mode is a plausible photograph of someone who is not quite the person
standing there — worse, because nothing about it looks broken.

### There is now a third option, and for some shoppers it is the right one

`make board-cn-tryon`, from the Stable Diffusion session
([docs/stable-diffusion-npu.md](stable-diffusion-npu.md)), generates the body
with ControlNet conditioned on the shopper's **own photo**, then composites
their real BiSeNet head onto it. Because the generation was conditioned on
that photo, the head lands at the right scale and pose with no warping.

That reaches the case this page cannot. A swap keeps the target photo's hair
and head shape, so a distinctive beard or hairstyle is **structurally**
unreachable here — not a tuning problem, and not something a wider blend mask
fixes. The ControlNet path never generates a face at all, so the beard
survives intact.

It is not strictly better, and the honest split is theirs:

| | Best when |
|---|---|
| **Head cutout** | the shopper must recognise themselves, and the product photo is fixed |
| **GHOST swap** (this page) | a seam would ruin it — the face integrates with the photo's own lighting |
| **ControlNet try-on** | hair or a beard has to survive, and pose should follow the shopper |

Tested here on a bearded subject: this swap returned his moustache and some
stubble and gave him the model's hair, which read as a different person. That
is the method working as designed, and it is the reason to keep all three.

## Verdict

- **It works, and it is fast.** 184 M parameters, whole-graph delegation, no
  shims, **82 ms a swap against 2064 ms through XNNPACK on the board's own
  CPU — 25×** — matching the fp32 reference to 1.78/255 and preserving
  identity (0.870 against fp32's 0.877). Shutter to picture is ~93 ms,
  against ~4.4 s for the CPU-only `inswapper_128` path in
  [docs/faceswap.md](faceswap.md).
- **There is a CPU fallback that ships without QNN at all.** XNNPACK fp32 at
  2064 ms is slow but numerically exact (0.0006/255), and
  `BACKEND=cpu make board-faceswap` runs it. That matters for a demo: the
  booth is not dead if the NPU path breaks.
- **The answer was to stop quantizing.** At 16a8w it was 4 % faster and
  visibly broken. Lowering the float graph lets the HTP run it in fp16, which
  this IoT-tier SoC turns out to support. The cost is a `.pte` about twice
  the size.
- **The identity encoder stays on the CPU in fp32.** It does not survive
  quantization at any config tried (cosine 0.118 at 16a8w, 0.301 at 16a16w),
  and it does not need to: it describes the person, not the moment, so a
  booth computes it while they walk up. This is the default; `IDENTITY=npu`
  opts back in for comparison.
- **The licence holds up better than feared** — the identity encoder is
  demonstrably not InsightFace's non-commercial r100 — **but it is not a
  clean bill of health**, and that was the entire reason for choosing GHOST
  over the better-looking `inswapper_128`. Worth a real legal read before a
  shop deploys it. This is now the only thing standing between this and a
  commercial deployment.
- **It is bigger than inswapper_128, not smaller** — 184 M against 138 M. The
  speed comes from the NPU, not from the model.
- **Whether it should replace the head-cutout composite is a product
  decision, not a technical one.** See the comparison above: the cutout is
  literally them and reads as pasted on; the swap reads as one photograph and
  gives them the model's hair. For selling a garment the swap is the better
  picture. For a mirror it is not. Both now run on this board, which is the
  useful outcome.

### What to try next

1. ~~**The `qnn_executor_runner` reload.**~~ **Done** — the Whisper session
   built the persistent-stdin runner and `scripts/warm_runner.py`. This
   swap's shutter went **960 ms to 120 ms**, output byte-identical. The
   numbers it was predicted to unlock, measured by the sessions that own
   them:

   | | Reloading | Predicted | **Measured** |
   |---|---|---|---|
   | This swap, per shutter | 960 ms | under 100 ms | **120 ms** |
   | Stable Diffusion 2.1, 20 steps | 28.3 s | ~8 s | **8.8 s** |
   | ControlNet, 20 steps | 48.0 s | ~21 s | **13.3 s** |

   ControlNet was expected to gain proportionally less — 2.7x rather than
   7x — because reloading was not its only overhead: ~0.65 s per step was its
   own CPU work requantising 13 mismatched residual tensors and writing
   ~27 MB of intermediate files. It beat the estimate (3.6x) because most of
   that 0.65 s turned out to be the *files*, not the arithmetic: moving the
   scratch to `/dev/shm` took 27 MB a step from 130 MB/s to 1.3 GB/s and left
   0.15 s of host-side work, of which 83 ms is real requantisation. See
   [persistent-runner.md](persistent-runner.md).
2. **Re-measure the other NPU models unquantized**, per the section above.
   Quantization bought 4 % here. It may be buying little elsewhere.
3. **Leave ArcFace alone.** It is exact on the CPU and free in wall-clock
   terms. Chasing its quantization is effort spent on the one stage that does
   not need it.
4. **Calibration is now mostly moot** for the generator, since the shipped
   path does not quantize it. The `--calib` plumbing stays for anyone
   re-testing a quantized config.
