# Face parsing on the Hexagon NPU

**Working.** BiSeNet face parsing, quantized 8a8w and lowered wholesale to the
QCS8275's Hexagon v75, produces a head/face/neck cutout in **12.3 ms**:

```bash
make board-faceparse-export                   # one-off, ~10 min, on the board
make board-faceparse IMAGE=frame.jpg          # finds the face itself
make board-faceparse IMAGE=frame.jpg BBOX=480,120,320,320   # or tell it where
```

This is the matte for the virtual-billboard demo: everything except background
and `cloth`, so the shirt is already excluded at the neckline.

## Measured, same board

| | Per inference | Model |
|---|---|---|
| **NPU** (8a8w, QNN) | **12.3 ms** | 12.7 MB |
| CPU (eager PyTorch, fp32, 8 threads) | 978 ms | 53 MB |

**80× faster**, over 100 warm iterations at 512×512. That is ~81 fps for the
segmentation step alone, so on this board the matte is not the bottleneck in a
live loop — the capture and composite around it will be.

## Quantization costs almost nothing here

Against the fp32 model run on the identical input:

| | |
|---|---|
| Per-pixel class agreement | **99.60 %** |
| Head-mask IoU, fp32 vs NPU | **0.9990** |
| Disagreeing pixels | 242 of 262,144 |

The two cutouts are visually indistinguishable. 8a8w was the right call — this
is a pure-convolutional network, which is exactly what 8-bit quantization is
good at, and there was no need for whisper's 16a8w.

`scripts/pte_backend.py` reports `QnnBackend ×1`: the entire graph lowered as a
single delegate, no partitioning.

## How it fits together

1. **Detect** — [scripts/face_detect.py](../scripts/face_detect.py) runs
   YuNet (OpenCV's own detector, 233 kB, a few ms on CPU) and returns the
   largest face's box plus five landmarks. Largest face wins: at a billboard
   that is the person standing in front of it.
2. **Preprocess** — [scripts/make_face_raw.py](../scripts/make_face_raw.py)
   crops a square around that box at 1.6×, so hair and neck survive, and emits
   the `(1, 3, 512, 512)` float32 NCHW tensor the runner wants.
3. **Run** — the bundle's prebuilt generic `qnn_executor_runner`. No
   model-specific runner needed, unlike whisper.
4. **Matte** — [scripts/face_matte.py](../scripts/face_matte.py) argmaxes the
   `(1, 19, 512, 512)` logits, unions the head classes, keeps the connected
   region under the nose landmark, erodes a pixel and feathers the alpha.

The crop and the landmarks go into `crop.json` beside the tensor, so the matte
lands back on exactly the pixels the model saw, and whatever composites next
gets the eye positions for scale and rotation without detecting again.

The model itself is vendored at [vendor/face_parsing](../vendor/face_parsing)
from `zllrunning/face-parsing.PyTorch` (MIT), because BiSeNet is not one of the
bundle's examples. Two edits, both noted in that package's `__init__.py`: a
relative import, and dropping an unused `import torchvision` that would
otherwise have dragged gigabytes of CUDA wheels onto the board.

### Classes

CelebAMask-HQ's 19. The cutout is everything except `background` and `cloth`:
skin, hair, both ears, neck, brows, eyes, nose, lips, and the things worn on a
head — glasses, earrings, a hat. Excluding `cloth` is the whole point; it is
the garment being replaced.

## Why the detector is not optional

BiSeNet was trained on CelebAMask-HQ, where every image is square and the face
fills it. Measured on the same face at three framings:

| Input | Classes found | Result |
|---|---|---|
| 512×512 face-filling | all 14 expected | Clean matte, hair edges intact |
| 1280×720 whole frame, squashed | hallucinated `hat`, `eye_g`; no neck, no brows | Unusable |
| 1280×720, detector-cropped | all 14 expected | Correct parse |

The middle row is what you get without a detector, and it is not a near miss —
the model invents classes. With YuNet in front, the wide frame produces exactly
the same class set as the ideal 512×512 input.

Detection costs a few milliseconds on CPU against 12.3 ms on the NPU, so the
detector roughly rounds to free.

## Resolved: flat backgrounds reading as hair was a test artifact

An earlier version of this document reported that BiSeNet labels flat
background beside the head as `hair`, producing a grey slab attached to the
cutout, and listed it as unsolved. **That was an artifact of the synthetic
test frame**, which was a face pasted onto a flat grey field.

Tested on a real camera frame from the board's own Brio — venue lighting, a
textured background, ceiling strip lights and two other people in shot — the
matte is clean. All 14 expected classes, no slab, hair edges intact, and the
background people and lights all correctly excluded. Nothing needed changing.

Worth keeping as a lesson: a synthetic test frame with a flat field is not a
conservative stand-in for a real scene. It was *harder* than reality in a way
that invented a problem.

## Resolution matters more than the crop

Matte quality tracks how many *real* pixels the face spans, not how well it is
cropped. A face spanning ~320 pixels upsampled to 512 is softer at the hairline
than a native 512 one. For the billboard, frame the subject so the face spans
400+ pixels — well within the Brio's range at conversational distance.

## Export notes

Same four workarounds as whisper, all in
[docs/whisper-npu.md](whisper-npu.md): `PYTHONPATH` ahead of site-packages, the
`QNN_SDK_ROOT` shim that claims to be x86, calibration supplied by us rather
than a dataset loader, and artifacts saved explicitly.

Two differences worth knowing:

- **The lowering is the bundle's own `build_executorch_binary`** — the same
  entry point its `deeplab_v3.py` example uses. Export, quantize, calibrate,
  lower and write in one call, so [board_faceparse_export.py](../scripts/board_faceparse_export.py)
  is mostly just the model and the calibration data.
- **Calibration is 40 CelebA-HQ faces**, fetched through the HF
  datasets-server rows API rather than a 460 MB parquet shard, preprocessed on
  the Mac and pushed as raw tensors. Preprocessing the calibration set through
  a different code path from the inference set is a good way to set every
  activation range slightly wrong, so both go through the same function.

The checkpoint is `79999_iter.pth`, SHA256 `468e13ca…b26567`, verified at fetch
time. Three independent HF mirrors agree on that digest.

## What this is not, yet

- **The detector runs on the CPU, on the Mac.** Fine at a few milliseconds,
  but in the on-board loop it will need to run there instead — YuNet is ONNX,
  so OpenCV's DNN module handles it, no export needed.
- **Fixed 512×512.** Compiled in, like every QNN graph.
- **argmax runs on the host.** The graph outputs 19 channels of logits —
  19.9 MB a frame. Folding the argmax into the graph would cut that to 262 KB
  and is worth trying if the transfer shows up in a live loop.
- **Not tested on a real webcam frame.** Two obstacles, one solved. The
  board's OpenCV wheel is built without V4L2 (`cv2.getBuildInformation()`
  lists no V4L lines), so `cv2.VideoCapture` cannot open `/dev/video0` — but
  `linuxpy`, a pure-Python v4l2 binding, installs with no system packages and
  does the job. The remaining obstacle is that a `gst-launch-1.0` process was
  holding the camera throughout. GStreamer being present is itself worth
  knowing: it is probably the better capture route for a live loop.

## Sources

- Model: [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) (MIT), checkpoint mirrored at [vivym/face-parsing-bisenet](https://huggingface.co/vivym/face-parsing-bisenet)
- Detector: YuNet, from [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet)
- Calibration: [korexyz/celeba-hq-256x256](https://huggingface.co/datasets/korexyz/celeba-hq-256x256)
- Board runs, 2026-10-17 — builds on [qualcomm-npu.md](qualcomm-npu.md) and [whisper-npu.md](whisper-npu.md)
