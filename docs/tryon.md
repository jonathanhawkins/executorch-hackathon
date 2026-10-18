# Garment overlay — the billboard effect

**Working.** Put a garment on the person in a frame, rather than the person's
head on a product photo:

```bash
make board-tryon IMAGE=frame.jpg               # CPU only, no board needed
OCCLUDE=1 make board-tryon IMAGE=frame.jpg     # + head matte from the NPU
GARMENT=assets/garments/other.png make board-tryon IMAGE=frame.jpg
```

The head-cutout path is untouched — `make board-faceparse` still does exactly
what it did, and [docs/faceparse-npu.md](faceparse-npu.md) still describes it.
This is an additional mode, not a replacement.

## Why this way round

Pasting a live head onto a studio product photo means a frozen body at one
angle and a live head at another, with a seam and a lighting mismatch across
it, and scale and roll to re-solve every frame. Pasting the garment onto the
live frame means the head is already correct — it is really their head — and
the whole thing moves with the person. For a billboard, moving is the point.

## Cost per frame, measured on the board

| Step | Where | Board | Mac |
|---|---|---|---|
| Face detection, full res | CPU | 70–83 ms | 9.2 ms |
| Face detection, `scale=3` | CPU | **11.5 ms** | — |
| Face parsing (BiSeNet) | **Hexagon NPU** | **12.3–15.8 ms** | — |
| Warp + composite, 1280×720 | CPU | 8.4 ms | 1.6 ms |
| JPEG encode 720p | CPU | 4.0 ms | — |

At full-resolution detection that is ~110 ms a frame, **about 9 fps**, and the
detector is three quarters of it — the "cheap" CPU model costing five times
what the NPU segmentation costs. At `scale=3` the same budget is ~37 ms,
**about 27 fps**.

### Detection scale is the cheapest lever available

Measured on the board, 1280×720 frame, 121 px face:

| scale | input | time | box drift |
|---|---|---|---|
| 1 | 1280×720 | 70.1 ms | — |
| 2 | 640×360 | 20.4 ms | 1 px |
| 3 | 426×240 | **11.5 ms** | 2 px |
| 4 | 320×180 | 7.1 ms | 7 px |

Six times faster for two pixels. `detect_frame(frame, scale=3)` — the default
stays 1 so nothing shifts under anyone tuning placement against full-resolution
boxes, but a live loop should pass 3.

Separately, detection was **127 ms** on the Mac until the detector got cached;
`cv2.FaceDetectorYN.create` costs roughly a hundred times running the thing.
`face_detect.detect_frame()` takes a decoded frame and reuses the detector, and
is the entry point a live loop should call.

## How placement works

[scripts/garment_overlay.py](../scripts/garment_overlay.py):

1. YuNet gives the face box and five landmarks.
2. The eye line gives roll; shoulders run parallel to it and "down the body"
   is its perpendicular.
3. Shoulder points are estimated by proportion: **1.15 × face width** either
   side of centre, **0.45 × face height** below the chin. Both are `--half-width`
   and `--drop` flags, because they are guesses.
4. A similarity transform maps the garment's own shoulder anchors onto those
   points — scale from the distance, rotation from the angle.
5. With `--matte`, the NPU head matte is drawn back over the garment, so hair
   across a shoulder stays in front of the shirt.

### Garments carry their own anchors

A garment is a PNG with alpha plus a JSON sidecar naming where its shoulder
line sits in its own pixels:

```json
{
  "shoulder_left":  [330, 210],
  "shoulder_right": [870, 210],
  "collar_center":  [600, 200]
}
```

Swap the artwork, update three numbers. `assets/garments/placeholder-tee.png`
is a drawn placeholder — replace it with the real product cut-out.

Both the garment anchors and the estimated shoulder points are in **image**
coordinates, left meaning smaller x. YuNet's landmarks use the subject's
left and right, which are mirrored on screen; crossing the two conventions
rotates the shirt 180° and paints it over the face, which is exactly what
happened the first time.

## What the face segmentation buys here

Occlusion, and it is the part a plain overlay cannot fake. Drawing the head
matte back on top puts hair in front of the collar where it belongs. Measured
on a frame where head and collar overlap, the occlusion layer changes ~1,800
pixels right at the chin and neck.

It also tells you where their *actual* shirt is — the `cloth` class — which is
where the new collar line should sit.

## The framing requirement — the real-world blocker

Tested against a live frame from the board's camera, the garment did not
appear at all, and the reason is physical rather than a bug.

At normal webcam distance the subject's face filled 290 px of a 480 px frame.
The shoulder line is estimated at chin + 0.45 face-heights, which put it at
y=526 — **46 px below the bottom of the frame**. The garment had nowhere to
go, the clip step reported `0% of the garment is on the body`, and the output
was the input. Three further live frames were worse, with the face at 307 px.

The arithmetic for what is needed: the placeholder tee extends ~750 px below
its own shoulder line against a 540 px shoulder span, so at a 209 px-wide face
the garment reaches ~1190 px down a 480 px frame. For the shirt to be fully in
shot the **face wants to be about a quarter of the frame height, not two
thirds** — the subject roughly 2.5x further back than someone sitting at a
laptop, or a wider lens, or the camera mounted higher and angled down.

This is an installation parameter, not something to fix in code, and it is
worth settling before tuning placement any further: every placement constant
here was fitted against synthetic frames at a framing the real camera has
never yet produced.

## What is not solved

- **The shoulder proportions are guesses.** They assume someone facing the
  camera roughly square on. A turn or a lean will break them. Real shoulder
  keypoints from a pose model are the fix, and `warp_garment` already takes a
  pair of points from anywhere, so that swap is one function.
- **The garment is rigid.** It scales and rotates; it does not drape, fold or
  respond to body shape. At billboard distance that reads acceptably. Close up
  it will not.
- **No temporal smoothing.** Per-frame detection jitters a few pixels and a
  vibrating garment reads as cheap immediately. An exponential moving average
  over the box and landmarks is the cheapest quality win still on the table —
  and it matters more at `scale=3`, where the box itself is coarser.
- **No arm or hand occlusion.** An arm crossing the chest will be painted
  over. That needs person segmentation, not face parsing.
- **Not tested on a real camera frame.** Everything here is synthetic test
  imagery; see the camera notes in [faceparse-npu.md](faceparse-npu.md).

## Sources

- Detector: YuNet, [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet)
- Matte: [faceparse-npu.md](faceparse-npu.md)
- Board runs, 2026-10-17

## Live mode in the web demo

The overlay is wired into the billboard panel as a second mode — pick
**Try-on — shirt on you** from the dropdown next to Start. The daemon re-reads
`/dev/shm/billboard/control.json` every loop, so switching mode or dragging a
slider takes effect on the next frame with no restart.

~500–590 ms per frame, which is *faster* than billboard mode: it skips the
full-resolution composite onto a 1080×1350 canvas.

### Debug markers

The **show shoulder markers** checkbox draws what the placement is actually
derived from:

- green box and size — the YuNet face box
- blue dots — the five landmarks
- red crosshairs, L and R — the estimated shoulder points
- an amber line between them, and a pixel readout top-left

This is the fastest way to tell *why* a garment sits wrong. Shoulder points are
inferred from the face box by proportion, so seeing them against the real
shoulders immediately separates "wrong parameters" from "wrong camera
distance".

In a close-up it is unmistakable: with a 202×277 face box the estimate puts the
shoulders at y≈487 in a 480px-tall frame, i.e. below the bottom edge. No slider
fixes that — the subject has to step back.

### Placement knobs

| Knob | Does |
|---|---|
| up / down | Shifts the shoulder line, in units of face height |
| width | Half-span of the shoulders, as a multiple of face width |
| shoulder drop | How far below the chin the shoulder line sits |

**Nudging the shirt up does not bring it into frame.** The head matte is drawn
over the garment as an occluder, so raising the shirt tucks it *behind* the
face and it disappears. Up/down is for fine alignment once the subject is
framed correctly, not for rescuing a close-up.

### A UI bug worth remembering

The sliders were first bound straight to server state. The poll is ~1.2 s
behind the pointer, so the thumb snapped back mid-drag and every drag event
fired a POST plus two fetches — which also made the panel look like it was
re-triggering the mode dropdown. They now hold local state, push on a 180 ms
debounce, and ignore incoming server values while a drag is in flight.
