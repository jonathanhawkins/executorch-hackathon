# Project brief

**ExecuTorch Hackathon — Founders, Inc., October 17–18, 2026.**
Everything below runs on the Arduino VENTUNO Q on the desk: no cloud, no
account, no network needed once the board is flashed.

## The pitch

A store fixture that puts the shopper *in* the product photo. Stand in front of
a camera and you appear wearing the garment — the head cutout, the body matte,
the garment placement and the generated try-on image are all computed on the
board's Hexagon NPU, with the web page that shows the result served from the
same device.

## What's built and measured

| Piece | Where it runs | Cost |
|---|---|---|
| Virtual billboard (BiSeNet head matte → product photo) | NPU | 12 ms compute, ~1 fps end to end |
| Person segmentation (SegFormer human parsing) | NPU | 24.9 ms |
| Generated try-on (Stable Diffusion + ControlNet-Canny) | NPU | ~0.3 s per step |
| Face swap into a studio shot (GHOST, Apache-2.0) | NPU | 82 ms, fp16 |
| LLM text generation (Gemma 4 E2B) | NPU | 188 tok/s prompt, 16.3 tok/s gen, 494 MiB |
| Vision Q&A (InternVL3-1B) | NPU | 0.5 s encode, ~33 tok/s |
| Live speech (Whisper-tiny → Gemma translate) | NPU | 1.8–2.6 s transcript, ~9 s translated |
| LLM baseline (Llama 3.2 1B, XNNPACK) | CPU | 10.1 tok/s |

The NPU numbers are the point: prompt eval is ~12× the CPU baseline on a model
twice the size, with resident memory a fraction of it, because the weights live
in the NPU context.

## How it fits together

Hugging Face model → `optimum-cli export executorch` → quantized `.pte` →
ExecuTorch runtime. The same `.pte` runs on the Mac and on the board, which is
what makes iterating on a laptop and demoing on hardware the same workflow.
NPU models go through the track bundle's QNN AOT path (Linux only) and are
driven by a warm persistent runner — 468 ms per inference launched cold, 26 ms
held open with scratch on tmpfs.

Two independent routes to the same effect, kept separate on purpose:
**overlay** moves a garment onto a live frame (`garment_overlay.py`,
`garment_clip.py`), and **face swap** moves the shopper's face into a studio
shot that already has the garment, lighting and pose (`board_faceswap_*`).

## Hardware

QCS8275, Cortex-A78C big cores, 14 GB RAM, Ubuntu 24.04 aarch64 over adb —
**not Android**, despite answering adb, and considerably better than its
published spec. No root; the compiler, the runners and the Python environment
were all installed without it.

## Status and limits

- Board path: working, demoed, including the web UI and the walkthrough deck.
- Android/XR app (`android/`): compiles, never run — no device was attached.
- macOS cannot do QNN ahead-of-time compilation, so NPU work happens on the board.
- Thin segmentation classes (arms) lose accuracy at 8a8w; silhouettes don't.
- The board reboots under heavy load, so long board-side jobs are resumable.
