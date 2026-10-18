# The three tracks, and what each one actually costs you

Picked from the event brief. The thing that matters for planning is the
**third column** — how far the hardware is from the laptop you develop on.

| Track | Device | Silicon | Status here |
|---|---|---|---|
| Compute | Copilot+ PC | Snapdragon X Elite | No hardware |
| Mobile + XR | Galaxy S25 Ultra, Galaxy XR | Snapdragon 8 Elite / XR2+ Gen 2 | App builds, no device |
| IoT | Arduino VENTUNO Q | **QCS8275 Snapdragon** | **Working — models running** |

## Recommendation: IoT, because it already works

The VENTUNO Q is plugged in and running Llama 3.2 1B at 10 tok/s today. That's
a demo you can build on this afternoon rather than one you're still debugging.

It's also better hardware than advertised — QCS8275 with four Cortex-A78C big
cores and 14 GB RAM, running full Ubuntu 24.04, not the quad-A53/4 GB part the
spec sheets describe. See [board.md](board.md).

The dev loop is the shortest of the three because the board is ordinary aarch64
Linux: export on the Mac, `adb push`, run the same Python. No NDK, no APK, no
cross-compilation.

### If you'd rather do Mobile + XR

Also viable, and the app in [`android/`](../android) is written and compiling —
it just needs a phone. Expect faster inference than the board (Snapdragon 8
Elite is a much bigger chip) at the cost of an untested runtime path.

## The CPU-vs-NPU decision

This is the one architectural choice worth making deliberately, because it
decides whether you need a Linux box.

### XNNPACK (CPU) — what this repo is set up for

- Exports **on macOS**, which is where you are.
- Runs on every one of the three devices.
- Fast enough for 1B-class models. Good enough to demo.
- **Start here.** Get the demo working, then optimise if there's time.

### QNN / Hexagon NPU — the performance story

Qualcomm's NPU is the reason these devices are interesting, and a QNN-delegated
model is several times faster than CPU. The catch:

- The Snapdragon 8 Elite (SM8750) **is** a supported SoC.
- But the ahead-of-time compile step is verified on **Ubuntu 22.04 / CentOS 9 /
  WSL2 only** — not macOS. You need the Qualcomm AI Engine Direct SDK (QNN
  2.37.0 is the version the docs recommend), g++ 13+, and Android NDK 26c.
- So: **a Linux machine, a cloud VM, or WSL on the Copilot+ PC** is a hard
  prerequisite for the NPU path. Budget half a day for the first successful
  QNN export.

Practical plan: demo on XNNPACK, and treat QNN as the stretch goal that wins
the track if you land it. Don't make the demo depend on it.

See [qualcomm-npu.md](qualcomm-npu.md) for the setup detail.

## IoT track note

Correcting what the spec sheets say: the board here is a **QCS8275**, not a
QRB2210 — 4× Cortex-A78C @ 2.36 GHz plus 4× Cortex-A55, and **14 GB RAM**, not
4 GB. A 1B model runs fine (10.1 tok/s measured). Don't size down to 135M on
the assumption that it's a weak part; check your own board with
`make board-info`.

Its Hexagon NPU is real, **proven to execute code** (`make board-npu`), and now
reachable: the track team's Drive bundle supplies both the QCS8275-capable
ExecuTorch source and pre-built NPU models (Gemma 3 1B, Gemma 4 E2B, InternVL3,
Depth Anything V2). Bring-up runs entirely on the board — no x86 host needed.
See [qualcomm-npu.md](qualcomm-npu.md).

The board is genuinely dual-brain — the Ubuntu MPU alongside an STM32H5F5
microcontroller running the Arduino Zephyr core — so the interesting projects
pair the model on Linux with real-time sensing or actuation on the MCU.

## Sources

- [Official hackathon track guide](https://github.com/aaishikasb/qc-executorch-guide)
- [Qualcomm AI Engine backend](https://docs.pytorch.org/executorch/main/backends-qualcomm.html)
- [Arduino VENTUNO Q](https://docs.arduino.cc/hardware/ventuno-q)
