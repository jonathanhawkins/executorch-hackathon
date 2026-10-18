# Arduino VENTUNO Q (IoT track)

**This is working.** All three models run on the board today.

## What the board actually is

Worth stating plainly, because the public spec sheets describe a weaker part:

| | Spec sheets say | What's actually here |
|---|---|---|
| SoC | QRB2210 | **QCS8275** (Snapdragon, "Monaco") |
| CPU | 4× Cortex-A53 | **4× Cortex-A78C @ 2.36 GHz + 4× Cortex-A55 @ 1.96 GHz** |
| RAM | 4 GB | **14 GB** |
| OS | — | Ubuntu 24.04.4 LTS, aarch64, Python 3.12.3 |
| Storage | — | 55 GB eMMC, ~34 GB free |

Read `make board-info` on your own board before assuming any of this — there
may be more than one hardware revision in the room.

The important consequence: this is a **real Linux machine with big cores**, not
a microcontroller. A 1B model is comfortable here.

## It is not Android

The board speaks adb, which is misleading. It's Ubuntu underneath:

```
Linux QCVENTUNO6 6.8.0-1078-qcom #83-Ubuntu SMP aarch64 GNU/Linux
```

So the Android APK in [`android/`](../android) is **irrelevant to this board**.
Instead, the same `.pte` runs under Python with the same ExecuTorch runtime the
Mac uses — ExecuTorch publishes `manylinux_2_28_aarch64` wheels, and the board's
Python 3.12.3 matches the `cp312` wheels exactly.

That makes the dev loop unusually short: export on the Mac, push, run. No
cross-compilation, no NDK, no app.

## Measured performance

On the board, XNNPACK CPU backend, `8da4w` quantization, 7 threads:

| Model | `.pte` | Prompt eval | **Generation** |
|---|---|---|---|
| SmolLM2-135M | 100 MB | 56.4 tok/s | **41.1 tok/s** |
| Qwen2.5-0.5B | 396 MB | 22.3 tok/s | **20.5 tok/s** |
| Llama-3.2-1B | 789 MB | 15.3 tok/s | **10.1 tok/s** |

For calibration: comfortable reading speed is roughly 10 tok/s, so **Llama 3.2
1B is genuinely usable** and Qwen 0.5B feels snappy.

Model load is effectively instant — the `.pte` is mmap'd and pages in lazily,
so the cost shows up in time-to-first-token (3.4 s for Llama 1B) rather than at
load. If first-token latency matters for your demo, warm the model with a
throwaway generation at startup.

## The loop

```bash
make board-info                        # confirm what's plugged in
make board-setup                       # one-off, ~15 min (see caveat below)
make board-run MODEL=llama3.2-1b-open  # push if needed, then run on board
```

USB transfer runs at ~90 MB/s, so even the 789 MB model pushes in about eight
seconds. `board_run.sh` skips the push when the board already has a
byte-identical copy.

## Setup caveat

`make board-setup` pulls roughly **8 GB**, because torch's `linux-aarch64`
wheel on PyPI is the CUDA (sbsa) build and drags in several GB of `nvidia-*`
packages this board can never use. It works — `torch.cuda.is_available()` is
`False` and everything runs on CPU — it's just fat.

Ubuntu 24.04 also marks its Python externally-managed (PEP 668) and ships no
`pip`, so the setup script uses `uv`, a static binary needing neither `sudo`
nor `ensurepip`.

To trim the install, pre-install CPU-only torch first — see the comment in
[`scripts/board_setup.sh`](../scripts/board_setup.sh).

## Where the headroom is

Generation is CPU-bound on the A78C cluster right now. Two directions:

- **Thread tuning.** The runtime reported "Number of efficient cores 4" and
  defaults accordingly. Pinning to the four big A78C cores with `taskset` is
  worth an experiment — `OMP_NUM_THREADS` is already set to `nproc - 1` in
  `board_infer.py`.
- **The Hexagon NPU.** It's a **v75** HTP, it's running, and the QNN runtime is
  `apt install`-able. But ExecuTorch doesn't list this chip as a target and its
  pip wheel ships no ahead-of-time QNN compiler, so reaching it needs an x86_64
  Linux host and a from-source build. Investigated in detail in
  [qualcomm-npu.md](qualcomm-npu.md) — worth reading before anyone commits half
  a day to it.

## The dual-brain angle

The board pairs this Linux MPU with an STM32H5F5 microcontroller. The projects
that actually show off the hardware put the model on Linux and real-time
sensing or actuation on the MCU — talk to it, it moves; sensors spike, it
explains what changed. See [ideas.md](ideas.md).
