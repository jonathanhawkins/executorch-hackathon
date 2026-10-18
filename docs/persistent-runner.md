# Keeping the runner warm

**Working.** Face parsing drops from **468 ms to 26 ms** per inference —
**18×** — with byte-identical output.

```bash
./scripts/board_build_runner.sh      # build it (needs ~/toolchain)
```

```python
from warm_runner import WarmRunner, board_env

with WarmRunner(pte_path, board_env(), name="faceparse") as r:
    for frame in frames:
        outputs = r.run([frame_raw])     # paths to output_0_N.raw
```

## The problem

The board's Python cannot load a QNN `.pte` in process, so every inference
goes out to `qnn_executor_runner`. Launching it per inference reloads the
entire QNN context each time, which costs far more than the inference:

| | compute | wall, one-shot | wall, held open |
|---|---|---|---|
| Face parsing | 12.5 ms | 468 ms | **26 ms** |
| GHOST face swap | ~190 ms | ~960 ms | not yet adopted |
| Stable Diffusion, per step | 0.30 s | 1.34 s | **0.31 s** |
| ControlNet, per step | 0.37 s | 2.33 s | **0.52 s** |

## Two fixes, and the second is the surprise

**1. A warm process.** A 30-line patch to
`examples/qualcomm/executor_runner/qnn_executor_runner.cpp` adds
`--input_list_path -`: the runner reads one request per line from stdin,
keeping method, program and QNN context loaded, and writes `FRAME_READY <n>`
to stdout per completed inference. Logs go to stderr, so that marker is a
clean synchronisation point with nothing to parse.

That alone takes face parsing from 468 ms to **138 ms**.

**2. Scratch files on tmpfs.** The remaining 138 ms is not compute — it is
*file I/O*. The runner exchanges tensors as files, and the face parser's
output is 19 channels of float32 at 512×512: **19.9 MB per inference**.

| | write speed |
|---|---|
| eMMC (`/home/arduino`) | 130 MB/s |
| tmpfs (`/dev/shm`) | **1.3 GB/s** |

Moving the scratch files to `/dev/shm` takes it from 138 ms to **26 ms**.
`WarmRunner` therefore defaults its workdir to tmpfs, and callers should keep
their input tensors there too.

| | per inference |
|---|---|
| One-shot launch, eMMC scratch | 468 ms |
| Warm runner, eMMC scratch | 138 ms |
| **Warm runner, tmpfs scratch** | **26 ms** |

26 ms against 12.5 ms of compute, with most of the remainder being the 19.9 MB
output still crossing a filesystem. Folding the face parser's `argmax` into
the graph would make that output 262 KB instead — the next worthwhile step,
and a model change rather than a plumbing one.

## Correctness

Verified before trusting any of it: the same input through the one-shot path
and the warm path gives **byte-identical** output — 19,922,944 bytes, matching
SHA256. A speedup that changes results is a bug, so this is checked rather
than assumed.

## Building it

The bundle's `build-x86` tree was configured on another machine, at
`/home/arduino/derrjohn_local/git/executorch`, against `/usr/bin/c++` and
`/usr/bin/gmake`. None of those paths exist here, so `cmake --build` would try
to reconfigure and fail.

But the tree records exactly how the target was compiled and linked, in
`flags.make` and `link.txt`. `scripts/board/build_runner.sh` replays those with
the stale prefix rewritten and our own compiler substituted. Only
`qnn_executor_runner.cpp` is recompiled; every other object and archive in the
link line is the bundle's, untouched. The original binary is left in place as
a fallback, and the new one is `qnn_executor_runner_persistent` beside it.

Two traps worth recording:

- **Rewrite the stale prefix only where it starts an argument.** The prebuilt
  `.o` files live in directories that embed the old path as a *name*
  (`CMakeFiles/…/home/arduino/derrjohn_local/…`), so a blanket substitution
  renames them out of existence.
- **`strings` is not installed on this image.** Checking the built binary with
  `strings | grep` returns a silent zero and looks like a failed build. Use
  `grep -a` on the binary.

Needs the toolchain from [board-toolchain.md](board-toolchain.md).

## The live daemon is on it

`scripts/board_billboard_daemon.py` now holds one warm runner per model,
started on first use and reused across frames. Measured against the deployed
daemon:

| model | first call (load) | then |
|---|---|---|
| BiSeNet face parsing | 276 ms | **30 ms** |
| SegFormer person segmentation | 221 ms | **31 ms** |

against the ~350 ms a frame the loop was logging before. Both contexts stay
resident at once without exhausting cDSP protection-domain memory, which was
the risk worth checking — that is what InternVL3-2B died of on this board.

Its scratch was already on tmpfs (`/dev/shm/billboard`), so for the daemon the
reload really was the whole cost.

Two things kept deliberately small, because the demo runs on this file:

- **`WARM_RUNNER=0`** in the daemon's environment restores the old per-call
  launch exactly. Nothing was removed.
- **A dead warm process cannot take the demo down.** If `run()` raises, the
  runner is dropped and that inference falls through to a fresh per-call
  launch, logging a line. Worst case is the old speed, not a broken billboard.

`warm_runner.py` was added to `board_web.sh`'s deploy list; without it the
daemon logs that the warm runner is unavailable and carries on per-call.

## Stable Diffusion is on it too

`scripts/board_sd_driver.py` now holds every graph open for the length of one
image instead of reloading it per use. The UNet was the whole of it: two
executions per denoising step, ~1.0 s to map its 881 MB context, and the old
driver paid that once per step.

| | 8 steps | 20 steps |
|---|---|---|
| ControlNet, reload per step | 20.2 s | 48.0 s |
| **ControlNet, held open** | **7.0 s** | **13.3 s** |
| SD 2.1, reload per step | 12.2 s | 28.3 s |
| **SD 2.1, held open** | **5.1 s** | **8.8 s** |

Per step, in steady state — the first step still pays the load:

| | NPU | wall, reloading | wall, held open |
|---|---|---|---|
| SD 2.1 | 0.30 s | 1.34 s | **0.31 s** |
| ControlNet | 0.37 s | 2.33 s | **0.52 s** |

SD 2.1 is now within 10 ms of its own NPU time, so there is nothing left to
win there. ControlNet's remaining 0.15 s a step is host-side, and the
follow-up work has since broken it down: **83 ms is the requantization
itself** — 13 residuals between ControlNet's output encodings and the UNet's
input encodings, twice per step, none of the 13 pairs sharing an encoding —
and the rest is the tensors crossing the filesystem (33 ms of the runner's own
file I/O, 14 ms for the UNet round trip, 13 ms reading ControlNet's outputs).
This paragraph originally attributed all 0.15 s to the arithmetic, which
overstated it. None of it is the runner's load.

What this buys the demo is **20 steps**, which the generate mode could not
afford before: 13.3 s against the 48 s it used to cost, for an image that is
visibly better than 8 steps on a figure.

Two details that the single-output camera loop never exercised, both fixed in
`warm_runner.py`:

- **`sorted(glob("output_0_*.raw"))` mis-orders ten or more outputs.**
  Lexicographically `output_0_10` sits between `output_0_1` and `output_0_2`.
  Invisible for BiSeNet's one output; for ControlNet's thirteen residuals it
  would have fed the UNet its blocks scrambled — a worse picture, not an
  error. `run()` now sorts by the trailing index.
- **The runner's own per-inference timing was being thrown away.** The drain
  thread kept the last 40 stderr lines for error reporting and discarded the
  rest, so a caller that wants to report NPU time separately from wall clock
  had nothing to read. `timing=True` parses those lines into `last_exec_ms`;
  it is off by default, because the camera loop has no use for it and an
  unread queue would grow for as long as the daemon runs.

### Memory: five contexts at once, with the billboard running

The risk worth checking, since holding several large contexts is what killed
InternVL3-2B. Sampled from `ps` through a real generation, the peak is **five
QNN contexts resident simultaneously**:

```
faceparse_qnn_8a8w.pte              the billboard daemon's
personseg_segformer-b0-atr_16a8w.pte
cn15_controlnet.pte                 369 MB  held across the loop
cn15_unet.pte                       881 MB  held across the loop
cn15_vae.pte                         64 MB  joins for the decode
```

1.31 GB of ControlNet weights on top of the daemon's two, no failure, 10.4 GB
of host RAM still free. The text encoder is closed as soon as its one call is
done, which is why it is not in that list — it is the one graph that is
genuinely finished early, and `Stage.close()` exists for it.

### Correctness

Checked the same way the face parser was, because a speedup that changes
results is a bug. Same seed, same prompt, same conditioning, both paths:

| | SHA256 of the PNG |
|---|---|
| ControlNet 8 steps, reloading / held open / the reference image from before this change | **all three identical** |
| SD 2.1 8 steps, reloading / held open | identical |
| SD 2.1 20 steps, reloading / held open | identical |

And the failure path, since the demo runs on this file: killing the warm UNet
mid-generation drops it, logs a line, relaunches per call for the rest of the
image, and produces the same bytes. `--persistent_runner off` forces that path
from the start, and it is also what happens on its own when the patched binary
is not there.

## Everything that runs a .pte in a loop is on it

- `scripts/board_billboard_daemon.py` — BiSeNet and the SegFormer, per frame.
- `scripts/ghost_booth.py` — the GHOST generator. ~770 ms of its ~960 ms
  shutter was reloading a 357 MB `.pte`; the shutter is 120 ms now.
- `scripts/board_sd_driver.py` — all four Stable Diffusion / ControlNet
  graphs, for the length of one image.

Anything added later should go the same way: construct a `WarmRunner` once
instead of building a command line per call. It costs nothing when the patched
binary is absent, because it falls back on its own.
