#!/usr/bin/env python3
"""Generate an image with Stable Diffusion v2.1 on the VENTUNO Q's Hexagon NPU.

Runs ON THE BOARD, driven by scripts/board_sd_run.sh.

Two things in the bundle do not fit this model, and between them they are why
this file exists:

  * The dedicated qaihub_stable_diffusion_runner executes the UNet as
    (latent, time_embedding[1,1280], text_emb). Release 0.62.2's QCS8275 graph
    takes (timestep[1,1], latent, text_emb) — a scalar timestep, with the time
    embedding folded into the graph. Wrong order, wrong shape.
  * The prebuilt generic qnn_executor_runner predates its own source's
    persistent-stdin mode, so it cannot be held open across inferences.

The first is still ours to fix, and the numpy below does it: the scheduler,
guidance and requantization that the bundle's runner does in C++ are ported
from its runner.cpp so the arithmetic matches.

The second is fixed now. The patched runner exists (scripts/board_build_runner.sh,
docs/persistent-runner.md), so every graph is held open across its executions
instead of being reloaded per use. That matters most for the UNet, which runs
twice per denoising step and costs ~1.04 s to load: at 8 steps the old path
spent 8.3 s of its 20.7 s loading the same 881 MB context eight times, for
1.7 s of actual compute. `Stage` keeps one warm process per graph and feeds it
one request per execution.

--persistent_runner off restores the old launch-per-stage path exactly, which
is also what happens automatically when the patched binary is not there. In
that mode each stage is one invocation over a batched input_list.txt — the
text encoder does both prompts in one launch, each denoising step does its two
classifier-free-guidance passes in one, and the VAE decodes in one.

Nothing here needs diffusers: a scalar timestep means no UNet2DConditionModel
download for its .time_embedding, and EulerDiscreteScheduler is thirty lines.
"""

import argparse
import contextlib
import json
import os
import re
import subprocess
import threading
import time

import numpy as np

# Stable Diffusion 2.1-base's training schedule: 1000 steps of scaled_linear
# betas. Same constants as the bundle's runner.h.
NUM_TRAIN_TIMESTEPS = 1000
BETA_START, BETA_END = 0.00085, 0.012

LATENT_SHAPE = (1, 64, 64, 4)  # NHWC — the layout the QNN graphs were built for
IMAGE_SHAPE = (1, 512, 512, 3)
UINT16_MAX = 65535

# Tensor names in the AI Hub graphs, confirmed against the context binaries by
# board_sd_export.py. Input *order* is read from encodings.json, not assumed.
OUTPUT_NAMES = {
    "text_encoder": "output_text_embedding",
    "unet": "output_latent",
    "vae": "output_image",
}

LOAD_DONE = re.compile(r"I (\d+):(\d+):([\d.]+) .*Inputs prepared")
# The runner logs one of these per input_list line, and then a "Total ..."
# summary in the same shape — which must not be counted as another execution.
TOOK = re.compile(r"(?<!Total )(\d+) inferences? took ([\d.]+) ms, avg ([\d.]+) ms")


class Profile:
    """Where a denoising step's wall clock goes, phase by phase.

    The NPU's own per-execution time comes from the runner's log. Everything
    else a step spends is host-side, and with ControlNet that is not a
    rounding error: 13 residual tensors, 6.7M elements, are read out of one
    graph, requantized into the next graph's encodings and written back to
    disk — twice per step, once per classifier-free-guidance branch. This
    exists because the difference between the two model sets is not evidence
    about which of those three phases costs what.
    """

    def __init__(self):
        self.phases = {}

    @contextlib.contextmanager
    def __call__(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phases.setdefault(name, []).append(time.perf_counter() - started)

    def summary(self):
        return {
            name: {
                "calls": len(times),
                "total_s": round(sum(times), 4),
                "mean_ms": round(1000 * sum(times) / len(times), 3),
            }
            for name, times in sorted(self.phases.items())
        }


PROFILE = Profile()


def find(encodings, stage, kind, name):
    for entry in encodings[stage][kind]:
        if entry["name"] == name:
            return entry
    raise KeyError(f"{stage}.{kind} has no tensor named {name!r}")


def quantize(data, encoding):
    """Float -> uint16, affine. QNN's offset is the negated zero point."""
    zero_point = abs(encoding["offset"])
    return np.clip(np.rint(data / encoding["scale"]) + zero_point, 0, UINT16_MAX).astype(
        np.uint16
    )


def dequantize(data, encoding):
    zero_point = abs(encoding["offset"])
    return (data.astype(np.float32) - zero_point) * np.float32(encoding["scale"])


def requantize(data, source, target, scale=1.0):
    """uint16 under one encoding -> uint16 under another, optionally scaled.

    Short-circuits when there is nothing to do, which matters: the 13
    ControlNet residuals are ~6.7M elements, and doing this the long way twice
    per step is real CPU time on the board. In practice the producer and
    consumer encodings differ for all 13, so the fast path rarely fires.

    `scale` is ControlNet's conditioning strength. Scaling the residuals is
    exactly what the reference implementation does, and it is free here
    because the dequantize/quantize round trip is already happening.
    """
    if scale == 1.0 and (
        source["scale"] == target["scale"] and source["offset"] == target["offset"]
    ):
        return data
    return quantize(dequantize(data, source) * np.float32(scale), target)


class Written:
    """A tensor that is already in the file the next graph will read it from.

    `Stage._write_batch` takes one of these in place of an array and writes
    nothing. It is how a ControlNet residual gets from that graph's uint16
    output into the UNet's input file in a single pass, instead of being
    materialized as a numpy array and then copied out again — 13.4 MB of copy
    per guidance branch, twice a step.
    """

    __slots__ = ("path", "size")

    def __init__(self, path, size):
        self.path = path
        self.size = size


class Requantizer:
    """ControlNet's 13 residuals, from its output encodings into the UNet's.

    Not one of the 13 producer/consumer pairs shares a scale and an offset, so
    every element of all 6.7 M has to be remapped, twice per step, once per
    guidance branch. Three ways to do it, and they must agree to the bit
    because the only acceptable difference in the image is none:

    ``lut``
        One 65536-entry uint16 table per residual. uint16 in, uint16 out, so
        the table *is* the function — and it is built by running ``pairs``
        below over every value a uint16 can hold, which makes it bit-exact by
        construction rather than by argument about float rounding. 128 KB
        each, 1.6 MB for all 13, built once per image and reused every step.
        Per element it is a gather.
    ``inplace``
        The same arithmetic as ``pairs``, op for op and so rounding for
        rounding, done in one preallocated float32 buffer. Also bit-exact.
        What it drops is not arithmetic but allocation: the expression
        ``quantize(dequantize(x, s) * c, t)`` builds eight temporaries of
        5.2 MB for the largest residual and walks each of them.
    ``affine``
        The dequantize/quantize pair folded into one multiply-add in float32,
        with the two constants hoisted out of the loop. The least work of the
        four, and *not* bit-exact: collapsing four float32 roundings into two
        moves ~0.2 % of elements by 1 LSB, every one of them by exactly 1.
        Kept because it is the obvious thing to try and a comparison nobody
        can re-run is not a result.
    ``pairs``
        What the driver did before any of this: dequantize to float32,
        multiply by the conditioning scale, quantize into the target. The
        reference, and the escape hatch.

    ``conditioning_scale`` folds into the table or the constants, so a
    strength other than 1.0 costs nothing here either.

    `lut` and `affine` write their result *into* the file the UNet will read
    it from, through a shared mapping held open for the life of the image. On
    tmpfs those pages are the file, so the remap's own stores are the write
    and the 13.4 MB per branch that used to be copied out again afterwards is
    not copied at all. `pairs` keeps the array-out-then-tofile shape it always
    had, which is the point of having it.
    """

    def __init__(self, residual_map, conditioning_scale, mode):
        self.mode = mode
        self.scale = conditioning_scale
        self.pairs = residual_map
        self.luts, self.consts = [], []
        self._maps, self._scratch = {}, {}
        every_uint16 = np.arange(1 + UINT16_MAX, dtype=np.uint16)
        for produced, consumed in residual_map:
            if mode == "lut":
                # Built by the reference path itself, over every value a
                # uint16 can hold. There is no rounding argument to get wrong.
                self.luts.append(
                    requantize(every_uint16, produced, consumed, conditioning_scale)
                )
            # `offset` is QNN's negated zero point, as everywhere else here.
            ratio = produced["scale"] * conditioning_scale / consumed["scale"]
            self.consts.append(
                (
                    np.float32(ratio),
                    np.float32(abs(consumed["offset"]) - abs(produced["offset"]) * ratio),
                )
            )

    def _mapping(self, key, path, size):
        """This residual's destination file, mapped once and kept.

        Remapping per call would be correct and much slower: `mode="w+"`
        truncates, so every store would take a fresh page fault — 3300 of them
        per tensor per branch, 86000 a step.
        """
        mapped = self._maps.get(key)
        if mapped is None or mapped.size != size:
            mapped = np.memmap(path, dtype=np.uint16, mode="w+", shape=(size,))
            mapped[:] = 0  # fault the pages in now rather than mid-step
            self._maps[key] = mapped
        return mapped

    def _buffer(self, slot, size):
        """This residual's float32 scratch. One per residual, not per branch:
        it is dead by the time the other branch wants it."""
        scratch = self._scratch.get(slot)
        if scratch is None or scratch.size != size:
            scratch = np.empty(size, dtype=np.float32)
            self._scratch[slot] = scratch
        return scratch

    def into(self, branch, slot, data, path):
        """Requantize residual `slot` of `branch` into the file at `path`.

        Returns the sentinel `Stage._write_batch` understands, so the tensor
        is not written a second time.
        """
        produced, consumed = self.pairs[slot]
        if self.mode == "pairs":
            out = requantize(data, produced, consumed, self.scale)
            out.tofile(path)
            return Written(path, out.size)

        mapped = self._mapping((branch, slot), path, data.size)
        if self.mode == "lut":
            # mode="clip" skips a bounds check that a uint16 index into a
            # 65536-entry table cannot fail.
            np.take(self.luts[slot], data, out=mapped, mode="clip")
        elif self.mode == "affine":
            ratio, bias = self.consts[slot]
            scratch = self._buffer(slot, data.size)
            np.copyto(scratch, data)
            scratch *= ratio
            scratch += bias
            np.rint(scratch, out=scratch)
            np.clip(scratch, 0, UINT16_MAX, out=scratch)
            np.copyto(mapped, scratch, casting="unsafe")
        else:
            # Op for op what `requantize` does, so float32 rounds in the same
            # places; see the class docstring.
            scratch = self._buffer(slot, data.size)
            np.copyto(scratch, data)
            scratch -= abs(produced["offset"])
            scratch *= np.float32(produced["scale"])
            if self.scale != 1.0:
                scratch *= np.float32(self.scale)
            scratch /= np.float32(consumed["scale"])
            np.rint(scratch, out=scratch)
            scratch += abs(consumed["offset"])
            np.clip(scratch, 0, UINT16_MAX, out=scratch)
            np.copyto(mapped, scratch, casting="unsafe")
        return Written(path, data.size)


def training_sigmas():
    """The 1000 sigmas SD 2.1-base was trained on, ascending with timestep."""
    betas = (
        np.linspace(BETA_START**0.5, BETA_END**0.5, NUM_TRAIN_TIMESTEPS, dtype=np.float64) ** 2
    )
    alphas_cumprod = np.cumprod(1.0 - betas)
    return np.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)


def build_schedule(num_steps, max_timestep=NUM_TRAIN_TIMESTEPS - 1, kind="karras"):
    """Timesteps and sigmas for the denoising loop.

    "linspace" is what diffusers' EulerDiscreteScheduler does by default and
    what the bundle's runner.cpp hardcodes. "karras" spaces the sigmas by the
    rho=7 rule from Karras et al., which spends more of the budget at the low
    noise levels where the detail is decided — it costs nothing per step and
    is visibly better at any step count we can afford.

    max_timestep exists because the graph's timestep input is quantized over
    [0, 951]: starting at the usual 999 would saturate it, and the first and
    noisiest step would silently be told the wrong noise level.
    """
    all_sigmas = training_sigmas()
    train_steps = np.arange(NUM_TRAIN_TIMESTEPS)

    if kind == "karras":
        sigma_max = float(np.interp(max_timestep, train_steps, all_sigmas))
        sigma_min = float(all_sigmas[0])
        rho = 7.0
        ramp = np.linspace(0, 1, num_steps, dtype=np.float64)
        inv_max, inv_min = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
        sigmas = (inv_max + ramp * (inv_min - inv_max)) ** rho
        # The UNet wants a timestep, not a sigma, so invert the training table.
        timesteps = np.interp(sigmas, all_sigmas, train_steps)
    else:
        timesteps = max_timestep * (
            1.0 - np.arange(num_steps, dtype=np.float64) / (num_steps - 1)
        )
        sigmas = np.interp(timesteps, train_steps, all_sigmas)

    return timesteps, np.append(sigmas, 0.0)


class Stage:
    """One of the three QNN graphs, run through the prebuilt qnn_executor_runner."""

    def __init__(self, name, args, encodings, env):
        self.name = name
        self.args = args
        self.env = env
        self.pte = os.path.join(args.model_dir, f"{args.prefix}_{name}.pte")
        self.inputs = sorted(encodings[name]["inputs"], key=lambda e: e["index"])
        self.outputs = sorted(encodings[name]["outputs"], key=lambda e: e["index"])
        # ControlNet has 13 outputs; everything else here has one.
        self.output = self.outputs[0]
        self.work = os.path.join(args.workdir, name)
        self.out_dir = os.path.join(self.work, "outputs")
        os.makedirs(self.out_dir, exist_ok=True)

        self.load_s = []
        self.exec_ms = []
        self.peak_rss_mb = 0.0
        # None = not tried yet, False = unavailable, else a live WarmRunner.
        self._warm = None

    # -- the resident runner -----------------------------------------------
    def warm(self):
        """This graph's warm process, started on first use.

        None means there is no patched runner to hold open, and run() keeps
        launching the bundle's binary per call exactly as it always did. A
        failure here is never fatal for the same reason: the fallback is the
        path that has generated every image so far.
        """
        if self._warm is not None:
            return self._warm or None
        if not self.args.persistent:
            self._warm = False
            return None
        try:
            from warm_runner import WarmRunner

            runner = WarmRunner(
                self.pte, self.env, workdir=self.work,
                runner=self.args.persistent, name=f"sd_{self.name}",
                timing=True,
            )
            if not runner.warm:
                print(f"    {self.name}: runner cannot be held open, launching per call",
                      flush=True)
                self._warm = False
                return None
            self._warm = runner
        except Exception as error:  # noqa: BLE001 — speed is optional, the image is not
            print(f"    {self.name}: warm runner unavailable ({error}), launching per call",
                  flush=True)
            self._warm = False
        return self._warm or None

    def close(self):
        """Let go of the cDSP context. Four of these graphs do not have to be
        resident at once — the text encoder is finished after one call."""
        if self._warm:
            self._warm.close()
        self._warm = False

    def _write_batch(self, batch):
        """One line of input_list.txt per inference, paths in graph input order."""
        lines = []
        with PROFILE(f"write:{self.name}"):
            for row, arrays in enumerate(batch):
                paths = []
                for entry in self.inputs:
                    data = arrays[entry["name"]]
                    expected = int(np.prod(entry["shape"]))
                    path = os.path.join(self.work, f"in{row}_{entry['index']}.raw")
                    if isinstance(data, Written):
                        # Already in this exact file, written straight out of
                        # the requantizer. See Requantizer.into().
                        if data.path != path or data.size != expected:
                            raise ValueError(
                                f"{self.name} input {entry['name']}: pre-written as "
                                f"{data.size} elements to {data.path}, graph wants "
                                f"{expected} at {path}"
                            )
                    elif data.size != expected:
                        raise ValueError(
                            f"{self.name} input {entry['name']}: {data.size} elements, "
                            f"want {expected}"
                        )
                    else:
                        data.tofile(path)
                    paths.append(path)
                lines.append(paths)

        list_path = os.path.join(self.work, "input_list.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(" ".join(row) for row in lines) + "\n")
        return list_path, lines

    def _watch_rss(self, proc, stop):
        """High-water RSS of the runner process, polled while it lives."""
        status = f"/proc/{proc.pid}/status"
        while not stop.wait(0.05):
            try:
                with open(status) as f:
                    for line in f:
                        if line.startswith("VmHWM:"):
                            self.peak_rss_mb = max(
                                self.peak_rss_mb, int(line.split()[1]) / 1024
                            )
                            break
            except OSError:
                return

    def run(self, batch, iteration=1):
        """Run every input set in `batch`. Returns one flat uint16 array each."""
        list_path, lines = self._write_batch(batch)
        runner = self.warm()
        if runner is not None:
            try:
                return self._run_warm(runner, lines, iteration)
            except Exception as error:  # noqa: BLE001
                # A warm process that has died must not cost a shopper the
                # picture they are standing there waiting for. Drop it and
                # fall through; worst case is the old speed. Same bargain
                # board_billboard_daemon.py makes for the camera loop.
                print(f"    {self.name}: warm runner failed ({error}), "
                      "relaunching per call", flush=True)
                self.close()

        for stale in os.listdir(self.out_dir):
            os.remove(os.path.join(self.out_dir, stale))

        command = [
            self.args.runner,
            "--model_path",
            self.pte,
            "--input_list_path",
            list_path,
            "--output_folder_path",
            self.out_dir,
            "--iteration",
            str(iteration),
            "--warm_up",
            "0",
        ]
        proc = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=self.work,
            env=self.env,
            text=True,
        )
        stop = threading.Event()
        watcher = threading.Thread(target=self._watch_rss, args=(proc, stop), daemon=True)
        watcher.start()
        stderr = proc.communicate()[1]
        stop.set()
        watcher.join(timeout=1)

        if proc.returncode != 0:
            tail = "\n".join(stderr.strip().split("\n")[-25:])
            raise RuntimeError(
                f"{self.name}: runner exited {proc.returncode}\n"
                f"--- stderr tail ---\n{tail}"
            )

        if self.args.verbose:
            print(stderr, flush=True)

        load = LOAD_DONE.search(stderr)
        if load:
            hours, minutes, seconds = load.groups()
            self.load_s.append(int(hours) * 3600 + int(minutes) * 60 + float(seconds))
        batch_exec_ms = [
            float(match.group(3))
            for line in stderr.split("\n")
            if "Total" not in line
            for match in [TOOK.search(line)]
            if match
        ]
        self.exec_ms.extend(batch_exec_ms)

        outputs = [self._read_outputs(row) for row in range(len(batch))]
        return outputs, batch_exec_ms

    def _read_outputs(self, row):
        """This inference's output tensors, by index.

        Read by name rather than by globbing: the ControlNet graph has 13
        outputs, and a lexicographic sort puts output_0_10 between
        output_0_1 and output_0_2 — which would feed the UNet its residuals
        in the wrong order, and produce a bad picture rather than an error.
        """
        row_outputs = []
        with PROFILE(f"read:{self.name}"):
            for index in range(len(self.outputs)):
                path = os.path.join(self.out_dir, f"output_{row}_{index}.raw")
                if not os.path.exists(path):
                    raise RuntimeError(
                        f"{self.name}: runner wrote no {os.path.basename(path)}"
                    )
                row_outputs.append(np.fromfile(path, dtype=np.uint16))
        # One output is the common case; hand it back bare so callers that
        # predate ControlNet keep reading naturally.
        return row_outputs[0] if len(row_outputs) == 1 else row_outputs

    def _run_warm(self, runner, lines, iteration):
        """One request per execution down a runner that is already loaded.

        The persistent runner writes every inference to output_0_N.raw — a
        live camera loop only ever wants the latest — so each row's outputs
        have to be read before the next request goes down the pipe.
        """
        outputs, batch_exec_ms = [], []
        for paths in lines:
            for _ in range(iteration):
                started = time.time()
                with PROFILE(f"runner:{self.name}"):
                    runner.run(paths)
                if runner.count == 1:
                    # The first request pays for the model load, and the
                    # runner's own log reports it relative to process start.
                    logged = None
                    for line in runner.stderr_tail():
                        logged = LOAD_DONE.search(line) or logged
                    if logged:
                        hours, minutes, seconds = logged.groups()
                        self.load_s.append(
                            int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                        )
                    else:
                        # The tail is 40 lines and QNN is chatty, so fall back
                        # to what the call actually cost us minus the compute.
                        elapsed = time.time() - started
                        self.load_s.append(
                            max(0.0, elapsed - (runner.last_exec_ms or 0) / 1000.0)
                        )
                if runner.last_exec_ms is not None:
                    batch_exec_ms.append(runner.last_exec_ms)
            outputs.append(self._read_outputs(0))
            self._sample_rss(runner.proc)
        self.exec_ms.extend(batch_exec_ms)
        return outputs, batch_exec_ms

    def _sample_rss(self, proc):
        if proc is None:
            return
        try:
            with open(f"/proc/{proc.pid}/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        self.peak_rss_mb = max(
                            self.peak_rss_mb, int(line.split()[1]) / 1024
                        )
                        return
        except OSError:
            return

    def summary(self):
        return {
            "load_s": self.load_s,
            "mean_load_s": float(np.mean(self.load_s)) if self.load_s else None,
            "exec_ms": self.exec_ms,
            "mean_exec_ms": float(np.mean(self.exec_ms)) if self.exec_ms else None,
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "launches": len(self.load_s),
        }


def generate(args, encodings, env):
    # ControlNet-Canny ships a fourth graph and a 16-input UNet; Stable
    # Diffusion v2.1 does not. Everything below keys off whether it is there.
    use_controlnet = "controlnet" in encodings

    te_out = find(encodings, "text_encoder", "outputs", OUTPUT_NAMES["text_encoder"])
    unet_timestep = find(encodings, "unet", "inputs", "timestep")
    unet_latent = find(encodings, "unet", "inputs", "latent")
    unet_text_emb = find(encodings, "unet", "inputs", "text_emb")
    unet_out = find(encodings, "unet", "outputs", OUTPUT_NAMES["unet"])
    vae_in = find(encodings, "vae", "inputs", "latent")
    vae_out = find(encodings, "vae", "outputs", OUTPUT_NAMES["vae"])

    # The highest timestep the graph's quantized input can actually carry.
    max_timestep = min(
        float(NUM_TRAIN_TIMESTEPS - 1), np.floor(UINT16_MAX * unet_timestep["scale"])
    )
    timesteps, sigmas = build_schedule(
        args.num_time_steps, max_timestep, args.schedule
    )

    residual_map, requantizer, cn_text_emb, cn_image_cond = [], None, None, None
    if use_controlnet:
        cn_text_emb = find(encodings, "controlnet", "inputs", "text_emb")
        cn_image_cond = find(encodings, "controlnet", "inputs", "image_cond")
        # controlnet's 13 outputs feed the UNet's 13 controlnet_* inputs. The
        # names differ (down_block_0 vs controlnet_downblock0), so pair them by
        # graph order, which metadata.json confirms is one-for-one.
        produced = sorted(encodings["controlnet"]["outputs"], key=lambda e: e["index"])
        consumed = [
            entry
            for entry in sorted(encodings["unet"]["inputs"], key=lambda e: e["index"])
            if entry["name"].startswith("controlnet_")
        ]
        if len(produced) != len(consumed):
            raise SystemExit(
                f"controlnet emits {len(produced)} residuals but the UNet takes "
                f"{len(consumed)} — these binaries are not a matched set"
            )
        residual_map = list(zip(produced, consumed))
        requantizer = Requantizer(
            residual_map, args.conditioning_scale, args.requant
        )

    stats = {
        "prompt": args.prompt,
        "max_timestep": max_timestep,
        "controlnet": use_controlnet,
        "conditioning_scale": args.conditioning_scale,
        "num_time_steps": args.num_time_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "prediction_type": args.prediction_type,
        "sampler": args.sampler,
        "schedule": args.schedule,
    }

    # ---- text encoder: the prompt and the empty prompt, for CFG. One launch.
    print("==> Text encoder", flush=True)
    text_encoder = Stage("text_encoder", args, encodings, env)
    tokens = [
        np.fromfile(os.path.join(args.tokens_dir, f"tokens_{name}.raw"), dtype=np.int32)
        for name in ("cond", "uncond")
    ]
    quantized, _ = text_encoder.run([{"tokens": t} for t in tokens])
    # The text encoder's output encoding is not the UNet's input encoding, so
    # the embedding has to be requantized between the two graphs — and again
    # separately for ControlNet, which has an encoding of its own.
    float_embeddings = [dequantize(q, te_out) for q in quantized]
    embeddings = [quantize(e, unet_text_emb) for e in float_embeddings]
    cn_embeddings = (
        [quantize(e, cn_text_emb) for e in float_embeddings] if use_controlnet else []
    )
    stats["text_encoder"] = text_encoder.summary()
    # Nothing needs it again, and its context is 437 MB (SD 2.1) of cDSP
    # protection-domain memory that the UNet and the VAE would rather have.
    text_encoder.close()
    print(
        f"    load {stats['text_encoder']['mean_load_s']:.2f} s, "
        f"2 x {stats['text_encoder']['mean_exec_ms']:.0f} ms, "
        f"peak RSS {stats['text_encoder']['peak_rss_mb']:.0f} MB",
        flush=True,
    )

    # ---- the Canny conditioning image, if this is the ControlNet set
    controlnet, quantized_image_cond = None, None
    if use_controlnet:
        cond_path = os.path.join(args.tokens_dir, "image_cond.raw")
        if not os.path.exists(cond_path):
            raise SystemExit(
                f"{cond_path} is missing — run scripts/make_canny_raw.py first"
            )
        # make_canny_raw.py writes the edge map as uint8 0/255; the graph wants
        # it quantized like everything else, in pixel space (no VAE encoder).
        edges = np.fromfile(cond_path, dtype=np.uint8).astype(np.float32) / 255.0
        quantized_image_cond = quantize(edges, cn_image_cond)
        controlnet = Stage("controlnet", args, encodings, env)
        print(
            f"==> ControlNet conditioning, {int(np.count_nonzero(edges))} edge pixels",
            flush=True,
        )

    # ---- UNet: the denoising loop, one launch of two inferences per step
    print(f"==> UNet, {args.num_time_steps} denoising steps", flush=True)
    unet = Stage("unet", args, encodings, env)
    rng = np.random.default_rng(args.seed)
    latent = (rng.standard_normal(LATENT_SHAPE) * sigmas[0]).astype(np.float32)

    step_s, saturated = [], []
    old_denoised = None
    for index, timestep in enumerate(timesteps):
        started = time.time()
        # Euler scales the sample by 1/sqrt(sigma^2+1) before the model sees it.
        scaled = latent / np.sqrt(sigmas[index] ** 2 + 1.0)
        quantized_latent = quantize(scaled, unet_latent)
        quantized_timestep = quantize(np.array([[timestep]], dtype=np.float64), unet_timestep)
        if timestep / unet_timestep["scale"] > UINT16_MAX:
            saturated.append(round(float(timestep), 1))

        cn_ms = []
        if use_controlnet:
            # One ControlNet pass per guidance branch, then its residuals are
            # appended to that branch's UNet inputs.
            residual_rows, cn_ms = controlnet.run(
                [
                    {
                        "text_emb": embedding,
                        "image_cond": quantized_image_cond,
                        "latent": quantized_latent,
                        "timestep": quantized_timestep,
                    }
                    for embedding in cn_embeddings
                ]
            )
            unet_rows = []
            with PROFILE("requant"):
                for branch, (embedding, residuals) in enumerate(
                    zip(embeddings, residual_rows)
                ):
                    row = {
                        "timestep": quantized_timestep,
                        "latent": quantized_latent,
                        "text_emb": embedding,
                    }
                    # `slot`, not `index` — `index` is the denoising step.
                    for slot, data in enumerate(residuals):
                        consumed = residual_map[slot][1]
                        # Straight into the file the UNet reads it from, under
                        # the name _write_batch would have given it.
                        row[consumed["name"]] = requantizer.into(
                            branch,
                            slot,
                            data,
                            os.path.join(
                                unet.work, f"in{branch}_{consumed['index']}.raw"
                            ),
                        )
                    unet_rows.append(row)
        else:
            unet_rows = [
                {
                    "timestep": quantized_timestep,
                    "latent": quantized_latent,
                    "text_emb": embedding,
                }
                for embedding in embeddings
            ]

        predictions, exec_ms = unet.run(unet_rows)
        exec_ms = list(exec_ms) + list(cn_ms)
        # Kept for the benchmark below: with ControlNet the UNet takes 16
        # inputs, and rebuilding them from scratch there is how you get a
        # KeyError on controlnet_downblock0.
        last_unet_row = unet_rows[0]
        noise_cond, noise_uncond = (dequantize(p, unet_out) for p in predictions)
        guided = (noise_uncond + args.guidance_scale * (noise_cond - noise_uncond)).reshape(
            LATENT_SHAPE
        )

        # Recover the denoised sample (the x0 prediction). Both samplers below
        # are written in terms of it, which is also what k-diffusion does.
        sigma, sigma_next = sigmas[index], sigmas[index + 1]
        if args.prediction_type == "v_prediction":
            denoised = guided * (-sigma / np.sqrt(sigma**2 + 1)) + latent / (sigma**2 + 1)
        else:
            denoised = latent - sigma * guided

        if args.sampler == "dpmpp_2m" and sigma_next > 0:
            # DPM-Solver++(2M): second order from the previous step's x0, so it
            # costs exactly one UNet pair per step like Euler but converges far
            # sooner. Falls back to first order on the first and last steps.
            lam, lam_next = -np.log(sigma), -np.log(sigma_next)
            h = lam_next - lam
            if old_denoised is None:
                extrapolated = denoised
            else:
                r = (lam - (-np.log(sigmas[index - 1]))) / h
                extrapolated = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            latent = (sigma_next / sigma) * latent - np.expm1(-h) * extrapolated
        else:
            # Euler: x <- x + ((x - x0) / sigma) * dsigma
            latent = latent + ((latent - denoised) / sigma) * (sigma_next - sigma)
        old_denoised = denoised
        latent = latent.astype(np.float32)
        step_s.append(time.time() - started)
        print(
            f"    step {index + 1:2d}/{args.num_time_steps}  t={timestep:6.1f}  "
            f"sigma={sigmas[index]:7.3f}  {step_s[-1]:5.2f} s wall, "
            f"{sum(exec_ms):6.0f} ms on the NPU",
            flush=True,
        )

    stats["unet"] = unet.summary()
    stats["step_s"] = step_s
    stats["seconds_per_step_wall"] = float(np.mean(step_s))
    # Two UNet executions per step: conditional and unconditional. With
    # ControlNet that is four graph runs, because its residuals are per-branch.
    stats["seconds_per_step_npu"] = 2 * float(np.mean(unet.exec_ms)) / 1000.0
    if use_controlnet:
        stats["controlnet_stage"] = controlnet.summary()
        stats["seconds_per_step_npu"] += 2 * float(np.mean(controlnet.exec_ms)) / 1000.0
    stats["timesteps_saturated"] = saturated
    stats["requant"] = args.requant if use_controlnet else None

    # ---- VAE decoder
    print("==> VAE decoder", flush=True)
    vae = Stage("vae", args, encodings, env)
    quantized_image, _ = vae.run([{"latent": quantize(latent, vae_in)}])
    image = dequantize(quantized_image[0], vae_out).reshape(IMAGE_SHAPE)[0]
    stats["vae"] = vae.summary()

    stats["host_phases"] = PROFILE.summary()
    stages = {"text_encoder": text_encoder, "unet": unet, "vae": vae}
    if use_controlnet:
        stages["controlnet"] = controlnet
    return image, stats, last_unet_row, stages


def benchmark_unet(args, encodings, env, unet_row, stats, unet=None):
    """Load-free per-execution number, for the viability question.

    Runs after the image has been written, and never takes it down with it:
    this is diagnostics, and nobody should lose a picture they waited a minute
    for because a timing run failed.

    `unet` is the stage the image was generated with. Reusing it is the point
    when it is warm — a second Stage would map another 881 MB context beside
    the live one purely to time it.
    """
    if not args.bench_iterations:
        return
    print(
        f"==> Benchmarking the UNet, {args.bench_iterations} back-to-back executions",
        flush=True,
    )
    try:
        bench = unet if (unet is not None and unet.warm()) else Stage(
            "unet", args, encodings, env
        )
        before = len(bench.exec_ms)
        # The same inputs the last denoising step used — including, under
        # ControlNet, its 13 residuals.
        _, bench_ms = bench.run([unet_row], iteration=args.bench_iterations)
        stats["unet_bench_mean_ms"] = float(np.mean(bench_ms))
        stats["unet_bench_load_s"] = bench.summary()["mean_load_s"]
        stats["unet_bench_peak_rss_mb"] = bench.summary()["peak_rss_mb"]
        # The loop's own mean is the number the report quotes; a 20-execution
        # benchmark appended to it would drag it toward the benchmark.
        del bench.exec_ms[before:]
        print(f"    {stats['unet_bench_mean_ms']:.0f} ms per execution", flush=True)
    except Exception as error:  # noqa: BLE001 — diagnostics must not be fatal
        print(f"    benchmark failed, image is already saved: {error}", flush=True)
        stats["unet_bench_error"] = str(error)


def report(stats):
    unet, text_encoder, vae = stats["unet"], stats["text_encoder"], stats["vae"]
    print()
    print("--- measured ---")
    print(
        f"  model load            text_encoder {text_encoder['mean_load_s']:.2f} s, "
        f"unet {unet['mean_load_s']:.2f} s, vae {vae['mean_load_s']:.2f} s"
    )
    print(
        f"  runner launches       text_encoder {text_encoder['launches']}, "
        f"unet {unet['launches']}, vae {vae['launches']}"
        + ("" if stats.get("persistent_runner") else "   (one per execution batch)")
    )
    bench = stats.get("unet_bench_mean_ms")
    if bench:
        print(f"  one UNet execution    {bench:.0f} ms (back-to-back, no reload)")
    print(f"  one UNet execution    {unet['mean_exec_ms']:.0f} ms (in the loop)")
    print(
        f"  seconds per step      {stats['seconds_per_step_npu']:.2f} s on the NPU "
        f"(2 executions), {stats['seconds_per_step_wall']:.2f} s wall"
    )
    phases = stats.get("host_phases") or {}
    if phases:
        steps = stats["num_time_steps"]
        # Per step, so it reads against the line above. The text encoder and
        # the VAE run once per image, not per step, and are excluded.
        per_step = {
            name: entry["total_s"] / steps
            for name, entry in phases.items()
            if name.endswith((":unet", ":controlnet")) or name == "requant"
        }
        ordered = sorted(per_step.items(), key=lambda kv: -kv[1])
        print(
            "  host-side per step    "
            + ", ".join(f"{name} {1000 * seconds:.0f} ms" for name, seconds in ordered)
        )
        if stats.get("requant"):
            print(f"  residual requant      --requant {stats['requant']}")
    print(f"  text encoder          {text_encoder['mean_exec_ms']:.0f} ms per prompt")
    print(f"  VAE decode            {vae['mean_exec_ms']:.0f} ms")
    print(
        f"  peak host RSS         unet {unet['peak_rss_mb']:.0f} MB, "
        f"text_encoder {text_encoder['peak_rss_mb']:.0f} MB, vae {vae['peak_rss_mb']:.0f} MB"
    )
    print(
        f"  {stats['num_time_steps']} steps              "
        f"{stats['seconds_per_step_npu'] * stats['num_time_steps']:.1f} s of NPU time, "
        f"{stats['total_s']:.1f} s wall end to end"
    )
    if stats["timesteps_saturated"]:
        print(
            "  NOTE  timesteps clipped to the graph's 951 ceiling: "
            f"{stats['timesteps_saturated']}"
        )


def resolve_persistent(args):
    """The patched runner that can be held open across executions, or None.

    `auto` looks for it beside the runner we were given, which is how the
    billboard daemon gets it without knowing it exists: it passes the
    bundle's binary, and the patched one is its sibling. See
    docs/persistent-runner.md.
    """
    if args.persistent_runner == "off":
        return None
    if args.persistent_runner != "auto":
        if os.path.exists(args.persistent_runner):
            return args.persistent_runner
        print(f"    {args.persistent_runner} is not there, launching per call", flush=True)
        return None
    candidate = f"{args.runner}_persistent"
    return candidate if os.path.exists(candidate) else None


def main(args):
    env = dict(os.environ)
    with open(os.path.join(args.model_dir, "encodings.json")) as f:
        encodings = json.load(f)
    os.makedirs(args.workdir, exist_ok=True)
    args.persistent = resolve_persistent(args)
    print(
        "  runner    "
        + (f"{os.path.basename(args.persistent)}, held open across executions"
           if args.persistent else "launched per execution (no patched runner)"),
        flush=True,
    )

    wall_start = time.time()
    stages = {}
    try:
        image, stats, last_unet_row, stages = generate(args, encodings, env)
        stats["total_s"] = time.time() - wall_start
        stats["persistent_runner"] = bool(args.persistent)

        from PIL import Image

        # Write the image before anything optional runs.
        Image.fromarray(
            np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
        ).save(args.output)
        stats["output"] = args.output
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=2)

        benchmark_unet(args, encodings, env, last_unet_row, stats, stages.get("unet"))
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=2)
    finally:
        # The warm processes hold cDSP contexts; the billboard daemon is
        # sharing that memory with us and wants them back either way.
        for stage in stages.values():
            stage.close()

    report(stats)
    print(f"  image                 {args.output}")
    print(f"  stats                 {args.stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="a photo of an astronaut riding a horse on mars")
    parser.add_argument("--model_dir", required=True, help="Directory of <prefix>_*.pte")
    parser.add_argument(
        "--prefix",
        default="sd21",
        help="Filename prefix of the .pte files: sd21 or cn15.",
    )
    parser.add_argument("--tokens_dir", required=True, help="Directory of tokens_*.raw")
    parser.add_argument("--runner", required=True, help="Path to qnn_executor_runner")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output", required=True, help="Where to write the PNG")
    parser.add_argument("--stats", required=True, help="Where to write the timings JSON")
    parser.add_argument("--num_time_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument(
        "--conditioning_scale",
        type=float,
        default=1.0,
        help="ControlNet strength. 1.0 follows the edge map hard and leaves "
        "the prompt little room; lower trades structural fidelity for prompt "
        "influence. Ignored without a ControlNet model set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampler",
        choices=("dpmpp_2m", "euler"),
        default="dpmpp_2m",
        help="DPM-Solver++(2M) costs the same per step as Euler and converges "
        "sooner. Euler is what the bundle's runner.cpp implements.",
    )
    parser.add_argument(
        "--schedule",
        choices=("karras", "linspace"),
        default="karras",
        help="How the sigmas are spaced. Karras rho=7 is free and better.",
    )
    parser.add_argument(
        "--prediction_type",
        choices=("v_prediction", "epsilon"),
        default="v_prediction",
        help="What the UNet's output means. This release's graph is "
        "v-prediction; the bundle's own runner assumes epsilon.",
    )
    parser.add_argument(
        "--bench_iterations",
        type=int,
        default=20,
        help="Back-to-back UNet executions in one launch, for the load-free "
        "per-step number. 0 to skip.",
    )
    parser.add_argument(
        "--persistent_runner",
        default="auto",
        help="The patched qnn_executor_runner that can be held open across "
        "executions (docs/persistent-runner.md). 'auto' looks for "
        "<runner>_persistent beside --runner; 'off' forces the old "
        "launch-per-stage path; or give a path.",
    )
    parser.add_argument(
        "--requant",
        choices=("lut", "inplace", "affine", "pairs"),
        default="lut",
        help="How ControlNet's 13 residuals are remapped into the UNet's "
        "input encodings. 'lut' is a 65536-entry table per residual, built "
        "once per image, bit-exact against 'pairs' by construction. "
        "'inplace' is the same arithmetic as 'pairs' in a reused buffer, also "
        "bit-exact. 'affine' is one float32 multiply-add per element: less "
        "work still, and 1 LSB off on ~0.2%% of elements. 'pairs' is the "
        "original dequantize/quantize round trip. Ignored without ControlNet.",
    )
    parser.add_argument("--verbose", action="store_true", help="Echo runner logs")
    main(parser.parse_args())
