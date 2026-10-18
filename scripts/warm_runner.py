#!/usr/bin/env python3
"""A qnn_executor_runner kept alive across inferences.

    from warm_runner import WarmRunner

    with WarmRunner(pte_path, env) as r:
        for frame in frames:
            outs = r.run([frame_raw_path])     # paths to output_0_N.raw

Self-test (on the board):  python3 warm_runner.py <model.pte>

The board's Python cannot load a QNN `.pte` in process, so every inference has
to go out to `qnn_executor_runner`. Launching it per inference reloads the
whole QNN context each time, which on this board costs far more than the
inference: face parsing is 12.5 ms of compute inside a 468 ms process.

This drives the patched runner's `--input_list_path -` mode instead — one
warm process, one request per line on stdin, one `FRAME_READY` line back on
stdout per completed inference. Measured on the face parser, 512x512:

    one-shot, per invocation            468 ms
    warm runner, scratch on eMMC        138 ms
    warm runner, scratch on tmpfs        26 ms

The tmpfs part is not incidental. The runner exchanges tensors as files, and
the face parser's output is 19 channels of float32 at 512x512 — 19.9 MB per
inference. eMMC writes at 130 MB/s and /dev/shm at 1.3 GB/s, which is the
whole difference between those last two rows. The default workdir is therefore
on tmpfs, and callers should keep their input tensors there too.

Needs the patched binary — see docs/persistent-runner.md. It falls back to
one-shot launches against an unpatched runner so nothing breaks, but then the
point is lost; `WarmRunner.warm` says which mode you got.

`timing=True` additionally captures the runner's own per-inference figure into
`last_exec_ms`, which is how board_sd_driver.py keeps reporting NPU time
separately from wall clock now that there is no per-call stderr to parse. It
is off by default: the billboard's camera loop has no use for it, and an
unread queue would grow for as long as the daemon runs.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
from pathlib import Path

BOARD_ET = Path("/home/arduino/git/executorch")
PATCHED = BOARD_ET / "build-x86/examples/qualcomm/executor_runner/qnn_executor_runner_persistent"
ORIGINAL = BOARD_ET / "build-x86/examples/qualcomm/executor_runner/qnn_executor_runner"
READY = "FRAME_READY"
# The runner logs this per inference, on stderr, before it writes FRAME_READY.
# "Total N inference took ..." has the same shape and is the end-of-run
# summary, so it must not be counted as another execution.
TOOK = re.compile(r"(?<!Total )(\d+) inferences? took ([\d.]+) ms, avg ([\d.]+) ms")


def _output_index(path: Path) -> int:
    """The N in output_0_N.raw. Needed because a lexicographic sort puts
    output_0_10 between output_0_1 and output_0_2 — invisible with one output
    and wrong for ControlNet's thirteen."""
    return int(path.stem.rsplit("_", 1)[1])


class WarmRunner:
    def __init__(self, pte, env, workdir=None, runner=None, name="runner", timing=False):
        self.pte = str(pte)
        self.env = env
        self.name = name
        self.runner = Path(runner) if runner else (PATCHED if PATCHED.exists() else ORIGINAL)
        self.warm = self.runner == PATCHED or "persistent" in self.runner.name

        base = Path(workdir) if workdir else Path(f"/dev/shm/warm_{name}")
        self.work = base
        self.out_dir = base / "outputs"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.proc = None
        self.count = 0
        self._stderr_tail: list[str] = []
        # Off by default: the live camera loop does not want its timings, and
        # a queue nobody drains would grow without bound.
        self.timing = timing
        self.last_exec_ms = None
        self._timings: queue.Queue = queue.Queue()

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if not self.warm or self.proc is not None:
            return
        self.proc = subprocess.Popen(
            [str(self.runner), "--model_path", self.pte,
             "--input_list_path", "-", "--output_folder_path", str(self.out_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(self.work), env=self.env, text=True, bufsize=1,
        )
        # Drain stderr continuously: the runner logs per inference, and a full
        # pipe buffer would deadlock it mid-request.
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in self.proc.stderr:
            self._stderr_tail.append(line.rstrip())
            del self._stderr_tail[:-40]
            if self.timing:
                match = TOOK.search(line)
                if match:
                    self._timings.put(float(match.group(3)))

    def stderr_tail(self, n: int = 40) -> list[str]:
        """The runner's most recent log lines. Bounded, so the model-load line
        is only here while it is still recent — see board_sd_driver.py."""
        return list(self._stderr_tail[-n:])

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()          # EOF ends the runner's loop
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        finally:
            self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()

    # -- inference ---------------------------------------------------------
    def run(self, input_paths) -> list[Path]:
        """One inference. Returns the output_0_N.raw paths, in index order."""
        line = " ".join(str(p) for p in input_paths)
        if not self.warm:
            return self._run_one_shot(line)

        self.start()
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"{self.name}: runner exited {self.proc.returncode}\n"
                + "\n".join(self._stderr_tail[-20:])
            )

        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

        while True:
            reply = self.proc.stdout.readline()
            if reply == "":
                raise RuntimeError(
                    f"{self.name}: runner closed stdout mid-request\n"
                    + "\n".join(self._stderr_tail[-20:])
                )
            if reply.startswith(READY):
                break

        if self.timing:
            # The runner logs the timing before FRAME_READY, so the line is
            # already in the pipe; this waits only for the drain thread to
            # pick it up. A timeout rather than a blocking get, because a
            # missing timing must not wedge an inference that succeeded.
            try:
                self.last_exec_ms = self._timings.get(timeout=2)
            except queue.Empty:
                self.last_exec_ms = None

        self.count += 1
        return sorted(self.out_dir.glob("output_0_*.raw"), key=_output_index)

    def _run_one_shot(self, line: str) -> list[Path]:
        """Fallback for an unpatched runner — correct, just not fast."""
        list_path = self.work / "input_list.txt"
        list_path.write_text(line + "\n")
        subprocess.run(
            [str(self.runner), "--model_path", self.pte,
             "--input_list_path", str(list_path),
             "--output_folder_path", str(self.out_dir)],
            cwd=str(self.work), env=self.env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        self.count += 1
        return sorted(self.out_dir.glob("output_0_*.raw"), key=_output_index)


def board_env() -> dict:
    """The LD_LIBRARY_PATH / ADSP_LIBRARY_PATH incantation every board run needs."""
    probe = "/home/arduino/qairt-probe/extracted/usr"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join([
        f"{BOARD_ET}/build-x86/backends/qualcomm",
        f"{BOARD_ET}/build-x86/lib",
        f"{probe}/lib",
        f"{probe}/lib/aarch64-linux-gnu",
    ])
    env["ADSP_LIBRARY_PATH"] = f"{probe}/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp"
    return env


if __name__ == "__main__":
    import sys
    import time

    pte = sys.argv[1]
    inputs = sys.argv[2:] or ["/dev/shm/fp/input_0.raw"]
    with WarmRunner(pte, board_env(), name="selftest") as r:
        print(f"runner: {r.runner.name}  warm={r.warm}")
        t0 = time.time(); r.run(inputs); first = time.time() - t0
        n = 20
        t0 = time.time()
        for _ in range(n):
            outs = r.run(inputs)
        marginal = (time.time() - t0) / n
        print(f"first (incl. load): {first*1000:.0f} ms")
        print(f"marginal          : {marginal*1000:.0f} ms over {n}")
        print(f"outputs           : {[p.name for p in outs]}")
