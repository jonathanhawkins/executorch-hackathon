#!/usr/bin/env python3
"""GHOST face swap for the billboard daemon — the NPU path, behind one class.

Runs ON the board, inside board_billboard_daemon.py. Deliberately exposes the
same three methods as `face_swap.FaceSwapper` (`warm`, `latent_for`, `apply`)
so the daemon can pick a backend and change nothing else.

Why this exists alongside face_swap.py: that path is `inswapper_128`, which is
~3.9 s of CPU per shutter and has **non-commercial weights**. GHOST is
Apache-2.0 and runs the swap on the Hexagon. See docs/faceswap-npu.md.

## Where each half runs, and why they differ

    identity   ArcFace iresnet100, XNNPACK .pte, in-process, ~400 ms
    swap       GHOST AEI-Net, QNN .pte, subprocess, ~850 ms

They use different runtimes on purpose.

**The generator goes through `qnn_executor_runner`**, the same one-shot
subprocess `parse_on_npu` already uses for BiSeNet, because the board's Python
cannot load a QNN `.pte` in-process: the bundle tree has no
`_portable_lib.so`, and the pip wheel's runtime has no Qualcomm backend
registered. So each swap pays ~770 ms of model load for ~82 ms of compute.
That is the same trade the face parse already makes (~350 ms load for ~12 ms),
and it is still 4.6x better than inswapper's 3.9 s. Rebuilding the runner with
its own source's persistent-stdin mode would cut this to ~82 ms; CLAUDE.md has
the detail.

**The identity encoder stays in-process** because it is exact there and it is
*not on the shutter's clock* — it describes whoever is standing in front of
the board, not the moment they say "take a photo". XNNPACK rather than eager
PyTorch: 403 ms against 1514 ms for the same weights, and numerically
identical (cosine 1.000000 against the fp32 reference). It is not on the NPU
because quantization destroys this particular model — embedding cosine 0.118
at 16a8w, which is a different person.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import make_ghost_raw as prep  # noqa: E402
from ghost_swap import paste_back, to_bgr  # noqa: E402
from warm_runner import WarmRunner  # noqa: E402

TARGET_SHAPE = (1, 3, 256, 256)
OUTPUT_SHAPE = (1, 3, 256, 256)


class GhostSwapper:
    """Both halves held open, and the two jobs kept apart.

    Same split as `face_swap.FaceSwapper`, for the same reason: `latent_for`
    depends only on the person, so a booth runs it while they are lining up;
    `apply` is the only thing the shutter pays for.
    """

    def __init__(self, arcface_pte, generator_pte, runner, npu_env, work_dir):
        self.arcface_pte = Path(arcface_pte)
        self.generator_pte = Path(generator_pte)
        self.npu_env = npu_env
        self.work = Path(work_dir)

        # `runner` is kept in the signature for callers that pass the stock
        # one-shot binary, but WarmRunner picks for itself: the persistent
        # build if it is present, the stock one otherwise. Passing the stock
        # path explicitly would pin it to the slow path forever.
        self._runner = WarmRunner(
            self.generator_pte, npu_env, workdir=self.work, name="ghost_aei"
        )
        self._method = None
        # The advertisement never changes, so its crop and inverse transform
        # are computed once and kept.
        self._target_key = None
        self._target_tensor = None
        self._target_matrix = None

    # -- loading ---------------------------------------------------------

    def warm(self) -> None:
        """Load the encoder and run each graph once.

        The first real shutter should not be the slow one — and for the NPU
        side, the first run also pays the cDSP's own warm-up on top of the
        model load.
        """
        import torch
        from executorch.runtime import Runtime

        program = Runtime.get().load_program(str(self.arcface_pte))
        self._method = program.load_method("forward")
        self._method.execute([torch.zeros(1, 3, 112, 112)])

        # Warm the NPU graph too: write a zero target and a zero latent and
        # take the model-load hit now rather than on someone's photo.
        self._run_generator(
            np.zeros(TARGET_SHAPE, dtype=np.float32),
            np.zeros((1, 512), dtype=np.float32),
        )

    # -- the person ------------------------------------------------------

    def latent_for(self, frame_bgr, face: dict) -> np.ndarray:
        """The shopper's 512-d identity vector, from the live frame.

        Raw ArcFace output, not L2-normalised: GHOST's AAD layers were trained
        against that scale. See vendor/ghost/__init__.py.
        """
        import torch

        if self._method is None:
            raise RuntimeError("GhostSwapper.warm() was not called")
        crop, _ = prep.align(frame_bgr, prep.landmarks_array(face), prep.CROP_SIZE)
        tensor = torch.from_numpy(prep.identity_tensor(crop))
        return self._method.execute([tensor])[0].numpy().astype(np.float32)

    # -- the shutter -----------------------------------------------------

    def apply(self, target_image, target_face: dict, latent: np.ndarray):
        """Their face into the advertisement. BGR in, BGR out."""
        tensor, matrix = self._target(target_image, target_face)
        out = self._run_generator(tensor, latent)
        return paste_back(target_image, to_bgr(out), matrix, prep.CROP_SIZE)

    # -- internals -------------------------------------------------------

    def _target(self, target_image, target_face: dict):
        key = (id(target_image), target_image.shape)
        if key != self._target_key:
            crop, matrix = prep.align(
                target_image, prep.landmarks_array(target_face), prep.CROP_SIZE
            )
            self._target_key = key
            self._target_tensor = prep.target_tensor(crop)
            self._target_matrix = matrix
        return self._target_tensor, self._target_matrix

    def _run_generator(self, target_tensor: np.ndarray, latent: np.ndarray):
        """One swap through the NPU.

        ~82 ms of compute. What it costs around that depends entirely on
        whether the persistent runner is present: warm it is just the
        compute, cold it is ~770 ms of reloading a 357 MB .pte every time.
        `self.warm` says which you got.

        The scratch files stay on tmpfs. The runner exchanges tensors as
        files, and the board's eMMC does 130 MB/s against tmpfs's 1.3 GB/s —
        on the face parser that was the difference between 138 ms and 26 ms
        warm, i.e. most of the win. `work_dir` comes from the daemon as
        /dev/shm/billboard/work/ghost, which is already tmpfs; WarmRunner
        would default to /dev/shm anyway.
        """
        work = self._runner.work
        work.mkdir(parents=True, exist_ok=True)
        target_path = work / "target_0.raw"
        latent_path = work / "embedding_0.raw"
        target_tensor.astype(np.float32).tofile(target_path)
        latent.astype(np.float32).reshape(1, -1).tofile(latent_path)

        produced = self._runner.run([target_path, latent_path])
        if not produced:
            raise RuntimeError("the NPU runner produced no output")

        array = np.fromfile(produced[0], dtype=np.float32)
        want = int(np.prod(OUTPUT_SHAPE))
        if array.size != want:
            raise RuntimeError(
                f"swap output was {array.size} floats, expected {want}"
            )
        return array.reshape(OUTPUT_SHAPE)

    @property
    def resident(self) -> bool:
        """True when the persistent runner is in use, so the .pte stays loaded.

        Not called `warm` — that name is already the *verb* on this class,
        the one the daemon calls to preload. Having both would silently
        replace the method with a property and break `swapper.warm()`.
        """
        return self._runner.warm

    def close(self) -> None:
        self._runner.close()


def available(arcface_pte, generator_pte) -> tuple[bool, str]:
    """Whether this backend can run at all, with a reason if not."""
    for path, what in ((arcface_pte, "ArcFace"), (generator_pte, "generator")):
        if not Path(path).exists():
            return False, f"no GHOST {what} at {path}"
    try:
        import executorch.runtime  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"executorch runtime unavailable: {type(exc).__name__}"
    return True, ""
