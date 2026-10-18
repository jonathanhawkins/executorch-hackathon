#!/usr/bin/env python3
"""Put a person's face onto a finished product photo — the photo-booth path.

    python scripts/face_swap.py --source me.jpg                     # onto the default template
    python scripts/face_swap.py --source me.jpg --target shot.png --out out.png
    python scripts/face_swap.py --source me.jpg --bench             # time each stage

The other way round from the garment overlay in garment_overlay.py: instead of
moving a shirt onto a live frame, this moves a face into a studio shot that
already has the shirt, the lighting and the pose. For a posed "step up, say
take a photo" booth that is the better trade — see docs/faceswap.md.

Three models, all ONNX on the CPU:

  YuNet          face box + 5 landmarks      (scripts/face_detect.py, cached)
  ArcFace r50    512-d identity embedding    112x112, ~175 ms on the board
  inswapper_128  the swap itself             128x128, ~3.9 s on the board

Only the last one is on the clock. The embedding describes the person, not the
moment, so a live booth computes it while they are lining up and spends the
shutter on the swap alone.

Both weight files are research/non-commercial. Fine for a demo; read
docs/faceswap.md before shipping anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import face_detect  # noqa: E402

# Four, not eight. The little cores make both models slower: the swap is 3.9 s
# on four threads and 5.2 s on eight, ArcFace 175 ms against 495 ms.
THREADS = 4

SWAPPER_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
# The deepinsight HF repos for both of these are disabled; the GitHub release
# still serves buffalo_l, and the recogniser inside it is the one the swapper
# was trained against. Mirrors of w600k_r50 alone are not all the same weights.
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ARCFACE_NAME = "w600k_r50.onnx"

# ArcFace's canonical 112x112 landmark positions, in YuNet's own order:
# right_eye, left_eye, nose, right_mouth, left_mouth. "Right" is the subject's
# right, which is the LEFT side of the image — the same crossed convention
# that CLAUDE.md warns about for garment anchors. Swap two rows here and the
# face lands upside down, convincingly enough that it is easy to miss.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # right_eye  (image left)
        [73.5318, 51.5014],  # left_eye   (image right)
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # right_mouth
        [70.7299, 92.2041],  # left_mouth
    ],
    dtype=np.float32,
)

LANDMARK_ORDER = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")

_SESSIONS: dict = {}


def _session(path: str):
    """One session per file. Creating one is far dearer than running it."""
    import onnxruntime as ort

    if path not in _SESSIONS:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.log_severity_level = 3  # the unused-emap warning is expected
        _SESSIONS[path] = ort.InferenceSession(
            path, opts, providers=["CPUExecutionProvider"]
        )
    return _SESSIONS[path]


def arcface_path(model_dir: Path | None = None) -> str:
    """Fetch the recogniser out of buffalo_l on first use, then cache it.

    It caches beside the swapper rather than at a fixed path, so staging the
    two models together is enough — otherwise a deployment that moves the
    swapper quietly re-downloads 289 MB on first use.
    """
    import io
    import urllib.request
    import zipfile

    if model_dir is None:
        model_dir = Path(__file__).resolve().parent.parent / "models/inswapper"
    cached = Path(model_dir) / ARCFACE_NAME
    if cached.exists():
        return str(cached)

    cached.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {ARCFACE_NAME} from buffalo_l (~289 MB, once)", file=sys.stderr)
    with urllib.request.urlopen(BUFFALO_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    cached.write_bytes(archive.read(ARCFACE_NAME))
    return str(cached)


def landmarks_array(face: dict) -> np.ndarray:
    return np.array([face["landmarks"][k] for k in LANDMARK_ORDER], dtype=np.float32)


def align(image, landmarks: np.ndarray, size: int):
    """Similarity transform onto the ArcFace template, scaled to `size`."""
    import cv2

    dst = ARCFACE_TEMPLATE * (size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks, dst, method=cv2.LMEDS, refineIters=20
    )
    if matrix is None:
        raise SystemExit("could not solve the alignment transform")
    crop = cv2.warpAffine(image, matrix, (size, size), borderValue=0.0)
    return crop, matrix


def embed(face_crop, emap: np.ndarray, session) -> np.ndarray:
    """512-d identity latent for the swapper.

    Not the raw ArcFace embedding: the swapper wants it through the model's
    own `emap` matrix. Feed it the bare embedding and you get a plausible
    face that belongs to nobody in particular.
    """
    blob = face_crop[:, :, ::-1].astype(np.float32)  # BGR -> RGB
    blob = ((blob - 127.5) / 127.5).transpose(2, 0, 1)[None]
    embedding = session.run(None, {session.get_inputs()[0].name: blob})[0].ravel()
    latent = (embedding / np.linalg.norm(embedding)).reshape(1, -1) @ emap
    return (latent / np.linalg.norm(latent)).astype(np.float32)


class FaceSwapper:
    """Both sessions held open, and the two halves of the job kept apart.

    A booth wants them apart. `latent_for` depends only on the person, so it
    runs while they are still lining up; `apply` is the only thing the shutter
    pays for. Building the sessions is ~1 s and belongs nowhere near either.
    """

    def __init__(self, swapper_onnx: str):
        self.path = str(swapper_onnx)
        self.emap = load_emap(self.path)
        self.swapper = _session(self.path)
        self.arcface = _session(arcface_path(Path(self.path).parent))

    def warm(self) -> None:
        """Run each graph once, so the first real shutter is not the slow one."""
        self.arcface.run(
            None, {self.arcface.get_inputs()[0].name: np.zeros((1, 3, 112, 112), np.float32)}
        )
        self.swapper.run(
            None,
            {"target": np.zeros((1, 3, 128, 128), np.float32),
             "source": np.zeros((1, 512), np.float32)},
        )

    def latent_for(self, image, face: dict) -> np.ndarray:
        crop, _ = align(image, landmarks_array(face), 112)
        return embed(crop, self.emap, self.arcface)

    def apply(self, target_image, target_face: dict, latent: np.ndarray):
        import cv2  # noqa: F401  (paste_back needs it; import here keeps the cost honest)

        crop, matrix = align(target_image, landmarks_array(target_face), 128)
        blob = (crop[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        output = self.swapper.run(None, {"target": blob, "source": latent})[0]
        swapped = np.clip(output[0].transpose(1, 2, 0)[:, :, ::-1] * 255.0, 0, 255)
        return paste_back(target_image, swapped, matrix)


def load_emap(swapper_onnx: str) -> np.ndarray:
    """The emap rides along in the graph as a dangling initializer.

    No node consumes it — onnxruntime says so on every load — so it has to be
    read out of the file rather than inferred. Doing that needs the `onnx`
    package and a pass over 554 MB, neither of which belongs in a booth loop,
    so the first read caches it as a 1 MB .npy beside the model and nothing
    but numpy is needed afterwards.
    """
    cached = Path(swapper_onnx).with_suffix(".emap.npy")
    if cached.exists():
        return np.load(cached)

    import onnx
    from onnx import numpy_helper

    model = onnx.load(swapper_onnx, load_external_data=False)
    emap = numpy_helper.to_array(model.graph.initializer[-1])
    np.save(cached, emap)
    return emap


def paste_back(frame, swapped, matrix):
    """Warp the 128x128 result home and feather it into the photo.

    The mask is an ellipse, not the crop's square. The swapper regenerates the
    whole 128 box, hair and background included, and its version of those is
    close but not identical — blend the square and its outline shows up as a
    rectangular halo against a flat studio backdrop. An ellipse also keeps the
    decision the billboard wants: their face, the model's hair.
    """
    import cv2

    size = swapped.shape[0]
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(
        mask,
        center=(size // 2, int(size * 0.52)),
        axes=(int(size * 0.37), int(size * 0.45)),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=1.0,
        thickness=-1,
    )
    mask = cv2.GaussianBlur(mask, (0, 0), size * 0.06)

    h, w = frame.shape[:2]
    inverse = cv2.invertAffineTransform(matrix)
    warped = cv2.warpAffine(swapped, inverse, (w, h), borderValue=0.0)
    mask = cv2.warpAffine(mask, inverse, (w, h), borderValue=0.0)[:, :, None]

    return (warped * mask + frame.astype(np.float32) * (1 - mask)).astype(np.uint8)


def swap(source_image, target_image, swapper_onnx: str, timings: dict | None = None):
    """Face from `source_image` onto the largest face in `target_image`.

    The one-shot form, for the CLI. A loop should hold a `FaceSwapper` and
    call its two halves at the moments that suit them.
    """

    def clock(name, fn):
        start = time.time()
        value = fn()
        if timings is not None:
            timings[name] = (time.time() - start) * 1000
        return value

    source_face = clock("detect_source", lambda: face_detect.detect_frame(source_image))
    if source_face is None:
        raise SystemExit("no face found in the source photo")
    target_face = clock("detect_target", lambda: face_detect.detect_frame(target_image))
    if target_face is None:
        raise SystemExit("no face found in the target photo")

    swapper = clock("sessions", lambda: FaceSwapper(swapper_onnx))
    latent = clock("embed", lambda: swapper.latent_for(source_image, source_face))
    return clock("swap", lambda: swapper.apply(target_image, target_face, latent))


def main() -> None:
    import cv2

    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="photo of the person")
    parser.add_argument(
        "--target",
        default=str(root / "assets/templates/model-navy-tee.png"),
        help="the advertisement their face goes into",
    )
    parser.add_argument("--out", default="swapped.png")
    parser.add_argument(
        "--swapper",
        default=str(root / "models/inswapper/inswapper_128.onnx"),
        help=f"inswapper_128.onnx — download from {SWAPPER_URL}",
    )
    parser.add_argument("--bench", action="store_true", help="print per-stage timings")
    args = parser.parse_args()

    if not Path(args.swapper).exists():
        sys.exit(
            f"missing {args.swapper}\n"
            f"  mkdir -p {Path(args.swapper).parent}\n"
            f"  curl -L -o {args.swapper} {SWAPPER_URL}"
        )

    source = cv2.imread(args.source)
    target = cv2.imread(args.target)
    if source is None:
        sys.exit(f"could not read {args.source}")
    if target is None:
        sys.exit(f"could not read {args.target}")

    timings: dict = {}
    result = swap(source, target, args.swapper, timings)
    cv2.imwrite(args.out, result)
    print(args.out)

    if args.bench:
        print(json.dumps({k: round(v, 1) for k, v in timings.items()}, indent=2))
        shutter = timings["swap"]
        print(f"shutter to picture: {shutter / 1000:.2f} s")


if __name__ == "__main__":
    main()
