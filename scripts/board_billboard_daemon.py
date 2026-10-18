#!/usr/bin/env python3
"""Virtual billboard: camera -> head cutout on the NPU -> composite on a tee.

Runs ON the board. Reads frames the camera daemon already writes to
/dev/shm/cam, detects a face, parses it with BiSeNet on the Hexagon NPU, cuts
out head/hair/neck, and drops it into the neck hole of a product photo.

Writes /dev/shm/billboard/latest.jpg plus state.json, which the web app serves.

Why the whole frame is not fed to the model: BiSeNet was trained on
CelebAMask-HQ, which is square and face-filling. A 16:9 webcam frame squashed
to 512x512 comes back with hallucinated `hat` and `eye_g` and no neck. YuNet
finds the face first and we crop a square around it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import face_detect  # noqa: E402
import face_swap  # noqa: E402
import ghost_booth  # noqa: E402
import garment_clip  # noqa: E402
import garment_overlay  # noqa: E402
import person_matte  # noqa: E402
from billboard_photo import PhotoBillboard, Smoother  # noqa: E402
from vendor.face_parsing import CLASSES, HEAD_CLASSES  # noqa: E402
import face_matte  # noqa: E402
import board_tryon_gen  # noqa: E402
from make_face_raw import preprocess, square_crop  # noqa: E402
import make_person_raw  # noqa: E402
from person_seg import MODELS as SEG_MODELS  # noqa: E402

BOARD = Path("/home/arduino")
ET = BOARD / "git/executorch"
PROBE = BOARD / "qairt-probe"
CAM_DIR = Path("/dev/shm/cam")
OUT_DIR = Path("/dev/shm/billboard")
WORK = OUT_DIR / "work"
PERSONSEG_WORK = OUT_DIR / "work-personseg"
ASSETS = HERE.parent / "assets"
GARMENTS = HERE.parent / "garments"
TEMPLATES = HERE.parent / "templates"
NECK_IDX = tuple(i for i, n in enumerate(CLASSES) if n in ("neck", "neck_l"))
SKIN_IDX = CLASSES.index("skin")
CONTROL = OUT_DIR / "control.json"
# ControlNet try-on: generate the garment onto the shopper's own pose, then
# put their real head back. See docs/stable-diffusion-npu.md.
CN_MODEL_DIR = BOARD / "models/controlnet-npu"
SD_DRIVER = HERE / "board_sd_driver.py"
PROMPTS = HERE.parent / "prompts"
TRYON_WORK = OUT_DIR / "work-generate"

SWAPPER_ONNX = BOARD / "models/inswapper/inswapper_128.onnx"
# The GHOST path: Apache-2.0 weights, and the swap runs on the Hexagon rather
# than 3.9s of CPU. SWAPPER=inswapper falls back to the old one.
# See docs/faceswap-npu.md.
GHOST_ARCFACE = BOARD / "models/faceswap-npu/ghost_arcface_xnnpack.pte"
GHOST_GENERATOR = BOARD / "models/faceswap-npu/ghost_aei_qnn_float.pte"
SWAPPER_BACKEND = os.environ.get("SWAPPER", "ghost")
# How stale the shopper's identity latent may be at the shutter. It only has
# to survive them saying "take a photo", and it is short enough that the next
# person in the queue cannot inherit it.
#
# Per backend, because the two encoders cost very differently and this runs
# inside the preview loop: inswapper's ArcFace is 175ms through onnxruntime,
# GHOST's is ~400ms through XNNPACK. Holding 2.0s for both would take the
# loop's duty cycle from ~9% to ~20% and make the preview visibly stutter
# while someone is lining up. These keep it at ~9% either way, and 4.5s is
# still far less than it takes one person to leave and the next to align.
LATENT_TTL = 2.0 if os.environ.get("SWAPPER") == "inswapper" else 4.5

MODEL = BOARD / "models/faceparse-npu/faceparse_qnn_8a8w.pte"
# The second segmentation model: whole body and limbs, for clipping the garment
# to the wearer's outline. 16a8w rather than 8a8w — see docs/personseg-npu.md.
PERSONSEG_MODEL = BOARD / "models/personseg-npu/personseg_segformer-b0-atr_16a8w.pte"
RUNNER = ET / "build-x86/examples/qualcomm/executor_runner/qnn_executor_runner"

INTERVAL = float(os.environ.get("BILLBOARD_INTERVAL", "0.8"))
SCORE = float(os.environ.get("FACE_SCORE", "0.6"))

NPU_ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": ":".join([
        str(ET / "build-x86/backends/qualcomm"),
        str(ET / "build-x86/lib"),
        str(PROBE / "extracted/usr/lib"),
        str(PROBE / "extracted/usr/lib/aarch64-linux-gnu"),
    ]),
    "ADSP_LIBRARY_PATH": str(
        PROBE / "extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp"
    ),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def newest_frame() -> Path | None:
    """Copy the newest camera frame somewhere stable, and return that.

    The camera daemon keeps only a handful of frames (`max-files=4`) and
    deletes them within a few hundred milliseconds, so reading the shared path
    directly races the writer — the file vanishes between the detector opening
    it and the compositor doing the same. Snapshot it once up front instead.
    """
    jpegs = sorted(CAM_DIR.glob("*.jpg"))
    if not jpegs:
        return None
    # Skip the very newest: multifilesink may still be writing it.
    src = jpegs[-2] if len(jpegs) > 1 else jpegs[0]
    dest = WORK / "frame.jpg"
    try:
        dest.write_bytes(src.read_bytes())
    except OSError:
        return None
    return dest if dest.stat().st_size > 0 else None


# One warm runner process per model, reused across frames. Launching the
# runner per inference reloads the whole QNN context each time, which is what
# made this loop cost ~350ms a frame for models that take 12-25ms. The
# scratch files are already on tmpfs (OUT_DIR is under /dev/shm), so the
# reload was the whole of it. WARM_RUNNER=0 restores the old per-call launch.
# See docs/persistent-runner.md.
WARM_RUNNERS: dict = {}
USE_WARM_RUNNER = os.environ.get("WARM_RUNNER", "1") != "0"


def _warm_runner(model: Path, work: Path):
    """The warm process for `model`, started on first use. None if unavailable."""
    if not USE_WARM_RUNNER:
        return None
    key = str(model)
    if key not in WARM_RUNNERS:
        try:
            from warm_runner import WarmRunner

            runner = WarmRunner(model, NPU_ENV, workdir=work, name=model.stem)
            if not runner.warm:
                log(f"warm runner unavailable for {model.name}, using per-call launch")
                WARM_RUNNERS[key] = None
            else:
                runner.start()
                WARM_RUNNERS[key] = runner
                log(f"warm runner up for {model.name}")
        except Exception as exc:
            log(f"warm runner failed for {model.name} ({exc}); using per-call launch")
            WARM_RUNNERS[key] = None
    return WARM_RUNNERS[key]


def run_on_npu(tensor: np.ndarray, model: Path, work: Path) -> Path:
    """Run one QNN graph on the Hexagon and hand back its output tensor.

    Each model gets its own working directory. They share the runner binary
    but not the scratch files, so a second model cannot read the first one's
    input list or pick up its stale outputs.
    """
    work.mkdir(parents=True, exist_ok=True)
    tensor.astype(np.float32).tofile(work / "input_0.raw")
    (work / "input_list.txt").write_text("input_0.raw\n")

    out_dir = work / "outputs"
    out_dir.mkdir(exist_ok=True)
    for stale in out_dir.glob("output_*.raw"):
        stale.unlink()

    runner = _warm_runner(model, work)
    if runner is not None:
        try:
            produced = runner.run([str(work / "input_0.raw")])
            if produced:
                return produced[0]
        except Exception as exc:
            # A warm process that has died must not take the demo with it:
            # drop it and fall through to a fresh per-call launch.
            log(f"warm runner for {model.name} failed ({exc}); relaunching per call")
            WARM_RUNNERS[str(model)] = None

    subprocess.run(
        [
            str(RUNNER),
            "--model_path", str(model),
            "--input_list_path", "input_list.txt",
            "--output_folder_path", "outputs",
        ],
        cwd=work, env=NPU_ENV, capture_output=True, timeout=60,
    )
    produced = sorted(out_dir.glob("output_*.raw"))
    if not produced:
        raise RuntimeError(f"the NPU runner produced no output for {model.name}")
    return produced[0]


def parse_on_npu(tensor: np.ndarray) -> Path:
    """Run BiSeNet on the Hexagon. ~12ms of compute, ~0.35s of model load."""
    return run_on_npu(tensor, MODEL, WORK)


def personseg_on_npu(tensor: np.ndarray) -> Path:
    """Run the human-parsing SegFormer on the Hexagon. ~25ms of compute."""
    return run_on_npu(tensor, PERSONSEG_MODEL, PERSONSEG_WORK)




def read_control() -> dict:
    """Mode is read every loop so the UI can switch without a restart."""
    try:
        return json.loads(CONTROL.read_text())
    except Exception:
        return {"mode": "billboard"}



def load_photo_billboard():
    """The photographic template, if one has been prepared.

    Same two rules as load_ad_photo, and for the same reasons. `sorted()`
    because bare glob order is filesystem order, so with two templates the
    pick is arbitrary and can differ between runs on identical files — a
    flaky demo rather than a wrong one. The stem test because every derived
    artifact is `<stem>.<what>.json`; this one is worse than load_ad_photo's
    was, since `stem` is then used to build every sibling path, so a wrong
    pick here fails as "no template prepared" rather than as a bad file.
    """
    meta = next((j for j in sorted(TEMPLATES.glob("*.json"))
                 if "." not in j.stem), None)
    if meta is None:
        return None
    stem = str(meta)[: -len(".json")]
    tpl, skin = Path(f"{stem}.template.png"), Path(f"{stem}.skin.png")
    if not (tpl.exists() and skin.exists()):
        return None
    return PhotoBillboard(tpl, skin, meta)


def load_ad_photo():
    """The product shot the shopper's face goes into — head still attached.

    Deliberately not `*.template.png`: that is the headless, inpainted one
    prep_template.py produces for the compositing path. The swapper replaces
    a face, so there has to be one there to replace.

    Picks by shape rather than by a list of names to avoid. Every derived
    artifact is `<stem>.<what>.png` — `.template`, `.skin`, `.head` — so the
    original is the one whose stem has no dot in it. A blacklist missed
    `.head.png` when prep_template.py started emitting it: at 3.7 kB it sorts
    before the 868 kB photo, so `sorted()` picked it and the mode died with
    "no face in model-navy-tee.head.png".
    """
    for png in sorted(TEMPLATES.glob("*.png")):
        if "." in png.stem:
            continue
        return png
    return None


class FaceSwapBooth:
    """Face-swap mode: their face into the model's photo, one shot at a time.

    Holds everything that does not change between shoppers — two onnxruntime
    sessions and the advertisement's own face detection — and the one thing
    that does, the identity latent. The split matters: the latent is computed
    while they are still lining up, so the shutter only pays for the swap.

    Loading is deferred to the first loop in this mode rather than done at
    startup, so the other three modes do not carry 730 MB of weights they
    never use. Once loaded it stays resident.
    """

    def __init__(self, photo_path: Path | None):
        self.photo_path = photo_path
        self.swapper = None
        self.target = None
        self.target_face = None
        self.latent = None
        self.latent_at = 0.0

    def ensure_loaded(self) -> tuple[bool, str]:
        if self.swapper is not None:
            return True, ""
        if self.photo_path is None:
            return False, f"no product photo in {TEMPLATES}"

        use_ghost = SWAPPER_BACKEND != "inswapper"
        if use_ghost:
            ok, why = ghost_booth.available(GHOST_ARCFACE, GHOST_GENERATOR)
            if not ok:
                # Say which backend failed and that the other one exists —
                # "no swapper" with two of them installed is a confusing thing
                # to read in a log at a demo.
                return False, f"{why} (SWAPPER=inswapper falls back to the CPU path)"
        elif not SWAPPER_ONNX.exists():
            return False, f"no swapper at {SWAPPER_ONNX}"

        import cv2

        t0 = time.time()
        self.target = cv2.imread(str(self.photo_path))
        if self.target is None:
            return False, f"could not read {self.photo_path}"
        # The advertisement never moves, so its detection is done once here
        # rather than on every shutter.
        self.target_face = face_detect.detect_frame(self.target, score_threshold=SCORE)
        if self.target_face is None:
            return False, f"no face in {self.photo_path.name}"

        if use_ghost:
            swapper = ghost_booth.GhostSwapper(
                GHOST_ARCFACE, GHOST_GENERATOR, RUNNER, NPU_ENV, WORK / "ghost"
            )
        else:
            swapper = face_swap.FaceSwapper(str(SWAPPER_ONNX))
        swapper.warm()
        self.swapper = swapper
        log(f"face swap ready [{'ghost/npu' if use_ghost else 'inswapper/cpu'}]: "
            f"{self.photo_path.name}, {(time.time() - t0):.1f}s to load")
        return True, ""

    def refresh_latent(self, frame_bgr, face: dict) -> None:
        if time.time() - self.latent_at < LATENT_TTL:
            return
        self.latent = self.swapper.latent_for(frame_bgr, face)
        self.latent_at = time.time()

    def shoot(self) -> Image.Image:
        result = self.swapper.apply(self.target, self.target_face, self.latent)
        return Image.fromarray(result[:, :, ::-1])  # BGR -> RGB for PIL


def load_garment():
    """The garment PNG and its anchors.

    `sorted()` for determinism: glob order is filesystem order, and
    garments/ holds one PNG today but state.json already advertises six
    labels. Without this, "it picked the wrong shirt sometimes" is a bug
    waiting for the second file to land.
    """
    png = next(iter(sorted(GARMENTS.glob("*.png"))), None)
    if png is None:
        return None, None
    meta = json.loads(png.with_suffix(".json").read_text())
    return Image.open(png).convert("RGBA"), meta


def _keep_face_region(alpha: Image.Image, seed: tuple[int, int]) -> Image.Image:
    """Drop every disconnected blob except the one containing `seed`."""
    import cv2

    arr = (np.asarray(alpha) > 8).astype(np.uint8)
    count, labels = cv2.connectedComponents(arr, connectivity=8)
    if count <= 2:
        return alpha
    x, y = int(seed[0]), int(seed[1])
    if not (0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]):
        return alpha
    keep = int(labels[y, x])
    if keep == 0:  # seed landed on a hole; fall back to the biggest region
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep = int(sizes.argmax())
    mask = (labels == keep).astype(np.uint8) * 255
    return Image.fromarray(np.minimum(np.asarray(alpha), mask), mode="L")


class Billboard:
    """The product photo and where the head belongs on it."""

    def __init__(self, assets: Path):
        meta = json.loads((assets / "billboard.json").read_text())
        self.size = (meta["width"], meta["height"])
        self.neck = tuple(meta["neck"])
        self.eye_y = meta["eyeY"]
        self.eye_span = meta["eyeSpan"]
        self.brand = meta.get("brand", "")
        self.backdrop = Image.open(assets / "backdrop.png").convert("RGBA")
        self.body = Image.open(assets / "body.png").convert("RGBA")
        self.shoulders = Image.open(assets / "shoulders.png").convert("RGBA")
        self.collar = Image.open(assets / "collar.png").convert("RGBA")

        # The head layer now stops just under the jaw. Everything below is the
        # drawn neck, which is cleaner than anything the parser gives back
        # there: the segmented neck carries wall pixels and frays at the
        # bottom. Cutting below the chin keeps the jaw and beard intact.
        chin_y = self.eye_y + int(1.2 * self.eye_span)
        cut_y = chin_y + 18
        clip = Image.new("L", self.size, 0)
        ImageDraw.Draw(clip).rectangle([0, 0, self.size[0], cut_y], fill=255)
        self.head_clip = clip.filter(ImageFilter.GaussianBlur(4))

    def draw_neck(self, skin_rgb) -> Image.Image:
        """A simple neck in the subject's own skin tone.

        The segmented neck is the weakest part of the matte — it carries wall
        pixels, frays at the bottom and ends in whatever the clip cut. Drawing
        it instead gives a clean column that always meets the collar, and
        sampling the colour from this person's face keeps it theirs rather than
        a generic tone.
        """
        layer = Image.new("RGBA", self.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        cx = self.neck[0]
        chin_y = self.eye_y + int(1.2 * self.eye_span)
        bottom = self.neck[1] + 70          # well inside the collar
        half_top = int(0.60 * self.eye_span)
        half_bot = int(0.78 * self.eye_span)  # widens into the shoulders

        d.polygon(
            [
                (cx - half_top, chin_y - 30),
                (cx + half_top, chin_y - 30),
                (cx + half_bot, bottom),
                (cx - half_bot, bottom),
            ],
            fill=skin_rgb + (255,),
        )

        # Contact shadow: the jaw casts onto the neck, and without it the
        # column reads as a flat plank stuck under the chin.
        shadow = Image.new("RGBA", self.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.ellipse(
            [cx - half_top - 10, chin_y - 64, cx + half_top + 10, chin_y + 54],
            fill=shade(skin_rgb, 0.62) + (205,),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(22))
        layer = Image.alpha_composite(layer, shadow)

        # Keep the shadow inside the neck column.
        clip = Image.new("L", self.size, 0)
        ImageDraw.Draw(clip).polygon(
            [
                (cx - half_top, chin_y - 30),
                (cx + half_top, chin_y - 30),
                (cx + half_bot, bottom),
                (cx - half_bot, bottom),
            ],
            fill=255,
        )
        layer.putalpha(ImageChops.multiply(layer.getchannel("A"),
                                           clip.filter(ImageFilter.GaussianBlur(3))))
        return layer

    def compose(self, head: Image.Image, eye_l, eye_r, skin_rgb=None) -> Image.Image:
        """Place `head` so its eye line matches the billboard's.

        Scaling by eye separation rather than by the face box keeps a person
        leaning in the same size as one standing back, which is what stops the
        head jumping around between frames.
        """
        dx, dy = eye_r[0] - eye_l[0], eye_r[1] - eye_l[1]
        span = max(math.hypot(dx, dy), 1e-3)
        angle = math.degrees(math.atan2(dy, dx))
        scale = self.eye_span / span

        mid = ((eye_l[0] + eye_r[0]) / 2.0, (eye_l[1] + eye_r[1]) / 2.0)

        # Pad before rotating, then rotate in place about the eye midpoint.
        # `expand=True` would shift the origin by an amount that depends on the
        # centre, which makes the midpoint hard to track; padding and rotating
        # without expand keeps it at a known pixel and cannot clip the corners.
        pw, ph = head.width // 2, head.height // 2
        padded = Image.new("RGBA", (head.width + 2 * pw, head.height + 2 * ph), (0, 0, 0, 0))
        padded.paste(head, (pw, ph), head)
        mid_pad = (mid[0] + pw, mid[1] + ph)

        rotated = padded.rotate(
            angle, resample=Image.BICUBIC, center=mid_pad, expand=False
        )

        new_size = (max(1, int(rotated.width * scale)), max(1, int(rotated.height * scale)))
        scaled = rotated.resize(new_size, Image.LANCZOS)
        mid_scaled = (mid_pad[0] * scale, mid_pad[1] * scale)

        canvas = self.backdrop.copy()
        canvas = Image.alpha_composite(canvas, self.body)
        # Drawn neck goes under the head, over the shirt.
        if skin_rgb is not None:
            canvas = Image.alpha_composite(canvas, self.draw_neck(skin_rgb))

        layer = Image.new("RGBA", self.size, (0, 0, 0, 0))
        paste_x = int(round(self.neck[0] - mid_scaled[0]))
        paste_y = int(round(self.eye_y - mid_scaled[1]))
        layer.paste(scaled, (paste_x, paste_y), scaled)

        # Trim everything below the neckline so the shirt stays visible.
        layer.putalpha(ImageChops.multiply(layer.getchannel("A"), self.head_clip))

        # Clipping can sever a hair strand from the head, leaving the upper
        # piece floating beside the collar. Keep only the region the face is
        # actually in — seeded at the eye line, which is always on the head.
        layer.putalpha(_keep_face_region(
            layer.getchannel("A"), (self.neck[0], self.eye_y)
        ))

        canvas = Image.alpha_composite(canvas, layer)
        # Shoulders then collar, both over the head: the yoke occludes the
        # sides of the neck and the collar hides its bottom edge, so the head
        # sits *in* the shirt rather than on top of it.
        canvas = Image.alpha_composite(canvas, self.shoulders)
        canvas = Image.alpha_composite(canvas, self.collar)
        return canvas.convert("RGB")



def draw_debug(canvas: Image.Image, face: dict, shoulder_l, shoulder_r) -> None:
    """Overlay what the placement is actually derived from.

    Without this you are guessing why a garment sits wrong: the shoulder points
    are inferred from the face box by proportion, so seeing them next to the
    real shoulders tells you immediately whether to change `drop`, `width`, or
    just the camera distance.
    """
    d = ImageDraw.Draw(canvas, "RGBA")
    x, y, w, h = face["bbox"]
    d.rectangle([x, y, x + w, y + h], outline=(94, 224, 160, 255), width=2)
    d.text((x, max(0, y - 14)), f"face {w}x{h}", fill=(94, 224, 160, 255))

    for pt, name in ((shoulder_l, "L"), (shoulder_r, "R")):
        px, py = int(pt[0]), int(pt[1])
        d.line([px - 14, py, px + 14, py], fill=(240, 102, 94, 255), width=3)
        d.line([px, py - 14, px, py + 14], fill=(240, 102, 94, 255), width=3)
        d.ellipse([px - 9, py - 9, px + 9, py + 9], outline=(240, 102, 94, 255), width=2)
        d.text((px + 12, py + 8), name, fill=(240, 102, 94, 255))

    d.line([int(shoulder_l[0]), int(shoulder_l[1]),
            int(shoulder_r[0]), int(shoulder_r[1])],
           fill=(240, 168, 94, 220), width=2)

    for pt in face["landmarks"].values():
        d.ellipse([pt[0] - 2, pt[1] - 2, pt[0] + 2, pt[1] + 2],
                  fill=(120, 190, 255, 255))

    span = int(abs(shoulder_r[0] - shoulder_l[0]))
    d.text((8, 8), f"shoulders {span}px  frame {canvas.width}x{canvas.height}",
           fill=(255, 255, 255, 230))


def compose_tryon(frame: Image.Image, face: dict, head_alpha, crop_box, garment, meta,
                  drop: float = 0.45, half_width: float = 1.15, offset_y: float = 0.0,
                  debug: bool = False):
    """Put the garment on the person, in their own scene.

    The opposite arrangement to the billboard: instead of lifting the head out
    and dropping it on a product photo, the shirt is warped onto the body that
    is already there. The head matte comes back over the top as an occluder, so
    hair falling across a shoulder stays in front of the fabric.
    """
    size = frame.size
    shoulder_l, shoulder_r = garment_overlay.estimate_shoulders(
        face, half_width=half_width, drop=drop
    )
    # A plain vertical nudge, in units of face height, for when the proportional
    # estimate is right but the framing is not — a close-up puts the shoulders
    # near the frame edge and the shirt body falls off the bottom.
    if offset_y:
        shift = offset_y * face["bbox"][3]
        shoulder_l = (shoulder_l[0], shoulder_l[1] + shift)
        shoulder_r = (shoulder_r[0], shoulder_r[1] + shift)
    warped = garment_overlay.warp_garment(garment, meta, shoulder_l, shoulder_r, size)

    canvas = frame.convert("RGBA")
    canvas = Image.alpha_composite(canvas, warped)

    # Head back on top, so the garment cannot cover hair or chin.
    left, top, side = crop_box
    occluder = Image.new("RGBA", size, (0, 0, 0, 0))
    head_rgb = frame.crop((left, top, left + side, top + side)).convert("RGBA")
    a = head_alpha.resize((side, side), Image.BILINEAR)
    head_rgb.putalpha(a)
    occluder.paste(head_rgb, (left, top), head_rgb)
    canvas = Image.alpha_composite(canvas, occluder)

    # numpy comparisons give np.bool_, which json.dumps refuses.
    on_screen = bool(
        0 <= float(shoulder_l[1]) < size[1] and 0 <= float(shoulder_r[1]) < size[1]
    )
    span = float(np.linalg.norm(np.asarray(shoulder_r) - np.asarray(shoulder_l)))
    out = canvas.convert("RGB")
    if debug:
        draw_debug(out, face, shoulder_l, shoulder_r)
    return out, span, on_screen


def compose_tryon_clipped(frame: Image.Image, face: dict, head_alpha, crop_box,
                          garment, meta, drop: float = 0.45,
                          half_width: float = 1.15, offset_y: float = 0.0,
                          debug: bool = False):
    """`compose_tryon`, plus the body: a second NPU pass that cuts the garment
    to the wearer's silhouette and puts their arms back in front of it.

    Placement is identical to `compose_tryon` — the same shoulder estimate and
    the same warp. What changes is which of the garment's pixels survive, and
    that is `garment_clip`'s job, not this one's. See docs/personseg-npu.md.
    """
    size = frame.size
    shoulder_l, shoulder_r = garment_overlay.estimate_shoulders(
        face, half_width=half_width, drop=drop
    )
    if offset_y:
        shift = offset_y * face["bbox"][3]
        shoulder_l = (shoulder_l[0], shoulder_l[1] + shift)
        shoulder_r = (shoulder_r[0], shoulder_r[1] + shift)
    warped = garment_overlay.warp_garment(garment, meta, shoulder_l, shoulder_r, size)

    # Body crop around the face we already have, so the detector does not run
    # a second time, then the SegFormer on the Hexagon.
    seg_model = SEG_MODELS["segformer-b0-atr"]
    tensor, crop = make_person_raw.prepare(frame, seg_model, face=face)
    PERSONSEG_WORK.mkdir(parents=True, exist_ok=True)
    crop_file = PERSONSEG_WORK / "person_crop.json"
    crop_file.write_text(crop.to_json(seg_model))
    logits = personseg_on_npu(tensor)
    masks = person_matte.body_masks(logits, crop_file, size)

    # The head matte is 512x512 over the face crop; garment_clip wants it at
    # frame size, like every other alpha it is handed.
    left, top, side = crop_box
    head_full = Image.new("L", size, 0)
    head_full.paste(head_alpha.resize((side, side), Image.BILINEAR), (left, top))

    canvas, stats = garment_clip.compose(frame, warped, masks, head_full, stats=True)

    on_screen = bool(
        0 <= float(shoulder_l[1]) < size[1] and 0 <= float(shoulder_r[1]) < size[1]
    )
    span = float(np.linalg.norm(np.asarray(shoulder_r) - np.asarray(shoulder_l)))
    out = canvas.convert("RGB")
    if debug:
        draw_debug(out, face, shoulder_l, shoulder_r)
    return out, span, on_screen, stats




SKIN_CLASS = 1
NECK_CLASSES = (14, 15)  # neck, neck_l


def refine_neck(parse, rgb512: Image.Image, alpha: Image.Image) -> Image.Image:
    """Drop `neck` pixels that are not the colour of this person's skin.

    BiSeNet reliably labels a band of *wall* beside and below the jaw as
    `neck`. Geometry cannot separate it — it is contiguous with the real neck
    and sits in the same place — but colour can: the wall is nothing like skin.

    So measure the skin the parser is confident about (the `skin` class on the
    face), and keep only neck pixels within a tolerance of it. The tolerance is
    derived from that person's own skin spread rather than fixed, so it travels
    across lighting and complexions instead of being tuned to one room.
    """
    arr = np.asarray(rgb512, dtype=np.float32)
    skin = parse == SKIN_CLASS
    if skin.sum() < 500:
        return alpha  # too little to measure; leave it alone

    ref = np.median(arr[skin], axis=0)
    skin_dist = np.linalg.norm(arr[skin] - ref, axis=1)
    # 90th percentile of genuine skin variation, with floor and ceiling so a
    # flat or blown-out face cannot collapse or explode the threshold.
    tol = float(np.clip(np.percentile(skin_dist, 90) * 1.25, 40.0, 110.0))

    neck = np.isin(parse, NECK_CLASSES)
    if not neck.any():
        return alpha

    dist = np.linalg.norm(arr - ref, axis=2)
    drop = neck & (dist > tol)
    if not drop.any():
        return alpha

    # Only remove *large contiguous* off-colour regions. Beard and shadow under
    # the jaw are also unlike skin and also get labelled `neck`, but they come
    # as scattered speckle; the wall comes as one slab. Dropping by colour
    # alone punches holes through facial hair.
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        drop.astype(np.uint8), connectivity=8
    )
    if count > 1:
        min_area = max(400, int(0.004 * drop.size))
        keep_drop = np.zeros_like(drop)
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep_drop |= labels == i
        drop = keep_drop

    if not drop.any():
        return alpha

    a = np.asarray(alpha).copy()
    a[drop] = 0
    return Image.fromarray(a, mode="L")




# The oval the shopper lines up with, as fractions of the frame. The UI draws
# the same ellipse, so what they aim at and what is checked cannot drift apart.
GUIDE = {"cx": 0.50, "cy": 0.44, "rx": 0.16, "ry": 0.29}
# Face width as a fraction of frame width. Below the floor the parse is noise;
# above the ceiling the head runs out of the frame.
FACE_MIN, FACE_MAX = 0.28, 0.40
TRIGGER_WORDS = ("take a photo", "take photo", "take the photo", "cheese",
                 "snap", "take my photo", "take a picture")

# The modes that dress the live person rather than compositing them onto a
# product photo. They share a shutter, and the gate on it is whether the
# shoulders are in frame rather than check_alignment's oval — the oval is
# drawn for a head going onto a photo, and a try-on needs a torso, not a
# centred face.
TRYON_MODES = ("tryon", "tryon-clip")


# Roll past this reads as a tilted head in the composite; yaw is measured as
# how far the nose sits from the midpoint of the eyes, in units of eye
# separation, which is scale-free and so works at any distance.
MAX_ROLL_DEG = 10.0
MAX_YAW_RATIO = 0.25


def head_pose(lm):
    """(roll degrees, yaw ratio) from the three landmarks that carry them."""
    try:
        rl, ll, nose = lm["right_eye"], lm["left_eye"], lm["nose"]
    except (KeyError, TypeError):
        return None
    dx, dy = ll[0] - rl[0], ll[1] - rl[1]
    sep = math.hypot(dx, dy)
    if sep < 1e-3:
        return None
    mid_x, mid_y = (rl[0] + ll[0]) / 2.0, (rl[1] + ll[1]) / 2.0
    # Project the nose offset onto the eye line, so a tilted head is not also
    # read as a turned one.
    ux, uy = dx / sep, dy / sep
    along = (nose[0] - mid_x) * ux + (nose[1] - mid_y) * uy
    return math.degrees(math.atan2(dy, dx)), along / sep


def check_alignment(bbox, frame_size, head_touches_top: bool, landmarks=None):
    """Is the shopper standing where the composite will actually work?

    The same thresholds the UI's oval is drawn from, so the guidance matches
    what the shopper sees rather than being a second opinion.
    """
    fw, fh = frame_size
    x, y, w, h = bbox
    rel = w / float(fw)
    cx = (x + w / 2.0) / fw
    cy = (y + h / 2.0) / fh

    if head_touches_top:
        # Naming the camera, not the shopper: crouching does not fix a camera
        # pointed too low, and asking them to step back makes it worse by
        # shrinking the face.
        return False, "Tilt the camera up — top of head is out of frame"
    if rel < FACE_MIN:
        return False, "Come closer"
    if rel > FACE_MAX:
        return False, "Step back a little"
    if cx < GUIDE["cx"] - 0.15:
        return False, "Move right"
    if cx > GUIDE["cx"] + 0.15:
        return False, "Move left"
    if cy < GUIDE["cy"] - 0.19:
        return False, "Lower a little"
    if cy > GUIDE["cy"] + 0.19:
        return False, "Raise a little"

    # Pose last: it is the hardest to act on, so only ask for it once the
    # shopper is otherwise in the right place.
    pose = head_pose(landmarks) if landmarks else None
    if pose is not None:
        roll, yaw = pose
        if abs(yaw) > MAX_YAW_RATIO:
            return False, "Face the camera \u2014 line your nose up"
        if abs(roll) > MAX_ROLL_DEG:
            return False, "Straighten your head"
    return True, "Hold still — say \u201ctake a photo\u201d"


def voice_triggered(since: float) -> bool:
    """Has the mic heard the shutter phrase since `since`?

    Reads the transcript the Whisper daemon already writes rather than adding
    another channel between the two.
    """
    path = Path("/dev/shm/mic/transcript.jsonl")
    try:
        lines = path.read_text().splitlines()[-6:]
    except OSError:
        return False
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("ts", 0) <= since:
            continue
        said = entry.get("text", "").lower()
        if any(word in said for word in TRIGGER_WORDS):
            return True
    return False


def skin_tone(parse, rgb512: Image.Image):
    """Median colour of the pixels the parser is confident are facial skin."""
    arr = np.asarray(rgb512, dtype=np.float32)
    skin = parse == SKIN_CLASS
    if skin.sum() < 400:
        return None
    med = np.median(arr[skin], axis=0)
    return tuple(int(v) for v in med)


def shade(rgb, factor: float):
    return tuple(int(max(0, min(255, c * factor))) for c in rgb)


def _checkerboard(size, square: int = 24):
    """Transparency backdrop, so an alpha edge is visible as an alpha edge."""
    img = Image.new("RGB", size, (58, 62, 70))
    d = ImageDraw.Draw(img)
    for y in range(0, size[1], square):
        for x in range(0, size[0], square):
            if ((x // square) + (y // square)) % 2 == 0:
                d.rectangle([x, y, x + square, y + square], fill=(44, 48, 55))
    return img


def compose_segment(frame: Image.Image, alpha: Image.Image, crop_box, parse=None):
    """Show the segmentation itself: the head cutout, isolated.

    Billboard and try-on both *use* the matte but neither shows it, so a
    failure there looks like a compositing bug when it is really the parse.
    This mode puts the cutout on a checkerboard at frame size, so you can see
    exactly what BiSeNet produced — hair edges, ear boundaries, where the neck
    stops.
    """
    left, top, side = crop_box
    canvas = _checkerboard(frame.size).convert("RGBA")

    head = frame.crop((left, top, left + side, top + side)).convert("RGBA")
    head.putalpha(alpha.resize((side, side), Image.BILINEAR))
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    layer.paste(head, (left, top), head)
    canvas = Image.alpha_composite(canvas, layer)

    out = canvas.convert("RGB")
    d = ImageDraw.Draw(out, "RGBA")
    d.rectangle([left, top, left + side, top + side],
                outline=(94, 224, 160, 180), width=2)
    covered = int((np.asarray(alpha) > 8).sum())
    d.text((8, 8),
           f"BiSeNet head matte · crop {side}px · {covered / (512 * 512) * 100:.1f}% of crop",
           fill=(255, 255, 255, 235))
    return out


def write_state(**kw) -> None:
    (OUT_DIR / "state.json").write_text(json.dumps({"ts": time.time(), **kw}))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    board = Billboard(ASSETS)
    garment, garment_meta = load_garment()
    photo = load_photo_billboard()
    booth = FaceSwapBooth(load_ad_photo())
    smoother = Smoother(alpha=0.3)
    # Inputs of the last captured shot, so slider moves re-composite it.
    shot_cache: dict = {}
    last_box = None
    waited_for_still = 0
    if not CONTROL.exists():
        CONTROL.write_text(json.dumps({"mode": "billboard"}))
    log(f"billboard {board.size} brand={board.brand}; polling {CAM_DIR}")
    log(f"garment: {'loaded' if garment else 'none found in ' + str(GARMENTS)}")
    log(f"photo template: {'loaded ' + str(photo.size) if photo else 'none; using the drawn shirt'}")
    log(f"face swap: {booth.photo_path.name if booth.photo_path else 'no product photo'}"
        f" (models load on first use of the mode)")
    last_trigger_check = time.time()

    while True:
        loop_start = time.time()
        frame_path = newest_frame()
        if frame_path is None:
            write_state(ok=False, reason="no camera frame")
            time.sleep(INTERVAL)
            continue

        try:
            found = face_detect.detect(frame_path, score_threshold=SCORE)
        except Exception as exc:
            write_state(ok=False, reason=f"detector error: {exc}")
            time.sleep(INTERVAL)
            continue

        if not found:
            write_state(ok=False, reason="no face in frame")
            time.sleep(INTERVAL)
            continue

        try:
            img = Image.open(frame_path).convert("RGB")
            bbox = found["bbox"]

            ctl0 = read_control()

            if ctl0.get("mode") in TRYON_MODES and ctl0.get("phase") == "shot":
                # The composite is latched. Unlike the billboard's shot phase
                # there is nothing to retune from cache — the placement knobs
                # re-run the warp, which needs the body mask, so a slider move
                # is a retake. Serving the frozen frame costs neither NPU pass.
                write_state(ok=True, mode=ctl0["mode"], phase="shot",
                            faceWidth=int(bbox[2]))
                time.sleep(INTERVAL)
                continue

            if ctl0.get("mode") == "faceswap":
                phase = ctl0.get("phase", "live")
                if phase == "shot":
                    # Nothing to retune here — the placement is the
                    # advertisement's own — so just keep serving it.
                    time.sleep(INTERVAL)
                    continue

                if booth.swapper is None:
                    # First loop in this mode: the weights come off the disk
                    # and both graphs run once. ~2s on the GHOST/NPU path,
                    # ~8s on inswapper's 730 MB of ONNX. Either way the
                    # preview freezes, so claim it before it happens.
                    write_state(ok=True, mode="faceswap", phase="working",
                                guidance="Loading the face swapper…")
                loaded, why = booth.ensure_loaded()
                if not loaded:
                    write_state(ok=False, mode="faceswap", reason=why)
                    time.sleep(INTERVAL)
                    continue

                aligned, guidance = check_alignment(
                    bbox, img.size, head_touches_top=bbox[1] < 0.30 * bbox[3],
                    landmarks=found.get("landmarks"),
                )
                tmp = OUT_DIR / "latest.tmp.jpg"
                img.save(tmp, quality=86)
                tmp.replace(OUT_DIR / "latest.jpg")

                # The embedding describes the person, not the moment, so it
                # comes off the shutter and onto the time they spend getting
                # into the frame. Only while aligned: the live branch stays
                # cheap so the preview keeps up with them.
                if aligned:
                    booth.refresh_latent(np.asarray(img)[:, :, ::-1].copy(), found)

                fire = aligned and booth.latent is not None and (
                    bool(ctl0.get("capture")) or voice_triggered(last_trigger_check)
                )
                last_trigger_check = time.time()

                if not fire:
                    write_state(ok=True, mode="faceswap", phase="live",
                                aligned=aligned, guidance=guidance,
                                identified=booth.latent is not None,
                                faceWidth=int(bbox[2]))
                    time.sleep(INTERVAL)
                    continue

                # The swap blocks this loop and the preview freezes for all
                # of it — ~1 s on the GHOST/NPU path (82 ms of compute, the
                # rest reloading the .pte through the one-shot runner), ~4 s
                # on inswapper's CPU. Say so before starting rather than
                # after: a state written afterwards arrives too late.
                write_state(ok=True, mode="faceswap", phase="working",
                            guidance="Making your billboard…",
                            faceWidth=int(bbox[2]))
                log("shutter: swapping")

                t0 = time.time()
                composed = booth.shoot()
                swap_ms = (time.time() - t0) * 1000
                composed.save(tmp, quality=90)
                tmp.replace(OUT_DIR / "latest.jpg")

                ctl0["phase"] = "shot"
                ctl0["capture"] = False
                CONTROL.write_text(json.dumps(ctl0))
                write_state(ok=True, mode="faceswap", phase="shot",
                            swapMs=round(swap_ms, 1), faceWidth=int(bbox[2]))
                log(f"swap {swap_ms:.0f}ms")
                continue

            if ctl0.get("mode") == "generate":
                phase = ctl0.get("phase", "live")
                if phase == "shot":
                    # The generated composite is already written; serving it
                    # is all there is to do. Nothing here is retunable from a
                    # slider — re-running means another ~20 s of diffusion.
                    time.sleep(INTERVAL)
                    continue

                catalogue = board_tryon_gen.load_catalogue(PROMPTS)
                slug = ctl0.get("garment") or next(iter(catalogue), None)
                if not catalogue or slug not in catalogue:
                    write_state(ok=False, mode="generate",
                                reason=f"no tokenized prompts under {PROMPTS}")
                    time.sleep(INTERVAL)
                    continue

                tmp = OUT_DIR / "latest.tmp.jpg"
                img.save(tmp, quality=86)
                tmp.replace(OUT_DIR / "latest.jpg")

                # Framing matters more here than in any other mode. The
                # conditioning is the centre square of the frame, so if the
                # shopper is close enough to fill it with their face there is
                # no torso left to put a garment on — and ControlNet will
                # faithfully generate a close-up of a face in a room. Reuse
                # the same gate and the same wording as the other posed modes.
                aligned, guidance = board_tryon_gen.framing_guidance(bbox, img.size)
                asked = bool(ctl0.get("capture"))
                if asked and not aligned:
                    # Asked for a shot from a pose that cannot produce one.
                    # Drop it rather than spending 26 s — but say so, with the
                    # numbers: a silently ignored shutter reads as a broken
                    # button, which is exactly how it was first reported.
                    ctl0["capture"] = False
                    CONTROL.write_text(json.dumps(ctl0))
                    want = int(board_tryon_gen.FACE_MAX * img.size[0])
                    guidance = (
                        f"{guidance} — your face is {int(bbox[2])}px wide, "
                        f"needs to be under {want}px"
                    )
                    log(f"capture declined: face {int(bbox[2])}px, want <{want}px")
                if not asked or not aligned:
                    write_state(ok=True, mode="generate", phase="live",
                                garment=slug, garmentLabel=catalogue[slug]["label"],
                                garments={k: v["label"] for k, v in catalogue.items()},
                                faceWidth=int(bbox[2]), aligned=aligned,
                                declined=asked and not aligned,
                                guidance=guidance if not aligned
                                else "Pick a garment and press the shutter.")
                    time.sleep(INTERVAL)
                    continue

                # Four graphs a step, two steps a second — the preview is
                # frozen for all of it. Claim it before starting, the way the
                # face-swap branch does: a state written afterwards is too late.
                write_state(ok=True, mode="generate", phase="working",
                            garment=slug, garmentLabel=catalogue[slug]["label"],
                            guidance=f"Generating {catalogue[slug]['label']}…")
                log(f"shutter: generating {slug}")

                try:
                    # The conditioning is built from the centre square of the
                    # frame, so the head matte has to come from that same
                    # square or the paste lands in the wrong place.
                    src512, _ = board_tryon_gen.square_512(img)
                    TRYON_WORK.mkdir(parents=True, exist_ok=True)
                    square_path = TRYON_WORK / "source_512.png"
                    src512.save(square_path)
                    found512 = face_detect.detect(square_path, score_threshold=SCORE)
                    if not found512:
                        raise RuntimeError("face left the frame")

                    bbox512 = found512["bbox"]
                    parse = face_matte.parse_map(parse_on_npu(preprocess(src512, bbox=bbox512)))
                    alpha = face_matte.alpha_from_parse(parse, feather=2.0, erode=1, seed=None)
                    crop_img, crop_box = square_crop(src512, bbox512)
                    head = crop_img.resize((512, 512), Image.BICUBIC).convert("RGBA")
                    head.putalpha(alpha)

                    # Mask the conditioning to the person. Without this the
                    # room's edges are conditioning too, and the model
                    # generates the room.
                    seg_model = SEG_MODELS["segformer-b0-atr"]
                    seg_tensor, seg_crop = make_person_raw.prepare(
                        src512, seg_model, face=found512
                    )
                    PERSONSEG_WORK.mkdir(parents=True, exist_ok=True)
                    seg_crop_file = PERSONSEG_WORK / "person_crop.json"
                    seg_crop_file.write_text(seg_crop.to_json(seg_model))
                    silhouette = person_matte.body_masks(
                        personseg_on_npu(seg_tensor), seg_crop_file, src512.size
                    ).silhouette

                    def segment_generated(generated, _model=seg_model):
                        """Silhouette of the GENERATED image, not the frame.

                        A second pass rather than reusing `silhouette` above:
                        the generation follows the real outline but does not
                        reproduce it, and a looser garment legitimately sits
                        outside the body that conditioned it. Cutting with the
                        real frame's mask would shave exactly the part of the
                        product worth seeing. ~25 ms against the generator's
                        seconds.

                        Whole-frame, with no detector: the square is already a
                        half-body shot, and the generated face is a stranger's
                        so there is nothing gained by finding it.
                        """
                        tensor, crop_meta = make_person_raw.prepare(
                            generated, _model, full_frame=True
                        )
                        meta_file = PERSONSEG_WORK / "generated_crop.json"
                        meta_file.write_text(crop_meta.to_json(_model))
                        masks = person_matte.body_masks(
                            personseg_on_npu(tensor), meta_file,
                            generated.size, with_head=True,
                        )
                        # Knock the generated head out of the silhouette. The
                        # real head is pasted back over this region a moment
                        # later, and the model's idea of someone's hair is
                        # both darker and larger than the real thing — keep it
                        # and whatever sticks out past BiSeNet's matte reads
                        # as a shadow glued to the side of their head.
                        body = (np.asarray(masks.silhouette, np.int16)
                                - np.asarray(masks.head, np.int16))
                        return Image.fromarray(
                            np.clip(body, 0, 255).astype(np.uint8), mode="L"
                        )

                    composed, measured = board_tryon_gen.run(
                        img, head, crop_box, PROMPTS / slug, TRYON_WORK,
                        silhouette=silhouette,
                        segment=segment_generated,
                        python=sys.executable, driver=str(SD_DRIVER),
                        model_dir=str(CN_MODEL_DIR), runner=str(RUNNER), env=NPU_ENV,
                        steps=int(ctl0.get("steps", 8)), seed=int(ctl0.get("seed", 7)),
                    )
                except Exception as exc:
                    log(f"generate failed: {exc}")
                    write_state(ok=False, mode="generate", reason=str(exc))
                    ctl0["capture"] = False
                    CONTROL.write_text(json.dumps(ctl0))
                    time.sleep(INTERVAL)
                    continue

                composed.save(tmp, quality=90)
                tmp.replace(OUT_DIR / "latest.jpg")
                ctl0["phase"] = "shot"
                ctl0["capture"] = False
                CONTROL.write_text(json.dumps(ctl0))
                write_state(ok=True, mode="generate", phase="shot", garment=slug,
                            garmentLabel=catalogue[slug]["label"],
                            generateS=measured.get("total_s"),
                            stepS=measured.get("seconds_per_step_npu"))
                log(f"generate {slug} {measured.get('total_s')}s")
                continue

            if ctl0.get("mode", "billboard") == "billboard":
                phase = ctl0.get("phase", "live")

                if phase == "shot":
                    # The composite is already written; keep serving it rather
                    # than re-running the NPU on a frame nobody is looking at.
                    # But if they move a placement slider, re-composite from
                    # the cached parse: placement is pure CPU, so tuning the
                    # shot costs ~200ms and no NPU round trip.
                    knobs = (round(float(ctl0.get("headScale", 1.12)), 3),
                             round(float(ctl0.get("fineX", 0.0)), 1),
                             round(float(ctl0.get("fineY", 36.0)), 1))
                    if shot_cache and knobs != shot_cache.get("knobs"):
                        t0 = time.time()
                        smoother.reset()
                        again, info = photo.compose(
                            shot_cache["crop"], shot_cache["parse"],
                            HEAD_CLASSES, NECK_IDX,
                            shot_cache["eye_l"], shot_cache["eye_r"], smoother,
                            face_box=shot_cache["fb"],
                            landmarks=shot_cache.get("lm"),
                            offsets={"dx": knobs[1], "dy": knobs[2],
                                     "headScale": knobs[0],
                                     "neckBlend": bool(ctl0.get("neckBlend", True)),
                        "neckFade": float(ctl0.get("neckFade", 1.0)),
                        "skin_idx": SKIN_IDX},
                        )
                        if again is not None:
                            tmp = OUT_DIR / "latest.tmp.jpg"
                            again.save(tmp, quality=90)
                            tmp.replace(OUT_DIR / "latest.jpg")
                            shot_cache["knobs"] = knobs
                            # Carry the measurements from the capture. The
                            # retune re-composites but does not re-parse, so
                            # recomputing them is impossible and omitting them
                            # blanks the stats panel to NaN.
                            write_state(ok=True, mode="billboard", phase="shot",
                                        retuneMs=round((time.time() - t0) * 1000, 1),
                                        **{**shot_cache["stats"], **info})
                    time.sleep(INTERVAL)
                    continue

                # Live framing. No parse here — alignment only needs the face
                # box, which is ~9ms of CPU against ~400ms for the parse, so
                # the preview stays responsive while they position themselves.
                aligned, guidance = check_alignment(
                    bbox, img.size, head_touches_top=bbox[1] < 0.30 * bbox[3],
                    landmarks=found.get("landmarks"),
                )
                # Surface the pose the gate is already measuring. Being told
                # "straighten your head" without being shown how far off you
                # are is guesswork; the number converges in about a second.
                pose = head_pose(found.get("landmarks"))
                tmp = OUT_DIR / "latest.tmp.jpg"
                img.save(tmp, quality=86)
                tmp.replace(OUT_DIR / "latest.jpg")

                # A button press is a deliberate choice and always fires; the
                # voice shutter still waits for alignment, because "take a
                # photo" said while walking past should not trigger it.
                fire = bool(ctl0.get("capture")) or (
                    aligned and voice_triggered(last_trigger_check)
                )
                last_trigger_check = time.time()

                # Don't shoot mid-motion. The parse and the matte are both
                # computed from one frame, so a frame caught while someone is
                # turning produces a smeared head that looks like a blending
                # bug rather than motion blur. Wait for the face box to settle,
                # but not forever — after a couple of seconds, take what we
                # have rather than appearing to ignore the shutter.
                box_now = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0, bbox[2])
                moved = (
                    max(abs(a - b) for a, b in zip(box_now, last_box))
                    if last_box else 1e9
                )
                last_box = box_now
                if fire and moved > 9 and waited_for_still < 3:
                    waited_for_still += 1
                    write_state(ok=True, mode="billboard", phase="live",
                                aligned=aligned, guidance="Hold still\u2026",
                                faceWidth=int(bbox[2]))
                    time.sleep(INTERVAL)
                    continue
                waited_for_still = 0

                if not fire:
                    write_state(ok=True, mode="billboard", phase="live",
                                aligned=aligned, guidance=guidance,
                                faceWidth=int(bbox[2]),
                                roll=round(pose[0], 1) if pose else None,
                                yaw=round(pose[1], 3) if pose else None,
                                # Normalised so the overlay can place the
                                # wireframe on the actual head without caring
                                # what resolution the camera happens to be.
                                faceCx=round((bbox[0] + bbox[2] / 2) / img.size[0], 4),
                                faceCy=round((bbox[1] + bbox[3] / 2) / img.size[1], 4),
                                faceRel=round(bbox[2] / img.size[0], 4))
                    time.sleep(INTERVAL)
                    continue

                # Falling through runs the full parse and composite once.
                smoother.reset()
                log("shutter: capturing")
            tensor = preprocess(img, bbox=bbox)
            # square_crop hands back the cropped image and the (left, top,
            # side) it used, so there is no need to crop a second time.
            crop_img, (crop_left, crop_top, crop_side) = square_crop(img, bbox)

            t0 = time.time()
            logits = parse_on_npu(tensor)
            npu_sec = time.time() - t0

            parse = face_matte.parse_map(logits)
            alpha = face_matte.alpha_from_parse(parse, feather=2.0, erode=1, seed=None)

            # The parse is in 512x512 crop space; build the cutout there, then
            # map the landmarks into the same space.
            crop512 = crop_img.resize((512, 512), Image.BICUBIC)
            # Opt-in: dropping `neck` pixels that are unlike skin removes the
            # wall the parser mislabels, but it also eats genuinely-lit neck
            # when the neck is shaded differently from the face, leaving the
            # background showing through. Off unless explicitly enabled.
            if read_control().get("refineNeck", False):
                alpha = refine_neck(parse, crop512, alpha)
            head = crop512.convert("RGBA")
            head.putalpha(alpha)

            k = 512.0 / crop_side
            to_crop = lambda p: ((p[0] - crop_left) * k, (p[1] - crop_top) * k)  # noqa: E731
            # YuNet names eyes from the subject's point of view, so
            # "right_eye" is the one on the left of the image. Taking them in
            # that order makes dx positive and the tilt angle come out right.
            lm = found["landmarks"]
            lm_crop = {k: to_crop(v) for k, v in lm.items()}
            eye_l = to_crop(lm["right_eye"])
            eye_r = to_crop(lm["left_eye"])

            mode = read_control().get("mode", "billboard")
            extra = {}
            if mode == "segment":
                composed = compose_segment(
                    img, alpha, (crop_left, crop_top, crop_side)
                )
            elif mode == "tryon" and garment is not None:
                ctl = read_control()
                composed, shoulder_px, on_screen = compose_tryon(
                    img, found, alpha, (crop_left, crop_top, crop_side),
                    garment, garment_meta,
                    drop=float(ctl.get("drop", 0.45)),
                    half_width=float(ctl.get("halfWidth", 1.15)),
                    offset_y=float(ctl.get("offsetY", 0.0)),
                    debug=bool(ctl.get("debug", False)),
                )
                extra = {
                    "shoulderPx": int(round(shoulder_px)),
                    "shouldersOnScreen": on_screen,
                }
            elif mode == "tryon-clip" and garment is not None:
                ctl = read_control()
                seg0 = time.time()
                composed, shoulder_px, on_screen, clip = compose_tryon_clipped(
                    img, found, alpha, (crop_left, crop_top, crop_side),
                    garment, garment_meta,
                    drop=float(ctl.get("drop", 0.45)),
                    half_width=float(ctl.get("halfWidth", 1.15)),
                    offset_y=float(ctl.get("offsetY", 0.0)),
                    debug=bool(ctl.get("debug", False)),
                )
                extra = {
                    "shoulderPx": int(round(shoulder_px)),
                    "shouldersOnScreen": on_screen,
                    "clipKept": round(clip.kept, 1),
                    "armsOccluded": clip.occluded_px,
                    "segMs": round((time.time() - seg0) * 1000, 1),
                }
            elif photo is not None:
                mode = "billboard"
                # Head above the top of the camera frame means the crop is
                # clamped and the hair is simply not in the picture; no amount
                # of compositing recovers it.
                head_only = photo.head_mask(parse, HEAD_CLASSES, NECK_IDX)
                if crop_top <= 1 and head_only[0].any():
                    smoother.reset()
                    write_state(ok=False, mode=mode, reason="Step back a little")
                    time.sleep(INTERVAL)
                    continue
                # The opposite failure, and the more common one: too far away.
                # Below roughly 110px of face the parse is noise, and
                # compositing it produces a torn, blurry head rather than an
                # obviously-wrong one, which is worse because it looks like a
                # bug in the blending.
                if int(bbox[2]) < 110:
                    smoother.reset()
                    write_state(ok=False, mode=mode,
                                reason="Come closer — face too small to segment",
                                faceWidth=int(bbox[2]))
                    time.sleep(INTERVAL)
                    continue

                cpu0 = time.time()
                # A capture is one frame, so place it from the measurement
                # rather than from an average over frames that do not exist.
                smoother.reset()
                # Face box in crop coordinates, so scale and anchor compare
                # like for like with the model's own box.
                kk = 512.0 / crop_side
                fb = [(bbox[0] - crop_left) * kk, (bbox[1] - crop_top) * kk,
                      bbox[2] * kk, bbox[3] * kk]
                composed, shot = photo.compose(
                    crop512, parse, HEAD_CLASSES, NECK_IDX, eye_l, eye_r, smoother,
                    face_box=fb,
                    landmarks=lm_crop,
                    offsets={
                        "dx": float(ctl0.get("fineX", 0.0)),
                        "dy": float(ctl0.get("fineY", 36.0)),
                        "headScale": float(ctl0.get("headScale", 1.12)),
                        "neckBlend": bool(ctl0.get("neckBlend", True)),
                        "neckFade": float(ctl0.get("neckFade", 1.0)),
                        "skin_idx": SKIN_IDX,
                    },
                )
                if composed is None:
                    write_state(ok=False, mode=mode, reason=shot.get("reason", "compose failed"))
                    time.sleep(INTERVAL)
                    continue
                extra = {**shot, "cpuMs": round((time.time() - cpu0) * 1000, 1),
                         "phase": "shot"}
                # Keep the compose inputs so the sliders can re-render this
                # exact shot without going back to the camera or the NPU.
                shot_cache.clear()
                shot_cache.update({
                    "crop": crop512, "parse": parse, "fb": fb, "lm": lm_crop,
                    "eye_l": eye_l, "eye_r": eye_r,
                    "faceWidth": int(bbox[2]),
                    "stats": {k: v for k, v in extra.items()
                              if k not in ("phase",)},
                    "knobs": (round(float(ctl0.get("headScale", 1.12)), 3),
                              round(float(ctl0.get("fineX", 0.0)), 1),
                              round(float(ctl0.get("fineY", 36.0)), 1)),
                })
                # Latch, and clear the one-shot capture flag so it does not
                # immediately re-fire on the next loop.
                ctl0["phase"] = "shot"
                ctl0["capture"] = False
                CONTROL.write_text(json.dumps(ctl0))
            else:
                mode = "billboard"
                composed = board.compose(
                    head, eye_l, eye_r, skin_rgb=skin_tone(parse, crop512)
                )

            tmp = OUT_DIR / "latest.tmp.jpg"
            composed.save(tmp, quality=86)
            tmp.replace(OUT_DIR / "latest.jpg")

            if mode in TRYON_MODES:
                # Latch after the frame is on disk, so the shot the UI freezes
                # is the one that was just written rather than the next one.
                ctl_now = read_control()
                # The button is a deliberate choice and always fires; the voice
                # shutter waits for the shoulders, because a try-on shot with
                # the garment off the bottom of the frame is not worth keeping
                # and "take a photo" carries across a room.
                fire = bool(ctl_now.get("capture")) or (
                    bool(extra.get("shouldersOnScreen"))
                    and voice_triggered(last_trigger_check)
                )
                last_trigger_check = time.time()
                if fire:
                    ctl_now["phase"] = "shot"
                    ctl_now["capture"] = False
                    CONTROL.write_text(json.dumps(ctl_now))
                    extra["phase"] = "shot"
                else:
                    extra["guidance"] = (
                        "Say “take a photo”"
                        if extra.get("shouldersOnScreen")
                        else "Step back — your shoulders are out of frame"
                    )

            head_px = int((np.asarray(alpha) > 8).sum())
            write_state(
                ok=True,
                mode=mode,
                npuMs=round(npu_sec * 1000, 1),
                totalMs=round((time.time() - loop_start) * 1000, 1),
                headCoverage=round(head_px / (512 * 512) * 100, 1),
                faceWidth=int(bbox[2]),
                brand=board.brand,
                **extra,
            )
            log(f"npu {npu_sec*1000:.0f}ms  face {bbox[2]}px  head {head_px/(512*512)*100:.1f}%")
        except Exception as exc:
            write_state(ok=False, reason=str(exc))
            log(f"compose failed: {exc}")

        slack = INTERVAL - (time.time() - loop_start)
        if slack > 0:
            time.sleep(slack)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
