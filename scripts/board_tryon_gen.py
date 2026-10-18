#!/usr/bin/env python3
"""Generate a garment onto the live camera frame, keeping the shopper's head.

Runs ON THE BOARD, called by board_billboard_daemon.py's `generate` mode. The
Mac-side equivalent is scripts/board_cn_tryon.sh, which orchestrates the same
three stages over adb; this is the same pipeline with the adb round trips
taken out, because the daemon is already on the board.

    1. Canny-edge the camera frame -> ControlNet's conditioning input.
    2. ControlNet + UNet + VAE generate a body wearing the described garment,
       following those edges, so the pose and build are the shopper's.
    3. The real head goes back on. Canny carries shape, not identity, so the
       generated face is a stranger's — but because the generation was
       conditioned on this very frame, the generated head is already in the
       right place at the right scale and the head is a paste, not a warp.

Prompts are pre-tokenized (scripts/make_tryon_prompts.py) rather than
tokenized here: a shop has a garment catalogue, not a free-text box, and it
keeps CLIP's vocabulary files off the board.

See docs/stable-diffusion-npu.md.
"""

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

SIZE = 512  # the context binaries are built for 512x512 and nothing else

# Framing thresholds, as a fraction of the SQUARE that actually gets
# conditioned — not of the raw frame, which is the subtlety. These are the
# OPPOSITE of the billboard's: that mode wants a big head to cut out, so it
# asks people to come close (FACE_MIN 0.28). This one needs a half-body shot,
# because a face that fills the square leaves no torso to dress and ControlNet
# will faithfully generate a close-up of a face in a room.
#
# 0.26 is the framing of the photo that produced the first good result, so the
# window is set around it.
FACE_MIN, FACE_MAX = 0.10, 0.32
# How much frame has to remain below the head for there to be a garment to
# generate, as a multiple of head height. Measured against a real desk-mounted
# camera: a seated person at a comfortable distance leaves ~0.93x, which is
# collar and upper chest — enough for a t-shirt. Asking for more than that
# asks for a camera position the hardware does not have.
TORSO_MIN = 0.8


def square_side(frame_size) -> int:
    """The side of the square the frame becomes. See square_512."""
    return max(frame_size)


def framing_guidance(bbox, frame_size) -> tuple[bool, str]:
    """Is there room below the head for a garment? Wording for the shopper."""
    side = square_side(frame_size)
    x, y, w, h = bbox
    rel = w / float(side)
    # What matters is not where the head sits in the square but how much of
    # the shopper is below it — that is the surface the garment goes on.
    # Measuring the square instead counts the letterbox bars as torso.
    torso = frame_size[1] - (y + h)
    if rel > FACE_MAX:
        return False, "Step back — we need your shoulders and chest in frame"
    if rel < FACE_MIN:
        return False, "Come a little closer"
    if torso < TORSO_MIN * h:
        return False, "Tilt the camera up — we need your chest in frame"
    return True, "Pick a garment and press the shutter."


def square_512(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int]]:
    """Letterbox to a square and resize. Returns the box the frame landed in.

    Padding rather than centre-cropping, because cropping a 640x480 frame to
    480x480 throws away a third of the width and forces the shopper much
    further back than a desk-mounted camera allows — measured: five declined
    captures plateauing at a 155px face against a 128px ceiling. Padding keeps
    the whole frame, so the same person at the same distance is 24% of the
    square instead of 32%.

    The bars cost nothing downstream: the conditioning is masked to the
    person's silhouette, so they carry no Canny edges.
    """
    width, height = img.size
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    square = Image.new("RGB", (side, side), (128, 128, 128))
    square.paste(img, (left, top))
    return square.resize((SIZE, SIZE), Image.BICUBIC), (left, top, side)


def write_conditioning(
    src512: Image.Image, work: Path, low=100, high=200, blur=3, silhouette=None
) -> float:
    """Canny edges -> image_cond.raw. Returns the edge density.

    `silhouette` is the whole point on a real camera frame. ControlNet follows
    every edge it is given, and a shop floor or a living room is full of them:
    conditioned on a raw frame it faithfully generates the room, picture
    frames and ceiling light included. Masking the edges to the person's
    outline leaves only the body to follow, which is all we wanted from it.
    """
    bgr = np.asarray(src512)[:, :, ::-1].copy()
    if blur:
        bgr = cv2.GaussianBlur(bgr, (blur | 1, blur | 1), 0)
    edges = cv2.Canny(bgr, low, high)
    if silhouette is not None:
        mask = np.asarray(silhouette.resize((SIZE, SIZE), Image.BILINEAR))
        # Keep the outline itself: dilating a little stops the silhouette's
        # own boundary being shaved off with the background behind it.
        mask = cv2.dilate((mask > 127).astype(np.uint8), np.ones((5, 5), np.uint8))
        edges = edges * mask
    conditioning = np.repeat(edges[:, :, None], 3, axis=2).astype(np.uint8)
    conditioning.reshape(1, SIZE, SIZE, 3).tofile(work / "image_cond.raw")
    return float(np.count_nonzero(edges)) / edges.size


def load_catalogue(prompts_dir: Path) -> dict:
    """The pre-tokenized garments, keyed by slug."""
    index = prompts_dir / "prompts.json"
    if not index.exists():
        return {}
    return json.loads(index.read_text())


def generate_body(
    work: Path,
    prompt_dir: Path,
    *,
    python: str,
    driver: str,
    model_dir: str,
    runner: str,
    env: dict,
    steps: int = 8,
    seed: int = 7,
    guidance: float = 7.5,
) -> tuple[Image.Image, dict]:
    """Run the four ControlNet graphs. Blocks for ~20 s at 8 steps."""
    for name in ("tokens_cond.raw", "tokens_uncond.raw"):
        (work / name).write_bytes((prompt_dir / name).read_bytes())

    output = work / "body.png"
    stats = work / "body_stats.json"
    command = [
        python, driver,
        "--model_dir", model_dir,
        "--prefix", "cn15",
        "--tokens_dir", str(work),
        "--runner", runner,
        "--workdir", str(work / "sd"),
        "--output", str(output),
        "--stats", str(stats),
        "--num_time_steps", str(steps),
        "--guidance_scale", str(guidance),
        "--seed", str(seed),
        "--prediction_type", "epsilon",
        # The benchmark is diagnostics and costs another model load; the
        # shopper is standing there.
        "--bench_iterations", "0",
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or not output.exists():
        tail = "\n".join(result.stderr.strip().split("\n")[-8:])
        raise RuntimeError(f"generation failed ({result.returncode})\n{tail}")

    measured = json.loads(stats.read_text()) if stats.exists() else {}
    return Image.open(output).convert("RGB"), measured


# Not pure white. 255 against the page's dark stage reads as a hole punched in
# the panel, and it blows out the shoulder of a white garment so the product
# loses its own edge. A light warm grey is what a lookbook actually shoots on.
STUDIO_GROUND = (247, 246, 243)


def on_studio_ground(body: Image.Image, mask: Image.Image,
                     ground=STUDIO_GROUND, feather: float = 1.0,
                     erode: int = 2) -> Image.Image:
    """Drop whatever the model invented behind the body and stand it on a flat ground.

    Masking the background away is much less work than restoring the camera's
    own background, and for this mode it is also the better answer. The
    generated garment is lit by light the model imagined; the room is lit by
    the room. Composite the two and every boundary has to be colour-matched or
    it shows. A flat ground has no lighting to disagree with, so the mismatch
    has nothing to be measured against — and a plain sweep is what a product
    shot looks like anyway, which is the mode's whole premise.

    The try-on modes are the ones that keep the real room; see
    docs/personseg-npu.md. This one is the studio.
    """
    alpha = mask.resize(body.size, Image.BILINEAR)
    if erode:
        # Pull the edge in before feathering it. A half-transparent edge pixel
        # still carries the colour of what it was sitting on, and what it was
        # sitting on is the dark scene the model invented — feather without
        # eroding and the body gets a grey halo against the light ground,
        # which reads as a bad cut-out rather than a photograph. Same reason
        # face_matte erodes a pixel before blurring.
        alpha = alpha.filter(ImageFilter.MinFilter(2 * erode + 1))
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    out = Image.new("RGB", body.size, tuple(ground))
    out.paste(body.convert("RGB"), (0, 0), alpha)
    return out


def restore_head(
    body: Image.Image,
    head_rgba: Image.Image,
    crop: tuple[int, int, int],
    *,
    feather: float = 2.0,
) -> Image.Image:
    """Paste the real head back at the position the conditioning implies.

    `crop` is the (left, top, side) that the head cut-out was taken from, in
    the same 512-space the generation was conditioned on — which is the whole
    reason this is a paste and not a registration problem.
    """
    left, top, side = crop
    head = head_rgba.resize((side, side), Image.BICUBIC)
    alpha = head.getchannel("A")
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    out = body.copy()
    out.paste(head.convert("RGB"), (left, top), alpha)
    return out


def run(
    frame: Image.Image,
    head_rgba: Image.Image,
    crop: tuple[int, int, int],
    prompt_dir: Path,
    work: Path,
    segment=None,
    ground=STUDIO_GROUND,
    **kwargs,
) -> tuple[Image.Image, dict]:
    """The whole thing: frame in, shopper-wearing-the-garment out.

    `segment` is an optional callable taking the generated image and returning
    an "L" silhouette for it. Pass one and the invented background is replaced
    with a flat studio ground; leave it None and the scene the model painted
    is kept. The NPU plumbing stays with the caller, which owns the runner.
    """
    work.mkdir(parents=True, exist_ok=True)
    src512, _ = square_512(frame)
    started = time.time()
    density = write_conditioning(
        src512, work,
        low=kwargs.pop("canny_low", 100),
        high=kwargs.pop("canny_high", 200),
        silhouette=kwargs.pop("silhouette", None),
    )
    body, measured = generate_body(work, prompt_dir, **kwargs)
    if segment is not None:
        # Before the head goes back, not after. The head carries BiSeNet's
        # 512x512 matte and the body carries a mask enlarged from 128x128 —
        # cutting the finished composite with the coarser of the two would
        # throw away the hairline, which is the edge anyone actually looks at.
        body = on_studio_ground(body, segment(body), ground)
        measured["ground"] = "studio"
    composed = restore_head(body, head_rgba, crop)
    measured["edge_density"] = round(density, 4)
    measured["total_s"] = round(time.time() - started, 1)
    return composed, measured
