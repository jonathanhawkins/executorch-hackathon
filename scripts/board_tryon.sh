#!/usr/bin/env bash
# Put a garment on the person in a frame — the billboard effect, end to end.
#
#   ./scripts/board_tryon.sh frame.jpg                  # overlay only, no NPU
#   OCCLUDE=1 ./scripts/board_tryon.sh frame.jpg        # + head matte on top
#   CLIP=1 ./scripts/board_tryon.sh frame.jpg           # + body clip and arms
#   OCCLUDE=1 CLIP=1 ./scripts/board_tryon.sh frame.jpg # both
#   GARMENT=assets/garments/other.png ./scripts/board_tryon.sh frame.jpg
#
# Without OCCLUDE this is pure CPU and needs nothing on the board: YuNet finds
# the face, the garment is warped onto the estimated shoulders, done. With it,
# the face parser runs on the Hexagon and the head is drawn back over the
# garment, so hair across a shoulder stays in front of the shirt.
#
# CLIP adds the second NPU model: person segmentation cuts the garment to the
# wearer's outline, so it stops hanging over the background beside a shoulder,
# and puts arms and hands back in front of it. Both flags are opt-in and
# independent — the default behaviour of this script has not changed, and
# scripts/board_faceparse_run.sh still works on its own for the head cutout.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${1:?usage: board_tryon.sh <image>}"
OUT="${OUT:-tryon.png}"
GARMENT="${GARMENT:-assets/garments/placeholder-tee.png}"
OCCLUDE="${OCCLUDE:-}"
CLIP="${CLIP:-}"
COVER="${COVER:-}"

MATTE_ARGS=()
if [ -n "$CLIP" ]; then
  echo "==> Person segmentation on the NPU (for clipping and arms)"
  # The body mask is the by-product we want; its own cutout goes to the bin.
  OUT=.personseg_stage/body.png COVER="$COVER" \
    ./scripts/board_personseg_run.sh "$IMAGE" >/dev/null 2>&1 || {
    echo "    person segmentation failed — continuing without clipping" >&2
    CLIP=""
  }
  if [ -n "$CLIP" ]; then
    MATTE_ARGS+=(--body .personseg_stage/run/output_0.raw)
  fi
fi

if [ -n "$OCCLUDE" ]; then
  echo "==> Face parsing on the NPU (for occlusion)"
  # Its own cutout is a by-product here; we only want the logits it leaves
  # behind in .faceparse_stage/run/.
  OUT=.faceparse_stage/head.png ./scripts/board_faceparse_run.sh "$IMAGE" >/dev/null 2>&1 || {
    echo "    face parse failed — continuing without occlusion" >&2
    OCCLUDE=""
  }
  if [ -n "$OCCLUDE" ]; then
    MATTE_ARGS+=(--matte .faceparse_stage/run/output_0.raw)
  fi
fi

echo "==> Compositing"
.venv/bin/python scripts/garment_overlay.py "$IMAGE" \
  --garment "$GARMENT" \
  -o "$OUT" \
  "${MATTE_ARGS[@]}" | sed 's/^/    /'
