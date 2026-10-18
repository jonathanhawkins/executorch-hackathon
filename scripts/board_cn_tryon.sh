#!/usr/bin/env bash
# Put a real person into a described garment, on the board, end to end.
#
#   ./scripts/board_cn_tryon.sh frame.jpg "a bright red crew neck t-shirt"
#   STEPS=8 ./scripts/board_cn_tryon.sh frame.jpg "a white linen shirt"
#
# Three stages, all on the Hexagon:
#   1. ControlNet-Canny generates a body wearing the described garment,
#      following the photo's edges — so the pose, framing and silhouette are
#      the real person's (docs/stable-diffusion-npu.md).
#   2. BiSeNet cuts the real head out of the same frame (docs/faceparse-npu.md).
#   3. ATR SegFormer labels the generated body, so the skin-tone correction
#      can tell an arm from a shirt (a chrominance guess cannot — skin and a
#      red garment share a Cr range). NO_SEG=1 skips it.
#   4. The head goes back on. Because stage 1 was conditioned on this very
#      photo, the generated head is already in the right place at the right
#      scale, so this is a paste rather than a warp.
#
# What that buys: the garment is generated and the face is genuinely theirs.
# ControlNet alone gets the clothes right and the face wrong; this fixes the
# face by not generating it.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${1:?usage: board_cn_tryon.sh <photo> [garment description]}"
GARMENT="${2:-a navy blue crew neck t-shirt}"
OUT="${OUT:-tryon.png}"
STEPS="${STEPS:-20}"
SEED="${SEED:-7}"
STAGE=.sd_stage

PROMPT="${PROMPT:-studio photo of a person wearing $GARMENT, plain background, soft lighting}"

echo "  photo    $IMAGE"
echo "  garment  $GARMENT"
echo

# 1. Generate the body. This also writes $STAGE/source_512.png — the exact
#    512x512 crop the conditioning was built from, which is the frame the head
#    matte has to be computed on for the two to line up.
echo "==> 1/4  ControlNet: generating the garment on this pose"
MODEL_DIR=/home/arduino/models/controlnet-npu PREFIX=cn15 BASE=sd15 \
  PREDICTION=epsilon IMAGE="$IMAGE" STEPS="$STEPS" SEED="$SEED" \
  OUT="$STAGE/body.png" STATS="$STAGE/body_stats.json" \
  ./scripts/board_sd_run.sh "$PROMPT" 2>&1 \
  | grep -E 'edge density|seconds per step|steps  ' | sed 's/^/    /'

# 2. The real head, from the same 512 frame so the coordinates agree.
echo "==> 2/4  BiSeNet: cutting out the real head"
OUT="$STAGE/head.png" ./scripts/board_faceparse_run.sh "$STAGE/source_512.png" 2>&1 \
  | grep -E 'head coverage|inference took' | sed 's/^/    /'

# 3. Label the generated body, so the skin-tone correction knows an arm from
#    a shirt. A chrominance guess cannot: skin and a red garment share a Cr
#    range. ATR SegFormer has an explicit class for each.
SEG_ARGS=""
if [ -z "${NO_SEG:-}" ]; then
  echo "==> 3/4  SegFormer: labelling the generated body"
  OUT="$STAGE/seg.png" ./scripts/board_personseg_run.sh "$STAGE/body.png" 2>&1 \
    | grep -E 'inference took|silhouette|arms' | sed 's/^/    /' || true
  SEG_ARGS="--person-logits .personseg_stage/run/output_0.raw --person-crop .personseg_stage/run/person_crop.json"
fi

# 4. Put the real head back.
echo "==> 4/4  Compositing"
# shellcheck disable=SC2086
.venv/bin/python scripts/sd_head_composite.py \
  "$STAGE/body.png" "$STAGE/head.png" \
  --crop .faceparse_stage/run/crop.json \
  $SEG_ARGS ${MATCH_BODY:+--match-body "$MATCH_BODY"} \
  -o "$OUT" | sed 's/^/    /'

echo
echo "    $OUT"
