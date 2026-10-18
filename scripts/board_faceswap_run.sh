#!/usr/bin/env bash
# Swap a shopper's face onto a product photo on the VENTUNO Q's Hexagon NPU.
#
#   ./scripts/board_faceswap_run.sh frame.jpg
#   ./scripts/board_faceswap_run.sh frame.jpg assets/templates/model-navy-tee.png
#   ITERATIONS=20 ./scripts/board_faceswap_run.sh frame.jpg      # for timing
#   REFERENCE=ref/x.raw ./scripts/board_faceswap_run.sh frame.jpg  # vs fp32
#
# **Frame and target, not a photo picker.** The source face is whoever is
# standing in front of the board; there is deliberately no path here for
# uploading a face that did not consent. See docs/faceswap-npu.md.
#
# A swap is two models, and only one of them belongs on the NPU:
#
#   identity   ArcFace iresnet100, **fp32 on the host by default**
#   swap       ghost_aei_qnn_*.pte on the Hexagon
#
# The encoder is on the CPU because quantizing it destroys the embedding —
# cosine 0.118 against fp32, which is a different person — and because it
# costs nothing to avoid: the identity describes whoever is standing there,
# not the instant the shutter fires, so a booth computes it while they walk
# up. `IDENTITY=npu` runs it on the Hexagon instead, for comparison.
#
# Preprocessing and the paste-back happen on the Mac; only the generator's
# convolutions run on the board. Export first with board_faceswap_export.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

FRAME="${1:?usage: board_faceswap_run.sh <frame> [target]}"
TARGET="${2:-assets/templates/model-navy-tee.png}"
ITERATIONS="${ITERATIONS:-1}"
OUT="${OUT:-swapped.png}"
# Which exported generator to run: the tag in its filename.
#   float  unquantized, lowered as fp16. 82 ms, 1.78/255 against fp32.
#   16a8w  quantized. 79 ms and visibly broken — 47.5/255, colour speckle.
# Quantizing this model buys 4 % and costs the picture; see docs/faceswap-npu.md.
QUANT="${QUANT:-float}"
# host (fp32, exact, off the critical path) or npu (quantized, currently wrong).
IDENTITY="${IDENTITY:-host}"
# An fp32 output_0.raw from ghost_swap.py --raw, to measure quantization cost.
REFERENCE="${REFERENCE:-}"
# Detect on a downscaled frame; a live loop should pass 3 (see CLAUDE.md).
SCALE="${SCALE:-1}"
# npu (Hexagon, 82 ms) or cpu (XNNPACK, ~2.1 s, numerically exact).
# The CPU path needs no QNN at all — see docs/faceswap-npu.md.
BACKEND="${BACKEND:-npu}"

BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
MODEL_DIR="${MODEL_DIR:-$BOARD/models/faceswap-npu}"
WORK=$BOARD/faceswap_input
RUNNER=$ET/build-x86/examples/qualcomm/executor_runner/qnn_executor_runner
CPU_RUNNER=$ET/build-xnnpack/executor_runner
STAGE=.faceswap_stage/run

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "==> Preprocessing${IDENTITY:+ (identity: $IDENTITY)}"
mkdir -p "$STAGE"
find "$STAGE" -maxdepth 1 -type f -delete
.venv/bin/python scripts/make_ghost_raw.py --source "$FRAME" --target "$TARGET" \
  --scale "$SCALE" -o "$STAGE" \
  $([ "$IDENTITY" = host ] && echo --embed) | sed 's/^/    /'

echo "==> Pushing to the board"
adb shell "mkdir -p $WORK/outputs && find $WORK -maxdepth 1 -type f -delete" >/dev/null
adb push "$STAGE"/. "$WORK/" >/dev/null

npu_env="
export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp
"

if [ "$IDENTITY" = npu ]; then
  echo "==> Identity on the NPU (ArcFace) — quantized, see docs/faceswap-npu.md"
  adb shell "
set -e
$npu_env
cd $WORK
$RUNNER \
  --model_path $MODEL_DIR/ghost_arcface_qnn_$QUANT.pte \
  --input_list_path identity_list.txt \
  --output_folder_path outputs \
  --iteration $ITERATIONS 2>&1 \
  | grep -iE 'inference latency|avg|elapsed|error' | head -5
# The generator's second input is this stage's output. Name it what
# target_list.txt already says it will be called.
cp outputs/output_0_0.raw embedding_0.raw 2>/dev/null || cp outputs/output_0.raw embedding_0.raw
rm -f outputs/output_0_0.raw outputs/output_0.raw
" 2>&1 | tr -d '\r'
else
  echo "==> Identity in fp32 on the host (not on the shutter's clock)"
fi

if [ "$BACKEND" = cpu ]; then
  echo "==> Swap on the board CPU (XNNPACK, fp32)"
  adb shell "
set -e
cd $WORK
$CPU_RUNNER \
  --model_path $MODEL_DIR/ghost_aei_xnnpack.pte \
  --inputs target_0.raw,embedding_0.raw \
  --num_executions $ITERATIONS --cpu_threads 8 \
  --output_file cpu_out --print_output none 2>&1 \
  | grep -iE 'executed successfully|error' | tail -2
" 2>&1 | tr -d '\r'
else
  echo "==> Swap on the NPU (GHOST AEI-Net)"
  adb shell "
set -e
$npu_env
cd $WORK
$RUNNER \
  --model_path $MODEL_DIR/ghost_aei_qnn_$QUANT.pte \
  --input_list_path target_list.txt \
  --output_folder_path outputs \
  --iteration $ITERATIONS 2>&1 \
  | grep -iE 'inference latency|avg|elapsed|error' | head -5
" 2>&1 | tr -d '\r'
fi

echo "==> Pasting back"
if [ "$BACKEND" = cpu ]; then
  adb pull "$WORK/cpu_out-0.bin" "$STAGE/output_0.raw" >/dev/null
else
  adb pull "$WORK/outputs/output_0_0.raw" "$STAGE/output_0.raw" >/dev/null 2>&1 \
    || adb pull "$WORK/outputs/output_0.raw" "$STAGE/output_0.raw" >/dev/null
fi
.venv/bin/python scripts/ghost_paste.py "$STAGE/output_0.raw" "$STAGE/crop.json" \
  -o "$OUT" ${REFERENCE:+--reference "$REFERENCE"} | sed 's/^/    /'
