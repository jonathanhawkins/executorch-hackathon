#!/usr/bin/env bash
# Segment the person in a frame on the VENTUNO Q's Hexagon NPU.
#
#   ./scripts/board_personseg_run.sh frame.jpg
#   ITERATIONS=100 ./scripts/board_personseg_run.sh frame.jpg    # for timing
#   OUT=body.png CLASSES=1 ./scripts/board_personseg_run.sh frame.jpg
#
# Preprocessing and matting happen on the Mac; only the convolutions and
# attention run on the board. Export the model first with
# scripts/board_personseg_export.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

IMAGE="${1:?usage: board_personseg_run.sh <image>}"
SEG_MODEL="${SEG_MODEL:-segformer-b0-atr}"
QUANT="${QUANT:-16a8w}"
ITERATIONS="${ITERATIONS:-1}"
OUT="${OUT:-body.png}"
COVER="${COVER:-}"
CLASSES="${CLASSES:-}"

BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
MODEL_DIR="${MODEL_DIR:-$BOARD/models/personseg-npu}"
WORK=$BOARD/personseg_input
RUNNER=$ET/build-x86/examples/qualcomm/executor_runner/qnn_executor_runner
STAGE=.personseg_stage/run

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "==> Preprocessing $IMAGE"
mkdir -p "$STAGE"
find "$STAGE" -maxdepth 1 -type f -delete
.venv/bin/python scripts/make_person_raw.py "$IMAGE" -o "$STAGE" \
  --model "$SEG_MODEL" ${COVER:+--cover "$COVER"} | sed 's/^/    /'

echo "==> Pushing to the board"
adb shell "mkdir -p $WORK/outputs && find $WORK -maxdepth 1 -type f -delete" >/dev/null
adb push "$STAGE"/. "$WORK/" >/dev/null

echo "==> Segmenting on the NPU"
adb shell "
set -e
export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp

cd $WORK
$RUNNER \
  --model_path $MODEL_DIR/personseg_${SEG_MODEL}_${QUANT}.pte \
  --input_list_path input_list.txt \
  --output_folder_path outputs \
  --iteration $ITERATIONS 2>&1 \
  | grep -iE 'inference latency|avg|elapsed|error' | head -10
" 2>&1 | tr -d '\r'

echo "==> Matting"
adb pull "$WORK/outputs/output_0_0.raw" "$STAGE/output_0.raw" >/dev/null 2>&1 \
  || adb pull "$WORK/outputs/output_0.raw" "$STAGE/output_0.raw" >/dev/null
.venv/bin/python scripts/person_matte.py "$STAGE/output_0.raw" "$IMAGE" -o "$OUT" \
  ${CLASSES:+--classes} | sed 's/^/    /'
