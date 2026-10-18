#!/usr/bin/env bash
# Parse a face on the VENTUNO Q's Hexagon NPU and write a head/neck cutout.
#
#   ./scripts/board_faceparse_run.sh portrait.jpg
#   ITERATIONS=20 ./scripts/board_faceparse_run.sh portrait.jpg   # for timing
#
# Preprocessing and matting happen on the Mac; only the convolutions run on the
# board. Export the model first with scripts/board_faceparse_export.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

IMAGE="${1:?usage: board_faceparse_run.sh <image>}"
ITERATIONS="${ITERATIONS:-1}"
OUT="${OUT:-head.png}"
# A face box, X,Y,W,H. Strongly recommended: BiSeNet wants a face-filling
# square, and a whole webcam frame is not one. See docs/faceparse-npu.md.
BBOX="${BBOX:-}"

BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
MODEL_DIR="${MODEL_DIR:-$BOARD/models/faceparse-npu}"
WORK=$BOARD/faceparse_input
RUNNER=$ET/build-x86/examples/qualcomm/executor_runner/qnn_executor_runner
STAGE=.faceparse_stage/run

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "==> Preprocessing $IMAGE"
mkdir -p "$STAGE"
find "$STAGE" -maxdepth 1 -type f -delete
.venv/bin/python scripts/make_face_raw.py "$IMAGE" -o "$STAGE" \
  ${BBOX:+--bbox "$BBOX"} | sed 's/^/    /'

echo "==> Pushing to the board"
adb shell "mkdir -p $WORK/outputs && find $WORK -maxdepth 1 -type f -delete" >/dev/null
adb push "$STAGE"/. "$WORK/" >/dev/null

echo "==> Parsing on the NPU"
adb shell "
set -e
export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp

cd $WORK
$RUNNER \
  --model_path $MODEL_DIR/faceparse_qnn_8a8w.pte \
  --input_list_path input_list.txt \
  --output_folder_path outputs \
  --iteration $ITERATIONS 2>&1 \
  | grep -iE 'inference latency|avg|elapsed|error' | head -10
" 2>&1 | tr -d '\r'

echo "==> Matting"
adb pull "$WORK/outputs/output_0_0.raw" "$STAGE/output_0.raw" >/dev/null 2>&1 \
  || adb pull "$WORK/outputs/output_0.raw" "$STAGE/output_0.raw" >/dev/null
.venv/bin/python scripts/face_matte.py "$STAGE/output_0.raw" "$IMAGE" -o "$OUT" --classes \
  | sed 's/^/    /'
