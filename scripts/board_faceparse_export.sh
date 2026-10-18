#!/usr/bin/env bash
# Export BiSeNet face parsing to a QNN .pte for the VENTUNO Q's Hexagon NPU.
#
#   ./scripts/board_faceparse_export.sh            # 40 calibration faces
#   CALIB=20 ./scripts/board_faceparse_export.sh
#
# Runs ENTIRELY ON THE BOARD. The ahead-of-time QNN compiler only exists as an
# aarch64 build inside the track team's bundle, so the Mac cannot do this —
# see docs/qualcomm-npu.md.
#
# Produces $ARTIFACT/faceparse_qnn_8a8w.pte, which the generic qnn_executor_runner
# runs. Budget ~10 minutes; calibration is 40 forward passes of a 13M-parameter
# conv net, which is far lighter than whisper's token-by-token loop.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

CALIB="${CALIB:-40}"
BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
SHIM=$BOARD/qnn-sdk
VENV=$BOARD/et-venv
ARTIFACT="${ARTIFACT:-$BOARD/models/faceparse-npu}"
CALIB_DIR=$BOARD/faceparse_calib
STAGE=.faceparse_stage
LOG=$BOARD/faceparse_export.log

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "  artifact  $ARTIFACT"
echo "  calib     $CALIB faces"
echo "  log       $LOG  (on the board)"
echo

echo "==> Preparing the calibration set on the Mac"
mkdir -p "$STAGE"
find "$STAGE" -maxdepth 1 -type f -delete
.venv/bin/python scripts/make_face_raw.py --calib "$CALIB" -o "$STAGE" 2>&1 | sed 's/^/    /'

echo "==> Fetching and checking the checkpoint"
CKPT=$(.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); from face_parse import checkpoint_path; print(checkpoint_path())" 2>/dev/null | tail -1)
echo "    $CKPT"

echo "==> Pushing model, calibration set and driver"
adb shell "mkdir -p $CALIB_DIR && find $CALIB_DIR -maxdepth 1 -type f -delete" >/dev/null
adb push "$STAGE"/. "$CALIB_DIR/" >/dev/null
adb push "$CKPT" $BOARD/79999_iter.pth >/dev/null
adb push scripts/board_faceparse_export.py $BOARD/board_faceparse_export.py >/dev/null
adb push vendor $BOARD/vendor >/dev/null

adb shell "
set -eo pipefail
# PYTHONPATH: the bundle tree first so 'executorch' is the QCS8275-capable
# source rather than the pip wheel, then \$BOARD so 'vendor.face_parsing'
# resolves to what we just pushed.
export PYTHONPATH=$BOARD/git:$BOARD
export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
# The AOT compiler hardcodes its host as x86_64; the shim points that name at
# the aarch64 libraries. See docs/whisper-npu.md.
export QNN_SDK_ROOT=$SHIM
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp

$VENV/bin/python $BOARD/board_faceparse_export.py \
  --compile_only \
  --soc_model QCS8275 \
  --build_folder $ET/build-x86 \
  --checkpoint $BOARD/79999_iter.pth \
  --calib_dir $CALIB_DIR \
  --calib_samples $CALIB \
  -a $ARTIFACT 2>&1 | tee $LOG | tail -30

ls -lh $ARTIFACT
" 2>&1 | tr -d '\r'
