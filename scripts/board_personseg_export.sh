#!/usr/bin/env bash
# Export person segmentation to a QNN .pte for the VENTUNO Q's Hexagon NPU.
#
#   ./scripts/board_personseg_export.sh                        # segformer-b0-atr
#   SEG_MODEL=dlv3-r101-voc ./scripts/board_personseg_export.sh # the baseline
#   CALIB=24 ./scripts/board_personseg_export.sh
#   QUANT=8a8w ./scripts/board_personseg_export.sh  # faster, loses the arms
#
# Runs ENTIRELY ON THE BOARD, for the same reason the face parser does: the
# ahead-of-time QNN compiler only exists as an aarch64 build inside the track
# team's bundle. See docs/qualcomm-npu.md.
#
# Produces $ARTIFACT/personseg_<model>_<quant>.pte, which the generic
# qnn_executor_runner runs. Budget ~10 minutes for segformer-b0.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

SEG_MODEL="${SEG_MODEL:-segformer-b0-atr}"
CALIB="${CALIB:-48}"
# 16a8w, not 8a8w. The silhouette barely notices the difference, but the arm
# classes do: against fp32 their IoU goes from 0.48-0.92 to 0.92-0.97, and an
# arm that is only sometimes occluded is a visible fault in the demo. It costs
# 24.9 ms against 11.2 ms on the NPU — which is the resource that was idle.
QUANT="${QUANT:-16a8w}"
BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
SHIM=$BOARD/qnn-sdk
VENV=$BOARD/et-venv
ARTIFACT="${ARTIFACT:-$BOARD/models/personseg-npu}"
CALIB_DIR=$BOARD/personseg_calib
CKPT_DIR=$BOARD/personseg_ckpt
STAGE=.personseg_stage/calib
CKPT_STAGE=.personseg_stage/ckpt
LOG=$BOARD/personseg_export.log

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "  model     $SEG_MODEL"
echo "  artifact  $ARTIFACT"
echo "  calib     $CALIB people"
echo "  log       $LOG  (on the board)"
echo

echo "==> Preparing the calibration set on the Mac"
mkdir -p "$STAGE"
find "$STAGE" -maxdepth 1 -type f -delete
.venv/bin/python scripts/make_person_raw.py --calib "$CALIB" --model "$SEG_MODEL" \
  -o "$STAGE" 2>&1 | sed 's/^/    /'

CKPT_ARGS=()
case "$SEG_MODEL" in
  segformer-*)
    echo "==> Fetching the checkpoint"
    CKPT=$(.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); \
      from person_seg import snapshot_path; print(snapshot_path('$SEG_MODEL'))" \
      2>/dev/null | tail -1)
    echo "    $CKPT"
    # Dereference first: a HF snapshot directory is symlinks into the blob
    # cache, and adb push would otherwise send the links.
    mkdir -p "$CKPT_STAGE"
    find "$CKPT_STAGE" -maxdepth 1 -type f -delete
    cp -RL "$CKPT"/. "$CKPT_STAGE/"
    adb shell "mkdir -p $CKPT_DIR && find $CKPT_DIR -maxdepth 1 -type f -delete" >/dev/null
    adb push "$CKPT_STAGE"/. "$CKPT_DIR/" >/dev/null
    CKPT_ARGS=(--checkpoint "$CKPT_DIR")
    ;;
  dlv3-*)
    # torchvision is not a dependency of this repo — the deeplab baseline is
    # the only thing that wants it, and it pulls no CUDA wheels here because
    # et-venv's torch already dragged them in. It fetches its own COCO weights
    # from download.pytorch.org on first use.
    echo "==> Installing torchvision on the board (deeplab baseline only)"
    adb shell "$BOARD/.local/bin/uv pip install --python $VENV/bin/python \
      --quiet torchvision 2>&1 | tail -3" | tr -d '\r' | sed 's/^/    /'
    ;;
esac

echo "==> Pushing the calibration set and driver"
adb shell "mkdir -p $CALIB_DIR && find $CALIB_DIR -maxdepth 1 -type f -delete" >/dev/null
adb push "$STAGE"/. "$CALIB_DIR/" >/dev/null
adb push scripts/board_personseg_export.py $BOARD/board_personseg_export.py >/dev/null
adb push scripts/person_seg.py $BOARD/person_seg.py >/dev/null

adb shell "
set -eo pipefail
# PYTHONPATH: the bundle tree first so 'executorch' is the QCS8275-capable
# source rather than the pip wheel, then \$BOARD for person_seg.py.
export PYTHONPATH=$BOARD/git:$BOARD
export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
# The AOT compiler hardcodes its host as x86_64; the shim points that name at
# the aarch64 libraries. See docs/whisper-npu.md.
export QNN_SDK_ROOT=$SHIM
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp
export HF_HUB_OFFLINE=1

$VENV/bin/python $BOARD/board_personseg_export.py \
  --compile_only \
  --soc_model QCS8275 \
  --build_folder $ET/build-x86 \
  --model $SEG_MODEL \
  ${CKPT_ARGS[*]:-} \
  --calib_dir $CALIB_DIR \
  --calib_samples $CALIB \
  --quant $QUANT \
  -a $ARTIFACT 2>&1 | tee $LOG | tail -30

ls -lh $ARTIFACT
" 2>&1 | tr -d '\r'
