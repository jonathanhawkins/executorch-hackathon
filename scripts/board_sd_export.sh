#!/usr/bin/env bash
# Wrap the AI Hub Stable Diffusion v2.1 context binaries into .pte on the board.
#
#   ./scripts/board_sd_export.sh            # once; ~1.4 GB of .pte, a few minutes
#   FORCE=1 ./scripts/board_sd_export.sh    # re-wrap
#
# Runs ENTIRELY ON THE BOARD: the ahead-of-time QNN path only exists as the
# bundle's aarch64 PyQnnManagerAdaptor, so the Mac cannot do this. See
# docs/qualcomm-npu.md.
#
# Nothing is compiled or quantized here — Qualcomm AI Hub already published
# these three graphs compiled for the QCS8275's Hexagon v75, w8a16. This only
# wraps each one in the ExecuTorch container the prebuilt runner can load.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
VENV=$BOARD/et-venv
SHIM=$BOARD/qnn-sdk
# Release 0.62.2 of the AI Hub export, built with QAIRT 2.45 — safely below the
# board's 2.49 runtime. See docs/stable-diffusion-npu.md on versions.
CTX="${CTX:-$BOARD/sd21/extracted/stable_diffusion_v2_1-qnn_context_binary-w8a16-qualcomm_qcs8275}"
ARTIFACT="${ARTIFACT:-$BOARD/models/sd21-npu}"
# ControlNet-Canny adds a fourth graph; see docs/stable-diffusion-npu.md.
STAGES="${STAGES:-text_encoder,unet,vae}"
PREFIX="${PREFIX:-sd21}"
LOG=$BOARD/sd_export.log

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "  context binaries  $CTX"
echo "  artifact          $ARTIFACT"
echo "  stages            $STAGES  (prefix $PREFIX)"
echo

adb push scripts/board_sd_export.py $BOARD/board_sd_export.py >/dev/null

adb shell "
set -eo pipefail
test -f $CTX/unet.bin || { echo 'No unet.bin under $CTX' >&2; exit 1; }

# The AOT compiler hardcodes its host as x86_64 and looks for the QNN libraries
# under \$QNN_SDK_ROOT/lib/x86_64-linux-clang. They are aarch64 here, so point
# that name at them — the loader only cares what it finds.
mkdir -p $SHIM/lib
ln -sfn $PROBE/extracted/usr/lib $SHIM/lib/x86_64-linux-clang

# PYTHONPATH ahead of site-packages so 'executorch' resolves to the bundle's
# QCS8275-capable tree, not the pip wheel that has no QNN AOT support at all.
export PYTHONPATH=$BOARD/git
export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
export QNN_SDK_ROOT=$SHIM
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp

mkdir -p $ARTIFACT
cd $BOARD
rm -f $LOG
# One stage per process. The bundle's preprocess_multimethod leaves a stale
# QnnManagerContext behind, so a second from_context_binary() in the same
# interpreter dies in qnn_preprocess.py. A fresh interpreter each time also
# guarantees only one set of weights is mapped onto the cDSP at a time.
for stage in $(echo "$STAGES" | tr ',' ' '); do
  $VENV/bin/python $BOARD/board_sd_export.py \
    --context_binaries $CTX \
    -a $ARTIFACT \
    --soc_model QCS8275 \
    --stages $STAGES \
    --prefix $PREFIX \
    --only \$stage ${FORCE:+--force} 2>&1 \
    | tee -a $LOG \
    | grep -vE '^\[(INFO|WARNING)( |\])|^W[0-9]{4} |rpcmem_|LIBNATIVE|unregisterAsync'
done

ls -lh $ARTIFACT
df -h /home | tail -1
" 2>&1 | tr -d '\r'
