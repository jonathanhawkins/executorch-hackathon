#!/usr/bin/env bash
# Export Whisper to a QNN .pte for the VENTUNO Q's Hexagon NPU.
#
#   ./scripts/board_whisper_export.sh                 # whisper-tiny, 256-token context
#   MAX_SEQ_LEN=128 ./scripts/board_whisper_export.sh
#
# Runs ENTIRELY ON THE BOARD. The ahead-of-time QNN compiler only exists as an
# aarch64 build inside the track team's bundle (PyQnnManagerAdaptor...so), so
# the Mac cannot do this — see docs/qualcomm-npu.md.
#
# Produces $ARTIFACT/whisper_qnn_16a8w.pte (encoder + decoder in one file) plus
# the tokenizer, which the runner needs alongside it.
#
# Slow: ~20 librispeech samples are decoded in eager PyTorch to calibrate the
# 16a8w quantizer before anything is lowered. Budget 20-40 minutes.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
VENV=$BOARD/et-venv
SHIM=$BOARD/qnn-sdk
ARTIFACT="${ARTIFACT:-$BOARD/models/whisper-tiny-npu}"
LOG=$BOARD/whisper_export.log

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "  artifact  $ARTIFACT"
echo "  seq_len   $MAX_SEQ_LEN"
echo "  log       $LOG  (on the board)"
echo

# Calibration reads LibriSpeech clips straight out of the dataset's parquet;
# see board_whisper_export.py for why it avoids `datasets` and torchcodec.
echo "==> Preflight: calibration dependencies"
adb push scripts/board_whisper_export.py $BOARD/board_whisper_export.py >/dev/null
adb shell "
set -e
export PATH=\$HOME/.local/bin:\$PATH
uv pip install --python $VENV/bin/python --prerelease=allow -q \
  pyarrow soundfile pydot

# The AOT compiler hardcodes its host as x86_64 and looks for the QNN
# libraries under \$QNN_SDK_ROOT/lib/x86_64-linux-clang. They are aarch64
# here, so point that name at them — the loader only cares what it finds.
mkdir -p $SHIM/lib
ln -sfn $PROBE/extracted/usr/lib $SHIM/lib/x86_64-linux-clang
" >/dev/null 2>&1 || echo "    (preflight had warnings; continuing)"

adb shell "
set -eo pipefail
# PYTHONPATH ahead of site-packages so 'executorch' resolves to the bundle's
# QCS8275-capable tree, not the pip wheel that has no QNN AOT support at all.
export PYTHONPATH=$BOARD/git
export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
export QNN_SDK_ROOT=$SHIM
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp

cd $ET/examples/qualcomm/oss_scripts/whisper
$VENV/bin/python $BOARD/board_whisper_export.py \
  --compile_only \
  --soc_model QCS8275 \
  --build_folder $ET/build-x86 \
  -a $ARTIFACT \
  --max_seq_len $MAX_SEQ_LEN 2>&1 | tee $LOG | tail -40

# --compile_only stops short of the inference path that would normally save
# the tokenizer, and the runner will not start without it.
$VENV/bin/python -c \
  \"from transformers import AutoTokenizer; \
   AutoTokenizer.from_pretrained('openai/whisper-tiny').save_pretrained('$ARTIFACT')\"

ls -lh $ARTIFACT
" 2>&1 | tr -d '\r'
