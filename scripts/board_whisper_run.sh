#!/usr/bin/env bash
# Transcribe an audio file on the VENTUNO Q's Hexagon NPU.
#
#   ./scripts/board_whisper_run.sh clip.wav
#   SEQ_LEN=128 ./scripts/board_whisper_run.sh interview.m4a
#
# Preprocessing happens on the Mac (ffmpeg + the Whisper feature extractor),
# because qnn_whisper_runner takes raw float32 mel tensors and nothing else.
# Export the model first with scripts/board_whisper_export.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

AUDIO="${1:?usage: board_whisper_run.sh <audio-file>}"
SEQ_LEN="${SEQ_LEN:-256}"

BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
MODEL_DIR="${MODEL_DIR:-$BOARD/models/whisper-tiny-npu}"
WORK=$BOARD/whisper_input
RUNNER=$ET/build-x86/examples/qualcomm/oss_scripts/whisper/qnn_whisper_runner
STAGE=.whisper_stage

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "==> Preprocessing $AUDIO"
mkdir -p "$STAGE"
find "$STAGE" -maxdepth 1 -type f -delete
.venv/bin/python scripts/make_whisper_raw.py "$AUDIO" -o "$STAGE"

echo "==> Pushing to the board"
# Stale chunks from a previous, longer clip would otherwise still be listed.
adb shell "mkdir -p $WORK/outputs && find $WORK -maxdepth 1 -type f -delete" >/dev/null
for f in "$STAGE"/*; do adb push "$f" "$WORK/" >/dev/null; done

echo "==> Transcribing on the NPU"
adb shell "
set -e
export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp

cd $WORK
$RUNNER \
  --model_path $MODEL_DIR/whisper_qnn_16a8w.pte \
  --tokenizer_json_path $MODEL_DIR/tokenizer.json \
  --input_list_path input_list.txt \
  --seq_len $SEQ_LEN \
  --output_folder_path outputs 2>&1 \
  | grep -E 'Model Load Time|Encoding Time|Decoding Time|Total Generated|rror'

echo
echo '--- transcript ---'
cat outputs/output_*.txt
echo
" 2>&1 | tr -d '\r'
