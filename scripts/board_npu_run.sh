#!/usr/bin/env bash
# Run a QNN model on the VENTUNO Q's Hexagon NPU.
#
#   ./scripts/board_npu_run.sh                                  # gemma4, default prompt
#   ./scripts/board_npu_run.sh gemma4_e2b "Explain edge AI."    # custom prompt
#   SEQ_LEN=600 ./scripts/board_npu_run.sh                      # longer answer
#
# Needs no sudo. Everything resolves from the extracted bundle plus the QAIRT
# .debs unpacked under ~/qairt-probe by scripts/board_npu_check.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

MODEL="${1:-gemma4_e2b}"
PROMPT="${2:-Name three reasons to run an AI model on a device instead of a server.}"
SEQ_LEN="${SEQ_LEN:-400}"
DECODER="${DECODER:-}"

BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
RUNNER=$ET/build-x86/examples/qualcomm/oss_scripts/llama/qnn_llama_runner

# The runner needs to be told which decoder family the .pte is, and it defaults
# to llama2. Guess from the directory name; override with DECODER=.
if [ -z "$DECODER" ]; then
  case "$MODEL" in
    *gemma4*|*gemma3n*) DECODER=gemma4 ;;
    *gemma3*)           DECODER=gemma3 ;;
    *gemma2*)           DECODER=gemma2 ;;
    *gemma*)            DECODER=gemma ;;
    *qwen3*)            DECODER=qwen3 ;;
    *qwen2*)            DECODER=qwen2_5 ;;
    *phi*)              DECODER=phi_4_mini ;;
    *llama3*)           DECODER=llama3 ;;
    *)                  DECODER=llama2 ;;
  esac
fi

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "  model    $MODEL  (decoder: $DECODER)"
echo "  seq_len  $SEQ_LEN"
echo

adb shell "
set -e
MODEL_DIR=$BOARD/models/$MODEL
test -d \$MODEL_DIR || { echo 'No such model dir: '\$MODEL_DIR >&2; exit 1; }
PTE=\$(find \$MODEL_DIR -maxdepth 1 -name '*.pte' ! -name '*vision_encoder*' ! -name '*tok_embedding*' | head -1)
test -n \"\$PTE\" || { echo 'No decoder .pte in '\$MODEL_DIR >&2; exit 1; }

# ExecuTorch's own QNN shim and the prebuilt libs, then QAIRT, then libatomic
# (which the image lacks and libQnnSystem.so needs).
export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
# Where the Hexagon skel libraries live, loaded onto the DSP over FastRPC.
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp

cd \$MODEL_DIR
$RUNNER \
  --model_path \"\$PTE\" \
  --tokenizer_path tokenizer.json \
  --decoder_model_version $DECODER \
  --prompt \"$PROMPT\" \
  --seq_len $SEQ_LEN \
  --eval_mode 1 \
  --output_path $BOARD/npu_out.txt 2>&1 \
  | grep -E 'Prompt evaluation|Generated .* tokens|Model Load Time|Time to first|RSS after finishing'

echo
echo '--- response ---'
cat $BOARD/npu_out.txt
" 2>&1 | tr -d '\r'
