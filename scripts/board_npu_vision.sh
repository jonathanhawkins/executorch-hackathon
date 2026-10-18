#!/usr/bin/env bash
# Run a vision-language model on the VENTUNO Q's Hexagon NPU.
#
#   ./scripts/board_npu_vision.sh internvl3-1b photo.jpg "What is in this image?"
#
# Preprocesses the image on the Mac, pushes the raw tensor, and drives the
# three-graph pipeline (vision encoder -> token embedding -> decoder).
#
# STATUS: the pipeline is verified up to the decoder — the vision encoder runs
# on the NPU in ~466 ms and emits 256x1536 features. Loading all three graphs
# at once only fits for the 1B variant; InternVL3-**2B** exhausts the cDSP's
# protection-domain memory (see docs/qualcomm-npu.md).
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

MODEL="${1:-internvl3-1b}"
IMAGE="${2:-}"
PROMPT="${3:-Describe this image.}"
# An image alone costs 256 tokens (img_seq_len), so a text-sized default like
# 128 fails before generating anything. Keep headroom under the model's
# compiled max context (1000 for the ctxt_1000 builds).
SEQ_LEN="${SEQ_LEN:-640}"
EVAL_MODE="${EVAL_MODE:-0}"
# The runner defaults to 0.0 = greedy argmax, which sends this quantized 1B
# into repetition loops ("New York, New York, New York..."). 0.7 is not enough;
# 1.0-1.2 produces answers that terminate properly. Short, specific prompts
# help more than long open-ended ones.
TEMPERATURE="${TEMPERATURE:-1.1}"

BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
DIR=$BOARD/models/$MODEL
RUNNER=$ET/build-x86/examples/qualcomm/oss_scripts/llama/qnn_multimodal_runner

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

# Check the model before the image — a missing model is the more likely problem
# and complaining about the image first sends you down the wrong path.
if ! adb shell "test -d $DIR" 2>/dev/null; then
  echo "No model at $DIR on the board." >&2
  echo >&2
  echo "Models currently on the board:" >&2
  adb shell "ls /home/arduino/models 2>/dev/null" | tr -d '\r' | sed 's/^/    /' >&2
  echo >&2
  echo "For a VLM you need all three .pte files (decoder, vision_encoder," >&2
  echo "tok_embedding). InternVL3-1B is the variant this board can fit;" >&2
  echo "2B exhausts the cDSP. See docs/qualcomm-npu.md." >&2
  exit 1
fi

SAMPLE=examples/qualcomm/oss_scripts/llama/assets/samples/images/Statue-of-Liberty-Island-New-York-Bay.png
if [ -z "$IMAGE" ]; then
  # Fall back to the sample image shipped in the bundle, so the command works
  # with no arguments.
  echo "==> No IMAGE given; using the bundle's sample image"
  TMP_PNG="${TMPDIR:-/tmp}/sample_vision.png"
  adb pull "$ET/$SAMPLE" "$TMP_PNG" >/dev/null 2>&1 || {
    echo "Could not fetch the sample image from the board." >&2; exit 1; }
  IMAGE="$TMP_PNG"
elif [ ! -f "$IMAGE" ]; then
  echo "No such image: $IMAGE" >&2
  echo "Pass a real path, e.g. IMAGE=~/Desktop/shot.png, or omit IMAGE to use" >&2
  echo "the sample image shipped with the bundle." >&2
  exit 1
fi

RAW="${TMPDIR:-/tmp}/$(basename "${IMAGE%.*}").raw"
echo "==> Preprocessing $(basename "$IMAGE") -> 448x448 NCHW float32"
.venv/bin/python scripts/make_vision_raw.py "$IMAGE" -o "$RAW" | sed 's/^/    /'
adb push "$RAW" "$DIR/input.raw" >/dev/null
REMOTE_RAW=input.raw

adb shell "
set -e
cd $DIR

# --image_path is an input LIST file, not the tensor itself: load_raw_files()
# opens it and reads newline-separated paths. Passing the .raw directly makes
# it try to open the tensor's bytes as filenames.
echo '$REMOTE_RAW' > image_list.txt

DEC=\$(ls *.pte | grep -v vision_encoder | grep -v tok_embedding | head -1)
ENC=\$(ls *vision_encoder*.pte 2>/dev/null | head -1)
EMB=\$(ls *tok_embedding*.pte 2>/dev/null | head -1)
test -n \"\$DEC\" -a -n \"\$ENC\" -a -n \"\$EMB\" || {
  echo 'Need all three .pte files (decoder, vision_encoder, tok_embedding).' >&2
  ls *.pte >&2; exit 1; }

export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp

$RUNNER \
  --decoder_path \"\$DEC\" \
  --encoder_path \"\$ENC\" \
  --tok_embedding_path \"\$EMB\" \
  --tokenizer_path tokenizer.json \
  --image_path image_list.txt \
  --decoder_model_version internvl3 \
  --prompt \"$PROMPT\" \
  --seq_len $SEQ_LEN \
  --eval_mode $EVAL_MODE \
  --temperature $TEMPERATURE \
  --output_path $BOARD/vlm_out.txt > $BOARD/vlm_log.txt 2>&1 || true

grep -aE 'Prompt evaluation|Generated .* tokens|Model Load Time|Time to first|available PD|shared weights' $BOARD/vlm_log.txt | tail -8
echo
echo '--- response ---'
cat $BOARD/vlm_out.txt 2>/dev/null | tail -c 1500
" 2>&1 | tr -d '\r'
