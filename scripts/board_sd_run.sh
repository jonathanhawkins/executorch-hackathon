#!/usr/bin/env bash
# Generate an image with Stable Diffusion v2.1 on the VENTUNO Q's Hexagon NPU.
#
#   ./scripts/board_sd_run.sh "a photo of an astronaut riding a horse on mars"
#   STEPS=8 ./scripts/board_sd_run.sh "a studio photo of a white cotton t-shirt"
#   BENCH=40 ./scripts/board_sd_run.sh "..."       # longer timing run
#   PERSISTENT=off ./scripts/board_sd_run.sh "..."  # reload every graph per use
#   REQUANT=pairs ./scripts/board_sd_run.sh "..."   # the old residual remap
#   SAMPLER=euler SCHEDULE=linspace PLAIN=1 ./scripts/board_sd_run.sh "..."  # the old way
#
# Export the .pte files once first with scripts/board_sd_export.sh.
#
# Prompt tokenization happens on the Mac, everything else on the board — the
# same split as board_faceparse_run.sh. The QNN graphs run through the patched
# qnn_executor_runner, held open across executions so the UNet is loaded once
# per image rather than once per denoising step; board_sd_driver.py drives the
# scheduler. PERSISTENT=off goes back to a load per step.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

PROMPT="${1:-a photo of an astronaut riding a horse on mars}"
STEPS="${STEPS:-20}"
GUIDANCE="${GUIDANCE:-7.5}"
SEED="${SEED:-42}"
# The unconditional pass runs either way for classifier-free guidance, so a
# negative prompt costs literally nothing — and on this w8a16 model it is the
# single biggest quality lever there is: 20 steps with one matches 40 without.
DEFAULT_NEGATIVE="blurry, low quality, distorted, deformed hands, extra limbs, watermark, text, cropped, out of focus"
NEGATIVE="${NEGATIVE:-$DEFAULT_NEGATIVE}"
[ -n "${PLAIN:-}" ] && NEGATIVE=""   # PLAIN=1 reverts to an empty uncond prompt
OUT="${OUT:-sd21.png}"
STATS="${STATS:-${PREFIX:-sd21}_stats.json}"

BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
VENV=$BOARD/et-venv
MODEL_DIR="${MODEL_DIR:-$BOARD/models/sd21-npu}"
PREFIX="${PREFIX:-sd21}"
# sd21 = Stable Diffusion 2.1 (v-prediction, 1024-wide text emb, pads with 0)
# sd15 = the ControlNet-Canny set (epsilon, 768-wide, pads with EOS)
BASE="${BASE:-sd21}"
# A photo to condition on, for the ControlNet model set. Its Canny edge map
# constrains pose and composition, which is what keeps the model from
# inventing a different scene. See docs/stable-diffusion-npu.md.
IMAGE="${IMAGE:-}"
# tmpfs, not eMMC. The runner exchanges tensors as files, and ControlNet
# writes ~27 MB of residuals per step; /dev/shm does 1.3 GB/s against eMMC's
# 130. Same reason WarmRunner defaults there — see docs/persistent-runner.md.
WORK="${WORK:-/dev/shm/sd_work}"
RUNNER=$ET/build-x86/examples/qualcomm/executor_runner/qnn_executor_runner
STAGE=.sd_stage

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "  prompt    $PROMPT"
echo "  steps     $STEPS   guidance $GUIDANCE   seed $SEED"
echo "  sampler   ${SAMPLER:-dpmpp_2m} / ${SCHEDULE:-karras}"
echo

echo "==> Tokenizing on the Mac"
mkdir -p "$STAGE"
rm -f "$STAGE"/tokens_*.raw
.venv/bin/python scripts/make_sd_tokens.py "$PROMPT" \
  ${NEGATIVE:+--negative-prompt "$NEGATIVE"} --base "$BASE" -o "$STAGE" | sed 's/^/    /'

if [ -n "$IMAGE" ]; then
  echo "==> Canny edges on the Mac"
  .venv/bin/python scripts/make_canny_raw.py "$IMAGE" -o "$STAGE" \
    ${CANNY_LOW:+--low "$CANNY_LOW"} ${CANNY_HIGH:+--high "$CANNY_HIGH"} \
    | sed 's/^/    /'
fi

echo "==> Pushing to the board"
adb shell "mkdir -p $WORK" >/dev/null
adb push "$STAGE"/tokens_cond.raw "$STAGE"/tokens_uncond.raw "$WORK/" >/dev/null
if [ -n "$IMAGE" ]; then adb push "$STAGE"/image_cond.raw "$WORK/" >/dev/null; fi
# warm_runner.py goes beside the driver so a plain sibling import finds it;
# without it the driver reloads every graph per execution and says so.
adb push scripts/board_sd_driver.py $BOARD/board_sd_driver.py >/dev/null
adb push scripts/warm_runner.py $BOARD/warm_runner.py >/dev/null

echo "==> Generating on the NPU"
adb shell "
set -eo pipefail
test -f $MODEL_DIR/${PREFIX}_unet.pte || {
  echo 'No ${PREFIX}_unet.pte under $MODEL_DIR — run scripts/board_sd_export.sh first.' >&2
  exit 1
}

# The runtime side only needs the QNN shim libraries and the Hexagon skels; the
# AOT shim root is irrelevant here. Same incantation as board_faceparse_run.sh.
export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp

cd $WORK
$VENV/bin/python $BOARD/board_sd_driver.py \
  --prompt \"$PROMPT\" \
  --model_dir $MODEL_DIR \
  --prefix $PREFIX \
  --tokens_dir $WORK \
  --runner $RUNNER \
  --workdir $WORK \
  --output $WORK/output.png \
  --stats $WORK/stats.json \
  --num_time_steps $STEPS \
  --guidance_scale $GUIDANCE \
  --conditioning_scale ${COND_SCALE:-1.0} \
  --seed $SEED \
  --prediction_type "${PREDICTION:-v_prediction}" \
  --sampler "${SAMPLER:-dpmpp_2m}" \
  --schedule "${SCHEDULE:-karras}" \
  --bench_iterations ${BENCH:-20} \
  --persistent_runner ${PERSISTENT:-auto} \
  --requant ${REQUANT:-lut} ${VERBOSE:+--verbose}
" 2>&1 | tr -d '\r'

echo "==> Pulling the image"
adb pull "$WORK/output.png" "$OUT" >/dev/null
adb pull "$WORK/stats.json" "$STATS" >/dev/null
echo "    $OUT"
echo "    $STATS"
