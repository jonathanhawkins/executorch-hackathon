#!/usr/bin/env bash
# Export GHOST face swap to QNN .pte files for the VENTUNO Q's Hexagon NPU.
#
#   ./scripts/board_faceswap_export.sh                 # 16a8w, 40 calibration faces
#   QUANT=8a8w ./scripts/board_faceswap_export.sh      # half the activation width
#   CALIB=20 STAGE=identity ./scripts/board_faceswap_export.sh
#
# Runs ENTIRELY ON THE BOARD. The ahead-of-time QNN compiler only exists as an
# aarch64 build inside the track team's bundle, so the Mac cannot do this —
# see docs/qualcomm-npu.md.
#
# Produces two .pte files, because a swap is two models:
#
#   ghost_arcface_qnn_$QUANT.pte   the identity encoder, 65 M params
#   ghost_aei_qnn_$QUANT.pte       the generator, 184 M params
#
# They are exported in SEPARATE PROCESSES on purpose. The QNN manager is
# process-global; two lowerings in one interpreter is the same trap as calling
# from_context_binary() twice (CLAUDE.md).
#
# Budget ~20 minutes for both: ArcFace ~4, the generator ~8, and calibration
# the rest. Calibration is a forward pass per face through a 184 M-parameter
# generator on the board's CPU, so it dominates.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

CALIB="${CALIB:-40}"
# none | 8a8w | 16a8w | 16a16w | fp16a8w.
#
# `none` is the default and the one that works: it hands QNN the float graph,
# the compile spec sets use_fp16=True, and the HTP runs it in fp16 at 82 ms
# with 1.78/255 against the fp32 reference.
#
# Quantizing this generator is a bad trade. 16a8w is 79 ms — 4 % faster — and
# 47.5/255, which is visible colour speckle across the face. The cause is not
# bit width; see docs/faceswap-npu.md for what was ruled out. 16a16w is far
# too slow (16-bit weights are not HTP-accelerated: 94x on ArcFace), and
# fp16a8w does not lower at all, because the HTP will not mix int8 weights
# with float activations.
#
# The identity encoder is not quantized either — it runs fp32 on the host.
QUANT="${QUANT:-none}"
# identity, generator, or both.
STAGE="${STAGE:-both}"
# The two quantizer knobs that decide whether this survives at all; see the
# note in board_faceswap_export.py. PER_CHANNEL must be false for fp16a8w.
PER_CHANNEL="${PER_CHANNEL:-true}"
ACT_OBSERVER="${ACT_OBSERVER:-moving_minmax}"
# Optional: a wider config for just the AAD_ResBlk submodules, which hold all
# 21 InstanceNorm2d layers. Empty = off.
AAD_QUANT="${AAD_QUANT:-}"

BOARD=/home/arduino
ET=$BOARD/git/executorch
PROBE=$BOARD/qairt-probe
SHIM=$BOARD/qnn-sdk
VENV=$BOARD/et-venv
ARTIFACT="${ARTIFACT:-$BOARD/models/faceswap-npu}"
CALIB_DIR=$BOARD/faceswap_calib
STAGE_DIR=.faceswap_stage
LOG=$BOARD/faceswap_export.log

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "  artifact  $ARTIFACT"
echo "  quant     $QUANT"
echo "  calib     $CALIB faces"
echo "  stage     $STAGE"
echo "  weights   $([ "$PER_CHANNEL" = true ] && echo per-channel || echo per-tensor), act observer $ACT_OBSERVER"
echo "  log       $LOG  (on the board)"
echo

echo "==> Preparing the calibration set on the Mac"
mkdir -p "$STAGE_DIR"
find "$STAGE_DIR" -maxdepth 1 -type f -delete
.venv/bin/python scripts/make_ghost_raw.py --calib "$CALIB" -o "$STAGE_DIR" \
  2>&1 | sed 's/^/    /'

echo "==> Fetching and checking the checkpoints"
# ghost_swap.py verifies both digests; it exits non-zero if either is wrong.
GEN_CKPT=$(.venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts')
from ghost_swap import weight_path
print(weight_path('G_unet_2blocks.pth'))" | tail -1)
ARC_CKPT=$(.venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts')
from ghost_swap import weight_path
print(weight_path('backbone.pth'))" | tail -1)
echo "    $GEN_CKPT"
echo "    $ARC_CKPT"

echo "==> Pushing calibration set, driver and model code"
adb shell "mkdir -p $CALIB_DIR && find $CALIB_DIR -maxdepth 1 -type f -delete" >/dev/null
adb push "$STAGE_DIR"/. "$CALIB_DIR/" >/dev/null
adb push scripts/board_faceswap_export.py $BOARD/board_faceswap_export.py >/dev/null
# `adb push vendor $BOARD/vendor` nests as vendor/vendor once the destination
# exists, which it does as soon as any other export has run. The trailing /.
# pushes the contents instead.
adb shell "mkdir -p $BOARD/vendor" >/dev/null
adb push vendor/. $BOARD/vendor/ >/dev/null

# ~1 GB of weights. Only worth pushing once.
for pair in "$ARC_CKPT:backbone.pth" "$GEN_CKPT:G_unet_2blocks.pth"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  size=$(wc -c <"$src" | tr -d ' ')
  have=$(adb shell "stat -c %s $BOARD/ghost/$dst 2>/dev/null || echo 0" | tr -d '\r')
  if [ "$have" = "$size" ]; then
    echo "    $dst already on the board"
  else
    echo "    pushing $dst ($((size / 1024 / 1024)) MB)"
    adb shell "mkdir -p $BOARD/ghost" >/dev/null
    adb push "$src" "$BOARD/ghost/$dst" >/dev/null
  fi
done

run_stage() {
  local stage="$1"
  echo
  echo "==> Exporting the $stage graph ($QUANT)"
  adb shell "
set -eo pipefail
# PYTHONPATH: the bundle tree first so 'executorch' is the QCS8275-capable
# source rather than the pip wheel, then \$BOARD so 'vendor.ghost' resolves to
# what we just pushed.
export PYTHONPATH=$BOARD/git:$BOARD
export LD_LIBRARY_PATH=$ET/build-x86/backends/qualcomm:$ET/build-x86/lib:$PROBE/extracted/usr/lib:$PROBE/extracted/usr/lib/aarch64-linux-gnu
# The AOT compiler hardcodes its host as x86_64; the shim points that name at
# the aarch64 libraries. See docs/whisper-npu.md.
export QNN_SDK_ROOT=$SHIM
export ADSP_LIBRARY_PATH=$PROBE/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp

\$HOME/et-venv/bin/python $BOARD/board_faceswap_export.py \
  --compile_only \
  --soc_model QCS8275 \
  --build_folder $ET/build-x86 \
  --stage $stage \
  --quant $QUANT \
  --per_channel $PER_CHANNEL \
  --act_observer $ACT_OBSERVER \
  ${AAD_QUANT:+--aad_quant $AAD_QUANT} \
  --identity_ckpt $BOARD/ghost/backbone.pth \
  --generator_ckpt $BOARD/ghost/G_unet_2blocks.pth \
  --calib_dir $CALIB_DIR \
  --calib_samples $CALIB \
  -a $ARTIFACT 2>&1 | tee -a $LOG | tail -25
" 2>&1 | tr -d '\r'
}

adb shell ": > $LOG" >/dev/null
case "$STAGE" in
  identity|generator) run_stage "$STAGE" ;;
  both) run_stage identity; run_stage generator ;;
  *) echo "STAGE must be identity, generator or both" >&2; exit 1 ;;
esac

echo
adb shell "ls -lh $ARTIFACT" 2>&1 | tr -d '\r'
