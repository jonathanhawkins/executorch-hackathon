#!/usr/bin/env bash
# NPU bring-up on the VENTUNO Q, following the track team's
# Ventuno_Q_ExecuTorch_Bringup.pdf.
#
#   ./scripts/board_npu_setup.sh            # does the no-sudo parts
#   ./scripts/board_npu_setup.sh --print    # just print the sudo block
#
# Split deliberately: the extraction and verification run unattended, but the
# apt steps need a password that only a human can type. This script does
# everything it can and prints the rest for you to paste.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

ARCHIVE=/home/arduino/ventuno_executorch_source.tar.gz
ET=/home/arduino/git/executorch

sudo_block() {
  cat <<'EOF'

Run these on the board (they need your password):

    make board-shell

then paste:

sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:ubuntu-qcom-iot/qcom-ppa
sudo apt update
sudo apt install -y libqnn1 libqnn-dev qnn-tools
sudo apt install -y python3.12-venv cmake ninja-build build-essential

Notes:
  * The qcom PPA is usually already configured on this image; add-apt-repository
    will just say so. Harmless.
  * python3.12-venv is genuinely missing from the image — that is why the CPU
    setup uses uv instead.
  * cmake/ninja/build-essential are not in the PDF but the image has no compiler
    at all, and the QNN runner has to be built from this checkout.

EOF
}

if [ "${1:-}" = "--print" ]; then
  sudo_block
  exit 0
fi

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "==> Checking the archive is on the board"
adb shell "test -f ${ARCHIVE}" || {
  echo "Missing ${ARCHIVE}. Push it first." >&2
  exit 1
}
adb shell "ls -lh ${ARCHIVE} | awk '{print \"    \"\$5}'" | tr -d '\r'

if adb shell "test -d ${ET}" 2>/dev/null; then
  echo "==> ${ET} already exists, skipping extraction"
else
  echo "==> Extracting (~6GB, a few minutes)"
  adb shell "cd /home/arduino && tar -xzf ventuno_executorch_source.tar.gz" | tr -d '\r'
fi

echo "==> Checkout contents"
adb shell "ls ${ET} | head -20; echo; du -sh ${ET}" | tr -d '\r'

echo "==> Does this checkout know QCS8275?"
# The whole point of the prepared tree: upstream's enum has no QCS8275.
adb shell "grep -rn 'QCS8275' ${ET}/backends/qualcomm/serialization/qc_schema.py 2>/dev/null | head -3" | tr -d '\r' \
  || echo "    (not found in qc_schema.py — check the checkout)"

echo "==> Is a QNN runner already built?"
adb shell "find ${ET} -name 'qnn_executor_runner' -o -name 'qnn_llama_runner' 2>/dev/null | head -5" | tr -d '\r' \
  || true

sudo_block
