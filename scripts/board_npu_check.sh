#!/usr/bin/env bash
# Prove whether the board's Hexagon NPU is reachable from userspace.
#
#   ./scripts/board_npu_check.sh      (or: make board-npu)
#
# Deliberately installs nothing and needs no sudo: it downloads the QAIRT
# .debs with `apt download`, unpacks them into ~/qairt-probe on the board, and
# runs Qualcomm's own platform validator against the extracted libraries.
#
# Last run: PASSED — Hexagon Architecture V75, unit test executed on the DSP.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

PROBE=/home/arduino/qairt-probe

adb shell "
set -e
mkdir -p ${PROBE} && cd ${PROBE}

# Fetch once. apt download needs no root; dpkg-deb -x unpacks without installing.
for p in qairt-libs qairt-dsp-binaries qairt-tools; do
  ls \${p}_*.deb >/dev/null 2>&1 || apt download \$p
done
[ -d extracted ] || { for f in qairt-libs_*.deb qairt-dsp-binaries_*.deb; do dpkg-deb -x \$f extracted/; done; }
[ -d tools ]     || dpkg-deb -x qairt-tools_*.deb tools/
chmod +x tools/usr/bin/qnn-platform-validator

export LD_LIBRARY_PATH=${PROBE}/extracted/usr/lib:${PROBE}/tools/usr/lib

# The Hexagon skel libraries are loaded onto the DSP over FastRPC, so the
# runtime needs to be told where they live. This board is QCS8300 / v75.
export ADSP_LIBRARY_PATH=${PROBE}/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp

mkdir -p out
./tools/usr/bin/qnn-platform-validator --backend dsp --coreVersion --testBackend \
  --targetPath ${PROBE}/out 2>&1 | grep -vE '^PF_VALIDATOR: DEBUG'
" 2>&1 | tr -d '\r'

cat <<'EOF'

If "Unit Test: Passed" appeared above, the NPU runtime half is proven: the
Hexagon is reachable and executing code. What remains is the ahead-of-time
compiler — see docs/qualcomm-npu.md.
EOF
