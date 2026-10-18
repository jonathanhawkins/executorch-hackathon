#!/usr/bin/env bash
# Push a model to the Arduino VENTUNO Q and run it there.
#
#   ./scripts/board_run.sh qwen2.5-0.5b "Explain edge AI in one sentence."
#
# The board is plain aarch64 Linux reachable over adb, not Android — so this
# copies the same .pte the Mac runs and executes it with the same Python
# runner, rather than going through an APK.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

MODEL="${1:-qwen2.5-0.5b}"
PROMPT="${2:-Explain what on-device AI is, in two sentences.}"
MAX_TOKENS="${MAX_TOKENS:-160}"

SRC="models/${MODEL}"
BOARD_HOME=/home/arduino
BOARD_DIR="${BOARD_HOME}/models/${MODEL}"
VENV="${BOARD_HOME}/et-venv"

if [ ! -d "$SRC" ]; then
  echo "No export at ${SRC}. Run:  make export MODEL=${MODEL}" >&2
  exit 1
fi

PTE="$(find "$SRC" -maxdepth 1 -name '*.pte' | head -1)"
[ -n "$PTE" ] || { echo "No .pte in ${SRC}" >&2; exit 1; }

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

echo "==> Board: $(adb shell 'uname -m; nproc' | tr -d '\r' | paste -sd' ' -) cores"

# Skip the push when the board already has a byte-identical copy — the .pte is
# hundreds of MB and USB is the slow part of this loop.
LOCAL_SIZE="$(stat -f%z "$PTE" 2>/dev/null || stat -c%s "$PTE")"
REMOTE_SIZE="$(adb shell "stat -c%s '${BOARD_DIR}/$(basename "$PTE")' 2>/dev/null || echo 0" | tr -d '\r')"

adb shell "mkdir -p '${BOARD_DIR}'"
if [ "$LOCAL_SIZE" = "$REMOTE_SIZE" ]; then
  echo "==> $(basename "$PTE") already on board ($(echo "$LOCAL_SIZE" | awk '{printf "%.0f MB", $1/1048576}')), skipping push"
else
  echo "==> Pushing $(basename "$PTE") ($(echo "$LOCAL_SIZE" | awk '{printf "%.0f MB", $1/1048576}'))"
  adb push "$PTE" "${BOARD_DIR}/"
fi

for f in tokenizer.json tokenizer_config.json chat_template.jinja special_tokens_map.json tokenizer.model; do
  [ -f "${SRC}/${f}" ] && adb push "${SRC}/${f}" "${BOARD_DIR}/" >/dev/null 2>&1 || true
done

adb push scripts/board_infer.py "${BOARD_HOME}/board_infer.py" >/dev/null

echo "==> Running on board"
adb shell "cd ${BOARD_HOME} && ${VENV}/bin/python board_infer.py \
  --model-dir '${BOARD_DIR}' \
  --prompt \"${PROMPT}\" \
  --max-seq-len ${MAX_TOKENS}" | tr -d '\r'
