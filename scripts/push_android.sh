#!/usr/bin/env bash
# Push an exported model + its tokenizer to an Android device for the
# Mobile/XR track (Galaxy S25 Ultra, Galaxy XR).
#
#   ./scripts/push_android.sh smollm2-135m
#
# The ExecuTorch LlamaDemo app reads from /data/local/tmp/llama by default.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:-smollm2-135m}"
SRC_DIR="models/${MODEL}"
DEST_DIR="${ANDROID_DEST:-/data/local/tmp/llama}"

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not found. Install it:" >&2
  echo "  brew install --cask android-platform-tools" >&2
  exit 1
fi

if [ ! -d "$SRC_DIR" ]; then
  echo "No export at ${SRC_DIR}." >&2
  echo "Export it first:  make export MODEL=${MODEL}" >&2
  exit 1
fi

PTE="$(find "$SRC_DIR" -maxdepth 1 -name '*.pte' | head -1)"
if [ -z "$PTE" ]; then
  echo "No .pte inside ${SRC_DIR}." >&2
  exit 1
fi

DEVICES="$(adb devices | awk 'NR>1 && /device$/ {print $1}')"
if [ -z "$DEVICES" ]; then
  echo "No device attached. Enable USB debugging and check \`adb devices\`." >&2
  exit 1
fi
echo "==> Device(s): ${DEVICES}"

echo "==> mkdir ${DEST_DIR}"
adb shell mkdir -p "${DEST_DIR}"

echo "==> Pushing $(basename "$PTE") ($(du -h "$PTE" | cut -f1))"
adb push "$PTE" "${DEST_DIR}/"

# The runner needs the tokenizer next to the model. optimum writes it into the
# export dir; push whichever form this model uses.
for tok in tokenizer.json tokenizer.model tokenizer_config.json chat_template.jinja special_tokens_map.json; do
  if [ -f "${SRC_DIR}/${tok}" ]; then
    echo "==> Pushing ${tok}"
    adb push "${SRC_DIR}/${tok}" "${DEST_DIR}/"
  fi
done

echo ""
echo "==> On device:"
adb shell ls -lh "${DEST_DIR}"
echo ""
echo "Next: build and install the demo app, see docs/android.md"
