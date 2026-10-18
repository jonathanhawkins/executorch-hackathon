#!/usr/bin/env bash
# Build qnn_executor_runner with the persistent-stdin patch, on the board.
#
#   ./scripts/board_build_runner.sh      (or: make board-build-runner)
#
# The build itself runs on the board — scripts/board/build_runner.sh — because
# replaying the recorded compile and link lines needs several layers of sed,
# and threading those through an inline `adb shell "..."` mangles the quoting.
# An earlier version of this script tried that and silently produced a broken
# link line, which then destroyed the previously working binary. Pushing a real
# script removes the whole class of problem.
#
# Leaves the bundle's original binary untouched; the patched one is
# qnn_executor_runner_persistent beside it. See docs/persistent-runner.md.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

adb shell "test -f /home/arduino/toolchain/env.sh" 2>/dev/null || {
  echo "No toolchain on the board. Run: ./scripts/board_toolchain.sh" >&2
  exit 1
}

adb push scripts/board/build_runner.sh /home/arduino/build_runner.sh >/dev/null
adb shell "bash /home/arduino/build_runner.sh" 2>&1 | tr -d '\r'
