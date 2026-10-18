#!/bin/bash
# Runs ON the board. Builds qnn_executor_runner with the persistent-stdin
# patch, leaving the bundle's original binary untouched.
#
# Not `cmake --build`: the bundle's build-x86 tree was configured on another
# machine, at /home/arduino/derrjohn_local/git/executorch, against /usr/bin/c++
# and /usr/bin/gmake. None of those exist here, so cmake would reconfigure and
# fail. The tree does record exactly how this target was compiled and linked,
# so we replay flags.make and link.txt with the stale prefix rewritten and our
# own compiler substituted. Only qnn_executor_runner.cpp is recompiled; every
# other object and archive in the link line is the bundle's, unchanged.
set -eo pipefail

ET=/home/arduino/git/executorch
DIR=$ET/build-x86/examples/qualcomm/executor_runner
CMF=$DIR/CMakeFiles/qnn_executor_runner.dir
STALE=/home/arduino/derrjohn_local/git/executorch
OUT=qnn_executor_runner_persistent

. /home/arduino/toolchain/env.sh
cd "$DIR"

# The prefix is rewritten only where it begins a real argument — start of
# line, after a space, or after a comma (-Wl,-rpath,...). The prebuilt .o
# files live in directories that EMBED the old path as a name, so a blanket
# substitution renames them out of existence.
fix_args() { sed -E "s#(^| |,)${STALE}#\1${ET}#g"; }
fix_all()  { sed -E "s#${STALE}#${ET}#g"; }

DEFINES=$(grep '^CXX_DEFINES =' "$CMF/flags.make" | cut -d= -f2- | fix_all)
INCLUDES=$(grep '^CXX_INCLUDES =' "$CMF/flags.make" | cut -d= -f2- | fix_all)
CXXFLAGS=$(grep '^CXX_FLAGS =' "$CMF/flags.make" | cut -d= -f2- | fix_all)

echo "==> Compiling qnn_executor_runner.cpp"
# shellcheck disable=SC2086
g++ $DEFINES $INCLUDES $CXXFLAGS \
  -c "$ET/examples/qualcomm/executor_runner/qnn_executor_runner.cpp" \
  -o "$CMF/${OUT}.cpp.o"

echo "==> Linking"
LINK=$(fix_args < "$CMF/link.txt" \
  | sed "s#^/usr/bin/c++#g++#" \
  | sed "s#CMakeFiles/qnn_executor_runner\.dir/qnn_executor_runner\.cpp\.o#${CMF}/${OUT}.cpp.o#" \
  | sed "s#-o qnn_executor_runner #-o ${OUT} #")
eval "$LINK"

echo "==> Result"
ls -la "$DIR/$OUT"
# `strings` is not installed on this image — grep -a the binary instead.
if grep -aq FRAME_READY "$DIR/$OUT"; then
  echo "    FRAME_READY present — the patch is in this binary"
else
  echo "    FRAME_READY MISSING — wrong source got compiled" >&2
  exit 1
fi
