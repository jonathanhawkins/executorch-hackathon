#!/usr/bin/env bash
# Install a C/C++ toolchain on the board without root.
#
#   ./scripts/board_toolchain.sh            # download, unpack, verify
#   ./scripts/board_toolchain.sh --verify   # just re-run the compile check
#
# The board's image has no compiler at all — no gcc, no headers, no crt
# objects — and `sudo` there needs a password nobody has written down. The same
# trick that got the QAIRT and ffmpeg libraries on (apt download + dpkg-deb -x,
# neither of which needs root) works for a toolchain too; it is just more
# packages, and gcc has to be told where its headers went.
#
# Everything lands under ~/toolchain. Source ~/toolchain/env.sh to use it.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

BOARD=/home/arduino
PREFIX=$BOARD/toolchain
ROOT=$PREFIX/root
TRIPLE=aarch64-linux-gnu

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

if [ "${1:-}" != "--verify" ]; then
  echo "==> Resolving the dependency closure"
  adb shell "
set -e
mkdir -p $PREFIX/debs $ROOT
cd $PREFIX/debs
apt-get install --print-uris -y build-essential cmake ninja-build 2>/dev/null \
  | grep -oE \"'http[^']+'\" | tr -d \\\"\\'\\\" > urls.txt
echo \"    \$(wc -l < urls.txt) packages\"
" 2>&1 | tr -d '\r'

  echo "==> Downloading"
  adb shell "
set -e
cd $PREFIX/debs
while read -r u; do
  f=\$(basename \"\$u\")
  [ -f \"\$f\" ] || curl -sSL -o \"\$f\" \"\$u\"
done < urls.txt
echo \"    \$(ls *.deb | wc -l) debs, \$(du -sh . | cut -f1)\"
" 2>&1 | tr -d '\r'

  echo "==> Unpacking (no root, nothing registered with dpkg)"
  adb shell "
set -e
cd $PREFIX/debs
for f in *.deb; do dpkg-deb -x \"\$f\" $ROOT/; done
echo \"    \$(du -sh $ROOT | cut -f1) unpacked\"
" 2>&1 | tr -d '\r'
fi

echo "==> Repointing libc's linker script"
# libc.so is a GNU ld script, not a library, and it names libc_nonshared.a by
# an absolute /usr path that only exists when libc6-dev is properly installed.
# The shared objects it also names (libc.so.6, the loader) DO exist in the real
# root, so only the .a reference is rewritten — repointing the rest at the
# prefix would break linking against the system libc.
adb shell "
set -e
L=$ROOT/usr/lib/${TRIPLE}
if [ -f \$L/libc.so ] && grep -q '/usr/lib/${TRIPLE}/libc_nonshared.a' \$L/libc.so; then
  sed -i 's|/usr/lib/${TRIPLE}/libc_nonshared.a|'\$L'/libc_nonshared.a|' \$L/libc.so
  echo '    libc.so repointed'
else
  echo '    already repointed'
fi
" 2>&1 | tr -d '\r'

echo "==> Writing the environment and compiler wrappers"
# Three things break when a toolchain is unpacked outside /usr, and all three
# are fixed here rather than left for the caller:
#
#   * gcc relocates its search for cc1/cc1plus to the prefix it was unpacked
#     into. cc1plus is there; cc1 is NOT, because cpp-13 is already installed
#     system-wide so apt never downloaded it. Both -B paths are passed, ours
#     first, so each compiler proper is found wherever it actually lives.
#   * `as` and `ld` need libbfd and libopcodes, which are in the prefix and
#     not on the loader's path — hence LD_LIBRARY_PATH.
#   * The system headers and crt objects live under the prefix, not /usr.
adb shell "
set -e
mkdir -p $PREFIX/bin

real_gcc=\$(ls $ROOT/usr/bin/${TRIPLE}-gcc-* 2>/dev/null | head -1)
real_gxx=\$(ls $ROOT/usr/bin/${TRIPLE}-g++-* 2>/dev/null | head -1)
test -n \"\$real_gcc\" || { echo 'no gcc in the unpacked tree' >&2; exit 1; }

libexec_ours=\$(ls -d $ROOT/usr/libexec/gcc/${TRIPLE}/* 2>/dev/null | head -1)
libexec_sys=\$(ls -d /usr/libexec/gcc/${TRIPLE}/* 2>/dev/null | head -1)
cxx_inc=\$(ls -d $ROOT/usr/include/c++/* 2>/dev/null | head -1)
cxx_tgt=\$(ls -d $ROOT/usr/include/${TRIPLE}/c++/* 2>/dev/null | head -1)
libdir=$ROOT/usr/lib/${TRIPLE}

write_wrapper() {
  out=\$1; compiler=\$2; shift 2
  {
    echo '#!/bin/sh'
    echo \"export LD_LIBRARY_PATH=\$libdir:\\\${LD_LIBRARY_PATH:-}\"
    echo \"exec \$compiler \\\\\"
    echo \"  -B\$libexec_ours -B\$libexec_sys -B\$libdir \\\\\"
    for inc in \"\$@\"; do echo \"  -isystem \$inc \\\\\"; done
    echo \"  -L\$libdir \\\"\\\$@\\\"\"
  } > \$out
  chmod +x \$out
}

write_wrapper $PREFIX/bin/gcc \"\$real_gcc\" \
  $ROOT/usr/include/${TRIPLE} $ROOT/usr/include
write_wrapper $PREFIX/bin/g++ \"\$real_gxx\" \
  \"\$cxx_inc\" \"\$cxx_tgt\" $ROOT/usr/include/${TRIPLE} $ROOT/usr/include

ln -sf $PREFIX/bin/gcc $PREFIX/bin/cc
ln -sf $PREFIX/bin/g++ $PREFIX/bin/c++

cat > $PREFIX/env.sh <<ENV
# Source this to use the unpacked toolchain:  . ~/toolchain/env.sh
export PATH=$PREFIX/bin:$ROOT/usr/bin:\\\$PATH
export LD_LIBRARY_PATH=\$libdir:\\\${LD_LIBRARY_PATH:-}
export CC=$PREFIX/bin/gcc
export CXX=$PREFIX/bin/g++
export CMAKE_ROOT=\$(ls -d $ROOT/usr/share/cmake-* 2>/dev/null | head -1)
ENV
echo '    wrappers + env.sh written'
" 2>&1 | tr -d '\r'

echo "==> Verifying"
adb shell "
. $PREFIX/env.sh
cd /tmp
printf '#include <stdio.h>\nint main(){puts(\"C ok\");return 0;}\n' > t.c
printf '#include <vector>\n#include <iostream>\nint main(){std::vector<int> v{1,2,3};std::cout<<\"C++ ok \"<<v.size()<<std::endl;}\n' > t.cpp
gcc t.c -o tc && ./tc
g++ -std=c++17 t.cpp -o tcpp && ./tcpp
echo \"gcc     \$(gcc --version | head -1)\"
echo \"cmake   \$(cmake --version 2>/dev/null | head -1 || echo MISSING)\"
echo \"ninja   \$(ninja --version 2>/dev/null || echo MISSING)\"
" 2>&1 | tr -d '\r'
