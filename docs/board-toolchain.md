# A compiler on the board, without root

**Working.** gcc 13.3, g++ with libstdc++, cmake 3.28.3 and ninja 1.11.1 all
run on the VENTUNO Q, installed with no password:

```bash
./scripts/board_toolchain.sh            # download, unpack, wrap, verify
./scripts/board_toolchain.sh --verify   # re-check an existing install
```

Then on the board: `. ~/toolchain/env.sh`.

## Why this exists

The image ships no compiler at all — no `gcc`, no `cc`, no headers, no crt
objects — and `sudo` on the board needs a password that is not in the bring-up
PDF, not in Arduino's VENTUNO Q documentation, and not the `arduino/arduino`
that the sibling UNO Q datasheet suggests (Arduino later landed a commit
titled *"removes incorrect reference to uno-q default password"*).

The same trick that put the QAIRT and ffmpeg libraries on this board works
here: `apt download` and `dpkg-deb -x` need no root. It is just more packages —
51 — and a toolchain minds where it lives rather more than a shared library
does.

## The four things that break outside /usr

Each of these is handled by the script; they are listed because they will
recur for anything else unpacked this way.

1. **`cc1` is missing but `cc1plus` is not.** `cpp-13` is already installed
   system-wide, so apt never downloaded it, and gcc relocates its search for
   the compiler proper into the prefix it finds itself in. The C++ one is
   there; the C one is at `/usr/libexec/gcc/aarch64-linux-gnu/13/cc1` in the
   real root. The wrappers pass **both** `-B` paths, ours first.
2. **`as` and `ld` cannot start.** They need `libbfd` and `libopcodes`, which
   are in the prefix and not on the loader's path. `LD_LIBRARY_PATH`, set
   inside the wrappers so it survives being called from cmake.
3. **`libc.so` is a linker script, not a library.** It names
   `libc_nonshared.a` by an absolute `/usr` path that only exists after a real
   `libc6-dev` install. The script rewrites that one reference to the prefix
   and leaves the rest alone — the shared objects it also names (`libc.so.6`,
   the loader) do exist in the real root, and repointing those would break
   linking against the system libc.
4. **Headers and crt objects.** Ordinary `-isystem`, `-B` and `-L` flags,
   baked into the wrappers so callers need no special flags.

## Downloads happen on the Mac

`apt download` on the board is unreliable — `ports.ubuntu.com` resolves to
IPv6 addresses the board cannot route and IPv4 mirrors that time out, which
also bit the ffmpeg fetch earlier. The board resolves the dependency closure
(`apt-get install --print-uris`, which needs no root), the Mac fetches the 51
debs over its own connection, and they go across by `adb push`. 67 MB down,
256 MB unpacked.

Doing the downloads from a long-lived `adb shell` is also what surfaced the
board rebooting under load — see below.

## The board rebooted mid-install

Partway through, the board dropped off USB entirely: `ioreg` showed the Mac's
controllers up with zero devices attached. It came back on its own, and
`uptime` then read 13 minutes, so it had restarted rather than lost its cable.

At the time it was running the billboard daemon, the camera pipeline, a
Next.js server, three other sessions' work and a 51-package download. Whether
that was thermal, power or USB is unknown, but it is worth knowing before a
demo that this board will reboot under enough load. Everything here is
idempotent, which is why the install simply resumed.

## Layout

```
~/toolchain/debs/   the 51 .deb files and urls.txt
~/toolchain/root/   unpacked, mirroring /usr (256 MB)
~/toolchain/bin/    gcc, g++, cc, c++ wrappers
~/toolchain/env.sh  PATH, LD_LIBRARY_PATH, CC, CXX, CMAKE_ROOT
```

Nothing is registered with dpkg and nothing outside `~/toolchain` is touched,
so this is removable by deleting one directory.

## What it unblocks

The persistent-runner patch. `qnn_executor_runner` is relaunched for every
inference, which costs 528 ms of wall clock for 12.5 ms of face-parsing
compute, and similar on every other model. A 30-line patch adding a
`--input_list_path -` stdin loop is already written in the board's checkout
and has never been compiled. Now it can be.
