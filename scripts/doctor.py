#!/usr/bin/env python3
"""Check the toolchain before you lose an hour to a broken environment.

    python scripts/doctor.py
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<22} {detail}")


def check_python() -> bool:
    v = sys.version_info
    good = (3, 10) <= (v.major, v.minor) < (3, 15)
    line(OK if good else BAD, "python", f"{platform.python_version()} ({sys.executable})")
    if not good:
        print("         ExecuTorch needs >=3.10,<3.15. Re-run ./scripts/setup.sh")
    return good


def check_packages() -> bool:
    required = ["executorch", "torch", "transformers", "optimum", "optimum-executorch", "torchao"]
    all_good = True
    for pkg in required:
        try:
            line(OK, pkg, md.version(pkg))
        except md.PackageNotFoundError:
            line(BAD, pkg, "not installed")
            all_good = False
    return all_good


def check_runtime() -> bool:
    """Import the runtime and lower a trivial model — catches broken wheels."""
    try:
        import torch
        from executorch.exir import to_edge_transform_and_lower
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        sample = (torch.ones(2, 2), torch.ones(2, 2))
        exported = torch.export.export(Add(), sample)
        prog = to_edge_transform_and_lower(
            exported, partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        from executorch.runtime import Runtime

        runtime = Runtime.get()
        program = runtime.load_program(prog.buffer)
        method = program.load_method("forward")
        out = method.execute(list(sample))
        good = bool(torch.allclose(out[0], torch.full((2, 2), 2.0)))
        line(OK if good else BAD, "xnnpack round-trip", "lowered, ran, and got 2+2" if good else "wrong result")
        return good
    except Exception as exc:  # noqa: BLE001 - we want the message, whatever it is
        line(BAD, "xnnpack round-trip", f"{type(exc).__name__}: {exc}")
        return False


def check_cli() -> bool:
    found = shutil.which("optimum-cli")
    if found:
        line(OK, "optimum-cli", found)
        return True
    venv_cli = REPO / ".venv" / "bin" / "optimum-cli"
    if venv_cli.exists():
        line(WARN, "optimum-cli", f"exists but not on PATH: {venv_cli}")
        print("         run `source .venv/bin/activate`, or use the make targets")
        return True
    line(BAD, "optimum-cli", "not found")
    return False


def check_hf_auth() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        line(OK, "hf auth", "token in environment")
        return
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists() and cached.stat().st_size > 0:
        line(OK, "hf auth", f"token cached at {cached}")
    else:
        line(WARN, "hf auth", "no token — gated models (Llama, Gemma) will 401")
        print("         fix: .venv/bin/hf auth login")


def check_adb() -> None:
    found = shutil.which("adb")
    if not found:
        line(WARN, "adb", "not installed — needed only for the Android track")
        print("         fix: brew install --cask android-platform-tools")
        return
    try:
        out = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=10).stdout
        devices = [l.split()[0] for l in out.splitlines()[1:] if l.strip() and "device" in l]
        if not devices:
            line(WARN, "adb", "installed, no device attached")
            return

        # adb also reaches the Arduino VENTUNO Q, which runs Ubuntu rather than
        # Android. Saying which is attached avoids a long detour building an
        # APK for a board that cannot install one.
        uname = subprocess.run(
            ["adb", "shell", "uname", "-s"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
        if "Linux" in uname:
            model = subprocess.run(
                ["adb", "shell", "getprop", "ro.product.model"],
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()
            if model:
                line(OK, "adb device", f"Android: {model} ({devices[0]})")
            else:
                soc = subprocess.run(
                    ["adb", "shell", "cat", "/sys/devices/soc0/machine"],
                    capture_output=True, text=True, timeout=15,
                ).stdout.strip() or "unknown SoC"
                line(OK, "adb device", f"Linux board: {soc} ({devices[0]}) — not Android")
                print("         use the board-* make targets, not apk/deploy")
        else:
            line(OK, "adb device", f"{len(devices)} attached: {', '.join(devices)}")
    except Exception as exc:  # noqa: BLE001
        line(WARN, "adb", f"{type(exc).__name__}: {exc}")


def check_android_build() -> None:
    """The Android side needs a JDK and an SDK that Gradle can find."""
    sdk = os.environ.get("ANDROID_HOME") or "/opt/homebrew/share/android-commandlinetools"
    if not Path(sdk).exists():
        line(WARN, "android sdk", "not found — needed only to build the app")
        print("         fix: brew install --cask android-commandlinetools")
        return
    platforms = sorted(p.name for p in (Path(sdk) / "platforms").glob("android-*"))
    line(OK, "android sdk", f"{sdk} ({', '.join(platforms) or 'no platforms'})")

    java = shutil.which("java")
    if java:
        try:
            out = subprocess.run(
                [java, "-version"], capture_output=True, text=True, timeout=15
            ).stderr.splitlines()[0]
            line(OK, "java", out)
        except Exception as exc:  # noqa: BLE001
            line(WARN, "java", f"{type(exc).__name__}: {exc}")
    else:
        line(WARN, "java", "not on PATH — fix: brew install openjdk@21")

    if (REPO / "android" / "gradlew").exists():
        line(OK, "gradle wrapper", "android/gradlew")
    else:
        line(WARN, "gradle wrapper", "missing — run `gradle wrapper` in android/")

    apk = REPO / "android/app/build/outputs/apk/debug/app-debug.apk"
    if apk.exists():
        line(OK, "demo apk", f"built, {apk.stat().st_size / 1024 / 1024:.0f} MB")
    else:
        line(WARN, "demo apk", "not built yet — run `make apk`")


def check_exports() -> None:
    models = REPO / "models"
    ptes = sorted(models.glob("*/*.pte")) if models.exists() else []
    if ptes:
        line(OK, "exported models", f"{len(ptes)} .pte")
        for pte in ptes:
            print(f"         {pte.relative_to(REPO)}  {pte.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        line(WARN, "exported models", "none yet — run `make smoke`")


def main() -> int:
    print(f"\nExecuTorch hackathon doctor — {platform.platform()}\n")
    critical = [check_python(), check_packages(), check_cli(), check_runtime()]
    check_hf_auth()
    check_adb()
    check_android_build()
    check_exports()

    if all(critical):
        print("\nToolchain is good. `make smoke` exports and runs a model end to end.\n")
        return 0
    print("\nSomething critical is broken — see the FAIL lines above.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
