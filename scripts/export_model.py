#!/usr/bin/env python3
"""Export a Hugging Face LLM to an ExecuTorch .pte file.

Thin, legible wrapper over `optimum-cli export executorch` that keeps the model
presets in models.toml so the team edits one file instead of pasting flags.

    python scripts/export_model.py --list
    python scripts/export_model.py smollm2-135m
    python scripts/export_model.py llama3.2-1b --recipe coreml
    python scripts/export_model.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRESETS_FILE = REPO / "models.toml"
DEFAULT_OUTDIR = REPO / "models"


def load_presets() -> dict:
    with PRESETS_FILE.open("rb") as fh:
        presets = tomllib.load(fh).get("presets", {})
    # An unquoted dot in a TOML key nests tables instead of naming a preset,
    # which otherwise shows up much later as a confusing KeyError.
    bad = [n for n, cfg in presets.items() if not isinstance(cfg, dict) or "model_id" not in cfg]
    if bad:
        raise SystemExit(
            f"models.toml: preset(s) {bad} have no model_id.\n"
            f"If the name contains a dot, quote it: [presets.\"llama3.2-1b\"]"
        )
    return presets


def human_size(num_bytes: int) -> str:
    val = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024 or unit == "GB":
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} GB"


def print_presets(presets: dict) -> None:
    print("\nAvailable presets (models.toml):\n")
    width = max(len(n) for n in presets)
    for name, cfg in presets.items():
        lock = "[gated]" if cfg.get("gated") else "       "
        print(f"  {name:<{width}}  {lock}  {cfg['model_id']}")
        if cfg.get("note"):
            print(f"  {'':<{width}}           {cfg['note']}")
    print()


def build_command(cfg: dict, outdir: Path) -> list[str]:
    cmd = [
        "optimum-cli", "export", "executorch",
        "--model", cfg["model_id"],
        "--task", cfg.get("task", "text-generation"),
        "--recipe", cfg.get("recipe", "xnnpack"),
        "--output_dir", str(outdir),
    ]
    # Fused SDPA + a ring KV cache. Both are large wins on device and are the
    # documented default for decoder-only models; skip only to isolate a bug.
    if cfg.get("use_custom_sdpa", True):
        cmd.append("--use_custom_sdpa")
    if cfg.get("use_custom_kv_cache", True):
        cmd.append("--use_custom_kv_cache")
    if cfg.get("qlinear"):
        cmd += ["--qlinear", str(cfg["qlinear"])]
    if cfg.get("qembedding"):
        cmd += ["--qembedding", str(cfg["qembedding"])]
    if cfg.get("max_seq_len"):
        cmd += ["--max_seq_len", str(cfg["max_seq_len"])]
    return cmd


def check_access(model_id: str) -> bool:
    """Fail fast and legibly on a gated repo.

    Note that huggingface_hub's model_info() succeeds on gated repos you cannot
    actually download, so the only honest test is fetching a real file.
    """
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(model_id, "config.json")
        return True
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "403" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
            print(f"\n  No access to {model_id}.", file=sys.stderr)
            print(f"  Open https://huggingface.co/{model_id} and accept the license,", file=sys.stderr)
            print("  then check you are logged in:  .venv/bin/hf auth login\n", file=sys.stderr)
            return False
        # Anything else (offline, rate limit) shouldn't block the attempt.
        print(f"  warning: access precheck inconclusive ({type(exc).__name__})")
        return True


def save_tokenizer(model_id: str, outdir: Path) -> None:
    """Drop the tokenizer next to the .pte.

    optimum writes only the .pte, but every on-device runner needs the
    tokenizer alongside it, so fetch it now rather than at `adb push` time.
    """
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(model_id).save_pretrained(str(outdir))
        print(f"  tokenizer saved to {outdir.name}/")
    except Exception as exc:  # noqa: BLE001 - non-fatal, the .pte is still good
        print(f"  warning: could not save tokenizer ({exc})", file=sys.stderr)


def main() -> int:
    presets = load_presets()

    ap = argparse.ArgumentParser(
        description="Export an LLM to ExecuTorch .pte",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("preset", nargs="?", help="preset name from models.toml")
    ap.add_argument("--list", action="store_true", help="list presets and exit")
    ap.add_argument("--model", help="HF model id, bypassing the presets")
    ap.add_argument("--recipe", help="override backend: xnnpack, coreml, portable, cuda")
    ap.add_argument("--qlinear", help="override linear quant, e.g. 8da4w, or 'none'")
    ap.add_argument("--qembedding", help="override embedding quant, e.g. 4w, or 'none'")
    ap.add_argument("--max-seq-len", type=int, help="override max sequence length")
    ap.add_argument("--output-dir", type=Path, help="defaults to models/<name>")
    ap.add_argument("--dry-run", action="store_true", help="print the command, run nothing")
    args = ap.parse_args()

    if args.list or (not args.preset and not args.model):
        print_presets(presets)
        return 0

    if args.preset and args.preset not in presets and not args.model:
        print(f"error: unknown preset {args.preset!r}", file=sys.stderr)
        print_presets(presets)
        return 2

    if args.model:
        name = args.model.split("/")[-1].lower()
        cfg = {"model_id": args.model, "recipe": "xnnpack", "qlinear": "8da4w",
               "qembedding": "8w", "max_seq_len": 2048}
    else:
        name = args.preset
        cfg = dict(presets[name])

    # CLI overrides win over the preset. 'none' disables a quant step.
    for key, val in (("recipe", args.recipe), ("qlinear", args.qlinear),
                     ("qembedding", args.qembedding), ("max_seq_len", args.max_seq_len)):
        if val is not None:
            cfg[key] = None if str(val).lower() == "none" else val

    outdir = args.output_dir or (DEFAULT_OUTDIR / name)
    cmd = build_command(cfg, outdir)

    print(f"\n  model   {cfg['model_id']}")
    print(f"  backend {cfg.get('recipe')}")
    print(f"  quant   linear={cfg.get('qlinear') or 'off'} embedding={cfg.get('qembedding') or 'off'}")
    print(f"  output  {outdir}")
    if cfg.get("gated"):
        print("\n  This model is gated. Accept its license on the HF model page and")
        print("  run `hf auth login` if the export 401s.")
    print("\n$ " + " ".join(cmd) + "\n")

    if args.dry_run:
        return 0

    if shutil.which("optimum-cli") is None:
        print("error: optimum-cli not found. Activate the venv:", file=sys.stderr)
        print("  source .venv/bin/activate", file=sys.stderr)
        return 127

    if not check_access(cfg["model_id"]):
        return 1

    outdir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nExport failed after {elapsed:.0f}s (exit {result.returncode}).", file=sys.stderr)
        return result.returncode

    save_tokenizer(cfg["model_id"], outdir)

    ptes = sorted(outdir.glob("*.pte"))
    print(f"\nExported in {elapsed:.0f}s")
    for pte in ptes:
        print(f"  {pte.relative_to(REPO)}  {human_size(pte.stat().st_size)}")
    if ptes:
        print(f"\nTry it:  python scripts/run_model.py {name} --prompt 'Hello'")
    else:
        print("  warning: no .pte landed in the output dir", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
