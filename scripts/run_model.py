#!/usr/bin/env python3
"""Run an exported .pte through the ExecuTorch runtime and time it.

This runs the same .pte you ship to the phone, through the same runtime, so a
number here is a real (if faster) preview of on-device behaviour.

    python scripts/run_model.py smollm2-135m
    python scripts/run_model.py smollm2-135m --prompt "Explain edge AI" --max-seq-len 256
    python scripts/run_model.py --model-dir models/custom --tokenizer Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRESETS_FILE = REPO / "models.toml"


def load_presets() -> dict:
    with PRESETS_FILE.open("rb") as fh:
        presets = tomllib.load(fh).get("presets", {})
    return {n: c for n, c in presets.items() if isinstance(c, dict) and "model_id" in c}


def main() -> int:
    presets = load_presets()

    ap = argparse.ArgumentParser(description="Run an exported ExecuTorch .pte")
    ap.add_argument("preset", nargs="?", help="preset name from models.toml")
    ap.add_argument("--model-dir", type=Path, help="export dir, defaults to models/<preset>")
    ap.add_argument("--tokenizer", help="HF id for the tokenizer, defaults to the preset's model")
    ap.add_argument("--prompt", default="Explain what on-device AI is, in two sentences.")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--raw", action="store_true",
                    help="skip the chat template and complete the prompt literally")
    args = ap.parse_args()

    if not args.preset and not args.model_dir:
        print("error: pass a preset name or --model-dir", file=sys.stderr)
        print(f"presets: {', '.join(presets)}", file=sys.stderr)
        return 2

    model_dir = args.model_dir or (REPO / "models" / args.preset)
    tokenizer_id = args.tokenizer
    if not tokenizer_id and args.preset in presets:
        tokenizer_id = presets[args.preset]["model_id"]
    if not tokenizer_id:
        print("error: could not resolve a tokenizer; pass --tokenizer", file=sys.stderr)
        return 2

    if not model_dir.exists() or not list(model_dir.glob("*.pte")):
        print(f"error: no .pte in {model_dir}", file=sys.stderr)
        print(f"export it first:  python scripts/export_model.py {args.preset or ''}".rstrip(), file=sys.stderr)
        return 1

    from optimum.executorch import ExecuTorchModelForCausalLM
    from transformers import AutoTokenizer

    pte = next(iter(sorted(model_dir.glob("*.pte"))))
    print(f"  model     {pte.relative_to(REPO)}  ({pte.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  tokenizer {tokenizer_id}")

    load_start = time.time()
    model = ExecuTorchModelForCausalLM.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    print(f"  load      {time.time() - load_start:.2f}s\n")

    # Instruct models ramble badly without their chat template — the model is
    # fine, the framing is wrong. Apply it unless asked for a raw completion.
    prompt = args.prompt
    if not args.raw and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        print("  prompt    chat template applied (--raw to skip)")

    print(f"\n> {args.prompt}\n")

    gen_start = time.time()
    text = model.text_generation(
        tokenizer=tokenizer,
        prompt=prompt,
        max_seq_len=args.max_seq_len,
    )
    elapsed = time.time() - gen_start

    print(text if isinstance(text, str) else "\n".join(text))

    # The runtime prints its own PyTorchObserver breakdown above; this is just
    # the wall-clock figure, which is the one you feel in a demo.
    print(f"\n  wall clock: {elapsed:.2f}s for the full generate() call")
    print("  (Mac CPU via XNNPACK. On-device numbers will differ.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
