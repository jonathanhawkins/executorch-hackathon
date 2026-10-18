#!/usr/bin/env python3
"""Run a .pte on the board. Executes ON the Arduino VENTUNO Q, not the Mac.

Pushed and invoked by scripts/board_run.sh. Kept separate from run_model.py
because it must work from a self-contained directory — the board shouldn't
need HF credentials or network to load a tokenizer.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# The board has no GPU; torch's aarch64 wheel is the CUDA build and will
# otherwise spend time probing for devices that aren't there.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Leave one core for the OS so the shell stays responsive during generation.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))


def describe_board() -> str:
    bits = []
    try:
        model = Path("/proc/device-tree/model").read_bytes().decode(errors="ignore").strip("\x00")
        if model:
            bits.append(model)
    except OSError:
        pass
    try:
        soc = Path("/sys/devices/soc0/machine").read_text().strip()
        if soc:
            bits.append(soc)
    except OSError:
        pass
    bits.append(f"{os.cpu_count()} cores")
    return " · ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a .pte on this board")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--prompt", default="Explain what on-device AI is, in two sentences.")
    ap.add_argument("--max-seq-len", type=int, default=160)
    ap.add_argument("--raw", action="store_true", help="skip the chat template")
    args = ap.parse_args()

    model_dir = args.model_dir
    ptes = sorted(model_dir.glob("*.pte"))
    if not ptes:
        print(f"error: no .pte in {model_dir}", file=sys.stderr)
        return 1
    pte = ptes[0]

    print(f"  board     {describe_board()}")
    print(f"  model     {pte.name}  ({pte.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  threads   {os.environ['OMP_NUM_THREADS']}")

    from optimum.executorch import ExecuTorchModelForCausalLM
    from transformers import AutoTokenizer

    load_start = time.time()
    # Both load from the pushed directory — no network, no HF token.
    model = ExecuTorchModelForCausalLM.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print(f"  load      {time.time() - load_start:.2f}s")

    prompt = args.prompt
    if not args.raw and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    print(f"\n> {args.prompt}\n")

    gen_start = time.time()
    text = model.text_generation(
        tokenizer=tokenizer,
        prompt=prompt,
        max_seq_len=args.max_seq_len,
    )
    elapsed = time.time() - gen_start

    print(text if isinstance(text, str) else "\n".join(text))
    print(f"\n  wall clock: {elapsed:.2f}s  (real on-device number)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
