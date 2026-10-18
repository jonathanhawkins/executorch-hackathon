#!/usr/bin/env python3
"""CLIP-tokenize a Stable Diffusion prompt into raw int32 tensors for the board.

Writes tokens_cond.raw and tokens_uncond.raw, each a [1, 77] int32 buffer that
the text encoder's QNN graph takes directly. Runs on the Mac so the board needs
no tokenizer at all — the same split as make_whisper_raw.py and make_face_raw.py.

Stable Diffusion 2.1 shares OpenAI CLIP's BPE vocabulary, which is why the
bundle's README points at openai/clip-vit-base-patch32 for vocab.json. It does
NOT share the padding: SD 2.x sets pad_token to "!" (id 0), so the 77-token
window is zero-padded after the EOS rather than filled with EOS. The bundle's
own C++ runner pads with 0 for the same reason.

The BPE is implemented here rather than imported because transformers 5.0.0rc1
— which optimum-executorch pins, see CLAUDE.md — ships a CLIPTokenizer that
raises on construction against the installed `tokenizers`. All this needs is
vocab.json and merges.txt, which is what the bundle's README asks for anyway.
Ported from CLIP's own simple_tokenizer.py.
"""

import argparse
import functools
import json
import os
import re

import numpy as np

MAX_TOKENS = 77
# Stable Diffusion 2.x sets pad_token to "!" (id 0); 1.x pads with EOS (49407).
# Getting this wrong quietly degrades conditioning — the padding is 65 of the
# 77 positions on a short prompt, and the text encoder attends to all of them.
PAD_SD2, PAD_SD15 = 0, 49407
BOS, EOS = "<|startoftext|>", "<|endoftext|>"

# CLIP's pretokenizer. The original uses \p{L} and \p{N} via the `regex`
# module; [^\W\d_] and \d are the stdlib `re` equivalents.
PATTERN = re.compile(
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d"
    r"|[^\W\d_]+|\d|[^\s\w]+|_+",
    re.IGNORECASE,
)


@functools.lru_cache(maxsize=1)
def bytes_to_unicode() -> dict:
    """CLIP's reversible byte <-> printable-codepoint map."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    code_points = list(byte_values)
    spare = 0
    for byte in range(2**8):
        if byte not in byte_values:
            byte_values.append(byte)
            code_points.append(2**8 + spare)
            spare += 1
    return {b: chr(c) for b, c in zip(byte_values, code_points)}


class ClipBPE:
    def __init__(self, vocab_json: str, merges_txt: str):
        with open(vocab_json, encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(merges_txt, encoding="utf-8") as f:
            lines = f.read().split("\n")
        # First line is a version header; the last is usually blank.
        merges = [tuple(line.split()) for line in lines[1:] if len(line.split()) == 2]
        self.ranks = {merge: i for i, merge in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.cache = {BOS: BOS, EOS: EOS}

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        # The end of a word is marked on its last character.
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = set(zip(word[:-1], word[1:]))
            bigram = min(pairs, key=lambda pair: self.ranks.get(pair, float("inf")))
            if bigram not in self.ranks:
                break
            first, second = bigram
            merged, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    merged.extend(word[i:])
                    break
                merged.extend(word[i:j])
                if j < len(word) - 1 and word[j + 1] == second:
                    merged.append(first + second)
                    i = j + 2
                else:
                    merged.append(word[j])
                    i = j + 1
            word = tuple(merged)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list:
        text = re.sub(r"\s+", " ", text).strip().lower()
        ids = []
        for chunk in PATTERN.findall(text):
            encoded = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            for piece in self._bpe(encoded).split(" "):
                if piece not in self.vocab:
                    raise KeyError(f"token {piece!r} is not in the CLIP vocabulary")
                ids.append(self.vocab[piece])
        return ids


def tokenizer_files(repo: str):
    """vocab.json and merges.txt, from the HF cache if they are already there."""
    from huggingface_hub import hf_hub_download

    paths = []
    for filename in ("vocab.json", "merges.txt"):
        try:
            paths.append(hf_hub_download(repo, filename, local_files_only=True))
        except Exception:
            paths.append(hf_hub_download(repo, filename))
    return paths


def main(args):
    pad_token = PAD_SD15 if args.base == "sd15" else PAD_SD2
    vocab_json, merges_txt = tokenizer_files(args.tokenizer)
    bpe = ClipBPE(vocab_json, merges_txt)
    os.makedirs(args.out, exist_ok=True)

    for name, prompt in (("cond", args.prompt), ("uncond", args.negative_prompt)):
        ids = [bpe.vocab[BOS]] + bpe.encode(prompt)
        ids = ids[: MAX_TOKENS - 1] + [bpe.vocab[EOS]]
        padded = ids + [pad_token] * (MAX_TOKENS - len(ids))
        tokens = np.asarray(padded, dtype=np.int32).reshape(1, MAX_TOKENS)

        path = os.path.join(args.out, f"tokens_{name}.raw")
        tokens.tofile(path)
        print(f"{os.path.basename(path)}  {len(ids)} tokens, pad {pad_token}  {ids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument(
        "-n",
        "--negative-prompt",
        default="",
        help="Unconditional prompt for classifier-free guidance. Default empty.",
    )
    parser.add_argument("-o", "--out", required=True, help="Directory to write the raws into")
    parser.add_argument(
        "--base",
        choices=("sd21", "sd15"),
        default="sd21",
        help="Which Stable Diffusion family, because they pad differently.",
    )
    parser.add_argument(
        "--tokenizer",
        default="openai/clip-vit-base-patch32",
        help="Any repo carrying CLIP's BPE vocab; the vocabulary is the same one.",
    )
    main(parser.parse_args())
