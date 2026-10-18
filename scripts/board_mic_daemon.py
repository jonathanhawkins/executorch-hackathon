#!/usr/bin/env python3
"""Live mic -> Whisper -> (optional) translation, all on the Hexagon NPU.

Runs ON the board, not the Mac. Started by the website or scripts/board_mic.sh.

Captures continuously, transcribes short overlapping windows on the NPU, and
appends results to a JSONL file the web app reads. Translation, when enabled,
is a second NPU pass through Gemma 4.

Latency budget, measured on this board:
    chunk wait      1.8-4.0 s   dominant, and the only knob that really moves
    Whisper load    0.52 s      paid per invocation; the runner is one-shot
    Whisper encode  0.03 s
    Whisper decode  ~0.008 s/token
So the NPU work is nearly free and the chunk length sets the delay. Chunks are
short when transcribing alone and longer when translating, where Gemma's ~4.5 s
dwarfs everything else and short chunks would only waste NPU passes.

The NPU is used serially on purpose: Whisper and Gemma each take the whole DSP
context, so overlapping them fails rather than sharing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

BOARD = Path("/home/arduino")
ET = BOARD / "git/executorch"
PROBE = BOARD / "qairt-probe"
RUN_DIR = Path("/dev/shm/mic")
TRANSCRIPT = RUN_DIR / "transcript.jsonl"
CONTROL = RUN_DIR / "control.json"
WORK = RUN_DIR / "work"

WHISPER_DIR = BOARD / "models/whisper-tiny-npu"
WHISPER_RUNNER = ET / "build-x86/examples/qualcomm/oss_scripts/whisper/qnn_whisper_runner"
GEMMA_DIR = BOARD / "models/gemma4_e2b"
LLAMA_RUNNER = ET / "build-x86/examples/qualcomm/oss_scripts/llama/qnn_llama_runner"

MIC = os.environ.get("MIC_DEVICE", "plughw:1,0")
SAMPLE_RATE = 16_000

# How much new audio to wait for before transcribing.
FAST_CHUNK = float(os.environ.get("FAST_CHUNK", "1.8"))   # transcribe only
SLOW_CHUNK = float(os.environ.get("SLOW_CHUNK", "4.0"))   # translating too
# Carried over from the previous window so a word split across the boundary is
# still whole in at least one of them.
OVERLAP = float(os.environ.get("OVERLAP", "0.6"))

# Whisper invents text from silence — "Thank you.", "Bye." and similar are its
# favourite hallucinations. Gating on RMS is cruder than a real VAD but it
# keeps the transcript clean and saves an NPU pass per quiet window.
SILENCE_RMS = float(os.environ.get("SILENCE_RMS", "0.012"))

HALLUCINATIONS = {
    "you", "thank you.", "thank you", "thanks for watching!", "bye.",
    ".", "...", "okay.", "oh.", "bye bye.", "thanks for watching.",
    "so", "so.", "yeah.", "uh.", "hmm.",
}

NPU_ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": ":".join([
        str(ET / "build-x86/backends/qualcomm"),
        str(ET / "build-x86/lib"),
        str(PROBE / "extracted/usr/lib"),
        str(PROBE / "extracted/usr/lib/aarch64-linux-gnu"),
    ]),
    "ADSP_LIBRARY_PATH": str(
        PROBE / "extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp"
    ),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class MicStream:
    """One long-lived arecord, read continuously into a ring buffer.

    The first version spawned arecord per chunk, which both cost process
    startup and — worse — dropped every word spoken while the NPU was busy,
    because nothing was listening during inference. Persistent capture means
    audio keeps arriving whatever the NPU is doing.
    """

    def __init__(self, device: str, rate: int, seconds: float = 30.0):
        self.rate = rate
        self.buf: deque = deque()
        self.samples = 0
        self.max_samples = int(rate * seconds)
        self.lock = threading.Lock()
        self.proc = subprocess.Popen(
            ["arecord", "-D", device, "-f", "S16_LE", "-r", str(rate),
             "-c", "1", "-t", "raw", "-q", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self) -> None:
        block = 2048  # bytes, ~64ms of audio
        while True:
            data = self.proc.stdout.read(block)
            if not data:
                break
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            with self.lock:
                self.buf.append(chunk)
                self.samples += chunk.size
                while self.samples > self.max_samples:
                    dropped = self.buf.popleft()
                    self.samples -= dropped.size

    def take(self, seconds: float) -> np.ndarray:
        want = int(self.rate * seconds)
        with self.lock:
            if self.samples == 0:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(list(self.buf))
        return audio[-want:] if audio.size > want else audio

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass


def read_control() -> dict:
    try:
        return json.loads(CONTROL.read_text())
    except Exception:
        return {"translate": False, "targetLang": "Spanish"}


def transcribe(audio: np.ndarray, extractor, seq_len: int = 128) -> str:
    """One Whisper window on the NPU. Short audio is zero-padded to 30s."""
    WORK.mkdir(parents=True, exist_ok=True)
    feats = extractor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
    ).input_features  # (1, 80, 3000)
    raw = WORK / "mel_0.raw"
    feats.astype(np.float32).tofile(raw)
    (WORK / "input_list.txt").write_text("mel_0.raw\n")

    out_dir = WORK / "outputs"
    out_dir.mkdir(exist_ok=True)
    for stale in out_dir.glob("output_*.txt"):
        stale.unlink()

    subprocess.run(
        [
            str(WHISPER_RUNNER),
            "--model_path", str(WHISPER_DIR / "whisper_qnn_16a8w.pte"),
            "--tokenizer_json_path", str(WHISPER_DIR / "tokenizer.json"),
            "--input_list_path", "input_list.txt",
            "--seq_len", str(seq_len),
            "--output_folder_path", "outputs",
        ],
        cwd=WORK, env=NPU_ENV, capture_output=True, timeout=120,
    )

    parts = [p.read_text().strip() for p in sorted(out_dir.glob("output_*.txt"))]
    joined = " ".join(x for x in parts if x).strip()
    # The runner emits Whisper's special tokens verbatim. <|nocaptions|> is its
    # way of saying "no speech here", so that becomes an empty result rather
    # than a line of markup in the transcript.
    if "<|nocaptions|>" in joined:
        return ""
    return re.sub(r"<\|[^|]*\|>", "", joined).strip()


def drop_overlap(prev: str, cur: str, max_words: int = 8) -> str:
    """Strip words at the start of `cur` that repeat the end of `prev`.

    Windows overlap so a word split across the boundary survives in one of
    them, which means consecutive results often share a few words. Trim the
    longest such repeat.
    """
    if not prev or not cur:
        return cur
    pw, cw = prev.split(), cur.split()
    norm = [w.lower().strip(".,!?;:") for w in pw]
    cnorm = [w.lower().strip(".,!?;:") for w in cw]
    for n in range(min(max_words, len(norm), len(cnorm)), 0, -1):
        if norm[-n:] == cnorm[:n]:
            return " ".join(cw[n:]).strip()
    return cur


def translate(text: str, target: str) -> str:
    """Second NPU pass: Gemma 4 rewrites the transcript in another language."""
    prompt = (
        f"Translate the following into {target}. "
        f"Reply with only the translation, nothing else.\n\n{text}"
    )
    framed = f"<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    out_path = RUN_DIR / "translate_out.txt"
    pte = next(GEMMA_DIR.glob("*.pte"), None)
    if pte is None:
        return ""
    subprocess.run(
        [
            str(LLAMA_RUNNER),
            "--model_path", pte.name,
            "--tokenizer_path", "tokenizer.json",
            "--decoder_model_version", "gemma4",
            "--prompt", framed,
            "--seq_len", "320",
            "--eval_mode", "1",
            "--temperature", "0.3",
            "--output_path", str(out_path),
        ],
        cwd=GEMMA_DIR, env=NPU_ENV, capture_output=True, timeout=180,
    )
    raw = out_path.read_text(errors="ignore") if out_path.exists() else ""
    # Keep only what follows the LAST model turn marker — the runner echoes the
    # prompt, which already contains one, so splitting on the first leaves it in.
    if "<|turn>model" in raw:
        raw = raw.rsplit("<|turn>model", 1)[1]
    return raw.replace("<turn|>", "").replace("<bos>", "").strip()


def append(entry: dict) -> None:
    with TRANSCRIPT.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if not CONTROL.exists():
        CONTROL.write_text(json.dumps({"translate": False, "targetLang": "Spanish"}))

    log("loading Whisper feature extractor")
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    stream = MicStream(MIC, SAMPLE_RATE)
    log(f"listening on {MIC} (continuous capture)")

    prev_text = ""
    last_emit = time.time()
    try:
        while True:
            ctl = read_control()
            chunk = SLOW_CHUNK if ctl.get("translate") else FAST_CHUNK

            # Wait only for audio not already consumed, so inference time is
            # overlapped with capture instead of added on top of it.
            wait = chunk - (time.time() - last_emit)
            if wait > 0:
                time.sleep(wait)
            last_emit = time.time()

            if not stream.alive():
                log("capture died, restarting")
                stream = MicStream(MIC, SAMPLE_RATE)
                continue

            audio = stream.take(chunk + OVERLAP)
            if audio.size < SAMPLE_RATE * 0.4:
                continue

            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < SILENCE_RMS:
                prev_text = ""  # a pause ends the overlap relationship
                continue

            started = time.time()
            try:
                text = transcribe(audio, extractor)
            except Exception as exc:
                log(f"transcribe failed: {exc}")
                continue
            asr_sec = time.time() - started

            cleaned = drop_overlap(prev_text, text.strip())
            if not cleaned or cleaned.lower() in HALLUCINATIONS:
                continue
            prev_text = text.strip()

            entry = {
                "ts": time.time(),
                "text": cleaned,
                "rms": round(rms, 4),
                "asrSec": round(asr_sec, 2),
                "chunkSec": chunk,
            }

            if ctl.get("translate"):
                t0 = time.time()
                try:
                    entry["translation"] = translate(cleaned, ctl.get("targetLang", "Spanish"))
                    entry["targetLang"] = ctl.get("targetLang", "Spanish")
                    entry["mtSec"] = round(time.time() - t0, 2)
                except Exception as exc:
                    log(f"translate failed: {exc}")
                # Gemma just consumed several seconds of wall clock; treat now
                # as the new baseline so the next window is not already stale.
                last_emit = time.time()

            append(entry)
            log(f"asr {entry['asrSec']}s  {cleaned[:70]}")
    finally:
        stream.stop()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
