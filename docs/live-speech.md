# Live speech: mic → Whisper → Gemma

Speak into the camera's microphone; the transcript and its translation appear
on the website. Both models run on the Hexagon NPU, so no audio leaves the
board.

Start it from the **Live microphone** panel at the bottom of the page, or:

```bash
make web            # the site, which has the Start listening button
make mic-logs       # what the listener is doing
```

## Measured

| Stage | Time | Where |
|---|---|---|
| Whisper-tiny **model load** | **0.52 s** | per invocation — the runner is one-shot |
| Whisper encode | **0.027 s** | Hexagon NPU |
| Whisper decode | **~0.008 s/token** | Hexagon NPU |
| Gemma 4 translate | **4.3–4.5 s** | Hexagon NPU |

The inference is nearly free. What you wait for is the audio chunk, and —
annoyingly — the half-second spent reloading Whisper every time, because
`qnn_whisper_runner` is a one-shot CLI with no server mode.

**Transcribe only:** 1.8 s chunks, so a line appears **~1.8–2.6 s** after it is
spoken.

**With translation:** chunks stretch to 4 s, because Gemma's 4.5 s dwarfs
everything else and short chunks would just burn NPU passes. A translated line
lands **~9 s** after speech.

### What made it faster

It used to be 5 s chunks with `arecord` spawned per chunk — about 6 s behind,
and it silently **dropped every word spoken while the NPU was busy**, because
nothing was listening during inference.

Now one long-lived `arecord` feeds a ring buffer on a reader thread, so capture
never stops. Windows overlap by 0.6 s so a word split across a boundary
survives in one of them, and `drop_overlap()` trims the repeated words from the
next result. Chunk length adapts to whether translation is on.

Net: roughly **2.3× faster**, and no more lost words.

### If you need it faster still

The remaining fat is that 0.52 s reload per window. Removing it means holding
the model resident, which needs a persistent process — the bundle ships
`PyQnnManagerAdaptor` built for aarch64 cp312, so the decode loop could be
reimplemented in Python against a loaded module. That would put per-window cost
near 0.1 s and allow ~1 s chunks, so about **1.1 s** end to end. It is a few
hours of work and is not done.

## How translation works, and why it isn't Whisper's

Whisper has a built-in `translate` task, but the ExecuTorch runner can't reach
it: `qnn_whisper_runner` has no task flag, and `runner.cpp` seeds the decoder
with only `<|startoftranscript|>` (50258), letting the model generate its own
language and task tokens — which defaults to transcribe. Forcing translate
would mean patching the C++ and rebuilding, and the board has no compiler.

So translation is a **second NPU pass through Gemma 4** instead. That turns out
better than Whisper's own translate task would have been:

- Whisper only ever translates **into English**. Gemma goes into any language.
- It reuses a model already proven on this board.
- It needs no rebuild.

Verified output: *"El reconocimiento de voz en el dispositivo está funcionando
en el procesador neuronal Hexagon."* and *"デバイスでの音声認識は、Hexagonの
Neural Processorで動作しています。"*

## Design notes

**Silence gating matters.** Whisper hallucinates confidently from near-silence
— "Thank you.", "Bye.", "Thanks for watching!" are its favourites. The daemon
drops chunks below an RMS threshold (`SILENCE_RMS`, default 0.012) and filters
a list of known hallucination phrases. Without this the transcript fills with
imaginary politeness.

**`<|nocaptions|>` means no speech.** The runner prints Whisper's special
tokens verbatim, so that gets mapped to an empty result rather than appearing
as markup on the page.

**The NPU is used serially.** Whisper and Gemma each take the whole DSP
context; they can't overlap. The daemon runs them one after the other, and the
vision model shares the same constraint — analysing a frame while the listener
is mid-utterance will contend.

**`pkill -f <name>` kills the shell you typed it in**, because the pattern
matches that shell's own command line. It bit us for `gst-launch` and again for
`mic_daemon.py`. Use `pgrep -af '[m]ic_daemon'` to check and stop it through
the API (which spawns `pkill` without a shell), or bracket the first letter.

**Control without restart.** The daemon re-reads `/dev/shm/mic/control.json`
every loop, so toggling translation or changing the target language from the UI
takes effect on the next chunk.

## About the whisper-small you downloaded

`whisper_small-voice_ai-float-qualcomm_qcs8275.zip` is built for this chip but
**can't be used by this runner**:

- It contains `encoder.bin` / `decoder.bin` — **QNN context binaries**, not
  ExecuTorch `.pte` files. `qnn_whisper_runner` only loads `.pte`.
- Its `metadata.json` says `"runtime": "voice_ai"` — Qualcomm's own Voice AI
  app framework, which we don't have.
- It wants **QAIRT 2.50.0**; the board has 2.49.0.

Using it would mean writing a Whisper decode loop directly against
`qnn-net-run` — autoregressive decoding, cross-attention KV cache management
and its custom `vocab.bin` — or obtaining the Voice AI runtime. Several hours
either way.

The upside would be real: whisper-small is markedly more accurate than tiny,
and float16 rather than 16a8w-quantized. If transcription quality becomes the
bottleneck for the demo, that's the thing to chase. It is not a drop-in.
