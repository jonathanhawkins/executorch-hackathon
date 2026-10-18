# Running on Android (Mobile + XR track)

For the Galaxy S25 Ultra and Galaxy XR. Shortest path of the three tracks —
same `.pte` you tested on the Mac, no cross-compiling.

There's a working demo app in [`android/`](../android). It builds today; it has
**not** been run on hardware yet, because no device has been attached.

## The whole loop

```bash
make export MODEL=llama3.2-1b-open   # once
make deploy MODEL=llama3.2-1b-open   # push model + build/install/launch app
make logcat                          # watch for load errors
```

`make deploy` is `make push` + `make install`. Use the pieces separately while
iterating on the app — pushing an 800 MB model every time is slow.

## First-time setup

Already done on this machine, listed so a teammate can catch up:

```bash
brew install --cask android-platform-tools      # adb
brew install --cask android-commandlinetools    # sdkmanager, platform SDKs
```

Then on the phone: Developer Options → USB debugging, plug in, accept the RSA
prompt. Confirm with `adb devices`, or `make doctor`.

The toolchain env (`ANDROID_HOME`, `JAVA_HOME`) lives in
[`scripts/android_env.sh`](../scripts/android_env.sh); the make targets source
it, so you don't need it in your shell profile.

## What the app does

A deliberately small harness, not a product — load a `.pte`, stream tokens,
show tok/s. Four files in `android/app/src/main/java/ai/executorch/hackathon/`:

| File | Job |
|---|---|
| `LlmEngine.kt` | Wraps `LlmModule`, bridges its callback to a Kotlin Flow |
| `PromptFormat.kt` | Chat templates — Llama 3, ChatML/Qwen, Gemma, raw |
| `ChatViewModel.kt` | State, threading, KV-cache reset |
| `MainActivity.kt` | Compose UI |

Model and tokenizer paths are editable in the UI and default to
`/data/local/tmp/llama/`, where `make push` puts them.

## Two things that will bite you

**The chat template.** `LlmModule.generate()` takes a raw string — there's no
`apply_chat_template` on device. Without the right framing an instruct model
free-associates instead of answering, which reads like a broken export but
isn't. `PromptFormat` handles Llama 3, ChatML and Gemma, and guesses from the
filename. We hit exactly this in Python before adding it.

**Threading.** `generate()` blocks its calling thread and pushes tokens through
a callback. On the main thread that freezes the UI for the whole generation.
`LlmEngine` puts it on `Dispatchers.IO`.

Also worth knowing: `resetContext()` clears the KV cache between prompts. Skip
it and the second question inherits the first one's context.

## Versions, and why they're pinned

| Piece | Version | Why |
|---|---|---|
| `org.pytorch:executorch-android` | **1.5.1** | Must match the `executorch` pip package that exported the `.pte`. Mismatches fail at load with an opaque error. |
| AGP | 9.4.1 | Has built-in Kotlin — the separate `kotlin-android` plugin is rejected. |
| Compose BOM | 2026.06.01 | Compose 1.12+ needs `compileSdk 37`, which is canary-only. This maps to 1.11.4 and builds on stable API 36. |

The AAR ships `arm64-v8a` and `x86_64` only — fine for devices and emulators,
but it means no 32-bit ABI.

## Sizing

The `.pte` is mmap'd, so resident memory is well under file size. Still:

| Model | `.pte` | Verdict |
|---|---|---|
| `llama3.2-1b-open` | 789 MB | Comfortable on an S25 Ultra |
| `qwen2.5-0.5b` | 396 MB | Safe everywhere |
| `smollm2-135m` | 100 MB | Use to prove the pipeline, not to demo |

## Getting a number you can trust

The app surfaces tok/s from the runtime's own `onStats` JSON, which is the
honest on-device figure. The Python number from `make run` is for comparing two
exports, not for predicting the phone.

## Sources

- [Running an LLM on Android](https://docs.pytorch.org/executorch/stable/llm/run-on-android.html)
- [Using ExecuTorch on Android](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)
