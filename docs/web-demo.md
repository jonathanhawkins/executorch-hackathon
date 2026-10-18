# The camera demo

A USB camera on the board, InternVL3-1B on the Hexagon NPU, and a Next.js page
served from the same device. Nothing leaves the board — pull the network cable
and it still works, which is the whole argument for on-device AI in one gesture.

```bash
make web          # start it (camera + server)
make web-logs     # tail the server log
make web-stop     # stop both
make web-push     # re-push sources after editing web/
```

## Reaching it

`make web` prints the board's own address, e.g. `http://10.104.8.211:3000`.
That works from anything on the same network as the board.

The Mac could **not** reach it directly here — hackathon wifi isolates clients
— so there's a fallback over USB:

```bash
adb forward tcp:3000 tcp:3000    # then open http://127.0.0.1:3000
```

Worth testing which one works on the venue network *before* you present.

## What it does

| Route | Job |
|---|---|
| `/` | Live preview, prompt box, result panel with timings |
| `/api/snapshot` | Newest camera frame as JPEG; the UI polls it |
| `/api/analyze` | Capture → preprocess → NPU inference → JSON |
| `/api/status` | Camera / model / runner health for the indicator pills |

Measured on the board: **~12 s end to end**, of which ~8 s is prompt eval
(270 tokens, 256 of them the image), 34 tok/s prompt eval, 33 tok/s generation.

## Design notes worth keeping

**One process owns the camera.** A single long-lived `gst-launch-1.0` writes a
bounded rolling window of JPEGs to `/dev/shm/cam/` (`max-files=4`). Everything
else reads those files. Two consumers opening `/dev/video0` fails with "device
busy", so the snapshot route and the analyze route both go through the files
rather than grabbing their own frames.

**Polling, not MJPEG.** The camera runs at 5fps, so polling `/api/snapshot`
every 220ms looks identical to a stream and avoids holding a long-lived
connection open through the server.

**One inference at a time.** The analyze route holds a `busy` flag and returns
429 if a second request arrives. Two concurrent runs fight over DSP memory and
both fail.

## Things that cost time, so you don't repeat them

- **`pkill -f gst-launch` kills the shell running it**, because the pattern
  matches that shell's own command line. Use `pkill -x gst-launch-1.0`.
- **Backgrounding inside `adb shell "... &"` doesn't survive.** The process
  dies when the shell exits. Write a script file on the board and `setsid` that
  instead — `board_web.sh` does this for both the camera and the server.
- **Dev mode doesn't hydrate over an adb port forward.** The HMR websocket
  can't connect and React never takes over, so the page renders but nothing is
  interactive. Use a production build (`npm run build && npm run start`), which
  is what you want for a demo anyway.
- **`output: "standalone"` conflicts with `next start`.** It's for container
  images; this isn't one.

## Honest limitation

The model recognises scenes reliably — people, rooms, lighting, landmarks — but
it's a heavily quantized 1B and its *wording* is erratic. The same prompt can
give a clean paragraph one run and a terse non-answer the next, and long
open-ended prompts ("describe this in detail") sometimes collapse into
repetition or filler.

Short closed questions are much steadier, which is why the preset buttons are
all short questions. Temperature 0 loops forever; ~1.0 is the sweet spot, and
there's a slider in the UI for live tuning.

If the demo needs guaranteed-good narration, pair the vision output with a
fixed script rather than reading the model's prose verbatim.
