# Project ideas

Scratchpad. Nothing decided yet — the repo runs a model today, so the idea can
change without the plumbing changing.

## What the judges are rewarding

"Demonstrate the potential of on-device AI." So the question every idea has to
answer is: **why does this have to run on the device?** If the demo would be
better with a server, it's the wrong demo. Good reasons:

- **Offline** — works with the wifi off. Always a strong live demo.
- **Latency** — sub-100ms loop that a round trip would ruin.
- **Privacy** — the data cannot leave the device (health, camera, audio, kids).
- **Cost** — thousands of inferences where per-token billing is absurd.

Pulling the network cable mid-demo and watching it keep working is the single
most persuasive thing you can do on stage.

## Shape of a winning build

- One model, working, with a tight loop. Not three models half-wired.
- Something physical or visible on stage — sensor, camera, mic, robot.
- A fallback path. Demo gods are hostile; have a recorded clip.

## Rough directions

### Mobile + XR
- **Offline field assistant** — camera + small VLM, answers about what it sees
  with the radio off. Plays well for travel, repair, accessibility.
- **Live meeting intelligence** — on-device transcription + summarization where
  audio never leaves the phone. The privacy argument writes itself.
- **XR spatial narrator** — Galaxy XR, describes the scene for low-vision users.
  Latency and privacy both genuinely require on-device.

### IoT (Arduino UNO Q)
- **Natural-language machine control** — speak to it, the MCU actuates. Shows
  off the dual-brain design: model on Linux, real-time control on the STM32.
- **Anomaly narrator** — sensors stream to the MCU, a small model on the A53
  explains in plain language what changed and what to do.
- Keep the model ≤0.5B. The QRB2210 is a quad-A53 with no NPU.

### Compute (Snapdragon X Elite)
- The one track where a 1B+ model is comfortable, and the only one with a
  realistic path to WSL for a QNN/NPU build. If the team wants to show raw
  speed, this is where.

## Open questions

- Which track? Affects everything — decide first.
- Text-only, or multimodal? Multimodal is more impressive and much riskier.
- Who owns the demo script? Worth naming a person on day one.
