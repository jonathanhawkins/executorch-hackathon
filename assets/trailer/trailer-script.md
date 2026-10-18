# 30-second device trailer — "Nobody waits for a fitting room"

Audience: a clothing retailer deciding whether to put one of these on the floor.
Format: 6 shots x 5s. Voiceover ~78 words. No on-screen UI mockups — the device
and the shopper's reaction carry it.

---

## Shot 1 — 0:00–0:05 · The problem
**Visual:** A queue of four people outside a fitting room, arms full of clothes.
One shopper checks the line, looks at the shirt in her hands, puts it back on
the rack and walks out of frame.
**VO:** "Every fitting room line is a sale walking out the door."
**Audio:** Store ambience, a single low tone under it.

## Shot 2 — 0:05–0:10 · The device
**Visual:** Macro push-in on a small matte box on a shelf beside the mirror.
One status LED. A camera lens. No screen, no cables beyond power.
**VO:** "So we built something that doesn't need one."
**Audio:** Tone resolves into a pulse.

## Shot 3 — 0:10–0:15 · The moment
**Visual:** A shopper steps in front of a large vertical display. On the screen,
she is already wearing the jacket she is holding — her face, her posture, moving
with her, in real time.
**VO:** "It sees the shopper, and puts them in the clothes. Instantly."
**Audio:** Pulse opens up.

## Shot 4 — 0:15–0:20 · The range
**Visual:** Fast cuts, same shopper on the display: jacket, dress, coat, knit —
four garments in four seconds, her reaction staying genuine throughout.
**VO:** "Every size. Every colourway. Nothing to unhang, nothing to refold."
**Audio:** Rhythmic, quickening.

## Shot 5 — 0:20–0:25 · The proof
**Visual:** Wide of the floor. Two more people now waiting to use the display,
phones out, filming themselves. A staff member watches, smiling, hands free.
**VO:** "It runs entirely on the device. No cloud, no customer images leaving
your store."
**Audio:** Warm, open.

## Shot 6 — 0:25–0:30 · The close
**Visual:** Back to the box on the counter, shallow depth of field, the store
soft behind it. Title card resolves over black.
**Title card:** *Try everything. Try on nothing.*
**VO:** "Try everything. Try on nothing."
**Audio:** Single resolving note.

---

## Seedance prompts (one per shot, 5s each)

1. Slow dolly past a fitting room queue in a bright modern clothing store, four
   people waiting with armfuls of clothes, a woman in the foreground glances at
   the line, looks down at the shirt in her hands, returns it to the rack and
   walks away. Handheld documentary feel, natural retail lighting, shallow
   depth of field, 35mm, cinematic colour grade.
2. Extreme close-up macro push-in on a small matte white electronic device on a
   retail shelf beside a mirror, single soft green status LED, one small camera
   lens, minimal industrial design, blurred store background with warm bokeh,
   product-film lighting, slow steady push.
3. Medium wide shot in a bright minimalist clothing store. On the left, a woman
   with empty hands at her sides stands wearing a plain black turtleneck and
   black trousers. On the right, directly beside her, a tall vertical display
   screen shows the same woman full length wearing a bright scarlet red evening
   dress instead of her black clothes. She raises one arm and the red-dressed
   figure on the screen raises the same arm at the same moment. The real woman
   is in black the whole time and never touches the dress. Both are sharp and
   fully in frame. Slow steady camera, clean bright retail lighting, realistic
   cinematic commercial, no text, no captions, no watermark.
4. Rapid series of a woman standing before a large vertical store display as the
   outfit shown on the screen changes between a leather jacket, a summer dress,
   a long wool coat and a cream knit. Quick cuts, her delighted reaction stays
   continuous, energetic pacing, modern boutique, crisp lighting.
5. Wide shot of a busy boutique floor, two young shoppers waiting their turn at
   a vertical display screen filming themselves on phones, a smiling staff
   member in the background folding nothing, relaxed and unhurried. Warm golden
   retail light, slow drifting camera, lifestyle commercial look.
6. A small matte white electronic device with one soft green LED sits sharply in
   focus in the centre of a dark retail counter, crisp and clearly lit from the
   first frame. The blurred warm store lights sit far behind it. Very slow
   push-in, the device stays perfectly sharp the entire time, premium product
   commercial lighting, the frame darkens gently to black at the very end, no
   text, no captions, no watermark.


---

## How the cut was made

Model `dreamina-seedance-2-5-xm` via TokenRouter (`POST /v1/video/generations`,
poll `GET /v1/video/generations/{task_id}`). Each clip comes back 720p/24fps at
6-10s with a generated audio bed; the cut takes the first 5s of shots 1-5 and
2-7s of shot 6, concatenated with a 1.5s fade at the tail.

Shot prompts 3 and 6 above are the second-pass versions. The first pass put the
coat in her hands *on the screen too* (the try-on beat didn't read) and opened
shot 6 out of focus. Shot 3 only worked once the real and virtual garments were
different colours — black turtleneck against a red dress — so the swap is legible
in a 5-second beat.

Files: `device-trailer-30s.mp4` is the cut, `shot1.mp4`-`shot6.mp4` the sources.
