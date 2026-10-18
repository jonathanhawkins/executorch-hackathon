# The virtual billboard

Stand in front of the camera and the billboard shows you wearing the shirt.
Head, hair and neck are segmented on the Hexagon NPU and dropped into the neck
of a product photo — the Minority Report idea, running entirely on the board.

```bash
make web        # the site; press "Start billboard"
make bb-logs    # what the compositor is doing
```

## Measured

| | |
|---|---|
| BiSeNet parse, pure compute | **12 ms** |
| Parse step as the daemon sees it | **~360–440 ms** |
| Whole frame: detect → parse → matte → composite | **~750–860 ms** |

That gap between 12 ms and 360 ms is **model load**, paid on every frame
because `qnn_executor_runner` is a one-shot CLI. The segmentation itself is
nearly free; the loop runs at a bit over 1 fps, which for a billboard that
reacts to someone standing in front of it reads as deliberate rather than slow.

## How it composites

The garment is split into pieces so the head can sit *between* them:

```
backdrop  ->  shirt body  ->  HEAD CUTOUT  ->  shoulders (yoke)  ->  collar
```

**Drawing garment pieces over the head is the whole trick.** The cutout ends in
a ragged neck. The shoulder band occludes the sides of that neck and the collar
hides its bottom edge, which is what makes the head read as being *in* the
shirt rather than pasted on top of it.

An earlier version had the shirt as one flat layer behind the head, with only
a collar in front. That left the neck sitting on the garment like a sticker —
the pieces matter more than the drawing quality does.

**Scale comes from eye separation, not the face box.** Pupil distance is a
stable biological constant, so someone leaning in and someone standing back
come out the same size on the billboard. Scaling by the detector's box makes
the head jump around between frames.

**Rotation too** — the eye line sets the tilt, so a head cocked to one side
still sits straight on the shirt.

### Proportions: anatomy is the starting point, not the answer

Three numbers in `make_billboard_assets.py` decide whether this reads as a
person: `EYE_SPAN`, `EYE_Y` and `NECK_Y`.

Strict anatomy says shoulder width is about **seven times** pupil distance,
which for the ~590 px shoulders here gives `EYE_SPAN ≈ 90`. That is correct and
looks wrong — the drawn tee is deliberately oversized and fills the frame, so
an anatomically-scaled head reads as tiny. A billboard is a portrait, not a
figure study. **`EYE_SPAN = 128`** is what actually looks right.

Raise it for a bigger head. `EYE_Y` is the knob for how much neck shows:
raising it drops the head and tucks the neck further under the collar. Current
values are `EYE_SPAN 128`, `EYE_Y 428`, `NECK_Y 648`, which puts the chin
essentially at the collar line.

Note the tilt interaction: scale and position are anchored to the **eye line**,
so someone tipping their head back shows more chin below the eyes and appears
to sit lower in the collar. That is inherent to eye-anchoring and is still the
right trade — the alternative, anchoring on the face box, makes the head jump
around between frames.

### The neck is drawn, not segmented

The segmented neck was the weakest part of the matte: it carries wall pixels
the parser mislabels as `neck`, it frays at the bottom, and whatever the clip
cuts shows as a hard or smudged edge against the collar.

So the head layer now **stops just under the jaw** and the neck is *drawn*: a
tapered column in the subject's own skin tone, widening slightly into the
shoulders, with a blurred contact shadow where the jaw would cast onto it.

The colour is sampled per frame — the median of the pixels the parser is
confident are facial `skin` — so it is this person's tone under this room's
light, not a generic value.

Why this beats cleaning up the segmentation: the neck is the one part of the
cutout nobody looks at closely but everybody notices when it is wrong. A drawn
column always meets the collar, always has a sane silhouette, and cannot leak
background. The parts that *do* get looked at — face, hair, jaw, ears — stay
fully segmented.

Shape knobs are in `Billboard.draw_neck()`: `half_top` and `half_bot` as
fractions of `EYE_SPAN`, and the shadow's ellipse and blur.

### An approach that did not work: colour-filtering the segmented neck

BiSeNet reliably labels a band of **wall** beside and below the jaw as `neck`.
Those pixels carry the background's colour, so they came through as a pale slab
with hard straight edges wherever the geometric clip cut them.

Geometry cannot separate it — the mislabelled region is contiguous with the
real neck and sits in the same place. **Colour can.** `refine_neck()` measures
the skin the parser is confident about (the `skin` class on the face), takes
its median, and drops `neck` pixels further than a tolerance from it. The
tolerance comes from that person's own skin spread — the 90th percentile of
in-face variation, clamped to 40–110 — so it travels across lighting and
complexions instead of being tuned to one room.

**Colour alone punches holes through facial hair.** Beard and jaw shadow are
also unlike skin and also get labelled `neck`, and the first version speckled
them out. The difference is shape: the wall arrives as one slab, beard as
scattered specks. So the drop is restricted to connected components above
~0.4% of the crop. That one constraint is what makes the filter usable.

**This was rolled back.** It deletes real neck too: a neck lit differently
from the face — shadowed under the jaw, bright where it catches a window — is
genuinely far from the face's median tone, so the filter punched through to the
backdrop and left white clipping. Widening the clip blur to compensate only
added a visible dithered fade across the bottom of the head layer.

It survives behind a `refineNeck` control flag, off by default. The drawn neck
above replaced it.

### Collar geometry: watch where the top arc lands

The collar ellipse spans `NECK_Y ± ~90`, so its **top** is at `NECK_Y - 92`.
Set `NECK_Y` too close to the eye line and that arc rises above the chin and
swallows the neck completely — the head appears to rest directly on the shirt.
The chin falls at roughly `EYE_Y + 1.2 * EYE_SPAN`; keep the collar's top arc
below it with ~45 px to spare.

### Clip the head at the neckline

BiSeNet's `neck` class runs to the bottom of the crop, so a scaled-up cutout
spills a slab of neck and chest **over** the shirt — the head layer is drawn
after the body. `Billboard.head_clip` cuts the head layer just below the collar
line, and the collar is drawn afterwards to hide the seam. Without it, enlarging
the head hides most of the garment you are trying to sell.

### A leaning neck is a rotation problem, not a placement one

"The neck goes off to the side" looks like a horizontal alignment error and
is not, which is why nudging left/right never quite fixes it. Rotation is
applied to the **whole cutout** — head and neck together — so straightening a
tilted head tilts the neck with it. The model's neck is vertical on an upright
body, so a leaning neck running down to a vertical collar reads as skew.

Two changes, because there are two causes:

* **Horizontal placement now comes from the fit**, not the face box centre.
  Taking scale and rotation from the landmark fit while leaving x on the box
  was an inconsistency: the box centre tracks the *face*, which slides off the
  skull's midline the moment the head turns, whereas a neck attaches under the
  midline. Mapping the anchor through the fit puts the face where the model's
  was, and his was over his own neck by construction.
* **The neck fade shortens with rotation** — `fade *= 1 - min(|angle|/12,
  0.45)`. The more the cutout is rotated, the sooner the shopper's leaning
  neck should give way to the model's vertical one. At the ±8° clamp that is
  about a third off the run.

### The pose wireframe

A simple 3D head drawn over the live preview in the pose the detector
currently measures: an ellipse, a meridian for the nose line, an equator for
the eye line, and two eye marks, the whole group rotated by the measured roll.

The point is the difference between stating a target and showing an error. The
dashed oval says where to be; it cannot say where you *are*. "Turn less" is
hard to act on — a meridian you can centre is not.

Three details carry it:

* **It is drawn on the head the detector found**, not parked at the target.
  A wireframe pinned to the target that only rotates shows your pose but gives
  you nothing to steer towards.
* **It is calibrated so a correctly-framed head lands exactly on the oval.**
  At the gate's nominal face width (0.34 of frame), `rx = rel * W * 0.471` and
  `ry = rx * 1.359` reproduce GUIDE's own `rx` and `ry`. Lining the solid head
  up with the dashed one is therefore the same test `check_alignment` applies,
  not a second opinion.
* **Yaw narrows the silhouette by its cosine** as well as bowing the meridian.
  Without the narrowing it reads as a flat oval being slid sideways rather
  than a head turning.

The viewBox is the camera's own 640x480, so GUIDE's fractions drop in
unchanged and the drawing cannot drift from what the daemon tests.

Worth checking the arithmetic rather than the look here: the first version
used an aspect correction that did not belong (the viewBox has square pixels),
giving `ry = 245` — a head 490px tall in a 480px frame. It would have been
obvious on screen, but only with a person in front of the camera, and there
wasn't one.

### Measuring the fit instead of eyeballing it

Two readouts turn "it looks slightly off" into numbers.

**Live: tilt and turn.** `check_alignment` was already measuring roll and yaw
to gate the shutter; they are now written into the state and shown beside the
framing indicator, amber past the thresholds. Being told "straighten your
head" without being shown how far off you are is guesswork — with the number
visible it converges in about a second.

**Shot: head vs model.** `head_match()` measures the placed head at **the eye
line, offset by however far the head has been moved vertically** — and reports
the ratio against what the model's head measures there. The eye line is a
stable height on a head, unlike a bounding box whose top depends entirely on
hairstyle; following the offset matters because the up/down knob spans 280px,
and at its extremes a fixed row would cut across the mouth or forehead and
report a sizing error that is purely placement. 1.00 means the shopper's head
occupies the width his did; it also reports `suggestScale`, the multiplier
that would make them match exactly.

It is **reported, not applied**, and that distinction matters: a ratio away
from 1.0 is not automatically an error. A genuinely narrower head *should*
measure narrower, and forcing every head to the model's proportions would
stretch people to fit the mannequin. The number tells you whether what is left
is a placement bug or a matter of taste.

#### Calibrate the reference, or the metric lies

The obvious reference is the stored head mask, and it is wrong. That mask was
grown so no hair fringe survived the inpaint, and it never went through the
feather, resample and alpha cut a shopper's matte does. Measured against it,
**feeding the model his own head back reads 0.87** — and anyone trusting that
number would scale up by 15% to correct a discrepancy that is entirely an
artefact of comparing two different things.

`scripts/calibrate_template.py` fixes it by recording what the model's own
head measures *after going through this exact path*, as `headSpanRef`. It
reads that figure from `head_match`'s own `headSpanPx` rather than re-deriving
it from pixel colour, and it **refuses to write** if the landmark fit fell
back — a reference built by the weaker estimator would skew every later
measurement, permanently and silently. Run it once per template, after
`prep_template.py`:

```bash
python scripts/calibrate_template.py assets/templates/model-navy-tee.png
```

That self-composite is also the sharpest end-to-end test in the project,
because it is the one case with a known right answer — the answer is the
original photograph. It caught the biased metric, and it is what confirmed the
landmark fit actually fires:

| | |
|---|---|
| `align` | `landmarks` |
| `scale` | **0.955** — exactly `side / 512`, undoing the crop's upscale |
| `angle` | **+0.0°** (the face-box fallback claims −4.0° on the same photo) |
| `headMatch` | **1.000** after calibration, 0.871 before |

The angle row is the landmark fit earning its keep: on a model who is
demonstrably not tilted, the eye-line method is 4° out and the five-point fit
is not.

### Two sessions, one board

Worth knowing when results stop making sense: `align` read `facebox` for a
long stretch and no amount of checking the code explained it, because another
agent working on the same board had switched `control.json` to its own
`faceswap` mode. The billboard branch simply never ran. The daemon, the
template and the compositor were all correct the whole time.

The tell is `state.mode`, not `control.mode` — check it first when the
readouts disagree with the code in front of you. The same contention shows up
as the daemon dying mid-session and as `board_billboard_daemon.py` changing
underneath a push.

### Make the guidance predict the shutter

The live gate said "Ready" for frames the capture then refused with "Step back
a little", so the shutter looked broken. They were testing different things:
the capture rejects when the square crop clamps at the top *and* the head mask
reaches row 0, while the live gate only checked `bbox[1] <= 2`. Hair runs
roughly a third of a face height above the detector's box, so the live gate
now tests `bbox[1] < 0.30 * bbox[3]` — close enough to predict the capture.

A guidance string that disagrees with the gate behind it is worse than no
guidance, because the shopper stops trusting all of it. Same principle as the
oval and the eye marks: if it is drawn, it has to be tested.

### Keep the head you removed

`prep_template.py` computes a mask of the model's head in order to inpaint it
away, and used to throw it away afterwards. It is now saved as
`<name>.head.png`, because it is the only ground truth in the whole pipeline
about where a head belongs on this body and how big it is — measured from the
original photograph rather than reconstructed from a handful of anchor
numbers.

Its immediate use is the halo. Everything inside that mask is inpainted
backdrop, so wherever the shopper's head stops short of where the model's jaw
used to be, the viewer sees grey under the chin and the head reads as floating
above the body. Nothing about the placement is wrong when that happens — the
two heads are simply different shapes, and no amount of tuning scale or offset
removes it, because moving the head to close the gap on one side opens it on
the other.

`bridge_gap()` fills it: inside the old head mask, within its **lower third**,
and not covered by the shopper's layer — feathered, and toned from the canvas
being painted. A flat fill is enough because the region sits under the jaw in
shadow, which is where the eye expects flat tone anyway.

Two details are load-bearing, and the first version got both wrong:

* **Band, not chin.** It originally excluded everything above the shopper's
  chin, reasoning that above the chin is sky. It is not — above the chin is
  the *head*, and the mask **is** the head, so it ends at the jaw. The guard
  excluded the entire mask and the function was inert at every default
  offset; it only did anything when the head was pushed above the model's jaw.
  Restricting to the lower third of the mask instead targets the jaw, which
  is where the notch appears. Backdrop beside someone's temples is left alone,
  because a genuinely smaller head *should* leave more wall visible.
* **Tone from the canvas, not the template.** `match_skin` has already shifted
  the neck and arms up to 26 L and 18 a/b toward the shopper by this point, so
  sampling the template's original mean put the model's skin directly against
  a neck recoloured to theirs — a visible step exactly where one was being
  hidden.

The mask is worth more than this one use — coverage against it is a direct
measure of how well a head fits the body, which is the thing "it doesn't line
up" actually means.

### The taper has to follow the silhouette

The neck column started at the chin and immediately narrowed to 0.56 of the
jaw. That cut the jaw rather than the background beside the neck, and left a
straight vertical edge running down where BiSeNet's blocky neck boundary was,
because the taper began below it.

The silhouette is an hourglass — wide at the jaw, in at the neck, out again
into the shoulders — so the column follows that, starting **above** the chin
(`rise`) at 1.6x the half-width, narrowing to 1.0 through the neck, then
flaring. Starting it above the chin is the part that matters: the blocky edge
is beside the *jaw*, so a taper that begins at the chin can never reach it.

### Placement: fit five landmarks, don't measure the box

Scale, rotation and position all used to come from the detector's bounding
box and the eye line. All three are better estimated at once, by fitting the
shopper's five YuNet landmarks onto the model's own five with a least-squares
similarity transform (Umeyama). The model's landmarks are measured during
`prep_template.py` and stored in the template JSON.

Why the box was the weak link:

* **The box width is a function of head yaw.** Turn away from the camera and
  it narrows while the head does not, so matching box widths silently enlarges
  a turned head. Landmark spread is a measurement *of the face*, not of a
  rectangle drawn around it.
* **The eye line overstates tilt on a turned head**, because it foreshortens.
  Fitting nose and mouth alongside the eyes constrains the rotation.
* **One anchor point concentrates all the error there.** A least-squares fit
  over five points spreads it instead.

The chin stays the anchor for the vertical offset, so the `up / down` knob
keeps meaning "how much neck shows". The fit supplies scale and tilt — the two
the box measured badly.

**The exact fit looks slightly small**, and that is not a bug. Mapping the
landmarks onto the model's makes the shopper's face exactly the size his was,
which is correct and reads as a touch small for a billboard, the same way the
drawn version needed `EYE_SPAN` above anatomical. `headScale` defaults to
**1.12** on top of the fit. On the old face-box basis the equivalent was ~1.30,
so the number is not comparable across the two — reset it when switching.

### Frame tightly, or the scale is guesswork

The guide oval was about 1.6x the face box, so a face could sit almost
anywhere inside it and still pass as aligned — and the resulting scale varied
by more than the knob does. Tightened to `rx 0.21, ry 0.30` with the size gate
at 0.28–0.40 of frame width.

`GUIDE` in the daemon and `.guide-oval` in the CSS are the same four numbers
(`left = cx - rx`, `top = cy - ry`, `width = 2rx`, `height = 2ry`). They have
to be edited together — what the shopper aims at is what `check_alignment`
tests, and if they drift apart the oval lies.

**Percentages against the stage are not percentages of the picture.** The
stage is 4:5, the composite's shape, but the live preview is a 4:3 camera
frame letterboxed inside it by `object-fit: contain`. An overlay positioned
against the stage is therefore measured against a box 1.67x taller than the
image it appears to sit on: the oval came out 1.67x too tall and centred too
high, which reads as "the oval is too big" and is invisible as a bug because
the oval still looks like an oval.

Matching the numbers on both sides does not fix it — they were already
matched. The guide has to live in a box that reproduces the *fitted content*:
`.guide-box` is full stage width and `75cqw` tall (`cqw` works because
`.billboard-stage` sets `container-type: inline-size`), centred. Inside it,
CSS percentages are frame fractions again.

The general form: any overlay on a `contain`-fitted image whose aspect differs
from its container is positioned against the wrong box. It only looks right
when the two aspects happen to agree.

### The eye marks are derived, not drawn by eye

Two dots inside the oval, at the position a face that passes
`check_alignment` actually puts its eyes. The numbers come from the model's
own landmarks in the template — eyes **0.337** down the detector's box,
separated by **0.457** of its width — mapped through the gate's nominal face
size. Aiming at them therefore satisfies the same test the oval does.

Drawing them where they merely look right would reintroduce exactly the
problem above: a guide the daemon never checks, quietly disagreeing with the
one it does.

### The centre line checks pose, it does not just suggest it

A vertical dashed line down the oval to line the nose up on. Like the eye
marks it is backed by the gate rather than being decoration, because a guide
the daemon never tests is the failure documented above:

* **Roll** from the eye line, rejected past **10°**. A tilted head is the
  single most "off"-looking thing in a composite, and the compositor clamps
  its own correction to ±8°, so beyond that the tilt survives into the poster.
* **Yaw** from where the nose sits between the eyes — the nose offset from the
  eye midpoint, *projected onto the eye line*, in units of eye separation.
  Rejected past **0.25**.

Two details matter. Projecting onto the eye line keeps a tilted head from also
reading as a turned one; without it, roll leaks into the yaw measure and the
shopper is told to turn when they should straighten. And dividing by eye
separation makes it scale-free, so it behaves the same at any distance —
which a pixel threshold would not.

Pose is checked **last**, after position and size. It is the hardest thing to
act on, so there is no point asking for it while the shopper is still in the
wrong place.

Yaw is worth gating on its own merits: a turned head is what the landmark fit
handles worst, since the landmarks foreshorten and no similarity transform
recovers the far side of a face the camera cannot see.

The oval itself is **32% x 58%** of the frame, a 1.37:1 ellipse. At the
previous 38% x 54% it rendered 160x170 px — near enough square that it read
as a circle, and a circle invites people to centre their whole head in it
rather than their face.

### Cross-fading the two necks

The first photo-template version cut the shopper at the jawline and used the
model's neck alone. That removes the frayed segmented neck, but it puts a hard
boundary exactly where tone changes fastest — under the jaw — so the seam shows
even when both sides are individually clean.

Keeping the shopper's neck and *dissolving* it into the model's works better,
and needs three things together. Any one alone is worse than the hard cut:

* **A vertical fade** from the jaw to the collar, on a smoothstep. A linear
  ramp leaves a visible edge where it reaches zero: the eye finds the
  discontinuity in the gradient, not in the value.
* **A horizontal taper** — a column under the chin, flaring slightly toward
  the shoulders. BiSeNet reliably labels a band of *wall* beside the jaw as
  `neck`, and that band is most of why the segmented neck looks frayed. The
  column removes it geometrically, so unlike the colour filter in
  `refine_neck` it cannot punch holes through a beard.
* **A tone match, ramped on the same profile.** Inside the fade both necks are
  visible at once, so if their tones disagree the cross-fade reads as a ghost
  rather than a blend. `match_neck_tone` pulls the shopper's neck toward the
  model's, weight zero at the jaw (their real jaw tone is kept — that is the
  part anyone looks at) rising to full at the collar.

Note the direction. `match_skin` moves the *template* toward the shopper, so
his arms read as theirs across the whole frame. `match_neck_tone` moves the
*shopper* toward the template, over the short run where the two overlap,
because at the collar it has to become his neck exactly. Running both in the
same direction leaves a mismatch somewhere by construction.

Cost: **+8 ms** per compose on the Mac (42 → 50 ms); the LAB work is confined
to the neck band rather than the frame. Toggle `neckBlend`, run length
`neckFade`.

### The EMA is for video, and a capture is not video

The single biggest placement bug in the photo-booth flow: `Smoother` is an
exponential moving average with `alpha=0.3`, there to stop the head twitching
between live frames. The live phase deliberately **skips the parse** — it only
needs the face box for alignment — so `compose()` is never called while the
shopper is framing up, and the EMA still holds whatever state it had at
startup or from the previous shot.

A capture then calls `compose()` exactly once, and alpha 0.3 means the head
lands **30% of the way** from that stale state toward the measured position.
The result reads as "the head is too small and the neck too long", because a
stale small scale and a stale high anchor are both pulled only a third of the
way to the truth. It is not a scaling problem and no amount of tuning
`faceWidth` fixes it.

`smoother.reset()` immediately before the capture compose is the fix: a still
frame should be placed from the measurement, not from an average over frames
that were never composited.

### Clamp the tilt

Rotation comes from the eye line. On a head that is also *turned* away from the
camera, correcting the full eye-line angle reads as a cocked head rather than a
corrected pose — the eye line foreshortens as the head turns, so the measured
angle overstates the real tilt. Clamped to ±8°.

### Don't shoot mid-motion

The parse and the matte both come from one frame, so a frame caught while
somebody is turning produces a smeared head that looks like a *blending* bug
rather than motion blur. The shutter waits for the face box to settle (centre
and width within 9 px across consecutive frames), for at most three loops —
after that it takes what it has rather than appearing to ignore the shutter.

### Tuning the shot without re-running the NPU

Placement is pure CPU, so the shot phase caches its compose inputs (crop,
parse, face box, eye points) and re-composites when a placement slider moves.
Tuning a captured shot costs ~160 ms and no NPU round trip, which is what makes
the sliders usable at all — re-parsing per slider move would be ~400 ms each.

Knobs are `headScale`, `fineX`, `fineY` in `control.json`, surfaced as sliders
in billboard mode once a shot is taken.

## Why BiSeNet rather than a segmentation model

Semantic segmentation (DeepLabV3, FFNet) labels the whole body as one `person`
class, which gives no seam at the neck — you'd cut out the head *and* the torso
you are trying to replace. BiSeNet trained on CelebAMask-HQ has 19 classes, so
the cutout is simply "everything except `background` and `cloth`", and the
garment is excluded at the neckline for free.

## The trap: BiSeNet wants a face-filling square

It was trained on square, face-filling portraits. A 16:9 webcam frame squashed
into 512×512 comes back as a blocky rectangle with hallucinated `hat` and
`eye_g` classes and no neck — the model invents classes rather than merely
missing. So YuNet finds the face first and we crop a square at 1.6× the face
box, which keeps hair and neck inside the frame.

Seen directly: pointing the camera at the ceiling gave
`['background', 'skin', 'eye_g', 'neck', 'cloth', 'hat']` on a centre crop with
nobody in it.

## Other things that bit

- **The camera frames vanish mid-read.** The capture daemon keeps only four
  (`max-files=4`) and deletes them within a few hundred ms, so reading the
  shared path races the writer. The daemon snapshots the newest frame to its
  own work directory first.
- **`square_crop` returns a tuple**, `(image, (left, top, side))`, not a dict.
- **PIL's `rotate(expand=True)` shifts the origin** by an amount that depends
  on the rotation centre, which makes the eye midpoint hard to track. Pad
  first, then rotate in place.
- **Clipping can create floating blobs.** Cutting the head layer at the
  neckline can sever a hair strand, leaving the upper piece hovering beside the
  collar. A connected-component pass *after* the clip, seeded at the eye line,
  removes it — doing it before the clip does not, because the strand is still
  attached at that point.

## Quality notes

Segmentation quality is genuinely good — hair, beard, ears and the neck come
out cleanly at 200 px+ face width. It degrades on small or heavily angled
faces, so **face the camera and get close**.

One known artifact: a sliver of skin can show just below the chin where the
neck cutout ends above the collar's inner edge. Widening the collar helps;
it is not fully solved.

Carried over from the earlier investigation and **still unverified in a real
scene**: on a flat, uniform backdrop BiSeNet labels the background beside the
head as `hair`, giving a grey slab on the cutout. It is connected to the real
hair, so the connected-component pass cannot remove it. A textured room has not
shown this, but a plain store wall might.

## What is *not* fixable in software

A flat horizontal cut across the top of the head means the camera cannot see
the top of the head — the crop is clamped to the frame, so there is nothing
there to segment. Same for a chin cut off at the bottom. No parameter helps:
tilt the camera or step back until the whole head is inside the frame.

Easy to confirm: the **Segmentation** mode draws the crop box, and if it is
flush against a frame edge the subject is outside it.

## The compare slider

Fades the composite down to reveal a reference underneath, which is the
fastest way to judge placement: "does this look right" becomes "does this land
where a head actually goes".

Two references, and the default is the useful one:

* **`original`** — the product photo with the model's head still on it. Fade to
  50% and the two heads ghost together, so size, angle and height are read
  directly rather than guessed at. This is the reference that answers whether
  the placement is right.
* **`base`** — the headless template the composite is built on, for checking
  the seam and the neck blend without the face distracting from it.

**Both are padded to the composite's canvas before being served.** The source
photo is 1122x1402 and the composite is 1122x1552, because `prep_template`
adds `headroom` above the model's hair. Serving the original unpadded would
misregister the two by exactly that padding and make the comparison worse than
useless — it would look like a placement error that is not there. The padding
is read from the template's own JSON rather than hardcoded, so regenerating
with a different headroom cannot silently reintroduce the offset.

The reference is a second `<img>` underneath with the composite's opacity
animated over it, rather than swapping one `src`. A swap flashes while the
other image decodes, and at this size the flash lasts longer than the glance
the comparison is for.

## The ad

The branding is an HTML overlay, not pixels baked into the composite. It stays
crisp at any size, a price change is a text edit, and it costs nothing per
frame.

**System fonts only, deliberately.** The claim this demo makes is that you can
unplug the network and it still works. A Google Fonts `<link>` would quietly
make that false — the page would still render, just not as designed, which is
the worst kind of broken because nobody notices until it is on a screen in
front of people. The stack is `"Helvetica Neue", Inter, "Segoe UI", system-ui`.

Three things do the work:

* **Tracking on the wordmark.** `0.44em` on a regular weight. Wide letter
  spacing is the house style of every modern basics label; it reads as
  confidence because it spends horizontal space on purpose. Note the
  `text-indent` / negative `margin-left` pair — tracking adds space *after*
  the last letter too, so without that the word sits visibly left of the
  margin it is supposed to align to.
* **A dark scrim under the bottom lockup.** It crosses both the pale backdrop
  and the navy shirt, so it needs its own ground. A light band would do it,
  but it bleaches the garment — and the garment is the product. White type on
  a dark scrim darkens the pale corners instead, which is why real lookbooks
  all look like this.
* **The stage matches the composite's aspect** (`1122 / 1552`). It used to be
  `1080 / 1350`, so the photograph never filled it and the overlay's top and
  bottom landed on letterbox bars rather than on the picture.

One trap worth naming: the scrim is `z-index: 0` beneath children at
`z-index: 1`, **not** `z-index: -1`. The overlay creates no stacking context,
so a negative index sends the scrim behind the photograph itself and it never
appears at all — the text just sits unreadable on the shirt, looking like the
gradient was too weak rather than absent.

## The store is fictional

`NORTHWIND` is made up. Rendering a real retailer's branding onto a stranger's
face is impersonation whatever the demo, so the backdrop, garment and collar
are all drawn programmatically in `scripts/make_billboard_assets.py`. Change
the brand, colour and price there; the anchors are written to `billboard.json`
and the daemon reads them.
