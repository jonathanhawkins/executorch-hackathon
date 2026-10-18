#!/usr/bin/env python3
"""Composite a shopper's head onto a photographic product template.

Replaces the drawn-garment billboard. The template is a real photo of a model
wearing the garment with his head removed and inpainted (see prep_template.py),
so the shopper's head lands on a real neck with real shading rather than a
drawn column.

Three things do most of the work:

* **Cut at the jawline, not the neck.** The shopper's neck is the worst part of
  their matte — it carries wall pixels and frays. The template already has a
  neck, properly lit. Using theirs would mean two necks fighting; using his
  means the seam sits under the jaw where a shadow hides it.
* **Scale by jaw width.** Matching the shopper's jaw to the model's jaw makes
  the head the right size for the body, independent of how close they stand.
* **Move the colour, not the face.** The template's neck and arms are shifted
  in LAB toward the shopper's skin. Recolouring their face instead would be
  both uglier and a lie about what they look like.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter

# Resolved by name from the parser's own label map by the caller.
NECK_NAMES = ("neck", "neck_l")


class Smoother:
    """Exponential moving average over placement, so the head stops jittering.

    The parse is recomputed from scratch each frame and the face box wobbles a
    pixel or two even when nobody moves; without this the head twitches.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.state: dict = {}

    def __call__(self, key: str, value: float) -> float:
        prev = self.state.get(key)
        if prev is None:
            self.state[key] = value
        else:
            self.state[key] = prev + self.alpha * (value - prev)
        return self.state[key]

    def reset(self) -> None:
        self.state.clear()


def _lab_mean(rgb: np.ndarray, mask: np.ndarray):
    import cv2

    if mask.sum() < 200:
        return None
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[mask].mean(axis=0)


class PhotoBillboard:
    def __init__(self, template: Path, skin: Path, meta: Path):
        self.meta = json.loads(Path(meta).read_text())
        self.size = tuple(self.meta["size"])
        self.jawline = tuple(self.meta["jawline"])
        self.jaw_width = float(self.meta["jawWidth"])
        self.face_width = float(self.meta.get("faceWidth", self.meta["jawWidth"]))
        self.chin = tuple(self.meta.get("chin", self.meta["jawline"]))
        self.landmarks = self.meta.get("landmarks") or None
        # The head prep removed, if it was kept. Everything inside it is
        # inpainted backdrop, so it doubles as a map of exactly where a gap
        # between the shopper's chin and the model's neck would show.
        head_path = Path(str(template).replace(".template.png", ".head.png"))
        self.head_mask_model = (
            np.asarray(Image.open(head_path).convert("L")) > 127
            if head_path.exists() else None
        )
        self.template = Image.open(template).convert("RGB")
        self.skin_mask = np.asarray(Image.open(skin).convert("L")) > 127
        self._template_skin_lab = _lab_mean(np.asarray(self.template), self.skin_mask)
        # Converting a 1122x1552 image to LAB twice a frame cost more than the
        # NPU parse did. Convert once, and remember the last tint so a shopper
        # standing still does not pay for it again.
        import cv2

        self._template_rgb = np.asarray(self.template)
        # Only the skin pixels ever change, and they are ~8% of the frame.
        # Keeping just those in LAB turns a full-image colour conversion per
        # frame into one over a thin strip.
        self._skin_lab = cv2.cvtColor(
            self._template_rgb[self.skin_mask].reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
        ).astype(np.float32).reshape(-1, 3)
        self._tint_cache_key = None
        self._tint_cache = self.template
        self._skin_rgb_mean = (
            self._template_rgb[self.skin_mask].mean(axis=0)
            if self.skin_mask.any() else None
        )

    # -- measurements on the shopper ------------------------------------

    @staticmethod
    def head_mask(parse: np.ndarray, head_idx, neck_idx,
                  with_neck: bool = False) -> np.ndarray:
        """The piece we transplant.

        Without the neck by default: the template already has one, properly
        lit, and two necks fight. With ``with_neck`` the shopper's own neck
        comes along so it can be cross-faded into the template's instead of
        butting the jaw against it — see ``neck_feather``.
        """
        drop = () if with_neck else tuple(neck_idx)
        keep = np.isin(parse, [i for i in head_idx if i not in drop])
        return keep.astype(np.uint8) * 255

    def neck_feather(self, chin_xy, half_top: float, fade: float,
                     flare: float = 1.45, rise: float = 0.34) -> Image.Image:
        """Alpha multiplier that dissolves the shopper's neck into the model's.

        Two separate jobs, both needed:

        * **Vertical fade.** Opaque at the jaw, gone by the collar. A hard cut
          anywhere along the neck shows as a line; a cut *at the jaw* shows
          worst of all, because that is where the tone changes fastest.
        * **Horizontal taper.** BiSeNet reliably labels a band of wall beside
          the jaw as ``neck``. Those pixels are the reason the segmented neck
          looks frayed. They sit outside the neck column, so restricting the
          neck to a column centred under the chin removes them geometrically —
          no colour test, so it cannot punch holes through a beard the way
          filtering by skin tone did.
        """
        import cv2

        w, h = self.size
        cx, cy = float(chin_xy[0]), float(chin_xy[1])
        ramp = np.ones((h, w), dtype=np.float32)

        y0 = int(max(0, math.floor(cy - fade * rise)))
        y1 = int(min(h, math.ceil(cy + fade)))
        if y1 <= y0:
            return Image.fromarray((ramp * 255).astype(np.uint8), mode="L")

        ys = np.arange(y0, y1, dtype=np.float32)
        t = np.clip((ys - cy) / max(fade, 1.0), 0.0, 1.0)
        # Fade on a smoothstep rather than a line: a linear ramp leaves a
        # visible edge where it reaches zero, because the eye finds the
        # discontinuity in the gradient.
        vert = 1.0 - (t * t * (3.0 - 2.0 * t))

        # The column follows the silhouette, which is an hourglass: wide at the
        # jaw, in at the neck, out again into the shoulders. Starting it at the
        # chin and narrow — as it was — cut the jaw instead of the background
        # beside the neck, and left a straight vertical edge where BiSeNet's
        # blocky neck boundary ran, because the taper began below it.
        tc = np.clip((ys - (cy - fade * rise)) / max(fade * (1.0 + rise), 1.0), 0.0, 1.0)
        half = half_top * np.interp(tc, [0.0, 0.34 / (1.0 + rise), 1.0],
                                    [1.60, 1.0, flare])

        xs = np.arange(w, dtype=np.float32)[None, :]
        dist = np.abs(xs - cx)
        # Soft shoulders on the column so its sides do not read as a cut.
        edge = np.clip((half[:, None] - dist) / np.maximum(half[:, None] * 0.28, 1.0), 0.0, 1.0)
        ramp[y0:y1] = vert[:, None] * edge
        ramp[y1:] = 0.0

        out = (np.clip(ramp, 0.0, 1.0) * 255).astype(np.uint8)
        out = cv2.GaussianBlur(out, (0, 0), 3.0)
        return Image.fromarray(out, mode="L")

    @staticmethod
    def jaw_metrics(mask: np.ndarray):
        """Chin point and jaw width, measured the same way as the template.

        Width comes from a band just above the lowest row rather than the
        widest row, which would be the hair.
        """
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return None
        chin_y = int(ys.max())
        band = mask[max(0, chin_y - 24):chin_y + 1]
        cols = np.nonzero(band.any(axis=0))[0]
        if cols.size == 0:
            return None
        return {
            "chin": (float((cols.min() + cols.max()) / 2.0), float(chin_y)),
            "width": float(cols.max() - cols.min()),
        }

    @staticmethod
    def similarity(src: np.ndarray, dst: np.ndarray):
        """Least-squares similarity (uniform scale, rotation, translation).

        Umeyama's estimate over all five landmarks, which beats measuring the
        detector's box for every part of the placement:

        * **Scale.** The box width is a function of head yaw — turn away from
          the camera and it narrows, so matching box widths silently enlarges
          a turned head. Landmark spread is a face measurement, not a bounding
          measurement, and five points average out the noise in any one.
        * **Rotation.** The eye line alone overstates tilt on a turned head,
          because it foreshortens. Fitting mouth and nose alongside the eyes
          constrains it.
        * **Position.** Anchoring on one point puts the whole head wherever
          that point lands; a least-squares fit spreads the error instead of
          concentrating it at the chin.

        Returns ``(scale, degrees, (tx, ty))`` mapping src onto dst.
        """
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        if src.shape != dst.shape or src.shape[0] < 2:
            return None

        src_mu, dst_mu = src.mean(axis=0), dst.mean(axis=0)
        src_c, dst_c = src - src_mu, dst - dst_mu
        var = (src_c ** 2).sum() / src.shape[0]
        if var < 1e-8:
            return None

        cov = (dst_c.T @ src_c) / src.shape[0]
        u, d, vt = np.linalg.svd(cov)
        # Guard against the fit reflecting rather than rotating, which a
        # mirrored or badly-ordered landmark set would otherwise produce.
        sign = np.ones(2)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            sign[-1] = -1.0
        rot = u @ np.diag(sign) @ vt
        scale = float((d * sign).sum() / var)
        trans = dst_mu - scale * (rot @ src_mu)
        angle = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
        return scale, angle, (float(trans[0]), float(trans[1]))

    # -- colour ----------------------------------------------------------

    def match_skin(self, shopper_rgb: np.ndarray, shopper_skin: np.ndarray,
                   strength: float = 0.85) -> Image.Image:
        """Shift the model's neck and arms toward the shopper's skin tone."""
        import cv2

        target = _lab_mean(shopper_rgb, shopper_skin)
        if target is None or self._template_skin_lab is None:
            return self.template

        delta = (target - self._template_skin_lab) * strength
        # A big L shift blows out the arms; chroma can move further than
        # luminance before it looks wrong.
        delta[0] = float(np.clip(delta[0], -26, 26))
        delta[1] = float(np.clip(delta[1], -18, 18))
        delta[2] = float(np.clip(delta[2], -18, 18))

        # Quantise the delta coarsely; a shopper holding still should not pay
        # for a re-tint every frame because their mean moved by one unit.
        key = tuple(np.round(delta / 5.0).astype(int))
        if key == self._tint_cache_key:
            return self._tint_cache

        lab = np.clip(self._skin_lab + delta, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(lab.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
        out = self._template_rgb.copy()
        out[self.skin_mask] = rgb
        tinted = Image.fromarray(out)
        self._tint_cache_key, self._tint_cache = key, tinted
        return tinted

    def light_match(self, head: Image.Image, strength: float = 0.35) -> Image.Image:
        """Nudge the head's brightness and white balance toward the template.

        Luminance and per-channel gain only — hue is left alone, because
        shifting a face's hue to match a backdrop is how people end up orange.
        """
        import cv2

        arr = np.asarray(head.convert("RGB"), dtype=np.float32)
        alpha = np.asarray(head.getchannel("A")) > 16
        if alpha.sum() < 200:
            return head

        ref = self._skin_rgb_mean
        if ref is None:
            return head

        cur = arr[alpha].mean(axis=0)
        gain = np.clip(ref / np.maximum(cur, 1e-3), 0.82, 1.22)
        gain = 1.0 + (gain - 1.0) * strength

        out = np.clip(arr * gain, 0, 255).astype(np.uint8)
        merged = Image.fromarray(out).convert("RGBA")
        merged.putalpha(head.getchannel("A"))
        return merged

    def match_neck_tone(self, layer: Image.Image, chin_xy, fade: float,
                        strength: float = 0.9) -> Image.Image:
        """Pull the shopper's neck toward the model's, increasing downward.

        The opposite direction from ``match_skin``, and deliberately so: that
        one moves the *template* toward the shopper so his arms read as theirs,
        which is the whole frame. This one moves the *shopper's neck* toward
        the template over the short run where the two overlap, because at the
        collar it has to become his neck exactly or the dissolve shows.

        At the jaw the weight is zero — their real jaw tone is kept, which is
        the part anyone actually looks at.
        """
        import cv2

        cx, cy = float(chin_xy[0]), float(chin_xy[1])
        w, h = self.size
        y0, y1 = int(max(0, cy)), int(min(h, cy + fade))
        if y1 - y0 < 8:
            return layer

        # Sample the model's neck from the lower half of the blend run, where
        # it is unambiguously neck rather than jaw shadow.
        half = max(12, int(fade * 0.45))
        sx0, sx1 = int(max(0, cx - half)), int(min(w, cx + half))
        sy0 = int(cy + fade * 0.45)
        sy1 = int(min(h, cy + fade))
        region = self._template_rgb[sy0:sy1, sx0:sx1]
        sel = self.skin_mask[sy0:sy1, sx0:sx1]
        if region.size == 0 or sel.sum() < 60:
            return layer
        ref = cv2.cvtColor(region[sel].reshape(-1, 1, 3),
                           cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3).mean(axis=0)

        band = np.asarray(layer.crop((0, y0, w, y1)).convert("RGB"))
        alpha = np.asarray(layer.getchannel("A"))[y0:y1] > 16
        if alpha.sum() < 60:
            return layer

        lab = cv2.cvtColor(band, cv2.COLOR_RGB2LAB).astype(np.float32)
        cur = lab[alpha].mean(axis=0)
        delta = (ref - cur) * strength

        # Ramp the correction in with depth, matching neck_feather's profile so
        # tone and opacity move together.
        t = ((np.arange(y0, y1, dtype=np.float32) - cy) / max(fade, 1.0))
        wgt = np.clip(t * t * (3.0 - 2.0 * t), 0.0, 1.0)[:, None, None]

        lab = np.clip(lab + delta[None, None, :] * wgt, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        out = layer.copy()
        patch = Image.fromarray(rgb).convert("RGBA")
        patch.putalpha(layer.crop((0, y0, w, y1)).getchannel("A"))
        out.paste(patch, (0, y0))
        return out

    def bridge_gap(self, canvas: Image.Image, layer: Image.Image,
                   band: float = 0.30) -> Image.Image:
        """Paint skin into the gap between the shopper's jaw and the neck.

        The model's head was removed and the hole inpainted with backdrop, so
        wherever the shopper's head is narrower than his was, the viewer sees
        grey. Nothing about the placement is wrong when it happens; the two
        heads are simply different shapes.

        Only the bottom of that region is filled. Backdrop showing beside
        someone's temples is *correct* — a smaller head really does leave more
        wall visible — but backdrop showing where the jaw meets the shoulders
        reads as a notch cut out of the neck. So the fill is restricted to the
        lower ``band`` of the removed head, which is the jaw, and left alone
        above it.

        An earlier version keyed this off the chin instead, which could not
        work: the mask *is* the head, so it ends at the jaw, and excluding
        everything above the chin excluded all of it. The function was inert
        at every default offset.
        """
        import cv2

        rows = np.nonzero(self.head_mask_model.any(axis=1))[0]
        if rows.size == 0:
            return canvas
        top, bottom = int(rows.min()), int(rows.max())
        band_top = int(bottom - band * max(bottom - top, 1))

        alpha = np.asarray(layer.getchannel("A"))
        gap = self.head_mask_model & (alpha < 110)
        gap[:band_top] = False
        if gap.sum() < 120:
            return canvas

        # Close pinholes along the matte edge, then feather, so the fill meets
        # the head without an outline of its own.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        soft = cv2.morphologyEx(gap.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k)
        soft = cv2.GaussianBlur(soft, (0, 0), 6.0)

        # Sample the tone from the canvas being painted, not from the original
        # template. By this point match_skin has already shifted the neck and
        # arms toward the shopper — up to 26 L and 18 a/b — so filling with the
        # model's own mean would put his skin tone directly against a neck
        # recoloured to theirs, a visible step exactly where one is being
        # hidden.
        base = np.asarray(canvas.convert("RGB"))
        near = self.skin_mask.copy()
        near[:band_top] = False
        if near.sum() < 200:
            near = self.skin_mask
        if near.sum() < 200:
            return canvas
        tone = base[near].mean(axis=0)

        fill = Image.new("RGBA", self.size, tuple(int(c) for c in tone) + (255,))
        fill.putalpha(Image.fromarray(soft, mode="L"))
        return Image.alpha_composite(canvas, fill)

    def head_match(self, layer: Image.Image, dy: float = 0.0) -> dict:
        """How the placed head measures against the one that belongs here.

        Turns "it looks slightly off" into a number. The model's head mask is
        the yardstick: measure both widths at his eye line — a stable height
        on a head, unlike the bounding box, whose top depends entirely on
        hairstyle — and the ratio is how much too big or small the shopper's
        head is. 1.00 means it occupies the same width his did.

        Reported rather than applied. A ratio away from 1.0 is not necessarily
        an error: a genuinely narrower head *should* measure narrower, and
        forcing it to match would stretch people to the model's proportions.
        It tells you whether the remaining disagreement is placement or taste.
        """
        if self.head_mask_model is None or not self.landmarks:
            return {}

        eyes = [self.landmarks[k] for k in ("right_eye", "left_eye")
                if k in self.landmarks]
        if len(eyes) < 2:
            return {}
        # Follow the head. The reference row is where the model's eyes are,
        # but the shopper is pasted `dy` lower, so measuring at the fixed row
        # samples a different height on their head — at the extremes of the
        # up/down slider that is their mouth or forehead, which are markedly
        # narrower than the eyes. The ratio would then report a sizing error
        # that is purely a placement offset, and suggestScale would advise a
        # correction that makes it worse.
        row = int(round(sum(e[1] for e in eyes) / len(eyes) + dy))
        h = self.size[1]
        if not (0 <= row < h):
            return {}

        # Average a few rows: one row through a stray hair strand is noise.
        lo, hi = max(0, row - 6), min(h, row + 7)
        def span(mask):
            widths = []
            for y in range(lo, hi):
                xs = np.nonzero(mask[y])[0]
                if xs.size:
                    widths.append(float(xs.max() - xs.min()))
            return float(np.median(widths)) if widths else 0.0

        # The reference is not the stored mask. That mask is grown so no hair
        # fringe survives the inpaint, and it never went through the feather,
        # resample and alpha cut the shopper's matte does — measured against
        # it a perfect self-composite reads 0.87, and anyone trusting that
        # would scale the head up by 15% to "fix" it.
        #
        # So the reference is what the model's own head measures *after being
        # put through this exact path*, recorded once by prep_template. Like
        # for like, by construction.
        got = span(np.asarray(layer.getchannel("A")) > 40)
        want = float(self.meta.get("headSpanRef") or 0.0)
        if want < 10:
            # No recorded reference: fall back to the mask, de-dilated, and
            # say so rather than reporting a number that is quietly biased.
            import cv2 as _cv2
            grown = int(self.meta.get("dilate", 0))
            model = self.head_mask_model
            if grown > 0:
                k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (grown * 2 + 1,) * 2)
                model = _cv2.erode(model.astype(np.uint8) * 255, k) > 127
            want = span(model)

        if want < 10 or got < 10:
            return {}
        return {
            "headMatch": round(got / want, 3),
            "suggestScale": round(want / got, 3),
            # The raw measurement, so calibrate_template can record a
            # reference taken by this exact code rather than re-deriving one
            # from pixel colour and hoping the two agree.
            "headSpanPx": round(got, 1),
        }

    # -- the composite ---------------------------------------------------

    def compose(self, crop_rgb: Image.Image, parse: np.ndarray, head_idx, neck_idx,
                eye_l, eye_r, smoother: Smoother, offsets=None, feather: float = 5.0,
                face_box=None, landmarks=None):
        import cv2

        offsets = offsets or {}
        # Measure on the head alone. jaw_metrics takes the lowest row as the
        # chin, so including the neck would put the "chin" at the collarbone
        # and throw both the scale and the anchor out.
        head_only = self.head_mask(parse, head_idx, neck_idx)
        metrics = self.jaw_metrics(head_only)
        if metrics is None or metrics["width"] < 10:
            return None, {"reason": "no head mask"}

        blend_neck = bool(offsets.get("neckBlend", True))
        mask = (self.head_mask(parse, head_idx, neck_idx, with_neck=True)
                if blend_neck else head_only)

        # Feather before any resampling, so the edge stays soft through it.
        alpha = Image.fromarray(mask, mode="L").filter(
            ImageFilter.GaussianBlur(feather)
        )
        head = crop_rgb.convert("RGBA")
        head.putalpha(alpha)

        # Trim to the head's own bounds. Scaling a mostly-transparent 512x512
        # canvas by 4x makes a 4500px image, nearly all of it empty.
        ys, xs = np.nonzero(mask)
        pad = int(feather * 3)
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(head.width, int(xs.max()) + pad)
        y1 = min(head.height, int(ys.max()) + pad)
        head = head.crop((x0, y0, x1, y1))

        fit = None
        if self.landmarks and landmarks:
            keys = [k for k in self.landmarks if k in landmarks]
            if len(keys) >= 3:
                fit = self.similarity(
                    np.array([landmarks[k] for k in keys], dtype=np.float64),
                    np.array([self.landmarks[k] for k in keys], dtype=np.float64),
                )

        fit_x = None
        if fit is not None:
            # Five-point fit: scale, rotation and position in one estimate.
            raw_scale, fit_angle, fit_trans = fit
            # Anchor stays the chin so the vertical knob keeps meaning "how
            # much neck shows"; the fit supplies the scale and the tilt, which
            # are the parts the box measured badly.
            anchor_src = (float(face_box[0] + face_box[2] / 2.0),
                          float(face_box[1] + face_box[3])) if face_box is not None \
                else metrics["chin"]
            # Horizontal placement comes from the fit as well. Taking scale and
            # rotation from it but leaving x on the face box's centre was an
            # inconsistency, and a visible one: the box centre tracks the
            # *face*, which shifts off the skull's midline as soon as the head
            # turns even slightly, while a neck attaches under the midline. The
            # head then sits a few px to one side of the collar and the neck
            # appears to run off at an angle. Mapping the anchor through the
            # fit puts the face where the model's was, and his was over his own
            # neck by construction.
            th = math.radians(fit_angle)
            cos_t, sin_t = math.cos(th), math.sin(th)
            ax, ay = anchor_src
            fit_x = raw_scale * (cos_t * ax - sin_t * ay) + fit_trans[0]
        elif face_box is not None:
            # Fallback: the detector's box on both sides. Comparable, but a
            # function of head yaw, so a turned head comes out too large.
            raw_scale = self.face_width / max(float(face_box[2]), 1.0)
            fit_angle = None
            anchor_src = (float(face_box[0] + face_box[2] / 2.0),
                          float(face_box[1] + face_box[3]))
        else:
            raw_scale = self.jaw_width / metrics["width"]
            fit_angle = None
            anchor_src = metrics["chin"]

        dx, dy = eye_r[0] - eye_l[0], eye_r[1] - eye_l[1]
        # Clamp the tilt. A billboard portrait wants a head that sits straight;
        # correcting the full eye-line angle on a head that is also turned
        # away from the camera reads as a cocked head rather than a pose.
        raw_angle = fit_angle if fit_angle is not None else math.degrees(math.atan2(dy, dx))
        raw_angle = max(-8.0, min(8.0, raw_angle))
        angle = smoother("angle", raw_angle)

        # A manual multiplier on top of the automatic scale. Matching face
        # width gets the head roughly right, but the model's head-to-body
        # proportion is his, not the shopper's, so the last 15% is taste.
        raw_scale *= float(offsets.get("headScale", 1.0))
        scale = smoother("scale", raw_scale)
        chin_x = smoother("chin_x", anchor_src[0]) - x0
        chin_y = smoother("chin_y", anchor_src[1]) - y0

        # Pad, rotate about the chin, then scale — the chin is the anchor
        # because that is the point that has to meet the template's jawline.
        pw, ph = head.width // 2, head.height // 2
        padded = Image.new("RGBA", (head.width + 2 * pw, head.height + 2 * ph), (0, 0, 0, 0))
        padded.paste(head, (pw, ph), head)
        anchor = (chin_x + pw, chin_y + ph)
        rotated = padded.rotate(angle, resample=Image.BICUBIC, center=anchor, expand=False)

        new_size = (max(1, int(rotated.width * scale)), max(1, int(rotated.height * scale)))
        scaled = rotated.resize(new_size, Image.LANCZOS)
        anchor_scaled = (anchor[0] * scale, anchor[1] * scale)

        scaled = self.light_match(scaled)

        shopper_rgb = np.asarray(crop_rgb.convert("RGB"))
        skin_idx = offsets.get("skin_idx", 1)
        base = self.match_skin(shopper_rgb, parse == skin_idx)

        canvas = base.convert("RGBA")
        # Chin sits a few px below the template jawline so it covers the old
        # cut edge rather than meeting it exactly.
        target_x = (fit_x if fit_x is not None else self.chin[0]) \
            + float(offsets.get("dx", 0.0))
        target_y = self.chin[1] + 5.0 + float(offsets.get("dy", 0.0))
        paste = (int(round(target_x - anchor_scaled[0])),
                 int(round(target_y - anchor_scaled[1])))

        layer = Image.new("RGBA", self.size, (0, 0, 0, 0))
        layer.paste(scaled, paste, scaled)

        if blend_neck:
            # Dissolve their neck into his over the run from jaw to collar, so
            # there is no single row where one becomes the other.
            collar_y = float(self.meta.get("collar", self.chin[1] + 125))
            fade = max(32.0, (collar_y - target_y) * float(offsets.get("neckFade", 1.0)))
            # Rotation is applied to the whole cutout, so straightening a
            # tilted head tilts the neck with it — and the model's neck is
            # vertical on an upright body. A leaning neck running down to a
            # vertical collar is what reads as "the neck goes off to the
            # side"; it is not a horizontal placement error, which is why
            # nudging left/right never quite fixes it. The more we rotate, the
            # sooner the shopper's neck should give way to his.
            fade *= 1.0 - min(abs(angle) / 12.0, 0.45)
            half_top = max(8.0, metrics["width"] * scale * 0.56)
            # Tone first, then alpha. Inside the fade both necks are visible at
            # once, so if their tones disagree the cross-fade shows as a ghost
            # rather than a blend — matching the tone is what makes the alpha
            # ramp invisible.
            layer = self.match_neck_tone(layer, (target_x, target_y), fade)
            neck_alpha = self.neck_feather((target_x, target_y), half_top, fade)
            layer.putalpha(ImageChops.multiply(layer.getchannel("A"), neck_alpha))

        if self.head_mask_model is not None:
            canvas = self.bridge_gap(canvas, layer)

        canvas = Image.alpha_composite(canvas, layer)

        return canvas.convert("RGB"), {
            **self.head_match(layer, dy=float(offsets.get("dy", 0.0))),
            "jawWidth": round(metrics["width"], 1),
            "scale": round(scale, 3),
            "angle": round(angle, 1),
            "neckBlend": blend_neck,
            "align": "landmarks" if fit is not None else "facebox",
        }
