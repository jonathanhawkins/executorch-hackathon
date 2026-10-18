"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// Short, closed questions. This quantized 1B sees accurately but its decoder
// is fragile: open-ended prompts like "describe this in detail" often collapse
// into repetition or filler, while direct questions come back clean.
const PRESETS = [
  "What do you see?",
  "Is there a person in this image?",
  "What colour is the largest object?",
  "How many people are visible?",
  "What room is this?",
];

const LANGUAGES = ["Spanish", "French", "German", "Japanese", "Mandarin Chinese", "Arabic"];

// Modes where the shopper lines up in the oval and the result is a held shot,
// rather than a live composite that tracks them. Both want the framing guide,
// the shutter button and the branding; they differ only in what the shutter
// runs.
const POSED = ["billboard", "faceswap", "generate"];
// Both try-on modes place the garment the same way, so they share the
// placement knobs and the shoulder readouts. They differ only in whether
// the body segmenter clips the result.
const TRYON = ["tryon", "tryon-clip"];
// Modes that take a shot, by button or by voice. The try-on modes gate the
// voice shutter on the shoulders being in frame rather than on the oval,
// so "ready" comes from a different field for them.
const SHUTTER = [...POSED, ...TRYON];


/**
 * A wireframe head drawn in the pose the detector currently measures, over
 * the fixed oval that marks the pose we want. The oval alone says where to
 * be; this says where you *are*, which is the half that was missing — "turn
 * less" is hard to act on, a meridian you can centre is not.
 *
 * Coordinates are the camera's own 640x480, so GUIDE's fractions drop
 * straight in and the drawing cannot drift from what check_alignment tests.
 */
function PoseHead({ roll = 0, yaw = 0, ok = false, cxN, cyN, relN }) {
  const W = 640;
  const H = 480;
  // Drawn on the head the detector actually found, not at the target. That
  // makes the instruction concrete: move until the solid head sits inside the
  // dashed oval and stands upright. A wireframe pinned at the target that
  // only rotates shows the pose but gives you nothing to steer towards.
  const cx = (cxN ?? 0.5) * W;
  const cy = (cyN ?? 0.44) * H;
  // Calibrated so a correctly-framed head draws exactly on the target oval:
  // at the gate's nominal face width (0.34 of frame) these give GUIDE's own
  // rx and ry. Lining the solid head up with the dashed one is then the whole
  // instruction, and it is the same test check_alignment applies.
  const rx = (relN ? relN * 0.471 : 0.16) * W;
  const ry = rx * 1.359;

  // Yaw is the nose's offset from the eye midpoint in units of eye
  // separation, so it is already scale-free — multiplying by rx turns it back
  // into pixels at whatever size the head happens to be.
  const shift = Math.max(-1.6, Math.min(1.6, yaw)) * rx * 1.9;
  // Turning the head foreshortens the far side. Narrowing the silhouette by
  // the cosine is what stops this reading as a flat oval slid sideways.
  const squash = Math.cos(Math.min(Math.abs(yaw) * 1.5, 1.1));
  const eyeY = cy - ry * 0.38;
  const stroke = ok ? "var(--accent, #4ade80)" : "rgba(255,255,255,0.78)";

  return (
    <svg className="pose-head" viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      <g
        transform={`rotate(${roll} ${cx} ${cy})`}
        fill="none"
        stroke={stroke}
        strokeWidth="2.5"
        strokeLinecap="round"
      >
        <ellipse cx={cx} cy={cy} rx={rx * squash} ry={ry} opacity="0.95" />
        {/* Meridian: the nose line. It bows to whichever side the nose is on,
            so centring it on the dashed line is the whole instruction. */}
        <path
          d={`M ${cx + shift * 0.45} ${cy - ry} Q ${cx + shift * 1.5} ${cy} ${cx + shift * 0.45} ${cy + ry}`}
          opacity="0.85"
        />
        {/* Equator: the eye line, bowed the way a sphere's would be. */}
        <path
          d={`M ${cx - rx * squash} ${eyeY} Q ${cx + shift * 0.5} ${eyeY + ry * 0.12} ${cx + rx * squash} ${eyeY}`}
          opacity="0.6"
        />
        <circle cx={cx + shift * 0.5 - rx * squash * 0.42} cy={eyeY} r="7" fill={stroke} stroke="none" opacity="0.9" />
        <circle cx={cx + shift * 0.5 + rx * squash * 0.42} cy={eyeY} r="7" fill={stroke} stroke="none" opacity="0.9" />
      </g>
    </svg>
  );
}

export default function Home() {
  const [prompt, setPrompt] = useState("What do you see?");
  const [temperature, setTemperature] = useState(1.0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null);
  const [feedSrc, setFeedSrc] = useState("/api/snapshot");
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);

  const [bb, setBb] = useState({ running: false, state: null, control: {} });
  const [bbBusy, setBbBusy] = useState(false);
  // Compare slider: 1 = the composite, 0 = the reference underneath.
  const [mix, setMix] = useState(1);
  const [ref, setRef] = useState("original");
  // Sliders are driven locally. Binding them straight to server state made the
  // thumb snap back mid-drag, because the poll is ~1.2s behind the pointer.
  const [knobs, setKnobs] = useState({ offsetY: 0, halfWidth: 1.15, drop: 0.45, headScale: 1.12, fineX: 0, fineY: 36, neckFade: 1.0 });
  const draggingRef = useRef(false);
  const pushRef = useRef(null);
  const [bbSrc, setBbSrc] = useState("/api/billboard?image=1");
  const [mic, setMic] = useState({ running: false, control: {}, entries: [] });
  const [micBusy, setMicBusy] = useState(false);
  const transcriptRef = useRef(null);

  // Poll the snapshot endpoint for a live preview. The camera runs at 5fps,
  // so there is nothing to gain from a faster loop.
  useEffect(() => {
    const id = setInterval(() => {
      setFeedSrc(`/api/snapshot?t=${Date.now()}`);
    }, 220);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const load = () =>
      fetch("/api/status")
        .then((r) => r.json())
        .then(setStatus)
        .catch(() => setStatus(null));
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const load = () =>
      fetch("/api/billboard")
        .then((r) => r.json())
        .then((data) => {
          setBb(data);
          if (!draggingRef.current && data.control) {
            setKnobs((k) => ({
              offsetY: data.control.offsetY ?? k.offsetY,
              halfWidth: data.control.halfWidth ?? k.halfWidth,
              headScale: data.control.headScale ?? k.headScale,
              fineX: data.control.fineX ?? k.fineX,
              fineY: data.control.fineY ?? k.fineY,
              neckFade: data.control.neckFade ?? k.neckFade,
              drop: data.control.drop ?? k.drop,
            }));
          }
        })
        .catch(() => {});
    load();
    const id = setInterval(load, 1200);
    return () => clearInterval(id);
  }, []);

  // Refresh the composite on its own cadence. The daemon rewrites it roughly
  // once a second, so polling faster only costs bandwidth.
  useEffect(() => {
    const id = setInterval(() => {
      setBbSrc(`/api/billboard?image=1&t=${Date.now()}`);
    }, 900);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const load = () =>
      fetch("/api/transcript")
        .then((r) => r.json())
        .then(setMic)
        .catch(() => {});
    load();
    const id = setInterval(load, 1500);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const el = transcriptRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [mic.entries.length]);

  const micPost = useCallback(async (body) => {
    setMicBusy(true);
    try {
      await fetch("/api/transcript", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const fresh = await fetch("/api/transcript").then((r) => r.json());
      setMic(fresh);
    } finally {
      setMicBusy(false);
    }
  }, []);

  const pushKnobs = useCallback((next) => {
    setKnobs(next);
    draggingRef.current = true;
    clearTimeout(pushRef.current);
    // Coalesce the drag into one request rather than one per pixel.
    pushRef.current = setTimeout(async () => {
      await fetch("/api/billboard", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "mode", ...next }),
      }).catch(() => {});
      draggingRef.current = false;
    }, 180);
  }, []);

  const bbPost = useCallback(async (action, extra = {}) => {
    // A reduced compare opacity is for inspecting the shot in front of you,
    // not a setting. Left in place it survives Retake and renders the *next*
    // capture as a ghost over the reference, which reads as a failed capture
    // with nothing on screen to explain it.
    if (action === "capture" || action === "retake") setMix(1);
    setBbBusy(true);
    try {
      await fetch("/api/billboard", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, ...extra }),
      });
      setBb(await fetch("/api/billboard").then((r) => r.json()));
    } finally {
      setBbBusy(false);
    }
  }, []);

  const analyze = useCallback(async () => {
    setBusy(true);
    setError(null);
    setElapsed(0);
    const startedAt = Date.now();
    timerRef.current = setInterval(
      () => setElapsed((Date.now() - startedAt) / 1000),
      100,
    );
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, temperature }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Inference failed.");
        setResult(null);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      clearInterval(timerRef.current);
      setBusy(false);
    }
  }, [prompt, temperature]);

  const s = result?.stats;
  const translating = Boolean(mic.control?.translate);

  return (
    <div className="wrap">
      <header>
        <h1>On-device AI</h1>
        <span className="muted" style={{ fontSize: 13 }}>
          Arduino VENTUNO Q · QCS8275 · Hexagon V75
        </span>
        {/* The walkthrough deck. Served from the board like everything else,
            so the talk survives the network being unplugged on stage. */}
        <a className="deck-link" href="/deck">
          Walkthrough deck →
        </a>
      </header>
      <p className="sub">
        A USB camera and microphone on the board, three models on the Hexagon
        NPU, and this page served from the same device. Nothing leaves the
        board — unplug the network and it still works.
      </p>

      <div className="pills">
        <span className="pill">
          <span className={`dot ${status?.camera ? "ok" : "bad"}`} />
          camera {status?.camera ? "live" : "offline"}
        </span>
        <span className="pill">
          <span className={`dot ${mic.running ? "ok" : "bad"}`} />
          mic {mic.running ? "listening" : "idle"}
        </span>
        <span className="pill">
          <span className={`dot ${status?.model ? "ok" : "bad"}`} />
          InternVL3-1B
        </span>
        <span className="pill">
          <span className={`dot ${bb.running ? "ok" : ""}`} />
          BiSeNet {bb.running ? "parsing" : "idle"}
        </span>
        <span className="pill">
          <span className="dot ok" />
          Whisper-tiny
        </span>
        <span className="pill">
          <span className={`dot ${translating ? "ok" : ""}`} />
          Gemma 4 {translating ? "translating" : "idle"}
        </span>
      </div>

      <div className="panel billboard-panel">
        <h2>Virtual billboard · BiSeNet face parsing on the NPU</h2>
        <div className="body">
          <div className="billboard-row">
            <div className="billboard-stage">
              {bb.state?.phase === "shot" && mix < 1 ? (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  className="billboard-base"
                  src={`/api/billboard?image=${ref}`}
                  alt="product photo without the shopper"
                />
              ) : null}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                className="billboard-img"
                src={bbSrc}
                alt="virtual billboard"
                style={bb.state?.phase === "shot" ? { opacity: mix } : undefined}
              />

              {/* Framing guide. Drawn from the same fractions the daemon
                  checks against, so what you aim at is what it tests. */}
              {POSED.includes(bb.control?.mode || "billboard") &&
               bb.state?.phase !== "shot" && bb.state?.phase !== "working" ? (
                <div className="guide-box">
                  <div
                    className={`guide-oval ${bb.state?.aligned ? "ok" : ""}`}
                  >
                    <span className="guide-centre" />
                  </div>
                  {bb.state?.roll != null ? (
                    <PoseHead
                      roll={bb.state.roll}
                      yaw={bb.state.yaw ?? 0}
                      ok={Boolean(bb.state.aligned)}
                      cxN={bb.state.faceCx}
                      cyN={bb.state.faceCy}
                      relN={bb.state.faceRel}
                    />
                  ) : null}
                  <div className="guide-text">
                    {bb.state?.guidance || "Step into frame"}
                  </div>
                </div>
              ) : null}

              {/* Branding is an overlay, not baked into the composite — so it
                  stays crisp at any size and a price change is a text edit. */}
              {POSED.includes(bb.control?.mode || "billboard") &&
               bb.state?.phase === "shot" ? (
                <div className="brand-overlay">
                  <div className="ad-top">
                    <div className="brand-mark">
                      <div className="brand-name">NORTHWIND</div>
                      <div className="brand-rule" />
                      <div className="brand-tag">EST. 2014 — SAN FRANCISCO</div>
                    </div>
                    <div className="ad-badge">NEW IN</div>
                  </div>

                  <div className="ad-bottom">
                    <div>
                      <div className="ad-product-name">The Everyday Tee</div>
                      <div className="ad-product-meta">
                        Heavyweight cotton &middot; Six colours &middot; XS&ndash;XXL
                      </div>
                    </div>
                    <div className="ad-price">$28</div>
                  </div>
                </div>
              ) : null}
              {bb.state?.phase === "working" ? (
                <div className="billboard-overlay working">
                  <span className="spin" />
                  {bb.state?.guidance || "Working\u2026"}
                </div>
              ) : !bb.running ? (
                <div className="billboard-overlay">Not running</div>
              ) : bb.state && !bb.state.ok ? (
                <div className="billboard-overlay">{bb.state.reason}</div>
              ) : null}
            </div>

            <div className="billboard-side">
              <div className="row" style={{ marginTop: 0 }}>
                <button
                  onClick={() => bbPost(bb.running ? "stop" : "start")}
                  disabled={bbBusy}
                >
                  {bbBusy ? <span className="spin" /> : null}
                  {bb.running ? "Stop" : "Start"}
                </button>

                {SHUTTER.includes(bb.control?.mode || "billboard") ? (
                  bb.state?.phase === "shot" ? (
                    <button className="ghost" onClick={() => bbPost("retake")}>
                      Retake
                    </button>
                  ) : (
                    <button
                      onClick={() => bbPost("capture")}
                      className={
                        (bb.state?.aligned ?? bb.state?.shouldersOnScreen)
                          ? ""
                          : "ghost"
                      }
                      disabled={bb.state?.phase === "working"}
                    >
                      Take photo
                    </button>
                  )
                ) : null}

                <select
                  className="lang-select"
                  value={bb.control?.mode || "billboard"}
                  onChange={(e) => bbPost("mode", { mode: e.target.value })}
                >
                  <option value="billboard">Billboard — head on product</option>
                  <option value="tryon">Try-on — shirt on you</option>
                  <option value="tryon-clip">Try-on + body clip — shirt cut to your outline</option>
                  <option value="faceswap">Face swap — your face on the model</option>
                  <option value="generate">Generate — the garment, made for you</option>
                  <option value="segment">Segmentation — head cutout</option>
                </select>
              </div>
              {bb.control?.mode === "generate" && (
                <div className="row">
                  <label>Garment</label>
                  <select
                    className="lang-select"
                    value={bb.control?.garment || "navy-tee"}
                    onChange={(e) =>
                      bbPost("mode", { mode: "generate", garment: e.target.value })
                    }
                  >
                    {Object.entries(bb.state?.garments || {}).map(([slug, label]) => (
                      <option key={slug} value={slug}>{label}</option>
                    ))}
                  </select>
                </div>
              )}

              {(bb.control?.mode || "billboard") === "billboard" &&
               bb.state?.phase === "shot" ? (
                <div className="knobs compare">
                  <label className="knob">
                    <span>compare</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.02}
                      value={mix}
                      onChange={(e) => setMix(Number(e.target.value))}
                    />
                    <b>{Math.round(mix * 100)}%</b>
                  </label>
                  <select
                    className="lang-select"
                    value={ref}
                    onChange={(e) => setRef(e.target.value)}
                  >
                    <option value="original">against the model&rsquo;s own head</option>
                    <option value="base">against the empty product photo</option>
                  </select>
                </div>
              ) : null}

              {(bb.control?.mode || "billboard") === "billboard" &&
               bb.state?.phase === "shot" ? (
                <div className="knobs">
                  {[
                    ["headScale", "head size", 0.75, 1.6, 0.02],
                    ["fineY", "up / down", -120, 160, 2],
                    ["fineX", "left / right", -80, 80, 2],
                    ["neckFade", "neck blend", 0.4, 1.8, 0.05],
                  ].map(([key, label, min, max, step]) => (
                    <label key={key} className="knob">
                      <span>{label}</span>
                      <input
                        type="range"
                        min={min}
                        max={max}
                        step={step}
                        value={knobs[key]}
                        onChange={(e) =>
                          pushKnobs({ ...knobs, [key]: Number(e.target.value) })
                        }
                      />
                      <b>
                        {key === "headScale"
                          ? Number(knobs[key]).toFixed(2)
                          : Math.round(Number(knobs[key]))}
                      </b>
                    </label>
                  ))}
                  <label className="knob-check">
                    <input
                      type="checkbox"
                      checked={bb.control?.neckBlend !== false}
                      onChange={(e) => bbPost("mode", { neckBlend: e.target.checked })}
                    />
                    blend my neck into the model&rsquo;s
                  </label>
                  <button
                    className="ghost"
                    style={{ alignSelf: "flex-start", fontSize: 12, padding: "5px 10px" }}
                    onClick={() =>
                      pushKnobs({ ...knobs, headScale: 1.12, fineX: 0, fineY: 36, neckFade: 1.0 })
                    }
                  >
                    Reset placement
                  </button>
                </div>
              ) : null}

              {TRYON.includes(bb.control?.mode || "billboard") ? (
                <div className="knobs">
                  {[
                    ["offsetY", "up / down", -1.2, 1.2, 0.05],
                    ["halfWidth", "width", 0.7, 2.0, 0.05],
                    ["drop", "shoulder drop", 0.1, 1.2, 0.05],
                  ].map(([key, label, min, max, step]) => (
                    <label key={key} className="knob">
                      <span>{label}</span>
                      <input
                        type="range"
                        min={min}
                        max={max}
                        step={step}
                        value={knobs[key]}
                        onChange={(e) =>
                          pushKnobs({ ...knobs, [key]: Number(e.target.value) })
                        }
                      />
                      <b>{Number(knobs[key]).toFixed(2)}</b>
                    </label>
                  ))}
                  <label className="knob-check">
                    <input
                      type="checkbox"
                      checked={Boolean(bb.control?.debug)}
                      onChange={(e) => bbPost("mode", { debug: e.target.checked })}
                    />
                    show shoulder markers
                  </label>
                  <button
                    className="ghost"
                    style={{ alignSelf: "flex-start", fontSize: 12, padding: "5px 10px" }}
                    onClick={() =>
                      pushKnobs({ offsetY: 0, halfWidth: 1.15, drop: 0.45 })
                    }
                  >
                    Reset placement
                  </button>
                </div>
              ) : null}

              <p className="muted" style={{ fontSize: 13, lineHeight: 1.6 }}>
                Stand in front of the camera, facing it. The head, hair and neck
                are segmented on the Hexagon NPU and dropped into the neck of
                the product photo. The collar is drawn last so the cut edge
                disappears under it.
              </p>

              {bb.state?.ok && bb.state?.phase === "live" ? (
                <div className="stats" style={{ marginTop: 4 }}>
                  <div className="stat">
                    <div className="k">Framing</div>
                    <div className="v" style={{ fontSize: 14 }}>
                      {bb.state.aligned ? "Ready" : "Adjust"}
                    </div>
                  </div>
                  <div className="stat">
                    <div className="k">Face width</div>
                    <div className="v">{bb.state.faceWidth} px</div>
                  </div>
                  {bb.state.roll != null ? (
                    <div className="stat">
                      <div className="k">Tilt</div>
                      <div
                        className="v"
                        style={{ color: Math.abs(bb.state.roll) > 10 ? "var(--warn, #e0a33e)" : undefined }}
                      >
                        {bb.state.roll > 0 ? "+" : ""}{bb.state.roll}&deg;
                      </div>
                    </div>
                  ) : null}
                  {bb.state.yaw != null ? (
                    <div className="stat">
                      <div className="k">Turn</div>
                      <div
                        className="v"
                        style={{ color: Math.abs(bb.state.yaw) > 0.25 ? "var(--warn, #e0a33e)" : undefined }}
                      >
                        {Math.round(bb.state.yaw * 100)}
                      </div>
                    </div>
                  ) : null}
                  <div className="stat">
                    <div className="k">Shutter</div>
                    <div className="v" style={{ fontSize: 14 }}>
                      say &ldquo;take a photo&rdquo;
                    </div>
                  </div>
                </div>
              ) : bb.state?.ok ? (
                <div className="stats" style={{ marginTop: 4 }}>
                  {bb.state.headMatch != null ? (
                    <div className="stat">
                      <div className="k">Head vs model</div>
                      <div
                        className="v"
                        title={`Scale that would match it exactly: ${bb.state.suggestScale}`}
                        style={{ color: Math.abs(bb.state.headMatch - 1) > 0.08 ? "var(--warn, #e0a33e)" : undefined }}
                      >
                        {Math.round(bb.state.headMatch * 100)}%
                      </div>
                    </div>
                  ) : null}
                  {/* Every mode writes a different subset of these — a
                      generated shot has no NPU parse time and no head
                      coverage — so each one renders only if it was measured,
                      rather than printing "NaN ms" and "undefined%". */}
                  {bb.state.npuMs != null ? (
                    <div className="stat">
                      <div className="k">NPU parse</div>
                      <div className="v">{bb.state.npuMs} ms</div>
                    </div>
                  ) : null}
                  {bb.state.totalMs != null ? (
                    <div className="stat">
                      <div className="k">Frame total</div>
                      <div className="v">{Math.round(bb.state.totalMs)} ms</div>
                    </div>
                  ) : null}
                  {bb.state.faceWidth != null ? (
                    <div className="stat">
                      <div className="k">Face width</div>
                      <div className="v">{bb.state.faceWidth} px</div>
                    </div>
                  ) : null}
                  {TRYON.includes(bb.state.mode) ? (
                    <div className="stat">
                      <div className="k">Shoulders</div>
                      <div className="v">{bb.state.shoulderPx ?? "—"} px</div>
                    </div>
                  ) : bb.state.headCoverage != null ? (
                    <div className="stat">
                      <div className="k">Head area</div>
                      <div className="v">{bb.state.headCoverage}%</div>
                    </div>
                  ) : null}
                  {bb.state.mode === "tryon-clip" ? (
                    <div className="stat">
                      <div className="k">On the body</div>
                      <div className="v">{bb.state.clipKept ?? "—"}%</div>
                    </div>
                  ) : null}
                  {TRYON.includes(bb.state.mode) ? (
                    <div className="stat">
                      <div className="k">Shutter</div>
                      <div
                        className="v"
                        style={{
                          fontSize: 13,
                          color: bb.state.shouldersOnScreen
                            ? undefined
                            : "var(--warn, #e0a33e)",
                        }}
                      >
                        {bb.state.guidance ?? "say “take a photo”"}
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : (
                <p className="muted" style={{ fontSize: 12 }}>
                  {bb.running
                    ? "Looking for a face…"
                    : "Press Start, then step into frame."}
                </p>
              )}

              {TRYON.includes(bb.state?.mode) && bb.state?.shouldersOnScreen === false ? (
                <p className="err" style={{ fontSize: 12 }}>
                  Your shoulders are below the frame — step back so your torso
                  is visible, or the shirt lands off-screen.
                </p>
              ) : null}

              <p className="muted" style={{ fontSize: 11, lineHeight: 1.6 }}>
                NORTHWIND is a fictional store. Face the camera straight on and
                get close enough that your face spans 200px or more — the
                segmentation degrades on small or heavily angled faces.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid">
        <div className="panel">
          <h2>Live camera</h2>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="feed" src={feedSrc} alt="camera feed" />
          <div className="body">
            <textarea
              rows={2}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Ask something about what the camera sees"
            />
            <div className="presets">
              {PRESETS.map((p) => (
                <button key={p} type="button" onClick={() => setPrompt(p)}>
                  {p}
                </button>
              ))}
            </div>
            <div className="row">
              <button onClick={analyze} disabled={busy || !prompt.trim()}>
                {busy ? <span className="spin" /> : null}
                {busy ? `Analyzing ${elapsed.toFixed(1)}s` : "Analyze frame"}
              </button>
              {result ? (
                <button
                  className="ghost"
                  onClick={() => {
                    setResult(null);
                    setError(null);
                  }}
                >
                  Clear
                </button>
              ) : null}
              <label className="muted" style={{ fontSize: 12, marginLeft: "auto" }}>
                temp{" "}
                <input
                  type="range"
                  min="0.4"
                  max="1.5"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  style={{ verticalAlign: "middle", width: 90 }}
                />{" "}
                {temperature.toFixed(1)}
              </label>
            </div>
          </div>
        </div>

        <div className="panel">
          <h2>Model output</h2>
          <div className="body">
            {error ? <p className="err">{error}</p> : null}

            {!result && !error ? (
              <p className="muted" style={{ fontSize: 14 }}>
                Point the camera at something and press Analyze.
              </p>
            ) : null}

            {result ? (
              <>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img className="shot" src={result.image} alt="analyzed frame" />
                <p className="muted" style={{ fontSize: 12, margin: "0 0 6px" }}>
                  {result.prompt}
                </p>
                <div className="answer">{result.text}</div>
                {s ? (
                  <div className="stats">
                    <div className="stat">
                      <div className="k">Total</div>
                      <div className="v">{s.wallClockSec?.toFixed(1)}s</div>
                    </div>
                    <div className="stat">
                      <div className="k">Prompt eval</div>
                      <div className="v">{s.promptEvalRate?.toFixed(0) ?? "—"} t/s</div>
                    </div>
                    <div className="stat">
                      <div className="k">Generation</div>
                      <div className="v">{s.generateRate?.toFixed(0) ?? "—"} t/s</div>
                    </div>
                    <div className="stat">
                      <div className="k">Tokens</div>
                      <div className="v">{s.generatedTokens ?? "—"}</div>
                    </div>
                  </div>
                ) : null}
              </>
            ) : null}
          </div>
        </div>
      </div>

      <div className="panel" style={{ marginTop: 18 }}>
        <h2>Live microphone · Whisper → Gemma</h2>
        <div className="body">
          <div className="row" style={{ marginTop: 0 }}>
            <button
              onClick={() => micPost({ action: mic.running ? "stop" : "start" })}
              disabled={micBusy}
            >
              {micBusy ? <span className="spin" /> : null}
              {mic.running ? "Stop listening" : "Start listening"}
            </button>

            <label className="muted" style={{ fontSize: 13 }}>
              <input
                type="checkbox"
                checked={translating}
                onChange={(e) =>
                  micPost({ action: "config", translate: e.target.checked })
                }
                style={{ verticalAlign: "middle", marginRight: 6 }}
              />
              translate
            </label>

            <select
              className="lang-select"
              value={mic.control?.targetLang || "Spanish"}
              onChange={(e) => micPost({ action: "config", targetLang: e.target.value })}
            >
              {LANGUAGES.map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>

            <button className="ghost" onClick={() => micPost({ action: "clear" })}>
              Clear
            </button>
          </div>

          <div className="transcript" ref={transcriptRef}>
            {mic.entries.length === 0 ? (
              <p className="muted" style={{ fontSize: 14, margin: 0 }}>
                {mic.running
                  ? "Listening… speak into the camera's microphone."
                  : "Press Start listening, then speak into the camera's microphone."}
              </p>
            ) : (
              mic.entries.map((e, i) => (
                <div className="line" key={`${e.ts}-${i}`}>
                  <div className="when">
                    {new Date(e.ts * 1000).toLocaleTimeString()} · asr {e.asrSec}s
                    {e.mtSec ? ` · mt ${e.mtSec}s` : ""}
                  </div>
                  <div className="said">{e.text}</div>
                  {e.translation ? (
                    <div className="translated">
                      <span className="lang">{e.targetLang}</span> {e.translation}
                    </div>
                  ) : null}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <footer>
        Vision encoder, token embedding and decoder all execute on the Hexagon
        V75 via QNN. Prompt eval covers 270 tokens, 256 of which are the image.
        <br />
        Speech is Whisper-tiny on the same NPU (about 1s per utterance);
        translation is a second NPU pass through Gemma 4 (about 4s), so a
        translated line lands roughly 10s after it is spoken.
        <br />
        The vision model is a heavily quantized 1B: it recognises scenes
        reliably but its wording varies run to run. Short questions give the
        steadiest answers.
      </footer>
    </div>
  );
}
