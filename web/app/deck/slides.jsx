/**
 * The deck's content, as one array of slides.
 *
 * Kept deliberately short: fourteen slides, one idea each, a headline and at
 * most a line or two per card. The detail all exists in docs/ — a slide that
 * needs a paragraph is a slide the speaker ends up reading out.
 *
 * Every number is measured on the attached VENTUNO Q and quoted from the
 * repo's own docs rather than retyped from memory.
 *
 * Each entry is { id, nav, section, el }:
 *   nav      the label in the overview grid
 *   section  the eyebrow line; absent on full-bleed title slides, which also
 *            suppresses the footer
 */

function Title() {
  return (
    <div className="deck-slide deck-title">
      <div className="kicker">Open Source AI Week · Founders Inc · Oct 17–18 2026</div>
      <h1>
        Nothing leaves
        <br />
        the board.
        <br />
        <span>Including the diffusion model.</span>
      </h1>
      <p className="tagline">
        A virtual try-on booth running entirely on an Arduino VENTUNO Q. Eleven models on the
        Hexagon NPU. No network, no cloud.
      </p>
      <div className="meta">
        <div>
          <b>IoT track</b>
          Arduino VENTUNO Q · QCS8275
        </div>
        <div>
          <b>ExecuTorch 1.5.1</b>
          QNN / Hexagon v75
        </div>
        <div>
          <b>11 models</b>
          LLM · VLM · ASR · seg · swap · diffusion
        </div>
      </div>
    </div>
  );
}

/**
 * The store setup, drawn as a plain elevation: a portrait screen on a stand,
 * a camera on top of it, and the shopper standing back far enough for the
 * framing gate to be happy.
 *
 * Drawn rather than photographed because the point is the *arrangement* — how
 * far back the shopper stands, where the camera is, and that the board doing
 * all of it is the small thing on the floor. A photo of our desk would show a
 * laptop and a cable nest instead.
 *
 * Coordinates are to scale against a 1.75 m shopper: 1 m is about 154 units,
 * so the screen is a 55" panel turned portrait and the gap really is ~2 m.
 */
function StoreSetup() {
  const FLOOR = 330;
  return (
    <svg className="diagram" viewBox="0 0 1180 368" role="img"
      aria-label="A portrait screen on a stand with a camera on top, a shopper standing about two metres back, and the VENTUNO Q board on the floor beside the stand">
      <defs>
        <linearGradient id="fov" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="var(--d-accent)" stopOpacity="0.07" />
          <stop offset="100%" stopColor="var(--d-accent)" stopOpacity="0.015" />
        </linearGradient>
      </defs>

      {/* floor */}
      <line x1="90" y1={FLOOR} x2="1090" y2={FLOOR} stroke="var(--d-line)" strokeWidth="2" />
      <ellipse cx="365" cy={FLOOR} rx="80" ry="6" fill="#000" opacity="0.5" />
      <ellipse cx="790" cy={FLOOR} rx="52" ry="6" fill="#000" opacity="0.5" />

      {/* what the camera sees */}
      <path d="M365 70 L 900 44 L 900 330 Z" fill="url(#fov)" />
      <path d="M365 70 L 900 44" stroke="var(--d-accent)" strokeWidth="1" strokeDasharray="4 6" opacity="0.3" />
      <path d="M365 70 L 900 330" stroke="var(--d-accent)" strokeWidth="1" strokeDasharray="4 6" opacity="0.3" />
      <text x="620" y="96" fontSize="12" fill="var(--d-accent)" opacity="0.75" textAnchor="middle">
        what the camera sees
      </text>

      {/* the screen, turned portrait */}
      <rect x="300" y="73" width="130" height="220" rx="7" fill="var(--d-panel-2)" stroke="var(--d-line)" strokeWidth="2" />
      <rect x="308" y="81" width="114" height="204" rx="3" fill="#0c1a16" />
      {/* ...showing the shopper wearing the garment */}
      <circle cx="365" cy="126" r="19" fill="var(--d-accent)" opacity="0.85" />
      <path d="M338 180 q27 -22 54 0 l6 62 h-66 z" fill="var(--d-accent)" opacity="0.55" />
      <rect x="312" y="246" width="106" height="35" rx="2" fill="#0a0c10" opacity="0.65" />
      <text x="320" y="262" fontSize="9" fill="var(--d-mute)" letterSpacing="1.4">NORTHWIRED</text>
      <text x="320" y="274" fontSize="9" fill="var(--d-accent)">Crew tee · $38</text>

      {/* stand */}
      <rect x="356" y="293" width="18" height="28" fill="var(--d-panel-2)" stroke="var(--d-line)" />
      <rect x="326" y="321" width="78" height="8" rx="4" fill="var(--d-panel-2)" stroke="var(--d-line)" />

      {/* camera */}
      <rect x="348" y="62" width="34" height="11" rx="5" fill="var(--d-panel-2)" stroke="var(--d-line)" />
      <circle cx="365" cy="67.5" r="3.4" fill="var(--d-accent)" />

      {/* the board, on the floor beside the stand */}
      <rect x="196" y="306" width="74" height="20" rx="3" fill="var(--d-panel-2)" stroke="var(--d-accent-2)" strokeWidth="1.5" />
      <rect x="203" y="312" width="16" height="8" rx="1.5" fill="var(--d-accent-2)" opacity="0.6" />
      <circle cx="232" cy="316" r="2.5" fill="var(--d-dim)" />
      <circle cx="242" cy="316" r="2.5" fill="var(--d-dim)" />
      <circle cx="252" cy="316" r="2.5" fill="var(--d-dim)" />

      {/* the shopper */}
      <circle cx="790" cy="83" r="21" fill="var(--d-mute)" opacity="0.85" />
      <rect x="766" y="111" width="48" height="104" rx="17" fill="var(--d-mute)" opacity="0.85" />
      <rect x="753" y="116" width="12" height="92" rx="6" fill="var(--d-mute)" opacity="0.7" />
      <rect x="815" y="116" width="12" height="92" rx="6" fill="var(--d-mute)" opacity="0.7" />
      <rect x="770" y="206" width="40" height="26" rx="8" fill="var(--d-mute)" opacity="0.85" />
      <rect x="772" y="226" width="15" height="104" rx="7" fill="var(--d-mute)" opacity="0.85" />
      <rect x="793" y="226" width="15" height="104" rx="7" fill="var(--d-mute)" opacity="0.85" />

      {/* how far back */}
      <line x1="410" y1="348" x2="745" y2="348" stroke="var(--d-dim)" strokeWidth="1" strokeDasharray="5 5" />
      <path d="M410 348 l9 -4 v8 z M745 348 l-9 -4 v8 z" fill="var(--d-dim)" />
      <text x="577" y="365" fontSize="13" fill="var(--d-mute)" textAnchor="middle">
        about 2 m back — the on-screen oval tells them when they are in the right place
      </text>

      {/* labels */}
      <line x1="432" y1="52" x2="386" y2="64" stroke="var(--d-line)" strokeWidth="1" />
      <text x="440" y="50" fontSize="13" fill="var(--d-ink)">USB camera</text>
      <text x="440" y="68" fontSize="12" fill="var(--d-dim)">one frame per shutter</text>

      <line x1="288" y1="150" x2="300" y2="160" stroke="var(--d-line)" strokeWidth="1" />
      <text x="280" y="146" fontSize="13" fill="var(--d-ink)" textAnchor="end">Screen, turned portrait</text>
      <text x="280" y="164" fontSize="12" fill="var(--d-dim)" textAnchor="end">shows them wearing it,</text>
      <text x="280" y="180" fontSize="12" fill="var(--d-dim)" textAnchor="end">priced, like an ad</text>

      <text x="233" y="283" fontSize="13" fill="var(--d-accent-2)" textAnchor="middle">VENTUNO Q</text>
      <text x="233" y="299" fontSize="12" fill="var(--d-dim)" textAnchor="middle">everything runs here</text>

      <text x="880" y="150" fontSize="13" fill="var(--d-ink)">The shopper</text>
      <text x="880" y="168" fontSize="12" fill="var(--d-dim)">names a garment, presses once,</text>
      <text x="880" y="184" fontSize="12" fill="var(--d-dim)">and watches themselves wear it</text>

      {/* the punchline, bottom right: nothing goes out */}
      <g transform="translate(880 258)">
        <path d="M4 14 a14 14 0 0 1 20 0" stroke="var(--d-dim)" strokeWidth="2" fill="none" />
        <path d="M9 19 a7 7 0 0 1 10 0" stroke="var(--d-dim)" strokeWidth="2" fill="none" />
        <circle cx="14" cy="24" r="2.2" fill="var(--d-dim)" />
        <line x1="1" y1="27" x2="27" y2="5" stroke="var(--d-err)" strokeWidth="2.2" />
        <text x="38" y="21" fontSize="13" fill="var(--d-mute)">No network. No cloud.</text>
      </g>
    </svg>
  );
}

const slides = [
  { id: "title", nav: "Title", el: <Title /> },

  {
    id: "why",
    nav: "Why on-device",
    section: "Premise",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">01</span> Premise <span className="rule" />
        </div>
        <h2>
          One question decides the build: <em>why must this run on the device?</em>
        </h2>
        <p className="deck-lede">
          If the demo would be better with a server, it is the wrong demo. We picked one where the
          answer is all four at once.
        </p>
        <div className="deck-body">
          <div className="cols c4 stretch">
            <div className="card accent">
              <div className="label">01</div>
              <h3>Offline</h3>
              <p>Pull the cable mid-demo and it keeps working — diffusion included.</p>
            </div>
            <div className="card accent">
              <div className="label">02</div>
              <h3>Latency</h3>
              <p>A 37 ms try-on frame and a 120 ms shutter. A round trip would ruin both.</p>
            </div>
            <div className="card accent">
              <div className="label">03</div>
              <h3>Privacy</h3>
              <p>A camera on a shopper&apos;s face. No frame leaves, by construction.</p>
            </div>
            <div className="card accent">
              <div className="label">04</div>
              <h3>Cost</h3>
              <p>A shop window generating images all day. Cloud billing is absurd here.</p>
            </div>
          </div>
          <div className="callout">
            <b>The demo:</b>{" "}
            <span className="mute">
              stand in front of a camera, name any garment, and get a photo of yourself wearing it —
              generated on a board the size of a paperback, with the network unplugged.
            </span>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "setup",
    nav: "In the store",
    section: "The setup",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">02</span> The setup <span className="rule" />
        </div>
        <h2>What it looks like in the store</h2>
        <p className="deck-lede">
          A screen turned portrait, a camera on top of it, and one small board on the floor doing
          all of the work.
        </p>
        <div className="deck-body">
          <StoreSetup />
          <div className="callout">
            <b>Portrait because people are, and they stand back because the garment needs a
            half-body shot.</b>{" "}
            <span className="mute">
              The whole installation is a screen and something the size of a paperback.
            </span>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "npu",
    nav: "Eleven models, one NPU",
    section: "Hardware",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">03</span> Hardware <span className="rule" />
        </div>
        <h2>Eleven models on the Hexagon, every one whole-graph</h2>
        <p className="deck-lede">
          Not one model plus plumbing. Each CPU baseline is a <em>fair</em> one — XNNPACK, never a
          scalar fallback dressed up as a win.
        </p>
        <div className="deck-body">
          <div className="card flush" style={{ padding: "6px 20px 10px" }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Job</th>
                  <th className="num">NPU</th>
                  <th className="num">Board CPU</th>
                  <th className="num">Speedup</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="name">BiSeNet</td>
                  <td>Head / hair / neck matte</td>
                  <td className="num hi">12.3 ms</td>
                  <td className="num">978 ms</td>
                  <td className="num hi">80×</td>
                </tr>
                <tr>
                  <td className="name">SegFormer-B0</td>
                  <td>Body + arms, 18 classes</td>
                  <td className="num hi">24.9 ms</td>
                  <td className="num">1668 ms</td>
                  <td className="num hi">67×</td>
                </tr>
                <tr>
                  <td className="name">GHOST</td>
                  <td>Face swap, 184M params</td>
                  <td className="num hi">82 ms</td>
                  <td className="num">2064 ms</td>
                  <td className="num hi">25×</td>
                </tr>
                <tr>
                  <td className="name">Whisper-tiny</td>
                  <td>30 s of speech</td>
                  <td className="num hi">0.22 s</td>
                  <td className="num">~9 s</td>
                  <td className="num hi">~40×</td>
                </tr>
                <tr>
                  <td className="name">Gemma 4 E2B</td>
                  <td>Text, ~2B params</td>
                  <td className="num hi">188 tok/s</td>
                  <td className="num">15.3 (1B)</td>
                  <td className="num hi">12×</td>
                </tr>
                <tr className="lead">
                  <td className="name">Stable Diffusion 2.1</td>
                  <td>512² image</td>
                  <td className="num hi">0.32 s / step</td>
                  <td className="num">—</td>
                  <td className="num">—</td>
                </tr>
                <tr className="lead">
                  <td className="name">ControlNet-Canny</td>
                  <td>Conditioned on a real photo</td>
                  <td className="num hi">0.37 s / step</td>
                  <td className="num">—</td>
                  <td className="num">—</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="cols c3">
            <div className="card">
              <h3>Transformers lower too</h3>
              <p>SegFormer is not a convnet and still goes down as a single delegate, no shims.</p>
            </div>
            <div className="card">
              <h3>Four compiled on the board</h3>
              <p>They are not in the bundle — the AOT compiler will not run on macOS.</p>
            </div>
            <div className="card">
              <h3>1.3 GB co-resident</h3>
              <p>All three diffusion graphs fit on the cDSP at once. Measured.</p>
            </div>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "t-quant",
    nav: "Quantization is not free",
    section: "What we learned",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">04</span> What we learned · 01 <span className="rule" />
        </div>
        <h2>
          Our headline finding: <em>quantization bought 4%</em> and cost everything
        </h2>
        <p className="deck-lede">
          Received wisdom is that you quantize for the NPU. On the face-swap generator, 16a8w ran
          faster and was visibly broken.
        </p>
        <div className="deck-body">
          <div className="cols wide-left">
            <div className="card flush" style={{ padding: "6px 20px 12px" }}>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>GHOST generator</th>
                    <th className="num">Latency</th>
                    <th className="num">Error vs fp32</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="name">16a8w quantized</td>
                    <td className="num">79 ms</td>
                    <td className="num bad">47.5 / 255</td>
                    <td className="bad">Colour speckle</td>
                  </tr>
                  <tr className="lead">
                    <td className="name hi">float → HTP fp16</td>
                    <td className="num">82 ms</td>
                    <td className="num hi">1.78 / 255</td>
                    <td className="hi">Indistinguishable</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="card accent">
              <h3>
                The QCS8275 does HTP fp16 <span className="tag ok">surprise</span>
              </h3>
              <p>
                QNN&apos;s docs say only premium-tier SoCs do. This IoT part does.{" "}
                <code>quant_dtype=None</code> is a real option.
              </p>
            </div>
          </div>
          <div className="cols c2">
            <div className="card">
              <h3>It is not bit width — we proved that</h3>
              <p>
                Simulating the same quantization in PyTorch predicts 0.2/255, not 47.5. The culprit
                is 21 <code>InstanceNorm2d</code> layers&apos; statistics in fixed point.
              </p>
            </div>
            <div className="card">
              <h3>The rule we now apply</h3>
              <p>
                Suspect anything with runtime normalisation statistics — and measure per-class, not
                overall. 8-bit costs the <em>thin</em> classes: a silhouette holds at IoU 0.96, the
                arms drop to 0.48.
              </p>
            </div>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "t-systems",
    nav: "The NPU wasn't the bottleneck",
    section: "What we learned",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">05</span> What we learned · 02 <span className="rule" />
        </div>
        <h2>The biggest wins were not model changes</h2>
        <p className="deck-lede">
          With segmentation at 12 ms, everything expensive turned out to be the code around it.
        </p>
        <div className="deck-body">
          <div className="cols c3">
            <div className="card accent">
              <div className="stat">
                <div className="n">468 → 26 ms</div>
                <div className="cap">
                  Per inference. The runner was relaunching every time — keep it warm, and put
                  scratch on tmpfs. <strong>Both halves matter.</strong>
                </div>
              </div>
            </div>
            <div className="card">
              <div className="stat">
                <div className="n blue">70 → 11.5 ms</div>
                <div className="cap">
                  Detect faces on a <strong>downscaled</strong> frame. 2 px of drift, and it was 5×
                  the NPU work it fed.
                </div>
              </div>
            </div>
            <div className="card">
              <div className="stat">
                <div className="n blue">127 → 9.2 ms</div>
                <div className="cap">
                  Build the detector <strong>once</strong>. Constructing one costs ~100× running it.
                </div>
              </div>
            </div>
          </div>
          <div className="flow">
            <div className="flow-step">
              <div className="where cpu">CPU</div>
              <div className="what">Detect face</div>
              <div className="cost">11.5 ms</div>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step">
              <div className="where npu">Hexagon</div>
              <div className="what">Head matte</div>
              <div className="cost">12–16 ms</div>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step">
              <div className="where npu">Hexagon</div>
              <div className="what">Body clip</div>
              <div className="cost">+56 ms, optional</div>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step">
              <div className="where cpu">CPU</div>
              <div className="what">Composite</div>
              <div className="cost">8.4 ms</div>
            </div>
            <div className="flow-arrow">=</div>
            <div className="flow-step">
              <div className="where">Live</div>
              <div className="what">37 ms · 27 fps</div>
              <div className="cost">93 ms with the clip</div>
            </div>
          </div>
          <div className="callout warn">
            <b>And the debugging lesson:</b>{" "}
            <span className="mute">
              the expensive bugs here never throw. The diffusion UNet is v-prediction, not epsilon —
              run it the documented way and every image is RGB static, with no error anywhere. We
              found it by correlating the model&apos;s output against its own input.
            </span>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "three-paths",
    nav: "Three paths to a try-on",
    section: "The product",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">06</span> The product <span className="rule" />
        </div>
        <h2>Three ways to put a shopper in a garment</h2>
        <p className="deck-lede">
          We built them in this order. All three run on the board today, and each taught us what the
          next had to fix.
        </p>
        <div className="deck-body">
          <div className="cols c3">
            <div className="card">
              <div className="label">Path 01 · 12 ms</div>
              <h3>Cut the head out</h3>
              <p>
                Segment the shopper&apos;s head and drop it into a product photo, scaled by eye
                separation.
              </p>
              <p>
                <strong>Right:</strong> it is literally them, live.
              </p>
              <p className="bad">
                <strong>Wrong:</strong> reads as pasted. A live head on a studio body never agrees
                about lighting.
              </p>
            </div>
            <div className="card">
              <div className="label">Path 02 · 82 ms</div>
              <h3>Swap the face in</h3>
              <p>
                Go the other way — move their face into a shot that already has the garment, pose
                and lighting.
              </p>
              <p>
                <strong>Right:</strong> reads as one photograph, identity at 0.870.
              </p>
              <p className="bad">
                <strong>Wrong:</strong> they get the model&apos;s hair and no beard. Structural, not
                tunable.
              </p>
            </div>
            <div className="card accent">
              <div className="label">Path 03 · 21–26 s</div>
              <h3>Generate the garment</h3>
              <p>
                Diffusion generates <em>their</em> body wearing it, and their real head goes back
                on.
              </p>
              <p>
                <strong>Right:</strong> their face, their hair, their pose — any garment from a
                sentence.
              </p>
              <p>
                <strong>Cost:</strong> press-and-wait, not a live mirror.
              </p>
            </div>
          </div>
          <div className="callout">
            <b>The shape of the problem:</b>{" "}
            <span className="mute">
              paths 1 and 2 both need a studio shoot <em>per garment</em>. Only path 3 generates the
              thing the shopper actually came for.
            </span>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "diffusion",
    nav: "Diffusion on the board",
    section: "The product",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">07</span> The product <span className="rule" />
        </div>
        <h2>
          Stable Diffusion on the board: <em>fast enough, and unusable</em>
        </h2>
        <p className="deck-lede">
          0.32 s per denoising step — six times inside our viability line. Then speed turned out not
          to be the constraint at all.
        </p>
        <div className="deck-body">
          <div className="cols wide-left">
            <div style={{ display: "flex", flexDirection: "column", gap: 15 }}>
              <div className="cols c3" style={{ gap: 12 }}>
                <div className="card" style={{ padding: "13px 15px" }}>
                  <div className="stat">
                    <div className="n" style={{ fontSize: 30 }}>
                      159<small>ms</small>
                    </div>
                    <div className="cap">One UNet run</div>
                  </div>
                </div>
                <div className="card" style={{ padding: "13px 15px" }}>
                  <div className="stat">
                    <div className="n" style={{ fontSize: 30 }}>
                      6.4<small>s</small>
                    </div>
                    <div className="cap">20 steps, NPU compute</div>
                  </div>
                </div>
                <div className="card warn" style={{ padding: "13px 15px" }}>
                  <div className="stat">
                    <div className="n" style={{ fontSize: 30, color: "var(--d-err)" }}>
                      1 in 6
                    </div>
                    <div className="cap">Generations usable</div>
                  </div>
                </div>
              </div>
              <div className="card warn">
                <h3>
                  The blocker was reliability <span className="tag no">not speed</span>
                </h3>
                <p>
                  Six seeds of one garment prompt gave one clean t-shirt and four images that were
                  not a t-shirt at all — swatches, stripes, three yellow tins. Raising guidance does
                  not help: the same seeds fail the same way.
                </p>
              </div>
              <div className="card">
                <h3>Quality is free; steps are not the lever</h3>
                <p>
                  DPM++(2M), Karras spacing and a negative prompt cost{" "}
                  <strong>zero extra NPU time</strong> and beat doubling the step count.
                </p>
              </div>
            </div>
            <figure className="figure fill">
              <img
                className="shot"
                src="/deck/sd21-tshirt.png"
                alt="A folded white t-shirt generated on the board"
              />
              <figcaption>
                <b>Generated on the board.</b> 20 steps, nothing on the network.
              </figcaption>
            </figure>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "controlnet",
    nav: "Conditioning fixes it",
    section: "The product",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">08</span> The product <span className="rule" />
        </div>
        <h2>
          ControlNet-Canny: <em>1-in-6 becomes 6-of-6</em>
        </h2>
        <p className="deck-lede">
          The 1-in-6 problem is a property of sampling from pure noise. Conditioning on the
          shopper&apos;s own photo is simply not that.
        </p>
        <div className="deck-body">
          <div className="cols wide-left">
            <div className="card flush" style={{ padding: "6px 20px 12px" }}>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Six seeds, same prompt</th>
                    <th className="num">Text-to-image</th>
                    <th className="num">ControlNet</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="lead">
                    <td className="name">Coherent subject</td>
                    <td className="num bad">1–2 of 6</td>
                    <td className="num hi">6 of 6</td>
                  </tr>
                  <tr>
                    <td className="name">Wrong subject entirely</td>
                    <td className="num bad">4 of 6</td>
                    <td className="num hi">0 of 6</td>
                  </tr>
                  <tr>
                    <td className="name">Pose and framing</td>
                    <td className="num bad">random</td>
                    <td className="num hi">follows the source</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="card accent">
              <h3>Zero compositional failures</h3>
              <p>
                <strong>
                  The difference between &ldquo;generate four and pick&rdquo; and something a
                  stranger can press once.
                </strong>
              </p>
            </div>
          </div>
          <div className="cols c3">
            <div className="card">
              <h3>No VAE encoder needed</h3>
              <p>The edge map conditions in pixel space, so it goes straight in.</p>
            </div>
            <div className="card">
              <h3>0.37 s / step</h3>
              <p>Four graph runs instead of two. 20 steps is 7.4 s of NPU time.</p>
            </div>
            <div className="card warn">
              <h3>It carries shape, not identity</h3>
              <p>
                Edges describe outlines, not faces — the person who comes out is not the shopper.
              </p>
            </div>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "winner",
    nav: "The winner",
    section: "The answer",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">09</span> The answer <span className="rule" />
        </div>
        <h2>
          The winning combination: <em>generated garment, real face</em>
        </h2>
        <p className="deck-lede">
          ControlNet gets the clothes right and the face wrong — so we do not generate the face at
          all. The real head is cut out in 11.9 ms and put back.
        </p>
        <div className="deck-body">
          <figure className="figure">
            <img
              className="strip capped"
              src="/deck/cn15-pipeline.png"
              alt="Source frame, Canny edge map, ControlNet output, and the real head restored"
            />
            <figcaption>
              <b>Source → Canny → ControlNet → head restored.</b> All four stages on the board.
            </figcaption>
          </figure>
          <div className="cols c3">
            <div className="card accent">
              <h3>A paste, not a warp</h3>
              <p>
                The generation was conditioned on <em>this photo</em>, so the head is already in the
                right place at the right scale.
              </p>
            </div>
            <div className="card accent">
              <h3>Identity by construction</h3>
              <p>
                The face is not improved, it is <strong>theirs</strong> — beard and hair included. No
                similarity score required.
              </p>
            </div>
            <div className="card">
              <h3>Mask the edges first</h3>
              <p>
                Canny takes <em>every</em> edge, so unmasked it generates the room rather than the
                shopper.
              </p>
            </div>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "compare",
    nav: "Why it wins",
    section: "The answer",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">10</span> The answer <span className="rule" />
        </div>
        <h2>Why the diffusion path is the best bet</h2>
        <p className="deck-lede">
          Only one path gets the garment, the body <em>and</em> the identity right at the same time.
        </p>
        <div className="deck-body">
          <div className="card flush" style={{ padding: "6px 20px 10px" }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>&nbsp;</th>
                  <th>1 · Head cutout</th>
                  <th>2 · Face swap</th>
                  <th>3 · ControlNet + real head</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="name">Identity</td>
                  <td className="hi">Exact</td>
                  <td>Close — 0.870</td>
                  <td className="hi">Exact</td>
                </tr>
                <tr>
                  <td className="name">Hair and beard</td>
                  <td className="hi">Kept</td>
                  <td className="bad">Lost</td>
                  <td className="hi">Kept</td>
                </tr>
                <tr>
                  <td className="name">Reads as one photo</td>
                  <td className="bad">No</td>
                  <td className="hi">Yes</td>
                  <td className="hi">Yes</td>
                </tr>
                <tr>
                  <td className="name">Pose and body</td>
                  <td className="bad">The model&apos;s</td>
                  <td className="bad">The model&apos;s</td>
                  <td className="hi">Theirs</td>
                </tr>
                <tr className="lead">
                  <td className="name">Garment range</td>
                  <td className="bad">One per studio shoot</td>
                  <td className="bad">One per studio shoot</td>
                  <td className="hi">Any, from a sentence</td>
                </tr>
                <tr>
                  <td className="name">Latency per shot</td>
                  <td className="hi">live, 37 ms/frame</td>
                  <td className="hi">~120 ms</td>
                  <td>21–26 s</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="cols c2">
            <div className="card accent">
              <h3>The product argument</h3>
              <p>
                A shop window that shows any garment anyone names, on a $299 board with the network
                unplugged, is a <strong>different product</strong> — not a faster version of the
                same one.
              </p>
            </div>
            <div className="card warn">
              <h3>The honest cost</h3>
              <p>
                21–26 s is press-and-wait. But only 6.7 s of that is compute — the rest is the UNet
                reloading per step. <strong>The fix is software, not silicon.</strong>
              </p>
            </div>
          </div>
        </div>
      </div>
    ),
  },


  {
    id: "cost",
    nav: "What it costs",
    section: "Cost",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">11</span> Cost <span className="rule" />
        </div>
        <h2>
          The board is <em>$299, once</em>
        </h2>
        <p className="deck-lede">
          Cost is one of the four reasons to be on-device, so it is worth doing the arithmetic
          rather than asserting it.
        </p>
        <div className="deck-body">
          <div className="cols c3">
            <div className="card accent">
              <div className="stat">
                <div className="n">$299</div>
                <div className="cap">
                  The VENTUNO Q, one-off. Camera, eleven models and the web page all run on it.
                </div>
              </div>
            </div>
            <div className="card">
              <div className="stat">
                <div className="n blue">$0.01–0.05</div>
                <div className="cap">
                  Per image through a hosted API, for a Stable&nbsp;Diffusion / ControlNet-class
                  model at list price.
                </div>
              </div>
            </div>
            <div className="card warn">
              <div className="stat">
                <div className="n" style={{ color: "var(--d-warn)" }}>
                  ~1,000,000
                </div>
                <div className="cap">
                  Segmentation inferences a day in live mode — 27 fps over a ten-hour day.
                </div>
              </div>
            </div>
          </div>
          <div className="cols c3">
            <div className="card">
              <h3>The generated shot pays it off</h3>
              <p>
                At 100 shoppers a day and $0.03 an image, the API bill passes $299 in about three
                months — and keeps going. The board does not.
              </p>
            </div>
            <div className="card">
              <h3>The live mode has no tariff</h3>
              <p>
                A million inferences a day is not a thing you buy per call. On the board it is
                electricity, and it is already paid for.
              </p>
            </div>
            <div className="card">
              <h3>And the frames must travel</h3>
              <p>
                Streaming the camera instead is roughly <strong>100 GB a day, per booth</strong>,
                up a shop&apos;s uplink — before anyone has generated anything.
              </p>
            </div>
          </div>
          <div className="callout">
            <b>Honest caveat:</b>{" "}
            <span className="mute">
              a self-hosted GPU changes the arithmetic — you trade per-image billing for an instance
              that bills whether or not anyone walks past. The board bills for neither. Hosted API
              list prices as of October 2026; board price from the Arduino store.
            </span>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "more-time",
    nav: "With more time",
    section: "Next",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">12</span> Next <span className="rule" />
        </div>
        <h2>What we would do with more time</h2>
        <p className="deck-lede">
          In priority order. The first one alone takes the shutter from 21 s to about 8.
        </p>
        <div className="deck-body">
          <div className="cols c3 stretch">
            <div className="card accent">
              <div className="label">01 · biggest win</div>
              <h3>Rebuild the runner</h3>
              <p>
                The prebuilt binary is older than its own source and cannot stay loaded, so the
                841 MB UNet reloads <em>every step</em>. Six seconds of compute inside 21 of wall
                clock.
              </p>
            </div>
            <div className="card">
              <div className="label">02</div>
              <h3>Shared memory, not files</h3>
              <p>
                ControlNet spends ~0.65 s a step requantizing residuals and writing 27 MB of raw
                files. One fused pass removes it.
              </p>
            </div>
            <div className="card">
              <div className="label">03</div>
              <h3>A cleaner edge map</h3>
              <p>
                Canny picks up collar bands and sleeve hems and ControlNet reads them as colour
                boundaries — ask for a red tee over a navy one and you get a <em>ringer</em> tee.
              </p>
            </div>
            <div className="card">
              <div className="label">04</div>
              <h3>Prove 8 conditioned steps</h3>
              <p>
                One conditioned 8-step run came out clean, which would halve the cost. That is a
                single observation, not a measurement.
              </p>
            </div>
            <div className="card">
              <div className="label">05</div>
              <h3>Re-test precision everywhere</h3>
              <p>
                Quantization bought <strong>4%</strong> on the one model we measured properly. Every
                other model here deserves the same fp16 check.
              </p>
            </div>
            <div className="card">
              <div className="label">06</div>
              <h3>Use the second brain</h3>
              <p>
                The board is dual-brain — an STM32 running Zephyr beside Linux. We never touched it,
                and real-time actuation is the IoT track&apos;s whole point.
              </p>
            </div>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "scorecard",
    nav: "Scorecard",
    section: "Scoring",
    el: (
      <div className="deck-slide">
        <div className="deck-eyebrow">
          <span className="num">13</span> Scoring <span className="rule" />
        </div>
        <h2>Against the criteria, with the evidence</h2>
        <div className="deck-body">
          <div className="card flush" style={{ padding: "6px 20px 10px", marginTop: 10 }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Criterion</th>
                  <th>What we show on stage</th>
                  <th className="num">The number</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="name">Offline</td>
                  <td>Unplug the network first, not last. This deck is served from the board too.</td>
                  <td className="num hi">0 network calls</td>
                </tr>
                <tr>
                  <td className="name">Latency</td>
                  <td>A try-on that tracks you, and a shutter faster than a round trip.</td>
                  <td className="num hi">37 ms · 120 ms</td>
                </tr>
                <tr>
                  <td className="name">Privacy</td>
                  <td>A camera on a face, where frames cannot leave by construction, not by policy.</td>
                  <td className="num hi">0 frames off-device</td>
                </tr>
                <tr>
                  <td className="name">Cost</td>
                  <td>A storefront generating images all day for the price of electricity.</td>
                  <td className="num hi">6.7 s NPU / image</td>
                </tr>
                <tr className="lead">
                  <td className="name">Technical depth</td>
                  <td>Eleven models whole-graph on the Hexagon, four compiled on the device.</td>
                  <td className="num hi">25–80×</td>
                </tr>
                <tr>
                  <td className="name">Rigour</td>
                  <td>Fair CPU baselines, per-class IoU, a result that contradicts the default advice.</td>
                  <td className="num hi">fp16 beats 16a8w</td>
                </tr>
                <tr>
                  <td className="name">Fallbacks</td>
                  <td>A CPU path for every NPU model, and an LLM baseline that predates the NPU work.</td>
                  <td className="num hi">Demo-god insurance</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    ),
  },

  {
    id: "end",
    nav: "Thank you",
    el: (
      <div className="deck-slide deck-title">
        <div className="kicker">Thank you</div>
        <h1>
          Eleven models,
          <br />
          one NPU,
          <br />
          <span>no network.</span>
        </h1>
        <p className="tagline">
          The best bet generates the garment and keeps the real face — because conditioning on the
          shopper&apos;s own photo makes the composite a paste rather than a warp, and takes
          reliability from 1-in-6 to 6-of-6.
        </p>
        <div className="meta">
          <div>
            <b>make web</b>
            Camera, NPU, UI, this deck
          </div>
          <div>
            <b>make board-cn-tryon</b>
            The winning path, one command
          </div>
          <div>
            <b>docs/</b>
            Every measurement, with its method
          </div>
        </div>
      </div>
    ),
  },
];

export default slides;
