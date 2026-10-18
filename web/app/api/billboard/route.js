import { execFile, spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export const dynamic = "force-dynamic";

const OUT_DIR = "/dev/shm/billboard";
const LATEST = `${OUT_DIR}/latest.jpg`;
const STATE = `${OUT_DIR}/state.json`;
const CONTROL = `${OUT_DIR}/control.json`;
const DAEMON = "/home/arduino/bb/scripts/board_billboard_daemon.py";
// References for the compare slider. "base" is the product photo with the
// model's head removed — what the shopper's head is actually laid onto.
// "original" still has his head, which is the more useful of the two for
// judging alignment: it shows where a head that genuinely belongs on this
// body sits, at what size and angle.
const TEMPLATES = "/home/arduino/bb/assets/templates";
const REFERENCE = {
  base: `${TEMPLATES}/model-navy-tee.template.png`,
  original: `${TEMPLATES}/model-navy-tee.png`,
};
const PYTHON = "/home/arduino/et-venv/bin/python";

// Matching the bare filename here makes Start silently do nothing, because
// `start` short-circuits on this. `pgrep -f` searches whole command lines, so
// a plain "board_billboard_daemon.py" also matches any *observer* holding that
// name in its argv — a `tail -f` on the log, an editor, a `watch pgrep`, or
// pgrep itself. One of those and `running()` returns true with no daemon
// alive, Start reports `{ok: true, running: true}`, spawns nothing, and the
// page sits on a frozen latest.jpg.
//
// Two defences, both needed. The leading `[e]` stops the pattern matching
// pgrep's own argv (the regex wants `e`, the literal argv has `[`), and
// requiring the interpreter path stops it matching an editor or a tail.
// Verified on the board against a decoy: this returns 1, the bare filename
// returns 3.
const DAEMON_PATTERN = "[e]t-venv/bin/python .*board_billboard_daemon";

async function running() {
  try {
    const { stdout } = await execFileAsync("pgrep", ["-f", DAEMON_PATTERN]);
    return stdout.trim().length > 0;
  } catch {
    return false;
  }
}

/**
 * ?image=1 returns the composited billboard JPEG; otherwise JSON state.
 * The image is served separately so the UI can refresh it with a plain <img>
 * rather than shuttling base64 through JSON every second.
 */
/**
 * Pad a reference image upward to the composite's canvas so the two line up
 * pixel for pixel. The padding amount is the template's own `headroom`, read
 * from its JSON rather than hardcoded, so regenerating the template with a
 * different value does not silently shift the overlay.
 */
async function padToComposite(file) {
  const meta = JSON.parse(
    await fs.readFile(`${TEMPLATES}/model-navy-tee.json`, "utf8")
  );
  const [w, h] = meta.size;
  const pad = meta.headroom ?? 0;
  const { stdout } = await execFileAsync(
    PYTHON,
    [
      "-c",
      `import sys,io
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB")
w, h, pad = ${w}, ${h}, ${pad}
if im.size != (w, h):
    out = Image.new("RGB", (w, h), (255, 255, 255))
    # Replicate the top row upward, the same way prep_template pads, so the
    # backdrop gradient continues instead of banding against a flat fill.
    if pad > 0:
        out.paste(im.crop((0, 0, im.width, 1)).resize((im.width, pad)), (0, 0))
    out.paste(im, (0, pad))
    im = out
buf = io.BytesIO(); im.save(buf, "PNG")
sys.stdout.buffer.write(buf.getvalue())`,
      file,
    ],
    { encoding: "buffer", maxBuffer: 64 * 1024 * 1024 }
  );
  return stdout;
}

export async function GET(request) {
  const which = new URL(request.url).searchParams.get("image");

  if (which) {
    // The original photo is 1122x1402; the composite is padded to 1122x1552
    // so there is headroom above the shopper's hair. Serving the original as
    // it is would misregister the two by exactly that padding and make the
    // comparison useless, so it is padded to match before going out.
    if (REFERENCE[which]) {
      try {
        const buf = await padToComposite(REFERENCE[which]);
        return new Response(buf, {
          headers: {
            "Content-Type": "image/png",
            "Cache-Control": "public, max-age=300",
          },
        });
      } catch {
        return new Response("no reference", { status: 404 });
      }
    }
    try {
      const buf = await fs.readFile(LATEST);
      return new Response(buf, {
        headers: {
          "Content-Type": "image/jpeg",
          "Cache-Control": "no-store, no-cache, must-revalidate",
        },
      });
    } catch {
      return new Response("no composite yet", { status: 503 });
    }
  }

  let state = null;
  try {
    state = JSON.parse(await fs.readFile(STATE, "utf8"));
  } catch {
    // Nothing written yet; the daemon may still be loading.
  }
  let control = { mode: "billboard", drop: 0.45, halfWidth: 1.15, offsetY: 0, debug: false, phase: "live", headScale: 1.12, fineX: 0, fineY: 36, neckBlend: true, neckFade: 1.0 };
  try { control = { ...control, ...JSON.parse(await fs.readFile(CONTROL, "utf8")) }; } catch {}
  return Response.json({ running: await running(), state, control });
}

export async function POST(request) {
  const body = await request.json().catch(() => ({}));
  const action = body.action || "start";

  if (action === "mode") {
    // The daemon re-reads this each loop, so switching needs no restart.
    await fs.mkdir(OUT_DIR, { recursive: true });
    let current = {};
    try { current = JSON.parse(await fs.readFile(CONTROL, "utf8")); } catch {}
    const next = {
      mode: ["tryon", "tryon-clip", "billboard", "segment", "faceswap", "generate"].includes(body.mode) ? body.mode : (current.mode || "billboard"),
      // ControlNet try-on: which catalogue garment to generate, and how many
      // denoising steps to spend on it.
      garment: body.garment ?? current.garment ?? "navy-tee",
      steps: body.steps ?? current.steps ?? 8,
      drop: body.drop ?? current.drop ?? 0.45,
      halfWidth: body.halfWidth ?? current.halfWidth ?? 1.15,
      offsetY: body.offsetY ?? current.offsetY ?? 0,
      debug: body.debug ?? current.debug ?? false,
      phase: body.phase ?? current.phase ?? "live",
      headScale: body.headScale ?? current.headScale ?? 1.12,
      fineX: body.fineX ?? current.fineX ?? 0,
      fineY: body.fineY ?? current.fineY ?? 36,
      neckBlend: body.neckBlend ?? current.neckBlend ?? true,
      neckFade: body.neckFade ?? current.neckFade ?? 1.0,
      capture: false,
    };
    await fs.writeFile(CONTROL, JSON.stringify(next));
    return Response.json({ ok: true, control: next });
  }

  if (action === "capture" || action === "retake") {
    await fs.mkdir(OUT_DIR, { recursive: true });
    let cur = {};
    try { cur = JSON.parse(await fs.readFile(CONTROL, "utf8")); } catch {}
    // The daemon reads these each loop; capture is one-shot and it clears the
    // flag itself once the composite lands.
    const next = action === "capture"
      ? { ...cur, phase: "live", capture: true }
      : { ...cur, phase: "live", capture: false };
    await fs.writeFile(CONTROL, JSON.stringify(next));
    return Response.json({ ok: true, control: next });
  }

  if (action === "stop") {
    try {
      // Same pattern as running(), and for a sharper reason: `pkill -f` on the
      // bare filename would kill any observer holding that name in its argv —
      // somebody's `tail -f` on the log, or their editor. Stop should end the
      // daemon, not their terminal.
      await execFileAsync("pkill", ["-f", DAEMON_PATTERN]);
    } catch {
      // Nothing matched; fine.
    }
    return Response.json({ ok: true, running: false });
  }

  if (action === "start") {
    if (await running()) return Response.json({ ok: true, running: true });
    await fs.mkdir(OUT_DIR, { recursive: true });
    const log = await fs.open(`${OUT_DIR}/daemon.log`, "a");
    const child = spawn(PYTHON, [DAEMON], {
      detached: true,
      stdio: ["ignore", log.fd, log.fd],
      cwd: "/home/arduino/bb",
    });
    child.unref();
    // Loading torch and the detector takes a few seconds before the first
    // composite appears.
    await new Promise((r) => setTimeout(r, 3500));
    return Response.json({ ok: true, running: await running() });
  }

  return Response.json({ error: `unknown action ${action}` }, { status: 400 });
}
