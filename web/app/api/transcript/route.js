import { execFile, spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export const dynamic = "force-dynamic";

const RUN_DIR = "/dev/shm/mic";
const TRANSCRIPT = `${RUN_DIR}/transcript.jsonl`;
const CONTROL = `${RUN_DIR}/control.json`;
const DAEMON = "/home/arduino/mic_daemon.py";
const PYTHON = "/home/arduino/et-venv/bin/python";

async function daemonRunning() {
  try {
    const { stdout } = await execFileAsync("pgrep", ["-f", "mic_daemon.py"]);
    return stdout.trim().length > 0;
  } catch {
    return false;
  }
}

async function readControl() {
  try {
    return JSON.parse(await fs.readFile(CONTROL, "utf8"));
  } catch {
    return { translate: false, targetLang: "Spanish" };
  }
}

/** Recent transcript lines, newest last. */
export async function GET(request) {
  const limit = Number(new URL(request.url).searchParams.get("limit")) || 40;
  let entries = [];
  try {
    const raw = await fs.readFile(TRANSCRIPT, "utf8");
    entries = raw
      .split("\n")
      .filter(Boolean)
      .slice(-limit)
      .map((line) => {
        try { return JSON.parse(line); } catch { return null; }
      })
      .filter(Boolean);
  } catch {
    // No transcript yet is normal before the daemon has heard anything.
  }
  const [running, control] = await Promise.all([daemonRunning(), readControl()]);
  return Response.json({ running, control, entries });
}

/**
 * Control the listener: { action: "start" | "stop" | "clear" | "config",
 *                         translate, targetLang }
 */
export async function POST(request) {
  const body = await request.json().catch(() => ({}));
  const action = body.action || "config";

  await fs.mkdir(RUN_DIR, { recursive: true });

  if (action === "config" || body.translate !== undefined || body.targetLang) {
    const current = await readControl();
    const next = {
      translate: body.translate ?? current.translate,
      targetLang: body.targetLang ?? current.targetLang,
    };
    // The daemon re-reads this file every loop, so changes take effect on the
    // next chunk without a restart.
    await fs.writeFile(CONTROL, JSON.stringify(next));
    if (action === "config") {
      return Response.json({ ok: true, control: next, running: await daemonRunning() });
    }
  }

  if (action === "clear") {
    await fs.writeFile(TRANSCRIPT, "");
    return Response.json({ ok: true });
  }

  if (action === "stop") {
    try {
      await execFileAsync("pkill", ["-f", "mic_daemon.py"]);
    } catch {
      // pkill exits non-zero when nothing matched; that is fine here.
    }
    return Response.json({ ok: true, running: false });
  }

  if (action === "start") {
    if (await daemonRunning()) {
      return Response.json({ ok: true, running: true });
    }
    const log = await fs.open(`${RUN_DIR}/daemon.log`, "a");
    const child = spawn(PYTHON, [DAEMON], {
      detached: true,
      stdio: ["ignore", log.fd, log.fd],
      env: { ...process.env, HF_HUB_DISABLE_TELEMETRY: "1" },
    });
    child.unref();
    // The feature extractor takes a moment to load before the first chunk.
    await new Promise((r) => setTimeout(r, 2500));
    return Response.json({ ok: true, running: await daemonRunning() });
  }

  return Response.json({ error: `unknown action ${action}` }, { status: 400 });
}
