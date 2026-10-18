import { execFile, spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export const BOARD = "/home/arduino";
export const ET = `${BOARD}/git/executorch`;
export const PROBE = `${BOARD}/qairt-probe`;
export const CAM_DIR = "/dev/shm/cam";
export const MODEL_DIR = `${BOARD}/models/internvl3-1b`;
export const VENV_PY = `${BOARD}/et-venv/bin/python`;
export const PREPROCESS = `${BOARD}/make_vision_raw.py`;
export const RUNNER = `${ET}/build-x86/examples/qualcomm/oss_scripts/llama/qnn_multimodal_runner`;

/**
 * The QNN runtime resolves out of unpacked .debs rather than an installed SDK,
 * so every invocation has to carry these. ADSP_LIBRARY_PATH is where the
 * Hexagon skel libraries live — without it the DSP has nothing to load.
 */
export const NPU_ENV = {
  ...process.env,
  LD_LIBRARY_PATH: [
    `${ET}/build-x86/backends/qualcomm`,
    `${ET}/build-x86/lib`,
    `${PROBE}/extracted/usr/lib`,
    `${PROBE}/extracted/usr/lib/aarch64-linux-gnu`,
  ].join(":"),
  ADSP_LIBRARY_PATH: `${PROBE}/extracted/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp`,
};

/** Newest JPEG the capture daemon has written, or null if it isn't running. */
export async function latestFrame() {
  let names;
  try {
    names = await fs.readdir(CAM_DIR);
  } catch {
    return null;
  }
  const jpegs = names.filter((n) => n.endsWith(".jpg"));
  if (jpegs.length === 0) return null;

  // Sort by mtime, not by name. multifilesink's counter is zero-padded to
  // five digits but does not stop there, so once it passes 99999 the names
  // are variable width and a lexicographic sort puts "f451354" after
  // "f15177" — older, but later in the sort. `max-files` also only reaps
  // files from the run that wrote them, so a gst restart orphans the old
  // ones and they win that sort forever: the preview freezes on a frame
  // from a previous session and no amount of restarting the server fixes it.
  const stats = await Promise.all(
    jpegs.map(async (n) => {
      const full = path.join(CAM_DIR, n);
      try {
        return { full, mtime: (await fs.stat(full)).mtimeMs };
      } catch {
        return null; // reaped between readdir and stat; normal at 30fps
      }
    })
  );
  const live = stats.filter(Boolean).sort((a, b) => a.mtime - b.mtime);
  if (live.length === 0) return null;
  // Skip the very newest: it may still be partially written.
  return live.length > 1 ? live[live.length - 2].full : live[0].full;
}

/** Is the capture daemon alive? */
export async function cameraRunning() {
  try {
    const { stdout } = await execFileAsync("pgrep", ["-x", "gst-launch-1.0"]);
    return stdout.trim().length > 0;
  } catch {
    return false;
  }
}

/**
 * Start the capture daemon if it isn't already running.
 *
 * One long-lived process owns /dev/video0 and writes a bounded rolling window
 * of frames. Everything else reads those files, which avoids two consumers
 * fighting over the camera — grabbing a separate frame mid-stream fails with
 * "device busy".
 */
export async function ensureCamera() {
  if (await cameraRunning()) return true;
  await fs.mkdir(CAM_DIR, { recursive: true });
  // Clear frames from any previous run before starting a new one. multifilesink
  // restarts its counter at zero and only reaps what it wrote itself, so
  // whatever is already here would otherwise sit in the directory forever.
  try {
    const stale = await fs.readdir(CAM_DIR);
    await Promise.all(
      stale
        .filter((n) => n.endsWith(".jpg"))
        .map((n) => fs.unlink(path.join(CAM_DIR, n)).catch(() => {}))
    );
  } catch {
    // Nothing to clear.
  }
  const child = spawn(
    "gst-launch-1.0",
    [
      "-q",
      "v4l2src", "device=/dev/video0", "!",
      "videoconvert", "!",
      "jpegenc", "quality=70", "!",
      "multifilesink", `location=${CAM_DIR}/f%05d.jpg`, "max-files=4",
    ],
    { detached: true, stdio: "ignore" },
  );
  child.unref();
  // Give it a moment to produce the first frame.
  await new Promise((r) => setTimeout(r, 2500));
  return cameraRunning();
}

/** Pull the timing block the runner prints to its log. */
export function parseStats(log) {
  const num = (re) => {
    const m = log.match(re);
    return m ? Number(m[1]) : null;
  };
  return {
    modelLoadSec: num(/Model Load Time:\s+([\d.]+)/),
    promptEvalRate: num(/Prompt evaluation:\s+[\d.]+ \(seconds\)\s+Rate:\s+([\d.]+)/),
    generateRate: num(/Generated \d+ tokens:\s+[\d.]+ \(seconds\)\s+Rate:\s+([\d.]+)/),
    timeToFirstTokenSec: num(/Time to first generated token:\s+([\d.]+)/),
    generatedTokens: num(/Generated (\d+) tokens:/),
  };
}

/** Strip the chat scaffolding so the UI shows only the model's answer. */
export function cleanResponse(raw) {
  if (!raw) return "";
  let text = raw;
  const marker = "assistant";
  const idx = text.lastIndexOf(marker);
  if (idx !== -1) text = text.slice(idx + marker.length);
  return text
    .replace(/<\|im_start\|>/g, "")
    .replace(/<\|im_end\|>/g, "")
    .replace(/<img>|<\/img>/g, "")
    .replace(/<image>/g, "")
    .trim()
    // The runner supports batched prompts and prefixes each answer with its
    // index, which is noise when there is only ever one.
    .replace(/^\[\d+\]\s*/, "");
}
