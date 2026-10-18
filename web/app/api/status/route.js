import { promises as fs } from "node:fs";
import { cameraRunning, latestFrame, MODEL_DIR, RUNNER } from "../../../lib/board.js";

export const dynamic = "force-dynamic";

const exists = async (p) => {
  try { await fs.access(p); return true; } catch { return false; }
};

export async function GET() {
  const [camera, frame, runner, model] = await Promise.all([
    cameraRunning(),
    latestFrame(),
    exists(RUNNER),
    exists(MODEL_DIR),
  ]);
  return Response.json({
    camera,
    hasFrame: Boolean(frame),
    runner,
    model,
  });
}
