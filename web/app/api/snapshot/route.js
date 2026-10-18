import { promises as fs } from "node:fs";
import { ensureCamera, latestFrame } from "../../../lib/board.js";

export const dynamic = "force-dynamic";

// Serves the newest camera frame. The UI polls this to make a live preview;
// the camera runs at 5fps so polling is as good as streaming here, and far
// less fragile than holding an MJPEG connection open.
export async function GET() {
  await ensureCamera();
  const frame = await latestFrame();
  if (!frame) {
    return new Response("no frame yet", { status: 503 });
  }
  const buf = await fs.readFile(frame);
  return new Response(buf, {
    headers: {
      "Content-Type": "image/jpeg",
      "Cache-Control": "no-store, no-cache, must-revalidate",
    },
  });
}
