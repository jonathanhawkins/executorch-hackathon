import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

import {
  BOARD,
  MODEL_DIR,
  NPU_ENV,
  PREPROCESS,
  RUNNER,
  VENV_PY,
  cleanResponse,
  ensureCamera,
  latestFrame,
  parseStats,
} from "../../../lib/board.js";

const execFileAsync = promisify(execFile);

export const dynamic = "force-dynamic";
export const maxDuration = 120;

// One inference at a time. The NPU has a single context set loaded; two
// concurrent runs fight over DSP memory and both fail.
let busy = false;

export async function POST(request) {
  if (busy) {
    return Response.json({ error: "Already analyzing a frame." }, { status: 429 });
  }

  const body = await request.json().catch(() => ({}));
  const prompt = (body.prompt || "What do you see?").slice(0, 400);
  // 256 of these tokens are the image, before any text — a text-sized value
  // aborts with "sequence length exceeded" before generating anything.
  const seqLen = Math.min(Math.max(Number(body.seqLen) || 480, 320), 1000);
  // 0.0 is greedy argmax, which sends this quantized 1B into endless
  // repetition loops. 1.0-1.2 terminates cleanly.
  const temperature = Math.min(Math.max(Number(body.temperature) || 1.1, 0.1), 2);

  busy = true;
  const started = Date.now();
  try {
    await ensureCamera();
    const frame = await latestFrame();
    if (!frame) {
      return Response.json({ error: "Camera produced no frame." }, { status: 503 });
    }

    const shot = path.join(MODEL_DIR, "shot.jpg");
    await fs.copyFile(frame, shot);

    // JPEG -> 448x448 ImageNet-normalized NCHW float32, which is what the
    // vision encoder consumes.
    await execFileAsync(VENV_PY, [PREPROCESS, shot, "-o", path.join(MODEL_DIR, "input.raw")]);

    // --image_path takes an input LIST file, not the tensor itself.
    await fs.writeFile(path.join(MODEL_DIR, "image_list.txt"), "input.raw\n");

    const outPath = path.join(BOARD, "web_vlm_out.txt");
    const args = [
      "--decoder_path", "InternVL3-1B_QCS8275_ctxt_1000.pte",
      "--encoder_path", "InternVL3-1B_QCS8275_ctxt_1000_vision_encoder_qnn.pte",
      "--tok_embedding_path", "InternVL3-1B_QCS8275_ctxt_1000_tok_embedding_qnn.pte",
      "--tokenizer_path", "tokenizer.json",
      "--image_path", "image_list.txt",
      "--decoder_model_version", "internvl3",
      "--prompt", prompt,
      "--seq_len", String(seqLen),
      // Hybrid mode (1) prefills ~8x faster but produces incoherent, looping
      // output for this model. KV-only is slower and correct.
      "--eval_mode", "0",
      "--temperature", String(temperature),
      "--output_path", outPath,
    ];

    const { stdout, stderr } = await execFileAsync(RUNNER, args, {
      cwd: MODEL_DIR,
      env: NPU_ENV,
      maxBuffer: 32 * 1024 * 1024,
      timeout: 110_000,
    }).catch((err) => ({ stdout: err.stdout || "", stderr: err.stderr || String(err) }));

    const log = `${stdout}\n${stderr}`;
    const raw = await fs.readFile(outPath, "utf8").catch(() => "");
    const text = cleanResponse(raw);

    if (!text) {
      const hint = log.includes("available PD")
        ? "The DSP ran out of protection-domain memory."
        : log.includes("sequence length exceeded")
          ? "seq_len too small for a 256-token image."
          : "The runner produced no output.";
      return Response.json({ error: hint, log: log.slice(-2000) }, { status: 500 });
    }

    const imageBase64 = (await fs.readFile(shot)).toString("base64");

    return Response.json({
      text,
      prompt,
      stats: { ...parseStats(log), wallClockSec: (Date.now() - started) / 1000 },
      image: `data:image/jpeg;base64,${imageBase64}`,
    });
  } catch (err) {
    return Response.json({ error: String(err?.message || err) }, { status: 500 });
  } finally {
    busy = false;
  }
}
