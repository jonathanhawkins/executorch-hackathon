#!/usr/bin/env bash
# One-shot environment setup. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_VERSION=3.12

echo "==> Checking for uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "    uv not found. Install it:"
  echo "      curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "==> Creating .venv on Python ${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}" .venv

# optimum-executorch pins transformers==5.0.0rc1, so pre-releases must be allowed
# or the resolver refuses the whole set.
echo "==> Installing ExecuTorch + optimum-executorch (this pulls torch, ~2GB)"
uv pip install --python .venv/bin/python --prerelease=allow \
  executorch==1.5.1 \
  optimum-executorch==1.1.0 \
  huggingface_hub \
  sentencepiece \
  tiktoken \
  blobfile

echo ""
echo "==> Done. Verify with:"
echo "      make doctor"
echo "      make smoke     # exports + runs a 135M model end to end"
echo ""
echo "    For gated models (Llama, Gemma) accept the license on the HF page, then:"
echo "      .venv/bin/hf auth login"
