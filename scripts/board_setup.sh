#!/usr/bin/env bash
# Set up the ExecuTorch runtime ON an aarch64 Linux board (Arduino VENTUNO Q).
#
# Runs on the board, not the Mac. Push and execute it with:
#   make board-setup
#
# Ubuntu 24.04 marks its Python externally-managed (PEP 668) and ships no pip,
# so this uses uv — a static binary that needs neither sudo nor ensurepip.
set -euo pipefail

VENV="${VENV:-$HOME/et-venv}"
export PATH="$HOME/.local/bin:$PATH"

echo "==> Board: $(uname -m), $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "==> Cores: $(nproc), RAM: $(free -h | awk '/^Mem:/{print $2}')"

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "==> uv $(uv --version)"

echo "==> Creating venv at ${VENV}"
uv venv --python 3.12 "$VENV"

# Same pins as the Mac side. optimum-executorch requires transformers==5.0.0rc1,
# hence --prerelease=allow. ExecuTorch publishes manylinux_2_28_aarch64 wheels,
# so nothing is built from source here.
#
# Heads up: torch's linux-aarch64 wheel on PyPI is the CUDA (sbsa) build, so
# this drags in several GB of nvidia-* packages that this board can never use.
# It works, it's just fat — budget ~8GB and 15 minutes on a first run. If disk
# or time is tight, pre-installing the CPU build trims most of that:
#
#   uv pip install --python "$VENV/bin/python" torch==2.14.1 \
#     --index-url https://download.pytorch.org/whl/cpu
#
# then re-run this script; the executorch resolve will reuse it.
echo "==> Installing ExecuTorch (aarch64 wheels, several GB — be patient)"
uv pip install --python "$VENV/bin/python" --prerelease=allow \
  executorch==1.5.1 \
  optimum-executorch==1.1.0 \
  huggingface_hub \
  sentencepiece \
  tiktoken \
  blobfile

echo ""
echo "==> Installed:"
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
for p in ("executorch", "torch", "transformers", "optimum-executorch"):
    try:
        print(f"    {p:<20} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"    {p:<20} MISSING")
PY

echo ""
echo "==> Done. Run a model with:  make board-run MODEL=<preset>"
