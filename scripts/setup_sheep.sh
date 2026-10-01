#!/bin/bash
# Set up DreamZero training environment on sheep.
# Creates ~/dreamzero-env venv with all training dependencies.
# Run once; safe to re-run (skips existing installs).
#
# Usage:
#   bash ~/final_project/code/dreamzero/scripts/setup_sheep.sh

set -euo pipefail

DREAMZERO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HOME}/dreamzero-env"

echo "=== DreamZero sheep setup ==="
echo "  repo:  $DREAMZERO_ROOT"
echo "  venv:  $VENV_DIR"

# ---- Create venv if it doesn't exist ----
if [ ! -d "$VENV_DIR" ]; then
    echo "--- Creating venv at $VENV_DIR ---"
    python3 -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

echo "--- Upgrading pip/wheel ---"
"$PIP" install --upgrade pip wheel setuptools -q

# ---- PyTorch (CUDA 12.4) ----
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "--- Installing PyTorch (CUDA 12.4) ---"
    "$PIP" install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
fi

# ---- Core training packages ----
echo "--- Installing training packages ---"
"$PIP" install \
    "transformers==4.51.3" \
    "deepspeed" \
    "accelerate" \
    "peft==0.5.0" \
    "hydra-core" \
    "omegaconf" \
    "tianshou==0.5.1" \
    "einops==0.8.1" \
    "diffusers==0.30.2" \
    "wandb" \
    "huggingface_hub" \
    "numpy==1.26.4" \
    "scipy==1.15.3" \
    "pandas" \
    "pyarrow" \
    "datasets==3.6.0" \
    "av==15.0.0" \
    "msgpack" \
    "msgpack-numpy" \
    "openpi-client==0.1.1" \
    "websockets" \
    "tyro" \
    "imageio==2.34.2" \
    "imageio-ffmpeg" \
    "opencv-python-headless" \
    "albumentations==1.4.18" \
    "ftfy" \
    "tiktoken" \
    "sentencepiece" \
    "lark" \
    "loguru" \
    "tqdm" \
    "psutil" \
    "requests" \
    "timm" \
    "click" \
    "termcolor" \
    "dm_tree" \
    "redis" \
    "flask" \
    "python-socketio>=5.13.0" \
    "flask_socketio"

# ---- decord (video backend) ----
if ! "$PYTHON" -c "import decord" 2>/dev/null; then
    echo "--- Installing decord ---"
    "$PIP" install decord2 || "$PIP" install decord || echo "WARNING: decord install failed; try manually"
fi

# ---- flash-attn (optional, speeds up attention) ----
if ! "$PYTHON" -c "import flash_attn" 2>/dev/null; then
    echo "--- Installing flash-attn (may take a few minutes) ---"
    "$PIP" install flash-attn --no-build-isolation || echo "WARNING: flash-attn install failed; training will use fallback attention"
fi

# ---- Install dreamzero itself (no deps, already installed above) ----
echo "--- Installing dreamzero package ---"
"$PIP" install -e "$DREAMZERO_ROOT" --no-deps

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Or use: $PYTHON"
echo ""
echo "Test: $PYTHON -c 'import torch; import groot; print(\"OK\", torch.cuda.is_available())'"
