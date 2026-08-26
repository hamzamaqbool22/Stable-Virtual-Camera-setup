#!/usr/bin/env bash
# One-shot Vast.ai install for Stable Virtual Camera (SEVA) v1.1.
# Run from the project root on a Linux GPU box. Do not run on Mac.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SEVA_DIR="$ROOT/stable-virtual-camera"
MODELS_DIR="$ROOT/models"
ENV_NAME="seva"
PYTHON_VERSION="3.11"
TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
SEVA_REPO="https://github.com/Stability-AI/stable-virtual-camera.git"
HF_REPO="stabilityai/stable-virtual-camera"
WEIGHT_FILE="modelv1.1.safetensors"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [[ "$(uname -s)" == "Darwin" ]]; then
  die "This script is for Vast.ai (Linux + NVIDIA). Do not run it on macOS."
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. This host has no NVIDIA driver."
log "GPU"
nvidia-smi -L || true
nvidia-smi | head -n 20 || true

# --- conda ---
if ! command -v conda >/dev/null 2>&1; then
  die "conda not found. Install Miniforge/Miniconda, then re-run."
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

log "conda $(conda --version)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "conda env '$ENV_NAME' already exists"
else
  log "creating conda env '$ENV_NAME' (python=${PYTHON_VERSION})"
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip git ninja -y
fi
conda activate "$ENV_NAME"

PYTHON_BIN="$(command -v python)"
log "python: $PYTHON_BIN"
python --version
[[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "$PYTHON_VERSION" ]] \
  || die "Expected Python ${PYTHON_VERSION} inside the '$ENV_NAME' env"

# --- PyTorch (cu128, NOT default PyPI CUDA 13 wheels) ---
log "installing torch==${TORCH_VERSION}+cu128"
python -m pip install --upgrade pip
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "$TORCH_INDEX"

log "verifying CUDA"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False. Stop and fix the GPU/driver/CUDA wheel.")
print("gpu", torch.cuda.get_device_name(0))
print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY

# --- official SEVA repo ---
if [[ -d "$SEVA_DIR/.git" ]]; then
  log "SEVA repo already cloned at $SEVA_DIR"
else
  log "cloning SEVA"
  git clone --recursive "$SEVA_REPO" "$SEVA_DIR"
fi
cd "$SEVA_DIR"
git submodule update --init --recursive
log "SEVA $(git log -1 --oneline)"

log "pip install -e .  (do not use requirements.txt)"
python -m pip install -e .

# SEVA pins numpy==1.24.4; that can break with torch 2.11. Relax only if import fails.
if ! python -c "import numpy, torch, seva" >/dev/null 2>&1; then
  log "relaxing numpy pin for torch 2.11"
  python -m pip install "numpy>=1.26,<2"
fi

log "verifying Python imports"
python - <<'PY'
import seva
import diffusers
import huggingface_hub
import torch
print("seva", getattr(seva, "__file__", "ok"))
print("diffusers", diffusers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("torch", torch.__version__)
PY

# --- Hugging Face auth + v1.1 weights ---
python -m pip install -U "huggingface_hub[cli]" hf_transfer || python -m pip install -U "huggingface_hub[cli]"
mkdir -p "$MODELS_DIR"

if [[ -f "$MODELS_DIR/$WEIGHT_FILE" ]]; then
  log "found existing $MODELS_DIR/$WEIGHT_FILE (skipping download)"
else
  if [[ -z "${HF_TOKEN:-}" ]]; then
    log "Hugging Face login (gated model). Accept the license at:"
    echo "  https://huggingface.co/stabilityai/stable-virtual-camera"
    hf auth login
  fi
  hf auth whoami || die "Hugging Face auth failed. Run: hf auth login  (or export HF_TOKEN)"
  log "downloading $WEIGHT_FILE into models/"
  hf download "$HF_REPO" "$WEIGHT_FILE" --local-dir "$MODELS_DIR"
  hf download "$HF_REPO" config.yaml --local-dir "$MODELS_DIR" || true
fi

# Prefetch SD 2.1 VAE used by seva.modules.autoencoder (OpenCLIP still downloads on first run).
log "prefetching SD 2.1 VAE"
hf download stabilityai/stable-diffusion-2-1-base --include "vae/*" || true

cd "$ROOT"
mkdir -p "$ROOT/images" "$ROOT/outputs"

cat <<EOF

Setup complete.

Next:
  conda activate ${ENV_NAME}
  # put a car photo in images/
  python generate.py --image images/car.jpg --traj "dolly zoom-in" --profile preview

Weights: ${MODELS_DIR}/${WEIGHT_FILE}
SEVA:    ${SEVA_DIR}
EOF
