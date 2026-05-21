#!/usr/bin/env bash
# LoRA fine-tuning with style-emotion disentanglement
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
# Avoid PyTorch multi-GPU init failure on 8-GPU nodes (device=7 assert).
# Override to use other GPUs, e.g. CUDA_VISIBLE_DEVICES=1 bash scripts/train_lora.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Use the activated conda env's Python (PATH may still point at /opt/anaconda3).
PYTHON="${CONDA_PREFIX:+$CONDA_PREFIX/bin/}python"
if ! "$PYTHON" -c "import accelerate" 2>/dev/null; then
  echo "Error: accelerate not found. Activate affectiveArt1 first:" >&2
  echo "  conda activate affectiveArt1" >&2
  echo "Current python: $(command -v "$PYTHON" || echo "$PYTHON")" >&2
  exit 1
fi

ABLATION="${ABLATION:-}"
EXTRA_ARGS=()
if [ -n "$ABLATION" ]; then
  EXTRA_ARGS+=(--ablation "$ABLATION")
fi

"$PYTHON" src/train_lora.py \
  --config configs/train_lora.yaml \
  "${EXTRA_ARGS[@]}" \
  "$@"

# Ablation examples:
# ABLATION=baseline_lora_only bash scripts/train_lora.sh
# ABLATION=full_model bash scripts/train_lora.sh
