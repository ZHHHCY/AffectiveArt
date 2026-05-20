#!/usr/bin/env bash
# LoRA fine-tuning with style-emotion disentanglement
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ABLATION="${ABLATION:-}"
EXTRA_ARGS=()
if [ -n "$ABLATION" ]; then
  EXTRA_ARGS+=(--ablation "$ABLATION")
fi

python src/train_lora.py \
  --config configs/train_lora.yaml \
  "${EXTRA_ARGS[@]}" \
  "$@"

# Ablation examples:
# ABLATION=baseline_lora_only bash scripts/train_lora.sh
# ABLATION=full_model bash scripts/train_lora.sh
