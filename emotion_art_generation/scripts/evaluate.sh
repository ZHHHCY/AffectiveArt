#!/usr/bin/env bash
# Run FID, LPIPS, and AAS evaluation
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CHECKPOINT="${CHECKPOINT:-outputs/lora/checkpoint-final}"

python src/evaluation/evaluate.py \
  --config configs/eval.yaml \
  --checkpoint "$CHECKPOINT" \
  "$@"
