#!/usr/bin/env bash
# Train style and emotion classifiers
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python src/train_classifiers.py \
  --config configs/train_classifiers.yaml \
  --task "${TASK:-both}" \
  "$@"
