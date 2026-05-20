#!/usr/bin/env bash
# Generate a single artwork from content + style + emotion
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CHECKPOINT="${CHECKPOINT:-outputs/lora/checkpoint-final}"
CONTENT="${CONTENT:-a lonely boat on a quiet river at sunset}"
STYLE="${STYLE:-Expressionism}"
EMOTION="${EMOTION:-Calm}"
OUTPUT="${OUTPUT:-outputs/samples/calm_expressionism_boat.png}"

python src/inference.py \
  --checkpoint "$CHECKPOINT" \
  --content "$CONTENT" \
  --style "$STYLE" \
  --emotion "$EMOTION" \
  --output "$OUTPUT" \
  "$@"
