#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
STUDY_NAME="${STUDY_NAME:-step6b0_static_traffic_v1}"
SOURCE_DIR="${SOURCE_DIR:-runs/chicago_july_v1/step6a/section_utility_1000_v1}"
OUTPUT_DIR="runs/chicago_july_v1/step6b0/$STUDY_NAME"
"$PY" road_vrp7_step6b0/train_baselines.py \
    --source-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR" \
    --protocol "${PROTOCOL:-both}" --seed "${SEED:-42}"
