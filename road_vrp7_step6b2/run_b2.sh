#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
SOURCE="${SOURCE_DIR:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}"
BASE="runs/chicago_july_v1/step6b2"
FEATURES="${FEATURES_DIR:-$BASE/features_boundary_v1}"
STUDY="${STUDY_NAME:-b2_seed42_v1}"
case "$MODE" in
 test) "$PY" -m unittest discover -s road_vrp7_step6b2/tests -v ;;
 pilot) "$PY" road_vrp7_step6b2/extract.py --source-dir "$SOURCE" \
    --output-dir "$BASE/${STUDY}" --max-states "${MAX_STATES:-5}" ;;
 extract) "$PY" road_vrp7_step6b2/extract.py --source-dir "$SOURCE" \
    --output-dir "$FEATURES" ${EXTRA_ARGS:-} ;;
 train) "$PY" road_vrp7_step6b2/train_b2.py --source-dir "$SOURCE" \
    --features-dir "$FEATURES" --output-dir "$BASE/${STUDY}" \
    --seed "${SEED:-42}" --trees "${TREES:-120}" \
    --protocol "${PROTOCOL:-both}" --methods "${METHODS:-P2,P4}" \
    --modes "${MODES:-F0,F1,F2,F3,F4,F5}" ;;
 *) echo 'Usage: bash road_vrp7_step6b2/run_b2.sh test|pilot|extract|train' >&2;exit 2 ;;
esac
