#!/usr/bin/env bash
# R4-K — No new ALNS instrumentation, only audit/read-only code plus optional R4-J resume.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
SOURCE="${R4K_SOURCE_STUDY:-r4j_train_early_s62s63_v1}"
CF_DIR="${R4K_CF_DIR:-runs/chicago_july_v1/step6c1_r4h/${SOURCE}_cf_full}"
PREFIX_DIR="${R4K_PREFIX_DIR:-runs/chicago_july_v1/step6c1_r4g_prefix/${SOURCE}_prefix_full}"
CAPTURE_DIR="${R4K_CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4g/${SOURCE}_capture}"
OUT="${R4K_ROOT:-runs/chicago_july_v1/step6c1_r4k}"
NAME="${STUDY_NAME:-r4k_${MODE}_${SOURCE}_v1}"
case "$MODE" in
  test)
    "$PY" -m unittest discover -s road_vrp7_step6c1_r4k/tests -p 'test_r4k*.py' -v
    "$PY" -m py_compile road_vrp7_step6c1_r4k/*.py
    bash -n road_vrp7_step6c1_r4k/run_r4k.sh
    ;;
  partial|complete)
    args=(--cf-dir "$CF_DIR" --prefix-dir "$PREFIX_DIR" --capture-dir "$CAPTURE_DIR" --output-dir "$OUT/$NAME")
    [[ "$MODE" != complete ]] || args+=(--require-complete)
    "$PY" road_vrp7_step6c1_r4k/audit_r4k.py "${args[@]}"
    ;;
  resume)
    # Full R4-H study was initialized already, keep the EXACT same manifest/plan.
    [[ -f "$CF_DIR/manifest.json" ]] || { echo '[R4K] Existing full CF manifest missing; abort.' >&2;exit 2; }
    echo "[R4K] Resuming frozen R4-J CF full: $CF_DIR; max NEW units this run = ${R4K_MAX_NEW_UNITS:-240}" >&2
    R4J_SOURCE_STUDY="$SOURCE" R4J_CF_STUDY="$(basename "$CF_DIR")" \
      R4H_ROOT="$(dirname "$CF_DIR")" PREFIX_ROOT="$(dirname "$PREFIX_DIR")" \
      R4J_MAX_NEW_UNITS="${R4K_MAX_NEW_UNITS:-240}" RESUME=1 \
      bash road_vrp7_step6c1_r4j/run_r4j.sh cf-full
    ;;
  old-pilot)
    echo '[R4K] For old 8-snapshot pilot use source-level R4-H report directly; do not mix its 8 snapshots with partial full statistics.' >&2
    ;;
  *) echo 'Usage: run_r4k.sh test|partial|resume|complete' >&2;exit 2;;
esac
