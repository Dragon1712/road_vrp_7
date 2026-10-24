#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
E_ROOT="${R4E_ROOT:-runs/chicago_july_v1/step6c1_r4e}"
F_ROOT="${R4F_ROOT:-runs/chicago_july_v1/step6c1_r4f}"
C_DIR="${R4C_DIR:-runs/chicago_july_v1/step6c1_r4c/r4c_pilot_16snaps_v1}"
PREVIEW="${PREVIEW_STUDY:-r4e_long_cf_8snaps_v1}"
FULL="${FULL_STUDY:-r4f_long_cf_16snaps_v1}"
NAME="${STUDY_NAME:-}"
case "$MODE" in
 test)
   "$PY" -m unittest discover -s road_vrp7_step6c1_r4f/tests -p 'test_r4f*.py' -v
   "$PY" -m py_compile road_vrp7_step6c1_r4f/*.py
   bash -n road_vrp7_step6c1_r4f/run_r4f.sh
   ;;
 audit-preview)
   NAME="${NAME:-r4f_preview_8snaps_v1}"
   "$PY" road_vrp7_step6c1_r4f/audit_r4f.py --r4e-dir "$E_ROOT/$PREVIEW" \
     --r4c-dir "$C_DIR" --output-dir "$F_ROOT/$NAME"
   ;;
 init-full)
   [[ ! -e "$E_ROOT/$FULL" ]] || { echo "Independent NEW target required" >&2; exit 2; }
   STUDY_NAME="$FULL" MAX_CASES=0 MAX_SNAPSHOTS_PER_RUN=2 \
     HORIZONS=100,200,500 MAX_ALTERNATIVES=8 \
     LONG_PROOF_STUDY="${LONG_PROOF_STUDY:-r4e_long_proof_16snaps_v1}" \
     MAX_NEW_UNITS=1 bash road_vrp7_step6c1_r4e/run_r4e.sh collect-pilot
   ;;
 reuse)
   "$PY" road_vrp7_step6c1_r4f/reuse_r4f.py --source-dir "$E_ROOT/$PREVIEW" --target-dir "$E_ROOT/$FULL"
   ;;
 collect-full)
   [[ -d "$E_ROOT/$FULL" ]] || { echo 'Run init-full, then reuse first' >&2; exit 2; }
   STUDY_NAME="$FULL" MAX_CASES=0 MAX_SNAPSHOTS_PER_RUN=2 \
     HORIZONS=100,200,500 MAX_ALTERNATIVES=8 \
     LONG_PROOF_STUDY="${LONG_PROOF_STUDY:-r4e_long_proof_16snaps_v1}" \
     MAX_NEW_UNITS="${MAX_NEW_UNITS:-0}" RESUME=1 \
     bash road_vrp7_step6c1_r4e/run_r4e.sh collect-pilot
   ;;
 audit-full)
   NAME="${NAME:-r4f_reversal_16snaps_v1}"
   "$PY" road_vrp7_step6c1_r4f/audit_r4f.py --r4e-dir "$E_ROOT/$FULL" \
      --r4c-dir "$C_DIR" --output-dir "$F_ROOT/$NAME" --require-full
   ;;
 *) echo 'Usage: run_r4f.sh test|audit-preview|init-full|reuse|collect-full|audit-full' >&2;exit 2;;
esac
