#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
PKG=road_vrp7_step6c1_r4d
BASE="${R4D_ROOT:-runs/chicago_july_v1/step6c1_r4d}"
R4C="${R4C_DIR:-runs/chicago_july_v1/step6c1_r4c/r4c_pilot_16snaps_v1}"
R4A="${R4A_CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4a/r4a_pilot_20k_fix1_capture}"
R4B="${R4B_DIR:-runs/chicago_july_v1/step6c1_r4b/r4b_exact_8cases_2snap_v1}"
case "${1:-}" in
  test)
    "$PY" -m unittest discover -s "$PKG/tests" -p 'test_r4d*.py' -v
    "$PY" -m py_compile "$PKG"/*.py
    bash -n "$PKG/run_r4d.sh"
    ;;
  audit)
    STUDY_NAME="${STUDY_NAME:-r4d_funnel_pilot16_v1}"
    "$PY" "$PKG/audit_r4d.py" --r4c-dir "$R4C" \
      --r4c-source-dir road_vrp7_step6c1_r4c \
      --capture-dir "$R4A" --r4b-dir "$R4B" --output-dir "$BASE/$STUDY_NAME"
    ;;
  *) echo 'Usage: run_r4d.sh test|audit' >&2;exit 2;;
esac
