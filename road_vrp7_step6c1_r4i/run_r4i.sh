#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
PKG=road_vrp7_step6c1_r4i
ROOT="${R4I_ROOT:-runs/chicago_july_v1/step6c1_r4i}"
case "${1:-}" in
 test)
    "$PY" -m unittest discover -s "$PKG/tests" -p 'test_r4i*.py' -v
    "$PY" -m py_compile "$PKG"/*.py
    bash -n "$PKG/run_r4i.sh"
    ;;
 probe)
    "$PY" "$PKG/probe_r4i.py" \
       --r4h-dir "${R4H_DIR:-runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1}" \
       --capture-dir "${R4G_CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture}" \
       --prefix-dir "${R4G_PREFIX_DIR:-runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1}" \
       --output-dir "$ROOT/${STUDY_NAME:-r4i_signal_loto_train_v1}"
    ;;
 *) echo 'Usage: run_r4i.sh test|probe' >&2; exit 2;;
esac
