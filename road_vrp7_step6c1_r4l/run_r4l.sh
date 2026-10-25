#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
OUT_ROOT="${R4L_ROOT:-runs/chicago_july_v1/step6c1_r4l}"
STUDY="${STUDY_NAME:-r4l_seed_transfer_loto_v1}"
OLD_CF="${R4L_OLD_CF_DIR:-runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1}"
OLD_CAPTURE="${R4L_OLD_CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture}"
OLD_PREFIX="${R4L_OLD_PREFIX_DIR:-runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1}"
NEW_R4K="${R4L_NEW_R4K_DIR:-runs/chicago_july_v1/step6c1_r4k/r4k_full_128audited_v1}"
NEW_CF="${R4L_NEW_CF_DIR:-runs/chicago_july_v1/step6c1_r4h/r4j_train_early_s62s63_v1_cf_full}"
NEW_CAPTURE="${R4L_NEW_CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4g/r4j_train_early_s62s63_v1_capture}"
NEW_PREFIX="${R4L_NEW_PREFIX_DIR:-runs/chicago_july_v1/step6c1_r4g_prefix/r4j_train_early_s62s63_v1_prefix_full}"
case "$MODE" in
  test)
    "$PY" -m unittest discover -s road_vrp7_step6c1_r4l/tests -p 'test_r4l*.py' -v
    "$PY" -m py_compile road_vrp7_step6c1_r4l/*.py
    bash -n road_vrp7_step6c1_r4l/run_r4l.sh
    ;;
  preflight)
    for f in "$OLD_CF/report.json" "$OLD_CAPTURE/snapshots.jsonl" "$OLD_PREFIX/report.json" \
             "$NEW_R4K/report.json" "$NEW_R4K/audited_labels.jsonl" \
             "$NEW_CF/report.json" "$NEW_CF/manifest.json" \
             "$NEW_CAPTURE/snapshots.jsonl" "$NEW_PREFIX/report.json"; do
       [[ -f "$f" ]] || { echo "[R4L] MISSING REQUIRED SOURCE: $f" >&2; exit 2; }
    done
    echo "[R4L] File presence OK; actual SHA, Train role and unit identity are verified by probe." >&2
    "$PY" - "$OLD_CF/report.json" "$NEW_R4K/report.json" "$NEW_CF/report.json" <<'PY'
import json,sys
old,r4k,new=(json.load(open(p)) for p in sys.argv[1:])
assert old['status']=='COMPLETE' and r4k['status']=='SOURCE_COMPLETE' and new['status']=='COMPLETE'
assert r4k['analysis']['completed_scenario_snapshots']==128
print('OLD',old['n_scenario_snapshots'],'NEW',r4k['analysis']['completed_scenario_snapshots'],
      'NEW-RECONSTRUCTED-UNITS',r4k['analysis']['existing_unit_files'])
PY
    ;;
  probe)
    "$PY" road_vrp7_step6c1_r4l/probe_r4l.py \
        --old-cf-dir "$OLD_CF" --old-capture-dir "$OLD_CAPTURE" --old-prefix-dir "$OLD_PREFIX" \
        --new-r4k-dir "$NEW_R4K" --new-cf-dir "$NEW_CF" \
        --new-capture-dir "$NEW_CAPTURE" --new-prefix-dir "$NEW_PREFIX" \
        --output-dir "$OUT_ROOT/$STUDY"
    ;;
  *) echo 'Usage: bash road_vrp7_step6c1_r4l/run_r4l.sh test|preflight|probe' >&2;exit 2;;
esac
