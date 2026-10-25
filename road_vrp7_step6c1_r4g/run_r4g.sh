#!/usr/bin/env bash
# R4-G v1.1: Do not inherit generic TASK_IDS/SCENARIO_IDS/SEEDS from earlier experiments.
# Use R4G_* environment overrides. Python TRAIN-role guards remain in force.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
ROOT="${R4G_ROOT:-runs/chicago_july_v1/step6c1_r4g}"
MODE="${1:-help}"
NAME="${STUDY_NAME:-r4g_train16_v1}"
BASE="$ROOT/$NAME"

# The 'show-config' mode displays exactly what would be run, without loading
# any dataset or requiring the ALNS package. Useful for reproducibility.
CONFIG_MODE="$MODE"
if [[ "$MODE" == show-config ]]; then
  CONFIG_MODE="${R4G_VIEW_MODE:-smoke}"
  [[ "$CONFIG_MODE" == smoke || "$CONFIG_MODE" == pilot ]] || {
    echo 'R4G_VIEW_MODE must be smoke or pilot' >&2;exit 2;
  }
fi

if [[ "$CONFIG_MODE" == smoke ]]; then
  TASK_IDS="${R4G_TASK_IDS:-task_0000}"
  SCENARIO_IDS="${R4G_SCENARIO_IDS:-2025-07-02_0800}"
  SEEDS="${R4G_SEEDS:-60}"
  MAX_ITERATIONS="${R4G_MAX_ITERATIONS:-2000}"
  SNAPSHOTS="${R4G_SNAPSHOTS:-250,700,1200,1700}"
else
  TASK_IDS="${R4G_TASK_IDS:-task_0000,task_0001,task_0002,task_0003}"
  SCENARIO_IDS="${R4G_SCENARIO_IDS:-2025-07-02_0800,2025-07-09_1700}"
  SEEDS="${R4G_SEEDS:-60,61}"
  MAX_ITERATIONS="${R4G_MAX_ITERATIONS:-20000}"
  SNAPSHOTS="${R4G_SNAPSHOTS:-1000,4000,8000,12000,16000,19000}"
fi
MAX_SNAPSHOTS="${R4G_MAX_SNAPSHOTS:-16}"

show_config() {
  printf 'MODE=%s\nTASK_IDS=%s\nSCENARIO_IDS=%s\nSEEDS=%s\nMAX_ITERATIONS=%s\nSNAPSHOTS=%s\nMAX_SNAPSHOTS=%s\nSTUDY_NAME=%s\n' \
    "$CONFIG_MODE" "$TASK_IDS" "$SCENARIO_IDS" "$SEEDS" "$MAX_ITERATIONS" "$SNAPSHOTS" "$MAX_SNAPSHOTS" "$NAME"
}

if [[ "$MODE" == smoke || "$MODE" == pilot || "$MODE" == show-config ]]; then
  for ambient in TASK_IDS SCENARIO_IDS SEEDS MAX_ITERATIONS SNAPSHOTS; do
    # The variables have been assigned to our scoped defaults above. Inspect
    # original exported environment via the process environment, not shell locals.
    if env | grep -q "^${ambient}="; then
      echo "[R4G] Ignoring inherited generic $ambient; use R4G_$ambient to override." >&2
    fi
  done
  IFS=',' read -r -a TASK_ARRAY <<< "$TASK_IDS"
  [[ ${#TASK_ARRAY[@]} -gt 0 ]] || { echo '[R4G] Empty task list' >&2;exit 2; }
  for t in "${TASK_ARRAY[@]}"; do
    case "$t" in
      task_0000|task_0001|task_0002|task_0003) ;;
      *) echo "[R4G] Non-TRAIN/unauthorized task '$t'. Allowed task_0000..task_0003; use R4G_TASK_IDS." >&2;exit 2;;
    esac
  done
  if [[ "$MODE" == show-config ]]; then show_config;exit 0; fi
  echo '[R4G] Effective TRAIN acquisition configuration:' >&2
  show_config >&2
fi

collect() {
  local arm="$1"
  "$PY" road_vrp7_step6c1_r4g/collect_train_r4g.py \
    --arm "$arm" --output-dir "${BASE}_${arm}" \
    --task-ids "$TASK_IDS" --scenario-ids "$SCENARIO_IDS" --seeds "$SEEDS" \
    --wall-sec 30 --max-iterations "$MAX_ITERATIONS" \
    --top-k 12 --anneal-iterations 5000 --snapshots "$SNAPSHOTS" \
    --max-snapshots "$MAX_SNAPSHOTS" --section-only
}
audit() {
  local extra=()
  [[ "${REQUIRE_STRICT:-1}" != 1 ]] || extra+=(--require-strict)
  "$PY" road_vrp7_step6c1_r4g/audit_r4g.py \
      --plain "${BASE}_plain" --capture "${BASE}_capture" \
      --output-dir "${BASE}_audit" "${extra[@]}"
}
proof() {
  local casecount="$1"
  local extra=()
  [[ "${RESUME:-0}" != 1 ]] || extra+=(--resume)
  "$PY" road_vrp7_step6c1_r4g/verify_train_r4g.py \
      --capture-dir "${CAPTURE_STUDY_DIR:-${BASE}_capture}" \
      --output-dir "$ROOT/${PROOF_STUDY_NAME:-${NAME}_proof}" \
      --max-cases "$casecount" \
      --max-snapshots-per-run "${PROOF_SNAPSHOTS_PER_RUN:-3}" \
      --horizons "${HORIZONS:-1,50,500}" "${extra[@]}"
}
case "$MODE" in
  test)
    "$PY" -m unittest discover -s road_vrp7_step6c1_r4g/tests -v
    "$PY" -m py_compile road_vrp7_step6c1_r4g/*.py
    bash -n road_vrp7_step6c1_r4g/run_r4g.sh
    ;;
  smoke|pilot)
    [[ ! -e "${BASE}_plain" && ! -e "${BASE}_capture" && ! -e "${BASE}_audit" ]] || {
      echo 'Independent NEW STUDY_NAME required for collection' >&2;exit 2;
    }
    collect plain
    collect capture
    audit
    ;;
  proof-smoke) proof 1;;
  proof-pilot) proof 0;;
  audit) audit;;
  diagnose)
    # Read-only post-hoc audit, new result directory, NO ALNS rerun.
    diag_name="${R4G_DIAG_NAME:-${NAME}_audit_diagnostic_v1_3}"
    "$PY" road_vrp7_step6c1_r4g/audit_r4g.py \
       --plain "${BASE}_plain" --capture "${BASE}_capture" \
       --output-dir "$ROOT/$diag_name"
    ;;
  *) echo 'Usage: run_r4g.sh test|show-config|smoke|pilot|audit|diagnose|proof-smoke|proof-pilot' >&2;exit 2;;
esac
