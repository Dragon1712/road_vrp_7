#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
VQ4=road_vrp7_step6c1_r3_vq4
VQ3=road_vrp7_step6c1_r3_vq3
BASE="${VQ4_ROOT:-runs/chicago_july_v1/step6c1_r3_vq4}"
OLD_ROOT="${OLD_VQ_ROOT:-runs/chicago_july_v1/step6c1_r3_vq}"
OLD="$OLD_ROOT/${OLD_STUDY:-vq_train48_H20H50_R3_v1}"
COLLECTION="${COLLECTION_STUDY:-vq4_train192_H20H50_R3_v1}"
TARGET="$BASE/$COLLECTION"
DATASET="${DATASET_DIR:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}"
MODEL_ROOT="$BASE/models"
case "${1:-help}" in
  test)
    "$PY" -m unittest discover -s "$VQ4/tests" -v
    "$PY" -m py_compile "$VQ4"/*.py
    bash -n "$VQ4/run_vq4.sh"
    ;;
  collect)
    # First run with MAX_NEW_UNITS=1 and do not set RESUME=1.
    # Following reuse, run again with RESUME=1 and unchanged settings.
    options=(); if [[ "${RESUME:-0}" == 1 ]];then options+=(--resume);fi
    "$PY" road_vrp7_step6c1_r3_vq/collect_vq.py \
      --output-dir "$TARGET" --dataset-dir "$DATASET" \
      --cost-dir "${COST_DIR:-datasets/processed/chicago/july_v1/dynamic_costs}" \
      --task-dir "${TASK_DIR:-datasets/processed/chicago/july_v1/tdcvrp30/tasks_10}" \
      --od-cache-dir "${OD_CACHE_DIR:-runs/chicago_july_v1/stage3/od_cache}" \
      --states "${STATES:-192}" --seed "${SELECT_SEED:-20261025}" \
      --horizons "${HORIZONS:-20,50}" --replicates "${REPLICATES:-3}" \
      --max-new-units "${MAX_NEW_UNITS:-0}" --wall-per-arm "${WALL_PER_ARM:-30}" \
      --road-replay "${options[@]}"
    ;;
  preflight|reuse)
    "$PY" "$VQ4/vq4_prepare.py" "${1}" --old48 "$OLD" --target192 "$TARGET"
    ;;
  views)
    "$PY" "$VQ4/vq4_prepare.py" views --old48 "$OLD" \
       --target192 "$TARGET" --out-root "$BASE"
    ;;
  train)
    : "${TRAIN_SEED:?Set TRAIN_SEED=42..47}"
    : "${SIZE:?Set SIZE=48|96|144|192}"
    case "$SIZE" in 48|96|144|192) ;; *) echo 'SIZE must be 48,96,144,192' >&2;exit 2;;esac
    dir="$BASE/vq4_view${SIZE}_v1"
    if [[ ! -f "$dir/report.json" ]];then echo "Missing COMPLETE VQ4 view: $dir" >&2;exit 2;fi
    "$PY" "$VQ3/train_vq3.py" --collection-dir "$dir" --dataset-dir "$DATASET" \
      --output-dir "$MODEL_ROOT/vq4_n${SIZE}_seed${TRAIN_SEED}_v1" \
      --seed "$TRAIN_SEED" --trees "${TREES:-160}"
    ;;
  aggregate)
    : "${SIZE:?Set SIZE=48|96|144|192}"
    case "$SIZE" in 48|96|144|192) ;; *) echo 'Bad SIZE' >&2;exit 2;;esac
    args=()
    for s in 42 43 44 45 46 47; do args+=("$MODEL_ROOT/vq4_n${SIZE}_seed${s}_v1");done
    IFS=,; paths="${args[*]}";unset IFS
    "$PY" "$VQ3/aggregate_vq3.py" --study-dirs "$paths" \
      --output-dir "$BASE/vq4_aggregate_n${SIZE}_seed42to47_v1"
    ;;
  analyze)
    "$PY" "$VQ4/analyze_curve.py" --model-root "$MODEL_ROOT" \
      --output-dir "$BASE/${ANALYSIS_STUDY:-vq4_learning_curve_seed42to47_v1}" \
      --baseline48-root "${BASELINE48_ROOT:-runs/chicago_july_v1/step6c1_r3_vq3}"
    ;;
  *) echo 'Usage: run_vq4.sh test|collect|preflight|reuse|views|train|aggregate|analyze' >&2;exit 2;;
esac
