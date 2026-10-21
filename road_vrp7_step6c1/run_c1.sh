#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
RUNS="runs/chicago_july_v1/step6c1"
SOURCE="${SOURCE_DIR:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}"
FEATURES="${FEATURES_DIR:-runs/chicago_july_v1/step6b2/features_boundary_v1}"
GRAPH="${GRAPH_DIR:-$RUNS/c1_graph_full_v1}"
SEED="${SEED:-42}"
STUDY_NAME="${STUDY_NAME:-}"
if [[ "$SEED" == 42 ]];then
  DEFAULT_ANCHOR="runs/chicago_july_v1/step6b2/b2_full_seed42_v1"
else
  DEFAULT_ANCHOR="runs/chicago_july_v1/step6b2_r/b2r_ablation_seed${SEED}_v1"
fi
ANCHOR="${ANCHOR_STUDY:-$DEFAULT_ANCHOR}"
case "$MODE" in
 test)
  "$PY" -m unittest discover -s road_vrp7_step6c1/tests -v ;;
 export)
  : "${STUDY_NAME:?Set a NEW STUDY_NAME, e.g. c1_graph_full_v1}"
  args=(--source-dir "$SOURCE" --features-dir "$FEATURES" --output-dir "$RUNS/$STUDY_NAME")
  [[ "${RESUME:-0}" == 1 ]] && args+=(--resume)
  [[ -z "${MAX_STATES:-}" ]] || args+=(--max-states "$MAX_STATES")
  [[ "${VALIDATE_F4:-1}" == 1 ]] && args+=(--validate-f4)
  "$PY" road_vrp7_step6c1/export_graph.py "${args[@]}" ;;
 train)
  : "${STUDY_NAME:?Set a NEW STUDY_NAME, e.g. c1_decision_seed42_v1}"
  "$PY" road_vrp7_step6c1/train_c1.py --source-dir "$SOURCE" --features-dir "$FEATURES" \
     --graph-dir "$GRAPH" --anchor-study "$ANCHOR" --output-dir "$RUNS/$STUDY_NAME" \
     --seed "$SEED" --protocol "${PROTOCOL:-both}" --variants "${VARIANTS:-D0,D1,D2T,D2}" \
     --epochs "${EPOCHS:-30}" --pretrain-epochs "${PRETRAIN_EPOCHS:-4}" \
     --decision-weight "${DECISION_WEIGHT:-1.0}" ;;
 online)
  : "${STUDY_NAME:?Set a NEW STUDY_NAME, e.g. c1_online_D2_match8_v1}"
  C1_VARIANT="${C1_VARIANT:-D2}"
  case "$C1_VARIANT" in D0|D1|D2T|D2) ;; *) echo "Expected C1_VARIANT D0,D1,D2T,D2" >&2;exit 2;; esac
  MODEL_STUDY="${MODEL_STUDY:-$RUNS/c1_decision_seed42_v1}"
  "$PY" road_vrp7_step6c1/online_c1.py --graph-dir "$GRAPH" \
    --control learned --model-method P2 --model-mode F4 --model-protocol task \
    --model-path "$MODEL_STUDY/task/${C1_VARIANT}_model.joblib" --model-source "$SOURCE" \
    --output-dir "$RUNS/$STUDY_NAME" --seeds "${SEEDS:-42,43}" \
    --split "${SPLIT:-validation}" \
    --task-ids "${TASK_IDS:-task_0008,task_0009}" \
    --scenario-ids "${SCENARIO_IDS:-2025-07-25_0800,2025-07-25_1700}" \
    --wall-sec 30 --max-iterations "${MAX_ITERATIONS:-20000}" \
    --anneal-iterations 5000 --top-k 12 --action-count 8 \
    --sampler-seed 20261019 --score-every "${SCORE_EVERY:-8}" \
    --max-scoring-sec "${MAX_SCORING_SEC:-2}" --cache-size 64 \
    --min-static-fraction 0 --fallback c3 ;;
 compare)
  : "${STUDY_NAME:?Set a new STUDY_NAME}"
  : "${D0_RUN:?D0_RUN required}";: "${D1_RUN:?D1_RUN required}";: "${D2T_RUN:?D2T_RUN required}";: "${D2_RUN:?D2_RUN required}"
  args=(--base "${BASE_RUN:-runs/chicago_july_v1/step6b2_r/b2r_online_P2_F2_match8_v1}" \
        --m3 "${M3_RUN:-runs/chicago_july_v1/step6b3/b3_online_M3_match8_v1}" \
        --arm "D0=$D0_RUN" --arm "D1=$D1_RUN" --arm "D2T=$D2T_RUN" --arm "D2=$D2_RUN" \
        --output-dir "$RUNS/$STUDY_NAME")
  [[ -z "${RANDOM_RUN:-}" ]] || args+=(--arm "random=$RANDOM_RUN")
  "$PY" road_vrp7_step6b3_r/online_compare_b3r.py "${args[@]}" ;;
 *) echo 'Usage: bash road_vrp7_step6c1/run_c1.sh test|export|train|online|compare' >&2;exit 2;;
esac
