#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
ROOT="runs/chicago_july_v1/step6c1_r"
GRAPH="${GRAPH_DIR:-runs/chicago_july_v1/step6c1/c1_graph_full_v1}"
SOURCE="${SOURCE_DIR:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}"
FEATURES="${FEATURES_DIR:-runs/chicago_july_v1/step6b2/features_boundary_v1}"
SEED="${SEED:-42}"
NAME="${STUDY_NAME:-}"
if [[ "$SEED" == 42 ]];then
 DEFAULT_ANCHOR="runs/chicago_july_v1/step6b2/b2_full_seed42_v1"
else
 DEFAULT_ANCHOR="runs/chicago_july_v1/step6b2_r/b2r_ablation_seed${SEED}_v1"
fi
ANCHOR="${ANCHOR_STUDY:-$DEFAULT_ANCHOR}"
case "$MODE" in
 test)
  "$PY" -m unittest discover -s road_vrp7_step6c1_r/tests -v
  "$PY" -m unittest discover -s road_vrp7_step6c1/tests -v ;;
 graph-audit)
  : "${NAME:?Set a NEW STUDY_NAME}"
  "$PY" road_vrp7_step6c1_r/graph_audit.py --graph-dir "$GRAPH" --output-dir "$ROOT/$NAME" ;;
 train)
  : "${NAME:?Set a NEW STUDY_NAME}"
  "$PY" road_vrp7_step6c1_r/train_c1r.py --source-dir "$SOURCE" --features-dir "$FEATURES" \
    --graph-dir "$GRAPH" --anchor-study "$ANCHOR" --output-dir "$ROOT/$NAME" \
    --seed "$SEED" --protocol "${PROTOCOL:-both}" \
    --variants "${VARIANTS:-D0,D1,D2T,D2}" \
    --arms "${ARMS:-OOF_ZERO,FULL_ZERO,FULL_DEFER,FULL_DEFER_SAFE}" \
    --epochs "${EPOCHS:-30}" --pretrain-epochs "${PRETRAIN_EPOCHS:-4}" ;;
 online)
  : "${NAME:?Set a NEW STUDY_NAME}"
  VARIANT="${C1_VARIANT:-D0}"
  ARM="${C1_ARM:-FULL_DEFER_SAFE}"
  case "$VARIANT" in D0|D1|D2T|D2) ;; *) echo 'Bad variant' >&2;exit 2;;esac
  case "$ARM" in OOF_ZERO|FULL_ZERO|FULL_DEFER|FULL_DEFER_SAFE) ;; *) echo 'Bad arm' >&2;exit 2;;esac
  STUDY="${MODEL_STUDY:-$ROOT/c1r_seed42_v1}"
  FEATURE_MODE=F4;[[ "$VARIANT" == D0 ]] && FEATURE_MODE=F2
  "$PY" road_vrp7_step6c1_r/online_c1r.py --graph-dir "$GRAPH" \
    --control learned --model-method P2 --model-mode "$FEATURE_MODE" --model-protocol task \
    --model-path "$STUDY/task/${VARIANT}_${ARM}_model.joblib" --model-source "$SOURCE" \
    --output-dir "$ROOT/$NAME" --seeds "${SEEDS:-42,43}" \
    --split "${SPLIT:-validation}" \
    --task-ids "${TASK_IDS:-task_0008,task_0009}" \
    --scenario-ids "${SCENARIO_IDS:-2025-07-25_0800,2025-07-25_1700}" \
    --wall-sec 30 --max-iterations "${MAX_ITERATIONS:-20000}" \
    --anneal-iterations 5000 --top-k 12 --action-count 8 \
    --sampler-seed 20261019 --score-every "${SCORE_EVERY:-8}" \
    --max-scoring-sec "${MAX_SCORING_SEC:-2}" --cache-size 64 \
    --min-static-fraction 0 --fallback c3 ;;
 aggregate)
  : "${NAME:?Set a NEW STUDY_NAME}"
  "$PY" road_vrp7_step6c1_r/aggregate_c1r.py --seeds "${MODEL_SEEDS:-42,43,44,45,46,47}" \
    --output-dir "$ROOT/$NAME" ;;
 compare)
  : "${NAME:?Set a NEW STUDY_NAME}"
  : "${R_RUN:?Set R_RUN to new online run path}"
  "$PY" road_vrp7_step6b3_r/online_compare_b3r.py \
    --base "${BASE_RUN:-runs/chicago_july_v1/step6b2_r/b2r_online_P2_F2_match8_v1}" \
    --m3 "${M3_RUN:-runs/chicago_july_v1/step6b3/b3_online_M3_match8_v1}" \
    --arm "C1R=$R_RUN" --output-dir "$ROOT/$NAME" ;;
 *) echo 'Usage: run_c1r.sh test|graph-audit|train|aggregate|online|compare' >&2;exit 2;;
esac
