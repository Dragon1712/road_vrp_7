#!/usr/bin/env bash
# Install Step5-B-v2 budget runner without replacing the original solver.
# Invoke from the repository root, after unpacking this patch.
set -euo pipefail
ROOT="$(pwd)"
SOURCE="$ROOT/step5b_v2_budget_patch/road_vrp7_step5b_v2"
TARGET="$ROOT/road_vrp7_step5b_v2"
SHELL_TARGET="$TARGET/run_stage5b_v2.sh"
if [ ! -f "$SHELL_TARGET" ] || [ ! -f "$TARGET/run_experiment.py" ] || [ ! -f "$SOURCE/budget_ablation.py" ];then
  echo 'ERROR: missing Step5-B-v2 sources; run from road_vrp_7 root after extracting patch' >&2
  exit 2
fi
for relative in budget_ablation.py tests/test_budget_ablation.py;do
  mkdir -p "$(dirname "$TARGET/$relative")"
  if [ -e "$TARGET/$relative" ] && ! cmp -s "$SOURCE/$relative" "$TARGET/$relative";then
    echo "ERROR: refusing to overwrite modified $TARGET/$relative" >&2
    exit 3
  fi
done
cp "$SOURCE/budget_ablation.py" "$TARGET/budget_ablation.py"
cp "$SOURCE/tests/test_budget_ablation.py" "$TARGET/tests/test_budget_ablation.py"
python -m py_compile "$TARGET/budget_ablation.py" "$TARGET/tests/test_budget_ablation.py"
if grep -q '^  budget)$' "$SHELL_TARGET";then
  echo 'Original stage script already has a budget entry; leaving it unchanged.'
else
  BACKUP="$SHELL_TARGET.before_budget_patch.bak"
  if [ -e "$BACKUP" ];then
    echo "ERROR: backup exists: $BACKUP. Review before patching." >&2
    exit 4
  fi
  cp "$SHELL_TARGET" "$BACKUP"
  python - "$SHELL_TARGET" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);source=p.read_text()
marker='  analyze)\n'
if source.count(marker)!=1 or 'case "$STAGE" in' not in source:
    raise SystemExit('Cannot safely find stage shell case statement')
addition='''  budget)
    "$PY" road_vrp7_step5b_v2/budget_ablation.py --action run "${@:2}" ;;
  analyze_budget)
    "$PY" road_vrp7_step5b_v2/budget_ablation.py --action analyze "${@:2}" ;;
  budget_all)
    "$PY" road_vrp7_step5b_v2/budget_ablation.py --action all "${@:2}" ;;
  budget_dry_run)
    "$PY" road_vrp7_step5b_v2/budget_ablation.py --action dry-run "${@:2}" ;;
'''
p.write_text(source.replace(marker,addition+marker,1))
PY
  if ! bash -n "$SHELL_TARGET";then
    cp "$BACKUP" "$SHELL_TARGET"
    echo 'ERROR: generated Bash syntax invalid; restored old runner' >&2
    exit 5
  fi
  echo "Patched $SHELL_TARGET; backed up as $BACKUP"
fi
# Existing Stage3/5-B scripts and data are not touched. No experiment is started.
echo 'PASS: files installed and Bash syntax validated.'
echo 'Next: bash road_vrp7_step5b_v2/run_stage5b_v2.sh budget_dry_run'
