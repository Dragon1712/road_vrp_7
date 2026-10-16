#!/usr/bin/env bash
# Run from road_vrp_7 repository root after extracting step5a_v2_analyze_fix/.
set -euo pipefail
ROOT="$(pwd)"
SRC="$ROOT/step5a_v2_analyze_fix/analyze_step5a.py"
DEST="$ROOT/road_vrp7_step5a_v2/analyze_step5a.py"
OLD_SHA='26314ea8eb9070bdd2799c33cbf004334e3d6864ce92685a110f63d689eab638'
if [ ! -f "$SRC" ] || [ ! -f "$DEST" ]; then
    echo 'Missing patch source or existing Step5-A v2 analyzer. Run in road_vrp_7 root.' >&2
    exit 2
fi
new_sha="$(sha256sum "$SRC" | awk '{print $1}')"
old_sha="$(sha256sum "$DEST" | awk '{print $1}')"
if [ "$old_sha" = "$new_sha" ]; then
    echo 'Analyzer is already patched; no overwrite needed.'
elif [ "$old_sha" = "$OLD_SHA" ]; then
    cp "$DEST" "$DEST.before_step5a_v2_fix.bak"
    cp "$SRC" "$DEST"
    echo 'Patched analyzer and backed up original file.'
else
    echo "Refusing to overwrite modified analyzer (sha256=$old_sha). Review the diff first." >&2
    exit 3
fi
python -m py_compile "$DEST"
python "$ROOT/step5a_v2_analyze_fix/tests/test_analyzer_fix.py" >/dev/null
printf 'PASS: patched analyzer compiles and offline regression checks pass.\n'
echo 'Next: bash road_vrp7_step5a_v2/run_stage5a_v2.sh analyze pilot'
