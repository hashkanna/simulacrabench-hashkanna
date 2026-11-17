#!/bin/bash
# Official-tool validation of a built submission on the DGX.
# usage: official_check.sh DIST_DIR PYTHON [phases]
# Runs tools/check_submission_zip.py on DIST_DIR.zip, then score.py on the three
# sandbox instruments for each phase, in the given interpreter's environment
# (requirements.txt is removed from a scratch copy so score.py does not build a
# fresh CPU-only venv; the interpreter already carries the pinned packages).
set -u
DIST=$1; PY=$2; PHASES=${3:-"1 2"}
KIT=$HOME/work/claude-sim/simulacra-kit
echo "== zip check"; $PY $KIT/tools/check_submission_zip.py $DIST.zip
SCR=$(mktemp -d); cp -r $DIST/. $SCR/; rm -f $SCR/requirements.txt
for phase in $PHASES; do
  for inst in unicef world_bank unhcr; do
    echo "== score.py phase $phase $inst"
    (cd $KIT && $PY score.py --submission $SCR --data $HOME/work/claude-sim/data/sandbox/$inst \
        --schema data/$inst.json --phase $phase 2>&1 | grep -E "PASS|FAIL|Error|phase\]|data\]" | tail -4)
  done
done
rm -rf $SCR
