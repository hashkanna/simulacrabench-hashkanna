#!/bin/bash
# Chained GPU jobs after the dev candidate is validated.
cd $HOME/work/claude-sim
echo "== dev dry run (dev2 zip, grader order, 900 s)"; date
claude/lab/dry_run.sh $HOME/work/claude-sim/dist/claude-v5-dev3 1 /tmp/claude_dry_dev3.json
echo "== official checks (dev zip, phase 1)"; date
claude/lab/official_check.sh $HOME/work/claude-sim/dist/claude-v5-dev3 $HOME/work/claude-sim/venvchk/bin/python "1" 2>&1 | grep -E "==|PASS|FAIL|OK:|Error"
echo "== final-path dry run (sandbox phase 2, real 3600 s budget)"; date
claude/lab/dry_run.sh $HOME/work/claude-sim/dist/claude-v5-dev3 2 /tmp/claude_dry_final.json
echo "== final-config proxy matrix"; date
claude/lab/final_cfg_matrix.sh
date
