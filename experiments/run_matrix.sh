#!/bin/bash
# usage: run_matrix.sh TAG SUBDIR PHASE "ENV=.. ENV=.." data_dirs...
# Runs one configuration on several datasets, saving json + per-cell logp.
TAG=$1; SUB=$2; PHASE=$3; ENVS=$4; shift 4
mkdir -p results/$TAG
env $ENVS ${PY:-./venv/bin/python} claude/lab/evaluate.py --sub $SUB --data "$@" --phase $PHASE \
  --kit simulacra-kit --out results/$TAG/summary.json --save-logp results/$TAG 2>&1 \
  | grep -v -i warn | grep -E "skill|total|Traceback|Error" 
