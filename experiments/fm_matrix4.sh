#!/bin/bash
# 10-fold vs 5-fold in-context OOF plan (90% vs 80% context) for the pooling fit.
cd $HOME/work/claude-sim
SUB=${SUB:-subs/v5h}
export PY=./venvchk/bin/python
CACHE=$HOME/work/claude-sim/fmcache
W="SIMULACRA_WEIGHTS=$HOME/work/claude-sim/weights SIMULACRA_BUDGET=20000 SIMULACRA_FORCE_PHASE=final SIMULACRA_LAB_FM_CACHE=$CACHE"
D=${D:-"data/proxy/reachsm_s1 data/proxy/reachsm_s2 data/proxy/wgm_a_s1 data/proxy/riasec_s1"}
echo "== p4_k10 (pfn x4, 10-fold FM OOF) $(date +%H:%M:%S)"
claude/lab/run_matrix.sh fm_p4_k10 $SUB 2 "$W SIMULACRA_EXPERTS=marginal,lr,mlp,pfn,dec SIMULACRA_FM_ORDER_FINAL=pfn SIMULACRA_FM_NEST=pfn=4 SIMULACRA_FM_KFOLD_FINAL=10" $D
$PY claude/lab/compare.py results/fm_p4 results/fm_p4_k10
echo "done4 $(date)"
