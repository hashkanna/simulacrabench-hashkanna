#!/bin/bash
# Which in-context experts, at what ensemble size, are worth their time?
# Phase-2 frames, final OOF plan, no time limit, lab FM cache (each expert x
# n_estimators is computed once; later configs reuse it). Pinned venv.
cd $HOME/work/claude-sim
SUB=${SUB:-subs/v5e}
export PY=./venvchk/bin/python
CACHE=$HOME/work/claude-sim/fmcache
W="SIMULACRA_WEIGHTS=$HOME/work/claude-sim/weights SIMULACRA_BUDGET=20000 SIMULACRA_FORCE_PHASE=final SIMULACRA_LAB_FM_CACHE=$CACHE"
D=${D:-"data/proxy/reachsm_s1 data/proxy/reachsm_s2 data/proxy/wgm_a_s1 data/proxy/riasec_s1"}
run() {
  echo "== $1 ($2 | $3) $(date +%H:%M:%S)"
  claude/lab/run_matrix.sh fm_$1 $SUB 2 "$W SIMULACRA_EXPERTS=marginal,lr,mlp,$2,dec SIMULACRA_FM_NEST=$3" $D
}
run f1i1   pfnf,icl     pfnf=1,icl=1
run p1i1   pfn,icl      pfn=1,icl=1
run f1i1p1 pfnf,icl,pfn pfnf=1,icl=1,pfn=1
run f4i4   pfnf,icl     pfnf=4,icl=4
run p4i4   pfn,icl      pfn=4,icl=4
run f4i4p1 pfnf,icl,pfn pfnf=4,icl=4,pfn=1
run f4i4p4 pfnf,icl,pfn pfnf=4,icl=4,pfn=4
run f1i4p4 pfnf,icl,pfn pfnf=1,icl=4,pfn=4
run f1i1p4 pfnf,icl,pfn pfnf=1,icl=1,pfn=4
run p4     pfn          pfn=4
$PY claude/lab/compare.py results/fm_f1i1 results/fm_p1i1 results/fm_f1i1p1 results/fm_f4i4 \
  results/fm_p4i4 results/fm_f4i4p1 results/fm_f4i4p4 results/fm_f1i4p4 results/fm_f1i1p4 results/fm_p4
echo "done $(date)"
