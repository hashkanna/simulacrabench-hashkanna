#!/bin/bash
# Pooling shrinkage strength on cached FM predictions (phase-2 proxies, final plan).
cd $HOME/work/claude-sim
SUB=${SUB:-subs/v5h}
export PY=./venvchk/bin/python
CACHE=$HOME/work/claude-sim/fmcache
W="SIMULACRA_WEIGHTS=$HOME/work/claude-sim/weights SIMULACRA_BUDGET=20000 SIMULACRA_FORCE_PHASE=final SIMULACRA_LAB_FM_CACHE=$CACHE"
D=${D:-"data/proxy/reachsm_s1 data/proxy/reachsm_s2 data/proxy/wgm_a_s1 data/proxy/riasec_s1"}
run() {
  echo "== $1 ($2 | $3 | strength x$4) $(date +%H:%M:%S)"
  claude/lab/run_matrix.sh fm_$1 $SUB 2 "$W SIMULACRA_EXPERTS=marginal,lr,mlp,$2,dec SIMULACRA_FM_ORDER_FINAL=$2 SIMULACRA_FM_NEST=$3 SIMULACRA_POOL_STRENGTH=$4" $D
}
run p4i4_s10   pfn,icl pfn=4,icl=4 10
run p4i4_s100  pfn,icl pfn=4,icl=4 100
run p4i4_sinf  pfn,icl pfn=4,icl=4 1000000
run p4_s10     pfn     pfn=4       10
run p4_s100    pfn     pfn=4       100
$PY claude/lab/compare.py results/fm_p4i4 results/fm_p4i4_s10 results/fm_p4i4_s100 results/fm_p4i4_sinf results/fm_p4 results/fm_p4_s10 results/fm_p4_s100
echo "done-pool $(date)"
