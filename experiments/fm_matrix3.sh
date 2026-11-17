#!/bin/bash
# Follow-up 2: TabPFN-only pools (full + multiclass variant) and 8 estimators,
# without TabICL (which slightly hurt next to full TabPFN x4). Reuses the cache.
cd $HOME/work/claude-sim
SUB=${SUB:-subs/v5h}
export PY=./venvchk/bin/python
CACHE=$HOME/work/claude-sim/fmcache
W="SIMULACRA_WEIGHTS=$HOME/work/claude-sim/weights SIMULACRA_BUDGET=20000 SIMULACRA_FORCE_PHASE=final SIMULACRA_LAB_FM_CACHE=$CACHE"
D=${D:-"data/proxy/reachsm_s1 data/proxy/reachsm_s2 data/proxy/wgm_a_s1 data/proxy/riasec_s1"}
run() {
  echo "== $1 ($2 | $3) $(date +%H:%M:%S)"
  claude/lab/run_matrix.sh fm_$1 $SUB 2 "$W SIMULACRA_EXPERTS=marginal,lr,mlp,$2,dec SIMULACRA_FM_ORDER_FINAL=$2 SIMULACRA_FM_NEST=$3" $D
}
run p4m4   pfn,pfnm  pfn=4,pfnm=4
run p8     pfn       pfn=8
run p8m4   pfn,pfnm  pfn=8,pfnm=4
$PY claude/lab/compare.py results/fm_p4 results/fm_p4m4 results/fm_p8 results/fm_p8m4 results/fm_p8i4 results/fm_p4i4m4
echo "done3 $(date)"
