#!/bin/bash
# Follow-up: TabPFN-3.5 multiclass variant (pfnm, already in the models.txt
# download) alone and on top of full TabPFN + TabICL. Same cache as fm_matrix.sh.
cd $HOME/work/claude-sim
SUB=${SUB:-subs/v5g}
export PY=./venvchk/bin/python
CACHE=$HOME/work/claude-sim/fmcache
W="SIMULACRA_WEIGHTS=$HOME/work/claude-sim/weights SIMULACRA_BUDGET=20000 SIMULACRA_FORCE_PHASE=final SIMULACRA_LAB_FM_CACHE=$CACHE"
D=${D:-"data/proxy/reachsm_s1 data/proxy/reachsm_s2 data/proxy/wgm_a_s1 data/proxy/riasec_s1"}
run() {
  echo "== $1 ($2 | $3) $(date +%H:%M:%S)"
  claude/lab/run_matrix.sh fm_$1 $SUB 2 "$W SIMULACRA_EXPERTS=marginal,lr,mlp,$2,dec SIMULACRA_FM_ORDER_FINAL=$2 SIMULACRA_FM_NEST=$3" $D
}
run m1i1     pfnm,icl      pfnm=1,icl=1
run p1i1m1   pfn,icl,pfnm  pfn=1,icl=1,pfnm=1
run p4i4m1   pfn,icl,pfnm  pfn=4,icl=4,pfnm=1
run p4i4m4   pfn,icl,pfnm  pfn=4,icl=4,pfnm=4
run p8i4     pfn,icl       pfn=8,icl=4
$PY claude/lab/compare.py results/fm_p1i1 results/fm_m1i1 results/fm_p1i1m1
$PY claude/lab/compare.py results/fm_p4i4 results/fm_p4i4m1 results/fm_p4i4m4 results/fm_f4i4p4 results/fm_p8i4
echo "done2 $(date)"
