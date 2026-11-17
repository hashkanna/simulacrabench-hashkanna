#!/bin/bash
# Panel-twin lab: TabPFN-3.5 x8 vs TabICL x4 vs both pooled, on the twin proxy and its
# no-twin source; final plan, fixed ensemble sizes, no time limit, cached FM outputs.
cd $HOME/work/claude-sim
export HF_HUB_CACHE=$HOME/work/claude-sim/hubcache
export PY=./venvchk/bin/python
SUB=${SUB:-dist/claude-v5l-icl-dev}
BASE="SIMULACRA_BUDGET=100000 SIMULACRA_FORCE_PHASE=final SIMULACRA_EXPERTS=marginal,lr,mlp,pfn,icl,dec SIMULACRA_FM_NEST=pfn=8,icl=4 SIMULACRA_LAB_FM_CACHE=$HOME/work/claude-sim/fmcache"
for data in ${DATA:-data/proxy/reachsmtwin_s1 data/proxy/reachsm_s1}; do
  for order in pfn,icl pfn icl; do
    echo "== $data $order $(date -u +%H:%M:%S)"
    claude/lab/run_matrix.sh twin_${order/,/_} $SUB 2 "$BASE SIMULACRA_FM_ORDER_FINAL=$order" $data
  done
done
$PY claude/lab/compare.py results/twin_pfn results/twin_pfn_icl results/twin_icl
$PY claude/lab/twin_breakdown.py data/proxy/reachsmtwin_s1 results/twin_pfn results/twin_pfn_icl results/twin_icl
echo "twin lab done $(date -u +%H:%M:%S)"
