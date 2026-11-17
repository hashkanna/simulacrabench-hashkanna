#!/bin/bash
# Overnight: value of final-phase FM settings on proxies (phase-2 frames).
cd $HOME/work/claude-sim
W="SIMULACRA_WEIGHTS=$HOME/work/claude-sim/weights SIMULACRA_BUDGET=20000"
D="data/proxy/wgm_a_s1 data/proxy/riasec_s1 data/proxy/reachsm_s1"
echo "== devcfg";  claude/lab/run_matrix.sh fc_dev   subs/v5e 2 "$W SIMULACRA_EXPERTS=marginal,lr,lri,mlp,pfnf,icl,dec" $D
echo "== finalcfg"; claude/lab/run_matrix.sh fc_final subs/v5e 2 "$W SIMULACRA_FORCE_PHASE=final SIMULACRA_EXPERTS=marginal,lr,lri,mlp,pfnf,icl,dec" $D
echo "== final+pfn"; claude/lab/run_matrix.sh fc_final_pfn subs/v5e 2 "$W SIMULACRA_FORCE_PHASE=final SIMULACRA_EXPERTS=marginal,lr,lri,mlp,pfnf,icl,pfn,dec" $D
./venv/bin/python claude/lab/compare.py results/fc_dev results/fc_final results/fc_final_pfn
