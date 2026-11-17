#!/bin/bash
# Full grader-order dry run of a built submission: one import, three predict()
# calls (UNICEF, World Bank, UNHCR), real phase budgets (no overrides).
# usage: dry_run.sh DIST_DIR PHASE OUT_JSON
DIST=$1; PHASE=$2; OUT=$3
cd $HOME/work/claude-sim
env -u SIMULACRA_WEIGHTS -u SIMULACRA_BUDGET -u SIMULACRA_EXPERTS \
  ./venvchk/bin/python claude/lab/evaluate.py --sub $DIST \
  --data data/sandbox/unicef data/sandbox/world_bank data/sandbox/unhcr \
  --phase $PHASE --kit simulacra-kit --out $OUT 2>&1 | grep -v -i warn | grep -E "skill|total|Traceback|Error"
./venv/bin/python - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("import %.1fs total %.1fs" % (d["import_seconds"], d["total_seconds"]))
for r in d["runs"]:
    i = r.get("info", {})
    print(r["data"].split("/")[-1], "%.1fs" % r["seconds"],
          {k: i.get(k) for k in ("phase", "budget_s", "dec_reserve_s", "fm_kinds", "t_probe",
                                  "nest_pfnf", "nest_icl", "nest_pfn", "cover_pfnf", "cover_icl",
                                  "cover_pfn", "err_pfnf", "err_icl", "err_pfn", "timings",
                                  "weights_mean")})
PY
