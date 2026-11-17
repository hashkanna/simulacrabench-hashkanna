#!/bin/bash
# Validate a candidate that takes full TabPFN-3.5 via models.txt. The platform's
# build-time snapshot_download is replicated in ~/work/claude-sim/hubcache.
# usage: rehearse.sh VERSION   (builds from claude/sub/v5)
cd $HOME/work/claude-sim
export HF_HUB_CACHE=$HOME/work/claude-sim/hubcache
VER=${1:-v5g}
PY=./venvchk/bin/python
$PY claude/lab/build_zip.py --sub claude/sub/v5 --weights weights --hub-pfn --out dist/claude-$VER-dev | tail -1
$PY claude/lab/build_zip.py --sub claude/sub/v5 --weights weights --hub-pfn --final-path-test --out dist/claude-$VER-finalpath | tail -1
echo "== final dry run (phase 2, 3600 s, pfn from HF cache)"; date
claude/lab/dry_run.sh $HOME/work/claude-sim/dist/claude-$VER-dev 2 /tmp/claude_dry_final_$VER.json
echo "== dev dry run (phase 1, 900 s)"; date
claude/lab/dry_run.sh $HOME/work/claude-sim/dist/claude-$VER-dev 1 /tmp/claude_dry_dev_$VER.json
echo "== rehearsal build (final path under the dev budget, phase 1)"; date
claude/lab/dry_run.sh $HOME/work/claude-sim/dist/claude-$VER-finalpath 1 /tmp/claude_dry_fp_$VER.json
echo "== official checks"; date
claude/lab/official_check.sh $HOME/work/claude-sim/dist/claude-$VER-dev $HOME/work/claude-sim/venvchk/bin/python "1 2" 2>&1 | grep -E "==|PASS|FAIL|OK:|Error"
echo "== rehearsal done"; date
