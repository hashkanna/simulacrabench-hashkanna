"""Paired comparison of saved per-cell log scores between configurations.

python compare.py results/A results/B [results/C ...]

For every dataset file present in all directories, reports each config's skill
and the paired skill difference vs the first config with a respondent-cluster
bootstrap standard error (cells within a respondent are correlated).
"""
import glob
import json
import os
import sys

import numpy as np


def load(d):
    out = {}
    refs_npz = {}
    for f in glob.glob(os.path.join(d, "*.npz")):
        z = np.load(f, allow_pickle=True)
        out[os.path.basename(f)] = (z["logp"], z["rid"])
        if "reference" in z.files:
            key = os.path.basename(f).rsplit("_p", 1)[0]
            refs_npz[key] = float(z["reference"])
    summ = os.path.join(d, "summary.json")
    refs = {}
    if os.path.exists(summ):
        s = json.load(open(summ))
        for r in s["runs"]:
            refs[os.path.basename(os.path.normpath(r["data"]))] = r["reference"]
    refs.update(refs_npz)
    return out, refs


def cluster_se(diff, rid, draws=1000, seed=0):
    _, idx = np.unique(rid, return_inverse=True)
    tot = np.bincount(idx, weights=diff)
    cnt = np.bincount(idx).astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(tot), size=(draws, len(tot)))
    means = tot[picks].sum(1) / cnt[picks].sum(1)
    return means.std(ddof=1)


def main():
    dirs = sys.argv[1:]
    data = [load(d) for d in dirs]
    common = sorted(set.intersection(*[set(d[0]) for d in data]))
    print("%-22s" % "dataset" + "".join("%27s" % os.path.basename(d.rstrip("/"))[:25] for d in dirs))
    agg = np.zeros(len(dirs))
    for name in common:
        base_lp, rid = data[0][0][name]
        key = name.rsplit("_p", 1)[0]
        R = data[0][1].get(key)
        row = "%-22s" % name[:22]
        for i, (d, refs) in enumerate(data):
            lp, rid_i = d[name]
            assert len(lp) == len(base_lp) and (rid_i == rid).all()
            skill = 1 + lp.mean() / R if R else lp.mean()
            if i == 0:
                row += "%27s" % ("%.5f" % skill)
            else:
                diff = (lp - base_lp) / (R or 1.0)
                se = cluster_se(diff, rid)
                row += "%27s" % ("%.4f %+.5f(%.5f)" % (skill, diff.mean(), se))
                agg[i] += diff.mean()
        print(row)
    if len(common) > 1:
        print("mean diff vs first:", " ".join("%+.5f" % (a / len(common)) for a in agg[1:]))


if __name__ == "__main__":
    main()
