"""Split a twin-lab comparison by whether the scored respondent's twin is visible.

    python twin_breakdown.py TWIN_DATA_DIR RESULT_DIR_A RESULT_DIR_B [...]

Reads <dataset>_p2.npz (per-cell floored log p, respondent ids) from each result
directory, as evaluate.py --save-logp writes them, and reports, for scored rows
whose twin is among the visible (TRAIN + DEV) rows and for the rest, each
configuration's mean log p per cell and its paired difference from the first
configuration in skill units (difference / marginal reference loss), with a
respondent-cluster bootstrap standard error.
"""
import os
import sys

import numpy as np
import pandas as pd


def base_id(rid):
    rid = str(rid)
    return rid[1:] if rid.startswith("T") else rid


def main():
    data, dirs = sys.argv[1], sys.argv[2:]
    df = pd.read_parquet(os.path.join(data, "respondents.parquet"),
                         columns=["respondent_id", "role"])
    visible = {base_id(r) for r, role in zip(df["respondent_id"], df["role"])
               if role in ("TRAIN", "DEV")}
    name = os.path.basename(os.path.normpath(data)) + "_p2.npz"
    runs = []
    for d in dirs:
        z = np.load(os.path.join(d, name), allow_pickle=True)
        runs.append((os.path.basename(os.path.normpath(d)), z["logp"], z["rid"],
                     float(z["reference"])))
    rid = runs[0][2]
    for _, _, r, _ in runs[1:]:
        assert (r == rid).all(), "cell order differs between runs"
    ref_loss = -runs[0][3] if runs[0][3] < 0 else runs[0][3]
    twin = np.array([base_id(r) in visible for r in rid])
    groups = {"twin visible": twin, "no visible twin": ~twin, "all": np.ones_like(twin)}
    rng = np.random.default_rng(0)
    for gname, mask in groups.items():
        ids = np.unique(rid[mask])
        index = {r: i for i, r in enumerate(ids)}
        cid = np.array([index[r] for r in rid[mask]])
        print("== %s: %d respondents, %d cells" % (gname, len(ids), int(mask.sum())))
        base = runs[0][1][mask]
        for label, logp, _, _ in runs:
            diff = (logp[mask] - base) / ref_loss
            per = np.bincount(cid, weights=diff, minlength=len(ids))
            cnt = np.bincount(cid, minlength=len(ids))
            boots = []
            for _ in range(400):
                pick = rng.integers(0, len(ids), len(ids))
                boots.append(per[pick].sum() / cnt[pick].sum())
            print("  %-16s mean logp %.5f   d(skill) vs %s %+.5f (se %.5f)"
                  % (label, logp[mask].mean(), runs[0][0], diff.mean(), np.std(boots)))


if __name__ == "__main__":
    main()
