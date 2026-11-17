"""Panel-twin mechanism lab: add "second-round" copies of some respondents to a
proxy instrument, the way a panel re-interview would appear in a file whose
cross-round key was removed.

    python twin_proxy.py SRC_DIR OUT_DIR [--frac 0.64] [--keep 0.8]
        [--given-noise 0.05] [--seed 0]

A share `frac` of respondents gets a copy. In the copy each PREDICT answer is
kept with probability `keep`, otherwise replaced by the same item's answer from
a random other respondent (marginals preserved). Each GIVEN value except `wave`
is replaced the same way with probability `given_noise`. Originals and copies are
pooled, shuffled and re-assigned TRAIN/DEV/TEST in the source split's proportions,
so twins straddle the visible/scored boundary as they would under a row-level
split. Real answers, one imposed mechanism: the lab asks whether an expert can
use a twin, not whether the competition files have twins.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--frac", type=float, default=0.64)
    ap.add_argument("--keep", type=float, default=0.8)
    ap.add_argument("--given-noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    schema = json.load(open(os.path.join(args.src, "schema.json")))
    df = pd.read_parquet(os.path.join(args.src, "respondents.parquet"))
    items = schema["items"]
    given = [k for k, v in items.items() if v["class"] == "GIVEN" and k != "wave" and k in df]
    predict = [k for k, v in items.items() if v["class"] == "PREDICT" and k in df]
    n = len(df)

    pick = rng.choice(n, size=int(round(args.frac * n)), replace=False)
    twins = df.iloc[pick].copy().reset_index(drop=True)
    twins["respondent_id"] = ["T" + str(r) for r in twins["respondent_id"]]
    m = len(twins)

    def resample(cols, prob):
        for c in cols:
            swap = rng.random(m) < prob
            if swap.any():
                donors = rng.integers(0, n, size=int(swap.sum()))
                twins.loc[swap, c] = df[c].to_numpy()[donors]

    resample(predict, 1.0 - args.keep)
    resample(given, args.given_noise)

    both = pd.concat([df, twins], ignore_index=True)
    both = both.iloc[rng.permutation(len(both))].reset_index(drop=True)
    split = schema["split"]
    total = split["n_train"] + split["n_dev"] + split["n_test"]
    n_all = len(both)
    n_dev = int(round(n_all * split["n_dev"] / total))
    n_test = int(round(n_all * split["n_test"] / total))
    n_train = n_all - n_dev - n_test
    both["role"] = ["TRAIN"] * n_train + ["DEV"] * n_dev + ["TEST"] * n_test

    schema = dict(schema)
    schema["split"] = {"n_train": n_train, "n_dev": n_dev, "n_test": n_test}
    schema["dataset"] = dict(schema["dataset"])
    schema["dataset"]["n_rows"] = n_all
    schema["dataset"]["description"] = (
        schema["dataset"].get("description", "") + " | panel-twin lab: frac %.2f keep %.2f "
        "given-noise %.2f seed %d" % (args.frac, args.keep, args.given_noise, args.seed))
    os.makedirs(args.out, exist_ok=True)
    both.to_parquet(os.path.join(args.out, "respondents.parquet"), index=False)
    with open(os.path.join(args.out, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=1)
    # How many scored rows have their twin among the visible rows.
    base = both["respondent_id"].astype(str).str.lstrip("T")
    vis = set(base[both["role"] == "TRAIN"])
    copy = both["respondent_id"].astype(str).str.startswith("T")
    has_twin = base.duplicated(keep=False)
    for role in ("DEV", "TEST"):
        r = both["role"] == role
        twin_visible = sum(1 for b, t, rr in zip(base, has_twin, r) if rr and t and b in vis)
        print("%s rows %d, with a twin %d, twin visible %d"
              % (role, int(r.sum()), int((r & has_twin).sum()), twin_visible))
    print("rows %d (copies %d)  split %s" % (n_all, int(copy.sum()), schema["split"]))


if __name__ == "__main__":
    main()
