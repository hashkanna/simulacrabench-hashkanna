"""UNHCR-shaped proxy from the public REACH Iraq CCCM camp profiling VII
dataset (HDX). Household survey of displaced households: governorate/camp,
household composition counts, head-of-household profile (GIVEN); move
intentions, services, select-multiple items with skip logic (PREDICT).

Gates are inferred from the data (as the organisers did for UNHCR): for a
column with not-asked cells, the parent whose levels best explain who was asked.
A synthetic `wave` (60/40) imposes UNHCR's documented recording rule: in wave 2
unselected select-multiple options are blank (NA_GATED) instead of "0".

python proxy_reach.py --pkl raw/reach_iraq_cccm.pkl --out data/proxy/reach --seeds 1 2
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

COUNT_BANDS = ["0", "1", "2", "3", "4", "5", "6+"]


def band_count(x):
    x = pd.to_numeric(x, errors="coerce")
    out = np.where(x.isna(), "Not recorded", np.where(x >= 6, "6+", x.fillna(0).clip(0, 6).astype(int).astype(str)))
    return out


def norm(v):
    if isinstance(v, bool) or isinstance(v, np.bool_):
        return "1" if v else "0"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()[:60]


def build(df, out, seed, split=(0.80, 0.06, 0.14)):
    rng = np.random.default_rng(seed)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    n = len(df)
    items, cols = {}, {}
    wave = np.where(rng.random(n) < 0.6, "1", "2")
    cols["wave"] = wave
    items["wave"] = {"question": "Survey round.", "class": "GIVEN", "values": ["1", "2"]}

    def add_given(name, values_series, q):
        s = pd.Series(values_series, dtype=object).fillna("Not recorded").map(norm)
        vals = sorted(s.unique().tolist(), key=lambda v: (len(v), v))
        cols[name] = s.to_numpy(dtype=object)
        items[name] = {"question": q, "class": "GIVEN", "values": vals}

    add_given("governorate", df["governorate"], "Governorate of the camp")
    add_given("camp", df["camp_name"], "Camp name")
    add_given("origin", df["governorate of origin"], "Governorate of origin")
    demo = [c for c in df.columns if c.startswith("demographics/")]
    for c in demo:
        name = c.split("/")[1]
        s = band_count(df[c])
        cols[name] = s
        items[name] = {"question": "Household members: " + name, "class": "GIVEN",
                       "values": COUNT_BANDS + (["Not recorded"] if (s == "Not recorded").any() else [])}
    add_given("hoh_gender", df["head of household gender"], "Head of household gender")
    age = pd.to_numeric(df["head of household age"], errors="coerce")
    ab = pd.cut(age, [0, 24, 34, 44, 54, 64, 200], labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
    add_given("hoh_age", ab.astype(object), "Head of household age")
    add_given("marital", df["marital_status"], "Head of household marital status")

    used = set(["governorate", "camp_name", "governorate of origin", "head of household gender",
                "head of household age", "marital_status"] + demo)
    skip_pat = re.compile(r"(other|text|date|today|calculation|tent_age|camp_name|confirmed|__)", re.I)
    candidates = []
    for c in df.columns:
        if c in used or skip_pat.search(c):
            continue
        s = df[c]
        vals = s.dropna().map(norm)
        nu = vals.nunique()
        if nu < 2 or nu > 14 or s.notna().mean() < 0.03:
            continue
        candidates.append(c)
    # Decide gates: asked = notna
    pred_names = []
    for c in candidates:
        s = df[c]
        asked = s.notna().to_numpy()
        name = re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")[:48]
        if name in items:
            name = name + "_x"
        gate = None
        if not asked.all():
            best = (0.0, None, None)
            for p in ["hoh_gender", "marital", "governorate"] + pred_names[-80:]:
                pv = cols[p]
                levels = pd.unique(pv)
                if len(levels) > 15:
                    continue
                opens = [v for v in levels if asked[pv == v].mean() >= 0.5]
                if not opens or len(opens) == len(levels):
                    continue
                agree = (asked == np.isin(pv, opens)).mean()
                if agree > best[0] + 1e-9:
                    best = (agree, p, opens)
            if best[1] is not None and best[0] >= 0.75:
                gate = {"parent": best[1], "observed_if": [str(v) for v in best[2] if v != "NA_GATED"]}
                if not gate["observed_if"]:
                    gate = None
        vals = s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else norm(v))
        if gate is None:
            vals = vals.fillna("Not recorded")
        else:
            vals = vals.fillna("NA_GATED")
        arr = vals.to_numpy(dtype=object)
        is_multi = set(pd.unique(arr)) <= {"0", "1", "NA_GATED", "Not recorded"} and "/" in c
        if is_multi and gate is not None:
            arr = np.where((wave == "2") & (arr == "0"), "NA_GATED", arr)
        levels = sorted([v for v in pd.unique(arr) if v != "NA_GATED"], key=lambda v: (len(v), v))
        if len(levels) < 1:
            continue
        cols[name] = arr
        rec = {"question": c[:120], "class": "PREDICT", "values": levels}
        if gate is not None:
            rec["gate"] = gate
        items[name] = rec
        pred_names.append(name)
    frame = pd.DataFrame(cols)
    n_train = int(round(split[0] * n))
    n_dev = int(round(split[1] * n))
    n_test = n - n_train - n_dev
    frame.insert(0, "respondent_id", ["H%06d" % i for i in range(1, n + 1)])
    frame["role"] = ["TRAIN"] * n_train + ["DEV"] * n_dev + ["TEST"] * n_test
    schema = {"dataset": {"n_rows": n, "version": "proxy-reach",
                          "description": "Proxy from REACH Iraq CCCM camp profiling VII"},
              "items": items, "split": {"n_train": n_train, "n_dev": n_dev, "n_test": n_test}}
    os.makedirs(out, exist_ok=True)
    frame.to_parquet(os.path.join(out, "respondents.parquet"), index=False)
    with open(os.path.join(out, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=1)
    n_given = sum(v["class"] == "GIVEN" for v in items.values())
    n_pred = sum(v["class"] == "PREDICT" for v in items.values())
    n_gated = sum(1 for v in items.values() if v.get("gate"))
    print(out, n, "given", n_given, "predict", n_pred, "gated", n_gated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    args = ap.parse_args()
    df = pd.read_pickle(args.pkl)
    for seed in args.seeds:
        build(df, "%s_s%d" % (args.out, seed), seed)


if __name__ == "__main__":
    main()
