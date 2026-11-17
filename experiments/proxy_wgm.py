"""Build UNICEF-shaped proxy instruments from the public Wellcome Global
Monitor 2018 microdata (https://wellcome.org/.../wgm2018-dataset-crosstabs-all-countries.xlsx).

Writes DATA_DIR/{respondents.parquet, schema.json} in the competition format,
with roles TRAIN/DEV/TEST in the UNICEF proportions (0.75/0.075/0.175).

python proxy_wgm.py --xlsx raw/wgm2018.xlsx --out data/proxy/wgm_a --seed 1 --variant a
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

LAB = {
    "AgeCategories": ("Age cohort", {1: "15-29", 2: "30-49", 3: "50+"}),
    "Gender": ("Gender", {1: "Male", 2: "Female"}),
    "Education": ("Education", {1: "Primary", 2: "Secondary", 3: "Tertiary"}),
    "Urban_Rural": ("Area type", {1: "Rural/small town", 2: "City/suburb"}),
    "Household_Income": ("Income quintile", {1: "Poorest 20%", 2: "Second 20%", 3: "Middle 20%",
                                             4: "Fourth 20%", 5: "Top 20%"}),
    "D1": ("Religion", {1: "Named a specific religion", 2: "Secular/Non-religious", 99: "DK/Refused"}),
    "Q27": ("Do you have any children?", {1: "Yes", 2: "No", 3: "Yes, no longer living", 4: "DK", 5: "Refused"}),
    "EMP_2010": ("Employment", {1: "FT employer", 2: "FT self", 3: "PT not want FT", 4: "Unemployed",
                                5: "PT want FT", 6: "Out of workforce"}),
    "Subjective_Income": ("Feelings about income", {1: "Comfortable", 2: "Getting by", 3: "Difficult"}),
}
AGREE5 = {1: "Strongly agree", 2: "Somewhat agree", 3: "Neither", 4: "Somewhat disagree",
          5: "Strongly disagree", 99: "DK/Refused"}
TRUST4 = {1: "A lot", 2: "Some", 3: "Not much", 4: "Not at all", 98: "DK", 99: "Refused"}
YESNO = {1: "Yes", 2: "No", 98: "DK", 99: "Refused"}
TARGETS = {
    "Q20": ("Trust most for medical advice", {1: "Family/friends", 2: "Religious leader", 3: "Doctor/nurse",
                                               4: "Famous person", 5: "Traditional healer", 97: "None/someone else",
                                               98: "DK", 99: "Refused"}),
    "Q21": ("Trust medical advice from government", TRUST4),
    "Q22": ("Trust medical advice from medical workers", TRUST4),
    "Q23": ("Heard of vaccines", YESNO),
    "Q24": ("Vaccines are important for children", AGREE5),
    "Q25": ("Vaccines are safe", AGREE5),
    "Q26": ("Vaccines are effective", AGREE5),
    "Q28": ("Children ever received a vaccine", YESNO),
    "Q29": ("Science disagreed with religion", YESNO),
    "Q30": ("Science vs religion: which do you believe", {1: "Science", 2: "Religion", 97: "It depends",
                                                         98: "DK", 99: "Refused"}),
    "Q11E": ("Trust doctors and nurses", TRUST4),
    "Q11G": ("Trust traditional healers", TRUST4),
    "Q12": ("Trust science", TRUST4),
    "Q17": ("Science benefits people like you", YESNO),
    "Q18": ("Science improves life next generation", YESNO),
    "Q7": ("Sought health info past 30 days", YESNO),
}
GATES = {  # child: (parent, observed_if codes)
    "Q24": ("Q23", [1]), "Q25": ("Q23", [1]), "Q26": ("Q23", [1]),
    "Q28": ("Q27", [1]), "Q29": ("D1", [1]), "Q30": ("Q29", [1]),
}
# Country codes from the data dictionary (WP5) for low/middle income countries.
COUNTRY_SETS = {
    "a": [33, 2, 36, 35, 9, 31, 11, 40, 41, 45, 48, 34, 10, 44, 42, 43, 37, 3, 4, 6],
    "b": [32, 26, 8, 5, 39, 30, 28, 29, 38, 1, 12, 13, 14, 17, 18, 19, 20, 21, 22, 24],
}


def load(xlsx):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    ws = wb["Full dataset"]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    data = pd.DataFrame(list(rows), columns=header)
    return data


def build(df, out, seed, variant, n_rows=None, split=(0.75, 0.075, 0.175)):
    countries = COUNTRY_SETS[variant]
    wp5 = pd.to_numeric(df["WP5"], errors="coerce")
    df = df[wp5.isin(countries)].copy()
    df["WP5"] = pd.to_numeric(df["WP5"], errors="coerce").astype(int)
    rng = np.random.default_rng(seed)
    if n_rows and len(df) > n_rows:
        df = df.iloc[rng.choice(len(df), n_rows, replace=False)]
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    items = {}
    cols = {}
    cname = {c: "C%03d" % c for c in countries}
    cols["Country"] = df["WP5"].map(cname).astype(object)
    items["Country"] = {"question": "Country", "class": "GIVEN",
                        "values": sorted(cname.values())}
    for v in ["AgeCategories", "Gender", "Education", "Urban_Rural", "Household_Income", "D1", "Q27"]:
        q, lab = LAB[v]
        x = pd.to_numeric(df[v], errors="coerce")
        s = x.map(lab)
        other = "Other/missing"
        if s.isna().any():
            s = s.fillna(other)
            values = list(lab.values()) + [other]
        else:
            values = list(lab.values())
        cols[v] = s.astype(object)
        items[v] = {"question": q, "class": "GIVEN", "values": values}
    for v, (q, lab) in TARGETS.items():
        x = pd.to_numeric(df[v], errors="coerce")
        s = x.map(lab)
        values = list(dict.fromkeys(lab.values()))
        gate = None
        if v in GATES:
            parent, opens = GATES[v]
            plab = LAB.get(parent, TARGETS.get(parent))[1]
            gate = {"parent": parent, "observed_if": [plab[c] for c in opens]}
            s = s.fillna("NA_GATED")
        else:
            other = "Not recorded"
            if s.isna().any():
                s = s.fillna(other)
                values = values + [other]
        cols[v] = s.astype(object)
        rec = {"question": q, "class": "PREDICT", "values": values}
        if gate:
            rec["gate"] = gate
        items[v] = rec
    frame = pd.DataFrame(cols)
    # Drop values not in declared values (safety)
    for v, rec in items.items():
        allowed = set(rec["values"]) | ({"NA_GATED"} if rec.get("gate") else set())
        bad = ~frame[v].isin(allowed)
        if bad.any():
            frame.loc[bad, v] = rec["values"][-1]
    n = len(frame)
    n_train = int(round(split[0] * n))
    n_dev = int(round(split[1] * n))
    n_test = n - n_train - n_dev
    roles = np.array(["TRAIN"] * n_train + ["DEV"] * n_dev + ["TEST"] * n_test, dtype=object)
    frame.insert(0, "respondent_id", ["P%06d" % i for i in range(1, n + 1)])
    frame["role"] = roles
    schema = {"dataset": {"n_rows": n, "version": "proxy-wgm-" + variant,
                          "description": "Proxy instrument from WGM 2018 (variant %s)" % variant},
              "items": items,
              "split": {"n_train": n_train, "n_dev": n_dev, "n_test": n_test}}
    os.makedirs(out, exist_ok=True)
    frame.to_parquet(os.path.join(out, "respondents.parquet"), index=False)
    with open(os.path.join(out, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=1)
    print(out, n, {k: (v["class"], len(v["values"])) for k, v in items.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--variant", default="a")
    ap.add_argument("--n-rows", type=int, default=None)
    args = ap.parse_args()
    cache = args.xlsx + ".parquet"
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
    else:
        df = load(args.xlsx)
        df.astype(str).to_parquet(cache)
        df = pd.read_parquet(cache)
    for seed in args.seeds:
        build(df, "%s_s%d" % (args.out, seed), seed, args.variant, args.n_rows)


if __name__ == "__main__":
    main()
