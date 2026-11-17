"""World-Bank-shaped proxy from the public OpenPsychometrics RIASEC dataset
(http://openpsychometrics.org/_rawdata/RIASEC_data12Dec2018.zip).

GIVEN: demographics (9). PREDICT: binned RIASEC / TIPI Big-Five subscale means,
vocabulary score, elapsed-time bins, raw items, and a few gated items with GIVEN
parents. Sizes follow the WB instrument (6,927 rows; 0.75/0.075/0.175 roles).

python proxy_riasec.py --csv raw/riasec/RIASEC_data12Dec2018/data.csv --out data/proxy/riasec --seeds 1 2
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

BINS7 = ["<2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0", "Not answered"]


def bin_scale(x, edges, labels, missing):
    out = np.full(len(x), missing, dtype=object)
    ok = np.isfinite(x)
    idx = np.digitize(x[ok], edges)
    out[ok] = np.asarray(labels, dtype=object)[idx]
    return out


def build(df, out, seed, n_rows=6927, split=(0.75, 0.075, 0.175)):
    rng = np.random.default_rng(seed)
    df = df.iloc[rng.choice(len(df), n_rows, replace=False)].reset_index(drop=True)
    items, cols = {}, {}

    def given(name, q, series, values):
        cols[name] = pd.Series(series, dtype=object)
        items[name] = {"question": q, "class": "GIVEN", "values": values}

    def target(name, q, series, values, gate=None):
        s = pd.Series(series, dtype=object)
        if gate:
            values = [v for v in values if v != "NA_GATED"]
        cols[name] = s
        rec = {"question": q, "class": "PREDICT", "values": values}
        if gate:
            rec["gate"] = gate
        items[name] = rec

    age = pd.to_numeric(df["age"], errors="coerce").to_numpy()
    age_lab = ["13-17", "18-21", "22-25", "26-30", "31-40", "41+", "Not answered"]
    given("age", "Age", bin_scale(np.where((age >= 13) & (age < 100), age, np.nan),
                                  [18, 22, 26, 31, 41], age_lab[:-1], "Not answered"), age_lab)
    maps = {
        "gender": ("Gender", {1: "Male", 2: "Female", 3: "Other"}),
        "education": ("Education", {1: "Less than high school", 2: "High school", 3: "University", 4: "Graduate"}),
        "urban": ("Area", {1: "Rural", 2: "Suburban", 3: "Urban"}),
        "engnat": ("Native English", {1: "Yes", 2: "No"}),
        "religion": ("Religion", {1: "Agnostic", 2: "Atheist", 3: "Buddhist", 4: "Christian (Catholic)",
                                  5: "Christian (Mormon)", 6: "Christian (Protestant)", 7: "Christian (Other)",
                                  8: "Hindu", 9: "Jewish", 10: "Muslim", 11: "Sikh", 12: "Other"}),
        "race": ("Race", {1: "Asian", 2: "Arab", 3: "Black", 4: "Indigenous", 5: "White", 6: "Other"}),
        "married": ("Marital status", {1: "Never married", 2: "Currently married", 3: "Previously married"}),
    }
    for name, (q, lab) in maps.items():
        x = pd.to_numeric(df[name], errors="coerce")
        s = x.map(lab).fillna("Not answered")
        given(name, q, s.to_numpy(), list(lab.values()) + ["Not answered"])
    cty = df["country"].astype(str)
    top = ["US", "GB", "CA", "AU", "IN", "PH", "MY"]
    given("country", "Country", np.where(cty.isin(top), cty, "Other"), top + ["Other"])

    # RIASEC subscales
    for code, label in zip("RIASEC", ["Realistic", "Investigative", "Artistic", "Social",
                                      "Enterprising", "Conventional"]):
        m = df[[f"{code}{i}" for i in range(1, 9)]].apply(pd.to_numeric, errors="coerce")
        m = m.where(m > 0)
        score = m.mean(axis=1).where(m.notna().sum(axis=1) >= 6).to_numpy()
        target(f"sc_{label.lower()}", f"RIASEC {label} subscale mean (1-5)",
               bin_scale(score, [2.5, 3.0, 3.5, 4.0, 4.5], BINS7[:-1], "Not answered"), BINS7)
    # TIPI Big Five (1-7) -> rescaled 1-5
    tipi = df[[f"TIPI{i}" for i in range(1, 11)]].apply(pd.to_numeric, errors="coerce")
    tipi = tipi.where(tipi > 0)
    rev = lambda c: 8 - tipi[c]
    big5 = {"extra": (tipi["TIPI1"], rev("TIPI6")), "agree": (rev("TIPI2"), tipi["TIPI7"]),
            "consc": (tipi["TIPI3"], rev("TIPI8")), "stability": (rev("TIPI4"), tipi["TIPI9"]),
            "open": (tipi["TIPI5"], rev("TIPI10"))}
    for name, (a, b) in big5.items():
        score = (((a + b) / 2.0 - 1) / 6.0 * 4 + 1).to_numpy()
        target(f"tipi_{name}", f"Big Five {name} (TIPI, rescaled 1-5)",
               bin_scale(score, [2.5, 3.0, 3.5, 4.0, 4.5], BINS7[:-1], "Not answered"), BINS7)
    # Vocabulary: real words known minus fake words claimed
    vcl = df[[f"VCL{i}" for i in range(1, 17)]].apply(pd.to_numeric, errors="coerce")
    fake = vcl[["VCL6", "VCL9", "VCL12"]].sum(axis=1)
    real = vcl.drop(columns=["VCL6", "VCL9", "VCL12"]).sum(axis=1)
    voc = (real - 2 * fake).to_numpy()
    voc_lab = ["<=4", "5-6", "7-8", "9-10", "11", "12", "13"]
    target("vocab", "Vocabulary check score", bin_scale(voc, [5, 7, 9, 11, 12, 13], voc_lab, "<=4"), voc_lab)
    # Elapsed times (log seconds) like trail-making bins
    for col, q in (("testelapse", "Test time"), ("surveyelapse", "Survey time")):
        t = pd.to_numeric(df[col], errors="coerce").to_numpy().astype(float)
        lt = np.log(np.clip(t, 1, None))
        qs = np.nanquantile(lt, [0.2, 0.45, 0.7, 0.9])
        lab = ["q1", "q2", "q3", "q4", "q5"]
        target(f"t_{col}", q + " (log seconds, banded)", bin_scale(lt, qs, lab, "q5"), lab)
    # Raw items: two per scale
    for code in "RIASEC":
        for i in (1, 5):
            x = pd.to_numeric(df[f"{code}{i}"], errors="coerce")
            s = x.map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}).fillna("Not answered")
            target(f"item_{code}{i}", f"Item {code}{i} (1=Dislike .. 5=Enjoy)", s.to_numpy(),
                   ["1", "2", "3", "4", "5", "Not answered"])
    # Other answers as targets
    voted = pd.to_numeric(df["voted"], errors="coerce").map({1: "Yes", 2: "No"}).fillna("Not answered")
    target("voted", "Voted in a national election in past year", voted.to_numpy(), ["Yes", "No", "Not answered"])
    orient = pd.to_numeric(df["orientation"], errors="coerce").map(
        {1: "Heterosexual", 2: "Bisexual", 3: "Homosexual", 4: "Asexual", 5: "Other"}).fillna("Not answered")
    target("orientation", "Orientation", orient.to_numpy(),
           ["Heterosexual", "Bisexual", "Homosexual", "Asexual", "Other", "Not answered"])
    fam = pd.to_numeric(df["familysize"], errors="coerce").to_numpy()
    fam_lab = ["0", "1", "2", "3", "4", "5", "6", "7+", "Not answered"]
    target("familysize", "Number of children your mother had", bin_scale(
        np.where((fam >= 0) & (fam < 50), fam, np.nan), [1, 2, 3, 4, 5, 6, 7], fam_lab[:-1],
        "Not answered"), fam_lab)
    hand = pd.to_numeric(df["hand"], errors="coerce").map({1: "Right", 2: "Left", 3: "Both"}).fillna("Not answered")
    target("hand", "Handedness", hand.to_numpy(), ["Right", "Left", "Both", "Not answered"])
    # Gated items with GIVEN parents (deterministic gates)
    edu = cols["education"].to_numpy()
    gate_open = np.isin(edu, ["University", "Graduate"])
    for code in "IS":
        x = pd.to_numeric(df[f"{code}3"], errors="coerce")
        s = x.map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}).fillna("Not answered").to_numpy()
        s = np.where(gate_open, s, "NA_GATED")
        target(f"gated_{code}3", f"Item {code}3 asked only of graduates", s,
               ["1", "2", "3", "4", "5", "Not answered"],
               gate={"parent": "education", "observed_if": ["University", "Graduate"]})
    frame = pd.DataFrame(cols)
    n = len(frame)
    n_train = int(round(split[0] * n))
    n_dev = int(round(split[1] * n))
    n_test = n - n_train - n_dev
    frame.insert(0, "respondent_id", ["P%06d" % i for i in range(1, n + 1)])
    frame["role"] = ["TRAIN"] * n_train + ["DEV"] * n_dev + ["TEST"] * n_test
    schema = {"dataset": {"n_rows": n, "version": "proxy-riasec",
                          "description": "Proxy instrument from OpenPsychometrics RIASEC"},
              "items": items, "split": {"n_train": n_train, "n_dev": n_dev, "n_test": n_test}}
    os.makedirs(out, exist_ok=True)
    frame.to_parquet(os.path.join(out, "respondents.parquet"), index=False)
    with open(os.path.join(out, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=1)
    print(out, n, sum(v["class"] == "PREDICT" for v in items.values()), "targets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--n-rows", type=int, default=6927)
    args = ap.parse_args()
    df = pd.read_csv(args.csv, sep="\t", low_memory=False)
    for seed in args.seeds:
        build(df, "%s_s%d" % (args.out, seed), seed, args.n_rows)


if __name__ == "__main__":
    main()
