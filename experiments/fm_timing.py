"""Timing probe for tabular foundation models on competition-shaped data.

python fm_timing.py --data DATA_DIR --weights WEIGHTS_DIR [--items 6]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch


def load(data_dir):
    schema = json.load(open(os.path.join(data_dir, "schema.json")))
    df = pd.read_parquet(os.path.join(data_dir, "respondents.parquet"))
    items = schema["items"]
    given = [k for k, v in items.items() if v["class"] == "GIVEN"]
    pred = [k for k, v in items.items() if v["class"] == "PREDICT"]
    X = np.column_stack([
        pd.Categorical(df[g], categories=items[g]["values"]).codes for g in given
    ]).astype(np.float32)
    Y = {}
    for p in pred:
        opts = list(items[p]["values"]) + (["NA_GATED"] if items[p].get("gate") else [])
        Y[p] = pd.Categorical(df[p], categories=opts).codes.astype(int)
    return X, Y, df["role"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--models", default="pfn35fast1,pfn35fast4,pfn35_1,icl1,icl4")
    args = ap.parse_args()
    X, Y, role = load(args.data)
    train = np.flatnonzero(role == "TRAIN")
    dev = np.flatnonzero(role == "DEV")
    rng = np.random.default_rng(0)
    perm = rng.permutation(train)
    fold = perm[: len(perm) // 5]
    ctx = perm[len(perm) // 5:]
    query = np.concatenate([fold, dev])
    names = [k for k in Y if len(np.unique(Y[k][train])) >= 2][: args.items]
    print("n_ctx", len(ctx), "n_query", len(query), "features", X.shape[1], flush=True)
    from tabpfn import TabPFNClassifier
    from tabicl import TabICLClassifier
    W = args.weights
    makers = {
        "pfn35fast1": lambda: TabPFNClassifier(device="cuda", n_estimators=1, random_state=0,
                                               model_path=f"{W}/tabpfn35_tabpfn-v3.5-fast-20260909.safetensors",
                                               ignore_pretraining_limits=True),
        "pfn35fast4": lambda: TabPFNClassifier(device="cuda", n_estimators=4, random_state=0,
                                               model_path=f"{W}/tabpfn35_tabpfn-v3.5-fast-20260909.safetensors",
                                               ignore_pretraining_limits=True),
        "pfn35_1": lambda: TabPFNClassifier(device="cuda", n_estimators=1, random_state=0,
                                            model_path=f"{W}/tabpfn35_tabpfn-v3.5-20260909.safetensors",
                                            ignore_pretraining_limits=True),
        "icl1": lambda: TabICLClassifier(device="cuda", n_estimators=1, random_state=0,
                                         model_path=f"{W}/tabicl-classifier-v2-20260212.ckpt",
                                         allow_auto_download=False),
        "icl4": lambda: TabICLClassifier(device="cuda", n_estimators=4, random_state=0,
                                         model_path=f"{W}/tabicl-classifier-v2-20260212.ckpt",
                                         allow_auto_download=False),
    }
    for key in args.models.split(","):
        times = []
        losses = []
        for j, name in enumerate(names):
            y = Y[name]
            torch.cuda.synchronize()
            t0 = time.time()
            m = makers[key]()
            m.fit(X[ctx], y[ctx])
            p = m.predict_proba(X[query])
            torch.cuda.synchronize()
            dt = time.time() - t0
            cls = list(m.classes_)
            yq = y[query]
            idx = np.array([cls.index(v) if v in cls else -1 for v in yq])
            pv = np.where(idx >= 0, p[np.arange(len(yq)), np.maximum(idx, 0)], 1e-3)
            losses.append(float(-np.log(np.maximum(pv, 1e-3)).mean()))
            times.append(dt)
            print(f"  {key} item {j} {name[:30]} K={len(cls)} {dt:.2f}s loss {losses[-1]:.4f}", flush=True)
        print(f"{key}: mean {np.mean(times[1:]) if len(times) > 1 else times[0]:.2f}s/item "
              f"(first {times[0]:.2f}s) mean loss {np.mean(losses):.4f}", flush=True)


if __name__ == "__main__":
    main()
