"""Make a smaller copy of a proxy instrument keeping a random subset of PREDICT
items (plus every gate ancestor of a kept item).

python subset_items.py SRC DST N_ITEMS [SEED]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

src, dst, n_keep = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
schema = json.load(open(os.path.join(src, "schema.json")))
df = pd.read_parquet(os.path.join(src, "respondents.parquet"))
items = schema["items"]
pred = [k for k, v in items.items() if v["class"] == "PREDICT"]
rng = np.random.default_rng(seed)
keep = set(rng.choice(pred, min(n_keep, len(pred)), replace=False).tolist())
changed = True
while changed:
    changed = False
    for k in list(keep):
        g = items[k].get("gate")
        if g and g["parent"] in items and items[g["parent"]]["class"] == "PREDICT" and g["parent"] not in keep:
            keep.add(g["parent"])
            changed = True
new_items = {k: v for k, v in items.items() if v["class"] != "PREDICT" or k in keep}
schema["items"] = new_items
cols = ["respondent_id"] + [k for k in new_items] + ["role"]
os.makedirs(dst, exist_ok=True)
df[cols].to_parquet(os.path.join(dst, "respondents.parquet"), index=False)
json.dump(schema, open(os.path.join(dst, "schema.json"), "w"), indent=1)
print(dst, "kept", len(keep), "predict items")
