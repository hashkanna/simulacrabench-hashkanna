"""Per-item cost of each in-context expert at several ensemble sizes, on the real
instrument shapes (sandbox data) with the final-phase OOF plan, for the
scheduler's cost model (FM_REL_COST, NEST_SCALE in main.py).

    python fm_bench.py --sub SUBDIR --data DIR [DIR ...] --phase 2 --items 6 --out bench.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as ev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--phase", type=int, default=2)
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--kinds", default="pfnf,icl,pfn")
    ap.add_argument("--nest", default="1,2,4")
    ap.add_argument("--kit", default=ev.DEFAULT_KIT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    m = ev.load_submission(args.sub)
    device = m._device()
    rows = []
    for d in args.data:
        prep = ev.prepare(d, args.phase, args.kit)
        inst = m.Instrument(prep["frame"].copy(), prep["schema"])
        X, cards, nominal = inst.given_matrix()
        Xf = X.astype(np.float32)
        extra = inst.numeric_extras()
        if extra is not None:
            Xf = np.concatenate([Xf, extra.astype(np.float32)], axis=1)
        cat_idx = [j for j, nom in enumerate(nominal) if nom]
        Xv, Xh = Xf[inst.vis], Xf[inst.hid]
        nv = len(inst.vis)
        plan, _ = m._fm_plan(nv, "final", m.SEED + 7)
        rng = np.random.default_rng(0)
        items = [str(t) for t in rng.choice(inst.targets, size=min(args.items, len(inst.targets)),
                                            replace=False)]
        name = os.path.basename(os.path.normpath(d))
        for kind in args.kinds.split(","):
            for e in (int(x) for x in args.nest.split(",")):
                fm = m._FM(kind, e, device)
                if not fm.available():
                    print(name, kind, e, "unavailable", flush=True)
                    continue
                t = items[0]
                t0 = time.monotonic()
                fm.run_item(Xv, inst.codes[t][inst.vis], Xh, plan, inst.K[t], cat_idx)
                warm = time.monotonic() - t0
                secs = []
                for t in items:
                    t0 = time.monotonic()
                    fm.run_item(Xv, inst.codes[t][inst.vis], Xh, plan, inst.K[t], cat_idx)
                    secs.append(time.monotonic() - t0)
                row = dict(data=name, kind=kind, n_est=e, nv=nv, nh=len(inst.hid),
                           n_feat=int(Xf.shape[1]), items=items,
                           K=[int(inst.K[t]) for t in items], secs=[round(s, 3) for s in secs],
                           mean=round(float(np.mean(secs)), 3), warm=round(warm, 2))
                rows.append(row)
                print("%-12s %-5s n_est=%d  mean %.2fs/item  (warm-up %.1fs, K=%s)"
                      % (name, kind, e, row["mean"], warm, row["K"]), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
