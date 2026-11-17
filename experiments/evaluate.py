"""In-process evaluator that mirrors the official grader exactly.

Uses the official kit's own functions (load_schema, load_frames, sample_rows,
marginal_reference, floored, score) so the numbers are the grader's numbers,
but calls predict() in-process so many runs are cheap.

    python evaluate.py --sub ../sub/v5 --data DATA_DIR --phase 1 [--kit KIT]

DATA_DIR holds respondents.parquet and schema.json (as make_sandbox.py writes,
or as our proxy builders write).
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KIT = os.environ.get(
    "SIMULACRA_KIT", os.path.join(HERE, "..", "..", "..", "simulacra-kit"))


def _kit(kit):
    sys.path.insert(0, os.path.abspath(kit))
    import make_sandbox  # noqa: E402
    import score  # noqa: E402
    return make_sandbox, score


def load_submission(path, name="sub_main"):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "main.py"))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, path)
    spec.loader.exec_module(module)
    return module


class _VirtualClock:
    """Stands in for the submission's `time` module: monotonic() runs at real
    speed except that in-context expert calls are charged their cached compute
    seconds divided by `speed` (a faster or slower GPU than the one that filled
    the cache)."""

    def __init__(self, real_time, speed):
        self._t = real_time
        self.speed = speed
        self.offset = 0.0

    def monotonic(self):
        return self._t.monotonic() + self.offset

    def charge(self, real_elapsed, cost_seconds):
        self.offset += cost_seconds / self.speed - real_elapsed

    def __getattr__(self, name):
        return getattr(self._t, name)


def install_fm_cache(module, cache_dir, speed=None):
    """Lab only: memoise the in-context experts' per-item predictions on disk.

    The key hashes everything the prediction depends on (kind, n_estimators,
    nominal encoding, context/query matrices, labels, fold plan, K), so a hit is
    exactly what a fresh run would compute. The original compute seconds are
    stored and summed in module._LAB_FM_COST so configs can be costed without
    re-running them.
    """
    import hashlib
    os.makedirs(cache_dir, exist_ok=True)
    orig = module._FM.run_item
    cost = {}
    module._LAB_FM_COST = cost
    clock = None
    if speed:
        clock = _VirtualClock(time, speed)
        module.time = clock

    def digest(*arrays):
        m = hashlib.sha1()
        for a in arrays:
            a = np.ascontiguousarray(a)
            m.update(str(a.dtype).encode())
            m.update(str(a.shape).encode())
            m.update(a.tobytes())
        return m.hexdigest()

    def run_item(self, Xv, y, Xh, plan, K, cat_idx):
        parts = [Xv, np.asarray(y), Xh, np.asarray(cat_idx or [], np.int64),
                 np.array([K], np.int64)]
        for ctx, q in plan:
            parts += [np.asarray(ctx, np.int64), np.asarray(q, np.int64)]
        key = "%s_n%d_%s_%s" % (self.kind, self.n_estimators,
                                getattr(module, "FM_NOMINAL", "cat"), digest(*parts))
        path = os.path.join(cache_dir, key + ".npz")
        c = cost.setdefault("%s%d" % (self.kind, self.n_estimators), [0.0, 0, 0])
        t_real = time.monotonic()
        if os.path.exists(path):
            z = np.load(path)
            c[0] += float(z["sec"])
            c[1] += 1
            if clock is not None:
                clock.charge(time.monotonic() - t_real, float(z["sec"]))
            return z["oof"], z["hid"]
        t0 = time.time()
        oof, hid = orig(self, Xv, y, Xh, plan, K, cat_idx)
        sec = time.time() - t0
        if clock is not None:
            clock.charge(time.monotonic() - t_real, sec)
        tmp = path[:-4] + ".tmp%d.npz" % os.getpid()
        np.savez(tmp, oof=oof, hid=hid, sec=np.array(sec))
        os.replace(tmp, path)
        c[0] += sec
        c[1] += 1
        c[2] += 1
        return oof, hid

    module._FM.run_item = run_item


def prepare(data_dir, phase, kit=DEFAULT_KIT, schema_path=None):
    make_sandbox, score = _kit(kit)
    config = make_sandbox.load_config(os.path.join(kit, "config.yml"))
    schema_path = schema_path or os.path.join(data_dir, "schema.json")
    schema = make_sandbox.load_schema(schema_path, config)
    respondents = score.load_frames(data_dir, schema)
    masked, cells, truth = score.sample_rows(schema, respondents, phase)
    floor = config["scoring"]["floor"]
    reference = score.marginal_reference(schema, masked, cells, truth, floor)
    # Round-trip through JSON-like objects the way the driver does: values
    # arrive as python objects, NaN for held-out cells.
    items = make_sandbox.generated_items(schema)
    frame = masked[["respondent_id"] + items].copy()
    frame = frame.astype(object).where(frame.notna(), np.nan)
    return dict(schema=schema, config=config, frame=frame, cells=cells,
                truth=truth, reference=reference, floor=floor, score=score,
                make_sandbox=make_sandbox)


def grade(prep, vectors):
    score = prep["score"]
    schema, cells, truth = prep["schema"], prep["cells"], prep["truth"]
    score.check(schema, vectors, cells)
    floored = score.floored(vectors, schema, cells, prep["floor"])
    result = score.score(schema, prep["config"], floored, truth, cells,
                         prep["reference"])
    return result


def per_cell_logp(prep, vectors):
    """Floored log p of the truth for every scored cell (for paired tests)."""
    score = prep["score"]
    floored = score.floored(vectors, prep["schema"], prep["cells"], prep["floor"])
    out = np.empty(len(floored))
    for i, (v, actual, (_, _, item)) in enumerate(
            zip(floored, prep["truth"], prep["cells"])):
        options = prep["make_sandbox"].options_for(prep["schema"], item)
        out[i] = np.log(v[options.index(actual)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--kit", default=DEFAULT_KIT)
    ap.add_argument("--out", default=None, help="write json summary here")
    ap.add_argument("--save-logp", default=None,
                    help="directory to save per-cell logp arrays (npz)")
    ap.add_argument("--fm-cache", default=os.environ.get("SIMULACRA_LAB_FM_CACHE"),
                    help="lab-only disk cache for in-context expert predictions")
    ap.add_argument("--fm-speed", type=float,
                    default=float(os.environ.get("SIMULACRA_LAB_FM_SPEED", "0") or 0),
                    help="with --fm-cache: simulate a GPU this many times faster")
    args = ap.parse_args()

    started = time.time()
    module = load_submission(args.sub)
    import_seconds = time.time() - started
    if args.fm_cache:
        install_fm_cache(module, args.fm_cache, args.fm_speed or None)
    summary = {"sub": args.sub, "phase": args.phase,
               "import_seconds": import_seconds, "runs": []}
    for data_dir in args.data:
        prep = prepare(data_dir, args.phase, args.kit)
        if args.fm_cache:
            module._LAB_FM_COST.clear()
        vclock = module.time if args.fm_cache and args.fm_speed else None
        v0 = vclock.monotonic() if vclock else 0.0
        t0 = time.time()
        vectors = module.predict(prep["frame"].copy(), prep["schema"])
        seconds = time.time() - t0
        result = grade(prep, vectors)
        info = getattr(module, "LAST_INFO", None)
        run = {"data": data_dir, "skill": result["skill"],
               "log_score": result["log_score"],
               "reference": result["marginal_reference"],
               "std_error": result["std_error"], "n_cells": result["n_cells"],
               "seconds": seconds}
        if info is not None:
            run["info"] = json.loads(json.dumps(info, default=str))
        if args.fm_cache:
            run["fm_cost"] = {k: [round(v[0], 2), v[1], v[2]]
                              for k, v in module._LAB_FM_COST.items()}
        if vclock:
            run["virtual_seconds"] = round(vclock.monotonic() - v0, 1)
            print("   virtual %.1fs at speed %.2f" % (run["virtual_seconds"], args.fm_speed))
        summary["runs"].append(run)
        print("%-50s skill %.5f  logS %.5f  R %.4f  se %.4f  %6.1fs"
              % (data_dir[-50:], result["skill"], result["log_score"],
                 result["marginal_reference"], result["std_error"], seconds),
              flush=True)
        if args.save_logp:
            os.makedirs(args.save_logp, exist_ok=True)
            name = os.path.basename(os.path.normpath(data_dir))
            np.savez_compressed(
                os.path.join(args.save_logp, "%s_p%d.npz" % (name, args.phase)),
                logp=per_cell_logp(prep, vectors),
                reference=np.array(result["marginal_reference"]),
                rid=np.array([c[1] for c in prep["cells"]], dtype=object),
                item=np.array([c[2] for c in prep["cells"]], dtype=object))
    total = sum(r["seconds"] for r in summary["runs"]) + import_seconds
    summary["total_seconds"] = total
    if len(summary["runs"]) > 1:
        print("mean skill %.5f   total %.1fs"
              % (np.mean([r["skill"] for r in summary["runs"]]), total))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
