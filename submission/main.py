"""SimulacraBench submission (Claude v5).

Per-cell conditional distributions P(answer | GIVEN block) for every blank cell,
estimated on the visible rows at run time (the competition site permits
training on the visible rows before predicting).

Pipeline per instrument
  1. Parse schema; integer-code every GIVEN/PREDICT column.
  2. Out-of-fold (OOF) predictions on the visible rows plus predictions for the
     held-out rows from several experts:
       marginal   smoothed visible counts (the reference forecast)
       lr         multi-task multinomial logistic regression, per-item ridge
                  chosen by OOF loss
       pfn*/icl   pretrained tabular in-context learners (TabPFN-3.5, TabICL v2),
                  bundled weights, no gradient updates, fold-bagged
  3. Per item, log-linear pooling of the experts with non-negative weights fit on
     the OOF floored log loss (shrunk toward instrument-wide weights).
  4. Skip-logic gates with a GIVEN parent: P(asked | parent value) is learned
     from the visible rows (schema as prior) and imposed exactly.
  5. A time-budget scheduler covers dev (900 s) and final (3600 s) phases; any
     expert failure or timeout falls back to the experts that completed.

No respondent identifiers are used; nothing is written to disk or network.
"""
import os
import sys
import time
import math

_T_IMPORT = time.monotonic()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "4")

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - torch is in the hosted image
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("SIMULACRA_WEIGHTS", os.path.join(HERE, "weights"))
FLOOR = 1e-3
LAST_INFO = {}
_SEEN = {}


def _env(name, default, cast=str):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return cast(value)


# ----------------------------------------------------------------- config --
EXPERTS = _env("SIMULACRA_EXPERTS", "marginal,lr,mlp,pfnf,icl,pfn,dec")
KFOLD = _env("SIMULACRA_KFOLD", 5, int)
FM_KFOLD_DEV = _env("SIMULACRA_FM_KFOLD_DEV", 2, int)
FM_KFOLD_FINAL = _env("SIMULACRA_FM_KFOLD_FINAL", 5, int)
FM_CAL_FRAC_DEV = _env("SIMULACRA_FM_CAL_FRAC_DEV", 0.4, float)
FM_CAL_FRAC_FINAL = _env("SIMULACRA_FM_CAL_FRAC_FINAL", 1.0, float)
BUDGET_OVERRIDE = _env("SIMULACRA_BUDGET", 0.0, float)
FORCE_PHASE = _env("SIMULACRA_FORCE_PHASE", "")
FM_NEST = _env("SIMULACRA_FM_NEST", "")
# In-context experts in priority order per phase (items an expert does not reach
# keep the others' pool). Full TabPFN-3.5 comes via models.txt; if it is missing,
# TabPFN-3.5-Fast takes its slot.
# Proxy matrix (phase-2 frames, 4 proxies, paired): full TabPFN-3.5 alone with 8
# estimators was the best pool (+0.00023 over 4 estimators, positive on all four);
# adding TabICL, TabPFN-3.5-Fast or the multiclass variant next to it added nothing
# or slightly hurt. Fast is the fallback for a missing full TabPFN-3.5.
FM_ORDER_FINAL = _env("SIMULACRA_FM_ORDER_FINAL", "pfn")
FM_ORDER_DEV = _env("SIMULACRA_FM_ORDER_DEV", "pfn")
# Experts that only fill leftover time (one estimator, not counted when sizing
# the others' ensembles) while full TabPFN-3.5 is present.
FM_FILLER = _env("SIMULACRA_FM_FILLER", "pfnf")
# Final phase: per-kind caps on the adaptive ensemble size, e.g. "pfnf=1", and the
# largest size the planner may choose for any expert.
FM_NEST_MAX = _env("SIMULACRA_FM_NEST_MAX", "")
NEST_TOP = _env("SIMULACRA_NEST_TOP", 8, int)
# Re-plan the ensemble size before each expert from the measured speed.
REPLAN = _env("SIMULACRA_REPLAN", 1, int)
# Multiplier on the ridge pulling per-item pooling weights toward the
# instrument-wide weights (base strength 2/sqrt(n)).
POOL_STRENGTH = _env("SIMULACRA_POOL_STRENGTH", 1.0, float)
FM_NOMINAL = _env("SIMULACRA_FM_NOMINAL", "cat")
SAFETY_DEV = _env("SIMULACRA_SAFETY_DEV", 0.72, float)
SAFETY_FINAL = _env("SIMULACRA_SAFETY_FINAL", 0.78, float)
SEED = _env("SIMULACRA_SEED", 20261109, int)
VERBOSE = _env("SIMULACRA_VERBOSE", 0, int)
STRUCT_ZEROS = _env("SIMULACRA_STRUCT_ZEROS", 1, int)
ORDINAL = _env("SIMULACRA_ORDINAL", 1, int)
# In the one-shot final phase a hard failure of the pretrained experts (no GPU,
# missing weights, import errors) raises instead of degrading silently: failed
# submissions do not count and can be retried. Timeouts never raise.
FAIL_LOUD = _env("SIMULACRA_FAIL_LOUD", "auto")


class FailLoud(RuntimeError):
    pass

CKPT_TABICL = os.path.join(WEIGHTS, "tabicl-classifier-v2-20260212.ckpt")
CKPT_PFN_FAST = os.path.join(WEIGHTS, "tabpfn-v3.5-fast-20260909.safetensors")
CKPT_PFN = os.path.join(WEIGHTS, "tabpfn-v3.5-20260909.safetensors")
# The full TabPFN-3.5 checkpoint is too large to bundle next to the others under
# the 1 GB cap; models.txt names its repo so the platform pre-downloads it into
# the Hugging Face cache at build time. Only the pinned commit is used: another
# snapshot qualifies only if the file's bytes hash to the pinned file's hash.
PFN_REPO = "Prior-Labs/tabpfn_3_5"
PFN_REV = "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"
PFN_SHA256 = "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"
# The same repo's multiclass variant ("slightly different architecture, sometimes
# better on multiclass"), already in the models.txt download; optional expert.
CKPT_PFN_MC = os.path.join(WEIGHTS, "tabpfn-v3.5-20260909_multiclass.safetensors")
PFN_MC_SHA256 = "4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149"
# TabPFN-3.5-Fast is in the same repo: a build without bundled weights takes it
# from the models.txt download too.
PFN_FAST_SHA256 = "14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550"
_RESOLVED = {}


def _log(*args):
    if VERBOSE:
        print("[v5 %7.1fs]" % (time.monotonic() - _T_IMPORT), *args,
              file=sys.stderr, flush=True)


def _hub_roots():
    """Hugging Face cache directories, in the order huggingface_hub resolves them."""
    roots = []
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if os.environ.get(var):
            roots.append(os.environ[var])
    if os.environ.get("HF_HOME"):
        roots.append(os.path.join(os.environ["HF_HOME"], "hub"))
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    roots.append(os.path.join(xdg, "huggingface", "hub"))
    try:
        from huggingface_hub import constants as hub_constants
        roots.append(str(hub_constants.HF_HUB_CACHE))
    except Exception:
        pass
    out = []
    for r in roots:
        if r and r not in out:
            out.append(r)
    return out


def _file_sha256(fname):
    import hashlib
    h = hashlib.sha256()
    with open(fname, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _hub_fallback_roots(folder):
    """Directories holding `folder` in common cache locations, in case the
    runtime does not export the cache variable the build step used. Bounded:
    a few fixed roots, at most three levels deep; never /proc, /sys or /dev."""
    import glob
    found = []
    for base in ("/root", "/home", "/opt", "/tmp", "/data", "/cache", "/models", "/mnt",
                 "/srv", "/var/cache", "/workspace", "/app", "/hf", "/huggingface"):
        if not os.path.isdir(base):
            continue
        for depth in range(0, 4):
            pattern = os.path.join(base, *(["*"] * depth), folder)
            try:
                hits = glob.glob(pattern)
            except Exception:
                hits = []
            for h in hits:
                root = os.path.dirname(h)
                if root not in found:
                    found.append(root)
    return found


def _hub_file(repo, filename, revision, sha256):
    """Path of a file pre-downloaded via models.txt, or "" if absent."""
    folder = "models--" + repo.replace("/", "--")
    found = _hub_file_in(_hub_roots(), folder, filename, revision, sha256)
    if not found:
        found = _hub_file_in(_hub_fallback_roots(folder), folder, filename, revision, sha256)
    return found


def _hub_file_in(roots, folder, filename, revision, sha256):
    for root in roots:
        snaps = os.path.join(root, folder, "snapshots")
        try:
            names = sorted(os.listdir(snaps))
        except OSError:
            continue
        if revision in names:
            cand = os.path.join(snaps, revision, filename)
            if os.path.exists(cand):
                return cand
        for name in names:
            cand = os.path.join(snaps, name, filename)
            if name == revision or not os.path.exists(cand):
                continue
            try:
                if _file_sha256(cand) == sha256:
                    return cand
            except OSError:
                pass
    return ""


def _pfn_full_path(kind="pfn"):
    """A TabPFN-3.5 checkpoint (kind "pfn" full, "pfnm" its multiclass variant,
    "pfnf" Fast): bundled copy if present, else the models.txt download."""
    if kind not in _RESOLVED:
        local, digest = {"pfnm": (CKPT_PFN_MC, PFN_MC_SHA256),
                         "pfnf": (CKPT_PFN_FAST, PFN_FAST_SHA256)}.get(kind, (CKPT_PFN, PFN_SHA256))
        found = local if os.path.exists(local) else ""
        if not found:
            try:
                found = _hub_file(PFN_REPO, os.path.basename(local), PFN_REV, digest)
            except Exception:
                found = ""
        _RESOLVED[kind] = found
    return _RESOLVED[kind]


# ------------------------------------------------------------- utilities --

def _floored(p):
    k = p.shape[-1]
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-300)
    return FLOOR + (1.0 - k * FLOOR) * p


def _nll(p, y):
    """Mean floored negative log likelihood; y are class indices (>=0)."""
    q = _floored(p)
    return float(-np.log(q[np.arange(len(y)), y]).mean())


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _parse_level(label):
    """Numeric value of an ordinal-looking label, else None."""
    text = str(label).strip()
    if not text:
        return None
    t = text.replace(",", "")
    for prefix in ("<=", ">=", "<", ">"):
        if t.startswith(prefix):
            t = t[len(prefix):]
    if t.endswith("+"):
        t = t[:-1]
    try:
        return float(t)
    except ValueError:
        pass
    if "-" in t:
        a, _, b = t.partition("-")
        try:
            return (float(a) + float(b)) / 2.0
        except ValueError:
            return None
    head = text.split(".")[0].strip()
    if head.isdigit() and len(head) <= 3 and "." in text:
        return float(head)
    return None


# ------------------------------------------------------------- instrument --

class Instrument:
    def __init__(self, frame, schema):
        records = schema["items"]
        self.schema = schema
        self.gated_value = schema.get("gated_value", "NA_GATED")
        self.items = [n for n, r in records.items()
                      if r.get("class") in ("GIVEN", "PREDICT") and n in frame.columns]
        self.given = [n for n in self.items if records[n]["class"] == "GIVEN"]
        self.targets = [n for n in self.items if records[n]["class"] == "PREDICT"]
        self.records = records
        self.options = {}
        for n in self.items:
            opts = list(records[n]["values"])
            if records[n].get("gate"):
                opts.append(self.gated_value)
            self.options[n] = opts
        self.K = {n: len(self.options[n]) for n in self.items}
        self.codes = {}
        for n in self.items:
            cat = pd.Categorical(frame[n].to_numpy(dtype=object),
                                 categories=self.options[n])
            self.codes[n] = np.asarray(cat.codes, dtype=np.int64)
        self.n = len(frame)
        if self.targets:
            known = np.column_stack([self.codes[t] >= 0 for t in self.targets])
            self.vis = np.flatnonzero(known.all(axis=1))
            self.hid = np.flatnonzero(~known.all(axis=1))
        else:
            self.vis = np.arange(self.n)
            self.hid = np.array([], dtype=np.int64)
        self.gate = {n: records[n].get("gate") for n in self.targets}

    # Features ---------------------------------------------------------
    def given_matrix(self):
        """Integer codes of GIVEN columns; unknown/NaN -> extra code."""
        cols, cards, nominal = [], [], []
        for g in self.given:
            k = self.K[g]
            c = self.codes[g]
            cols.append(np.where(c >= 0, c, k))
            cards.append(k + 1)
            vals = [_parse_level(v) for v in self.records[g]["values"]]
            numeric = sum(v is not None for v in vals) >= max(2, len(vals) - 2)
            nominal.append(not numeric)
        if not cols:
            return np.zeros((self.n, 1), np.int64), [1], [False]
        return np.column_stack(cols).astype(np.int64), cards, nominal

    def numeric_extras(self):
        """Sums of count-like GIVEN columns that share a question text."""
        groups = {}
        for g in self.given:
            vals = [_parse_level(v) for v in self.records[g]["values"]]
            if all(v is not None for v in vals) and len(vals) >= 3:
                q = str(self.records[g].get("question", ""))[:60]
                groups.setdefault(q, []).append((g, vals))
        extras = []
        for q, members in groups.items():
            if len(members) < 3:
                continue
            total = np.zeros(self.n)
            for g, vals in members:
                lut = np.asarray(vals + [0.0])
                c = self.codes[g]
                total += lut[np.where(c >= 0, c, len(vals))]
            extras.append(total)
        if not extras:
            return None
        return np.column_stack(extras)


# ----------------------------------------------------------------- budget --

_KNOWN_SHARE = {12: 0.15, 76: 0.27, 213: 0.58}
# Development-phase rehearsal of the final code path (final FM settings and
# fail-loud) under the development budget. Flipped only in rehearsal builds.
FINAL_PATH_TEST = False


def _phase_budget(inst):
    if BUDGET_OVERRIDE > 0:
        return BUDGET_OVERRIDE, (FORCE_PHASE or "override"), SAFETY_FINAL
    split = inst.schema.get("split") or {}
    n_hid = len(inst.hid)
    if split and n_hid == split.get("n_test") and n_hid != split.get("n_dev"):
        return 3600.0, "final", SAFETY_FINAL
    if FINAL_PATH_TEST:
        return 900.0, "final", SAFETY_DEV
    return 900.0, "dev", SAFETY_DEV


def _instrument_deadline(inst):
    total, phase, safety = _phase_budget(inst)
    end = _T_IMPORT + total * safety
    key = (len(inst.targets), inst.n)
    _SEEN[key] = True
    now = time.monotonic()
    share_this = _KNOWN_SHARE.get(len(inst.targets), 0.3)
    if BUDGET_OVERRIDE > 0:
        # Lab mode: the override is the budget for this instrument alone.
        return now + total * safety, phase
    remaining = [s for t, s in _KNOWN_SHARE.items()
                 if not any(k[0] == t for k in _SEEN)]
    denom = share_this + sum(remaining)
    return now + max(0.0, end - now) * share_this / denom, phase


# ---------------------------------------------------------------- experts --

def _fold_ids(n, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ids = np.empty(n, np.int64)
    ids[perm] = np.arange(n) % k
    return ids


def _marginal_expert(inst, folds, kfold):
    out = {}
    nv = len(inst.vis)
    for t in inst.targets:
        y = inst.codes[t][inst.vis]
        K = inst.K[t]
        oof = np.empty((nv, K))
        for f in range(kfold):
            tr = folds != f
            c = np.bincount(y[tr], minlength=K) + 0.5
            oof[folds == f] = c / c.sum()
        c = np.bincount(y, minlength=K) + 0.5
        out[t] = (oof, np.tile(c / c.sum(), (len(inst.hid), 1)))
    return out


def _onehot(X, cards, pairs=()):
    blocks = [np.eye(c, dtype=np.float32)[X[:, j]] for j, c in enumerate(cards)]
    for a, b in pairs:
        code = X[:, a] * cards[b] + X[:, b]
        blocks.append(0.5 * np.eye(cards[a] * cards[b], dtype=np.float32)[code])
    M = np.concatenate(blocks, axis=1)
    keep = M.std(axis=0) > 0
    return M[:, keep]


def _lr_expert(inst, folds, kfold, X, cards, interactions, deadline, device):
    """Multi-task multinomial logistic regression with per-item ridge."""
    if torch is None:
        return {}
    pairs = []
    if interactions:
        for a in range(len(cards)):
            for b in range(a + 1, len(cards)):
                if cards[a] * cards[b] <= 64:
                    pairs.append((a, b))
    F = _onehot(X, cards, pairs)
    if F.shape[1] > 6000:
        return {}
    targets = inst.targets
    Ks = [inst.K[t] for t in targets]
    groups = {}
    for i, k in enumerate(Ks):
        groups.setdefault(k, []).append(i)
    Y = np.column_stack([inst.codes[t] for t in targets])
    vis, hid = inst.vis, inst.hid
    dev = torch.device(device)
    Ft = torch.as_tensor(F, device=dev)
    Yt = torch.as_tensor(np.maximum(Y, 0), device=dev)
    def fit(rows, lam):
        rows_t = torch.as_tensor(rows, device=dev)
        Xr = Ft[rows_t]
        n = len(rows)
        params = []
        for k, idx in groups.items():
            W = torch.zeros(F.shape[1], len(idx), k, device=dev, requires_grad=True)
            yk = Yt[rows_t][:, idx]
            cnt = torch.stack([torch.bincount(yk[:, j], minlength=k) for j in range(len(idx))]).float()
            b0 = torch.log((cnt + 0.5) / (cnt.sum(1, keepdim=True) + 0.5 * k))
            b = b0.clone().requires_grad_(True)
            params.append((k, idx, W, b, yk))
        opt = torch.optim.LBFGS([p for _, _, W, b, _ in params for p in (W, b)],
                                lr=1.0, max_iter=100, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-6,
                                tolerance_change=1e-9)

        def closure():
            opt.zero_grad()
            loss = 0.0
            for k, idx, W, b, yk in params:
                logits = torch.einsum("nd,djk->njk", Xr, W) + b
                lp = torch.log_softmax(logits, dim=-1)
                loss = loss - lp.gather(2, yk[:, :, None]).sum() / n
                loss = loss + 0.5 * lam * (W * W).sum() / n
            loss.backward()
            return loss
        opt.step(closure)
        return [(k, idx, W.detach(), b.detach()) for k, idx, W, b, _ in params]

    def apply(model, rows):
        rows_t = torch.as_tensor(rows, device=dev)
        Xr = Ft[rows_t]
        out = [None] * len(targets)
        with torch.no_grad():
            for k, idx, W, b in model:
                p = torch.softmax(torch.einsum("nd,djk->njk", Xr, W) + b, dim=-1)
                p = p.cpu().numpy().astype(np.float64)
                for jj, i in enumerate(idx):
                    out[i] = p[:, jj, :]
        return out

    nv = len(vis)
    # The first ridge value always completes (a complete expert beats none);
    # further values only while time remains, then each item keeps its best.
    lambdas = (4.0, 1.0, 16.0, 64.0) if not interactions else (16.0, 4.0, 64.0, 256.0)
    oof, done_lams = {}, []
    for lam in lambdas:
        if done_lams and time.monotonic() > deadline:
            break
        cur = [np.zeros((nv, k)) for k in Ks]
        for f in range(kfold):
            tr = vis[folds != f]
            te_pos = np.flatnonzero(folds == f)
            model = fit(tr, lam)
            preds = apply(model, vis[te_pos])
            for i in range(len(targets)):
                cur[i][te_pos] = preds[i]
        oof[lam] = cur
        done_lams.append(lam)
    yv = Y[vis]
    best = []
    for i in range(len(targets)):
        losses = [_nll(oof[lam][i], yv[:, i]) for lam in done_lams]
        best.append(done_lams[int(np.argmin(losses))])
    out = {}
    for lam in sorted(set(best)):
        model = fit(vis, lam)
        preds = apply(model, hid) if len(hid) else [np.zeros((0, k)) for k in Ks]
        for i, t in enumerate(targets):
            if best[i] == lam:
                out[t] = (oof[lam][i], preds[i])
    return out


def _mlp_expert(inst, folds, kfold, X, cards, deadline, device, seeds=2):
    """Multi-task MLP on one-hot GIVEN (shared trunk, per-item softmax heads,
    linear skip). Epoch count chosen on an inner split of fold 0."""
    if torch is None:
        return {}
    dev = torch.device(device)
    F = _onehot(X, cards)
    targets = inst.targets
    Ks = [inst.K[t] for t in targets]
    offs = np.concatenate([[0], np.cumsum(Ks)]).astype(np.int64)
    Y = np.column_stack([np.maximum(inst.codes[t], 0) for t in targets])
    vis, hid = inst.vis, inst.hid
    Ft = torch.as_tensor(F, device=dev)
    Yt = torch.as_tensor(Y, device=dev)
    groups = {}
    for i, k in enumerate(Ks):
        groups.setdefault(k, []).append(i)
    gidx = {k: torch.as_tensor(np.concatenate([np.arange(offs[i], offs[i] + k) for i in idx]),
                               device=dev) for k, idx in groups.items()}
    gcols = {k: torch.as_tensor(idx, device=dev) for k, idx in groups.items()}
    n_out = int(offs[-1])

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.skip = torch.nn.Linear(F.shape[1], n_out)
            self.body = torch.nn.Sequential(
                torch.nn.Linear(F.shape[1], 128), torch.nn.GELU(), torch.nn.Dropout(0.25),
                torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Dropout(0.25))
            self.head = torch.nn.Linear(128, n_out)
            torch.nn.init.zeros_(self.skip.weight)
            torch.nn.init.normal_(self.head.weight, std=0.01)

        def forward(self, x):
            return self.skip(x) + self.head(self.body(x))

    def loss_fn(logits, y):
        tot = 0.0
        for k, cols in gcols.items():
            lg = logits[:, gidx[k]].reshape(len(logits), len(cols), k)
            lp = torch.log_softmax(lg, dim=-1)
            tot = tot - lp.gather(2, y[:, cols][:, :, None]).sum()
        return tot / len(logits)

    def train(rows, epochs, seed, valid=None):
        torch.manual_seed(seed)
        net = Net().to(dev)
        rows_t = torch.as_tensor(rows, device=dev)
        with torch.no_grad():
            yv_ = Yt[rows_t]
            for i, k in enumerate(Ks):
                c = torch.bincount(yv_[:, i], minlength=k).float() + 0.5
                net.skip.bias[int(offs[i]):int(offs[i]) + k] = torch.log(c / c.sum())
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
        best, best_ep = np.inf, 0
        for ep in range(epochs):
            net.train()
            perm = rows_t[torch.randperm(len(rows_t), device=dev)]
            for s in range(0, len(perm), 512):
                b = perm[s:s + 512]
                opt.zero_grad()
                loss = loss_fn(net(Ft[b]), Yt[b])
                loss.backward()
                opt.step()
            if valid is not None:
                net.eval()
                with torch.no_grad():
                    vl = float(loss_fn(net(Ft[valid]), Yt[valid]))
                if vl < best - 1e-5:
                    best, best_ep = vl, ep + 1
                elif ep + 1 - best_ep >= 4:
                    break
            if time.monotonic() > deadline:
                break
        return net, best_ep

    def predict_rows(net, rows):
        net.eval()
        with torch.no_grad():
            lg = net(Ft[torch.as_tensor(rows, device=dev)])
            return [torch.softmax(lg[:, int(offs[i]):int(offs[i]) + k], dim=-1).cpu().numpy().astype(np.float64)
                    for i, k in enumerate(Ks)]

    nv = len(vis)
    tr0 = vis[folds != 0]
    inner = np.random.default_rng(SEED + 3).permutation(tr0)
    nval = max(200, len(inner) // 8)
    _, ep_best = train(inner[nval:], 60, SEED, valid=torch.as_tensor(inner[:nval], device=dev))
    ep_best = max(3, ep_best)
    oof = [np.zeros((nv, k)) for k in Ks]
    for f in range(kfold):
        if time.monotonic() > deadline:
            return {}
        te = np.flatnonzero(folds == f)
        acc = None
        for s in range(seeds):
            net, _ = train(vis[folds != f], ep_best, SEED + 11 * s + f)
            pr = predict_rows(net, vis[te])
            acc = pr if acc is None else [a + p for a, p in zip(acc, pr)]
        for i in range(len(Ks)):
            oof[i][te] = acc[i] / seeds
    acc = None
    for s in range(seeds):
        net, _ = train(vis, ep_best, SEED + 101 * s)
        pr = predict_rows(net, hid) if len(hid) else [np.zeros((0, k)) for k in Ks]
        acc = pr if acc is None else [a + p for a, p in zip(acc, pr)]
    return {t: (oof[i], acc[i] / seeds) for i, t in enumerate(targets)}


def _mtlr_fit(Ft, Yt, rows_t, groups, lam, d, dev, max_iter=100, init=None):
    """Multi-task multinomial LR on feature tensor rows; groups: {K: [cols]}.
    `init` (a previous fit with the same groups) warm-starts the optimiser."""
    Xr = Ft[rows_t]
    n = len(rows_t)
    params = []
    start = {k: (W0, b0) for k, _, W0, b0 in (init or [])}
    for k, idx in groups.items():
        yk = Yt[rows_t][:, idx]
        if k in start:
            W = start[k][0].clone().requires_grad_(True)
            b = start[k][1].clone().requires_grad_(True)
        else:
            W = torch.zeros(d, len(idx), k, device=dev, requires_grad=True)
            cnt = torch.stack([torch.bincount(yk[:, j], minlength=k) for j in range(len(idx))]).float()
            b = torch.log((cnt + 0.5) / (cnt.sum(1, keepdim=True) + 0.5 * k)).clone().requires_grad_(True)
        params.append((k, idx, W, b, yk))
    opt = torch.optim.LBFGS([p for _, _, W, b, _ in params for p in (W, b)],
                            lr=1.0, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-6,
                            tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = 0.0
        for k, idx, W, b, yk in params:
            logits = torch.einsum("nd,djk->njk", Xr, W) + b
            lp = torch.log_softmax(logits, dim=-1)
            loss = loss - lp.gather(2, yk[:, :, None]).sum() / n
            loss = loss + 0.5 * lam * (W * W).sum() / n
        loss.backward()
        return loss
    opt.step(closure)
    return [(k, idx, W.detach(), b.detach()) for k, idx, W, b, _ in params]


def _mtlr_apply(model, Fq, n_out):
    out = [None] * n_out
    with torch.no_grad():
        for k, idx, W, b in model:
            p = torch.softmax(torch.einsum("nd,djk->njk", Fq, W) + b, dim=-1)
            p = p.cpu().numpy().astype(np.float64)
            for jj, i in enumerate(idx):
                out[i] = p[:, jj, :]
    return out


def _dec_expert(inst, parent, children, X, cards, folds, kfold, pi_oof, pi_hid,
                device, lam=4.0):
    """Children of a PREDICT parent: P(child | x, parent value) by multi-task LR,
    marginalised over the parent's pooled predicted distribution pi."""
    dev = torch.device(device)
    vis, hid = inst.vis, inst.hid
    Kp = inst.K[parent]
    base = _onehot(X, cards)
    small = [j for j, c in enumerate(cards) if c <= 4]

    def feats(rows, v):
        n = len(rows)
        onev = np.zeros((n, Kp), np.float32)
        onev[np.arange(n), v] = 1.0
        blocks = [base[rows], onev]
        for j in small:
            xg = np.eye(cards[j], dtype=np.float32)[X[rows, j]]
            blocks.append((onev[:, :, None] * xg[:, None, :]).reshape(n, -1))
        return np.concatenate(blocks, axis=1)

    Ks = [inst.K[c] for c in children]
    groups = {}
    for i, k in enumerate(Ks):
        groups.setdefault(k, []).append(i)
    Y = np.column_stack([inst.codes[c] for c in children])
    pv_true = inst.codes[parent]
    nv = len(vis)
    oof = [np.full((nv, k), np.nan) for k in Ks]
    Fv = torch.as_tensor(feats(vis, np.maximum(pv_true[vis], 0)), device=dev)
    d = Fv.shape[1]
    Yt = torch.as_tensor(np.maximum(Y[vis], 0), device=dev)

    def mix(model, rows, pi):
        acc = [np.zeros((len(rows), k)) for k in Ks]
        for v in range(Kp):
            w = pi[:, v]
            if not (w > 1e-6).any():
                continue
            Fq = torch.as_tensor(feats(rows, np.full(len(rows), v)), device=dev)
            pr = _mtlr_apply(model, Fq, len(children))
            for i in range(len(children)):
                acc[i] += w[:, None] * pr[i]
        return acc

    full = _mtlr_fit(Fv, Yt, torch.arange(nv, device=dev), groups, lam, d, dev, max_iter=60)
    for f in range(kfold):
        tr = np.flatnonzero(folds != f)
        te = np.flatnonzero((folds == f) & np.isfinite(pi_oof).all(axis=1))
        if not len(te):
            continue
        model = _mtlr_fit(Fv, Yt, torch.as_tensor(tr, device=dev), groups, lam, d, dev,
                          max_iter=25, init=full)
        got = mix(model, vis[te], pi_oof[te])
        for i in range(len(children)):
            oof[i][te] = got[i]
    hidp = mix(full, hid, pi_hid) if len(hid) else [np.zeros((0, k)) for k in Ks]
    return {c: (oof[i], hidp[i]) for i, c in enumerate(children)}


class _FM:
    """Pretrained in-context tabular classifier, fold-bagged per item."""

    def __init__(self, kind, n_estimators, device):
        self.kind = kind
        self.n_estimators = n_estimators
        self.device = device

    def available(self):
        return bool(self.path())

    def path(self):
        if self.kind == "icl":
            return CKPT_TABICL if os.path.exists(CKPT_TABICL) else ""
        return _pfn_full_path(self.kind)

    def make(self, cat_idx):
        if self.kind == "icl":
            from tabicl import TabICLClassifier
            return TabICLClassifier(device=self.device, n_estimators=self.n_estimators,
                                    random_state=0, model_path=CKPT_TABICL,
                                    allow_auto_download=False, use_fa3=False)
        from tabpfn import TabPFNClassifier
        _patch_tabpfn_checkpoint()
        path = self.path()
        return TabPFNClassifier(device=self.device, n_estimators=self.n_estimators,
                                random_state=0, model_path=path,
                                ignore_pretraining_limits=True,
                                categorical_features_indices=cat_idx or None)

    def _predict(self, Xc, yc, Xq, K, cat_idx):
        classes = np.unique(yc)
        c = np.bincount(yc, minlength=K) + 0.5
        prior = c / c.sum()
        if len(classes) < 2 or len(Xq) == 0:
            return np.tile(prior, (len(Xq), 1))
        if FM_NOMINAL == "target" and cat_idx:
            Xc, Xq = _target_order(Xc, yc, Xq, K, cat_idx, prior)
            cat_idx = []
        elif FM_NOMINAL == "raw":
            cat_idx = []
        model = self.make(cat_idx)
        model.fit(Xc, yc)
        proba = model.predict_proba(Xq)
        full = np.zeros((len(Xq), K))
        full[:, np.asarray(model.classes_, dtype=np.int64)] = proba
        # Classes unseen in this context keep a tiny smoothed mass.
        full = 0.999 * full + 0.001 * prior
        return full / full.sum(axis=1, keepdims=True)

    def run_item(self, Xv, y, Xh, plan, K, cat_idx):
        """plan: list of (context_rows, query_rows) over visible rows."""
        oof = np.full((len(y), K), np.nan)
        for ctx, q in plan:
            oof[q] = self._predict(Xv[ctx], y[ctx], Xv[q], K, cat_idx)
        hid = self._predict(Xv, y, Xh, K, cat_idx)
        return oof, hid


_SMOKE = {}

# Per-item cost of each in-context expert with one estimator relative to
# TabPFN-3.5-Fast, and of n estimators relative to one, measured on an H100
# (8 CPU cores, Python 3.13, final OOF plan, the three instrument shapes):
# Fast 0.48-0.85 s/item; full TabPFN 1.2-1.5x that; TabICL 2.4-4.6x, but its
# cost is mostly per-call overhead, so 4 estimators cost only ~1.6x one.
# Re-planning rescales these by the speed actually measured.
FM_REL_COST = {"pfnf": 1.0, "icl": 3.0, "pfn": 1.4, "pfnm": 1.4}
NEST_SCALE = {1: 1.0, 2: 1.8, 4: 3.4, 8: 6.6}
NEST_SCALE_KIND = {"icl": {1: 1.0, 2: 1.5, 4: 1.7, 8: 2.6}}


def _nest_scale(kind, e):
    return NEST_SCALE_KIND.get(kind, NEST_SCALE).get(e, NEST_SCALE.get(e, 0.85 * e))


def _parse_kv(text):
    out = {}
    for kv in (text or "").split(","):
        k, _, v = kv.partition("=")
        if k.strip() and v.strip():
            out[k.strip()] = int(v)
    return out


def _fm_requested(experts, phase):
    """In-context experts to run for this phase, in priority order."""
    order = FM_ORDER_DEV if phase == "dev" else FM_ORDER_FINAL
    known = ("pfnf", "icl", "pfn", "pfnm")
    kinds = []
    for k in (x.strip() for x in order.split(",")):
        if k in known and k in experts and k not in kinds:
            kinds.append(k)
    return kinds


def _fm_resolve(kinds, experts, device):
    """Drop unavailable experts; a missing full TabPFN-3.5 is replaced by the best
    pool without it (Fast + TabICL, whichever are available)."""
    out = []
    for k in kinds:
        if _FM(k, 1, device).available():
            if k not in out:
                out.append(k)
        elif k == "pfn":
            for sub in ("pfnf", "icl"):
                if (sub in experts and sub not in kinds and sub not in out
                        and _FM(sub, 1, device).available()):
                    out.append(sub)
    return out


def _pick_nest(kinds, n_items, t1, remaining, caps, frac=0.85):
    """Largest common ensemble size with which every kind covers every item."""
    for e in (8, 4, 2, 1):
        if e > NEST_TOP:
            continue
        cost = sum(FM_REL_COST.get(k, 1.8) * _nest_scale(k, min(e, caps.get(k, 8))) for k in kinds)
        if n_items * t1 * max(cost, 1.0) <= frac * remaining:
            return e
    return 1


def _patch_tabpfn_checkpoint():
    """tabpfn resolves symlinks before picking a loader by file suffix, and the
    Hugging Face cache keeps files as suffix-less blobs behind symlinks, so a
    checkpoint pre-downloaded via models.txt would be read as a pickle and fail.
    Recognise the safetensors header (u64 length, then a JSON object) instead."""
    try:
        from tabpfn import checkpoint as ck
    except Exception:
        return
    cls = getattr(ck, "Checkpoint", None)
    prop = getattr(cls, "is_safetensors", None) if cls is not None else None
    if not isinstance(prop, property) or getattr(cls, "_simulacra_sniff", False):
        return

    def is_safetensors(self):
        try:
            if prop.fget(self):
                return True
        except Exception:
            pass
        try:
            with open(self.path, "rb") as fh:
                head = fh.read(9)
            size = int.from_bytes(head[:8], "little")
            return len(head) == 9 and head[8:9] == b"{" and 2 <= size < (1 << 31)
        except Exception:
            return False

    cls.is_safetensors = property(is_safetensors)
    cls._simulacra_sniff = True


def _required_kinds(fm_kinds):
    """In-context experts whose absence should fail a loud (final) run: the
    bundled ones, plus full TabPFN-3.5 when models.txt was shipped for it."""
    req = [k for k in ("pfnf", "icl") if k in fm_kinds]
    if os.path.exists(os.path.join(HERE, "models.txt")):
        req += [k for k in ("pfn", "pfnm") if k in fm_kinds]
    return req


def _smoke(kind, device):
    """Load the expert and fit/predict a tiny synthetic task once per process;
    returns an error string or ""."""
    if kind not in _SMOKE:
        try:
            rng = np.random.default_rng(0)
            Xs = rng.normal(size=(96, 4)).astype(np.float32)
            ys = (Xs[:, 0] + 0.5 * rng.normal(size=96) > 0).astype(np.int64)
            p = _FM(kind, 1, device)._predict(Xs[:80], ys[:80], Xs[80:], 2, [])
            ok = p.shape == (16, 2) and np.isfinite(p).all()
            _SMOKE[kind] = "" if ok else "non-finite output"
        except Exception as exc:
            _SMOKE[kind] = type(exc).__name__ + ": " + str(exc)[:200]
    return _SMOKE[kind]


def _target_order(Xc, yc, Xq, K, cols, prior, strength=10.0):
    """Recode nominal columns by a smoothed per-level target score computed on
    the context rows only (rank order; unseen levels get the prior score)."""
    Xc = Xc.copy()
    Xq = Xq.copy()
    main = int(np.argmax(prior))
    onehot = np.zeros((len(yc), K))
    onehot[np.arange(len(yc)), yc] = 1.0
    for j in cols:
        lv_c = Xc[:, j].astype(np.int64)
        lv_q = Xq[:, j].astype(np.int64)
        m = int(max(lv_c.max(initial=0), lv_q.max(initial=0))) + 1
        cnt = np.zeros((m, K))
        np.add.at(cnt, lv_c, onehot)
        n = cnt.sum(axis=1, keepdims=True)
        post = (cnt + strength * prior) / (n + strength)
        # score: 1-D projection maximising between-level spread (main class vs rest
        # for binary; leading direction of the level means otherwise)
        if K == 2:
            score = post[:, 1]
        else:
            centred = (post - prior) * np.sqrt(n + 1e-9)
            try:
                _, _, vt = np.linalg.svd(centred, full_matrices=False)
                score = post @ vt[0]
            except np.linalg.LinAlgError:
                score = post[:, main]
        rank = np.empty(m)
        rank[np.argsort(score, kind="stable")] = np.arange(m)
        Xc[:, j] = rank[lv_c]
        Xq[:, j] = rank[lv_q]
    return Xc, Xq


def _fm_plan(nv, phase, seed):
    """OOF plan for in-context learners: high-context folds over a calibration
    subset (dev) or all visible rows (final)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nv)
    if phase == "final":
        frac, m = FM_CAL_FRAC_FINAL, FM_KFOLD_FINAL
    else:
        frac, m = FM_CAL_FRAC_DEV, FM_KFOLD_DEV
    n_cal = int(min(nv, max(min(nv, 1500), round(frac * nv))))
    cal = perm[:n_cal]
    parts = np.array_split(cal, max(1, m))
    plan = []
    everyone = np.arange(nv)
    for q in parts:
        mask = np.ones(nv, bool)
        mask[q] = False
        plan.append((everyone[mask], np.sort(q)))
    return plan, np.sort(cal)


# ---------------------------------------------------------------- pooling --

def _pool_weights(logps, y, prior_w=None, strength=0.0):
    """Non-negative log-linear pooling weights minimising floored NLL.

    logps: list of (n, K) log-probabilities; y: (n,) labels. Optional ridge
    toward prior_w with the given strength.
    """
    from scipy.optimize import minimize
    M = len(logps)
    L = np.stack(logps, axis=0)  # M, n, K
    n, K = L.shape[1], L.shape[2]
    rows = np.arange(n)
    onehot = np.zeros((n, K))
    onehot[rows, y] = 1.0
    prior = None if prior_w is None else np.asarray(prior_w, float)

    def fg(w):
        z = np.tensordot(w, L, axes=1)
        p = _softmax(z)
        q = FLOOR + (1 - K * FLOOR) * p
        qy = q[rows, y]
        f = -np.log(qy).mean()
        a = (1 - K * FLOOR) * p[rows, y] / qy
        dz = -(a[:, None]) * (onehot - p)
        g = np.einsum("mnk,nk->m", L, dz) / n
        if prior is not None and strength > 0:
            f += strength * ((w - prior) ** 2).sum()
            g = g + 2 * strength * (w - prior)
        return f, g

    w0 = np.full(M, 1.0 / M) if prior is None else np.maximum(prior, 1e-3)
    try:
        res = minimize(fg, w0, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 5.0)] * M, options={"maxiter": 100})
        w = res.x
        f = float(res.fun)
    except Exception:
        w, f = w0, fg(w0)[0]
    return w, f


def _pool(experts_oof, experts_hid, y, rows, shared_w=None, strength=0.0):
    """Fit log-linear weights on `rows` (bool mask), apply everywhere."""
    logs_fit = [np.log(np.clip(p[rows], 1e-7, 1.0)) for p in experts_oof]
    w, f = _pool_weights(logs_fit, y[rows], shared_w, strength)
    oof = np.full(experts_oof[0].shape, np.nan)
    oof[rows] = _softmax(np.tensordot(w, np.stack(logs_fit, 0), axes=1))
    if len(experts_hid[0]):
        logs_hid = [np.log(np.clip(p, 1e-7, 1.0)) for p in experts_hid]
        hid = _softmax(np.tensordot(w, np.stack(logs_hid, 0), axes=1))
    else:
        hid = experts_hid[0]
    return oof, hid, w


# ------------------------------------------------------------------ gates --

def _given_gate_structure(inst, item, p_hid, p_oof, folds, kfold, rows):
    """Impose P(asked | GIVEN parent value) learned from visible rows."""
    gate = inst.gate[item]
    parent = gate["parent"]
    if parent not in inst.given:
        return p_hid, p_oof, False
    K = inst.K[item]
    na = K - 1
    pc = inst.codes[parent]
    Kp = inst.K[parent] + 1
    pcode = np.where(pc >= 0, pc, Kp - 1)
    allowed = set(gate.get("observed_if") or [])
    s = np.array([1.0 if (j < len(inst.options[parent]) and inst.options[parent][j] in allowed)
                  else 0.0 for j in range(Kp)])
    y = inst.codes[item][inst.vis]
    asked = (y != na).astype(float)
    pv = pcode[inst.vis]

    def rates(mask):
        n = np.bincount(pv[mask], minlength=Kp).astype(float)
        a = np.bincount(pv[mask], weights=asked[mask], minlength=Kp)
        return (a + 0.5 + 0.5 * s) / (n + 1.0 + 0.5)

    def impose(p, r):
        out = p.copy()
        ans = out[:, :na]
        tot = ans.sum(axis=1, keepdims=True)
        ans = np.where(tot > 0, ans / np.maximum(tot, 1e-300), 1.0 / max(na, 1))
        out[:, :na] = ans * r[:, None]
        out[:, na] = 1.0 - r
        return out

    oof_new = p_oof.copy()
    for f in range(kfold):
        tr = folds != f
        te = (folds == f) & rows
        if te.any():
            oof_new[te] = impose(p_oof[te], rates(tr)[pv[te]])
    use = _nll(oof_new[rows], y[rows]) <= _nll(p_oof[rows], y[rows]) + 1e-9
    if not use:
        return p_hid, p_oof, False
    r_all = rates(np.ones(len(y), bool))
    hid_new = impose(p_hid, r_all[pcode[inst.hid]]) if len(inst.hid) else p_hid
    return hid_new, oof_new, True


def _ordinal_index(options):
    """Indices of options that parse as ordered numbers (sorted by value), or None."""
    vals = [(_parse_level(o), i) for i, o in enumerate(options)]
    num = [(v, i) for v, i in vals if v is not None]
    if len(num) < 4 or len(num) < 0.6 * len(options):
        return None
    num.sort()
    return np.array([i for _, i in num], dtype=np.int64)


def _ordinal_smooth(p, idx, alpha):
    if alpha <= 0:
        return p
    out = p.copy()
    block = p[:, idx]
    mass = block.sum(axis=1, keepdims=True)
    left = np.concatenate([block[:, :1], block[:, :-1]], axis=1)
    right = np.concatenate([block[:, 1:], block[:, -1:]], axis=1)
    sm = (left + block + right) / 3.0
    sm = sm * mass / np.maximum(sm.sum(axis=1, keepdims=True), 1e-300)
    out[:, idx] = (1 - alpha) * block + alpha * sm
    return out


def _ordinal_post(inst, t, p_hid, p_oof, rows):
    idx = _ordinal_index(inst.options[t])
    if idx is None:
        return p_hid, p_oof, 0.0
    y = inst.codes[t][inst.vis]
    best_a, best_l = 0.0, _nll(p_oof[rows], y[rows])
    for a in (0.05, 0.1, 0.2, 0.35, 0.5):
        l = _nll(_ordinal_smooth(p_oof[rows], idx, a), y[rows])
        if l < best_l - 1e-6:
            best_a, best_l = a, l
    if best_a == 0.0:
        return p_hid, p_oof, 0.0
    p_oof = p_oof.copy()
    p_oof[rows] = _ordinal_smooth(p_oof[rows], idx, best_a)
    return _ordinal_smooth(p_hid, idx, best_a) if len(p_hid) else p_hid, p_oof, best_a


def _zero_table(X, cards, y, K, rows, min_n):
    """For each (feature, value): mask of classes never seen among >= min_n rows."""
    tables = []
    for j, c in enumerate(cards):
        x = X[rows, j]
        cnt = np.zeros((c, K))
        np.add.at(cnt, (x, y[rows]), 1.0)
        n = cnt.sum(axis=1)
        zero = (cnt == 0) & (n[:, None] >= min_n)
        tables.append((zero, n))
    return tables


def _apply_zeros(p, Xq, tables, strength=0.5):
    """Cap probabilities of structurally absent classes at strength/(n+1)."""
    out = p.copy()
    cap = np.ones_like(out)
    for j, (zero, n) in enumerate(tables):
        v = Xq[:, j]
        z = zero[v]                       # q, K
        c = strength / (n[v] + 1.0)       # q
        cap = np.where(z, np.minimum(cap, c[:, None]), cap)
    limited = np.minimum(out, cap)
    excess = (out - limited).sum(axis=1, keepdims=True)
    free = np.where(limited < cap, limited, 0.0)
    fs = free.sum(axis=1, keepdims=True)
    out = limited + np.where(fs > 0, excess * free / np.maximum(fs, 1e-300), 0.0)
    return out / out.sum(axis=1, keepdims=True)


def _structural_zeros(inst, t, X, cards, p_hid, p_oof, folds, kfold, rows, min_n=60):
    y = inst.codes[t][inst.vis]
    K = inst.K[t]
    Xv = X[inst.vis]
    oof_new = p_oof.copy()
    for f in range(kfold):
        tr = np.flatnonzero(folds != f)
        te = (folds == f) & rows
        if te.any():
            tab = _zero_table(Xv, cards, y, K, tr, min_n)
            oof_new[te] = _apply_zeros(p_oof[te], Xv[te], tab)
    if _nll(oof_new[rows], y[rows]) >= _nll(p_oof[rows], y[rows]) - 1e-7:
        return p_hid, p_oof, False
    tab = _zero_table(Xv, cards, y, K, np.arange(len(y)), min_n)
    hid_new = _apply_zeros(p_hid, X[inst.hid], tab) if len(inst.hid) else p_hid
    return hid_new, oof_new, True


# ---------------------------------------------------------------- predict --

def _device():
    forced = os.environ.get("SIMULACRA_DEVICE")
    if forced:
        return forced
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _run(frame, schema):
    t_start = time.monotonic()
    inst = Instrument(frame, schema)
    deadline, phase = _instrument_deadline(inst)
    info = {"phase": phase, "n_vis": int(len(inst.vis)), "n_hid": int(len(inst.hid)),
            "targets": len(inst.targets), "budget_s": round(deadline - t_start, 1)}
    LAST_INFO.clear()
    LAST_INFO.update(info)
    targets = inst.targets
    nv = len(inst.vis)
    experts = [e.strip() for e in EXPERTS.split(",") if e.strip()]
    device = _device()

    if nv < 30 or not inst.given:
        # Too little to cross-validate: smoothed visible counts (+ gates).
        kfold = 2 if nv >= 2 else 1
    else:
        kfold = KFOLD
    folds = _fold_ids(nv, max(kfold, 1), SEED) if nv else np.zeros(0, np.int64)

    preds = {"marginal": _marginal_expert(inst, folds, max(kfold, 1))} if nv else {}
    if nv == 0:
        out = {}
        for t in targets:
            out[t] = np.full((len(inst.hid), inst.K[t]), 1.0 / inst.K[t])
        return inst, out

    X, cards, nominal = inst.given_matrix()
    timings = {}
    if nv >= 30 and inst.given:
        for name, inter in (("lr", False), ("lri", True)):
            if name not in experts:
                continue
            t0 = time.monotonic()
            try:
                frac = 0.30 if name == "lr" else 0.20
                res = _lr_expert(inst, folds, kfold, X, cards, inter,
                                 time.monotonic() + frac * max(0.0, deadline - time.monotonic()),
                                 "cpu" if device == "mps" else device)
                if res:
                    preds[name] = res
            except Exception as exc:  # fall back to other experts
                info["err_" + name] = type(exc).__name__ + ": " + str(exc)[:200]
            timings[name] = round(time.monotonic() - t0, 2)
            _log(name, timings[name])

    if "mlp" in experts and nv >= 300 and inst.given and torch is not None:
        t0 = time.monotonic()
        try:
            res = _mlp_expert(inst, folds, kfold, X, cards,
                              time.monotonic() + 0.2 * max(0.0, deadline - time.monotonic()),
                              "cpu" if device == "mps" else device)
            if res:
                preds["mlp"] = res
        except Exception as exc:
            info["err_mlp"] = type(exc).__name__ + ": " + str(exc)[:200]
        timings["mlp"] = round(time.monotonic() - t0, 2)

    # Priority of items for the expensive experts: current best OOF loss.
    yv = {t: inst.codes[t][inst.vis] for t in targets}

    def best_loss(t):
        return min(_nll(preds[m][t][0], yv[t]) for m in preds if t in preds[m])

    fm_requested = _fm_requested(experts, phase)
    n_dec_children = sum(1 for t in targets
                         if (inst.gate.get(t) or {}).get("parent") in inst.targets)
    dec_reserve = 0.0
    if "dec" in experts and n_dec_children and nv >= 200:
        # Its cost does not grow with the budget (13-16 s for 155 children on the
        # GB10 in both phases), so the reserve is absolute, ~5x that, capped.
        dec_reserve = min(0.25 * max(0.0, deadline - t_start), 10.0 + 0.4 * n_dec_children)
    fm_deadline = deadline - dec_reserve
    info["dec_reserve_s"] = round(dec_reserve, 1)
    loud = FAIL_LOUD == "1" or (FAIL_LOUD == "auto" and phase == "final")
    required = _required_kinds(fm_requested)
    if loud and fm_requested and nv >= 30 and inst.given:
        if torch is None or device not in ("cuda", "mps"):
            raise FailLoud("pretrained experts need a GPU; none available")
        for kind in required:
            if not _FM(kind, 1, device).available():
                raise FailLoud("missing weights for " + kind)
            err = _smoke(kind, device)
            if err:
                raise FailLoud("%s cannot run: %s" % (kind, err))
    fm_kinds = _fm_resolve(fm_requested, experts, device) if fm_requested else []
    info["fm_kinds"] = ",".join(fm_kinds)
    if nv >= 30 and inst.given and fm_kinds and torch is not None and device in ("cuda", "mps"):
        # Gate parents first (their forecasts feed every descendant), then by
        # current OOF loss.
        n_desc = {t: 0 for t in targets}
        for t in targets:
            seen, p = set(), (inst.gate.get(t) or {}).get("parent")
            while p in n_desc and p not in seen:
                n_desc[p] += 1
                seen.add(p)
                p = (inst.gate.get(p) or {}).get("parent")
        order = sorted(targets, key=lambda t: -best_loss(t) * (1.0 + math.log1p(n_desc[t])))
        Xf = X.astype(np.float32)
        extra = inst.numeric_extras()
        if extra is not None:
            Xf = np.concatenate([Xf, extra.astype(np.float32)], axis=1)
        cat_idx = [j for j, nom in enumerate(nominal) if nom]
        Xv, Xh = Xf[inst.vis], Xf[inst.hid]
        fm_plan, fm_cal = _fm_plan(nv, phase, SEED + 7)
        n_est = {k: (4 if phase == "final" else 1) for k in ("pfnf", "icl", "pfn", "pfnm")}
        if FM_NEST:
            n_est.update(_parse_kv(FM_NEST))
        caps = _parse_kv(FM_NEST_MAX)
        # Both phases size the ensemble to the measured speed, so development
        # runs rehearse the final configuration.
        adaptive = phase in ("final", "dev") and not FM_NEST and bool(order)
        t1 = 1.0
        if adaptive:
            # Size the ensemble to the measured speed: every in-context expert must
            # cover every item before extra estimators are worth buying.
            probe_kind = fm_kinds[0]
            probe = _FM(probe_kind, 1, device)
            times = []
            for t in order[:2]:
                tp = time.monotonic()
                try:
                    probe.run_item(Xv, yv[t], Xh, fm_plan, inst.K[t], cat_idx)
                except Exception:
                    pass
                times.append(time.monotonic() - tp)
            # Seconds per item for Fast with one estimator (the cost unit).
            t1 = max(0.02, min(times)) / FM_REL_COST.get(probe_kind, 1.0)
            info["t_probe"] = round(t1, 3)
        calib = []
        filler = ({k.strip() for k in FM_FILLER.split(",") if k.strip()}
                  if "pfn" in fm_kinds else set())
        for i, kind in enumerate(fm_kinds):
            if adaptive:
                if kind in filler:
                    n_est[kind] = 1
                else:
                    ratio = float(np.median(calib)) if (calib and REPLAN) else 1.0
                    e = _pick_nest([k for k in fm_kinds[i:] if k not in filler], len(order),
                                   t1 * ratio, max(0.0, fm_deadline - time.monotonic()), caps)
                    n_est[kind] = min(e, caps.get(kind, 8))
                info["nest_" + kind] = n_est[kind]
                if i == 0:
                    info["final_nest"] = n_est[kind]
            fm = _FM(kind, n_est[kind], device)
            t0 = time.monotonic()
            done = {}
            per_item = []
            cur = []        # seconds per item at the current ensemble size
            n_err = 0
            for idx, t in enumerate(order):
                now = time.monotonic()
                if cur:
                    est = float(np.median(cur))
                elif per_item:
                    est = float(np.median(per_item)) / 1.9
                else:
                    est = 2.0
                if now + 1.5 * est > fm_deadline:
                    break
                # Give up ensemble members before coverage: if the items left do
                # not fit at this size (slower GPU than the probe suggested),
                # halve it. Items pooled at different sizes are fine.
                if adaptive and len(cur) >= 3 and n_est[kind] > 1:
                    left = len(order) - idx
                    # Median after the first item at this size (which carries the
                    # warm-up), so one slow start does not trigger a cut.
                    per = float(np.median(cur[1:][-5:]))
                    avail = 0.95 * (fm_deadline - now)
                    if left * per > avail:
                        # Jump to the largest size whose projected cost fits.
                        e_cur = n_est[kind]
                        e_new = 1
                        for e in (8, 4, 2):
                            if e < e_cur and left * per * _nest_scale(kind, e) / \
                                    _nest_scale(kind, e_cur) <= avail:
                                e_new = e
                                break
                        n_est[kind] = e_new
                        fm = _FM(kind, e_new, device)
                        info["nest_" + kind + "_cut"] = (info.get("nest_" + kind + "_cut", "")
                                                         + "%d@%d " % (e_new, idx))
                        est = per * _nest_scale(kind, e_new) / _nest_scale(kind, e_cur)
                        cur = []
                ti = time.monotonic()
                try:
                    oof, hid = fm.run_item(Xv, yv[t], Xh, fm_plan, inst.K[t], cat_idx)
                    if np.isfinite(oof[fm_cal]).all() and np.isfinite(hid).all():
                        done[t] = (oof, hid)
                except Exception as exc:
                    n_err += 1
                    info.setdefault("err_" + kind, type(exc).__name__ + ": " + str(exc)[:200])
                    if loud and kind in required and n_err >= 3 and n_err > 0.2 * (len(per_item) + 1):
                        raise FailLoud("%s failing repeatedly: %s" % (kind, info["err_" + kind]))
                dt = time.monotonic() - ti
                per_item.append(dt)
                cur.append(dt)
            if done:
                preds[kind] = done
            if adaptive and (cur or per_item):
                planned = t1 * FM_REL_COST.get(kind, 1.8) * _nest_scale(kind, n_est[kind])
                calib.append(float(np.mean(cur or per_item)) / max(planned, 1e-6))
            timings[kind] = round(time.monotonic() - t0, 2)
            info["cover_" + kind] = len(done)
            _log(kind, "covered", len(done), "of", len(targets), timings[kind])
            if torch is not None and device == "cuda":
                torch.cuda.empty_cache()

    # ---- pooling ------------------------------------------------------
    names = list(preds.keys())

    def avail_rows(t, ms):
        r = np.ones(nv, bool)
        for m in ms:
            r &= np.isfinite(preds[m][t][0]).all(axis=1)
        return r

    # Instrument-wide prior weights from a sample of fully covered items.
    shared = {}
    full_items = [t for t in targets if all(t in preds[m] for m in names)]
    if len(names) > 1 and full_items:
        sub = full_items[:: max(1, len(full_items) // 60)]
        ws = []
        for t in sub:
            r = avail_rows(t, names)
            if r.sum() < 30:
                continue
            w, _ = _pool_weights([np.log(np.clip(preds[m][t][0][r], 1e-7, 1)) for m in names],
                                 yv[t][r])
            ws.append(w)
        if ws:
            shared[tuple(names)] = np.median(np.stack(ws), axis=0)

    out, used = {}, {}
    oof_final, oof_rows = {}, {}
    n_gate_used = n_zero_used = n_ord_used = 0
    # Parents before children so a child can use its parent's pooled forecast.
    order, placed = [], set()
    pending = list(targets)
    while pending:
        progressed = False
        for t in list(pending):
            g = inst.gate.get(t)
            if not g or g["parent"] not in pending:
                order.append(t)
                pending.remove(t)
                progressed = True
        if not progressed:
            order.extend(pending)
            break
    children_of = {}
    for t in targets:
        g = inst.gate.get(t)
        if g and g["parent"] in inst.targets:
            children_of.setdefault(g["parent"], []).append(t)
    use_dec = "dec" in experts and torch is not None and nv >= 200
    dec_preds = {}
    for t in order:
        g = inst.gate.get(t)
        if use_dec and g and g["parent"] in children_of and t not in dec_preds \
                and g["parent"] in out and time.monotonic() < deadline - 2.0:
            p = g["parent"]
            try:
                t0d = time.monotonic()
                pi_oof = np.full((nv, inst.K[p]), np.nan)
                rr = oof_rows[p]
                pi_oof[rr] = _floored(oof_final[p][rr])
                pi_hid = _floored(out[p]) if len(inst.hid) else out[p]
                dec_preds.update(_dec_expert(inst, p, children_of[p], X, cards, folds, kfold,
                                             pi_oof, pi_hid, "cpu" if device == "mps" else device))
                timings["dec"] = round(timings.get("dec", 0.0) + time.monotonic() - t0d, 2)
            except Exception as exc:
                info.setdefault("err_dec", type(exc).__name__ + ": " + str(exc)[:200])
        ms = [m for m in names if t in preds[m]]
        rows = avail_rows(t, ms)
        if rows.sum() < 30:
            ms = ["marginal"]
            rows = avail_rows(t, ms)
        if t in dec_preds:
            preds.setdefault("dec", {})[t] = dec_preds[t]
            ms = ms + ["dec"]
            rows = avail_rows(t, ms)
            if rows.sum() < 30:
                ms = ms[:-1]
                rows = avail_rows(t, ms)
        if len(ms) == 1:
            p_oof, p_hid = preds[ms[0]][t]
            w = np.array([1.0])
        else:
            prior = shared.get(tuple(ms))
            strength = (POOL_STRENGTH * 2.0 / max(1.0, math.sqrt(rows.sum()))
                        if prior is not None else 0.0)
            p_oof, p_hid, w = _pool([preds[m][t][0] for m in ms],
                                    [preds[m][t][1] for m in ms], yv[t], rows, prior, strength)
        if ORDINAL:
            p_hid, p_oof, a_used = _ordinal_post(inst, t, p_hid, p_oof, rows)
            n_ord_used += int(a_used > 0)
        if inst.gate.get(t):
            p_hid, p_oof, g_used = _given_gate_structure(inst, t, p_hid, p_oof, folds, kfold, rows)
            n_gate_used += int(g_used)
        if STRUCT_ZEROS:
            p_hid, p_oof, z_used = _structural_zeros(inst, t, X, cards, p_hid, p_oof, folds,
                                                     kfold, rows)
            n_zero_used += int(z_used)
        out[t] = p_hid
        oof_final[t] = p_oof
        oof_rows[t] = rows
        used[t] = dict(zip(ms, np.round(w, 3).tolist()))
    info["timings"] = timings
    info["gate_struct_used"] = n_gate_used
    info["zero_struct_used"] = n_zero_used
    info["ordinal_used"] = n_ord_used
    info["oof_loss"] = float(np.mean([_nll(oof_final[t][oof_rows[t]], yv[t][oof_rows[t]])
                                      for t in targets]))
    info["oof_loss_marginal"] = float(np.mean([_nll(preds["marginal"][t][0][oof_rows[t]],
                                                    yv[t][oof_rows[t]]) for t in targets]))
    info["weights_example"] = {t: used[t] for t in targets[:3]}
    agg = {}
    for t in targets:
        for m, w in used[t].items():
            agg.setdefault(m, []).append(w)
    info["weights_mean"] = {m: round(float(np.mean(v)), 3) for m, v in agg.items()}
    info["weights_n"] = {m: len(v) for m, v in agg.items()}
    info["seconds"] = round(time.monotonic() - t_start, 1)
    LAST_INFO.update(info)
    _log("done", info)
    return inst, out


def _safe(vec, K):
    v = np.asarray(vec, dtype=np.float64)
    if v.shape != (K,) or not np.isfinite(v).all() or (v < 0).any() or v.sum() <= 0:
        return np.full(K, 1.0 / K)
    return v / v.sum()


def predict(frame, schema):
    records = schema["items"]
    items = [n for n, r in records.items()
             if r.get("class") in ("GIVEN", "PREDICT") and n in frame.columns]
    try:
        inst, out = _run(frame, schema)
        row_pos = {r: i for i, r in enumerate(inst.hid)}
    except FailLoud:
        raise
    except Exception as exc:  # last-resort: smoothed visible counts
        LAST_INFO["fatal"] = type(exc).__name__ + ": " + str(exc)[:300]
        inst, out, row_pos = None, {}, {}
    gated_value = schema.get("gated_value", "NA_GATED")
    grid = frame[items].to_numpy(dtype=object)
    blank = pd.isna(grid)
    fallback = {}
    vectors = []
    rows, cols = np.nonzero(blank)
    for r, c in zip(rows, cols):
        name = items[c]
        opts = list(records[name]["values"]) + ([gated_value] if records[name].get("gate") else [])
        K = len(opts)
        vec = None
        if name in out and r in row_pos:
            vec = out[name][row_pos[r]]
        if vec is None:
            if name not in fallback:
                counts = frame[name].value_counts()
                w = np.array([counts.get(o, 0) + 0.5 for o in opts], float)
                fallback[name] = w / w.sum()
            vec = fallback[name]
        vectors.append(_safe(vec, K).tolist())
    return vectors
