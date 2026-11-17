"""Assemble a submission directory + ZIP from claude/sub/<version>.

python build_zip.py --sub claude/sub/v5 --weights WEIGHTS_DIR --out dist/claude-v5
    [--with-full-pfn] [--hub-pfn]

--hub-pfn writes models.txt naming Prior-Labs/tabpfn_3_5, so the platform
pre-downloads the full TabPFN-3.5 checkpoint into the Hugging Face cache at build
time (main.py only uses it at the pinned commit / hash).

Copies main.py, requirements.txt, licence notices and the bundled checkpoints
(verifying SHA-256), then zips from inside the directory so main.py sits at the
archive root. Prints the ZIP size and SHA-256.
"""
import argparse
import hashlib
import os
import shutil
import zipfile

CHECKPOINTS = {
    "tabicl-classifier-v2-20260212.ckpt":
        "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
    "tabpfn-v3.5-fast-20260909.safetensors":
        "14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550",
    "tabpfn-v3.5-20260909.safetensors":
        "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
}


ICL_NOTICE = """
2. TabICL v2 (tabicl-classifier-v2-20260212.ckpt), named in models.txt and
   pre-downloaded by the platform, from https://huggingface.co/jingang/TabICL at
   commit 4dcd344ece2c00be9e831fdd35bed57b5ad83e19 (used only at that commit or if
   its SHA-256 matches), BSD-3-Clause (see LICENSE.tabicl).
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def selfcheck(main_py):
    """Refuse to package a main.py that calls an undefined module-level helper."""
    import ast
    tree = ast.parse(open(main_py).read())
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    names |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name)}
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id.startswith("_")}
    missing = called - names
    if missing:
        raise SystemExit("selfcheck: undefined helpers called in main.py: %s" % sorted(missing))
    for required in ("predict", "_dec_expert", "_lr_expert", "_FM", "_pool", "_pfn_full_path"):
        if required not in names:
            raise SystemExit("selfcheck: %s missing from main.py" % required)
    print("selfcheck ok (%d top-level definitions)" % len(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-full-pfn", action="store_true")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--hub-pfn", action="store_true",
                    help="write models.txt so the platform pre-downloads full TabPFN-3.5")
    ap.add_argument("--final-path-test", action="store_true",
                    help="dev rehearsal build: final FM settings + fail-loud under the dev budget")
    ap.add_argument("--hub-icl", action="store_true",
                    help="also name jingang/TabICL in models.txt (TabICL v2 from the platform's "
                         "download when it is not bundled)")
    ap.add_argument("--fm-order", default="",
                    help="default in-context expert order for both phases, e.g. pfn,icl")
    ap.add_argument("--slim", action="store_true",
                    help="no bundled weights: every checkpoint comes via models.txt (implies "
                         "--no-weights --hub-pfn; NOTICE.slim becomes NOTICE)")
    args = ap.parse_args()
    if args.slim:
        args.no_weights = args.hub_pfn = True
    selfcheck(os.path.join(args.sub, "main.py"))
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)
    if not args.no_weights:
        os.makedirs(os.path.join(args.out, "weights"))
    files = ["main.py", "requirements.txt", "NOTICE", "LICENSE.tabpfn-3.5", "LICENSE.tabicl"]
    if args.slim:
        files = ["main.py", "requirements.txt", "LICENSE.tabpfn-3.5"]
        notice = open(os.path.join(args.sub, "NOTICE.slim")).read()
        if args.hub_icl:
            files.append("LICENSE.tabicl")
            notice += ICL_NOTICE
        with open(os.path.join(args.out, "NOTICE"), "w") as fh:
            fh.write(notice)
    for name in files:
        src = os.path.join(args.sub, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, name))
    if args.hub_pfn or args.hub_icl:
        # One bare repo id per line: the platform reads it a line at a time.
        repos = (["Prior-Labs/tabpfn_3_5"] if args.hub_pfn else []) + (
            ["jingang/TabICL"] if args.hub_icl else [])
        with open(os.path.join(args.out, "models.txt"), "w") as fh:
            fh.write("".join(r + "\n" for r in repos))
        print("models.txt:", ", ".join(repos))
    if args.fm_order:
        path = os.path.join(args.out, "main.py")
        text = open(path).read()
        for name in ("FM_ORDER_FINAL", "FM_ORDER_DEV"):
            old = '%s = _env("SIMULACRA_%s", "pfn")' % (name, name)
            assert text.count(old) == 1, old
            text = text.replace(old, '%s = _env("SIMULACRA_%s", "%s")' % (name, name, args.fm_order))
        open(path, "w").write(text)
        print("fm order:", args.fm_order)
    if args.final_path_test:
        path = os.path.join(args.out, "main.py")
        text = open(path).read()
        assert "FINAL_PATH_TEST = False" in text
        open(path, "w").write(text.replace("FINAL_PATH_TEST = False", "FINAL_PATH_TEST = True", 1))
        print("final-path rehearsal build")
    if not args.no_weights:
        for name, digest in CHECKPOINTS.items():
            if name == "tabpfn-v3.5-20260909.safetensors" and not args.with_full_pfn:
                continue
            src = os.path.realpath(os.path.join(args.weights, name))
            got = sha256(src)
            if digest and got != digest:
                raise SystemExit("hash mismatch for %s: %s" % (name, got))
            print("weight", name, got)
            shutil.copy(src, os.path.join(args.out, "weights", name))
    zpath = args.out.rstrip("/") + ".zip"
    if os.path.exists(zpath):
        os.remove(zpath)
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(args.out):
            for f in sorted(files):
                full = os.path.join(root, f)
                arc = os.path.relpath(full, args.out)
                if arc.endswith(".safetensors") or arc.endswith(".ckpt"):
                    zf.write(full, arc, compress_type=zipfile.ZIP_STORED)
                else:
                    zf.write(full, arc)
    print("zip", zpath, os.path.getsize(zpath), sha256(zpath))


if __name__ == "__main__":
    main()
