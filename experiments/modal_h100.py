"""H100 runs on Modal that mirror the hosted runtime: Python 3.13, one H100,
8 CPU cores and 16 GB RAM as hard limits, and the full TabPFN-3.5 repo
pre-downloaded into HF_HUB_CACHE at image build exactly as the platform does for
models.txt (snapshot_download of the whole repo). The submission directory is
assembled like the ZIP: main.py + bundled Fast/TabICL weights + models.txt.

    SIM_BUNDLE=/path/to/bundle modal run --detach claude/lab/modal_h100.py \
        --tasks bench,final,dev --tag h100a

The bundle holds sub/ (submission files without weights), lab/ (evaluate.py,
fm_bench.py), kit/ (official kit) and data/ (synthetic sandbox + public proxies).
Logs and JSON results go to the Modal volume `simulacra-claude` under /<tag>/.
"""
import os

import modal

BUNDLE = os.environ.get("SIM_BUNDLE", "/nonexistent-bundle")
GPU = os.environ.get("SIM_GPU", "H100!")

app = modal.App("simulacra-claude-h100")
vol = modal.Volume.from_name("simulacra-claude", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("time")
    .pip_install("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "torch", "torchvision",
                 "transformers", "huggingface_hub", "accelerate", "safetensors", "PyYAML")
    .pip_install("tabpfn==9.1.0", "tabicl==2.2.0")
    .env({"HF_HUB_CACHE": "/hfcache", "PYTHONUNBUFFERED": "1"})
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download as s; "
        "print(s('Prior-Labs/tabpfn_3_5', cache_dir='/hfcache'))\"",
        "python -c \"from huggingface_hub import hf_hub_download as h; "
        "print(h('jingang/TabICL', 'tabicl-classifier-v2-20260212.ckpt', "
        "revision='4dcd344ece2c00be9e831fdd35bed57b5ad83e19', cache_dir='/iclcache'))\"",
    )
    # Slim builds name jingang/TabICL in models.txt too: whole-repo download, as the
    # platform does for each models.txt line.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download as s; "
        "print(s('jingang/TabICL', cache_dir='/hfcache'))\"",
    )
    .add_local_dir(BUNDLE, "/bundle")
)

SUMMARY = r'''
import json, sys
d = json.load(open(sys.argv[1]))
print("import %.1fs total %.1fs" % (d["import_seconds"], d["total_seconds"]))
for r in d["runs"]:
    i = r.get("info", {})
    print(r["data"].split("/")[-1], "skill %.5f" % r["skill"], "%.1fs" % r["seconds"],
          {k: i.get(k) for k in ("phase", "budget_s", "dec_reserve_s", "fm_kinds", "t_probe",
                                  "nest_pfn", "nest_icl", "nest_pfnf", "cover_pfn", "cover_icl",
                                  "cover_pfnf", "err_pfn", "err_icl", "err_pfnf", "timings",
                                  "weights_mean")})
'''


@app.function(image=image, gpu=GPU, cpu=(8.0, 8.0), memory=(16384, 16384),
              timeout=3 * 3600, volumes={"/out": vol})
def run(tasks: str, tag: str):
    import glob
    import shutil
    import subprocess

    out = "/out/" + tag
    os.makedirs(out, exist_ok=True)
    sub = "/tmp/sub"
    shutil.copytree("/bundle/sub", sub)
    if os.path.exists(sub + "/models.txt"):
        # Slim build: no bundled weights, every checkpoint from the HF cache.
        print("slim build; models.txt:", open(sub + "/models.txt").read().split(), flush=True)
    else:
        os.makedirs(sub + "/weights", exist_ok=True)
        snap = glob.glob("/hfcache/models--Prior-Labs--tabpfn_3_5/snapshots/*/")[0]
        shutil.copy(os.path.realpath(snap + "tabpfn-v3.5-fast-20260909.safetensors"),
                    sub + "/weights/tabpfn-v3.5-fast-20260909.safetensors")
        icl = glob.glob("/iclcache/models--jingang--TabICL/snapshots/*/"
                        "tabicl-classifier-v2-20260212.ckpt")[0]
        shutil.copy(os.path.realpath(icl), sub + "/weights/tabicl-classifier-v2-20260212.ckpt")
        with open(sub + "/models.txt", "w") as fh:
            fh.write("Prior-Labs/tabpfn_3_5\n")
    fp = "/tmp/sub_fp"
    shutil.copytree(sub, fp)
    text = open(fp + "/main.py").read()
    assert "FINAL_PATH_TEST = False" in text
    open(fp + "/main.py", "w").write(text.replace("FINAL_PATH_TEST = False",
                                                  "FINAL_PATH_TEST = True", 1))
    shutil.copytree("/bundle", "/work")
    with open("/tmp/summary.py", "w") as fh:
        fh.write(SUMMARY)
    log = open(out + "/log.txt", "a")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SIMULACRA_")}

    def sh(cmd):
        print("$ " + cmd, flush=True)
        log.write("$ " + cmd + "\n")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env=env, cwd="/work")
        for line in p.stdout:
            if "warn" in line.lower():
                continue
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        p.wait()
        vol.commit()
        return p.returncode

    sh("nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv; nproc; "
       "cat /sys/fs/cgroup/memory.max 2>/dev/null; cat /sys/fs/cgroup/cpu.max 2>/dev/null; "
       "python -c 'import sys, torch, tabpfn, numpy, pandas, sklearn; print(sys.version); "
       "print(torch.__version__, torch.version.cuda, tabpfn.__version__, numpy.__version__, "
       "pandas.__version__, sklearn.__version__)'")
    D = "data/sandbox/unicef data/sandbox/world_bank data/sandbox/unhcr"
    for task in tasks.split(","):
        if task == "bench":
            sh("python lab/fm_bench.py --sub %s --data %s --phase 2 --items 6 --kit kit "
               "--out %s/bench.json" % (sub, D, out))
        elif task == "cuttest":
            for budget in ("150", "100"):
                sh("SIMULACRA_BUDGET=%s SIMULACRA_FORCE_PHASE=final python lab/cut_test.py %s "
                   "data/sandbox/world_bank kit" % (budget, sub))
        elif task == "bench8":
            sh("python lab/fm_bench.py --sub %s --data %s --phase 2 --items 6 --kit kit "
               "--kinds pfn --nest 8 --out %s/bench8.json" % (sub, D, out))
        elif task == "twin":
            # Panel-twin lab (twin_proxy.py) next to its no-twin source: the same pools
            # with fixed ensemble sizes and no time limit; in-context predictions are
            # cached, so each pool after the first only refits the pooling.
            base = ("SIMULACRA_BUDGET=100000 SIMULACRA_FORCE_PHASE=final "
                    "SIMULACRA_EXPERTS=marginal,lr,mlp,pfn,icl,dec SIMULACRA_FM_NEST=pfn=8,icl=4 ")
            for data in ("data/proxy/reachsmtwin_s1", "data/proxy/reachsm_s1"):
                for order in ("pfn,icl", "pfn", "icl"):
                    res = "%s/twin_%s" % (out, order.replace(",", "_"))
                    sh(base + "SIMULACRA_FM_ORDER_FINAL=%s python lab/evaluate.py --sub %s "
                       "--data %s --phase 2 --kit kit --fm-cache /tmp/fmc --save-logp %s 2>&1 "
                       "| grep -E 'skill|Traceback|Error'" % (order, sub, data, res))
            sh("python lab/compare.py %s/twin_pfn %s/twin_pfn_icl %s/twin_icl" % (out, out, out))
        elif task in ("final", "dev", "finalpath"):
            phase = 2 if task == "final" else 1
            target = fp if task == "finalpath" else sub
            sh("/usr/bin/time -v python lab/evaluate.py --sub %s --data %s --phase %d --kit kit "
               "--out %s/%s.json 2>&1 | grep -E 'skill|total|Traceback|Error|Maximum resident|"
               "Elapsed'" % (target, D, phase, out, task))
            sh("python /tmp/summary.py %s/%s.json" % (out, task))
    vol.commit()


@app.local_entrypoint()
def main(tasks: str = "bench,final,dev", tag: str = "h100a"):
    run.remote(tasks, tag)
