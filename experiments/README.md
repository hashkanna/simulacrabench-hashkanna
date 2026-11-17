# Experiments

These scripts built the proxy instruments and ran the comparisons and timings reported in
Appendix B of the report. They are published as they were run. None of them is needed to run the
submission, and none of them touches SimulacraBench microdata.

## Read this first

- The shell scripts were run from a workspace directory on the development machine in which this
  directory was `claude/lab`, the competition kit was `simulacra-kit`, proxy instruments were under
  `data/proxy`, sandbox data under `data/sandbox`, and results under `results`. Their paths have
  not been rewritten. The Python scripts take their paths as arguments.
- Several scripts name other builds of the submission (`subs/v5e`, `subs/v5g`, `subs/v5h`,
  `dist/claude-v5l-icl-dev`, and `claude/sub/v5`, the development tree that `build_zip.py` packs).
  Those builds are not in this repository; they are other versions of `submission/main.py`.
  Experts and ensemble sizes are chosen through `SIMULACRA_*` environment variables, which the
  submitted `main.py` still reads, so the scripts show which configurations were compared.
- The "earlier baseline" row of the report's ablation table is the team's earlier
  development-phase submission. It is not in this repository.
- Runs with TabICL used `tabicl==2.2.0` and the checkpoint `tabicl-classifier-v2-20260212.ckpt`
  from the Hugging Face repository `jingang/TabICL` (commit
  `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`). The submitted `main.py` looks for that file in the
  `SIMULACRA_WEIGHTS` directory only.
- `proxy_wgm.py` takes `--seeds`, not `--seed` as its docstring says.

## Proxy instruments

Each builder converts a public survey into the competition's file format (a `respondents.parquet`
with TRAIN, DEV and TEST roles and a `schema.json` with GIVEN and PREDICT items and gates). The
surveys are not redistributed here.

| Script | Source | Proxy for |
|---|---|---|
| `proxy_wgm.py` | Wellcome Global Monitor 2018, <https://wellcome.org/sites/default/files/wgm2018-dataset-crosstabs-all-countries.xlsx> (149,014 respondents, 144 countries) | UNICEF: vaccine and science attitudes in two sets of 20 low- and middle-income countries |
| `proxy_riasec.py` | Open Psychometrics, "Answers to the Holland Code (RIASEC) Test", <http://openpsychometrics.org/_rawdata/RIASEC_data12Dec2018.zip> (145,828 respondents) | World Bank: binned scale scores and raw items, in 6,927-row samples |
| `proxy_reach.py` | REACH Initiative, "Iraq - CCCM IDP Camp Profiling VII", <https://data.humdata.org/dataset/cccm-idp-camp-profiling-vii> (5,479 households, CC BY-IGO) | UNHCR: a household survey of displaced households with skip logic |
| `subset_items.py` | a proxy instrument | A smaller copy with a random subset of the PREDICT items. The `reachsm_*` instruments in the scripts are the REACH proxy reduced to about half of its items. |
| `twin_proxy.py` | a proxy instrument | The repeated-household experiment: adds noisy copies of some proxy respondents. It uses proxy rows only. |

`proxy_reach.py` reads a pandas pickle of the REACH workbook's household sheet (5,479 rows and
442 columns). This makes it from the downloaded workbook:

```python
import openpyxl, pandas as pd

book = openpyxl.load_workbook("reach_iraq_cccm.xlsx", read_only=True)
rows = list(book["CCCM Camp Profiling VII dataset"].iter_rows(values_only=True))
names, seen = [], {}
for name in map(str, rows[0]):  # one column name occurs twice; the second becomes "<name>__1"
    seen[name] = seen.get(name, -1) + 1
    names.append(name if seen[name] == 0 else "%s__%d" % (name, seen[name]))
pd.DataFrame(rows[1:], columns=names).to_pickle("reach_iraq_cccm.pkl")
```

## Evaluation

| Script | Purpose |
|---|---|
| `evaluate.py` | Runs a submission directory's `predict()` in-process on one or more instruments and grades it with the kit's own functions. `--kit` gives the kit's path, `--sub` the submission directory. It can cache the in-context experts' forecasts on disk so that pooling variants are cheap to compare. |
| `compare.py` | Paired skill differences between saved runs, with a bootstrap standard error clustered by respondent. |
| `twin_breakdown.py` | Splits a repeated-household comparison by whether the scored respondent's copy was visible. |
| `run_matrix.sh` | One configuration on several instruments, saving per-cell log scores. |
| `final_cfg_matrix.sh` | Development plan against test-phase plan, then with the full TabPFN-3.5 checkpoint added. |
| `fm_matrix.sh`, `fm_matrix2.sh`, `fm_matrix3.sh`, `fm_matrix4.sh` | Which in-context experts to pool and with how many ensemble members; five or ten folds. |
| `fm_pool.sh` | Strength of the pooling ridge. |
| `twin_lab.sh` | TabPFN-3.5, TabICL and both, on the repeated-household proxy. |

## Timing and build checks

| Script | Purpose |
|---|---|
| `fm_timing.py`, `fm_bench.py` | Cost per item of the in-context experts on sandbox data of the real shapes. |
| `cut_test.py` | Forces eight ensemble members on a budget that cannot afford them, to exercise the planner's mid-run cut. |
| `modal_h100.py` | The rented-H100 runs: Python 3.13, one H100, 8 CPU cores and 16 GB of RAM as hard limits. |
| `build_zip.py` | Assembles a submission directory and its zip from the development tree. `--slim` gave the submitted archive, which bundles no weights. |
| `dry_run.sh` | One import and three `predict()` calls in the grader's order (UNICEF, World Bank, UNHCR) with the real phase budgets, on sandbox data. |
| `official_check.sh` | The kit's `check_submission_zip.py` and `score.py` on a built archive, in an interpreter that already has the pinned packages. |
| `rehearse.sh`, `overnight.sh` | Chains of the two above for a new build. |
