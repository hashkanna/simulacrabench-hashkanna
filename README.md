# SimulacraBench: test-phase submission of team HashKanna

Code and technical report for team **HashKanna**'s entry in the
[SimulacraBench](https://simulacrabench.org) competition (NeurIPS 2026 Competition Track,
Codabench competition 17822).

| Leaderboard (two decimals, as displayed) | Grand Prize | UNICEF | World Bank | UNHCR |
|---|---|---|---|---|
| Test phase, submission 1053554 (15 November 2026) | 0.49 | 0.34 | 0.24 | 0.88 |

The method forecasts every blank cell by its conditional marginal given the respondent's GIVEN
columns. Everything is fitted at run time, inside `predict()`, on the visible respondents: smoothed
marginals, a multi-task logistic regression, a multi-task MLP, the frozen tabular foundation model
TabPFN-3.5 used in context (up to eight ensemble members), and a gate-decomposition model for
skip-logic chains. The experts are pooled log-linearly per item, and three post-processing steps
are kept only where they do not raise the out-of-fold loss. The report in `paper/` describes all
of it; Appendix A of the report lists every setting.

## Layout

| Path | Contents |
|---|---|
| `submission/` | The five files of the submitted archive, byte for byte. `main.py` is the whole method. |
| `archive/claude-v5k-slim-dev.zip` | The archive exactly as it was uploaded. |
| `SHA256SUMS` | SHA-256 of the five files and of the archive. |
| `experiments/` | Scripts behind the offline proxy experiments and timing runs (Appendix B of the report). See `experiments/README.md`. |
| `paper/` | The technical report: `report.pdf` and its LaTeX source. Building it needs `neurips_2026.sty` from the NeurIPS 2026 formatting instructions beside `report.tex`; that file is not copied here. |

## Rebuilding the submission archive

```bash
cd submission && zip -r ../submission.zip .
```

This packs the same five files as the uploaded archive. A zip made today differs from the uploaded
one in its container bytes (timestamps, compressor), so compare the files, not the archives:

```bash
sha256sum -c SHA256SUMS        # macOS: shasum -a 256 -c SHA256SUMS
```

## Running it

**On the platform.** Upload the archive. The platform installs `requirements.txt`, downloads the
Hugging Face repository named in `models.txt` into `HF_HUB_CACHE` when it builds the image, and
calls `predict()` once per instrument with no network, on one H100 GPU with 8 CPU cores and 16 GB
of RAM. Its time limit covers the three instruments together: 3,600 s in the test phase and 900 s
in the development phase. `main.py` plans each instrument within its share of that limit.

**Locally, with the competition kit.** You need Linux, one NVIDIA GPU, and the
[competition kit](https://github.com/SituatedEvals/public). The kit ships no microdata;
`make_sandbox.py` writes synthetic respondents with the shapes of the real instruments, so a local
run checks that the code runs as submitted, and the skill it prints is not the leaderboard score.

```bash
git clone https://github.com/SituatedEvals/public.git kit
git clone https://github.com/hashkanna/simulacrabench-hashkanna.git
cd kit
pip install -r requirements.txt huggingface_hub

# 1. TabPFN-3.5 at the pinned commit, laid out as the platform does it (2 GB, needs the network once)
export HF_HUB_CACHE=$PWD/_hf
python -c "from huggingface_hub import snapshot_download; snapshot_download('Prior-Labs/tabpfn_3_5', revision='06bf2ba35c80a92a3b9abb436b99cf49e7a0365e', cache_dir='$HF_HUB_CACHE')"

# 2. Sandbox data for one instrument (unicef, world_bank or unhcr)
python make_sandbox.py --schema data/unicef.json --out _sandbox/unicef

# 3. The test-phase layout: TRAIN and DEV rows visible, TEST rows scored, 3,600 s limit
python score.py --submission ../simulacrabench-hashkanna/submission \
    --data _sandbox/unicef --schema data/unicef.json --phase 2
```

`--phase 1` gives the development-phase layout (TRAIN visible, DEV scored, 900 s limit). The code
picks its test-phase plan when the number of held-out rows equals the schema's test split, which is
what `--phase 2` produces on full-size sandbox data.

Things to know about a local run:

- `score.py` builds a fresh virtual environment from `submission/requirements.txt` (the only step
  that needs the network) and then calls `predict()` with sockets disabled. PyTorch in that
  environment is whatever wheel `pip` picks for your machine, and it must be a CUDA build. If it is
  not, install `tabpfn==9.1.0` and a CUDA build of PyTorch into your own environment and give the
  kit a copy of the directory without `requirements.txt`; the kit then runs the code in the
  interpreter that runs `score.py`:

  ```bash
  cp -r ../simulacrabench-hashkanna/submission /tmp/submission-noreq
  rm /tmp/submission-noreq/requirements.txt
  python score.py --submission /tmp/submission-noreq \
      --data _sandbox/unicef --schema data/unicef.json --phase 2
  ```

- In the test-phase plan the code stops with an error rather than degrade when there is no GPU,
  when the TabPFN-3.5 checkpoint is not found, or when a smoke test of the model fails. The kit's
  `--docker` mode starts a container without a GPU and without the weights, so it cannot run this
  submission.
- Only `tabpfn-v3.5-20260909.safetensors` is needed. In the Hugging Face cache, `main.py` takes it
  from the pinned commit's snapshot folder, or from another snapshot if its SHA-256 matches.
  Instead of the cache you can put that one file in a directory and set
  `SIMULACRA_WEIGHTS=/that/directory`; a file found there is used without a hash check, so check
  it against the table below yourself.
- `SIMULACRA_VERBOSE=1` makes `main.py` print the planner's progress to stderr. The kit's
  `score.py` captures the submission's output and shows it only when the run fails;
  `experiments/evaluate.py` calls `predict()` in-process and lets it through. The other
  `SIMULACRA_*` variables that `main.py` reads are development switches. The platform run set none
  of them.
- The time budget assumes an H100. On a slower GPU the planner uses fewer TabPFN ensemble members
  (8, 4, 2 or 1), so the forecasts can differ slightly from an H100 run. Seeds are fixed.

## Environment

- Platform image: Python 3.13 with numpy, pandas and torch pre-installed, plus
  `requirements.txt` (`tabpfn==9.1.0`, `tabicl==2.2.0`). TabICL was evaluated during development
  and is installed by `requirements.txt`, but the submitted configuration never imports it.
- One CUDA GPU. In H100 runs of an earlier build with the same expert pool, peak host memory was
  about 8 GB of the 16 GB allowed.

## Pretrained weights

No weights are stored in this repository.

| Model | File | SHA-256 | Role |
|---|---|---|---|
| TabPFN-3.5 | `tabpfn-v3.5-20260909.safetensors` (876,027,932 bytes) | `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3` | In-context expert for every item |
| TabPFN-3.5-Fast | `tabpfn-v3.5-fast-20260909.safetensors` | `14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550` | Fallback if the full checkpoint is missing, development phase only |

Both are files of the Hugging Face repository
[`Prior-Labs/tabpfn_3_5`](https://huggingface.co/Prior-Labs/tabpfn_3_5) at commit
`06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`. They are used frozen, for in-context inference only.
`main.py` uses a file only if it sits in that commit's snapshot folder or its SHA-256 matches. The
weights are under the TabPFN-3.5 License v1.0 (non-commercial); a copy is in
`submission/LICENSE.tabpfn-3.5`.

## Data

The repository holds no SimulacraBench microdata and no other survey data. The method sees
SimulacraBench rows only at run time, as passed to `predict()`. Three public surveys were used
offline as proxy instruments for choosing components; none of them enters the submitted pipeline.
`experiments/README.md` lists them with their sources.

## Notes

- `submission/main.py` is the file that ran, unedited. Its module docstring is older than the final
  build: it mentions TabICL and bundled weights, but the submitted configuration uses TabPFN-3.5
  alone among the pretrained models and takes the weights from the Hugging Face cache (see
  `submission/NOTICE` and `submission/models.txt`).
- This repository was exported after the test-phase run from a private development repository,
  which is why its history is short.
- The code and the experiments were produced with AI coding assistants under the author's
  direction, as the report's Appendix C states.

## Licence

The code in this repository is under the MIT licence (`LICENSE`). The TabPFN-3.5 weights are not
part of the repository and keep their own licence (above). The competition kit is a separate
project under its own MIT licence.
