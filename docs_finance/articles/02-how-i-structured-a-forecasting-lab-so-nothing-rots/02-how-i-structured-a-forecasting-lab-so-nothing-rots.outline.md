# Outline — How I structured a forecasting lab so nothing rots

## One-line pitch
The repo, not the models. Numbered self-contained experiments, YAML as single source of truth, shared `src/btc_ai/eval/` so every leaderboard row is comparable, Makefile-as-API, uv for envs. Written for ML practitioners who keep meaning to clean up their notebook graveyard.

## Audience
ML practitioners, quant researchers, and ML-curious engineers who have ever opened a folder named `notebooks_v3_FINAL_2/` and felt physical pain. Series readers who want to know *how* the leaderboard rows in articles 1, 3, 4, 6, 9 are made comparable before trusting them.

## Thesis
A research repo only stays useful if every experiment can be re-run, re-compared, and audited against the same baseline a year later. The way to enforce that isn't a tool — it's a small set of structural decisions made before you write the first model.

## Structure

### 1. The hook — the lab notebook is the product
- The thing I'm shipping with this series isn't a model. It's a notebook.
- "Lab notebook" not "leaderboard": negative results count, partial results count, dead branches count.
- This article is the engineering scaffolding behind every other article in the series.

### 2. Six rules I'd give my past self
Numbered, opinionated, short. Each rule gets a paragraph and a code/file pointer.

1. **Numbered, self-contained experiments.** `experiments/${index}_${name}/`, never edit a finished one. `cp -r experiments/01_baseline experiments/02_arima` is the workflow.
2. **YAML is the single source of truth.** Both `run.py` and `scripts/fetch_data.py` read the same `config.yaml`. There is no "the run I did Tuesday with the 48-bar setting" — it's in the YAML or it didn't happen.
3. **Shared `src/btc_ai/` for loader, splitter, metrics.** Everyone uses the same `mae()`, the same `directional_accuracy()`, the same time-ordered split. If the floor moves, every model is judged against the new floor automatically.
4. **Makefile is the canonical command surface.** `make 02_arima`, never `python run.py`. One-liner README, fewer "wait, what flags?" mistakes.
5. **`uv` for envs and runs.** `uv.lock` is committed; `.venv/` is not. New machine = `brew install uv && uv sync && make fetch`. No "works on my laptop".
6. **Every experiment ships a report — even the negative ones.** README + `metrics.json` + `predictions.parquet` + `plot.png`, in `results/`. A bad result in a tidy folder is more useful than a good result in `notebooks/scratch_v8.ipynb`.

### 3. The shape of an experiment folder
Walk through `experiments/02_arima/` as the canonical example. What each file is for, what gets gitignored (just the data and `.venv/`), what gets committed (everything else, including the `results/`).

### 4. Why YAML > flags
- A bad day in this lab is "I changed `input_chunk_length` from 24 to 48 yesterday and forgot." With YAML, the diff is visible in git.
- Inline comments matter — every field has a `# what does this do` annotation. You're writing the YAML once but reading it for years.
- No auto-pick defaults. If the field is ambiguous (`period: monthly` vs `daily`, `backend: chronos` vs `timesfm`), the user has to specify it. The lab refuses to guess.

### 5. The shared eval module is the load-bearing decision
This is the rule that makes the leaderboard real.
- One `mae()`, one `rmse()`, one `directional_accuracy()`, one `annualized_sharpe()`. Every experiment imports them ([src/btc_ai/eval/metrics.py](../../../src/btc_ai/eval/metrics.py)).
- The `directional_accuracy(y_true, y_pred, ref)` signature is the subtle bit — it takes a `ref` argument so a model that "predicts the last value" returns NaN by design. Article 1 explained why that matters; this one shows where the decision lives in code.
- `periods_per_year` is a lookup table, not a `1h` magic number. Change interval, the Sharpe annualization tracks it.

### 6. Reproducibility on someone else's laptop
The exact thing a reader does:
```
brew install uv
uv sync
make fetch
make 01_baseline
make 02_arima
make 03_xgboost
…
```
The point: the article 1 numbers, on a fresh clone, on a different machine. That's the bar.

### 7. Negative results as first-class artifacts
- 03_xgboost (XGBoost dir_acc 0.4872 — worse than chance) ships exactly the same files as 02_arima (sharpe +7.28). Both have a README, both have a metrics.json, both are linked from the leaderboard.
- The repo does not delete failed experiments. They're the most informative folders in the tree.
- The series posts (especially Articles 4 and 9) are essentially "let me explain why this folder underperformed" — those posts only exist because the underperformance is *captured*, not lost.

### 8. What this is not
- Not a framework. There's no `MyExperimentBase` class to inherit. Each `run.py` is a flat script.
- Not MLflow / W&B / DVC. For 7 experiments × 1 dataset × 1 person, a folder of `metrics.json` files is enough. (Bring out the heavier tools when you have multiple humans or multiple datasets.)
- Not "production". This repo is a notebook, not a service.

### 9. Closing — the discipline pays off in Article 6
Tease forward: Article 6 will compare the same model on two different test windows and find the verdict flips. That comparison is only believable because *the entire pipeline* is shared between runs. The discipline isn't aesthetic; it's what lets the rest of the series make claims.

## Key code/file references
- [README.md](../../../README.md) — Repository Layout + Conventions
- [Makefile](../../../Makefile) — every target wraps `uv run`
- [src/btc_ai/eval/metrics.py](../../../src/btc_ai/eval/metrics.py) — the load-bearing module
- [experiments/02_arima/config.yaml](../../../experiments/02_arima/config.yaml) — YAML with inline comments, no auto-pick
- [pyproject.toml](../../../pyproject.toml) + [uv.lock](../../../uv.lock) — env definition

## Tone notes
- Lived-in, not preachy. Each rule has a "I made this mistake first" tone.
- Concrete > abstract. Real file names, real targets, real `make` invocations.
- This is the only purely engineering post in the series; the rest are model results. Lean on that contrast.

## Length target
~1,500–1,800 words. Long enough to walk through the rules, short enough to be re-read at the start of someone else's lab project.
