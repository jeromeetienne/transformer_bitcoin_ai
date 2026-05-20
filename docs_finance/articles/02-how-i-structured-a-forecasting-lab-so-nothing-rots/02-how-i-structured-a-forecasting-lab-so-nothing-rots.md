# How I structured a forecasting lab so nothing rots

*Article 2 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

This post is about the repository, not the models. If you're here for transformer architectures and Sharpe ratios, the next eleven articles are for you. This one is the scaffolding.

I keep trying to clean up other people's research notebook graveyards — `notebooks_v3_FINAL_2/`, `lstm_attempt_works_maybe.ipynb`, `metrics_old.json` — and I keep failing. The reason is that those repos were never structured to be re-run. They were structured to be *finished*. Once the model worked once, the notebook ossified into a screenshot.

The lab notebook I'm shipping with this series is structured the other way around. **Every experiment can be re-run by anyone with `uv` installed, and every leaderboard row is comparable to every other leaderboard row by construction.** This article is six rules that get me there.

If that sounds modest, the test is Article 6. In Article 6 I'll claim that the same model on two different test windows produces opposite verdicts. That claim is only credible if you trust that the only thing that changed between the runs was the test window. Everything in this post exists to make that trust earned, not asked for.

---

## Rule 1 — Numbered, self-contained experiments

Every experiment lives in `experiments/${index}_${name}/`. The numbering is chronological, not categorical. I never edit a finished one. The workflow is:

```
cp -r experiments/01_baseline experiments/02_arima
# now edit 02_arima/run.py and 02_arima/config.yaml
```

That's it. There is no `experiments/baselines/` or `experiments/deep/` directory; the index *is* the categorization. When Article 6 claims that "the LSTM was broken on Q1 2024 but competitive on Jan–Nov 2024", it is comparing two committed runs of [04_lstm](../../experiments/04_lstm/), each with a fully reproducible `config.yaml`. The chronology is the leaderboard.

The non-negotiable rule: **never edit a past experiment**. If 02_arima needs to change, that's a 02b or a new index. Past experiments are read-only artifacts. This is the single most important habit in the lab. It's also the one I want to break the most often.

---

## Rule 2 — YAML is the single source of truth

Each experiment has a `config.yaml`. Both `run.py` and `scripts/fetch_data.py` read the same file. There is no "the run I did Tuesday with the 48-bar setting" — it's in the YAML or it didn't happen.

The YAML pulls double duty:
- **What data?** `symbol`, `interval`, `start`, `end`, `period`, `test_fraction`.
- **What model?** `order: [1, 1, 1]` for ARIMA, `input_chunk_length`, `hidden_size`, etc., for the deep models.

Concretely, `experiments/02_arima/config.yaml`:

```yaml
symbol: BTCUSDT       # Binance trading pair
interval: 1h          # kline interval: 1m, 5m, 15m, 1h, 4h, 1d, ...
start: 2024-01-01     # inclusive, UTC
end: 2024-12-01       # exclusive, UTC
period: monthly       # 'monthly' or 'daily'. Must be set explicitly.
                      # Use 'daily' for partial / not-yet-complete months.
test_fraction: 0.2    # tail fraction held out for evaluation
order: [1, 1, 1]      # ARIMA (p, d, q)
```

Two principles fall out of this:

**Inline comments on every field.** I'm writing the YAML once and reading it for years. The `# Use 'daily' for partial / not-yet-complete months.` line on `period:` saved me ten minutes of head-scratching three weeks after I wrote it. The convention is non-negotiable: every field gets a `# what does this do` annotation.

**No auto-pick defaults.** If a field is ambiguous, the user has to specify it. `period: monthly` vs `daily`. `backend: chronos` vs `timesfm`. `order: [1, 1, 1]`. The lab refuses to guess. Defaults that "do the right thing most of the time" do the wrong thing the moment the right thing changes — and you don't notice because the run still passes. Forcing the explicit choice catches it at the YAML, not in `metrics.json`.

The result: a `git diff` between two configs is the entire change between two runs. That's the whole audit trail.

---

## Rule 3 — Shared `src/btc_ai/` for loader, splitter, metrics

This is the load-bearing decision. Every experiment imports the same:
- **Loader.** `BinanceVisionLoader` ([src/btc_ai/data/binance_vision.py](../../src/btc_ai/data/binance_vision.py)) reads the on-disk cache and returns a `pd.DataFrame` with a tz-aware DatetimeIndex.
- **Splitter.** Time-ordered, last `test_fraction` held out, no shuffling. Spelled out in `run.py` itself rather than abstracted, so it's auditable.
- **Metrics.** [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py) — `mae`, `rmse`, `mape`, `directional_accuracy`, `strategy_returns`, `cumulative_return`, `annualized_sharpe`, plus a `periods_per_year` lookup so the Sharpe annualization tracks the interval.

The subtle bit is the `directional_accuracy(y_true, y_pred, ref)` signature. It takes a third argument — `ref`, the reference price (typically `close[t-1]`). That's how the naive baseline returns `NaN`: its predicted move is identically zero, no opinion is expressed, no row counts. Article 1 explained why that matters. This article points at where the decision lives:

```python
def directional_accuracy(y_true, y_pred, ref):
    true_dir = np.sign(y_true.to_numpy() - ref.to_numpy())
    pred_dir = np.sign(y_pred.to_numpy() - ref.to_numpy())
    mask = (true_dir != 0) & (pred_dir != 0)
    if not mask.any():
        return float('nan')
    return float(np.mean(true_dir[mask] == pred_dir[mask]))
```

That signature has to be imposed at the *shared module* level. If every experiment defined its own `directional_accuracy`, the leaderboard would be a fiction. If the floor of the leaderboard moves — say, a future article changes the convention to "directional accuracy excluding bars with absolute return < 0.1 %" — every model gets re-judged against the new floor automatically. That's only possible because the metric lives in one file.

---

## Rule 4 — Makefile is the canonical command surface

I never run `python experiments/02_arima/run.py`. I run:

```
make 02_arima
```

Why? Three reasons:
1. **Discoverability.** `make help` enumerates everything that can be done. README points to `make`, so newcomers learn one tool, not seven.
2. **Wrap `uv run`.** Every target wraps `uv run`, so the `.venv/` is automatic. No "wait, did I activate the venv?".
3. **Argument hygiene.** `make fetch CONFIG=experiments/02_arima/config.yaml` is the *only* way to override the data config — there are no half-remembered flags scattered across READMEs.

The Makefile is short — a target per experiment plus a few utility targets (`lint`, `test`, `clean`) — and that's the point. It's not a build system; it's a one-page index of what this repo can do. The full file is ~60 lines, half of them whitespace.

---

## Rule 5 — `uv` for envs and runs

The repo uses [`uv`](https://docs.astral.sh/uv/) for environment management. `uv.lock` is committed; `.venv/` is not. A new contributor's path is:

```
brew install uv          # macOS
uv sync                  # creates .venv/ from pyproject.toml + uv.lock
make fetch               # pre-warm data cache
make 01_baseline   # reproduce article 1's numbers
```

Four commands. No "works on my laptop". `uv` has resolved every "did you install the right pandas version?" headache I've had since adopting it. There's no requirements.txt, no conda environment.yaml, no Dockerfile (yet — I'll add one if/when somebody asks). The lockfile is the contract.

---

## Rule 6 — Every experiment ships a report, even the negative ones

This is the discipline. Each experiment ends with:
- `README.md` — what the experiment is, why it exists, how to interpret the result.
- `results/metrics.json` — the canonical numbers.
- `results/predictions.parquet` — the per-bar test-set predictions.
- `results/plot.png` — close vs. prediction on the test window.

For the wins, this is obvious. For the losses, it's structurally necessary. [03_gradient_boosting](../../experiments/03_gradient_boosting/) — XGBoost with 31 engineered features — clocks `directional_accuracy = 0.4872`. That's worse than chance. The folder ships the same artifacts as [02_arima](../../experiments/02_arima/): README, metrics, predictions, plot. The next post in the series (Article 4) is *only* able to make the claim "ARIMA(1,1,0) beat XGBoost" because the XGBoost loss is captured in a tidy folder, not lost in a notebook.

The lab does not delete failed experiments. They're the most informative folders in the tree. A bad result in `results/metrics.json` is more useful than a good result in `notebooks/scratch_v8.ipynb`, because the bad result can still be compared to next week's model. The good result that lives in a notebook can't.

---

## What an experiment folder actually looks like

Walk through `experiments/02_arima/` as the canonical example:

```
experiments/02_arima/
├── README.md           # what / why / how / how to interpret
├── config.yaml         # data slice + ARIMA order, every field commented
├── run.py              # ~120 lines: load, split, fit, predict, metrics, plot
├── sweep.py            # iterate over (p, d, q) orders, write sweep.csv
└── results/
    ├── metrics.json        # produced by run.py
    ├── predictions.parquet # close, pred, ref, strategy_return per test bar
    ├── plot.png            # close vs ARIMA pred on the test window
    └── sweep.csv           # produced by sweep.py
```

Two flat scripts, one config, one results dir. No subclasses, no plugin system, no decorators. `run.py` is meant to be read top-to-bottom in five minutes. The repo deliberately does not have a `MyExperimentBase` class to inherit; each script duplicates the load-split-evaluate boilerplate, which is annoying right up until you have to debug exactly one experiment, at which point the duplication is a feature.

What's gitignored: `data/raw/` and `data/processed/`, plus `.venv/`. Everything else — including every committed `metrics.json` — is in the repo. The leaderboard is part of the source tree.

---

## What this is *not*

A few honest disclaimers:

- **Not a framework.** No `Experiment` base class, no plugin registry, no "best practices" decorators. If your repo grows to 100 experiments and 10 contributors, you'll outgrow this scheme. For 7 experiments and one me, a flat folder per run beats a clever abstraction every time.
- **Not MLflow / W&B / DVC.** A folder of `metrics.json` files is enough at this scale. Bring out the heavier tools when you have multi-human, multi-dataset, or multi-deployment requirements. None of those apply here.
- **Not "production".** This repo is a lab notebook. There is no serving stack, no inference endpoint, no monitoring. The artifact is the notebook. If a model from here ever ships to production, it'll be re-implemented behind a proper service boundary.

The general principle: pick scaffolding proportional to the work. Five experiments don't need a workflow engine. Twenty might. The scaffolding here is the smallest set of decisions that makes the leaderboard honest.

---

## Why this discipline matters — Article 6 spoiler

The cleanest illustration of why this scaffolding earns its keep is in Article 6. There, I'll run every model on two non-overlapping test windows: Q1 2024 only, and Jan–Nov 2024. The verdict will flip. XGBoost is *leader* on Q1 (`dir_acc 0.5185`) and *last* on Jan–Nov (`0.4872`). The LSTM is *broken* on Q1 (`Sharpe -0.54`) and *competitive* on Jan–Nov (`+4.95`).

That comparison is only believable if the only thing that changed between the runs was the test window. Same loader, same splitter, same metrics, same Sharpe annualization, same `make` target, same lockfile. If any of those varied, the article evaporates — every flip becomes "well, maybe it's the metric, maybe it's the data". The scaffolding is the difference between a real finding and a methodological accident.

That's the punchline. The repo isn't structured this way because it's elegant. It's structured this way because the *next* claim I want to make depends on it.

---

## How to reproduce, fully

If you want every leaderboard row in this series, on your laptop, the path from a clean clone is six commands:

```
brew install uv
git clone <repo>
cd transformer_bitcoin_ai
uv sync
make fetch
make 01_baseline
make 02_arima
make 03_gradient_boosting
make 04_lstm
make 05_transformer
make 06_pretrained
```

The deep-learning ones (04 / 05 / 06) take single-digit minutes each on Apple MPS or a recent CUDA GPU; the rest are seconds. Every `metrics.json` you produce should match the ones I quote in this series, modulo seed-level noise on the deep models.

If they don't match, that's a bug in the lab — file an issue. The whole point of the scaffolding is that "should match" is meaningful at all.

---

The next post is Article 4 — *ARIMA(1,1,0) beat my XGBoost. One parameter beat thirty-one features.* — and it's where the leaderboard gets a soul. A single AR coefficient on differenced returns from [02_arima](../../experiments/02_arima/) produces a non-trivial directional edge, while XGBoost with 24 lagged returns + rolling stats + volume + OHLC from [03_gradient_boosting](../../experiments/03_gradient_boosting/) lands below chance. The contrast is what makes the lab notebook useful: every later model has something concrete to beat.

---

*Code: [README.md](../../README.md) · [Makefile](../../Makefile) · [src/btc_ai/](../../src/btc_ai/) · [experiments/](../../experiments/)*
