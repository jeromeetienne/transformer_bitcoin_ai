# Prompt — generate `docs/reports/${experiment_id}.report.md`

Reverse-engineered from the existing reports in [docs/reports/](reports/). Use this when an experiment finishes and you want a `${experiment_id}.report.md` written from its artifacts.

## Inputs

Read these files for the experiment under report (substitute `${experiment_id}`, e.g. `02_arima`):

- `experiments/${experiment_id}/config.yaml` — config table + data slice description
- `experiments/${experiment_id}/run.py` — what the model actually does (target definition, walk-forward shape, library used)
- `experiments/${experiment_id}/README.md` — narrative source for "What this experiment is"
- `experiments/${experiment_id}/results/metrics.json` — single-fit headline metrics
- `experiments/${experiment_id}/results/sweep.csv` — if present, becomes the sweep section
- `experiments/${experiment_id}/results/plot.png` — referenced under "Files produced"
- `src/btc_ai/eval/metrics.py` — authoritative metric definitions (use these when explaining NaN, sign conventions, ppy, etc.)
- `docs/reports/*.report.md` — every prior report. **Required** for the cross-experiment leaderboard table and inline links to sibling reports.

Do **not** rerun experiments, recompute metrics, or invent numbers. Every numeric value in the report must come verbatim from `metrics.json` / `sweep.csv` of the experiment being reported on, or from a sibling experiment's existing report.

## Output

Single file at `docs/reports/${experiment_id}.report.md`. UTF-8, LF, no trailing whitespace. Use the section skeleton below in order; omit a section only when its source artifact does not exist (e.g. no `sweep.csv` → no sweep section).

## Skeleton

```markdown
# Report — `${experiment_id}`

**Run date:** YYYY-MM-DD
**Status:** completed | completed (single fit + N-order sweep) | completed (single fit, no sweep yet)

## What this experiment is

One paragraph naming the model class, what it predicts, and what makes this experiment different from the previous one in the ladder. Follow with a fenced code/pseudo-math block showing the target definition or model equation. Then one paragraph naming the library and the walk-forward shape (e.g. `historical_forecasts(retrain=False)`), and a link to `experiments/${experiment_id}/README.md` for full details.

## Configuration

From [experiments/${experiment_id}/config.yaml](../../experiments/${experiment_id}/config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `1h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| <model-specific fields…> | … |

Total bars: **N**. Train: **N**. (Val: **N**.) Test: **N**.

## Results — single fit (and the model's identifier if applicable, e.g. ARIMA(1,1,1))

From [experiments/${experiment_id}/results/metrics.json](../../experiments/${experiment_id}/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | **…** USD |
| RMSE | … USD |
| MAPE | … % |
| Directional accuracy | … |
| Cumulative return | … |
| Annualized Sharpe | … |

(Add AIC/BIC, training-loss, or other model-native metrics above the shared block if the experiment exposes them.)

## Cross-experiment comparison

Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| 01_baseline | … | … | … | NaN | — | — |
| 02_arima (1, 1, 1) | … | … | … | … | … | … |
| **${experiment_id}** | **…** | **…** | … | **…** | **…** | **…** |
| …other siblings… | … | … | … | … | … | … |

One paragraph of editorial after the table: which row leads which column, what flipped vs a prior window, what to read into it. Link to sibling reports inline.

## Sweep (omit if no `sweep.csv`)

From [experiments/${experiment_id}/results/sweep.csv](../../experiments/${experiment_id}/results/sweep.csv) — same data slice, N configs:

| <param columns…> | <metric columns…> |
|---|---|
| … bolded leaders … |

Sort by the metric that matters most for the experiment (Sharpe for trading-aware models, RMSE/MAE for level-only).

## Interpretation

Numbered list. Each item is one observation about *why* the numbers came out the way they did — not a restatement of the table. Cite specific cells. Call out:

1. The leader on the directional / strategy metric and what makes it the leader.
2. Any tie-to-naive on MAE — it's the random-walk floor and is expected.
3. Disagreements between in-sample selection criteria (AIC/BIC/val_loss) and out-of-sample Sharpe.
4. Cases where adding parameters / features / depth *hurt* — and what that implies about signal vs noise at this horizon.
5. Any NaN or by-design "no opinion" behavior, and why the metric definition produces it (cite `src/btc_ai/eval/metrics.py`).
6. Anything the AIC / sweep flagged that wasn't visible in MAE alone.

### Bottom line (or `### The non-obvious bit`)

One short paragraph stating the headline finding, in bold where appropriate, plus a per-bar Sharpe sanity check when the strategy looks too good (`SE ≈ √(1/N_test_bars)` against zero). Always name the regime caveat — which months the test slice covers and whether the finding has been tested out-of-regime.

## Caveats (include for any model that fit parameters)

- Single split, no rolling-origin CV.
- Walk-forward without re-fitting (or note if it does refit).
- Anything the model assumes that BTC log-returns violate (Gaussianity, homoskedasticity, stationarity).
- Anything missing from the inputs (exog regressors, covariates, on-chain features).

## Files produced

- [experiments/${experiment_id}/results/metrics.json](../../experiments/${experiment_id}/results/metrics.json)
- [experiments/${experiment_id}/results/predictions.parquet](../../experiments/${experiment_id}/results/predictions.parquet)
- [experiments/${experiment_id}/results/plot.png](../../experiments/${experiment_id}/results/plot.png)
- [experiments/${experiment_id}/results/sweep.csv](../../experiments/${experiment_id}/results/sweep.csv)  ← only if exists

## How to reproduce

```
make ${experiment_id}
make ${experiment_id}_sweep   # only if there is a sweep target in the Makefile
```
```

## Style rules

- **Voice:** technical, honest, editorialized. Treat the reader as a peer reading the leaderboard, not a student. State what's surprising; don't pad with summaries of what the table already shows.
- **Numbers:** copy from `metrics.json` / `sweep.csv` verbatim. Format USD with thousands separators using a thin non-breaking space (e.g. `1 608`, `93 381.54`). Percentages with `%` and one or two decimal places matching the source magnitude.
- **Bold the leader** in every leaderboard / sweep table, including the row for the experiment under report. Bold key claims in prose.
- **Links** use relative paths from `docs/reports/` (i.e. `../../experiments/...`, `02_arima.report.md` for siblings).
- **No emojis.** No marketing language. No "this exciting result" / "remarkable".
- **NaN is not a bug** — when a metric is NaN by design (e.g. dir_acc on the naive baseline, Sharpe on a flat strategy), say *why* with reference to the metric's definition in `src/btc_ai/eval/metrics.py`.
- **AIC vs out-of-sample disagreement** is a recurring lesson in this repo — surface it whenever the sweep makes it visible.
- **Regime caveat is mandatory** when reporting a positive Sharpe. Name the months covered by the test slice and whether the finding has held out of regime.
- **No fabricated history.** If you cannot derive a fact from the inputs listed above, omit it. Do not invent author names, prior results, or dates.

## Cross-report consistency

When this report is written, **do not rewrite the leaderboard rows in older reports**. If the new experiment's row would change a prior report's editorial paragraph (e.g. it dethrones the previous leader), say so in the new report's "Cross-experiment comparison" paragraph and leave the older reports alone. Each report is a snapshot at its run date.
