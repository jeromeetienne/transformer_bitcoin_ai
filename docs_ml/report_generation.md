# Prompts — generate per-experiment reports and the global cross-experiment report

This document specifies two related prompts:

1. **Per-experiment report** — `docs_ml/reports/${experiment_id}.report.md` for a single experiment. The bulk of this file.
2. **Global cross-experiment report** — `docs_ml/reports/XX_global.report.md`, a synthesis over every per-experiment report. See [Global cross-experiment report](#global-cross-experiment-report) at the bottom.

Use prompt 1 when an experiment finishes; use prompt 2 after any per-experiment report has been (re)generated.

---

## Prompt 1 — `docs_ml/reports/${experiment_id}.report.md`

Reverse-engineered from the existing reports in [docs_ml/reports/](reports/). Use this when an experiment finishes and you want a `${experiment_id}.report.md` written from its artifacts.

## Inputs

The report covers a single `(experiment, trial)` pair. The trial name is the config filename with the `.config.yaml` extension stripped — e.g. `experiments/02_arima/configs/btc_4h_2024.config.yaml` → trial `btc_4h_2024`, with artefacts under `experiments/02_arima/results/btc_4h_2024/`. Substitute `${experiment_id}` (e.g. `02_arima`) and `${trial}` (e.g. `btc_4h_2024`) throughout.

Read these files for the experiment+trial under report:

- `experiments/${experiment_id}/configs/${trial}.config.yaml` — config table + data slice description
- `experiments/${experiment_id}/run.py` — what the model actually does (target definition, walk-forward shape, library used)
- `experiments/${experiment_id}/README.md` — narrative source for "What this experiment is"
- `experiments/${experiment_id}/results/${trial}/metrics.json` — single-fit headline metrics
- `experiments/${experiment_id}/results/${trial}/sweep.csv` — if present, becomes the sweep section
- `experiments/${experiment_id}/results/${trial}/plot.png` — referenced under "Files produced"
- `src/btc_ai/eval/metrics.py` — authoritative metric definitions (use these when explaining NaN, sign conventions, ppy, etc.)
- `docs_ml/reports/*.report.md` — every prior report. **Required** for the cross-experiment leaderboard table and inline links to sibling reports.

Do **not** rerun experiments, recompute metrics, or invent numbers. Every numeric value in the report must come verbatim from `metrics.json` / `sweep.csv` of the experiment+trial being reported on, or from a sibling experiment's existing report.

## Output

Single file at `docs_ml/reports/${experiment_id}.report.md`. UTF-8, LF, no trailing whitespace. Use the section skeleton below in order; omit a section only when its source artifact does not exist (e.g. no `sweep.csv` → no sweep section).

## Skeleton

```markdown
# Report — `${experiment_id}`

**Run date:** YYYY-MM-DD
**Status:** completed | completed (single fit + N-order sweep) | completed (single fit, no sweep yet)

## What this experiment is

One paragraph naming the model class, what it predicts, and what makes this experiment different from the previous one in the ladder. Follow with a fenced code/pseudo-math block showing the target definition or model equation. Then one paragraph naming the library and the walk-forward shape (e.g. `historical_forecasts(retrain=False)`), and a link to `experiments/${experiment_id}/README.md` for full details.

## Configuration

From [experiments/${experiment_id}/configs/${trial}.yaml](../../experiments/${experiment_id}/configs/${trial}.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `4h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| <model-specific fields…> | … |

Total bars: **N**. Train: **N**. (Val: **N**.) Test: **N**.

## Results — single fit (and the model's identifier if applicable, e.g. ARIMA(1,1,1))

From [experiments/${experiment_id}/results/${trial}/metrics.json](../../experiments/${experiment_id}/results/${trial}/metrics.json):

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
| 01_baseline_naive | … | … | … | NaN | — | — |
| 01b_moving_average (window=N) | … | … | … | … | … | … |
| 02_arima (1, 1, 1) | … | … | … | … | … | … |
| **${experiment_id}** | **…** | **…** | … | **…** | **…** | **…** |
| …other siblings… | … | … | … | … | … | … |

One paragraph of editorial after the table: which row leads which column, what flipped vs a prior window, what to read into it. Link to sibling reports inline.

## Sweep (omit if no `sweep.csv`)

From [experiments/${experiment_id}/results/${trial}/sweep.csv](../../experiments/${experiment_id}/results/${trial}/sweep.csv) — same data slice, N configs:

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

- [experiments/${experiment_id}/results/${trial}/metrics.json](../../experiments/${experiment_id}/results/${trial}/metrics.json)
- [experiments/${experiment_id}/results/${trial}/predictions.parquet](../../experiments/${experiment_id}/results/${trial}/predictions.parquet)
- [experiments/${experiment_id}/results/${trial}/plot.png](../../experiments/${experiment_id}/results/${trial}/plot.png)
- [experiments/${experiment_id}/results/${trial}/sweep.csv](../../experiments/${experiment_id}/results/${trial}/sweep.csv)  ← only if exists

## How to reproduce

```
make ${experiment_id} CONFIG=experiments/${experiment_id}/configs/${trial}.yaml
make ${experiment_id}_sweep CONFIG=experiments/${experiment_id}/configs/${trial}.yaml   # only if there is a sweep target in the Makefile
```

(The CONFIG= override is optional when running the trial that matches the per-experiment Makefile's default `CONFIG ?=`; required for any other trial.)
```

## Style rules

- **Voice:** technical, honest, editorialized. Treat the reader as a peer reading the leaderboard, not a student. State what's surprising; don't pad with summaries of what the table already shows.
- **Numbers:** copy from `metrics.json` / `sweep.csv` verbatim. Format USD with thousands separators using a thin non-breaking space (e.g. `1 608`, `93 381.54`). Percentages with `%` and one or two decimal places matching the source magnitude.
- **Bold the leader** in every leaderboard / sweep table, including the row for the experiment under report. Bold key claims in prose.
- **Links** use relative paths from `docs_ml/reports/` (i.e. `../../experiments/...`, `02_arima.report.md` for siblings).
- **No emojis.** No marketing language. No "this exciting result" / "remarkable".
- **NaN is not a bug** — when a metric is NaN by design (e.g. dir_acc on the naive baseline, Sharpe on a flat strategy), say *why* with reference to the metric's definition in `src/btc_ai/eval/metrics.py`.
- **AIC vs out-of-sample disagreement** is a recurring lesson in this repo — surface it whenever the sweep makes it visible.
- **Regime caveat is mandatory** when reporting a positive Sharpe. Name the months covered by the test slice and whether the finding has held out of regime.
- **No fabricated history.** If you cannot derive a fact from the inputs listed above, omit it. Do not invent author names, prior results, or dates.

## Cross-report consistency

When this report is written, **do not rewrite the leaderboard rows in older reports**. If the new experiment's row would change a prior report's editorial paragraph (e.g. it dethrones the previous leader), say so in the new report's "Cross-experiment comparison" paragraph and leave the older reports alone. Each report is a snapshot at its run date.

After regenerating any per-experiment report, also regenerate [`docs_ml/reports/XX_global.report.md`](reports/XX_global.report.md) so the synthesis stays in sync. See [Global cross-experiment report](#global-cross-experiment-report) below.

---

## Global cross-experiment report

The single-file synthesis across every per-experiment report. Lives at `docs_ml/reports/XX_global.report.md`. The `XX_` prefix is deliberate — sorts after every numbered experiment so the synthesis sits at the bottom of `ls`.

### When to (re)generate

- After **any** per-experiment report is written or updated. The global report is a derived artifact; it goes stale the moment a row in any leaderboard changes.
- After data-slice shifts that affect comparability (e.g. the recent `1h → 4h` switch). The data-slice picture section is load-bearing — it tells the reader which rows are apples-to-apples.

### Inputs

Read these files for the global report:

- Every `docs_ml/reports/*.report.md` (except `XX_global.report.md` itself). These are the audit trail; every number in the global report should be traceable to one of them.
- Every `experiments/${experiment_id}/results/${trial}/metrics.json` — for verbatim numeric values when reproducing leaderboard rows.
- Every `experiments/${experiment_id}/results/${trial}/sweep.csv` — only when the global report cites a sweep row (e.g. ARIMA's `(3, 1, 3)` AIC leader).
- [`src/btc_ai/eval/metrics.py`](../src/btc_ai/eval/metrics.py) — for NaN semantics and the `periods_per_year` mapping used in the per-bar Sharpe significance table.
- [`docs_ml/articles_todo.md`](articles_todo.md) — to keep the "where this goes next" section pointed at the right downstream consumer.

Do **not** rerun experiments, recompute metrics, or invent numbers. The global report is a *synthesis*, not a re-run; every value is a verbatim copy from `metrics.json`, `sweep.csv`, or a per-experiment report.

### Output

Single file at `docs_ml/reports/XX_global.report.md`. UTF-8, LF, no trailing whitespace.

### Skeleton

Use the section order below. Omit a section only when its source data does not exist (e.g. no probabilistic forecasts → no probabilistic-interval section).

```markdown
# Report — `XX_global` — cross-experiment comparison

**Run date:** YYYY-MM-DD
**Status:** synthesis of N per-experiment reports under [docs_ml/reports/](.)

## What this report is

One paragraph naming the experiments synthesized, plus the rule that every cited number is verbatim from per-experiment artifacts (this report does not recompute anything).

## Data-slice picture (load-bearing)

A table with one row per experiment showing `interval (config)`, `rows_total (artefact)`, `rows_test`, `metrics.json mtime`, and a "Slice" column flagging fresh vs. stale slices. Surface any divergence in data slices upfront — readers need to know which leaderboards are apples-to-apples before they read further.

## Leaderboard — `<slice>` slice

One leaderboard table per data slice present (e.g. one 4h, one 1h). Each is a `| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | annualized_sharpe |` table, sorted by experiment id. **Bold the leader** in each metric column. Editorial paragraph below the table: who leads what, what's surprising, what flipped vs. expectations.

## The complexity-doesn't-pay arc (or equivalent ladder section)

A table or list ordering experiments by parameter count (approx.) alongside Sharpe / dir_acc / MAE-rank. The observation usually surfaces: parameter count and signal extraction are weakly correlated on near-random-walk financial data. Use whatever framing the data supports — if a *different* ladder is meaningful (e.g. "engineered features vs. raw sequence"), use that.

## Point-error vs. trading-metric tradeoff

A paragraph naming the MAE leader and the Sharpe leader and explaining the mechanism by which they diverge (strategy aggressiveness, hedging, etc.).

## Where capacity helps and where it hurts

A pair-by-pair table walking the ladder one transition at a time. Columns: `Pair | What's added | Δ MAE | Δ dir_acc | Δ Sharpe | Verdict`. Surfaces the *gradient* across the ladder rather than absolute numbers.

## AIC vs. out-of-sample

Only present if a sweep surfaces it (currently 02_arima's sweep). State which selection criterion picked which order, and where the criteria agree vs. disagree.

## NaN-by-design as a teaching surface

Enumerate every NaN that appears in any leaderboard, with the metric-definition reason (cite [src/btc_ai/eval/metrics.py](../src/btc_ai/eval/metrics.py)). A leaderboard reader can mistake NaN for a missing run; this section ensures they don't.

## Pipeline sanity checks

Numbered list of cross-experiment identities the artifacts confirm (e.g. ARIMA (0, 1, 0) MAE equals naive's MAE bit-for-bit). Also list checks that *don't* exist yet but would be useful.

## Per-bar Sharpe significance

Table with one row per experiment whose `annualized_sharpe` is not NaN. Columns: `Experiment | Sharpe (annualized) | √ppy | per-bar Sharpe | √N_test | SE | ratio (σ)`. `per_bar = annualized / √ppy`; `SE = 1 / √N_test`. Bold any row with ratio > 2.5 σ as cleanly significant. State plainly that this approximation assumes i.i.d. per-bar strategy returns (which they probably aren't on crypto) and therefore upper-bounds the real significance.

## Regime caveat

One paragraph naming the dates the test slice covers and the regime (e.g. "post-election BTC rally"). State that no row has been tested out-of-regime and that every positive Sharpe is conditional on this regime.

## Bottom line

Short paragraph. The headline finding in bold where appropriate. One-sentence "read in one line" reduction at the end.

## Where this goes next

Bulleted operational follow-ups surfaced by the per-experiment reports (re-runs needed, missing sanity checks, missing analyses, downstream consumers like the article series).

## Caveats (shared by all rows)

Bulleted: single split, walk-forward without re-estimation, one seed per row, long/flat strategy with no costs, single test regime, every cited number verbatim.

## Source artefacts

Bulleted list of links to each per-experiment report and a pointer to `experiments/${id}/results/` for raw artifacts.

## How to regenerate this report

A short shell block: regenerate per-experiment artifacts via `make`, regenerate per-experiment reports per Prompt 1, regenerate this report per Prompt 2.
```

### Style rules (in addition to those for per-experiment reports)

- **Synthesis, not duplication.** A section that just restates what a single per-experiment report already says is the wrong section. The global report's job is to put rows *next to each other* and read the gradient — comparative claims, ladder views, pair-by-pair deltas, significance tables. If a paragraph could have lived in `02_arima.report.md`, it belongs there, not here.
- **Every numeric cell is a verbatim copy.** Do not round, recompute, or interpolate. If a number would need to be computed (e.g. a Δ between two MAEs in the pair-by-pair table), compute it from the verbatim values and show the math implicitly in the table — never derive a number that isn't either copy-pasted or trivially-derivable on screen.
- **Bold the load-bearing claim, not every claim.** The global report is dense; over-bolding kills the signal. Limit bold to: leader cells in tables, the headline finding in the Bottom line, and any σ-ratio above the significance threshold.
- **Honour the data-slice picture.** If two experiments are on different slices, **never** put them in the same leaderboard. Two leaderboards are better than one misleading one. The "Data-slice picture" section is the explicit promise this rule is being kept.
- **Cite the per-experiment reports inline.** When making a claim about an experiment, link the claim to that experiment's report (e.g. `[02_arima's sweep](02_arima.report.md#sweep--12-p-d-q-orders)`). The audit trail must be one click away.
- **Regenerate, don't append.** This file is fully regenerated each time. There is no "history" section — the per-experiment reports hold history; the global report is always the current synthesis.

### Cross-report consistency

When the global report is regenerated:

- It **may** state that a leaderboard row in a per-experiment report would change if that report were re-generated today. It **must not** edit the per-experiment report to fix it. Per-experiment reports are snapshots; the global report is the live synthesis.
- Conversely, when a per-experiment report changes (Prompt 1 was just run), the global report is now stale and must be regenerated (Prompt 2). The two are coupled.
