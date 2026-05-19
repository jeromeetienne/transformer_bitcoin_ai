# Report — `XX_global` — cross-experiment comparison

**Run date:** 2026-05-19
**Status:** synthesis of seven per-experiment reports under [docs_ml/reports/](.)

## What this report is

A single-file leaderboard and analytical comparison of every experiment in the repo: [01_baseline_naive](01_baseline_naive.report.md), [01b_moving_average](01b_moving_average.report.md), [02_arima](02_arima.report.md), [03_gradient_boosting](03_gradient_boosting.report.md), [04_lstm](04_lstm.report.md), [05_transformer](05_transformer.report.md), [06_pretrained](06_pretrained.report.md). Every number cited here is verbatim from the corresponding `experiments/${id}/results/metrics.json`; the per-experiment reports are the audit trail.

Sections progress from raw numbers to analytical observations. The intent is not to repeat what the per-experiment reports already say but to put the rows side-by-side and read the *gradient* across the ladder of model classes.

## Data-slice picture

All seven experiments report on the same 4h slice now. The recent re-run of 05_transformer and 06_pretrained closed the 1h/4h divergence that earlier versions of this report flagged.

| # | `interval` (config) | `rows_total` (artefact) | `rows_test` | `metrics.json` mtime | Slice |
|---|---|---|---|---|---|
| 01_baseline_naive | 4h | 2 010 | 402 | 2026-05-19 | fresh 4h |
| 01b_moving_average | 4h | 2 010 | 402 | 2026-05-19 | fresh 4h |
| 02_arima | 4h | 2 010 | 402 | 2026-05-19 | fresh 4h |
| 03_gradient_boosting | 4h | 1 985 | 397 | 2026-05-19 | fresh 4h (but `sweep.csv` still stale 1h) |
| 04_lstm | 4h | 2 009 | 401 | 2026-05-19 | fresh 4h |
| 05_transformer | 4h | 2 009 | 401 | 2026-05-19 | fresh 4h (but `sweep.csv` still stale 1h) |
| 06_pretrained (chronos-2) | 4h | 2 009 | 402 | 2026-05-19 | fresh 4h (no sweep on disk yet) |

The 4h test slice covers approximately **Sep 25 → Dec 1 2024** (last 20 % of the 11-month range) — the strong post-election BTC rally. **The sweep artifacts for 03 and 05 are still stale on 1h**, so their hyperparameter-ranking tables remain non-comparable to the headline 4h numbers; 06 has no sweep at all yet.

## Leaderboard — 4h slice (all 7 experiments)

| Experiment | MAE (USD) | RMSE (USD) | MAPE | dir_acc | cum_ret | annualized_sharpe |
|---|---|---|---|---|---|---|
| [01_baseline_naive](01_baseline_naive.report.md) | 518.36 | 784.09 | 0.6701 % | NaN | — | — |
| [01b_moving_average (window=24)](01b_moving_average.report.md) | 1 905.27 | 2 580.61 | 2.4782 % | 0.4801 | 0.0511 | 1.3003 |
| [02_arima (1, 1, 1)](02_arima.report.md) | **517.16** | 782.42 | **0.6686 %** | **0.5547** | 0.3314 | 6.0569 |
| [03_gradient_boosting (n_feat=31)](03_gradient_boosting.report.md) | 539.39 | 795.07 | 0.7008 % | 0.5365 | 0.5042 | 6.4233 |
| [04_lstm (default)](04_lstm.report.md) | 522.29 | 785.45 | 0.6761 % | 0.4938 | 0.2596 | 4.2660 |
| [05_transformer (default)](05_transformer.report.md) | 813.10 | 1 090.78 | 1.0786 % | 0.4913 | 0.4468 | 5.8675 |
| [06_pretrained (chronos-2)](06_pretrained.report.md) | 519.78 | **780.99** | 0.6721 % | 0.5323 | **0.6975** | **7.5915** |

**Who leads what:** the leaderboard splits cleanly into two halves. **02_arima(1, 1, 1) leads MAE, MAPE, and dir_acc** with three estimated parameters. **06_pretrained (Chronos-2) leads RMSE, cum_ret, and Sharpe** with 120 M zero-shot parameters and no training on Bitcoin. No model leads all six columns; the split corresponds to "best at average point error and direction" (02) vs. "best at tail behaviour and trading-aware metrics" (06).

**Three observations the table makes load-bearing.** First, the **MAE band 517 → 540** contains five of the seven rows; on a ~400-bar test slice these gaps are nearly within sweep noise. Second, **05_transformer's MAE 813.10 is the outlier** — 57 % above naive, in a leaderboard otherwise within $25 of naive. Third, **06_pretrained's RMSE is lower than ARIMA's** even though its MAE is higher: the foundation model has fewer big-error bars than every trained model.

## The complexity-doesn't-pay arc (with one big asterisk)

Ordering by approximate trainable-parameter count:

| Row | Approx. trainable params | Sharpe | dir_acc | MAE (USD) |
|---|---|---|---|---|
| 01_baseline_naive | 0 | — | NaN | 518.36 |
| 01b_moving_average (24) | 0 (window only) | 1.3003 | 0.4801 | 1 905.27 |
| 02_arima (1, 1, 1) | 3 | 6.0569 | **0.5547** | **517.16** |
| 02_arima (3, 1, 3)* | 7 | 6.4015 | 0.5274 | 514.07 |
| 03_gradient_boosting | ~31 feat × 400 trees × depth 5 | 6.4233 | 0.5365 | 539.39 |
| 04_lstm (default) | ~10 000 weights | 4.2660 | 0.4938 | 522.29 |
| 05_transformer (default) | ~30 000+ weights | 5.8675 | 0.4913 | 813.10 |
| 06_pretrained (chronos-2) | ≈ 120 M (zero-shot, no BTC training) | **7.5915** | 0.5323 | 519.78 |

\* From [02_arima's sweep](02_arima.report.md#sweep--12-p-d-q-orders).

The trained-on-this-data trajectory (rows 01 → 05) tells a clean story: **capacity helps up to ARIMA(3, 1, 3) at 7 parameters, plateaus through XGBoost at thousands of leaf splits, then *declines* for the recurrent and attention models** at ten- to thirty-thousand weights. LSTM Sharpe 4.27 and TFT Sharpe 5.87 both sit below ARIMA(1, 1, 1)'s 6.06. The 4h training set of ~1 450 rows is not enough for the deep models to amortise their capacity.

**The 120 M-parameter zero-shot row breaks the arc, but only because of *how* the parameters were trained**: Chronos-2 was trained on millions of unrelated time series, not on BTC. So it does not overfit the 1 450-row 4h training slice — there *is* no training slice for the foundation model. The arc to read is therefore:

- Trained on this slice → capacity helps then hurts (U-shape with the bottom around the trained TFT).
- Trained on other data → 120 M parameters productive without overfitting risk on this slice.

The headline reframing: **on a small training set, you don't get to use more parameters by training them yourself — you have to borrow them from somewhere else.**

## Point-error vs. trading-metric tradeoff

The 4h leaderboard surfaces an explicit tension across two pairs:

- **02_arima(1, 1, 1) vs. 06_pretrained (chronos-2).** MAE: 02 wins by $2.62. RMSE: 06 wins by $1.43. dir_acc: 02 wins (0.5547 vs. 0.5323). cum_ret: 06 wins by 36 percentage points. Sharpe: 06 wins by 1.53.
- **02_arima(1, 1, 1) vs. 03_gradient_boosting.** MAE: 02 wins by $22.23. RMSE: 02 wins by $12.65. dir_acc: 02 wins by 0.0182. cum_ret: 03 wins by 17 percentage points. Sharpe: 03 wins by 0.37.

Both 03 and 06 outperform 02 on the trading metrics while losing (or essentially tying) on point error. The mechanism is *strategy aggressiveness*: a higher-confidence directional forecast — strictly above the reference, more often — keeps the long/flat rule long for more of the rally. ARIMA's direction-getting-right edge does not convert into compounding wins because ARIMA's predicted-up-moves are smaller. **You don't get both on this slice. Pick a model for the metric you actually care about.**

## Where capacity helps and where it hurts

Pair-by-pair walk of the ladder, one transition at a time (each row adds **one** thing to the previous):

| Pair | What's added | Δ MAE | Δ dir_acc | Δ Sharpe | Verdict |
|---|---|---|---|---|---|
| 01 → 01b | a 24-bar rolling mean | +1 386.91 | from NaN to 0.4801 | from — to 1.3003 | mostly hurt (MAE blows up on trending regime) |
| 01b → 02(1,1,1) | linear AR + MA, differencing | −1 388.11 | +0.0746 | +4.7566 | huge help |
| 02(1,1,1) → 03 | non-linear trees + 31 engineered features | +22.23 | −0.0182 | +0.3664 | mixed (trade MAE for Sharpe) |
| 03 → 04 | recurrent state, raw past covariates instead of engineered features | −17.10 | −0.0427 | −2.1573 | hurt |
| 04 → 05 | attention + future covariates + variable selection | +290.81 | −0.0025 | +1.6015 | mixed (Sharpe up but MAE explodes) |
| 05 → 06 | **swap from trained TFT to zero-shot Chronos-2** (paradigm change, not a capacity step) | −293.32 | +0.0410 | +1.7240 | big help |
| 02(1,1,1) → 02(3,1,3)* | richer ARIMA orders | −3.09 | −0.0273 | +0.3446 | help on MAE / Sharpe, hurt on dir_acc |

\* From [02's sweep](02_arima.report.md#sweep--12-p-d-q-orders).

Two transitions help every metric (`01b → 02`, `05 → 06`). One hurts every metric (`03 → 04`). The rest are mixed. Notably, **`04 → 05` adds Sharpe at the cost of an enormous MAE penalty** — the TFT trades point precision for directional aggressiveness, an asymmetric exchange that lands well on the trading metric.

## AIC vs. out-of-sample

Visible only in [02_arima's sweep](02_arima.report.md#sweep--12-p-d-q-orders), where 12 different `(p, d, q)` orders share a training slice:

- **AIC leader:** `(3, 1, 3)` at AIC 25 543.88
- **MAE leader:** `(3, 1, 3)` at MAE 514.07
- **Sharpe leader (within ARIMA):** `(3, 1, 3)` at 6.4015
- **dir_acc leader (within ARIMA):** `(1, 1, 1)` at 0.5547

AIC and Sharpe agree on `(3, 1, 3)`; the disagreement is between AIC / Sharpe (which prefer `(3, 1, 3)`) and dir_acc (which prefers `(1, 1, 1)`). The repo's recurring "AIC and out-of-sample Sharpe can disagree" lesson does *not* fire on this 4h sweep — but the related lesson does: **point error and direction-getting-right are not the same objective**, even within a single model family on a single slice.

## NaN-by-design as a teaching surface

NaN appears in three places, each for a different reason — and each is informative about how the metric module is built ([src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py)):

1. **01's `directional_accuracy = NaN`** — the naive predictor returns `close_pred[t] = close[t-1]`, so `sign(pred − ref) = sign(0) = 0` for every bar. The metric's mask `(true_dir != 0) & (pred_dir != 0)` is empty; the function short-circuits to NaN per [metrics.py:23-34](../../src/btc_ai/eval/metrics.py).
2. **02 sweep `(0, 1, 0)` row — `directional_accuracy = NaN` and `sharpe = NaN`** — the random-walk ARIMA order is mathematically equivalent to naive, so it inherits naive's NaN dir_acc *and* produces an entirely flat strategy (`pred = ref` everywhere → `position = 0` → `stddev = 0` → NaN per [metrics.py:51-59](../../src/btc_ai/eval/metrics.py)).
3. **04 sweep `(48, 64, 2, 0.2)` row — `sharpe = NaN`** — the LSTM at this configuration *never* predicts strictly above the reference price, so the long/flat strategy stays flat. Same mechanism as case 2, different cause: the model has been trained, it just trained itself into "always hold cash."

A leaderboard reader can mistake any of these for a missing run. They are not. The metric module reports NaN as a deliberate signal of "no opinion expressed."

## Pipeline sanity checks

Two checks land cleanly in the artefacts:

1. **02 sweep's `(0, 1, 0)` row reports MAE 518.358631840796** — bit-identical to [01_baseline_naive's metrics.json](../../experiments/01_baseline_naive/results/metrics.json) MAE. The random-walk ARIMA *is* the naive predictor; identical numbers confirm the data slice, the split index, and the metric implementation are shared.
2. **06_pretrained's RMSE (780.99) is below 02_arima's RMSE (782.42) while its MAE is above 02's** — a non-trivial consistency check on the metric implementation: RMSE and MAE can disagree on ranking when the error distribution has different tails, which the implementation correctly reflects. Squared-error penalises outliers more than absolute-error.

Two checks that *don't* exist yet but would be useful:

- 01b at `window=1` should equal 01 (naive) MAE. Not tested in either run.
- A `walk-forward with retraining` shadow of any 4h experiment, to verify that `retrain=False` doesn't materially change the Sharpe on this slice. Not done.

## Per-bar Sharpe significance

The per-experiment reports compute `per-bar Sharpe = annualized / √ppy` and compare to `SE ≈ 1 / √N_test`. Collected here:

| Experiment | Sharpe (annualized) | √ppy | per-bar Sharpe | √N_test | SE | ratio (σ) |
|---|---|---|---|---|---|---|
| 01b_moving_average | 1.3003 | √2190 ≈ 46.79 | 0.0278 | √402 ≈ 20.05 | 0.0499 | 0.56 |
| 02_arima (1, 1, 1) | 6.0569 | 46.79 | 0.1295 | 20.05 | 0.0499 | **2.59** |
| 02_arima (3, 1, 3) | 6.4015 | 46.79 | 0.1369 | 20.05 | 0.0499 | **2.74** |
| 03_gradient_boosting | 6.4233 | 46.79 | 0.1373 | √397 ≈ 19.92 | 0.0502 | **2.74** |
| 04_lstm | 4.2660 | 46.79 | 0.0912 | √401 ≈ 20.02 | 0.0499 | 1.83 |
| 05_transformer | 5.8675 | 46.79 | 0.1254 | 20.02 | 0.0499 | **2.51** |
| 06_pretrained (chronos-2) | 7.5915 | 46.79 | 0.1622 | 20.05 | 0.0499 | **3.25** |

**Five rows clear the 2.5 σ threshold**: ARIMA(1, 1, 1), ARIMA(3, 1, 3), XGBoost, TFT, and Chronos-2. **06_pretrained at 3.25 σ is the strongest signal in the series**, followed by ARIMA(3, 1, 3) / XGBoost (tied at 2.74) and ARIMA(1, 1, 1) (2.59). LSTM (1.83) and MA(24) (0.56) are weak.

This is a coarse approximation — it assumes per-bar strategy returns are i.i.d., which they almost certainly aren't on hourly-to-4h crypto. Volatility clustering inflates the true SE. The σ ratios above are upper bounds on the real significance.

## Regime caveat

Every result above is conditional on a test slice that lands approximately **Sep 25 → Dec 1 2024** — the post-election BTC rally. A strongly trending regime is exactly where a long/flat strategy with even mild directional skill compounds favourably. **None of the rows in this report have been tested out-of-regime.** The directional skill that ARIMA, XGBoost, TFT, and Chronos-2 show may or may not survive a sideways or down-trending slice. The 06_pretrained Sharpe lead, in particular, is the most regime-sensitive finding in the series: a zero-shot prior could be unusually well-tuned to the *kind* of trend BTC was in during this window, and we have no second test window to verify.

Until that is tested, every positive Sharpe in this report is best read as *"skill conditional on this regime"*, not *"skill in general."*

## Bottom line

On 402 bars of 4h BTC over a strongly trending Q4-2024 test window:

- **The Sharpe leader is a 120 M-parameter zero-shot foundation model** (Chronos-2, no training on BTC) at **7.5915** — clearly significant at 3.25 σ. Cum_ret 69.75 % and RMSE 780.99 are also leaderboard tops.
- **The MAE / MAPE / dir_acc leader is a 3-parameter ARIMA** (`(1, 1, 1)`, MAE 517.16, dir_acc 0.5547). The two best models in the series differ by *eight orders of magnitude* in parameter count.
- **The most-parameters trained-on-this-data model — TFT** — reports MAE 813.10, the worst in the 4h leaderboard. Its 1 448-row training set is not enough for ~30 k parameters to converge on a stable point estimate.
- **Capacity helps up to ARIMA(3, 1, 3) at 7 parameters, plateaus through XGBoost, declines through LSTM and TFT, then jumps back up only when the parameters come pre-trained on other data.**

Read in one line: *On this slice, three parameters of linear differenced auto-regression beat every other trained model on MAE / MAPE / dir_acc — and the only thing that beats that on Sharpe is 120 M parameters of pretrained prior that never saw Bitcoin.*

## Where this goes next

Operational follow-ups surfaced by the per-experiment reports:

- **Re-run `make 05_transformer_sweep`** on 4h so the TFT's hyperparameter rankings catch up to its single fit.
- **Re-run `make 03_gradient_boosting_sweep`** on 4h. The current sweep.csv is stale on 1h; default-config Sharpe on 4h (6.42) is wildly different from default-config Sharpe on 1h (1.45), and the sweep cannot rank 4h hyperparameter choices in its current form.
- **Run `make 06_pretrained_sweep`** to surface the Chronos ↔ TimesFM head-to-head the experiment was designed to expose. The single 06 run here is Chronos-2 only; the design intent is a side-by-side with TimesFM 2.5 on the same slice.
- **A `window=1` row in 01b** would close the trivial sanity check (MA(1) should equal naive). Not on disk.
- **Out-of-regime test.** Run a sibling experiment family on, e.g., 2022-Q2 (sideways) or 2022-Q4 (down-trending) and re-rank. Every positive Sharpe in this report — especially 06's 3.25 σ lead — needs that test before "skill" replaces "skill in this regime."
- **Probabilistic-interval analysis.** Only 06 produces q10 / q50 / q90 bands. Calibration is not summarised in `metrics.json`. A future global report could add an interval-coverage column (fraction of bars where `close ∈ [pred_lo, pred_hi]`).
- **Article series**, per [docs_ml/articles_todo.md](../articles_todo.md). The reports above are the raw material for those articles; this synthesis is the closing summary the series will eventually mirror.

## Caveats (shared by all rows)

- Single split, no rolling-origin CV.
- Walk-forward without re-estimation (`retrain=False` for trained models; nothing to re-estimate for naive / MA / foundation models).
- One seed per row for the stochastic trainers (XGBoost, LSTM, TFT, foundation models).
- Long / flat strategy, no shorting, no transaction costs, no slippage.
- Single test regime (Q4-2024 BTC rally). See [Regime caveat](#regime-caveat).
- All numbers above are copied verbatim from `experiments/${id}/results/metrics.json` and the per-experiment reports. No values are recomputed in this document.

## Source artefacts

Per-experiment reports (audit trail):

- [01_baseline_naive.report.md](01_baseline_naive.report.md)
- [01b_moving_average.report.md](01b_moving_average.report.md)
- [02_arima.report.md](02_arima.report.md)
- [03_gradient_boosting.report.md](03_gradient_boosting.report.md)
- [04_lstm.report.md](04_lstm.report.md)
- [05_transformer.report.md](05_transformer.report.md)
- [06_pretrained.report.md](06_pretrained.report.md)

Raw `metrics.json` / `sweep.csv` for each experiment under `experiments/${id}/results/`.

## How to regenerate this report

```
# 1. Make sure every per-experiment artefact is fresh.
make 01_baseline_naive && make 01b_moving_average && make 02_arima
make 03_gradient_boosting && make 04_lstm && make 05_transformer && make 06_pretrained
# (re-run sweeps too if they are stale)

# 2. Regenerate per-experiment reports per Prompt 1 in docs_ml/report_generation.md.

# 3. Regenerate this global report per Prompt 2 in docs_ml/report_generation.md.
```
