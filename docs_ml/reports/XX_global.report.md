# Report — `XX_global` — cross-experiment comparison

**Run date:** 2026-05-19
**Status:** synthesis of seven per-experiment reports under [docs_ml/reports/](.)

## What this report is

A single-file leaderboard and analytical comparison of every experiment in the repo: [01_baseline_naive](01_baseline_naive.report.md), [01b_moving_average](01b_moving_average.report.md), [02_arima](02_arima.report.md), [03_gradient_boosting](03_gradient_boosting.report.md), [04_lstm](04_lstm.report.md), [05_transformer](05_transformer.report.md), [06_pretrained](06_pretrained.report.md). Each number cited here is verbatim from the corresponding `experiments/${id}/results/metrics.json`; the per-experiment reports are the audit trail.

Sections progress from raw numbers to analytical observations. The intent is not to repeat what the per-experiment reports already say but to put the rows side-by-side and read the *gradient* across the ladder of model classes.

## Data-slice picture (load-bearing)

The seven experiments are **not** all on the same data slice. Commit 81d4fcb switched every `config.yaml` from `interval: 1h` to `interval: 4h`; experiments 01 / 01b / 02 / 03 / 04 have been re-run since, but 05 and 06 still report on stale 1h artefacts.

| # | `interval` (config) | `rows_total` (artefact) | `rows_test` | `metrics.json` mtime | Slice |
|---|---|---|---|---|---|
| 01_baseline_naive | 4h | 2 010 | 402 | 2026-05-19 | fresh 4h |
| 01b_moving_average | 4h | 2 010 | 402 | 2026-05-19 | fresh 4h |
| 02_arima | 4h | 2 010 | 402 | 2026-05-19 | fresh 4h |
| 03_gradient_boosting | 4h | 1 985 | 397 | 2026-05-19 | fresh 4h (but `sweep.csv` is stale 1h) |
| 04_lstm | 4h | 2 009 | 401 | 2026-05-19 | fresh 4h |
| 05_transformer | 4h | 8 039 | 1 607 | 2026-04-29 | **stale 1h** |
| 06_pretrained | 4h | 8 039 | 1 608 | 2026-04-29 | **stale 1h** |

The 4h test slice is roughly **402 bars from approximately Sep 25 → Dec 1 2024** (last 20 % of the 11-month range). The 1h test slice is the same calendar window with 4× the granularity — same regime, finer resolution. Two leaderboards follow.

## Leaderboard — 4h slice (01, 01b, 02, 03, 04)

| Experiment | MAE (USD) | RMSE (USD) | MAPE | dir_acc | cum_ret | annualized_sharpe |
|---|---|---|---|---|---|---|
| [01_baseline_naive](01_baseline_naive.report.md) | 518.36 | 784.09 | 0.6701 % | NaN | — | — |
| [01b_moving_average (window=24)](01b_moving_average.report.md) | 1 905.27 | 2 580.61 | 2.4782 % | 0.4801 | 0.0511 | 1.3003 |
| [02_arima (1, 1, 1)](02_arima.report.md) | **517.16** | **782.42** | **0.6686 %** | **0.5547** | 0.3314 | 6.0569 |
| [03_gradient_boosting (n_feat=31)](03_gradient_boosting.report.md) | 539.39 | 795.07 | 0.7008 % | 0.5365 | **0.5042** | **6.4233** |
| [04_lstm (default)](04_lstm.report.md) | 522.29 | 785.45 | 0.6761 % | 0.4938 | 0.2596 | 4.2660 |

**Who leads what:** 02_arima takes four columns out of six (MAE, RMSE, MAPE, dir_acc) with three estimated parameters. 03_gradient_boosting takes cum_ret and Sharpe — at the cost of being **worse than naive on MAE**. 04_lstm leads nothing and trails 02_arima on every column. 01b is the anti-baseline: a row everything else must clear by more than 3×.

**The narrow MAE band.** Three of the five rows — 01, 02, 04 — sit within $5 of each other on MAE (518, 517, 522). 03's MAE is $22 higher than 01's. On a 402-bar test slice these gaps are small enough that a different split could flip 01 vs. 02 or 01 vs. 04. Treat sub-$5 MAE deltas as noise.

**The MA(24) shape.** 01b's MAE 1 905 is 3.68× the naive floor — not because moving averages are bad in general, but because this particular window lags hard on a trending slice. On a sideways slice, 01b would look very different. This row teaches a *regime* lesson more than a model lesson.

## Leaderboard — 1h slice (05, 06, stale)

| Experiment | MAE (USD) | RMSE (USD) | MAPE | dir_acc | cum_ret | annualized_sharpe |
|---|---|---|---|---|---|---|
| [05_transformer (default)](05_transformer.report.md) | 373.70 | 546.68 | 0.4879 % | **0.5053** | **0.2957** | **4.5864** |
| [06_pretrained (timesfm)](06_pretrained.report.md) | **268.89** | **405.14** | **0.3521 %** | 0.4677 | 0.1690 | 2.4444 |

A trained TFT with covariates leads the trading metrics; a zero-shot 200M-parameter foundation model leads point error by hedging. **Both rows will change** once 05 and 06 are re-run on 4h (see [Where this goes next](#where-this-goes-next)).

## The complexity-doesn't-pay arc

Ordering by approximate parameter count for the 4h experiments:

| Row | Approx. trainable params | Sharpe | dir_acc | MAE rank (5 = best) |
|---|---|---|---|---|
| 01_baseline_naive | 0 | — | NaN | 4 |
| 01b_moving_average (24) | 0 (just a window) | 1.3003 | 0.4801 | 1 |
| 02_arima (1, 1, 1) | 3 | 6.0569 | **0.5547** | **5** |
| 02_arima (3, 1, 3)* | 7 | **6.4015** | 0.5274 | — |
| 03_gradient_boosting | ~31 features × 400 trees × depth 5 (effectively thousands of leaf splits) | 6.4233 | 0.5365 | 2 |
| 04_lstm (default) | ~10 000 weights (`hidden_dim=32`, 2 layers) | 4.2660 | 0.4938 | 3 |

\* From [02_arima's sweep](02_arima.report.md#sweep--12-p-d-q-orders), included here to surface the AIC winner.

Three observations:

1. **The Sharpe peak is somewhere between 3 and 31 parameters.** ARIMA(3, 1, 3) at 7 params and XGBoost at 31 features sit ~3 % apart on Sharpe (6.40 vs. 6.42); both clear ARIMA(1, 1, 1) at 3 params (6.06) by a comparable margin. After that, doubling parameters to ~10 000 (the LSTM) *drops* Sharpe by ~30 %. There is no monotone capacity → skill relationship on this signal.
2. **The directional-accuracy peak is at 3 parameters.** ARIMA(1, 1, 1) leads dir_acc on the 4h slice at 0.5547. Every model with more parameters is *lower* on dir_acc — including the (3, 1, 3) ARIMA, the gradient-booster, and the LSTM. This is the cleanest "simple beats complex" beat in the leaderboard.
3. **04_lstm has the right shape for the "deep learning didn't pay off" punchline.** It is the first row in the ladder where adding capacity costs trading metrics relative to a lower-capacity model on the same data. dir_acc 0.4938 is below coin-flip; Sharpe trails ARIMA's by 30 %. The LSTM is using its weights, just not productively.

## Point-error vs. trading-metric tradeoff

The 4h leaderboard surfaces an explicit tension: **the MAE leader and the Sharpe leader are different models**.

- 02_arima(1, 1, 1) leads MAE (517.16), RMSE (782.42), MAPE (0.6686 %), and dir_acc (0.5547).
- 03_gradient_boosting leads cum_ret (50.42 %) and Sharpe (6.4233).

The mechanism is visible in the strategy column. XGBoost's directional confidence is *higher* (i.e. it predicts more strictly above-reference more often) than ARIMA's, which makes the long/flat rule stay long for more bars during the rally. That converts a lower dir_acc (0.5365 vs. 0.5547) into a higher cum_ret (50.42 % vs. 33.14 %). The cost is paid in MAE — being further from the reference price on average inflates point error even when the direction is right.

**On this slice, you don't get both.** Pick a model for the metric you actually care about.

## Where capacity helps and where it hurts

Pair-by-pair on the 4h slice (each pair compares with **one** more "thing" added to the lower row):

| Pair | What's added | Δ MAE | Δ dir_acc | Δ Sharpe | Verdict |
|---|---|---|---|---|---|
| 01 → 01b | a 24-bar rolling mean | +1 386.91 | from NaN to 0.4801 | from — to 1.3003 | mostly hurt (MAE blows up on trending regime) |
| 01b → 02(1,1,1) | linear AR + MA, differencing | −1 388.11 | +0.0746 | +4.7566 | huge help |
| 02(1,1,1) → 03 | non-linear trees + 31 engineered features | +22.23 | −0.0182 | +0.3664 | mixed (trade MAE for Sharpe) |
| 03 → 04 | recurrent state, raw past covariates instead of engineered features | −17.10 | −0.0427 | −2.1573 | hurt (lower dir_acc and Sharpe) |
| 02(1,1,1) → 02(3,1,3)* | richer ARIMA orders | −3.09 | −0.0273 | +0.3446 | help on MAE / Sharpe, hurt on dir_acc |

\* From [02's sweep](02_arima.report.md#sweep--12-p-d-q-orders).

The pattern: linear non-stationary modelling (`d ≥ 1`) is the one transition that pays off across all metrics. Every other transition is mixed at best.

## AIC vs. out-of-sample

Visible only in [02_arima's sweep](02_arima.report.md#sweep--12-p-d-q-orders), where 12 different `(p, d, q)` orders share a training slice:

- **AIC leader:** `(3, 1, 3)` at AIC 25 543.88
- **MAE leader:** `(3, 1, 3)` at MAE 514.07
- **Sharpe leader:** `(3, 1, 3)` at 6.4015
- **dir_acc leader:** `(1, 1, 1)` at 0.5547

AIC and Sharpe agree here; the disagreement is between AIC / Sharpe (which prefer `(3, 1, 3)`) and dir_acc (which prefers `(1, 1, 1)`). The repo's recurring lesson — "AIC and out-of-sample Sharpe can disagree" — does *not* fire in this sweep on the 4h slice. The lesson it does deliver is **"point error and direction-getting-right are not the same"**, even within a single model family on a single slice.

## NaN-by-design as a teaching surface

NaN appears in three places, each for a different reason — and each is informative about how the metric module is built ([src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py)):

1. **01's `directional_accuracy = NaN`** — the naive predictor returns `close_pred[t] = close[t-1]`, so `sign(pred - ref) = sign(0) = 0` for every bar. The metric's mask `(true_dir != 0) & (pred_dir != 0)` is empty; the function short-circuits to NaN per [metrics.py:23-34](../../src/btc_ai/eval/metrics.py).
2. **02 sweep `(0, 1, 0)` row — `directional_accuracy = NaN` and `sharpe = NaN`** — the random-walk ARIMA order is mathematically equivalent to naive, so it inherits naive's NaN dir_acc *and* produces an entirely flat strategy (`pred = ref` everywhere → `position = 0` → `stddev = 0` → NaN per [metrics.py:51-59](../../src/btc_ai/eval/metrics.py)).
3. **04 sweep `(48, 64, 2, 0.2)` row — `sharpe = NaN`** — the LSTM at this configuration *never* predicts strictly above the reference price, so the long/flat strategy stays flat. Same mechanism as case 2, different cause: the model has been trained, it just trained itself into "always hold cash."

A leaderboard reader can mistake any of these for a missing run. They are not. The metric module reports NaN as a deliberate signal of "no opinion expressed."

## Pipeline sanity checks

Two checks land cleanly in the artefacts:

1. **02 sweep's `(0, 1, 0)` row reports MAE 518.358631840796** — exactly equal to [01_baseline_naive's metrics.json](../../experiments/01_baseline_naive/results/metrics.json) MAE. The random-walk ARIMA *is* the naive predictor; bit-identical numbers confirm the data slice, the split index, and the metric implementation are shared.
2. **05_transformer sweep's `(48, 32, 4, 1, 0.1)` row matches 05's single-fit `metrics.json`** to all decimals. Same code path, same hyperparameters; runs agree. (Both rows are stale on 1h; the check still validates the sweep harness.)

Two checks that *don't* exist yet but would be useful:

- 01b at `window=1` should equal 01 (naive). Not tested in either run.
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
| 05_transformer | 4.5864 | √8760 ≈ 93.60 | 0.0490 | √1607 ≈ 40.09 | 0.0249 | 1.97 |
| 06_pretrained (timesfm) | 2.4444 | 93.60 | 0.0261 | √1608 ≈ 40.10 | 0.0249 | 1.05 |

Two cleanly significant rows (>2.5 σ): the two ARIMA variants and XGBoost. Three borderline rows (1–2 σ): LSTM, TFT, MA. One within-noise row (<1 σ): TimesFM. **The cleanest evidence of an edge on the 4h slice is in the ARIMA family**; XGBoost matches it on Sharpe but pays for it in MAE.

This is a coarse approximation — it assumes per-bar strategy returns are i.i.d., which they probably aren't on hourly-to-4h crypto. Volatility clustering inflates the *true* SE. The σ ratios above are therefore upper bounds on the real significance.

## Regime caveat

Every result above is conditional on a test slice that lands approximately **Sep 25 → Dec 1 2024**. That window contains the post-election BTC rally — a strongly trending regime where a long/flat strategy with even mild directional skill compounds favourably. None of the rows in this report have been tested out-of-regime. The directional skill that ARIMA, XGBoost, and (on 1h) TFT show may or may not survive a sideways or down-trending slice. **Until that is tested, every positive Sharpe in this report is best read as "skill conditional on this regime", not "skill in general."**

## Bottom line

On 402 bars of 4h BTC over a strongly trending Q4-2024 test window:

- **The directional-accuracy leader is a 3-parameter ARIMA** (`(1, 1, 1)`, dir_acc 0.5547). The most-parameters model in the ladder — `04_lstm` with ~10 000 weights — comes in *below coin-flip* at 0.4938.
- **The Sharpe leader is XGBoost on 31 engineered features** at 6.4233. It is **worse than naive on MAE**. ARIMA(3, 1, 3) is within 0.3 % of it on Sharpe with 7 parameters.
- **The two strongest signals (>2.5 σ above zero) belong to the ARIMA family and XGBoost.** LSTM, TFT, MA are borderline (1–2 σ); the zero-shot foundation model is within noise.
- **The 1h leaderboard is stale.** 05 and 06 have not been re-run since the project switched to 4h. Their numbers are honest but not directly comparable.

Read in one line: *on this regime, 3 parameters of linear differenced auto-regression beat every other model in the ladder on every metric that does not directly reward directional aggressiveness — and the model that does win on directional aggressiveness (XGBoost) pays for it in point-error fit.*

## Where this goes next

Operational follow-ups surfaced by the per-experiment reports:

- **Re-run `make 05_transformer` and `make 06_pretrained`** on 4h to put 05 and 06 in the same leaderboard as 01–04. Then re-run `make 06_pretrained_sweep` to surface the Chronos ↔ TimesFM head-to-head the experiment was designed to expose.
- **Re-run `make 03_gradient_boosting_sweep`** on 4h. The current sweep.csv is stale on 1h; default-config Sharpe on 4h (6.42) is wildly different from default-config Sharpe on 1h (1.45), and the sweep cannot rank 4h hyperparameter choices in its current form.
- **A `window=1` row in 01b** would close the trivial sanity check (MA(1) should equal naive). Not on disk.
- **Out-of-regime test.** Run a sibling experiment family on, e.g., 2022-Q2 (sideways) or 2022-Q4 (down-trending) and re-rank. Every positive Sharpe in this report needs that test before "skill" replaces "skill in this regime."
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
# 1. Make sure every per-experiment report is up to date.
make 01_baseline_naive && make 01b_moving_average && make 02_arima
make 03_gradient_boosting && make 04_lstm && make 05_transformer && make 06_pretrained
# (re-run sweeps too if they are stale)

# 2. Regenerate per-experiment reports per docs_ml/report_generation.md.

# 3. Regenerate this global report per the "Global cross-experiment report"
#    section in docs_ml/report_generation.md.
```
