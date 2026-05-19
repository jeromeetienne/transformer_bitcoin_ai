# Report — `02_arima`

**Run date:** 2026-05-19
**Status:** completed (single fit + 12-order sweep)

## What this experiment is

First "real" model in the lineup. Fits an `ARIMA(p, d, q)` on the training slice and produces walk-forward 1-step-ahead forecasts on the held-out test slice. ARIMA is a linear, stationary model of price *increments* (`d=1` differences the price level once before modelling). It can capture short-term momentum (positive AR coefficient → recent up-moves continue) or mean-reversion (negative AR) plus a short residual-correction term via MA. The differentiator vs. 01 / 01b: this is the first model that actually *estimates parameters from data*.

```
ARIMA(p, d, q)
   p : AR — autoregressive lags of the differenced series
   d : I  — differencing order (d=1 → model price changes, not the level)
   q : MA — moving-average lags of the residuals
```

Library: **statsmodels** (`statsmodels.tsa.arima.model.ARIMA`). Walk-forward shape: fit once on the train slice, extend via `fit.apply(full_series, refit=False)` so the trained `(p, d, q)` coefficients stay frozen while the model sees new observations, then `predict(start=split, end=last, dynamic=False)` — `dynamic=False` is the key flag, each prediction at `t` uses *actual* past values (true 1-step-ahead walk-forward). See [experiments/02_arima/README.md](../../experiments/02_arima/README.md) for the full narrative.

## Configuration

From [experiments/02_arima/config.yaml](../../experiments/02_arima/config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `4h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| order | `(1, 1, 1)` |

Total bars: **2 010**. Train: **1 608**. Test: **402**.

## Results — single fit (ARIMA(1, 1, 1))

From [experiments/02_arima/results/metrics.json](../../experiments/02_arima/results/metrics.json):

| Metric | Value |
|---|---|
| AIC | 25 549.66 |
| BIC | 25 565.80 |
| MAE | 517.16 USD |
| RMSE | 782.42 USD |
| MAPE | 0.6686 % |
| Directional accuracy | **0.5547** |
| Cumulative return | 0.3314 |
| Annualized Sharpe | **6.0569** |

## Cross-experiment comparison

4h-slice leaderboard. Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| [01_baseline_naive](01_baseline_naive.report.md) | 518.36 | 784.09 | 0.6701 % | NaN | — | — |
| [01b_moving_average (window=24)](01b_moving_average.report.md) | 1 905.27 | 2 580.61 | 2.4782 % | 0.4801 | 0.0511 | 1.3003 |
| **02_arima (1, 1, 1)** | **517.16** | **782.42** | **0.6686 %** | **0.5547** | 0.3314 | 6.0569 |
| [03_gradient_boosting (n_feat=31)](03_gradient_boosting.report.md) | 539.39 | 795.07 | 0.7008 % | 0.5365 | **0.5042** | **6.4233** |
| [04_lstm](04_lstm.report.md) | 522.29 | 785.45 | 0.6761 % | 0.4938 | 0.2596 | 4.2660 |

02_arima leads four columns out of six (MAE, RMSE, MAPE, dir_acc) on the 4h slice with the smallest non-trivial parametric model in the leaderboard — three estimated parameters. It only loses cum_ret and Sharpe to 03_gradient_boosting (50 % vs. 33 % cumulative; 6.42 vs. 6.06 Sharpe), and only by a hair. The MAE win over naive is **$1.20** — a margin that would not survive a different train/test split, and is best read as "ARIMA ties naive on point error but adds directional skill where naive has none." (Experiments 05 and 06 currently report on a stale 1h slice — see [05_transformer.report.md](05_transformer.report.md) and [06_pretrained.report.md](06_pretrained.report.md).)

## Sweep — 12 (p, d, q) orders

From [experiments/02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv) on the same 4h data slice. Sorted in source order (Box-Jenkins canonical ramp + endpoints). Leaders bolded per column.

| order | AIC | BIC | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|---|
| (0, 1, 0) | 25 546.20 | 25 551.58 | 518.36 | 784.09 | 0.6701 % | NaN | 0.0000 | NaN |
| (1, 1, 0) | 25 547.86 | 25 558.62 | 517.58 | 782.96 | 0.6692 % | 0.5174 | 0.2785 | 5.2043 |
| (0, 1, 1) | 25 547.87 | 25 558.64 | 517.61 | 783.01 | 0.6692 % | 0.5124 | 0.2619 | 4.9076 |
| **(1, 1, 1)** | 25 549.66 | 25 565.80 | 517.16 | 782.42 | 0.6686 % | **0.5547** | 0.3314 | 6.0569 |
| (2, 1, 0) | 25 549.35 | 25 565.50 | 516.87 | 781.95 | 0.6682 % | 0.5224 | 0.2316 | 3.9724 |
| (0, 1, 2) | 25 549.35 | 25 565.49 | 516.88 | 781.95 | 0.6682 % | 0.5224 | 0.2321 | 3.9809 |
| (2, 1, 2) | 25 553.35 | 25 580.26 | 516.87 | 781.94 | 0.6682 % | 0.5249 | 0.2335 | 4.0009 |
| **(3, 1, 3)** | **25 543.88** | 25 581.56 | **514.07** | **776.69** | **0.6659 %** | 0.5274 | **0.5101** | **6.4015** |
| (5, 1, 0) | 25 554.83 | 25 587.12 | 517.06 | 780.59 | 0.6685 % | 0.5075 | 0.3577 | 5.4892 |
| (0, 1, 5) | 25 554.89 | 25 587.18 | 517.02 | 780.71 | 0.6684 % | 0.5000 | 0.3500 | 5.1418 |
| (5, 1, 5) | 25 550.49 | 25 609.69 | 522.58 | 786.41 | 0.6747 % | 0.5224 | 0.3312 | 4.5889 |
| (1, 0, 1) | 25 571.25 | 25 592.78 | 521.05 | 788.42 | 0.6728 % | 0.4726 | 0.0087 | 0.5560 |

The (3, 1, 3) row leads on five columns simultaneously: AIC, MAE, RMSE, MAPE, cum_ret, Sharpe — all five point-error and strategy metrics. The configured default (1, 1, 1) leads only on directional accuracy. Differentiation between adjacent orders on MAE is well under a dollar — treat sub-dollar MAE deltas as noise.

## Interpretation

1. **AIC and Sharpe agree on (3, 1, 3).** Lowest AIC (25 543.88), lowest MAE (514.07), highest Sharpe (6.40). Unusually, the in-sample selection criterion picks the same order the out-of-sample trading metric prefers. The disagreement that *does* show up is between AIC/Sharpe and **directional accuracy** — the configured (1, 1, 1) leads dir_acc at 0.5547, while (3, 1, 3) is at 0.5274 despite a richer AR / MA spec. Different criteria can rank differently; this sweep makes that visible.
2. **(0, 1, 0) returns the naive baseline exactly.** AIC 25 546.20 with MAE 518.358631840796 — bit-identical to [01_baseline_naive's metrics.json](../../experiments/01_baseline_naive/results/metrics.json). That row is the random-walk specification (no AR, no MA, one difference); mathematically it *is* `close_pred[t] = close[t-1]`. Pipeline sanity check passes.
3. **(1, 0, 1) — only un-differenced order in the sweep — collapses on Sharpe** to 0.5560. Differencing is load-bearing: BTC price levels are non-stationary, and an ARMA without differencing fits the level and fails the strategy. d ≥ 1 is non-optional.
4. **MAE differences between (1, 1, 1) and (3, 1, 3) are $3** on a 402-bar slice. On a different split, the rank order between them could flip. The robust reading is that *both* are useful, with (3, 1, 3) more aggressive on direction and (1, 1, 1) marginally better at the no-opinion direction-call.
5. **(5, 1, 5) is the overfit shape.** Higher AIC than smaller orders, worse MAE, lower Sharpe than (3, 1, 3) — extra AR / MA coefficients past 3 deliver no signal on this data slice and add variance.

### Bottom line

ARIMA(1, 1, 1) ties naive on MAE within **$1.20** (517.16 vs. 518.36) and takes Sharpe to **6.06** with `dir_acc 0.5547`. Per-bar Sharpe ≈ 6.0569 / √2190 = 0.1295; standard error ≈ 1 / √402 = 0.0499; ratio ≈ **2.59 σ** — significant at roughly the 99 % level. The richer (3, 1, 3) order pushes Sharpe to 6.40 (per-bar 0.1369, ratio ≈ 2.74 σ) on the same slice. Both numbers are conditional on the test window: approximately Sep 25 → Dec 1 2024, which covers the post-election BTC rally. The directional skill ARIMA shows here is genuine but has not been tested out-of-regime.

## Caveats

- Single fit, no order search inside the experiment — orders are chosen explicitly. Sweep above samples a discrete grid; no auto-ARIMA.
- Walk-forward without re-estimation. `(p, d, q)` coefficients are frozen at train-time values; the model sees new observations but does not refit at each step.
- ARIMA assumes residuals are roughly Gaussian and homoskedastic; BTC log-returns are heavy-tailed and exhibit volatility clustering. Point forecasts may look reasonable; interval forecasts (not produced here) would be badly miscalibrated.
- No exogenous regressors (this is ARIMA, not ARIMAX). Volume, OHLC, on-chain features, sentiment all enter starting at 03.
- Long/flat strategy, no shorting, no transaction costs.

## Files produced

- [experiments/02_arima/results/metrics.json](../../experiments/02_arima/results/metrics.json)
- [experiments/02_arima/results/predictions.parquet](../../experiments/02_arima/results/predictions.parquet)
- [experiments/02_arima/results/plot.png](../../experiments/02_arima/results/plot.png)
- [experiments/02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv)

## How to reproduce

```
make 02_arima
make 02_arima_sweep
```
