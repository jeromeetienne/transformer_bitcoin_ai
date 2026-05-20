# Report — `03_gradient_boosting`

**Run date:** 2026-05-19
**Status:** completed (single fit on fresh 4h slice; sweep present but stale on 1h — see Sweep section)

## What this experiment is

First non-linear model in the lineup, and the first one that ingests engineered covariates. Fits an XGBoost regressor on a flat feature vector — lagged log-returns, rolling means and standard deviations of past log-returns, and (optionally) a per-bar OHLCV summary — with the bar-T log-return as target. Predictions are reconstructed to price as `close_pred = close[t-1] * exp(r_pred)` so MAE / RMSE / MAPE / dir_acc / Sharpe are directly comparable to 01 / 02. The differentiator vs. 02_arima: a non-linear estimator, and features beyond past closes alone.

```
XGBoost(reg:squarederror)
   ├── lagged log-returns        r_{T-1}, r_{T-2}, ..., r_{T-N}
   ├── rolling mean / std        on past log-returns (windows = 6, 24)
   ├── log_volume_{T-1}          log1p(volume_{T-1})
   └── hl_range_{T-1}, oc_body_{T-1}   per-bar OHLC summaries
```

Library: **xgboost** (`xgb.XGBRegressor` with `tree_method='hist'`). Walk-forward shape: every feature is `.shift(1)` over its source, so a single `model.predict(X_test)` call *is* a walk-forward of 1-step-ahead forecasts — no per-step refit needed. See [experiments/03_gradient_boosting/README.md](../../experiments/03_gradient_boosting/README.md) for the full narrative.

## Configuration

From [experiments/03_gradient_boosting/configs/btc_4h_2024.config.yaml](../../experiments/03_gradient_boosting/configs/btc_4h_2024.config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `4h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| features.return_lags | `24` |
| features.rolling_windows | `[6, 24]` |
| features.use_volume | `true` |
| features.use_ohlc | `true` |
| model.n_estimators | `400` |
| model.max_depth | `5` |
| model.learning_rate | `0.05` |
| model.subsample | `0.8` |
| model.colsample_bytree | `0.8` |
| model.reg_lambda | `1.0` |
| model.random_state | `42` |

Total bars: **1 985** (25 rows dropped by lag / rolling-window construction). Train: **1 588**. Test: **397**. Total features: **31** (24 lagged returns + 4 rolling stats over windows 6 and 24 + 1 `log_volume` + 2 OHLC scalars).

## Results — single fit

From [experiments/03_gradient_boosting/results/btc_4h_2024/metrics.json](../../experiments/03_gradient_boosting/results/btc_4h_2024/metrics.json):

| Metric | Value |
|---|---|
| MAE | 539.39 USD |
| RMSE | 795.07 USD |
| MAPE | 0.7008 % |
| Directional accuracy | 0.5365 |
| Cumulative return | **0.5042** |
| Annualized Sharpe | **6.4233** |

## Cross-experiment comparison

4h-slice leaderboard. Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| [01_baseline_naive](01_baseline_naive.report.md) | 518.36 | 784.09 | 0.6701 % | NaN | — | — |
| [02_arima (1, 1, 1)](02_arima.report.md) | **517.16** | **782.42** | **0.6686 %** | **0.5547** | 0.3314 | 6.0569 |
| **03_gradient_boosting (n_feat=31)** | 539.39 | 795.07 | 0.7008 % | 0.5365 | **0.5042** | **6.4233** |
| [04_lstm](04_lstm.report.md) | 522.29 | 785.45 | 0.6761 % | 0.4938 | 0.2596 | 4.2660 |

03_gradient_boosting is the **cum_ret / Sharpe leader** on the 4h slice but is *third* on MAE — **31 engineered features lose to a 3-parameter ARIMA on point error**, and lose to a *zero*-parameter naive predictor too. The Sharpe lead over ARIMA(1, 1, 1) is narrow (6.42 vs. 6.06); the cum_ret lead is wider (50 % vs. 33 %) because XGBoost's directional confidence translates into more bars long during the rally. The honest reading: extra capacity buys directional aggression at the cost of point-error fit. (Experiments 05 and 06 currently report on a stale 1h slice — see [05_transformer.report.md](05_transformer.report.md) and [06_pretrained.report.md](06_pretrained.report.md).)

## Sweep — 8 (n_estimators, max_depth, learning_rate) configs

> **Stale-slice warning.** The contents of [experiments/03_gradient_boosting/results/btc_4h_2024/sweep.csv](../../experiments/03_gradient_boosting/results/btc_4h_2024/sweep.csv) date from the previous 1h-data run and have **not** been regenerated since commit 81d4fcb (kline interval switch). The single-fit `metrics.json` above is fresh on 4h (MAE 539). The sweep MAEs (range 269–276) are on the 1h scale and **not directly comparable** to the headline numbers. Treat the sweep as a historical hyperparameter-ranking reference until re-run on 4h with `make 03_gradient_boosting_sweep`.

From the stale sweep, sorted in source order, leaders bolded per column:

| n_estimators | max_depth | learning_rate | MAE (1h) | RMSE (1h) | MAPE (1h) | dir_acc (1h) | cum_ret (1h) | sharpe (1h) |
|---|---|---|---|---|---|---|---|---|
| 200 | 3 | 0.10 | **269.05** | **412.09** | **0.3514 %** | 0.5041 | 0.2088 | 3.2624 |
| 200 | 5 | 0.10 | 274.86 | 416.41 | 0.3598 % | 0.4978 | 0.1371 | 2.2555 |
| 400 | 3 | 0.05 | 269.57 | 412.44 | 0.3519 % | 0.4978 | 0.1240 | 2.0892 |
| 400 | 5 | 0.05 | 270.73 | 414.07 | 0.3541 % | 0.4872 | 0.0809 | 1.4539 |
| 400 | 7 | 0.05 | 276.33 | 420.43 | 0.3609 % | 0.5022 | 0.0735 | 1.3215 |
| 800 | 5 | 0.03 | 272.53 | 415.99 | 0.3565 % | **0.5078** | **0.2578** | **3.9260** |
| 800 | 7 | 0.03 | 275.08 | 416.35 | 0.3593 % | 0.4972 | 0.1864 | 3.0612 |
| 1200 | 5 | 0.02 | 271.91 | 413.94 | 0.3556 % | 0.4891 | 0.0897 | 1.5543 |

Even within the stale slice the smallest configs (n_estimators=200, max_depth=3) lead on MAE, and (800, 5, 0.03) leads on Sharpe — bigger is not better on near-random-walk data. The default config `(400, 5, 0.05)` is not the sweep winner on either MAE or Sharpe.

## Interpretation

1. **31 features lose to a 3-parameter ARIMA on MAE** (539.39 vs. 517.16). Lagged log-returns + rolling moments + OHLC summaries cannot extract signal that a linear AR / MA on differenced prices misses. The lesson is about signal-to-noise at this horizon, not about XGBoost — feature breadth doesn't manufacture predictability.
2. **Sharpe leadership comes from directional aggression, not from point-error fit.** dir_acc 0.5365 < ARIMA's 0.5547, but cum_ret 0.5042 > ARIMA's 0.3314 — XGBoost makes the long bets *bigger* (predicts further above the reference price more often), so the long/flat rule stays long more often during the rally. That is the entire mechanism behind the Sharpe lead. It would invert on a sideways slice.
3. **MAE is higher than naive's** (539 vs. 518). This is the structurally interesting result: a model that *trains* manages to be worse on point error than the model that doesn't. The tree ensemble is fitting noise in the residuals while the linear momentum signal that AR(1) captures cheaply gets washed out.
4. **The sweep is stale on 1h and cannot rank current 4h configs.** Within the stale slice, the (800, 5, 0.03) configuration leads on Sharpe (3.93) and (n_estimators=200, max_depth=3, learning_rate=0.1) leads on MAE (269.05). The configured default (400, 5, 0.05) is neither — its 1h sweep Sharpe was 1.45. The 4h regime evidently behaves very differently (single-fit Sharpe 6.42 on 4h vs. 1.45 on the same default on 1h), almost certainly because the post-election rally fills a much larger fraction of the shorter 4h test slice than it did of the longer 1h test slice. **Rerun the sweep on 4h before reading hyperparameter rankings into the 4h leaderboard.**
5. **The (default) row of the sweep is not the Sharpe winner — and was not even close on 1h.** This is the recurring lesson in the repo: defaults set by `config.yaml` are starting points, not optima. Sweeps regularly find configs that do better on the trading metric, often at *lower* capacity than the default.

### Bottom line

XGBoost on 31 engineered features takes the 4h Sharpe to **6.4233** and cum_ret to **50.42 %** — both leaderboard tops — while *losing* to naive on MAE (539.39 vs. 518.36). Per-bar Sharpe ≈ 6.4233 / √2190 = 0.1373; standard error ≈ 1 / √397 = 0.0502; ratio ≈ **2.74 σ** — significant at roughly the 99 % level. Test slice covers approximately Sep 25 → Dec 1 2024 (post-election BTC rally); directional aggression pays disproportionately well in this regime and has not been tested out-of-regime. The pending re-sweep on 4h will tell whether the default `(400, 5, 0.05)` is the right config or whether a smaller tree count gets to the same Sharpe more cheaply.

## Caveats

- Single deterministic fit. `random_state` is fixed in config, but tree-building has run-to-run variance from `subsample` / `colsample_bytree`. A more rigorous comparison would average over 3–5 seeds.
- Walk-forward without re-estimation. Trees are frozen at train-time values; the test window is predicted in one pass.
- No early stopping. `n_estimators` is taken at face value; a held-out validation tail with `early_stopping_rounds` would prevent overfitting on the larger configs.
- Features are minimal — lagged returns + rolling moments + OHLCV summaries. No on-chain features, no exogenous price-of-other-assets, no sentiment.
- Squared-error loss on log-returns. Heavy tails bias the price MAE slightly relative to a quantile-loss reconstruction; the 0.7 % MAPE here likely under-states the worst-case bars.
- Sweep on stale 1h slice; do not use its MAE figures to rank against the headline 4h numbers above.

## Files produced

- [experiments/03_gradient_boosting/results/btc_4h_2024/metrics.json](../../experiments/03_gradient_boosting/results/btc_4h_2024/metrics.json) (fresh 4h)
- [experiments/03_gradient_boosting/results/btc_4h_2024/predictions.parquet](../../experiments/03_gradient_boosting/results/btc_4h_2024/predictions.parquet)
- [experiments/03_gradient_boosting/results/btc_4h_2024/plot.png](../../experiments/03_gradient_boosting/results/btc_4h_2024/plot.png)
- [experiments/03_gradient_boosting/results/btc_4h_2024/sweep.csv](../../experiments/03_gradient_boosting/results/btc_4h_2024/sweep.csv) (**stale 1h**)

## How to reproduce

```
make 03_gradient_boosting
make 03_gradient_boosting_sweep
```
