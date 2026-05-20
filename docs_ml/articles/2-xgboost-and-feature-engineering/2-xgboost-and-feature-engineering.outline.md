# Outline — Article 2: XGBoost and feature engineering

Working title: *"When the features do the work — XGBoost on engineered Bitcoin features."*

Source experiment: [`03_xgboost`](experiments/03_xgboost/).

## Editorial framing

The classical-machine-learning article. The contract is the opposite of the deep-learning articles that follow: the model is generic and shallow (a gradient-boosted tree ensemble), and the features carry the signal. By the time articles 3 and 4 arrive, the contract inverts — the deep-learning models eat raw past observations and learn their own representation.

The point of this article is not "XGBoost is or is not a good Bitcoin forecaster." It is the comparison: thirty-one engineered features against the linear ARIMA(3, 1, 3) from article 1. Result: XGBoost ties ARIMA on directional accuracy (0.5082 = 0.5082), loses on MAE by $11.70, and loses on Sharpe by 0.72. Feature breadth doesn't manufacture signal that wasn't there.

## Beats

1. **Hook.** Thirty-one engineered features versus three estimated ARIMA parameters. The thirty-one features lose on MAE, lose on Sharpe, and *tie* on directional accuracy. Feature engineering is real but the signal-to-noise ratio at this horizon doesn't reward it.

2. **Methodology recap** — the standard 1-paragraph block. Same target, same 366-bar test slice, same metric module, same walk-forward shape with weights frozen.

3. **What the model is — XGBoost in plain English.** A gradient-boosted ensemble of regression trees. Each tree fits the residual of the previous trees' ensemble. The model is generic and non-linear; what you feed it determines what it can learn. Library: `xgboost` (`XGBRegressor`, `tree_method='hist'`).

4. **The features — what 31 of them looks like.**
   - 24 lagged log-returns (`r_lag_1` through `r_lag_24`) — 24 bars at 4h = 96 hours of price-change history per row.
   - 4 rolling moments — `r_mean_6`, `r_std_6`, `r_mean_24`, `r_std_24` — short-term and day-long momentum and volatility.
   - 1 log-volume — `log1p(volume)`.
   - 2 OHLC summaries — `hl_range` (high - low) and `oc_body` (close - open).
   - Code link: [`experiments/03_xgboost/features.py`](experiments/03_xgboost/features.py).
   - Critical: every feature is `.shift(1)` over its source, so a row at time `T` only sees data observed strictly before `T`. No current-bar leakage. Walk-forward is implicit — a single `model.predict(X_test)` *is* the walk-forward, no per-step refit needed.

5. **The fit.** Standard XGBoost hyperparameters from [`configs/btc_4h_2024.config.yaml`](experiments/03_xgboost/configs/btc_4h_2024.config.yaml): 400 estimators, max_depth 5, lr 0.05, subsample 0.8, `colsample_bytree` 0.8, `random_state` 42. The target is the log-return `r_T`; the model predicts in log-return space; prediction is reconstructed to price as `close_pred = close[t-1] * exp(r_pred)`.

6. **The result — verbatim from [`metrics.json`](experiments/03_xgboost/results/btc_4h_2024/metrics.json).**
    - MAE 550.85 USD (worse than naive's 540.96)
    - RMSE 808.65 USD
    - MAPE 0.7096 %
    - dir_acc 0.5082 (identical to ARIMA — tie)
    - cum_ret 0.4149
    - Sharpe **6.1388**

7. **The honest comparison to ARIMA from article 1.**
   | | naive | ARIMA(3, 1, 3) | XGBoost |
   |---|---|---|---|
   | MAE | 540.96 | **539.15** | 550.85 |
   | Sharpe | — | **6.8559** | 6.1388 |
   | dir_acc | NaN | 0.5082 | 0.5082 |
   
   Two things to say honestly: XGBoost ties ARIMA on direction but doesn't beat it; XGBoost loses on MAE to *naive* (extra capacity fitted noise); XGBoost loses Sharpe to ARIMA. **31 engineered features lose to a 3-parameter ARIMA on every metric except directional accuracy, where they tie.**

8. **The sweep — eight configs.** Stale on 1h, can't be compared head-to-head. But the *ranking* lesson stands: the smallest configs (n_estimators=200, max_depth=3) lead on MAE; the configured default (400, 5, 0.05) is *not* the sweep's Sharpe winner on the older 1h data; bigger trees / deeper / longer is not better. Mention the slice-staleness explicitly.

9. **Why this happened — the model class teaching beat.**
   - 31 features lose to 3 parameters on MAE. The lesson is signal-to-noise at this horizon, not the choice between linear and non-linear. On a 4h BTC slice, lagged returns are mostly noise; rolling moments are mostly autocorrelation; OHLC summaries are mostly noise about the prior bar's volatility. A tree ensemble *can* model interactions among them, but on this signal-to-noise it mostly models noise.
   - The Sharpe pattern is the same shape as ARIMA's — both models get long during the rally, both are right about direction at roughly 0.51. The Sharpe difference is entirely in *magnitude calibration*; XGBoost's residual structure is noisier, so the long bets get sized smaller and fewer up-bars are collected.
   - When XGBoost wins is also informative: in older 1h sweeps it could clear ARIMA on Sharpe. On 4h with fewer training bars and stronger trend, it doesn't.

10. **Per-bar Sharpe significance.** 6.1388 / √2190 = 0.1312. SE ≈ 1 / √366 = 0.0523. Ratio ≈ **2.51 σ** — borderline significant. Lower than ARIMA's 2.80; same caveats apply.

11. **The XGBoost-is-still-useful caveat.** This is one data slice, one horizon, one model class. On 1h data or with richer features (on-chain, sentiment, cross-asset), XGBoost can do meaningfully better. The lesson of this article is **what 31 engineered features buy on the same 4h slice that article 1 worked on** — not "feature engineering is dead."

12. **Reproduce.** `make 03_xgboost` + `make 03_xgboost_sweep`.

## What this article is NOT

- An XGBoost tutorial. The library's documentation is comprehensive.
- A feature-engineering tutorial.
- A claim that this is the best XGBoost run possible. A richer feature set (on-chain data, sentiment, cross-asset prices) would change the picture and is out of scope.
