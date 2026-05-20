# Outline — Article 1: Baselines you need to beat

Working title: *"Forecasting Bitcoin with machine learning, Part 1: the baselines you need to beat."*

The first model article. Pairs the zero-parameter naive last-value baseline with the 3-parameter linear ARIMA. Establishes the methodology that every later article inherits and shows, in detail, the floor every later model has to clear.

Source experiments: [`01_baseline`](experiments/01_baseline/), [`02_arima`](experiments/02_arima/).

## Beats

1. **Hook.** Before claiming a deep-learning model "works on Bitcoin," show what the dumbest possible predictor reports on the same slice. On 366 test bars at 4h, the dumbest predictor's MAE is **540.96 USD**. Every model later in the series will sit within $11 of that number, in either direction.

2. **The target, the slice, the split.** A 1-paragraph methodology recap (the reusable one mentioned in articles_todo.md). Log-return target, 4h `BTCUSDT`, three slices (train 2023-01-01 → 2024-08-01, validation 2024-08-01 → 2024-10-01, test 2024-10-01 → 2024-12-01), walk-forward 1-step-ahead, weights frozen.

3. **The naive baseline.** `close_pred[t] = close[t-1]`. No fit, no parameters, no walk-forward in the model-fitting sense — the entire experiment is a one-pass slice through the test bars. The whole point is the architecture: data loader, config schema, metric module, `results/` layout — every later experiment reuses them unchanged. Show the prediction code (it's three lines).

4. **The metrics report `NaN` and that is a feature.** Directional accuracy is `NaN` by design — `sign(pred − ref) = sign(0) = 0` on every bar, the mask is empty, the function short-circuits. Cumulative return and Sharpe are omitted entirely from `metrics.json`, not zero — the model expressed no strategy. Show the relevant lines of [`src/btc_ai/eval/metrics.py:23-34`](src/btc_ai/eval/metrics.py). Three NaN-by-design surfaces in the leaderboard. Recognize them; don't mistake them for missing runs.

5. **The naive numbers (verbatim from `metrics.json`).** MAE 540.96 / RMSE 813.14 / MAPE 0.6922 %. dir_acc NaN. No cumulative return, no Sharpe.

6. **ARIMA in 90 seconds of plain English.** Auto-regressive (`p` lags of the differenced series) + Integrated (`d` differences before modelling) + Moving Average (`q` lags of residuals). On BTC, `d = 1` is non-optional — the price is non-stationary, the difference is stationary-ish. AR captures short-term momentum or mean-reversion; MA cleans up residual structure.

7. **The fit.** statsmodels' `ARIMA(p, d, q)`. Fit once on the train slice, then `fit.apply(full_series, refit=False)` so the coefficients freeze and the model sees new observations, then `predict(start=split, end=last, dynamic=False)` — the `dynamic=False` flag is load-bearing because it tells statsmodels each prediction at `t` uses the *actual* past values, not the model's own past predictions. That is the walk-forward shape.

8. **The sweep — 12 (p, d, q) orders.** Box-Jenkins ramp plus endpoints. Result: ARIMA(3, 1, 3) is the simultaneous winner of AIC, MAE, RMSE, MAPE, cumulative return, and Sharpe. The naive-equivalent (0, 1, 0) row produces MAE bit-identical to the naive baseline — pipeline sanity check passes. The un-differenced (1, 0, 1) row has Sharpe 0.5560 — confirming `d ≥ 1` is structural.

9. **The result — verbatim from `02_arima/results/btc_4h_2024/metrics.json`.** ARIMA(3, 1, 3) on 366 test bars: MAE 539.15, RMSE 808.79, MAPE 0.6903 %, dir_acc 0.5082, cum_ret 0.5269, annualized Sharpe 6.86. Beat naive on MAE by $1.81. Take Sharpe to 6.86.

10. **What this means about the model class.** Three takeaways:
    - The MAE gap between naive and a tuned linear AR / MA model is small enough to matter. *Three* parameters move the needle by $1.81. Any later model that does not match this gap is not earning its complexity.
    - AIC and Sharpe agree on this slice — both pick (3, 1, 3). On other slices they don't, and the global report has a separate disagreement to point to. Different selection criteria can rank differently.
    - Differencing is structural. The `(1, 0, 1)` row of the sweep collapses to Sharpe 0.56 — confirming that fitting price levels directly fails on BTC.

11. **Per-bar Sharpe significance check.** Per-bar Sharpe ≈ 6.86 / √2190 = 0.147. SE ≈ 1 / √366 = 0.052. Ratio ≈ **2.80 σ**. Significant at roughly the 99 % level, with the standard caveats about i.i.d. residual assumption being optimistic on 4h crypto.

12. **Regime caveat.** Test slice is post-election rally. Long-flat on an uptrending slice helps any model with mild directional skill. The 6.86 number is best read as "skill conditional on this regime."

13. **Reproduce.** `make 01_baseline` and `make 02_arima` plus `make 02_arima_sweep`.

## What this article is NOT

- An ARIMA tutorial. Refer the curious reader to Box & Jenkins, or to statsmodels.
- A claim that ARIMA is the right tool for Bitcoin trading. The series will show in later articles where it loses (XGBoost takes Sharpe; the LSTM gets directional accuracy higher).
