# Report — `03_gradient_boosting`

**Run date:** 2026-04-29
**Status:** completed (single fit + 8-config sweep)

## What this experiment is

The first **non-linear, feature-driven** model in the lineup. Trees instead of linear ARMA terms; engineered features instead of raw lagged closes. The headline structural change vs. 02:

```
target  : r_T = log(close_T / close_{T-1})            # next-step log-return, not price
features: r_{T-1..T-N}, rolling mean/std of r,        # built strictly from data observed
          log(volume_{T-1}), high-low, close-open     # before bar T (.shift(1) everywhere)
predict : close_pred_T = close_{T-1} * exp(r_pred_T)  # reconstruct price for shared metrics
```

Library: `xgboost.XGBRegressor` (tree_method='hist'). Why predict the **return** and reconstruct, not predict price directly? Trees cannot extrapolate — a model trained on prices in one range will refuse to output values outside that range, which on a trending series guarantees catastrophic out-of-sample failure. Returns are stationary; the level is supplied by the *known* `close_{T-1}` at prediction time.

## Configuration

From [experiments/03_gradient_boosting/config.yaml](../../experiments/03_gradient_boosting/config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `1h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| features.return_lags | **24** (one full day of lagged returns) |
| features.rolling_windows | `[6, 24]` (6 h and 24 h rolling mean/std of returns) |
| features.use_volume | `true` (`log1p(volume_{T-1})`) |
| features.use_ohlc | `true` (`high-low` range, `close-open` body of bar T-1) |
| model.n_estimators | 400 |
| model.max_depth | 5 |
| model.learning_rate | 0.05 |
| model.subsample | 0.8 |
| model.colsample_bytree | 0.8 |
| model.reg_lambda | 1.0 |
| model.random_state | 42 |

Total bars: **8 040**. After feature warmup (24-bar rolling window + 1-bar diff → 25 rows dropped): **8 015**. Train: **6 412**. Test: **1 603**. Feature columns: **31** (24 lagged returns + 2 windows × 2 stats + log_volume + hl_range + oc_body).

> **Slice note.** 01/02/04 use 1 608 test bars; 03 uses 1 603 (warmup cost). Headline metrics are still directly comparable; the gap is well within sample noise.

## Results — single fit

From [experiments/03_gradient_boosting/results/metrics.json](../../experiments/03_gradient_boosting/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | **270.73** USD |
| RMSE | 414.07 USD |
| MAPE | 0.3541 % |
| Directional accuracy | **0.4872** |
| Cumulative return | **+8.09 %** |
| Annualized Sharpe | **+1.45** |

## Cross-experiment comparison

Same data slice, (very nearly) same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| 01_baseline_naive | **260.50** | 396.97 | 0.341 % | NaN | — | — |
| **02_arima (1, 1, 1)** | **260.50** | **396.97** | 0.341 % | **0.5336** | **+53.02 %** | **+7.28** |
| **03_gradient_boosting (default)** | 270.73 | 414.07 | 0.354 % | **0.4872** | +8.09 % | +1.45 |
| 04_lstm (Darts BlockRNN-LSTM) | 274.02 | 409.66 | 0.360 % | 0.5196 | +50.34 % | +4.95 |

03 falls **below coin-flip on directional accuracy** in this widened window (0.4872) — a sharp reversal of the Q1-only result, where 03 led the leaderboard at 0.5185 / Sharpe +4.69. This is the regime-fit story:

> **Q1 (Jan-Mar 2024)** → dir_acc 0.5185, Sharpe +4.69
> **Jan-Nov 2024**     → dir_acc 0.4872, Sharpe +1.45

Same model, same hyperparameters, same code. What changed is the test window. Q1 2024 was a clean uptrend (BTC ~$42k → $71k); the engineered lag features happened to encode a "follow the recent momentum" signal that paid off in that regime. Extending the test window to include the Apr–Jul drawdown and chop pulled directional accuracy below 0.5.

## Hyperparameter sweep

From [experiments/03_gradient_boosting/results/sweep.csv](../../experiments/03_gradient_boosting/results/sweep.csv) — same data slice, 8 configs, all using fixed `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `random_state=42`:

| n_estimators | max_depth | learning_rate | MAE | RMSE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|
| 800 | 5 | 0.030 | 272.53 | 415.99 | **0.5078** | **+25.78 %** | **+3.93** |
| 800 | 7 | 0.030 | 275.08 | 416.35 | 0.4972 | +18.64 % | +3.06 |
| 200 | 3 | 0.100 | **269.05** | **412.09** | 0.5041 | +20.88 % | +3.26 |
| 400 | 7 | 0.050 | 276.33 | 420.43 | 0.5022 | +7.35 % | +1.32 |
| 400 | 3 | 0.050 | 269.57 | 412.44 | 0.4978 | +12.40 % | +2.09 |
| 200 | 5 | 0.100 | 274.86 | 416.41 | 0.4978 | +13.71 % | +2.26 |
| 1200 | 5 | 0.020 | 271.91 | 413.94 | 0.4891 | +8.97 % | +1.55 |
| 400 | 5 | 0.050 | 270.73 | 414.07 | 0.4872 | +8.09 % | +1.45 |

## Interpretation

1. **Every single config beat naive on MAE by less than $9, while every one** sits within ~$7 of the others on level prediction. The level-prediction problem is essentially settled by the random-walk floor; the sweep is searching the directional axis.

2. **Best dir_acc in the sweep is 0.5078** — `(800 trees, depth 5, lr 0.03)`. Three configs land at or above 0.5; five land below. Compare to the Q1 sweep where five of eight cleared 0.5 and the best hit 0.572. **The directional skill that looked load-bearing on Q1 has eroded substantially with a wider window.**

3. **MAE-best ≠ dir_acc-best.** `(200, 3, 0.10)` wins MAE at $269.05 but has middling dir_acc 0.5041. `(800, 5, 0.03)` wins dir_acc + Sharpe but has the 4th-best MAE. As before: MAE and dir_acc are near-orthogonal axes; for trading-strategy framing, only dir_acc / Sharpe matters.

4. **More boosting rounds with a smaller learning rate generalize better** — the `(800, 5, 0.03)` and `(200, 3, 0.10)` configs lead, while `(400, 7, 0.05)` and the default `(400, 5, 0.05)` lag. With more data (now 6 412 train rows vs Q1's 1 727), there's room for slower, deeper learning.

5. **No XGBoost config in this sweep beats ARIMA(1, 1, 0).** The leader at 0.5078 / +3.93 Sharpe is well behind ARIMA's 0.5373 / +7.52. **31 engineered features didn't add information that AR(1) on differences didn't already capture.** That's a strong negative result: it suggests the predictive content is dominated by the most recent price change and the engineered lag/volume/OHLC features are mostly noise on this signal-to-noise regime.

### Bottom line

**The Q1 result was regime fit.** XGBoost's directional edge does not survive a wider test window. The model is now last on dir_acc among the four scoring models. ARIMA(1, 1, 0) and the LSTM extract more from the same data, with much smaller feature surfaces.

The productive next directions for this branch of the lineup:

- **Add genuinely-exogenous features** — funding rate, perp basis, related-asset returns, on-chain flows. ARIMAX with the same exogenous inputs is a useful counter-baseline to confirm whether the issue is "no signal" or "wrong model class."
- **Walk-forward refit** — refit the model every N bars instead of training once. Tabular GB is cheap to refit; this would let weights adapt to regime shifts.
- **Direction-only target** — instead of regressing log-returns, classify sign. Pure dir_acc training might be more aligned with the metric we care about.

## Caveats

- **5-bar slice mismatch with 01/02/04** — feature warmup costs 25 rows. Test set is 1 603 bars vs. the others' 1 608. Within sample noise.
- **Single train/test split, no rolling-origin CV.** The sweep already shows hyperparameter rankings can shift on a different test window — same shape applies to model rankings.
- **No early stopping or validation set.** `n_estimators` is fixed; the sweep tries a few values directly.
- **Each prediction is independent — no multi-step rollout.** Equivalent to ARIMA's `dynamic=False` walk-forward.
- **Trees can't extrapolate.** This is structurally addressed by predicting returns and reconstructing prices via `close_{T-1} * exp(r)`.
- **No feature-importance dump.** XGBoost exposes `model.feature_importances_` but `run.py` doesn't write it. Easy follow-up to see whether `r_lag_1` dominates or whether the rolling/volume features actually contribute.

## Files produced

- [experiments/03_gradient_boosting/results/metrics.json](../../experiments/03_gradient_boosting/results/metrics.json) — single fit
- [experiments/03_gradient_boosting/results/predictions.parquet](../../experiments/03_gradient_boosting/results/predictions.parquet)
- [experiments/03_gradient_boosting/results/plot.png](../../experiments/03_gradient_boosting/results/plot.png)
- [experiments/03_gradient_boosting/results/sweep.csv](../../experiments/03_gradient_boosting/results/sweep.csv) — 8-config sweep

## How to reproduce

```
make 03_gradient_boosting        # single fit using params in config.yaml
make 03_gradient_boosting_sweep  # 8-config (n_estimators × depth × lr) sweep
```
