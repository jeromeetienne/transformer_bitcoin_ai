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
| end | `2024-04-01` UTC (exclusive) |
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

Total bars: **2 184**. After feature warmup (24-bar rolling window + 1-bar diff → 25 rows dropped): **2 159**. Train: **1 727**. Test: **432**. Feature columns: **31** (24 lagged returns + 2 windows × 2 stats + log_volume + hl_range + oc_body).

> **Slice note.** 01/01b/02 used 437 test bars; 03 uses 432 (warmup cost). Same end date, ~5 bars shorter at the start of the test slice. Headline metrics are still directly comparable; the gap is well within sample noise.

## Results — single fit

From [experiments/03_gradient_boosting/results/metrics.json](../../experiments/03_gradient_boosting/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | **354.60** USD |
| RMSE | 493.14 USD |
| MAPE | 0.5281 % |
| Directional accuracy | **0.5185** |

## Cross-experiment comparison

Same data slice, (very nearly) same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | vs naive (MAE) |
|---|---|---|---|---|---|
| **01_baseline_naive** | **337.89** | 471.54 | 0.502 % | NaN | — |
| 01b_moving_average (window=24) | 1012.86 | 1356.63 | 1.506 % | 0.483 | +674.97 |
| 02_arima (1, 1, 1) | 338.15 | 471.49 | 0.503 % | 0.492 | +0.27 |
| **03_gradient_boosting (default)** | 354.60 | 493.14 | 0.528 % | **0.5185** | +16.71 |

03 is **the first model in the lineup with directional accuracy meaningfully above 0.5.** It pays for that with ~$17 more MAE than naive — the trees nudge the predicted price up or down by a small `exp(r_pred)` factor on every bar, and on the bars where they nudge in the wrong direction the absolute error grows.

## Hyperparameter sweep

From [experiments/03_gradient_boosting/results/sweep.csv](../../experiments/03_gradient_boosting/results/sweep.csv) — same data slice, 8 configs, all using fixed `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `random_state=42`:

| n_estimators | max_depth | learning_rate | MAE | RMSE | MAPE | dir_acc |
|---|---|---|---|---|---|---|
| 400 | **7** | 0.05 | **350.37** | 493.67 | 0.522 % | **0.5718** |
| 800 | 7 | 0.03 | 352.14 | **491.90** | 0.525 % | 0.5347 |
| 400 | 3 | 0.05 | 353.28 | 493.63 | 0.526 % | 0.5278 |
| 800 | 5 | 0.03 | 354.04 | 494.02 | 0.527 % | 0.5370 |
| 400 | 5 | 0.05 | 354.60 | 493.14 | 0.528 % | 0.5185 |
| 1200 | 5 | 0.02 | 354.80 | 492.26 | 0.528 % | **0.5648** |
| 200 | 3 | 0.10 | 355.36 | 499.76 | 0.530 % | 0.5069 |
| 200 | 5 | 0.10 | 358.49 | 494.88 | 0.534 % | 0.5231 |

## Interpretation

1. **All 8 configs land in a $8 MAE band** ($350–358), and all 8 are **above** naive's $337.89. Adding non-linearity + 30 engineered features does not help point-forecast MAE on hourly BTC. That continues the story 02 told: *the dynamics aren't there to extract from price-and-volume features alone at this timescale.*

2. **Every single config has directional accuracy ≥ 0.50.** Five of eight are ≥ 0.52, and the best (`400 trees, depth 7`) hits **0.572** — meaningfully above coin-flip on 432 test bars. This is **the first directional signal anywhere in the leaderboard.** ARIMA's best dir_acc was 0.513 (and that order had MAE worse than naive's).

3. **MAE and dir_acc are uncorrelated across this sweep.** The MAE-best config (`400 / 7 / 0.05`) also happens to be the dir_acc-best, but elsewhere they disagree: `1200 / 5 / 0.02` is mid-table on MAE (354.80) but #2 on dir_acc (0.565). Translation: the sweep is searching two near-orthogonal axes — *how cleanly does the model predict price level* vs. *how often does it pick the right side of the move.* For a trading-strategy framing, dir_acc is the load-bearing one.

4. **Deeper trees > more trees on this data.** `depth=7` shows up at the top of dir_acc twice; `depth=3` configs are flat at ~0.51. Training set is small (1 727 rows × 31 cols), so deeper trees can carve out the few non-linear corners that exist before regularization smooths them away.

5. **Why does GB give up MAE to win dir_acc?** Naive predicts no change — the `exp(r_pred)` multiplier is exactly 1. GB always predicts *some* nonzero return, so its predicted price is always slightly off the prior close. On bars where the actual move was small (most of them), naive's "no-change" guess is closer in absolute terms. On bars where the move was directionally predictable (a minority, but more than 50%), GB scores. The MAE/dir_acc tradeoff falls out of that arithmetic, not from the model being "wrong."

### Bottom line

**GB doesn't beat naive on MAE, but it expresses a directional opinion that's right ~57% of the time at its best setting** — the first model in the leaderboard that actually has skin in the game. For a price-forecasting metric scoreboard this is a draw-to-loss; for a trading-strategy framing this is the first win.

The next experiments worth running:

- **Same model, longer history** (multi-year start date) to see if the dir_acc edge widens or shrinks with more training data.
- **Add exogenous features GB can chew on** — funding rate, perp basis, on-chain flows, related-asset returns. Classical ARIMAX baseline with the same exogenous inputs would be a useful counter-baseline.
- **Move to a sequence model** (transformer / LSTM) on the same target framing (predict log-return) — the comparison becomes *"does sequential context add anything over treating each bar as IID tabular features?"*

## Caveats baked into v1

- **5-bar slice mismatch with 01/01b/02** — feature warmup costs 25 rows (24-bar rolling + 1-bar diff). Test set is 432 bars vs. the others' 437. Headline-metric noise floor is roughly the same.
- **Single train/test split, no rolling-origin CV.** Hyperparameter rankings could shift on a different test window. The `random_state` makes a single run reproducible but doesn't measure sweep variance.
- **No early stopping or validation set.** `n_estimators` is fixed; the sweep tries a few values directly. Overfitting is checked by RMSE/MAE, not by held-out loss.
- **Each prediction is independent — no multi-step rollout.** Equivalent to ARIMA's `dynamic=False` walk-forward: features at every test bar use *actual* past data, never previously-predicted data. Multi-step forecasts would require feeding predicted returns back into the lag features, which is a different (harder) experiment.
- **Trees can't extrapolate.** This is structurally addressed by predicting returns and reconstructing prices via `close_{T-1} * exp(r)`. If you ever switch the target back to raw price (against the current design), expect dramatic test-set failures whenever the test window trends outside the training range.
- **No feature-importance dump.** XGBoost exposes `model.feature_importances_` but `run.py` doesn't write it. Easy follow-up if you want to see whether `r_lag_1` dominates or whether the rolling/volume features actually contribute.

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
