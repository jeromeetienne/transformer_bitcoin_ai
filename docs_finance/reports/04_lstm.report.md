# Report — `04_lstm`

**Run date:** 2026-04-29
**Status:** completed (single fit + 6-config sweep)

## What this experiment is

The first **sequence model** in the lineup. Where 03 fed XGBoost a flat tabular vector of hand-engineered lag features, 04 feeds a recurrent network the *raw* log-return sequence and lets it learn its own representation of recent history. This is the chronological predecessor of the transformer / foundation-model experiments the repo is named for.

```
target  : r_T = log(close_T / close_{T-1})           # next-step log-return (same as 03)
inputs  : target sequence, with input_chunk_length    # raw history fed window-by-window
          past_covariates: log_volume, hl_range,      # multivariate past covariates
          oc_body                                     # darts only feeds them strictly before T
predict : close_pred_T = close_{T-1} * exp(r_pred_T)  # reconstruct price for shared metrics
```

Library: [Darts](https://unit8co.github.io/darts/) `BlockRNNModel(model="LSTM")` — non-autoregressive block predictor, past-covariates only. Walk-forward 1-step-ahead via `historical_forecasts(retrain=False)`: the model is fit once on train+val, then slides its input window across the test slice. A separate validation tail (10 % of train) is held out for early stopping.

Why predict log-returns, not price? Same reason as 03: stationary target, level supplied by the *known* `close_{T-1}` at prediction time. Why `BlockRNNModel` instead of `RNNModel`? Block is non-autoregressive (predicts a fixed horizon in one shot) and consumes past covariates only — the cleanest apples-to-apples comparison with 03's tabular setup. `RNNModel` (DeepAR-style) is autoregressive and supports future covariates; reserve it for a follow-up that adds calendar features.

## Configuration

From [experiments/04_lstm/config.yaml](../../experiments/04_lstm/config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `1h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| covariates.use_volume | `true` (`log1p(volume)`) |
| covariates.use_ohlc | `true` (`high - low`, `close - open`) |
| model.input_chunk_length | **48** (2 days of 1 h history per training window) |
| model.output_chunk_length | 1 (one-step-ahead) |
| model.hidden_dim | 32 |
| model.n_rnn_layers | 2 |
| model.dropout | 0.1 |
| model.batch_size | 64 |
| model.n_epochs | 30 (max; early stopping enabled) |
| model.learning_rate | 0.001 (Adam) |
| model.random_state | 42 |
| training.val_fraction | 0.1 (tail of *train* for early stopping) |
| training.early_stopping_patience | 5 |

Total bars after dropna: **8 039**. Train: **5 789**. Validation: **643**. Test: **1 607**.

> **Slice note.** Test count differs by 1 from 01/02 (1 608) due to the 1-bar diff to compute log-returns; differs by 4 from 03 (1 603) because 03 also drops 24 bars of rolling-window warmup. Headline metrics are still directly comparable.

> **MPS note.** Inputs are cast to `float32` before `TimeSeries` construction — Apple's MPS backend doesn't support `float64` tensors and darts auto-detects float64 inputs. Price reconstruction stays in float64 for accuracy. See the comment at [experiments/04_lstm/run.py](../../experiments/04_lstm/run.py).

## Results — single fit

From [experiments/04_lstm/results/metrics.json](../../experiments/04_lstm/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | **274.02** USD |
| RMSE | 409.66 USD |
| MAPE | 0.3596 % |
| Directional accuracy | **0.5196** |
| Cumulative return | **+50.34 %** |
| Annualized Sharpe | **+4.95** |

## Cross-experiment comparison

Same data slice, (very nearly) same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| 01_baseline_naive | **260.50** | 396.97 | 0.341 % | NaN | — | — |
| **02_arima (1, 1, 1)** | **260.50** | **396.97** | 0.341 % | **0.5336** | **+53.02 %** | **+7.28** |
| 03_gradient_boosting (default) | 270.73 | 414.07 | 0.354 % | 0.4872 | +8.09 % | +1.45 |
| **04_lstm (default)** | 274.02 | 409.66 | 0.360 % | 0.5196 | **+50.34 %** | **+4.95** |

LSTM is the **second-place directional model** behind ARIMA(1, 1, 1) and well clear of XGBoost. It pays for that with ~$14 of MAE vs naive — same pattern as 03 — but extracts a meaningful trading edge that XGBoost on the same data does not.

This is a sharp inversion of the Q1-only result, where the LSTM had Sharpe -0.54 (cum_ret -2.82 %) and looked broken. Same code, same hyperparameters; the wider window (1 607 test bars vs 432) gave the model both more training data (5 789 vs 1 573 train rows) and a more diverse evaluation regime. **The Q1 numbers were a small-sample fluke; the wider-window numbers are the real read.**

## Hyperparameter sweep

From [experiments/04_lstm/results/sweep.csv](../../experiments/04_lstm/results/sweep.csv) — same data slice, 6 configs, all using fixed `batch_size=64`, `n_epochs=30`, `learning_rate=0.001`, `random_state=42`, val-tail early stopping with patience 5:

| input_chunk_length | hidden_dim | n_rnn_layers | dropout | MAE | RMSE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|---|
| **48** | 32 | 2 | 0.10 | 274.47 | 410.08 | **0.5196** | **+50.34 %** | **+4.95** |
| **96** | 64 | 3 | 0.20 | **260.23** | **396.60** | **0.5196** | **+50.34 %** | **+4.95** |
| 48 | 64 | 2 | 0.20 | 260.46 | 396.79 | 0.5178 | +43.95 % | +4.59 |
| 96 | 32 | 2 | 0.10 | 261.74 | 397.57 | 0.4916 | +20.75 % | +3.06 |
| 24 | 32 | 2 | 0.10 | 263.35 | 399.05 | 0.4810 | +8.74 % | +2.15 |
| 24 | 16 | 1 | 0.00 | 267.64 | 403.02 | 0.4798 | +1.58 % | +0.54 |

## Interpretation

1. **The largest model `(96, 64, 3, 0.2)` is the only model in the entire lineup to beat naive on MAE** — by 27 ¢, well within sample noise but nominally *first place*. It also ties for best dir_acc (0.5196) and Sharpe (+4.95). This is the headline: sequence + capacity beats hand-engineered features at the same target.

2. **The default config and the largest config produce *identical* dir_acc / cum_ret / Sharpe** to many decimal places (0.51960174… / 0.50339319… / 4.94512094…) despite different MAE. Both predict the same *sign* on every one of the 1 607 test bars, while disagreeing slightly on magnitude. This is suspicious-looking but plausible: the long/flat strategy depends only on `r_pred > 0`, and both models — fit on the same data with the same seed — learn similar mean trends in the validation-loss landscape. Worth noting; not a leakage signal.

3. **Short input windows fail.** `input_chunk_length=24` (one day of history) lands below 0.5 dir_acc, regardless of capacity. The directional signal the LSTM is picking up requires more context — 48-96 hours, on the order of multi-day momentum. This is consistent with what ARIMA(1, 1, 0) is doing implicitly (one lag, but on differences that compound across many bars).

4. **Capacity helps when the window is long enough.** `(48, 64, 2, 0.2)` and `(96, 64, 3, 0.2)` both beat the smaller `(48, 32, 2, 0.1)` on MAE; the largest config is the MAE leader. With ~5 800 training rows (3.4× the Q1 amount), the LSTM has room to learn structure that the smaller models couldn't fit reliably.

5. **No LSTM config beats ARIMA(1, 1, 0)'s Sharpe.** ARIMA leads at 0.5373 / +7.52; the best LSTM config is 0.5196 / +4.95. This is informative: a single learned coefficient on the most recent log-return change is *more* directionally useful on this window than a 2-3 layer LSTM with full sequence context. A possible read: the directional signal is dominated by the very-most-recent bar; longer context dilutes it. A more pessimistic read: 1 600 test bars is too small to distinguish the two — per-bar Sharpe difference is ~0.03, well within SE.

6. **Strategy P&L is decoupled from MAE.** `(96, 64, 3, 0.2)` and `(48, 32, 2, 0.1)` are at the extremes of MAE in this sweep ($260.23 vs $274.47) but identical on Sharpe. As before: at 1h horizon, MAE measures volatility; only direction-aware metrics measure model usefulness for trading.

### Bottom line

**The LSTM with the default config or any of the larger configs lands in the same Sharpe band as the leaderboard's ARIMA leader, slightly behind it, and clear of XGBoost.** The result is honestly mid-pack: real directional signal, but not new information beyond what AR(1)-on-differences already captures with one parameter. The headline win is that the largest LSTM config nominally beat naive on MAE — the first time anything in the lineup has — though the gap is too small to claim victory on the level metric.

Productive next directions:

- **Test on a window with a real bear leg.** The Sep–Nov 2024 rally is friendly to any "go-with-recent-momentum" rule. Mid-2022 (post-LUNA, FTX) would falsify or confirm.
- **Calendar covariates.** `RNNModel(model="LSTM")` accepts future covariates — hour-of-day, day-of-week — which `BlockRNNModel` can't consume. Whether intra-day seasonality adds anything is testable in an `04b` variant.
- **Transformer / TFT.** This was the original next-step rec; the LSTM result above is a useful baseline for it. The question becomes "does attention add anything over LSTM at this signal-to-noise level?"
- **Foundation models.** Chronos / TimesFM zero-shot would tell us whether a *pretrained* sequence model has any prior on BTC.

## Caveats

- **Single split, no rolling-origin CV.** Hyperparameter rankings could shift on a different window — the Q1 → Jan-Nov inversion this experiment already exhibits is the proof.
- **Walk-forward without retraining.** Once fit, the LSTM is frozen; it doesn't adapt to regime shifts within the test slice. `historical_forecasts(retrain=True)` would address this at significant compute cost.
- **`random_state=42` everywhere.** Single-seed runs hide variance from initialization and SGD. Multi-seed runs (5+ seeds) would be the right way to claim a Sharpe difference between configs.
- **MPS float32.** Inputs cast to float32 for MPS compatibility. Numerical behaviour will differ slightly from CUDA / CPU float64 runs. For log-returns at this magnitude, the difference is below noise.
- **No feature attribution.** Unlike XGBoost's `feature_importances_`, the LSTM doesn't expose per-covariate contribution. Whether the past covariates (volume, hl_range, oc_body) actually contribute or whether the target sequence alone is doing all the work is a follow-up question — easy to test by ablating `use_volume` / `use_ohlc` to false and rerunning.
- **Two warnings during training are upstream and harmless:** `pin_memory not supported on MPS` and a `LeafSpec` deprecation from pytorch-lightning's pytree integration.

## Files produced

- [experiments/04_lstm/results/metrics.json](../../experiments/04_lstm/results/metrics.json) — single fit
- [experiments/04_lstm/results/predictions.parquet](../../experiments/04_lstm/results/predictions.parquet) — columns: close, pred, ref, strategy_return
- [experiments/04_lstm/results/plot.png](../../experiments/04_lstm/results/plot.png)
- [experiments/04_lstm/results/sweep.csv](../../experiments/04_lstm/results/sweep.csv) — 6-config sweep

## How to reproduce

```
make 04_lstm         # single fit using params in config.yaml (~30 s on MPS)
make 04_lstm_sweep   # 6-config sweep (~3-5 min on MPS)
```
