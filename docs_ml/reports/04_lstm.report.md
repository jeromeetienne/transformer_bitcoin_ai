# Report — `04_lstm`

**Run date:** 2026-05-19
**Status:** completed (single fit + 6-config sweep)

## What this experiment is

First **deep-learning** model in the lineup. Fits a Darts `BlockRNNModel` configured as a stacked-LSTM with past covariates (`log_volume`, OHLC range, OHLC body) and the bar-T log-return as target. Predictions are reconstructed to price as `close_pred = close[t-1] * exp(r_pred)` so MAE / RMSE / MAPE / dir_acc / Sharpe are directly comparable to 01 / 02 / 03. Differentiator vs. 03_gradient_boosting: a *sequence* model that sees an ordered window of bars and carries hidden state across them, rather than a flat feature vector.

```
BlockRNN(LSTM)
   ├── stacked LSTM cells       (carry state across input_chunk_length bars)
   ├── dropout between layers   (regularization)
   └── linear head              (single-step regression on r_T)
```

Library: **darts** (`darts.models.BlockRNNModel(model='LSTM')`) on PyTorch Lightning. Walk-forward shape: `model.historical_forecasts(start=test_start, forecast_horizon=1, retrain=False, last_points_only=True)` — slides the input window across the test slice with weights frozen at train-time values. Three-way time split (train / val / test) with `EarlyStopping(monitor='val_loss')`. See [experiments/04_lstm/README.md](../../experiments/04_lstm/README.md) for the full narrative.

## Configuration

From [experiments/04_lstm/configs/btc_4h_2024.config.yaml](../../experiments/04_lstm/configs/btc_4h_2024.config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `4h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| covariates.use_volume | `true` |
| covariates.use_ohlc | `true` |
| model.input_chunk_length | `48` (= 2 days at 4h) |
| model.output_chunk_length | `1` |
| model.hidden_dim | `32` |
| model.n_rnn_layers | `2` |
| model.dropout | `0.1` |
| model.batch_size | `64` |
| model.n_epochs | `30` (early-stopped) |
| model.learning_rate | `0.001` (Adam) |
| model.weight_decay | `0.0` |
| model.random_state | `42` |
| training.val_fraction | `0.1` |
| training.early_stopping_patience | `5` |

Total bars: **2 009**. Train: **1 448**. Val: **160**. Test: **401**.

## Results — single fit

From [experiments/04_lstm/results/btc_4h_2024/metrics.json](../../experiments/04_lstm/results/btc_4h_2024/metrics.json):

| Metric | Value |
|---|---|
| MAE | 522.29 USD |
| RMSE | 785.45 USD |
| MAPE | 0.6761 % |
| Directional accuracy | **0.4938** (below coin-flip — see Interpretation) |
| Cumulative return | 0.2596 |
| Annualized Sharpe | 4.2660 |

## Cross-experiment comparison

4h-slice leaderboard. Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| [01_baseline_naive](01_baseline_naive.report.md) | 518.36 | 784.09 | 0.6701 % | NaN | — | — |
| [02_arima (1, 1, 1)](02_arima.report.md) | **517.16** | **782.42** | **0.6686 %** | **0.5547** | 0.3314 | 6.0569 |
| [03_gradient_boosting (n_feat=31)](03_gradient_boosting.report.md) | 539.39 | 795.07 | 0.7008 % | 0.5365 | **0.5042** | **6.4233** |
| **04_lstm** | 522.29 | 785.45 | 0.6761 % | 0.4938 | 0.2596 | 4.2660 |

04_lstm leads no column. It ties naive on MAE within $4, loses on direction (0.4938 < 0.5), loses Sharpe to both ARIMA(1, 1, 1) and XGBoost. On this 4h slice the LSTM is the first model to add capacity *and* go backwards on the trading metrics relative to a 3-parameter linear baseline. (Experiments 05 and 06 currently report on a stale 1h slice — see [05_transformer.report.md](05_transformer.report.md) and [06_pretrained.report.md](06_pretrained.report.md).)

## Sweep — 6 (input_chunk_length, hidden_dim, n_rnn_layers, dropout) configs

From [experiments/04_lstm/results/btc_4h_2024/sweep.csv](../../experiments/04_lstm/results/btc_4h_2024/sweep.csv) on the same 4h data slice. Sorted in source order; leaders bolded per column.

| icl | hidden | layers | dropout | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 16 | 1 | 0.0 | 553.34 | 815.33 | 0.7215 % | 0.5137 | 0.4174 | **5.0295** |
| 24 | 32 | 2 | 0.1 | 519.66 | 782.69 | 0.6726 % | 0.5062 | 0.3422 | 4.3749 |
| 48 | 32 | 2 | 0.1 | 522.17 | 785.40 | 0.6759 % | 0.4913 | 0.2421 | 3.9947 |
| 48 | 64 | 2 | 0.2 | 543.97 | 806.02 | 0.7055 % | 0.4638 | 0.0000 | NaN |
| 96 | 32 | 2 | 0.1 | 518.49 | 780.38 | 0.6714 % | 0.5137 | 0.4253 | 4.5246 |
| 96 | 64 | 3 | 0.2 | **517.29** | **779.74** | **0.6697 %** | **0.5362** | **0.4956** | 5.0096 |

The smallest configuration in the sweep — `(icl=24, hidden=16, n_rnn_layers=1, dropout=0.0)` — wins on Sharpe (5.0295). The largest configuration — `(96, 64, 3, 0.2)` — wins on every other column (MAE, RMSE, MAPE, dir_acc, cum_ret). The configured default `(48, 32, 2, 0.1)` is in between on every metric and is **not** the Sharpe winner.

## Interpretation

1. **dir_acc 0.4938 — below coin-flip.** The default LSTM is wrong on direction more often than right on this slice. ARIMA(1, 1, 1) clears 0.55 and XGBoost clears 0.53 on the same slice; the LSTM does not. Reading: a recurrent network with this many degrees of freedom and only ~1 400 training rows is fitting noise in the val_loss objective while the directional signal that AR(1) captures cheaply gets washed out. The same shape as 03's MAE failure, on a different metric.
2. **Sharpe 4.2660 is the lowest among the trained 4h models.** Per-bar 4.2660 / √2190 = 0.0912; SE 1 / √401 = 0.0499; ratio ≈ **1.83 σ** — only weakly significant. The strategy is positive but not by much over noise.
3. **The (48, 64, 2, 0.2) sweep row returns NaN Sharpe.** The strategy went entirely flat (`pred ≤ ref` on every test bar), so `cum_ret = 0` and the strategy returns are all zero — `stddev = 0` → annualized Sharpe degenerates to NaN per [src/btc_ai/eval/metrics.py:51-59](../../src/btc_ai/eval/metrics.py). This is a useful failure mode to recognize: a "trained model that never goes long" looks identical to a missing run on the leaderboard.
4. **The default config is not the sweep winner on either MAE or Sharpe.** Sharpe leader is the *smallest* config `(24, 16, 1, 0.0)` at 5.03; the MAE / dir_acc / cum_ret leader is the *largest* `(96, 64, 3, 0.2)` at 517.29 / 0.5362 / 0.4956. The two leaders point in opposite directions on capacity — smaller is better for the trading metric, larger is better for fit. The configured default splits the difference and wins neither.
5. **Where adding capacity hurt.** Moving from `(96, 64, 3, 0.2)` (largest) backwards to `(48, 32, 2, 0.1)` (default) costs **dir_acc 0.045** and **Sharpe 1.01**, with MAE essentially unchanged. The deep model is more confident *in the wrong direction* in the default config than in either the smallest or largest sweep config.

### Bottom line

LSTM is **the first model in the lineup whose extra capacity does not pay back on the trading metrics.** Default-config dir_acc 0.4938 sits below coin-flip; Sharpe 4.2660 trails ARIMA's 6.0569 and XGBoost's 6.4233 on the same slice. Per-bar Sharpe sanity check: 0.0912 against an SE of 0.0499 — about 1.83 σ, weakly significant. Test slice covers approximately Sep 25 → Dec 1 2024 (post-election rally); the LSTM's failure to find direction in a clean uptrending regime is the load-bearing observation. The sweep suggests the smallest LSTM `(icl=24, hidden=16, layers=1, dropout=0.0)` would have given a higher Sharpe (5.03) — a real signal that on this signal-to-noise ratio, *less* recurrent capacity is more honest.

## Caveats

- Single deterministic fit. `random_state` is fixed but neural-net training has run-to-run variance from MPS / CUDA non-determinism. A more rigorous comparison would average metrics over 3–5 seeds.
- Walk-forward without re-estimation. `retrain=False` keeps train-time weights frozen across the entire test window.
- `output_chunk_length: 1`. One-step-ahead only; multi-horizon is a one-line change but changes the loss surface.
- No future covariates. `BlockRNN` is past-covariate-only; calendar / cyclical time features that an LSTM could legitimately consume at prediction time are out of reach here (TFT in 05 adds them).
- No probabilistic head. Squared-error regression on log-returns; no quantile / Gaussian likelihood.
- Long/flat strategy, no shorting, no transaction costs.

## Files produced

- [experiments/04_lstm/results/btc_4h_2024/metrics.json](../../experiments/04_lstm/results/btc_4h_2024/metrics.json)
- [experiments/04_lstm/results/btc_4h_2024/predictions.parquet](../../experiments/04_lstm/results/btc_4h_2024/predictions.parquet)
- [experiments/04_lstm/results/btc_4h_2024/plot.png](../../experiments/04_lstm/results/btc_4h_2024/plot.png)
- [experiments/04_lstm/results/btc_4h_2024/sweep.csv](../../experiments/04_lstm/results/btc_4h_2024/sweep.csv)

## How to reproduce

```
make 04_lstm
make 04_lstm_sweep
make 04_lstm_optuna             # adaptive TPE search; ignored by the headline numbers above
```
