# Report — `05_transformer`

**Run date:** 2026-04-29
**Status:** completed (single fit + 6-config sweep) — **on stale 1h slice; pending re-run on 4h**

> The `config.yaml` for this experiment now declares `interval: 4h` (matching commit 81d4fcb's project-wide kline switch), but the `metrics.json` and `sweep.csv` artefacts in this report were last regenerated on **2026-04-29** against the previous 1h slice. Total bars 8 039 / test 1 607 corresponds to 1h, not 4h (which would give ~2 010 / 402, as seen in 01–04). All numbers below are honest reports of the artefacts on disk; they are **not** comparable to the 4h leaderboard in 01–04's reports.

## What this experiment is

First **attention-based** model in the lineup. Fits a Darts Temporal Fusion Transformer with separate past- and future-covariate channels and the bar-T log-return as target. Predictions are reconstructed to price as `close_pred = close[t-1] * exp(r_pred)`. Differentiator vs. 04_lstm: attention over an explicit time window (can look at any past step directly instead of compressing it into a hidden state), plus a future-covariate channel — cyclical encodings of hour-of-day and day-of-week — that TFT can legitimately consume at prediction time because they are deterministic functions of the timestamp.

```
TFT
   ├── variable selection networks  (which inputs matter, per-step)
   ├── LSTM encoder (past) + decoder (future)
   ├── multi-head self-attention    (long-range dependencies)
   └── gated residual networks       (feature mixing & skips)
```

Library: **darts** (`darts.models.TFTModel`) on PyTorch Lightning. Walk-forward shape: `model.historical_forecasts(start=test_start, forecast_horizon=1, retrain=False, last_points_only=True)`. Same harness as [04_lstm](04_lstm.report.md) so the two are directly comparable when *both* are on the same slice (they currently are not). See [experiments/05_transformer/README.md](../../experiments/05_transformer/README.md) for the full narrative.

## Configuration

From [experiments/05_transformer/config.yaml](../../experiments/05_transformer/config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `4h` (config) — **1h on the stale artefacts** |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| covariates.use_volume | `true` |
| covariates.use_ohlc | `true` |
| covariates.use_time_features | `true` |
| model.input_chunk_length | `48` |
| model.output_chunk_length | `1` |
| model.hidden_size | `32` |
| model.lstm_layers | `1` |
| model.num_attention_heads | `4` |
| model.dropout | `0.1` |
| model.hidden_continuous_size | `8` |
| model.add_relative_index | `false` |
| model.full_attention | `false` |
| model.batch_size | `64` |
| model.n_epochs | `30` (early-stopped) |
| model.learning_rate | `0.001` (Adam) |
| model.random_state | `42` |
| training.val_fraction | `0.1` |
| training.early_stopping_patience | `5` |

Total bars on the stale 1h artefact: **8 039**. Train: **5 789**. Val: **643**. Test: **1 607**.

## Results — single fit (on stale 1h slice)

From [experiments/05_transformer/results/metrics.json](../../experiments/05_transformer/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | 373.70 USD |
| RMSE | 546.68 USD |
| MAPE | 0.4879 % |
| Directional accuracy | 0.5053 |
| Cumulative return | 0.2957 |
| Annualized Sharpe | 4.5864 |

The MAE 373.70 is **lower** than every 4h number reported in 01–04 — but only because it is computed over 1h bars on a different prediction horizon. It is not a TFT advantage; it is an interval artefact. The honest comparison is the 1h-only leaderboard below.

## Cross-experiment comparison

1h-slice leaderboard. Same data slice, same split, same metrics. Only experiments still on the stale 1h slice appear here.

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| **05_transformer** | 373.70 | 546.68 | 0.4879 % | **0.5053** | **0.2957** | **4.5864** |
| [06_pretrained (timesfm)](06_pretrained.report.md) | **268.89** | **405.14** | **0.3521 %** | 0.4677 | 0.1690 | 2.4444 |

On the 1h slice, 05_transformer leads the trading metrics (dir_acc, cum_ret, Sharpe) and 06_pretrained leads point error (MAE, RMSE, MAPE). The split is informative: a trained model with covariates squeezes more direction out of the slice than a zero-shot 200M-parameter generic prior does — and the zero-shot model returns a lower MAE because it commits less to a direction. (Experiments 01–04 are on the fresh 4h slice; their reports contain a separate leaderboard.)

## Sweep — 6 (input_chunk_length, hidden_size, num_attention_heads, lstm_layers, dropout) configs

From [experiments/05_transformer/results/sweep.csv](../../experiments/05_transformer/results/sweep.csv), still on the stale 1h slice. Sorted in source order; leaders bolded per column.

| icl | hidden | heads | layers | dropout | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 16 | 2 | 1 | 0.1 | 398.59 | 580.50 | 0.5232 % | 0.5028 | **0.3008** | 4.3677 |
| **48** | **32** | **4** | **1** | **0.1** | **373.70** | **546.68** | **0.4879 %** | 0.5053 | 0.2957 | 4.5864 |
| 48 | 32 | 4 | 2 | 0.2 | 420.61 | 605.08 | 0.5538 % | 0.4984 | 0.2898 | 4.1173 |
| 48 | 64 | 8 | 1 | 0.1 | 392.46 | 574.22 | 0.5127 % | **0.5109** | 0.3076 | **4.8331** |
| 96 | 32 | 4 | 1 | 0.1 | 409.96 | 601.69 | 0.5412 % | 0.4984 | 0.2294 | 3.4472 |
| 96 | 64 | 8 | 2 | 0.2 | 395.20 | 610.71 | 0.5119 % | 0.5047 | 0.3095 | 4.4670 |

The default config `(48, 32, 4, 1, 0.1)` leads on MAE / RMSE / MAPE; the wider config `(48, 64, 8, 1, 0.1)` leads on dir_acc and Sharpe. Larger `input_chunk_length` (96 bars) consistently underperforms — extra history does not help on a near-i.i.d. signal.

## Interpretation

1. **The 1h slice is what is on disk.** Total rows 8 039 corresponds to roughly 24h × 335 days ≈ 8 040 hourly bars; the 4h slice would have ~2 010. The `interval: 4h` in the current config has not been propagated through to the artefacts. Re-running with `make 05_transformer` will overwrite `metrics.json` and `sweep.csv` against the 4h slice and make the row comparable to 04_lstm.
2. **dir_acc 0.5053 is the median outcome for a TFT on near-random-walk data** — a tiny edge over coin-flip, just enough to produce a positive Sharpe of 4.5864. Per-bar Sharpe ≈ 4.5864 / √8760 = 0.0490; standard error ≈ 1 / √1607 = 0.0249; ratio ≈ **1.97 σ** — borderline significant at ~95 %. Not strong evidence of an edge.
3. **Sweep Sharpe winner is wider, not deeper.** `(48, 64, 8, 1, 0.1)` reaches Sharpe 4.8331 at the same `icl` as the default with twice the hidden size and twice the attention heads — but the same single LSTM layer. Adding a second `lstm_layers` (the `(48, 32, 4, 2, 0.2)` row) costs Sharpe.
4. **`input_chunk_length = 96` consistently underperforms in this sweep.** Both 96-row configs lose Sharpe relative to their 48-bar counterparts. Either the slice is too short for the model to learn cross-day structure, or there is no cross-day structure to learn — the second is the more probable explanation on hourly BTC.
5. **Where capacity may have hurt.** Doubling `lstm_layers` from 1 → 2 with `dropout` raised to 0.2 (`(48, 32, 4, 2, 0.2)` row) drops Sharpe by 0.47 vs. the default — even though more parameters and more regularization usually go together. A near-i.i.d. residual punishes extra recurrent depth.

### Bottom line

**On the stale 1h slice**, 05_transformer reports **MAE 373.70 / dir_acc 0.5053 / Sharpe 4.5864**. The Sharpe per-bar (0.0490) is 1.97 σ above zero relative to a 1 607-bar SE — borderline significant. Sweep Sharpe winner `(48, 64, 8, 1, 0.1)` at 4.8331 sits just above the default; the wins from extra width are modest and the wins from extra depth are negative. Test slice covers approximately Sep 25 → Dec 1 2024 (post-election BTC rally) regardless of which interval is used. **The load-bearing item is operational, not architectural**: the artefacts need to be regenerated on 4h before this row can sit in the same leaderboard as 02_arima / 03_gradient_boosting / 04_lstm.

## Caveats

- **Stale 1h slice.** Re-run `make 05_transformer` after confirming the config's `interval: 4h`, then re-run `make 05_transformer_sweep`.
- Single deterministic fit. `random_state` is fixed but neural-net training has run-to-run variance from MPS / CUDA non-determinism.
- Walk-forward without re-estimation. `retrain=False`.
- `output_chunk_length: 1`. One-step-ahead only.
- No probabilistic head. TFT supports quantile regression; not used here (foundation models in 06 do use it).
- Future covariates are calendar-only (`hour_sin / hour_cos / dow_sin / dow_cos`). No exogenous price-of-other-assets, on-chain features, or news.
- Long/flat strategy, no shorting, no transaction costs.
- Attention weights are not interpretation. A high attention weight on lag −24 does not mean lag −24 *caused* the prediction.

## Files produced

- [experiments/05_transformer/results/metrics.json](../../experiments/05_transformer/results/metrics.json) (**stale 1h**)
- [experiments/05_transformer/results/predictions.parquet](../../experiments/05_transformer/results/predictions.parquet) (**stale 1h**)
- [experiments/05_transformer/results/plot.png](../../experiments/05_transformer/results/plot.png) (**stale 1h**)
- [experiments/05_transformer/results/sweep.csv](../../experiments/05_transformer/results/sweep.csv) (**stale 1h**)

## How to reproduce

```
make 05_transformer
make 05_transformer_sweep
make 05_transformer_optuna       # adaptive TPE search; ignored by the headline numbers above
```
