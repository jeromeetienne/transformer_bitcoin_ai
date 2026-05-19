# Report — `05_transformer`

**Run date:** 2026-05-19
**Status:** completed (single fit on fresh 4h slice). Sweep present but **stale on 1h** — pending re-run via `make 05_transformer_sweep`.

## What this experiment is

First **attention-based** model in the lineup. Fits a Darts Temporal Fusion Transformer with separate past- and future-covariate channels and the bar-T log-return as target. Predictions are reconstructed to price as `close_pred = close[t-1] * exp(r_pred)`. Differentiator vs. [04_lstm](04_lstm.report.md): attention over an explicit time window (any past step can be addressed directly instead of compressed into a hidden state), plus a future-covariate channel — cyclical encodings of hour-of-day and day-of-week — that TFT can legitimately consume at prediction time because they are deterministic functions of the timestamp.

```
TFT
   ├── variable selection networks  (which inputs matter, per-step)
   ├── LSTM encoder (past) + decoder (future)
   ├── multi-head self-attention    (long-range dependencies)
   └── gated residual networks       (feature mixing & skips)
```

Library: **darts** (`darts.models.TFTModel`) on PyTorch Lightning. Walk-forward shape: `model.historical_forecasts(start=test_start, forecast_horizon=1, retrain=False, last_points_only=True)`. Same harness as [04_lstm](04_lstm.report.md); only the model class and the future-covariate channel differ. See [experiments/05_transformer/README.md](../../experiments/05_transformer/README.md) for the full narrative.

## Configuration

From [experiments/05_transformer/config.yaml](../../experiments/05_transformer/config.yaml):

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
| covariates.use_time_features | `true` |
| model.input_chunk_length | `48` (= 8 days at 4h) |
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

Total bars: **2 009**. Train: **1 448**. Val: **160**. Test: **401**.

## Results — single fit

From [experiments/05_transformer/results/metrics.json](../../experiments/05_transformer/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | 813.10 USD |
| RMSE | 1 090.78 USD |
| MAPE | 1.0786 % |
| Directional accuracy | **0.4913** (below coin-flip — see Interpretation) |
| Cumulative return | 0.4468 |
| Annualized Sharpe | 5.8675 |

## Cross-experiment comparison

4h-slice leaderboard. Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| [01_baseline_naive](01_baseline_naive.report.md) | 518.36 | 784.09 | 0.6701 % | NaN | — | — |
| [01b_moving_average (window=24)](01b_moving_average.report.md) | 1 905.27 | 2 580.61 | 2.4782 % | 0.4801 | 0.0511 | 1.3003 |
| [02_arima (1, 1, 1)](02_arima.report.md) | **517.16** | 782.42 | **0.6686 %** | **0.5547** | 0.3314 | 6.0569 |
| [03_gradient_boosting (n_feat=31)](03_gradient_boosting.report.md) | 539.39 | 795.07 | 0.7008 % | 0.5365 | 0.5042 | 6.4233 |
| [04_lstm](04_lstm.report.md) | 522.29 | 785.45 | 0.6761 % | 0.4938 | 0.2596 | 4.2660 |
| **05_transformer** | 813.10 | 1 090.78 | 1.0786 % | 0.4913 | 0.4468 | 5.8675 |
| [06_pretrained (chronos-2)](06_pretrained.report.md) | 519.78 | **780.99** | 0.6721 % | 0.5323 | **0.6975** | **7.5915** |

05_transformer has **the worst MAE in the 4h leaderboard** at 813.10 — **57 % worse than naive's 518.36** and **51 % worse than the next-worst trained model** (03_gradient_boosting at 539.39). dir_acc 0.4913 is below coin-flip and essentially tied with LSTM's 0.4938. Sharpe 5.8675 is decent but trails 06_pretrained's 7.5915 and the ARIMA / XGBoost cluster around 6.0–6.4. **The TFT's extra architectural capacity does not pay off on the 4h training set of 1 448 rows.**

## Sweep — stale 1h artifact

> **Stale-slice warning.** [experiments/05_transformer/results/sweep.csv](../../experiments/05_transformer/results/sweep.csv) was last regenerated on 2026-04-29 against the previous 1h slice (5 789 training rows, 1 607 test rows). The single-fit `metrics.json` above is fresh on 4h. The sweep MAE figures (range 373–420 USD) are on the 1h scale and **not directly comparable** to the headline numbers above. Treat the sweep as a historical hyperparameter-ranking reference until re-run on 4h with `make 05_transformer_sweep`.

## Interpretation

1. **Worst MAE in the 4h leaderboard.** MAE 813.10 is 57 % above naive's 518.36 — no other model in the 4h leaderboard is wrong by this much on average. The TFT is underfit on point precision: the 1 448 training rows are not enough for ~30 k parameters across variable-selection networks, gated residual networks, an LSTM encoder/decoder, and multi-head attention to converge on a stable point estimate.
2. **dir_acc 0.4913 is below coin-flip** — and almost identical to 04_lstm's 0.4938. Adding attention, future covariates (calendar cyclical), variable selection, and gated residuals over an LSTM-only baseline does **not** pull more direction out of this signal at this training-set size. The TFT's extra machinery is exercised but does not earn its keep.
3. **Sharpe 5.8675 is decent despite the poor MAE.** Per-bar Sharpe = 5.8675 / √2190 = 0.1254; SE = 1 / √401 = 0.0499; ratio ≈ **2.51 σ** — borderline significant. The TFT is wrong-about-magnitude but expresses enough strong-positive forecasts that the long/flat rule (`pred > ref → long`) keeps the strategy long during the rally. Strategy aggressiveness rescues Sharpe even as MAE collapses.
4. **The 4h re-run dramatically changes the picture from the prior 1h artifact.** On 1h with 5 789 training rows, the same architecture reported MAE 373.70 (comparable to naive on the 1h scale) and Sharpe 4.5864. On 4h with 1 448 training rows, MAE jumps to 813.10 (much worse than naive) but Sharpe rises to 5.8675. The TFT is now *wrong-by-more* per bar but *more directional* — an artifact of the smaller training slice and the strong-trending test regime.
5. **Sweep is stale on 1h and cannot rank current 4h configs.** The 1h sweep's `(48, 64, 8, 1, 0.1)` was the Sharpe winner at 4.83. Whether that ranking survives the 4h re-run is unknown until `make 05_transformer_sweep` runs.

### Bottom line

TFT on 4h reports **MAE 813.10 (worst in the 4h leaderboard) / dir_acc 0.4913 (below coin-flip) / Sharpe 5.8675**. Per-bar Sharpe 0.1254 against SE 0.0499 → **2.51 σ**, borderline significant. Test slice covers approximately Sep 25 → Dec 1 2024 (post-election BTC rally). The load-bearing observation: **the TFT needs more training data than the 4h slice supplies** — its extra capacity over the LSTM does not buy meaningful skill at 1 448 rows, and it pays a large MAE penalty for the parameters that *are* exercised.

## Caveats

- Single deterministic fit. `random_state` is fixed but neural-net training has run-to-run variance from MPS / CUDA non-determinism.
- Walk-forward without re-estimation. `retrain=False` keeps train-time weights frozen across the entire test window.
- `output_chunk_length: 1`. One-step-ahead only.
- No probabilistic head. TFT supports quantile regression; not used here (the foundation models in [06_pretrained](06_pretrained.report.md) do use it).
- Future covariates are calendar-only (`hour_sin / hour_cos / dow_sin / dow_cos`). No exogenous price-of-other-assets, on-chain features, or news.
- Sweep on stale 1h slice; do not read it as ranking 4h configs.
- Long/flat strategy, no shorting, no transaction costs.
- Attention weights are not interpretation. A high attention weight on lag −24 does not mean lag −24 *caused* the prediction.

## Files produced

- [experiments/05_transformer/results/metrics.json](../../experiments/05_transformer/results/metrics.json) (fresh 4h)
- [experiments/05_transformer/results/predictions.parquet](../../experiments/05_transformer/results/predictions.parquet) (fresh 4h)
- [experiments/05_transformer/results/plot.png](../../experiments/05_transformer/results/plot.png) (fresh 4h)
- [experiments/05_transformer/results/sweep.csv](../../experiments/05_transformer/results/sweep.csv) (**stale 1h**)

## How to reproduce

```
make 05_transformer
make 05_transformer_sweep
make 05_transformer_optuna       # adaptive TPE search; ignored by the headline numbers above
```
