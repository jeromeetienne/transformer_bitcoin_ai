# 05_transformer

First **attention-based** model in the lineup. Fits a [Temporal Fusion Transformer (TFT)](https://arxiv.org/abs/1912.09363) — via [darts](https://unit8co.github.io/darts/) — on 1h BTC log-returns, with separate **past** and **future** covariate channels, then walk-forward 1-step forecasts on the held-out test slice.

```
TFT
   ├── variable selection networks  (which inputs matter, per-step)
   ├── LSTM encoder (past) + decoder (future)
   ├── multi-head self-attention    (long-range dependencies)
   └── gated residual networks       (feature mixing & skips)
```

The target is the bar-T log-return `r_T = log(close_T / close_{T-1})` — the same stationary target used by [`04_lstm`](../04_lstm/) — and predictions are reconstructed to price as `close_pred = close_{T-1} * exp(r_pred)` so MAE/RMSE/MAPE/directional_accuracy/Sharpe are directly comparable to all earlier experiments.

## Why this experiment

Three reasons TFT comes after LSTM in the progression:

1. **Long-range structure.** LSTM compresses history into a single hidden state and re-reads it sequentially. Attention can look at any past step directly, so if BTC has any non-trivial dependence beyond a few hours (intraday seasonality, day-of-week effects), TFT has a better shot at finding it.
2. **Future covariates as a first-class input.** TFT cleanly separates *past* covariates (only known up to `t-1`: log-volume, OHLC range/body) from *future* covariates (deterministic functions of time: hour-of-day, day-of-week). LSTM in [`04_lstm`](../04_lstm/) had no way to consume future-known signal — TFT does.
3. **Interpretability hooks.** TFT exposes per-feature variable-selection weights and per-head attention maps. Not used yet in this v1 run, but this is the first model in the sequence where post-hoc "what did it look at" analysis is meaningful.

If TFT can't beat the LSTM Sharpe (and especially the ARIMA(1,1,1) Sharpe leader from [`02_arima`](../02_arima/)), the conclusion is the same as for LSTM: the hourly BTC log-return signal is too weak for added model capacity to find anything an AR(1) on differences couldn't.

## How it's done

1. Load OHLCV via the shared loader ([src/btc_ai/data](../../src/btc_ai/data)).
2. Build target `r_T = log(close_T / close_{T-1})` and `ref_T = close_{T-1}` for downstream price reconstruction.
3. Build **past covariates** (optional, controlled by `config.yaml`):
   - `log_volume = log1p(volume)`
   - `hl_range  = high − low`
   - `oc_body   = close − open`
4. Build **future covariates** — cyclical encodings of the timestamp itself:
   - `hour_sin / hour_cos` — `sin/cos(2π · hour-of-day / 24)`
   - `dow_sin  / dow_cos`  — `sin/cos(2π · day-of-week / 7)`

   These are deterministic functions of any future bar's index, so TFT may legitimately consume them at prediction time.
5. Three-way time split: `[0:train_end] [train_end:val_end] [val_end:n]` = `(train, val, test)`. The val slice is the tail of the would-be train, never overlapping test, used only for early stopping.
6. Fit a `darts.dataprocessing.transformers.Scaler` on the **training slice only** for target / past / future covariates separately. Cast to `float32` so the model can use Apple MPS or CUDA.
7. Train `TFTModel` with `EarlyStopping(monitor='val_loss', patience=…)` on `val_loss`. Architecture knobs (`hidden_size`, `lstm_layers`, `num_attention_heads`, `dropout`, `hidden_continuous_size`) all come from `config.yaml`.
8. Walk-forward predict on the test slice via `model.historical_forecasts(start=test_start, forecast_horizon=1, retrain=False, last_points_only=True)` — slides the input window across test, parameters frozen at train-time values.
9. Inverse-scale to `r_pred`, reconstruct prices `close_pred = ref * exp(r_pred)`, then compute MAE, RMSE, MAPE, directional accuracy, cumulative return, and annualized Sharpe via the shared metrics in [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py).
10. Write `results/metrics.json`, `results/predictions.parquet`, `results/plot.png`.

Library: **[darts](https://unit8co.github.io/darts/)** (`darts.models.TFTModel`). Same harness as [`04_lstm`](../04_lstm/) — only the model class and the future-covariate channel differ — so the two runs are directly comparable on identical data.

## How to run

```
make 05_transformer
```

Try a different architecture by editing [`config.yaml`](config.yaml):

```yaml
model:
  input_chunk_length: 48        # 24 / 48 / 96 — bars of history fed to the encoder
  hidden_size: 32               # transformer width (16 / 32 / 64)
  lstm_layers: 1                # depth of TFT's encoder/decoder LSTM stack
  num_attention_heads: 4        # multi-head attention; must divide hidden_size
  dropout: 0.1
  hidden_continuous_size: 8     # GRN width for continuous covariates
  add_relative_index: false     # if you disable use_time_features, set this true
  full_attention: false         # darts' interpretable masked attention vs. full
```

To turn off the cyclical time features (e.g. for an ablation against [`04_lstm`](../04_lstm/)):

```yaml
covariates:
  use_time_features: false      # ⇒ then model.add_relative_index must be true
model:
  add_relative_index: true      # darts auto-generates a relative position index
```

The runner raises early if both are off — TFT cannot run without future covariates of some kind.

## How to interpret

Compare against earlier experiments **on the same data slice**:

| | Naive | MA(24) | ARIMA(1,1,1) | XGBoost | LSTM | **TFT** |
|---|---|---|---|---|---|---|
| Family | last-value | rolling mean | linear, stationary | tabular GBT | recurrent NN | **attention NN** |
| Past covariates | — | — | — | engineered lags | OHLCV | OHLCV |
| Future covariates | — | — | — | — | — | **hour/dow cyclical** |
| Headline number | MAE floor | MAE ceiling | best Sharpe so far | — | beat naive on MAE | *this experiment* |

What the numbers tell you:

- **`mae` similar to LSTM / naive** — the typical outcome on hourly BTC. Adding attention over the same target doesn't conjure signal that isn't there. Don't be disappointed; this is the *truth* about hourly BTC log-returns.
- **`mae` clearly below LSTM** — TFT's variable-selection + attention + future covariates are pulling something out that LSTM missed. Worth investigating which heads/inputs are doing the work (re-fit and inspect `model.predict(num_samples=1).explain()`-style hooks).
- **`directional_accuracy > 0.51`** — TFT is finding *some* short-term directional signal. The strategy column in `predictions.parquet` will translate that into per-bar P&L.
- **`annualized_sharpe` close to ARIMA's** — the deep model finally caught up on the metric that matters for trading. Usually requires the bigger configs (icl=96, hidden=64, heads=8).
- **`annualized_sharpe` below LSTM** — TFT may be overfitting val_loss while underfitting trading P&L. Try lowering capacity (`hidden_size: 16`, `lstm_layers: 1`) and increasing dropout.

### Caveats baked into this v1

- **Future covariates are calendar-only.** No exogenous price-of-other-assets, on-chain features, or news/sentiment yet. The TFT machinery is here; the inputs are still mostly the same five OHLCV columns.
- **Single deterministic fit.** `random_state` is fixed in `config.yaml`, but neural-net training has run-to-run variance from non-determinism in some MPS / CUDA kernels. A more rigorous comparison would average metrics over 3–5 seeds.
- **Walk-forward without re-estimation.** `retrain=False` keeps the train-time weights frozen across the entire test window. Re-fitting at every step would be slower and might capture regime drift, but is out of scope here.
- **`output_chunk_length: 1`.** One-step-ahead only. Multi-horizon TFT is a one-line change but changes the loss surface.
- **No probabilistic head.** This run uses a deterministic regression loss. TFT supports quantile regression (`likelihood=QuantileRegression`) and would give calibrated 50/90% intervals; not used here.
- **Attention is not interpretation.** A high attention weight on lag −24 doesn't mean lag −24 *causes* the prediction. Explain the model with care.

## Sweeping hyperparameters

For comparing multiple `(input_chunk_length, hidden_size, num_attention_heads, lstm_layers, dropout)` combinations on the same data slice without re-running the full experiment each time:

```
make 05_transformer_sweep
```

Edit the `GRID` list at the top of [`sweep.py`](sweep.py) to change which configs are tried. Output: a printed table to stdout (sorted in source order, easy to scan) and a `results/sweep.csv` for downstream analysis. The sweep does not touch `metrics.json` / `predictions.parquet` / `plot.png` — those reflect the single config in [`config.yaml`](config.yaml) set via `make 05_transformer`.

What to look at in the sweep:

- **MAE on the same test slice** — only number that says "this config predicts better than that one." On hourly BTC the differences are usually within a few dollars on a ~$370 baseline; treat sub-1% MAE deltas as noise.
- **`directional_accuracy` and `annualized_sharpe`** — the metrics that *trade*. They often rank configs differently from MAE because point error and direction-getting-right are not the same thing.
- **`hidden_size` × `num_attention_heads`** — `hidden_size` must be divisible by `num_attention_heads`. The grid respects this; if you add new rows, double-check.
- **Bigger ≠ better.** TFT has many moving parts; on a 5.7k-sample training set the smaller configs (icl=24, hidden=16) are often more honest than the bigger ones.

## Files

```
experiments/05_transformer/
├── README.md           # this file
├── config.yaml         # data slice + test_fraction + covariates + TFT model knobs
├── run.py              # entry point — fits the single config in config.yaml
├── sweep.py            # sweeps multiple TFT configs, writes results/sweep.csv
└── results/
    ├── metrics.json        # produced by run.py
    ├── predictions.parquet # close + pred + ref + strategy_return on the test slice
    ├── plot.png            # produced by run.py
    └── sweep.csv           # produced by sweep.py
```
