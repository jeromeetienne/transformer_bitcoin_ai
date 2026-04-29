# 04_lstm

First **deep learning** model in the lineup. Fits a [BlockRNN-LSTM](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.block_rnn_model.html) — via [darts](https://unit8co.github.io/darts/) — on 1h BTC log-returns with a small set of past covariates, then walk-forward 1-step-ahead forecasts on the held-out test slice.

```
BlockRNN(LSTM)
   ├── stacked LSTM cells       (carry state across input_chunk_length bars)
   ├── dropout between layers   (regularization)
   └── linear head              (single-step regression on r_T)
```

The target is the bar-T log-return `r_T = log(close_T / close_{T-1})` — the same stationary target used by [`03_gradient_boosting`](../03_gradient_boosting/) — and predictions are reconstructed to price as `close_pred = close_{T-1} * exp(r_pred)` so MAE/RMSE/MAPE/directional_accuracy/Sharpe are directly comparable to all earlier experiments.

## Why this experiment

Three reasons LSTM comes after gradient boosting in the progression:

1. **Sequence model, not a tabular one.** XGBoost sees an unordered bag of features per row; LSTM sees an *ordered* window of bars and carries hidden state across them. If the next-bar log-return depends on a *trajectory* (e.g. "three up-bars then a down-bar") rather than a snapshot, LSTM has the right inductive bias and trees do not.
2. **Past covariates as time series, not engineered scalars.** Where `03_gradient_boosting` summarised volume / OHLC into a single shifted scalar per row, the BlockRNN model ingests `log_volume`, `hl_range`, `oc_body` as *parallel time series* aligned with the target. The model decides what to do with the temporal pattern — no rolling-window choice required.
3. **Bridge to the transformer.** [`05_transformer`](../05_transformer/) is a step up in capacity *and* mechanism (attention, future covariates, variable selection). Running the same darts harness with a vanilla LSTM first isolates the contribution of attention itself: any gap between LSTM and TFT on the same data slice is attributable to the architecture, not the framework.

If LSTM can't beat the ARIMA(1,1,1) Sharpe leader from [`02_arima`](../02_arima/), the conclusion is the same as for the linear baselines: the hourly BTC log-return signal is too weak for added model capacity to find anything an AR(1) on differences couldn't.

## How it's done

1. Load OHLCV via the shared loader ([src/btc_ai/data](../../src/btc_ai/data)). Drop the timezone so darts' time indexing is unambiguous (matches [`02_arima`](../02_arima/)'s convention).
2. Build target `r_T = log(close_T / close_{T-1})` and `ref_T = close_{T-1}` for downstream price reconstruction.
3. Build **past covariates** (optional, controlled by `config.yaml`):
   - `log_volume = log1p(volume)`
   - `hl_range  = high − low`
   - `oc_body   = close − open`

   darts' `BlockRNNModel` only feeds past covariates strictly before each forecast step, so no current-bar leakage at evaluation time.
4. Three-way time split: `[0:train_end] [train_end:val_end] [val_end:n]` = `(train, val, test)`. The val slice is the tail of the would-be train, never overlapping test, used only for early stopping.
5. Fit a `darts.dataprocessing.transformers.Scaler` on the **training slice only** for target / past covariates separately. Cast to `float32` so the model can use Apple MPS or CUDA (MPS does not support `float64`).
6. Train `BlockRNNModel(model='LSTM', ...)` with `EarlyStopping(monitor='val_loss', patience=…)`. Architecture knobs (`input_chunk_length`, `hidden_dim`, `n_rnn_layers`, `dropout`, `batch_size`, `n_epochs`, `learning_rate`) all come from `config.yaml`.
7. Walk-forward predict on the test slice via `model.historical_forecasts(start=test_start, forecast_horizon=1, retrain=False, last_points_only=True)` — slides the input window across test, parameters frozen at train-time values.
8. Inverse-scale to `r_pred`, reconstruct prices `close_pred = ref * exp(r_pred)`, then compute MAE, RMSE, MAPE, directional accuracy, cumulative return, and annualized Sharpe via the shared metrics in [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py).
9. Write `results/metrics.json`, `results/predictions.parquet`, `results/plot.png`.

Library: **[darts](https://unit8co.github.io/darts/)** (`darts.models.BlockRNNModel`). Same harness used by [`05_transformer`](../05_transformer/) — only the model class differs — so the two runs are directly comparable on identical data.

## How to run

```
make 04_lstm
```

Try a different architecture by editing [`config.yaml`](config.yaml):

```yaml
model:
  input_chunk_length: 48      # 24 / 48 / 96 — bars of history fed to the LSTM
  hidden_dim: 32              # LSTM hidden state size (16 / 32 / 64)
  n_rnn_layers: 2             # stacked LSTM layers (1 / 2 / 3)
  dropout: 0.1                # dropout between layers (regularization)
  batch_size: 64
  n_epochs: 30                # max passes; early stopping may halt sooner
  learning_rate: 0.001        # Adam learning rate

training:
  val_fraction: 0.1           # tail fraction of *train* carved off for early stopping
  early_stopping_patience: 5  # epochs of no val_loss improvement before stopping
```

To run with target only (no past covariates) for an ablation against the recurrent component alone:

```yaml
covariates:
  use_volume: false
  use_ohlc: false
```

## How to interpret

Compare against earlier experiments **on the same data slice**:

| | Naive | MA(24) | ARIMA(1,1,1) | XGBoost | **LSTM** |
|---|---|---|---|---|---|
| Family | last-value | rolling mean | linear, stationary | tabular GBT | **recurrent NN** |
| Sees ordered history | — | window mean only | yes (linear) | engineered lags | **yes (learned)** |
| Past covariates | — | — | — | engineered scalars | **OHLCV time series** |
| Headline number | MAE floor | MAE ceiling | best Sharpe so far | first non-linear | *this experiment* |

What the numbers tell you:

- **`mae` close to XGBoost / naive** — the typical outcome on hourly BTC. Recurrent capacity over the same target doesn't conjure signal that isn't there. Don't be disappointed; this is the *truth* about hourly BTC log-returns.
- **`mae` clearly below XGBoost** — the LSTM is using the *order* of past bars in a way trees can't. Worth re-running with `use_volume: false` / `use_ohlc: false` to see how much of the gain is from the covariate channel vs. the recurrent target alone.
- **`directional_accuracy > 0.51`** — LSTM is finding *some* short-term directional signal even if MAE is flat. The strategy column in `predictions.parquet` translates that into per-bar P&L.
- **`annualized_sharpe` close to ARIMA's** — the deep model finally caught up on the metric that matters for trading. Usually requires the bigger configs (`hidden_dim ≥ 32`, `n_rnn_layers ≥ 2`).
- **`annualized_sharpe` below ARIMA's** — LSTM is overfitting val_loss while underfitting trading P&L. Try lowering capacity (`hidden_dim: 16`, `n_rnn_layers: 1`) and increasing `dropout`. Also lower `n_epochs` or `early_stopping_patience` to halt sooner.

### Caveats baked into this v1

- **Single deterministic fit.** `random_state` is fixed in `config.yaml`, but neural-net training has run-to-run variance from non-determinism in some MPS / CUDA kernels. A more rigorous comparison would average metrics over 3–5 seeds.
- **Walk-forward without re-estimation.** `retrain=False` keeps the train-time weights frozen across the entire test window. Re-fitting at every step would be slower and might capture regime drift, but is out of scope here.
- **`output_chunk_length: 1`.** One-step-ahead only. Multi-horizon LSTM is a one-line change but changes the loss surface.
- **No future covariates.** BlockRNN is a *past-covariate-only* model — calendar / cyclical time features that LSTM could legitimately consume at prediction time are out of reach here. [`05_transformer`](../05_transformer/) adds them.
- **Squared-error regression on log-returns.** No probabilistic head. darts supports quantile / Gaussian likelihoods on other model classes; not used here.

## Sweeping hyperparameters

For comparing multiple `(input_chunk_length, hidden_dim, n_rnn_layers, dropout)` combinations on the same data slice without re-running the full experiment each time:

```
make 04_lstm_sweep
```

Edit the `GRID` list at the top of [`sweep.py`](sweep.py) to change which configs are tried. Output: a printed table to stdout (sorted in source order, easy to scan) and a `results/sweep.csv` for downstream analysis. The sweep does not touch `metrics.json` / `predictions.parquet` / `plot.png` — those reflect the single config in [`config.yaml`](config.yaml) set via `make 04_lstm`.

What to look at in the sweep:

- **MAE on the same test slice** — only number that says "this config predicts better than that one." On hourly BTC the differences are usually within a few dollars on a multi-hundred-dollar baseline; treat sub-1% MAE deltas as noise.
- **`directional_accuracy` and `annualized_sharpe`** — the metrics that *trade*. They often rank configs differently from MAE because point error and direction-getting-right are not the same thing.
- **`input_chunk_length` matters more than `hidden_dim`.** For a near-random-walk like hourly BTC, doubling the lookback often changes nothing; doubling the hidden size mostly buys variance. Sweep `icl` first.
- **Bigger ≠ better.** With a few thousand training rows and a noisy target, the smaller configs (`icl=24`, `hidden_dim=16`, `n_rnn_layers=1`) are often more honest than the bigger ones.

### Optuna search (TPE + median pruning)

For an adaptive search over the same 4-D space — Tree-structured Parzen Estimator picks each next trial based on previous results, and `MedianPruner` kills unpromising trials mid-training:

```
make 04_lstm_optuna
```

The search is parameterised in the `optuna:` section of [`config.yaml`](config.yaml): `n_trials`, `timeout_seconds`, `objective` (any metric key returned by the run, e.g. `annualized_sharpe`), `direction` (`minimize` / `maximize`), `seed`, and a declarative `search_space` (`categorical` / `int` / `float`). No silent defaults — `objective` and `direction` must both be set explicitly.

Outputs (under `results/`):

- `optuna_study.db` — SQLite-backed Optuna storage. Resumable: rerunning `make 04_lstm_optuna` adds new trials to the same study (via `load_if_exists=True`).
- `optuna_trials.csv` — full `study.trials_dataframe()` snapshot for CLI inspection.
- `optuna_best.json` — best params, best objective value, count of completed vs pruned trials, storage URL.

To explore the study interactively (parallel coordinates, history, parameter importance):

```
make 04_lstm_optuna_dashboard
```

This launches [`optuna-dashboard`](https://optuna-dashboard.readthedocs.io/) against `results/optuna_study.db` on `http://127.0.0.1:8080`. Ctrl-C to stop. The dashboard reads the same SQLite file the sweep is writing to, so you can leave it running while a longer search is in progress and refresh.

The Optuna run **does not** touch `metrics.json` / `predictions.parquet` / `plot.png` / `sweep.csv` — they reflect the manual GRID and the single `model:` config, not the Optuna search.

## Files

```
experiments/04_lstm/
├── README.md           # this file
├── config.yaml         # data slice + test_fraction + covariates + LSTM model + training + optuna sections
├── run.py              # entry point — fits the single config in config.yaml; train_and_evaluate() is reused by sweep.py / optuna_sweep.py
├── sweep.py            # manual GRID sweep, writes results/sweep.csv
├── optuna_sweep.py     # Optuna TPE search with PyTorchLightning pruning, writes results/optuna_*
└── results/
    ├── metrics.json        # produced by run.py
    ├── predictions.parquet # close + pred + ref + strategy_return on the test slice
    ├── plot.png            # produced by run.py
    ├── sweep.csv           # produced by sweep.py
    ├── optuna_study.db     # produced by optuna_sweep.py (SQLite, read by optuna-dashboard)
    ├── optuna_trials.csv   # produced by optuna_sweep.py
    └── optuna_best.json    # produced by optuna_sweep.py
```
