# 03_gradient_boosting

First **non-linear** model in the lineup. Fits an [XGBoost regressor](https://xgboost.readthedocs.io/) on hand-engineered features of past 1h BTC data, with the **bar-T log-return** as target. Walk-forward 1-step-ahead forecasts on the held-out test slice.

```
XGBoost(reg:squarederror)
   ├── lagged log-returns        r_{T-1}, r_{T-2}, ..., r_{T-N}
   ├── rolling mean / std        on past log-returns (windows = 6, 24)
   └── (optional) OHLCV summaries  log_volume_{T-1}, hl_range_{T-1}, oc_body_{T-1}
```

The target is the bar-T log-return `r_T = log(close_T / close_{T-1})` — a stationary quantity, unlike the price level itself — and predictions are reconstructed to price as `close_pred = close_{T-1} * exp(r_pred)` so MAE/RMSE/MAPE/directional_accuracy are directly comparable to [`01_baseline`](../01_baseline/) / [`02_arima`](../02_arima/). A simple long/flat strategy on the predicted direction also yields cumulative return + annualized Sharpe.

## Why this experiment

Three reasons gradient boosting comes after ARIMA in the progression:

1. **First non-linear estimator.** ARIMA is a linear model of price increments; if there's any short-term *interaction* between, say, "the last hour was an unusually large up-move" *and* "volume was below recent average," ARIMA cannot represent it. Trees can. If GBT can't beat ARIMA, the residual signal in hourly BTC is either purely linear or pure noise.
2. **First model to ingest covariates.** Previous baselines only saw past closes. XGBoost gets lagged log-returns, rolling moments of those returns, and (optionally) a per-bar volume / OHLC summary. This is also the first place in the lineup where feature engineering is the lever — model architecture is fixed, what you feed it isn't.
3. **A tabular, low-variance benchmark for the deep models that follow.** [`04_lstm`](../04_lstm/) and [`05_transformer`](../05_transformer/) need to clear this bar to justify their training cost. GBT is the right comparator: same features could go to either, but trees train in seconds and rarely overfit catastrophically.

If GBT can't beat the ARIMA(1,1,1) Sharpe leader, the conclusion is the same as for the linear baselines: the hourly BTC log-return signal is too weak for engineered features alone to find anything useful.

## How it's done

1. Load OHLCV via the shared loader ([src/btc_ai/data](../../src/btc_ai/data)).
2. Build target `r_T = log(close_T / close_{T-1})` and `ref_T = close_{T-1}` for downstream price reconstruction.
3. Build the feature matrix in [`features.py`](features.py) — every column is `.shift(1)` over its source so a row at `T` sees only data observed strictly before `T` (no current-bar leakage):
   - `r_lag_1 ... r_lag_N` — last `N` log-returns (`return_lags`)
   - `r_mean_w / r_std_w` — rolling mean / std of past log-returns for each `w` in `rolling_windows`
   - `log_volume` (optional) — `log1p(volume_{T-1})`
   - `hl_range / oc_body` (optional) — `(high − low)_{T-1}` and `(close − open)_{T-1}`
4. Time-split: hold out the last `test_fraction` (default 20%) as the test set. No random shuffle.
5. Fit `xgb.XGBRegressor(...)` on the training slice — `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda`, and `random_state` all come from `config.yaml`.
6. Predict on the test slice in one shot. Because every feature already encodes only past information, a single `model.predict(X_test)` call *is* a walk-forward of 1-step-ahead forecasts — no per-step refit needed.
7. Reconstruct prices `close_pred = ref * exp(r_pred)`, then compute MAE, RMSE, MAPE, directional accuracy, cumulative return, and annualized Sharpe via the shared metrics in [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py).
8. Write `results/metrics.json`, `results/predictions.parquet`, `results/plot.png`.

Library: **[xgboost](https://xgboost.readthedocs.io/)** (`xgb.XGBRegressor` with `tree_method='hist'`). Chosen for speed on tabular features and explicit hyperparameter control. A LightGBM swap would be a one-line change.

## How to run

```
make 03_gradient_boosting
```

Try a different model / feature set by editing [`config.yaml`](config.yaml):

```yaml
features:
  return_lags: 24            # 12 / 24 / 48 — number of lagged log-returns
  rolling_windows: [6, 24]   # add/remove windows; each adds mean + std cols
  use_volume: true
  use_ohlc: true

model:
  n_estimators: 400          # boosting rounds
  max_depth: 5               # 3-8 typical for tabular GB
  learning_rate: 0.05        # smaller → more trees, better generalization
  subsample: 0.8             # row sub-sampling per tree (stochastic GB)
  colsample_bytree: 0.8      # column sub-sampling per tree
  reg_lambda: 1.0            # L2 regularization on leaf weights
```

## How to interpret

Compare against earlier experiments **on the same data slice**:

| | Naive | ARIMA(1,1,1) | **XGBoost** |
|---|---|---|---|
| Family | last-value | linear, stationary | **tabular GBT** |
| What it sees | `close_{T-1}` | past closes only | **engineered lags + rolling moments + OHLCV** |
| Non-linear | — | — | **yes** |
| Headline number | MAE floor | best Sharpe so far | *this experiment* |

What the numbers tell you:

- **`mae` close to naive's `mae`** — the typical outcome on hourly BTC. Trees over the same lagged returns can't conjure signal that isn't there. Don't be disappointed; this is the *truth* about hourly BTC log-returns.
- **`mae` clearly below naive** — the engineered features are pulling something out that ARIMA missed. Worth checking which features XGBoost actually used (`model.feature_importances_`, not currently dumped in this v1).
- **`directional_accuracy > 0.51`** — GBT is finding *some* short-term directional signal even if MAE is flat. The strategy column in `predictions.parquet` translates that into per-bar P&L.
- **`annualized_sharpe` close to ARIMA's** — the non-linear model finally caught up on the metric that matters for trading. Usually requires the bigger configs (`n_estimators ≥ 800`, `max_depth ≥ 5`) or richer features.
- **`annualized_sharpe` below ARIMA's** — XGBoost is fitting noise in the residuals while losing the linear momentum signal AR(1) captured cheaply. Try more regularization (`reg_lambda` ↑, `max_depth` ↓) or fewer features.

### Caveats baked into this v1

- **Single deterministic fit.** `random_state` is fixed in `config.yaml`, but tree-building has run-to-run variance from `subsample` / `colsample_bytree`. A more rigorous comparison would average metrics over 3–5 seeds.
- **Walk-forward without re-estimation.** The model is fit once on the training slice; the entire test window is predicted with frozen trees. Re-fitting at every step would be slower and might capture regime drift, but is out of scope here.
- **No early stopping.** `n_estimators` is taken at face value. A held-out validation tail with `early_stopping_rounds` would prevent overfitting on the larger configs but adds another knob.
- **Features are minimal.** Lagged log-returns + rolling moments + a per-bar OHLCV summary. No exogenous price-of-other-assets, on-chain features, or news/sentiment. The XGBoost machinery is here; the inputs are still mostly the same five OHLCV columns.
- **Price reconstruction loses calibration on extreme bars.** `close_pred = ref * exp(r_pred)` is exact in expectation only when `r_pred` is the median of the conditional distribution. Squared-error trees fit the mean; on heavy-tailed log-returns that biases the price MAE slightly. A quantile loss (`reg:quantileerror`) would be more honest but is not used here.

## Sweeping hyperparameters

For comparing multiple `(n_estimators, max_depth, learning_rate)` combinations on the same data slice without re-running the full experiment each time:

```
make 03_gradient_boosting_sweep
```

Edit the `GRID` list at the top of [`sweep.py`](sweep.py) to change which configs are tried. Output: a printed table to stdout (sorted in source order, easy to scan) and a `results/sweep.csv` for downstream analysis. The sweep does not touch `metrics.json` / `predictions.parquet` / `plot.png` — those reflect the single config in [`config.yaml`](config.yaml) set via `make 03_gradient_boosting`.

What to look at in the sweep:

- **MAE on the same test slice** — only number that says "this config predicts better than that one." On hourly BTC the differences are usually within a few dollars on a multi-hundred-dollar baseline; treat sub-1% MAE deltas as noise.
- **`directional_accuracy` and `annualized_sharpe`** — the metrics that *trade*. They often rank configs differently from MAE because point error and direction-getting-right are not the same thing.
- **Depth × learning rate × tree count** — these three trade off against each other. A common pattern: `(800 trees, depth 5, lr 0.03)` and `(400 trees, depth 5, lr 0.05)` reach the same MAE; the bigger one is just slower.
- **Bigger ≠ better.** On a short training set, deep trees with many rounds memorise training noise. The smaller configs (`max_depth=3`, `n_estimators ≤ 400`) are often more honest.

## Files

```
experiments/03_gradient_boosting/
├── README.md           # this file
├── config.yaml         # data slice + test_fraction + features + XGBoost model knobs
├── features.py         # builds the lag/rolling/OHLCV feature matrix; no leakage
├── run.py              # entry point — fits the single config in config.yaml
├── sweep.py            # sweeps multiple XGBoost configs, writes results/sweep.csv
└── results/
    ├── metrics.json        # produced by run.py
    ├── predictions.parquet # close + pred + ref + strategy_return on the test slice
    ├── plot.png            # produced by run.py
    └── sweep.csv           # produced by sweep.py
```
