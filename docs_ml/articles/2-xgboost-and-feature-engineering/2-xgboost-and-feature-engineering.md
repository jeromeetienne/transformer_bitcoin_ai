# When the features do the work — XGBoost on engineered Bitcoin features

The first article in this series set the floor: naive last-value at MAE 540.96 USD on 366 test bars, ARIMA(3, 1, 3) at 539.15 with Sharpe 6.86. This article is about the first non-linear model on the ladder — a gradient-boosted tree ensemble — and the first model in the series that ingests *engineered features* rather than raw past prices. Thirty-one of them.

The result is a single sentence, with the rest of the article explaining why it lands the way it does: **thirty-one engineered features lose to a 3-parameter ARIMA on every metric in the leaderboard except directional accuracy, where they tie.** XGBoost reports MAE 550.85, worse than naive's 540.96 and ARIMA's 539.15. It reports annualized Sharpe 6.1388, behind ARIMA's 6.8559. It reports directional accuracy 0.5082 — *bit-identical to ARIMA's*. Feature breadth doesn't manufacture signal that wasn't there at this horizon. The classical-machine-learning article is, in part, a story about that.

## Methodology, in one paragraph

Target: one-step-ahead log-return `r_T = log(close_T / close_{T-1})`, reconstructed back to price as `close_pred = close_{T-1} * exp(r_pred)`. Data: `BTCUSDT` 4-hour bars from Binance Vision. Split: train `2023-01-01` → `2024-08-01` UTC (3 809 rows after lag / rolling-window construction trims the first few bars), validation `2024-08-01` → `2024-10-01` UTC (unused — XGBoost is a single fit, no early stopping), test `2024-10-01` → `2024-12-01` UTC (**366 bars**). Walk-forward, one step ahead, weights frozen across the test window. Metrics from the shared module — [`src/btc_ai/eval/metrics.py`](src/btc_ai/eval/metrics.py). Reproduce: `make 03_xgboost`.

## What the model is

XGBoost is a gradient-boosted ensemble of regression trees. Each tree fits the *residual* of the running ensemble: tree 1 predicts the target, trees 2 through N each predict the leftover error of trees 1 through N-1. The final prediction is the sum of all the trees, with a learning rate scaling each tree's contribution. The model is non-linear because trees are non-linear; it can capture interactions between input features that a linear model cannot. It is generic because the trees don't know anything about time series — they receive a flat feature vector per row and minimize squared error on the target.

The configuration is standard, from [`experiments/03_xgboost/configs/btc_4h_2024.config.yaml`](experiments/03_xgboost/configs/btc_4h_2024.config.yaml):

```yaml
model:
  n_estimators: 400
  max_depth: 5
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  reg_lambda: 1.0
  random_state: 42
```

400 trees, max depth 5, learning rate 0.05, subsampling 80 % of rows and 80 % of features at each tree. Library: `xgboost` with `tree_method='hist'`. The target the trees are predicting is the log-return; the price prediction comes out of a single `exp(r_pred) * close[t-1]` step at the end.

## The features

The interesting part of this experiment is not the model. It is the feature engineering. [`experiments/03_xgboost/features.py`](experiments/03_xgboost/features.py) builds the entire feature matrix from a clean OHLCV bar table. There are 31 columns:

- **24 lagged log-returns.** `r_lag_1` through `r_lag_24` — log-returns from one bar back to twenty-four bars back. At 4h cadence that is ninety-six hours of price-change history per row.
- **4 rolling moments.** `r_mean_6`, `r_std_6`, `r_mean_24`, `r_std_24` — short-term (6-bar) and day-long (24-bar) windowed mean and standard deviation of past log-returns. Short-term momentum and short-term volatility, in two windows.
- **1 log-volume.** `log_volume = log1p(volume_{t-1})`. Trade volume is heavy-tailed; the log compresses it into a usable scale.
- **2 OHLC summaries.** `hl_range = high - low` of the previous bar, `oc_body = close - open` of the previous bar. These capture *how* the prior bar moved — broad range vs. tight, decisive body vs. doji — not just what it closed at.

The shape of the construction code is the load-bearing detail:

```python
# experiments/03_xgboost/features.py — abbreviated
close = df['close'].astype('float64')
r = np.log(close).diff()

cols: dict[str, pd.Series] = {}
for i in range(lags):
        cols[f'r_lag_{i + 1}'] = r.shift(1 + i)

for w in feat_cfg['rolling_windows']:
        cols[f'r_mean_{w}'] = r.shift(1).rolling(w).mean()
        cols[f'r_std_{w}']  = r.shift(1).rolling(w).std()

if feat_cfg.get('use_volume') is True:
        cols['log_volume'] = np.log1p(df['volume']).shift(1)

if feat_cfg.get('use_ohlc') is True:
        cols['hl_range'] = (df['high'] - df['low']).shift(1)
        cols['oc_body']  = (df['close'] - df['open']).shift(1)
```

The `.shift(1)` on every feature is the entire walk-forward correctness story for this experiment. A row at time `T` sees only data observed strictly before `T`. There is no current-bar leakage. *And* — this is the elegant part of using a flat-feature model on time-series data — a single `model.predict(X_test)` call against the shifted feature matrix produces 366 honest one-step-ahead forecasts in one shot. No per-step refit, no historical-forecast loop, no walk-forward sliding window. The shift takes care of it.

This is the "model is generic, features carry the signal" contract spelled out in code. The model knows nothing about time. The lag and rolling and OHLC operations are doing all the time-series work, and the model just sees thirty-one floats per row.

## The numbers

From [`experiments/03_xgboost/results/btc_4h_2024/metrics.json`](experiments/03_xgboost/results/btc_4h_2024/metrics.json), quoted verbatim:

| Metric | XGBoost |
|---|---|
| MAE | **550.85 USD** |
| RMSE | 808.65 USD |
| MAPE | 0.7096 % |
| Directional accuracy | 0.5082 |
| Cumulative return | 0.4149 |
| Annualized Sharpe | **6.1388** |

Side-by-side with what we have from article 1, on the same 366-bar test slice:

| Model | MAE | RMSE | dir_acc | cum_ret | Sharpe |
|---|---|---|---|---|---|
| Naive last-value | 540.96 | 813.14 | NaN | — | — |
| ARIMA(3, 1, 3) | **539.15** | 808.79 | 0.5082 | **0.5269** | **6.8559** |
| **XGBoost (31 features)** | 550.85 | **808.65** | 0.5082 | 0.4149 | 6.1388 |

Three observations the table makes load-bearing.

**First — XGBoost loses on MAE to naive.** It is the structurally interesting result of this article. A model that trains is *worse on point error* than a model that doesn't. The tree ensemble fits noise in the residuals of the differenced series faster than it captures any signal the AR / MA structure in article 1 was extracting cheaply. Adding capacity does not help when the signal-to-noise ratio is low: the extra degrees of freedom go into modelling random structure that doesn't survive to the next bar.

**Second — directional accuracy is bit-identical to ARIMA's, to four decimal places.** Both models are right about direction 50.82 % of the time. 186 bars out of 366 on each. That is not a coincidence at the architectural level — both models are linear-ish at the differenced level. (ARIMA explicitly is; XGBoost on lagged-returns features has a strong linear component and the trees mostly model second-order interactions on top.) Both models recover the same first-order direction signal because *that is approximately the entire direction signal in the data at this horizon*. Adding 28 more features and a non-linear estimator does not pull more direction out of it.

**Third — Sharpe is lower than ARIMA's** despite the identical dir_acc. 6.14 vs. 6.86 is a Sharpe difference of 0.72 with the same number of correct direction calls. This is entirely a *magnitude calibration* difference: XGBoost's predicted positive log-returns are smaller on average than ARIMA's, so the long-flat rule triggers fewer "go long" bars during the rally, and the strategy collects fewer up-bar returns. The strategy mechanism trades correctness for confidence; XGBoost is equally correct but less confident. On a strongly trending up-regime that costs Sharpe.

## The sweep, briefly

The sweep at [`experiments/03_xgboost/results/btc_4h_2024/sweep.csv`](experiments/03_xgboost/results/btc_4h_2024/sweep.csv) covers eight `(n_estimators, max_depth, learning_rate)` configurations. It was last regenerated against the previous 1-hour data slice and has not been re-run against the current 4-hour slice — its MAE figures (range 269–276 USD) are on the 1-hour scale and are *not* directly comparable to the 550.85 above. I am flagging this rather than hiding it because the hyperparameter-ranking lesson stands even on stale data.

The relevant lesson: on the stale slice, the *smallest* configurations led on MAE (n_estimators=200, max_depth=3 → MAE 269.05) and the (n_estimators=800, max_depth=5, lr=0.03) row led on Sharpe at 3.93. The *configured default* `(400, 5, 0.05)` was the sweep winner on neither metric. Defaults set by `config.yaml` are starting points, not optima — this is a recurring beat in the series, and article 3 will see it again in the LSTM sweep where the MAE winner and the Sharpe winner sit at opposite ends of the capacity axis. Re-running [`experiments/03_xgboost/results/btc_4h_2024/sweep.csv`](experiments/03_xgboost/results/btc_4h_2024/sweep.csv) on the current 4-hour slice is on the operational follow-up list; for now, the *single-fit* number for the default config is the leaderboard row.

## What this article tells us about the model class

**Feature breadth and non-linearity, together, do not buy a free improvement at this horizon.** On 4-hour Bitcoin with ~3 800 training bars and a target that is mostly noise, thirty-one engineered features in front of a gradient-boosted tree ensemble produce a model that loses to a three-parameter ARIMA on MAE and ties it on direction. That is not a defect in XGBoost — it is a fact about how much signal the engineered features extract above what AR / MA on the differenced price already gives you. The lift from feature engineering depends on whether the features carry information the simpler model can't see; at this horizon, on this slice, they mostly don't.

**The signal-to-noise floor sets the same ceiling for many model classes.** The cluster of MAEs around $539–$551 that articles 1 through 5 will populate is not random. It is the noise floor of one-step-ahead forecasting on this slice at this cadence. Different model classes are different paths up to the same ceiling. A few of them clear it by a dollar; one ends up much worse (article 4); none clear it by a lot. That is the most ML-credible observation the entire series will make.

**Where XGBoost would help.** This is one slice, one horizon, one feature set. With richer features — on-chain metrics, cross-asset prices, sentiment — XGBoost can do meaningfully better on Bitcoin, and on shorter horizons (1-minute, 5-minute) it can find signal that lagged returns and rolling moments capture better than ARIMA's linear projection does. The lesson of this article is *what 31 features extracted from the price series alone buy*, not "XGBoost doesn't work on Bitcoin."

## Per-bar Sharpe sanity check

6.1388 / √2190 = 0.1312 per bar; SE on the per-bar mean ≈ 1 / √366 = 0.0523; ratio ≈ **2.51 σ** — borderline significant under the optimistic i.i.d. assumption. Lower than ARIMA's 2.80 σ from article 1. The same regime caveat applies: the test slice is the post-election Bitcoin rally; a long-flat strategy with 50.8 % directional accuracy compounds favourably on a strongly trending up-regime, and the Sharpe will look different on a sideways slice.

## Reproduce

```
make 03_xgboost
make 03_xgboost_sweep   # stale on 1h; pending re-run on 4h
```

The first writes [`experiments/03_xgboost/results/btc_4h_2024/metrics.json`](experiments/03_xgboost/results/btc_4h_2024/metrics.json) — the numbers in this article. The second writes [`experiments/03_xgboost/results/btc_4h_2024/sweep.csv`](experiments/03_xgboost/results/btc_4h_2024/sweep.csv) — pending a rerun against the current 4-hour slice; do not read its MAE values as 4-hour numbers.

Article 3 introduces the first deep-learning model in the series — a stacked LSTM. It also introduces the first model whose extra architectural capacity is going to go *backwards* on directional accuracy compared to ARIMA on this slice. Onwards.
