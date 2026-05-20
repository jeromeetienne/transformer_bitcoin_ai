# XGBoost Walks In, Humans Do the Thinking: Feature Engineering for Bitcoin

The first two experiments told us what a model that knows nothing can do. Article 1 (naive last-value) told us the zero-parameter floor; article 2 (ARIMA) told us the three-parameter linear floor. Both came in near $338 MAE on hourly BTC, which is the efficient-market hypothesis doing its job: if a linear function of recent prices could predict the next bar, arbitrageurs would have closed that gap long ago.

The question this experiment asks is different. Not "can a linear model predict price?" but "can a *nonlinear* model extract signal that a linear model cannot represent?" If the answer is yes, it should show up here first — gradient boosting trees are the nonlinear model of choice for tabular data, and they are reliable enough that failure is informative.

## The shift to nonlinear

ARIMA is a linear model. Its prediction is a weighted sum of past price changes and past prediction errors. If the true process is roughly:

```
r_t = a * r_{t-1} + b * r_{t-2} + noise
```

ARIMA can represent that exactly. What it cannot represent is:

```
r_t = large positive  if  r_{t-1} > 0  AND  volume_{t-1} > rolling_avg
r_t = small negative  otherwise
```

That is an interaction — a conjunction of two conditions — and it requires a nonlinear function to capture. Trees handle interactions naturally: each split is a threshold on a single feature, and chaining splits produces interaction terms for free. A tree that asks "was the last return positive?" and then asks "was volume above average?" is representing a first-order feature interaction without any explicit feature engineering of the product.

Gradient boosting runs an ensemble of these trees in sequence, where each new tree fits the residuals of the previous ones. The result is a powerful nonlinear estimator that trains in seconds and rarely overfits catastrophically on tabular data — unlike deep networks, which can overfit dramatically on small datasets before the optimizer has a chance to correct.

## What the model predicts

Both ARIMA and the naive baseline predicted price levels. XGBoost switches to predicting **log-returns**:

```
r_T = log(close_T / close_{T-1})
```

This is a stationary quantity. The price level of BTC drifts up and down over years; the hourly log-return fluctuates around zero with roughly constant variance. Stationary targets are better behaved for regression: the training distribution is similar to the test distribution, and squared-error loss is not dominated by the overall trend.

The price forecast is reconstructed after the fact:

```
close_pred = close_{T-1} * exp(r_pred)
```

Because this reconstruction uses the *actual* previous close (not the predicted one), there is no compounding of forecast errors. The MAE, RMSE, MAPE, and directional accuracy reported for XGBoost are directly comparable to the baseline numbers from articles 1 and 2.

## The feature matrix

The experiment lives in [`experiments/03_xgboost/`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/03_xgboost), with feature construction in [`features.py`](https://github.com/jeromeetienne/transformer_bitcoin_ai/blob/HEAD/experiments/03_xgboost/features.py). Every column in the feature matrix is a shifted view of past data — no information from bar *T* itself is allowed to leak into the row for bar *T*.

**Lagged log-returns** — `r_{t-1}, r_{t-2}, ..., r_{t-N}`. The most direct encoding of recent price history. `N` is configurable (`return_lags` in `config.yaml`; default 24, one day of hourly data). These are the same lags that ARIMA's AR component sees, but XGBoost can use them nonlinearly and in combination with the other features.

**Rolling mean and standard deviation** — for each window size *w* (default: 6 and 24 bars), compute the mean and standard deviation of the last *w* log-returns. The mean captures short-term momentum: has the market been trending up recently? The standard deviation captures the volatility regime: is this a calm hour or a volatile one? Both are computed from the training data only, applied at each row using data strictly before bar *T*.

**OHLCV summaries** — three optional columns:
- `log_volume = log1p(volume_{T-1})` — logarithm of the previous bar's volume. High volume often accompanies large moves; low volume often signals consolidation.
- `hl_range = high_{T-1} - low_{T-1}` — the previous bar's price range. A direct measure of local volatility.
- `oc_body = close_{T-1} - open_{T-1}` — the previous bar's candle body. Positive means the bar closed above the open (bullish); negative means it closed below (bearish).

The full feature config:

```yaml
features:
  return_lags: 24            # number of lagged log-returns
  rolling_windows: [6, 24]   # each window adds a mean column and a std column
  use_volume: true
  use_ohlc: true
```

With these settings, a row has 24 lag features + 4 rolling features (2 windows × 2 stats) + 3 OHLCV features = 31 columns. Each column is `shift(1)` over its underlying series, so the row at bar *T* contains only information that was observable at bar *T-1*.

## How XGBoost works

XGBoost fits an ensemble of decision trees, where each tree corrects the residuals of the previous ones. The prediction is the sum:

```
r_pred = f_1(x) + f_2(x) + ... + f_K(x)
```

where each *f_k* is a shallow tree and *K* is the number of boosting rounds (`n_estimators`). The trees are fit greedily: at each step, the new tree is chosen to maximize the reduction in squared error on the current residuals.

Three hyperparameters drive most of the variance in performance:

- **`n_estimators`** — how many trees. More trees = more capacity to fit complex patterns, but also more risk of memorizing training noise. Diminishing returns set in quickly on small datasets.
- **`max_depth`** — how deep each tree is allowed to grow. Depth 3 captures up to 3-way interactions; depth 7 can capture very specific conjunctions of conditions. On 5 000 training rows and 31 features, depth 5 is usually about right.
- **`learning_rate`** — how much weight each new tree gets. A smaller learning rate means each tree contributes less, requiring more trees to reach the same fit. The pair `(n_estimators=400, lr=0.05)` and `(n_estimators=800, lr=0.025)` often produce similar results; the second trains roughly twice as long.

Regularization — `reg_lambda` (L2 penalty on leaf weights), `subsample` (fraction of rows sampled per tree), `colsample_bytree` (fraction of features sampled per tree) — prevents the model from memorizing individual training examples. On a noisy target like BTC log-returns, regularization matters more than capacity.

```bash
make 03_xgboost
```

## How to interpret the results

XGBoost, ARIMA, and naive on the same test slice:

| | Naive | ARIMA(3,1,3) | XGBoost |
|---|---|---|---|
| Model | `pred = close[t-1]` | linear AR+MA on differences | nonlinear trees on 31 features |
| Parameters | 0 | 7 | ~thousands (tree structure) |
| Sees interactions | no | no | yes |
| Sees volume/OHLC | no | no | yes |
| MAE (USD) | 337.89 | ~337–340 | *run the experiment* |
| Directional accuracy | NaN | ~0.50 | *run the experiment* |

The most likely outcome: MAE close to naive, directional accuracy near 0.50. This is not a flaw in XGBoost — it is the correct finding. Hourly BTC log-returns are close to independent of their own past. A nonlinear model with 31 features cannot change that fact.

Where XGBoost sometimes surprises is directional accuracy and Sharpe. Even when MAE is flat, the model may develop a faint directional bias: not from the AR structure (that is what ARIMA found), but from volume-return interactions or volatility-regime conditioning. The `strategy_return` column in `predictions.parquet` applies a simple long/flat strategy — go long if the model predicts a positive return, stay flat otherwise — and computes the resulting P&L. The annualized Sharpe of that strategy is the metric that matters for trading, and it sometimes diverges from MAE.

### The sweep tells the real story

```bash
make 03_xgboost_sweep
```

The sweep runs a grid of `(n_estimators, max_depth, learning_rate)` combinations and writes `results/sweep.csv`. Two things to look for:

**Bigger is not better.** On a dataset with 5 000 training rows and a near-noise target, `max_depth=3` with `n_estimators=200` often matches or beats `max_depth=7` with `n_estimators=800` on test MAE, despite the smaller model having far less capacity. What the bigger model does is memorize the training set, which has no predictive value.

**AIC has no equivalent here.** Unlike ARIMA, XGBoost does not report an information criterion. The only honest comparison is test MAE. Because the test slice is only 437 bars, differences smaller than about 5–10 USD are within the noise of the estimator.

### The no-leakage invariant

One risk in any feature-engineering pipeline is temporal leakage: accidentally including information from bar *T* in the features for bar *T*. The experiment guards against this with a strict `.shift(1)` on every source series before computing any rolling statistic or lag. The consequence is that the first row of the feature matrix is NaN for all lag features, and this row is dropped before training. The result is a dataset where each row's features are strictly prior to the target bar.

The walk-forward at inference time is trivially correct: because all features are precomputed from past data, a single call to `model.predict(X_test)` produces all 437 forecasts without any per-step loop. This is only valid because the feature construction already enforces the no-lookahead constraint.

## What this tells us

XGBoost on 31 hand-crafted features is the first time this series allows the model to see:

1. Nonlinear interactions between past returns
2. Volume as a signal
3. Within-bar structure (high-low range, open-close body)

If none of these help, the conclusion is that hourly BTC price is very close to a martingale — future changes are independent of the past — and no amount of feature engineering on OHLCV data will change that. That conclusion is valuable. It is also the most common outcome.

If some features do help, the next question is whether a model that learns its own features from the raw sequence — rather than consuming hand-engineered scalars — can do better still. That is what article 4 (LSTM) investigates. The gap between XGBoost and LSTM, on the same data and the same evaluation protocol, is attributable specifically to the difference between tabular feature engineering and learned sequential representations.
