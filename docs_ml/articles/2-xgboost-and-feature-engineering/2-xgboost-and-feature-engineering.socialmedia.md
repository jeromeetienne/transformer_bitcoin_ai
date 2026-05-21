# Social Media Posts — Article 2: XGBoost Walks In, Humans Do the Thinking

## Twitter

**Thread: Can XGBoost beat a zero-parameter Bitcoin predictor? 🧵**

1/ The naive floor: $337 MAE on hourly BTC. Zero parameters.

I gave XGBoost 31 hand-crafted features — lagged returns, rolling volatility, volume, OHLC structure — and pointed it at the same test slice.

Here's what nonlinear trees can and can't do.

2/ First design decision: predict log-returns, not price levels.

`r_T = log(close_T / close_{T-1})`

Stationary, zero-mean, comparable across time. Reconstruct price after the fact using the actual previous close. No compounding errors.

3/ The 31 features — all strictly no-lookahead:
- 24 lagged log-returns (one full day of hourly history)
- Rolling mean + std over 6h and 24h windows
- Log-volume, high-low range, open-close candle body

Every column is `.shift(1)` before any rolling stat is computed.

4/ XGBoost finds interactions ARIMA fundamentally cannot.

"Volume above average AND last return positive" is a conjunction. Trees capture it by chaining splits. Linear models can't represent it at all.

5/ Most likely result: MAE near $337. Directional accuracy near 0.50.

That's the correct finding — not a broken experiment. Hourly BTC is close to a martingale. No feature engineering on OHLCV changes that.

The question that sets up article 4: can learned features do what hand-engineering couldn't?

Full article → [link]

#MachineLearning #Bitcoin

---

## Bluesky

**I gave XGBoost 31 features and pointed it at hourly Bitcoin. Here's what nonlinear trees actually find.**

The key move: predict log-returns instead of price levels — stationary, zero-mean, comparable across the full series. Reconstruct price after the fact.

31 features covering lagged returns, rolling volatility, volume, and OHLC structure. Every column strictly prior to the target bar — no leakage.

XGBoost can capture interactions ARIMA can't: "high volume AND recent upward move" is a conjunction, and trees handle it for free.

Most likely result: MAE still near the $337 floor. That's the honest signal-to-noise ratio of hourly BTC. The hyperparameter sweep shows it clearly: on a noisy target, regularization matters more than capacity.

Article 2 → [link]

#MachineLearning #TimeSeries

---

## LinkedIn

**XGBoost on Bitcoin: what 31 hand-crafted features reveal about signal-to-noise ratio.**

Article 2 of this forecasting series introduces the first nonlinear model: gradient-boosted trees with feature engineering. Three design decisions drove the experiment.

**Predict log-returns, not price levels.** Raw BTC price drifts by thousands of dollars over the training period — a terrible regression target. Log-returns are stationary and zero-mean. The price forecast is reconstructed after the fact using the actual previous close, so there is no compounding of errors across the test set.

**31 features, all strictly no-lookahead.** Lagged returns (24 lags, one full day of hourly history), rolling mean and standard deviation (6h and 24h windows), log-volume, high-low range, and open-close candle body. Every column is derived from data strictly prior to the target bar — enforced by a `.shift(1)` before any rolling operation is computed.

**Nonlinear interactions.** XGBoost handles feature conjunctions that ARIMA fundamentally cannot represent: volume-return interactions, volatility-regime conditioning. Each split in a tree is a threshold on a single feature; chaining splits produces interaction terms automatically, with no explicit feature engineering of the product.

The expected result: MAE near the $337 naive floor. That is not a disappointing outcome — it is an accurate description of the signal available in hourly OHLCV data. The hyperparameter sweep (depth, n_estimators, learning rate) confirms it: on a noisy target with 5,000 training rows, regularization matters more than model capacity. The bigger model memorizes the training set; it does not generalize.

The value of this experiment is in what it rules out. If nonlinear trees with 31 hand-crafted features can't move the needle, the question for the LSTM and Transformer becomes sharp: can a model that learns its own features from the raw sequence do what explicit engineering could not?

Article 2 is live → [link]

#MachineLearning #TimeSeries #FeatureEngineering
