# Social Media Posts — Article 2: XGBoost Walks In, Humans Do the Thinking

## Twitter

**Thread: Can XGBoost beat a zero-parameter Bitcoin predictor?**

1/ The naive last-value predictor costs $337 MAE on hourly BTC.

I threw XGBoost at it — 31 hand-crafted features, nonlinear trees, volume, OHLC.

Here's what happened. 🧵

2/ The key move: stop predicting price levels. Predict log-returns instead.

`r_T = log(close_T / close_{T-1})`

Stationary, zero-mean, comparable across time. Then reconstruct the price after the fact.

3/ The features I built:
- 24 lagged log-returns (1 day of hourly history)
- Rolling mean + std over 6h and 24h windows
- Log-volume, high-low range, open-close candle body

31 columns total. Every one of them strictly shifted by 1 bar. No leakage.

4/ XGBoost finds interactions ARIMA cannot see.

"Did the last return go up AND was volume above average?"

Trees handle conjunctions like that naturally. Linear models cannot.

5/ The most likely outcome: MAE near $337. Directional accuracy near 0.50.

That's not a broken experiment. That's the honest signal-to-noise ratio of hourly BTC.

But watch the Sharpe on the long/flat strategy — it sometimes diverges from MAE.

Full article → [link]

#MachineLearning #Bitcoin

---

## Bluesky

**I gave XGBoost 31 features and pointed it at hourly Bitcoin. Here's what nonlinear trees can (and can't) do.**

The setup: predict log-returns using lagged returns, rolling volatility, volume, and OHLC structure. Strictly no-lookahead — every feature is shifted one bar before any rolling stat is computed.

XGBoost can find interactions that ARIMA fundamentally cannot: "high volume AND recent upward move" is a conjunction, and trees capture it for free.

The most common result: MAE still near the $337 naive floor. That's not a failure. It's the correct finding about hourly BTC — the signal-to-noise ratio is low, and no amount of feature engineering on OHLCV changes that fact.

The interesting metric is directional accuracy and strategy Sharpe, not MAE.

Article 2 → [link]

#MachineLearning #TimeSeries

---

## LinkedIn

**XGBoost on Bitcoin: what 31 hand-crafted features teach you about signal-to-noise ratio.**

In this series on ML for Bitcoin price forecasting, Article 2 introduces the first nonlinear model: XGBoost with feature engineering.

The key design decisions:

**Predict log-returns, not price levels.** Log-returns are stationary and zero-mean — a much better regression target than raw price, which drifts by thousands of dollars over the training period.

**31 features, all strictly no-lookahead.** Lagged returns (24 lags), rolling mean and standard deviation (6h and 24h windows), log-volume, high-low range, and candle body. Every column is computed from data prior to the target bar. No exceptions.

**Nonlinear interactions.** XGBoost handles feature conjunctions that ARIMA cannot represent: things like "volume was high AND the last return was positive." Each tree split is a threshold on a single feature; chaining splits produces interaction terms automatically.

The likely result: MAE near $337, close to the naive floor established in Article 1.

That outcome is informative, not disappointing. Hourly Bitcoin price is close to a martingale — the signal available in OHLCV is thin. XGBoost's value here is not beating the baseline; it is establishing that the baseline can't be beaten with classical feature engineering. That sets up a sharper question for the LSTM and Transformer: can a model that learns its own features from the raw sequence do what hand-engineering could not?

The hyperparameter sweep (depth, n_estimators, learning rate) also tells a consistent story: on a noisy target with 5 000 training rows, regularization matters more than capacity.

Article 2 is live → [link]
