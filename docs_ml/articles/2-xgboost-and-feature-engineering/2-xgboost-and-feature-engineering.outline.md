# Outline: XGBoost and Feature Engineering

## 1. The shift to nonlinear models
- What ARIMA cannot represent: interactions between past observations
- Why gradient boosting is the right next step (tabular, fast, interpretable)
- The feature engineering mandate: the model is generic, the features carry the signal

## 2. Target transformation
- Why log-returns instead of prices (stationarity, comparability)
- Reconstructing price predictions from log-return forecasts: close_pred = ref * exp(r_pred)
- The walk-forward guarantee: every feature row is shifted so no current-bar leakage

## 3. The feature matrix
- Lagged log-returns: r_{t-1}, r_{t-2}, ..., r_{t-N}
- Rolling mean and standard deviation: capturing momentum and volatility regime
- OHLCV summaries: log_volume, hl_range, oc_body
- The no-leakage invariant: every column is .shift(1) over its source

## 4. How XGBoost works (briefly)
- Trees as weak learners, boosting as additive ensemble
- Why depth × learning-rate × tree-count trade off against each other
- Regularization: L2 on leaf weights, subsampling rows and columns

## 5. The numbers and what they mean
- Comparing MAE / directional accuracy / Sharpe against the baselines
- Why MAE close to naive is the expected outcome, not a failure
- Where XGBoost may surprise: directional signal even when point error stays flat
- The sweep: bigger ≠ better on a short, noisy dataset

## 6. What this tells us about the problem
- Feature engineering as an explicit modeling choice vs. learned representations
- Setting up the LSTM: what happens when the model learns features from the sequence directly
