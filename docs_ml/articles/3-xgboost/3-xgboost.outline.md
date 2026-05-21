# Outline — XGBoost

## Opening hook
- ARIMA beat naive by $1.20 on point error and added direction
- What if we break the linearity assumption? What if we engineer features?
- XGBoost: the "generic" model powered by engineered feature work

## The paradigm shift
- Move from univariate to multivariate: use lagged returns, rolling windows, OHLC
- Move from linear to non-linear: gradient boosting ensemble
- The thesis: capacity + engineered features should beat linear + univariate

## Feature engineering (the real work)
- Lagged log-returns: 24 past bar-returns (r[t-1] through r[t-24])
- Rolling statistics: mean and std of 6-bar and 24-bar windows (4 features)
- OHLC summaries: high-low range, open-close body (2 features)
- log_volume[t-1]: log of previous bar volume (1 feature)
- Total: 31 features, all lagged by one bar (walk-forward ready)
- The insight: the model is generic; the features carry the signal

## XGBoost: the model
- Gradient boosting: sequential trees, each one correcting the previous
- Target: log-return (same as ARIMA)
- Regularization: max_depth 5, subsample 0.8, colsample_bytree 0.8, lambda 1.0
- No early stopping in the baseline run
- Single pass prediction on test (no per-step retraining like ARIMA)

## The results
- MAE: 539.39 USD (worse than naive 518.36!)
- RMSE: 795.07 USD
- MAPE: 0.7008 %
- Directional accuracy: 0.5365
- Cumulative return: **50.42%** (best in leaderboard)
- Sharpe: **6.4233** (best in leaderboard)

## The paradox: 31 features lose to 3 parameters on MAE
- XGBoost is third on MAE - worse than naive, worse than ARIMA(1,1,1)
- But it's first on cum_ret and Sharpe
- The lesson: capacity doesn't manufacture signal you don't have
- On a near-random-walk, trees fit noise in residuals while linear momentum (AR(1)) gets washed out

## How XGBoost wins on Sharpe despite losing on MAE
- Directional aggression: XGBoost makes bigger bets on its directional calls
- On bars it predicts "up", it predicts further up than ARIMA
- In a uptrending rally, more-aggressive bets accumulate more returns
- The strategy is long/flat: on up predictions, go long; on down predictions, stay flat
- XGBoost stays long more often and bigger

## The sweep (stale on 1h - caveat)
- 8 configurations tested on the *old* 1h data slice
- Cannot rank current 4h configurations
- Smallest configs (n_estimators=200, max_depth=3) lead on MAE
- (800, 5, 0.03) leads on Sharpe
- Lesson: bigger is not better on near-random-walk data
- Note: sweep must be re-run on 4h to be meaningful

## What this tells us
- Point error ceiling: on this signal-to-noise ratio, engineered features hit a ceiling
- The ceiling is higher than ARIMA on direction/strategy metrics
- But it's lower than ARIMA on point error
- The tree ensemble trades small-magnitude precision for directional confidence

## Caveats
- Single fit, no multi-seed runs
- No early stopping (sweep later tests this)
- Features minimal (no on-chain, no sentiment, no macro)
- Sweep is stale and cannot guide 4h config selection
- Long/flat strategy, no shorting, no transaction costs

## Closing
- 31 features + non-linearity don't beat linear + univariate on point error
- But they do generate a trading strategy with better Sharpe
- The paradigm: if point error is your floor, features are your ceiling
- Next: can sequence models (LSTM) do better by seeing patterns across time?
