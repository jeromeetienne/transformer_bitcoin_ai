# Outline: Baselines You Need to Beat

## 1. Why baselines matter
- The floor problem: without a reference, any number can look good or bad
- The naive last-value predictor as the most honest zero-parameter model
- Historical context: random-walk hypothesis and efficient markets

## 2. The naive baseline (experiment 01_baseline)
- Definition: close_pred[t] = close[t-1]
- Why it's harder to beat than it looks on a near-random-walk asset
- The architecture it establishes: data loader, config schema, metrics, results layout

## 3. The ARIMA baseline (experiment 02_arima)
- What ARIMA actually models: integrated, differenced price changes
- The three parameters: p (AR lags), d (differencing), q (MA residuals)
- Why d=1 and what stationarity means in practice
- Walk-forward evaluation without coefficient re-estimation

## 4. The numbers
- Reporting the results from both experiments on the same slice
- Reading the metrics table: what each number tells you
- Why MAE close to naive is the *expected* outcome, not a failure
- The directional accuracy signal: where ARIMA may add value even if point MAE doesn't improve

## 5. What these baselines tell us about the problem
- The signal-to-noise ratio at hourly BTC
- What a model needs to improve on naive (and why most don't)
- Setting up the right expectations for articles 3–7
