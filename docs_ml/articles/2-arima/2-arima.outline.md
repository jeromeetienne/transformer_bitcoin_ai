# Outline — ARIMA

## Opening hook
- After naive baseline: can classical statistics do better?
- Introduce ARIMA as the workhorse of time-series forecasting for decades
- The question: does a model that explicitly models *changes* in price (not price itself) work better?

## What ARIMA is (without the math)
- Unpack ARIMA(p,d,q) conceptually:
  - d: differencing (work with price changes, not price levels)
  - p: autoregression (recent changes predict future changes - momentum)
  - q: moving average on residuals (corrections to recent surprises)
- Why differencing matters for financial data: price levels are non-stationary; changes are more stationary
- Intuition: short-term momentum + mean reversion + residual momentum

## Setup and the order sweep
- Same data as baseline (4h BTC, 2024, 402 test bars)
- Config: ARIMA(1,1,1) as the headline model
- Why a sweep: try 12 different (p,d,q) configurations and see which explains the test slice best
- The (0,1,0) row: mathematically identical to naive baseline (sanity check)

## The results on (1,1,1)
- MAE: 517.16 (beats naive by $1.20, within noise)
- RMSE: 782.42 (marginally better)
- MAPE: 0.6686 %
- Directional accuracy: 0.5547 (55.47% - the first directional opinion!)
- Sharpe: 6.0569 (significant directional edge on the rally)
- Cumulative return: 33.14%

## The sweep results
- The (3,1,3) row: leads on AIC, MAE, RMSE, MAPE, Sharpe (five columns!)
- MAE 514.07 - actually better than (1,1,1)
- But we're comparing MAEs that differ by dollars on a 402-bar slice
- Key lesson: (0,1,0) = naive (sanity check passes)
- Lesson: (1,0,1) collapses on Sharpe - differencing is non-optional
- Overfit shape: (5,1,5) is worse than (3,1,3) - diminishing returns on parameter count

## What the numbers mean
- The MAE win over naive is real but small ($1.20 / $518 = 0.23%)
- The directional win is real (0.5547 > 0.5 coin-flip)
- But directional accuracy has limitations: you can be right 55% and still lose money if sizes are wrong
- Sharpe translates direction into money: 6.06 Sharpe = about 2.59 standard errors above noise

## Walking forward vs. fitting once
- The difference between training parameters once and walking forward with frozen coefficients
- Why this matters: you're not retrain at each step, so early bars constrain later bars
- The integrity: train on 1608 bars, predict the next 402 without retraining

## Caveats and limitations
- ARIMA assumes roughly Gaussian, homoskedastic residuals (BTC is heavy-tailed, volatile clusters)
- No exogenous features (Volume, OHLC, sentiment all absent)
- No cointegration (relationship with other assets)
- Long/flat strategy, no shorting
- Single split, no rolling-window validation

## Closing
- ARIMA ties naive on point error, adds directional skill
- This is the classical-stats approach: linear, interpretable, limited
- Next up: what if we add engineered features and a non-linear model?
- The question: can we do better than $1.20 on MAE?
