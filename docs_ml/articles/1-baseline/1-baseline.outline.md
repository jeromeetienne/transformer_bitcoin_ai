# Outline — Baseline

## Opening hook
- Frame the problem: predicting Bitcoin price movements is hard. But how hard?
- Introduce the concept of a baseline: the dumbest possible model that everything else must beat
- The thesis: naive-last-value is simpler than you think, and nothing else beats it by much

## The naive predictor (what it does)
- Define the model: `close_pred[t] = close[t-1]`
- Explain why this is called the "zero-parameter floor"
- Walk through the logic: if price doesn't change (a lot) from bar to bar, why predict it will?

## Setup and methodology
- Data: 4h Bitcoin OHLCV from 2024-01-01 to 2024-12-01 (2,010 bars)
- Train/test split: 80/20 monthly walk-forward (402 test bars)
- Metrics: MAE, RMSE, MAPE, directional accuracy (the "can you call the move direction" metric)
- Why walk-forward matters: realistic backtest that respects time flow

## The results
- MAE: 518.36 USD - what it means for a volatile asset
- RMSE: 784.09 USD - the higher variance of outliers
- MAPE: 0.6701% - percentage error
- Directional accuracy: NaN (by design - naive predictor never picks a direction)
- Sharpe, cumulative return: N/A (no directional opinion, no strategy)

## Why directional accuracy is NaN
- The metric's design: counts bars where predicted direction matches actual direction
- The naive baseline's behavior: predicts "no change" everywhere, so pred_direction = 0
- The mask filters out all bars - result: NaN by design, not a failure

## The important insight
- This isn't just a floor to beat - it's a *commentary on signal-to-noise*
- The gap between naive and the next best model (ARIMA) is embarrassingly small: $1.20
- What that tells us: on this slice, point error and directional skill are nearly orthogonal
- You can be "right on average price" but still get the direction wrong (or vice versa)

## Closing
- What comes next: can we beat $518 MAE? Can we add directional skill?
- Frame the series: each next model trades something for something else
- This baseline is the constant yardstick for the entire series
