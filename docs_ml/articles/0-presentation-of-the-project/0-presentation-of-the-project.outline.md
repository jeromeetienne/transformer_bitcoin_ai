# Outline - Presentation of the project

## Opening hook
- Bitcoin price prediction: the most popular machine learning failure
- Every model class from naive to state-of-the-art tested on the same slice
- The whole series in one chart: who actually beats the random walk

## What the project is
- Seven experiments, one data slice, one set of metrics
- Each experiment is its own folder with config, run script, results, report
- Each model gets a fair fight: same train/val/test split, same walk-forward, same metric module
- Everything reproducible from `make 01_baseline` to `make 07_finetuned`

## The model ladder (preview the seven articles)
- 01_baseline: naive last-value (zero parameters)
- 02_arima: classical linear time series (3 to 7 parameters)
- 03_xgboost: gradient boosting on engineered features (~31 features)
- 04_lstm: recurrent network (thousands of weights)
- 05_transformer: Temporal Fusion Transformer (tens of thousands of weights)
- 06_pretrained: zero-shot Chronos-2 and TimesFM 2.5 (28M to 200M frozen parameters)
- 07_finetuned: the same foundation models, but fine-tuned on Bitcoin

## The data
- BTCUSDT 4h bars from Binance
- Test slice: 2024-10-01 to 2024-12-01 UTC (post-election rally)
- Train slice: varies by experiment (1.6 years for most; 4.7 years for fine-tuned)
- The asset is Bitcoin in spot USDT, the bar is 4 hours, the question is "what's the next bar's close"

## The target
- 1-step-ahead log-return: `r_T = log(close_T / close_{T-1})`
- Reconstructed to price: `close_pred = close_{T-1} * exp(r_pred)`
- Why log-returns: stationary, symmetric, scales naturally

## The metrics (and why each matters)
- MAE: dollars wrong per bar - the closest-to-the-price metric
- RMSE: penalizes big misses more than small ones
- MAPE: percentage error, comparable across price levels
- Directional accuracy: how often is the sign of the predicted move right?
- Cumulative return: a long/flat strategy's gain over the test window
- Annualized Sharpe: risk-adjusted return of that strategy

## The evaluation philosophy
- Single split, walk-forward without retraining
- Long/flat strategy: long if predicted up, flat if predicted down
- No shorting, no leverage, no transaction costs
- Every number is verbatim from `metrics.json` - nothing is recomputed

## The recurring lessons (preview)
- Capacity does not equal skill: bigger models can perform worse
- Point error and trading metrics often disagree
- In-sample loss does not rank out-of-sample Sharpe
- NaN is a feature, not a bug: it signals "no opinion expressed"
- Every positive Sharpe is conditional on the regime

## How to read the series
- Each article is self-contained: what the model does, how it was set up, what it found
- Articles can be read in order (the ladder) or in any order (each one stands alone)
- The seven model articles culminate in a cross-experiment synthesis

## Closing
- Bitcoin is hard to predict, but the experiments are not a waste
- What this series is actually about: how to evaluate a model honestly, and why the model class matters less than the data-to-parameter ratio
- The reader leaves with a working mental model of the ladder, the metrics, and the gotchas
