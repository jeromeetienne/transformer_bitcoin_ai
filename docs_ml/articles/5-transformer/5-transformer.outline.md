# Outline — Transformer

## Opening hook
- LSTM got directional accuracy below coin-flip on 1448 rows
- The pitch for Transformers: attention solves the LSTM problem
- Attention lets any past step address any other step directly
- But does it help when you have 1448 rows and 30k parameters?

## What Temporal Fusion Transformer (TFT) is
- Multi-head self-attention: each bar can "look at" every other bar in the window
- No bottleneck like LSTM's hidden state
- Variable selection networks: learns which inputs matter at each time step
- Gated residual networks: mixes features adaptively
- Future covariates: cyclical time features (hour-of-day, day-of-week) the model can use

## The architecture
- TFT backbone: LSTM encoder + decoder, multi-head attention (4 heads), gated residuals
- Future covariates: sin/cos encodings of hour and day-of-week
- Input window: 48 bars
- Training: PyTorch Lightning, 30 epochs with early stopping, same val/test split as LSTM

## The theory
- Attention should learn dependencies the LSTM hidden state can't represent
- Variable selection should focus on inputs that matter
- Future covariates give the model deterministic time structure
- All this capacity should buy better predictions than the LSTM

## The results
- MAE: **813.10 USD** (worst in the leaderboard by far!)
- RMSE: 1090.78 USD
- MAPE: 1.0786 %
- Directional accuracy: 0.4913 (below coin flip)
- Cumulative return: 44.68%
- Sharpe: 5.8675 (better than LSTM, still below ARIMA)

## The catastrophic failure: worst MAE in the 4h leaderboard
- 813.10 is 57% worse than naive's 518.36
- 51% worse than XGBoost's 539.39
- The gap between LSTM and TFT is huge
- dir_acc 0.4913 is below coin flip (like LSTM)

## The paradox: bad MAE, decent Sharpe
- Sharpe 5.87 is not terrible - still 2.5 standard deviations above noise
- The TFT predicts with low precision but high confidence
- On uptrending slice, confident-wrong (up when it's up, wrong magnitude) still makes money
- But per-bar Sharpe of 0.1254 is barely above the 0.0499 SE - borderline

## The insight: the TFT needs more data
- 30k+ parameters on 1448 training rows is completely wrong
- The model can't converge on a stable point estimate
- Variable selection networks add capacity; gated residuals add capacity
- Attention adds capacity
- All of it fits noise, none of it generalizes

## The sweep (stale 1h artifact)
- 8 configurations tested on *old* 1h data
- Different scale, cannot rank 4h configs
- Cannot guide configuration selection on 4h
- Sweep must be re-run

## What this tells us about capacity
- LSTM with ~1k parameters failed on direction
- TFT with ~30k parameters failed catastrophically on both
- Adding architectural complexity on insufficient data is self-defeating
- The pattern is clear: you need data proportional to parameters

## Architecture enhancements that didn't help
- Variable selection networks: should focus on inputs that matter (didn't help enough)
- Multi-head attention: should learn multiple dependency patterns (didn't help)
- Gated residuals: should mix features adaptively (noise-fitting instead)
- Future covariates: should give time structure (not enough data to learn their relationships)

## Caveats
- Single seed
- Frozen weights across test window
- Future covariates are calendar-only (no macro, sentiment, volume)
- Attention weights are not interpretation (high attention weight doesn't mean causation)
- Sweep is stale

## Closing
- TFT is the most dramatic failure yet
- Not just below ARIMA - 57% worse on MAE than naive
- The lesson: attention is not magic on small datasets
- The question: can pre-trained foundation models transfer a prior from general time-series data?
