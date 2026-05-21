# Outline — LSTM

## Opening hook
- XGBoost saw features in isolation: each bar's returns, volatility, volume on their own
- What if we treat bars as a *sequence*? A time series where order matters?
- LSTM: deep learning's answer - a recurrent network that carries hidden state across bars

## What LSTM is (conceptually)
- Recurrent networks: a hidden state h[t] that captures "what we know so far"
- Updated at each step: h[t] = f(x[t], h[t-1])
- LSTM: an LSTM cell is a fancy recurrent gate mechanism (forget gate, input gate, output gate)
- Solve the vanishing gradient problem that simpler RNNs have (historical context can influence far-future predictions)

## The architecture
- Stack of LSTM cells: two layers, 32 hidden units per layer
- Dropout 0.1 between layers (regularization)
- Input window: 48 bars (2 days at 4h), fed to the LSTM
- Covariates: volume and OHLC (same as XGBoost)
- Output: single 1-step-ahead forecast of log-return
- Training: PyTorch Lightning + Adam optimizer, 30 epochs with early stopping

## The difference from XGBoost
- XGBoost sees flat feature vectors, finds non-linear patterns in those vectors
- LSTM sees ordered sequences, passes hidden state through time
- LSTM should learn long-range dependencies (is a bar 24 hours ago relevant?)
- But with only 1448 training rows, does it have enough data to learn those dependencies?

## The results
- MAE: 522.29 USD (4th place, worse than naive, worse than ARIMA and XGBoost)
- RMSE: 785.45 USD
- MAPE: 0.6761 %
- Directional accuracy: **0.4938** (below coin flip!)
- Cumulative return: 25.96%
- Sharpe: 4.2660 (lowest among trained models)

## The critical failure: directional accuracy below 0.5
- On this slice, the LSTM is *wrong* on direction more than it's right
- This is a failure to find signal that even naive 3-parameter ARIMA found
- Interpretation: overfitting noise in val_loss while real signal gets washed out
- Same pattern as XGBoost on MAE - capacity without enough data fits noise

## The sweep results
- 6 configurations tested on same 4h data
- Smallest config (icl=24, hidden=16, layers=1, dropout=0.0) wins on Sharpe at 5.03
- Largest config (96, 64, 3, 0.2) wins on MAE and directional accuracy (but still below 0.5)
- Default (48, 32, 2, 0.1) wins neither metric
- Key insight: bigger models perform worse on the trading metric (Sharpe)

## What the failure means
- LSTM with 1448 training rows is too much capacity
- The model learns noise in the validation loss signal
- The directional edge that AR(1) captures gets washed out
- One row (48, 64, 2, 0.2) even produces NaN Sharpe - strategy goes entirely flat!

## Early stopping paradox
- Using validation loss to stop training (good practice)
- But val_loss doesn't rank the trading metrics
- A config with better val_loss might have worse Sharpe
- The model stops when val_loss plateaus, which isn't when Sharpe is best

## Caveats
- Single seed (neural nets have run-to-run variance from hardware non-determinism)
- No early stopping tweaking (5-epoch patience might be too aggressive)
- Frozen weights across test window (walk-forward without retraining)
- Only past covariates, no future covariates (unlike Transformer in next article)

## Closing
- LSTM is the first model to add significant capacity and clearly fail
- Not just fail on point error (like XGBoost) but fail on *direction*
- The lesson: more parameters + same amount of data = worse generalization
- The question: can attention mechanisms do better with the same data?
