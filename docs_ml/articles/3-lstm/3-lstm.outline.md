# Outline: LSTM — Learning Sequences

## 1. The step from tabular to sequential
- What XGBoost cannot represent: order within the feature window
- The LSTM's inductive bias: hidden state that summarizes arbitrary history
- Why the same features that go into trees become a different kind of signal for a recurrent net

## 2. How an LSTM works
- The cell state and the three gates: forget, input, output
- What "memory" means in practice at hourly BTC timescales
- Stacked layers and dropout as regularization

## 3. The experiment setup
- Darts BlockRNN(LSTM): same framework as the Transformer article that follows
- Target: log-return, same as XGBoost — for direct comparability
- Past covariates as parallel time series: log_volume, hl_range, oc_body
- Three-way split: train / val (early stopping) / test
- Walk-forward via historical_forecasts(retrain=False)

## 4. The hyperparameter surface
- input_chunk_length: how much history the LSTM sees
- hidden_dim and n_rnn_layers: capacity vs. overfitting risk
- Early stopping on val_loss: why the training curve matters
- What the Optuna sweep reveals about the parameter landscape

## 5. The numbers and what they mean
- MAE / directional accuracy / Sharpe against the baseline table
- When LSTM beats XGBoost and why (trajectory vs. snapshot)
- The typical honest outcome: capacity doesn't help on near-i.i.d. targets

## 6. Bridging to the Transformer
- What LSTM misses: non-local dependencies, future covariates
- The attention mechanism as a different solution to the same problem
