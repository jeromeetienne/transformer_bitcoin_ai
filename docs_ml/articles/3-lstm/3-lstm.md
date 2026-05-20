# LSTM: Learning Sequences

XGBoost saw 31 features per bar and made an unordered prediction: a row of numbers goes in, a number comes out. The order in which those lag features were computed was implicit in their names — `r_lag_1` is more recent than `r_lag_24` — but nothing in the model architecture enforced that. A tree doesn't care which column comes first; it treats every feature as an independent axis in feature space.

The long short-term memory network (LSTM) cares about order. It is designed to process a sequence of observations one at a time, maintaining a hidden state that summarizes everything it has seen so far. The prediction for bar *T* is a function not of a row of features but of the entire *trajectory* of bars leading up to it. That is a fundamentally different inductive bias, and this experiment tests whether it is a useful one for hourly Bitcoin price.

## How an LSTM works

The LSTM is a recurrent neural network with a specific gating mechanism designed to solve the vanishing gradient problem that made early recurrent networks nearly untrainable on long sequences.

At each timestep *t*, the LSTM takes the current input *x_t* and its previous hidden state *h_{t-1}*, and produces an updated hidden state *h_t* and cell state *c_t*. The cell state is the long-term memory; the hidden state is what gets passed to the next layer or to the output.

Three gates control what information flows through:

**Forget gate** — decides what fraction of the current cell state to discard. If the model has been tracking a trending regime and sees a sudden reversal, the forget gate should activate to clear the trend signal.

**Input gate** — decides what new information from the current input to write into the cell state. The new information itself is computed by a separate tanh transform of the input and previous hidden state.

**Output gate** — decides what to expose from the cell state as the hidden state. Even if the cell carries rich information, the output gate can choose to only expose a relevant subset.

Mathematically:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)      # forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)      # input gate
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)   # candidate cell update
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t          # cell state update
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)      # output gate
h_t = o_t ⊙ tanh(c_t)                    # hidden state
```

Where σ is the sigmoid function and ⊙ is elementwise multiplication. The trainable parameters are the weight matrices *W_f, W_i, W_g, W_o* and the biases *b_f, b_i, b_g, b_o*. For a hidden dimension of 32, each gate matrix is 32×(32+input_dim), so a single LSTM layer is a fairly compact model.

Stacking layers connects the output of one LSTM to the input of the next, allowing the network to build hierarchical representations of the sequence. Dropout is applied between layers as regularization — at each training step, a random fraction of hidden activations is zeroed out, forcing the network not to rely on any single activation path.

## The experiment

The experiment is in [`experiments/04_lstm/`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/04_lstm), using the Darts `BlockRNNModel` with `model='LSTM'`. Darts provides the same high-level API as the Transformer experiment that follows, so the only architectural difference between articles 4 and 5 is the model class.

**Target**: bar-*T* log-return `r_T = log(close_T / close_{T-1})`, same as XGBoost. Predictions are reconstructed to price as `close_pred = close_{T-1} * exp(r_pred)` for metric comparability.

**Past covariates**: three parallel time series alongside the target log-return — `log_volume`, `hl_range`, and `oc_body` — each aligned to the same timestamp and shifted so the row at bar *T* sees only data from bar *T-1*. Darts' `BlockRNNModel` feeds past covariates only for timesteps before the forecast horizon, so there is no leakage.

**Three-way split**: the dataset is divided into train (the bulk of the series), validation (the tail of the training window, carved off for early stopping), and test (the final 20% of the full series). The validation slice is used only for early stopping — the model never sees test labels during training.

**Walk-forward evaluation**: `model.historical_forecasts(start=test_start, forecast_horizon=1, retrain=False, last_points_only=True)` slides the input window across the test slice bar by bar, each time predicting one step ahead using only actual past values. The `retrain=False` flag keeps the weights frozen: this matches the evaluation protocol used by every other experiment in the series.

```bash
make 04_lstm
```

## The hyperparameter surface

The four knobs that matter most, and what they trade:

**`input_chunk_length`** (default: 48) — how many bars of history the LSTM processes before making a prediction. On hourly BTC, 48 bars is two days. The question is whether signal from two days ago is useful. On a near-random-walk, the answer is usually no: the LSTM's hidden state at the end of a 48-bar window carries approximately the same signal as at the end of a 24-bar window. The sweep often shows that `input_chunk_length` matters less than intuition suggests.

**`hidden_dim`** (default: 32) — the size of the LSTM's hidden state and cell state. Larger hidden dimensions give the model more capacity to represent complex patterns. On a 5 000-row training set, `hidden_dim=64` is often worse than `hidden_dim=32` because the extra capacity is consumed by overfitting rather than learning.

**`n_rnn_layers`** (default: 2) — how many stacked LSTM layers. One layer is often enough; the second layer can learn a higher-level representation of the sequence but also doubles the parameter count. Two layers is a reasonable default; three layers rarely helps on datasets this small.

**Early stopping**: training runs for up to `n_epochs` passes through the training data, but stops early if `val_loss` does not improve for `early_stopping_patience` consecutive epochs (default: 5). The model that is saved is the one with the best validation loss, not the last epoch — this matters because validation loss often turns upward well before `n_epochs` is reached, and using the last epoch's weights would mean deploying an overfit model.

The Optuna adaptive hyperparameter search (`make 04_lstm_optuna`) uses Tree-structured Parzen Estimation to select promising configurations and median pruning to kill unpromising trials early. This is more efficient than a manual grid sweep, especially when the hyperparameter space is large. The results are stored in `results/optuna_study.db` — a SQLite database that can be resumed if the search is interrupted and explored interactively via `make 04_lstm_optuna_dashboard`.

## How to interpret the results

The comparison table across all experiments so far:

| | Naive | ARIMA(3,1,3) | XGBoost | **LSTM** |
|---|---|---|---|---|
| Family | last-value | linear, stationary | tabular trees | **recurrent NN** |
| Sees ordered history | no | yes (linear) | engineered lags | **yes (learned)** |
| Past covariates | no | no | engineered scalars | **OHLCV sequences** |
| Parameters | 0 | 7 | ~thousands | ~tens of thousands |
| MAE (USD) | 337.89 | ~337–340 | ~337–340 | *run the experiment* |

The expected outcome is MAE close to the floor, with directional accuracy near 0.50 and Sharpe near zero. This is the honest result on hourly BTC: more model capacity does not conjure signal from a near-i.i.d. process.

If the LSTM does beat XGBoost on MAE, the explanation is likely in the covariates. XGBoost saw `log_volume_{T-1}` and `hl_range_{T-1}` as single scalar features — yesterday's value only. The LSTM sees these as time series over the full input window: not just what volume was last bar, but how volume has been behaving over the past 48 bars. That richer temporal context can sometimes extract signal that a lag-feature snapshot misses.

If the LSTM's directional accuracy or Sharpe beats XGBoost even when MAE does not, the model is finding direction from sequential patterns that tabular trees cannot represent — trajectory shapes rather than point values.

The ablation is instructive: run with `use_volume: false` and `use_ohlc: false` to strip the model down to a univariate LSTM on log-returns only. If performance drops, the covariates were contributing. If performance is unchanged, the LSTM was ignoring them — which is also informative.

### What typically goes wrong

**Overfitting val_loss**: if training loss decreases monotonically but val_loss starts rising after 3–5 epochs, the model is memorizing training patterns. The early-stopping mechanism should catch this, but only if the validation slice is large enough to be representative. A val_fraction of 10% on a 5 000-row training set is 500 rows — borderline adequate.

**input_chunk_length too long**: on a near-random-walk series, asking the LSTM to summarize 96 hours of history creates a long gradient path that is hard to train. The gates help, but they don't fully solve the problem when the relevant signal (if any) is in the last few bars.

**Numerical instability**: `float64` data on Apple Metal Performance Shaders causes errors in some PyTorch kernels. The experiment casts all series to `float32` before training. If the loss becomes NaN, check the data type first.

## Bridging to the Transformer

The LSTM has one fundamental limitation: it compresses all of its history into a single fixed-size hidden state before making the prediction. The hidden state at the end of a 48-bar window is a 32-dimensional vector — the entire past is encoded into those 32 numbers. If the most relevant bar is 40 steps back (say, a large move that set the current volatility regime), the LSTM must have kept that information alive through 39 subsequent forget-gate applications.

The Transformer's attention mechanism sidesteps this entirely. Instead of compressing history into a state and reading it once, attention allows the model to look directly at any past bar when making the prediction. A bar 40 steps back is as directly accessible as a bar 1 step back. Whether that matters for hourly BTC — where the relevant signal is usually local — is exactly the question article 5 answers.
