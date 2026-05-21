# The First Deep Learning Failure: Why LSTMs Lose on This Data

Up until now, the models have been transparent. ARIMA is three parameters you can inspect. XGBoost is a bunch of decision trees you can walk through. When they succeed, you understand why. When they fail (like XGBoost on point error), you can see the mechanism: overfitting noise.

Now we enter deep learning. LSTMs are recurrent neural networks - a more complex creature. They carry hidden state across time, learning long-range patterns through thousands of parameters. The pitch is that they can learn *sequences* in ways that flat feature vectors can't.

The reality? On 1,448 training bars, the LSTM doesn't just lose. It breaks. It gets directional accuracy *below a coin flip*. It's the first model in the series that clearly fails - not by a little, by a lot.

## What an LSTM is

An LSTM (Long Short-Term Memory) cell is a gating mechanism that learns what to remember and what to forget as it processes a sequence. At each time step, it takes the current input and the hidden state from the previous step, and outputs a new hidden state that captures "what matters about the history so far."

```
h[t] = LSTM(x[t], h[t-1])
```

The LSTM cell has three gates - forget, input, and output - that control what information flows. This solves a problem that simpler recurrent networks have: the vanishing gradient. If you have a 48-bar input window, gradients can propagate backwards through all 48 time steps, giving the early bars a voice in learning. Simpler RNNs forget what happened too far back.

In practice, this means an LSTM can learn that "bar 24 bars ago affects the next bar," or "volatility clusters persist across days." It's a learner of temporal patterns.

## The architecture

The experiment stacks two LSTM layers, each with 32 hidden units. Think of it as two recurrent networks, one feeding into the next. The first layer processes the 48-bar input window and outputs a sequence of hidden states. The second layer processes those hidden states and outputs a final encoding. A linear head on top regresses from that encoding to the next bar's log-return.

```
input [48 bars]
  │
  ├─ LSTM layer 1 (32 units) ─┐
  │                           ├─ LSTM layer 2 (32 units) ─ Linear head ─ log-return pred
  └───────────────────────────┘
        (dropout 0.1)
```

Training uses PyTorch Lightning, Adam optimizer, 30 epochs max with early stopping when validation loss plateaus after 5 epochs without improvement.

Covariates: volume and OHLC (same as XGBoost), giving the LSTM real structure to work with, not just close prices.

Three-way split: 1,448 training bars, 160 validation bars (for early stopping), 401 test bars.

## The results

| Metric | Value |
| --- | --- |
| MAE | 522.29 USD |
| RMSE | 785.45 USD |
| MAPE | 0.6761 % |
| **Directional accuracy** | **0.4938** |
| Cumulative return | 0.2596 |
| Annualized Sharpe | 4.2660 |

MAE is $522.29 - worse than both baselines (naive 518.36, ARIMA 517.16). Better than XGBoost (539.39), but worse than XGBoost is not a compliment.

But look at directional accuracy: **0.4938**. Below 0.5. The LSTM is *wrong more often than right* on direction.

This is the critical failure. ARIMA with three parameters gets direction right 55.47% of the time. XGBoost with 31 features gets it right 53.65%. The LSTM with thousands of parameters gets it right less than 50%. The model is actively *worse* than a coin flip.

Sharpe 4.2660 is the lowest among trained models. Per-bar: 4.2660 / √2190 ≈ 0.0912 against SE ≈ 0.0499 gives ratio ≈ 1.83 σ - weakly significant, close to noise.

## The sweep: smaller is better

The experiment swept 6 configurations on the same 4h data:

| Config | icl | hidden | layers | dropout | MAE | dir_acc | Sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smallest | 24 | 16 | 1 | 0.0 | 553.34 | 0.5137 | **5.0295** |
| default | 48 | 32 | 2 | 0.1 | 522.29 | 0.4938 | 4.2660 |
| largest | 96 | 64 | 3 | 0.2 | **517.29** | **0.5362** | 5.0096 |

The tiny LSTM (24-bar input, 16 hidden units, 1 layer, no dropout) wins on Sharpe at 5.03. The huge LSTM (96-bar input, 64 hidden units, 3 layers, 0.2 dropout) wins on MAE and directional accuracy. The configured default is in between and wins neither.

But notice: even the *best* configuration in the sweep (largest, MAE 517.29) barely ties ARIMA(1,1,1) on MAE (517.16). And its dir_acc 0.5362 is still below ARIMA's 0.5547.

The smallest LSTM shows the pattern: dropping capacity improves Sharpe. On this signal-to-noise ratio, *more parameters make the model worse*.

## One row produces NaN Sharpe

The `(48, 64, 2, 0.2)` configuration produces NaN Sharpe. Why? Because the strategy went entirely flat. The LSTM predicted `close_pred[t] <= reference_price[t-1]` on every test bar, so the long/flat rule never triggered a long position. Zero returns means zero standard deviation means NaN Sharpe.

This is a revealing failure mode. The model trained, converged, produced forecasts - but the forecasts never expressed a "go long" opinion. On a strongly uptrending slice (the post-election rally), a model that never goes long loses all the gains.

## Why the LSTM fails

With 1,448 training rows and parameters in the thousands, the LSTM has more capacity to fit noise than signal exists in the training data. The early stopping mechanism monitors validation loss. But validation loss doesn't rank the trading metrics.

A configuration with lower val_loss might have worse Sharpe because it's more confident in the wrong direction. The model optimizes for point loss (MSE) during training, but we care about Sharpe and direction during testing.

The LSTM learns patterns in the 1,448-bar training window that don't generalize. It fits the noise, washes out the signal, and stumbles into a regime (the test slice) where its learned patterns don't apply.

This is the deepest lesson of the three-model sequence so far:

| Model | Capacity | Data | On point error | On direction |
| --- | --- | --- | --- | --- |
| ARIMA | 3 | 1,608 | Good | Good |
| XGBoost | 31 features | 1,588 | Bad | OK |
| LSTM | 1000s parameters | 1,448 | Bad | Bad |

Capacity grows. Data shrinks (val/test split). Performance cascades.

## The frozen-weight assumption

The LSTM trains once on the training set, then predicts the entire test slice with frozen weights. It's not updated as it sees new bars in the test window. This is realistic - you deploy a model and it runs - but it's also conservative. After 100 test bars, the coefficients are "100 bars stale."

A rolling-window approach would retrain the LSTM every N bars, updating its view as new data arrives. That might help. But the experiment doesn't do that - it freezes weights for the walk-forward.

## Caveats

This is a single seed (`random_state: 42`). Neural network training has run-to-run variance from hardware non-determinism (GPU / MPS floating-point operations aren't bit-deterministic). Three to five seed runs would tighten the variance estimate.

Early stopping with 5-epoch patience might be too aggressive - the model stops as soon as val_loss stops improving for 5 epochs. A longer patience or a different monitor (like val_Sharpe, if you could compute it) might give the model more epochs to learn.

No future covariates. The next article (Transformer) adds cyclical time features (hour-of-day, day-of-week). These are deterministic and can be used at prediction time. The LSTM only sees past covariates.

## Closing

The LSTM is the first model to clearly fail. Not just on one metric - on the primary metric we care about (Sharpe) and on direction. The smaller LSTM (5.03 Sharpe) would have been better than the configured default (4.27 Sharpe), but even that loses to the 3-parameter ARIMA (6.06 Sharpe).

The lesson: **on 1,448 rows of data, adding thousands of parameters doesn't buy you signal. It buys you overfitting.**

Next comes the Transformer - an even more complex architecture with even more parameters. It will face the same challenge: can attention mechanisms extract patterns that LSTMs can't, or does the capacity problem only get worse?

## How to reproduce

```bash
make 04_lstm
make 04_lstm_sweep
```

Results in [experiments/04_lstm/results/btc_4h_2024/](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/04_lstm/results/btc_4h_2024). Sweep rankings in [sweep.csv](https://github.com/jeromeetienne/transformer_bitcoin_ai/blob/HEAD/experiments/04_lstm/results/btc_4h_2024/sweep.csv).
