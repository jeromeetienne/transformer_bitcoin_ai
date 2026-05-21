# When Attention Fails: The Transformer That Couldn't

The LSTM's failure was disappointing but understandable. Recurrent networks compress history into a hidden state - a bottleneck. If the hidden state doesn't capture what matters, later layers have no way to recover. The LSTM on 1,448 rows couldn't learn enough to avoid overfitting.

The Transformer was supposed to fix that. Instead of a bottleneck, use *attention* - let every step of the sequence look at every other step directly. Attend to the past bars that actually matter, ignore the rest. Add variable selection networks to learn which inputs are relevant. Add gated residual connections for feature mixing. Add cyclical time features so the model knows what time of day and day of week it is.

The result? The worst point error in the leaderboard. By 57%.

## The Temporal Fusion Transformer

The TFT is a Transformer specifically designed for time-series forecasting. It's structured differently from language models:

- **Encoder (past):** LSTM that processes the input window (48 bars of history)
- **Decoder (future):** Another LSTM that predicts forward
- **Attention:** Multi-head self-attention across the encoded sequence
- **Variable selection:** Networks that learn which inputs matter at each step
- **Gated residuals:** Adaptive feature mixing across the entire architecture

The key difference from the LSTM: attention. The encoder LSTM outputs a sequence of 48 hidden vectors (one per bar). Attention lets the decoder look at all 48 simultaneously, with learned weights on each. A small bar from 24 bars ago might be relevant; a large bar from 5 bars ago might be noise. The attention mechanism learns which.

Future covariates: hour-of-day (sin/cos encoded) and day-of-week (sin/cos encoded). These are deterministic - you know the hour and day when making a prediction. An LSTM can't use them (it only knows the time implicitly from the bar sequence). A Transformer's attention can: "when it's 8 AM on Tuesday, these patterns apply."

## The architecture

```
input [48 bars + covariates]
         │
         ├─ Variable selection networks ──┐
         │                               ├─ LSTM encoder
         └───────────────────────────────┘
                      │
              Multi-head self-attention
                      │
         ┌────────────┴────────────┐
         │                         │
    LSTM decoder            Gated residuals
         │                         │
         └────────────┬────────────┘
                      │
              Linear regression head
                      │
              log-return prediction
```

Training: 30 epochs, early stopping at 5 epochs without improvement, validation loss monitored. Same three-way split as LSTM (1,448 train, 160 val, 401 test).

## The results

| Metric | Value |
| --- | --- |
| **MAE** | **813.10 USD** |
| **RMSE** | 1,090.78 USD |
| MAPE | 1.0786 % |
| Directional accuracy | 0.4913 |
| Cumulative return | 44.68 % |
| Annualized Sharpe | 5.8675 |

Put this in context:

| Model | MAE |
| --- | --- |
| 01_baseline | 518.36 |
| 02_arima | 517.16 |
| 03_xgboost | 539.39 |
| 04_lstm | 522.29 |
| **05_transformer** | **813.10** |

The TFT is wrong by $813.10 on average. Compared to the naive baseline at $518.36, that's **57% worse**. Compared to the next-worst trained model (XGBoost at 539.39), it's **51% worse**.

This is not a small difference. This is a failure mode.

## Why catastrophic failure

The Temporal Fusion Transformer has approximately 30,000+ parameters across:

- Variable selection networks (thousands of parameters)
- Two LSTM layers (thousands each)
- Multi-head attention (thousands)
- Gated residual networks (thousands)
- Projection layers (thousands)

On 1,448 training rows, the model has a parameter-to-example ratio of ~20:1. For comparison:

- ARIMA: 3 parameters / 1,608 examples = 0.002:1
- XGBoost with 31 features: ~100 parameters (tree leaves vary) / 1,588 examples = 0.063:1
- LSTM: ~1,000 parameters / 1,448 examples = 0.69:1
- TFT: ~30,000 parameters / 1,448 examples = 20:1

The TFT doesn't have enough examples to learn a stable estimate of its parameters. Early stopping on validation loss helps, but it doesn't solve the core problem: there's more capacity than data.

The model overfits catastrophically. It learns spurious patterns in the 1,448-bar training set that don't generalize. When it sees the test set, its learned patterns fail. The resulting predictions are huge swings - hence the 1,090.78 RMSE (outliers are enormous relative to MAE).

## A passing grade on Sharpe, despite the failure

Sharpe 5.8675 is not great (ARIMA is 6.0569, XGBoost is 6.4233) but it's not terrible either. Per-bar Sharpe = 5.8675 / √2190 ≈ 0.1254. Standard error on a 401-bar slice ≈ 1 / √401 ≈ 0.0499. Ratio ≈ 2.51 σ - borderline significant.

Why is Sharpe acceptable when MAE is a disaster?

The TFT expresses enough directional confidence (even though it's wrong on direction 51% of the time) that the long/flat strategy generates reasonable returns on an uptrending slice. On the post-election Bitcoin rally, being long sometimes and correctly capturing the uptrend buoys the numbers even when point predictions are wildly off.

This is the paradox: **a model can be catastrophically wrong on magnitude but right enough on direction (in the right regime) to generate money.**

Sharpe doesn't tell you that the magnitude is broken. It just tells you that the directional positions, sized by the predicted magnitude, beat the spread on this slice.

## Directional accuracy: below coin flip again

Dir_acc 0.4913 is below 0.5. Like the LSTM, the TFT is wrong on direction more often than right. The attention mechanism, variable selection networks, and gated residuals all failed to extract the directional signal that AR(1) captures with a single coefficient.

The future covariates (hour-of-day, day-of-week) didn't help. These are structural features that should give the model temporal context. Apparently, 30,000 parameters aren't enough to learn their relationships on 1,448 rows.

## What the sweep would show (but can't, because it's stale)

A sweep of 8 configurations exists on disk, but it was run on the old 1h data slice. The MAE figures (373-420 USD range) are on a different scale. You can't compare 373 on 1h data to 813 on 4h data. The test-bar count is different, the regime is different, everything is different.

A sweep on the current 4h slice would show: does a smaller TFT (fewer attention heads, smaller hidden size) avoid the catastrophic failure? Almost certainly yes - the sweep would show a tradeoff between capacity and generalization. But that sweep doesn't exist on disk yet.

## The data problem, illustrated

Here's the structure of the pattern:

- On 3,000+ rows: deep learning with attention can extract patterns XGBoost can't
- On 1,500 rows: deep learning with attention overfits and collapses
- On 1,500 rows with pre-training: deep learning can leverage the prior, partly recovering

The third option is article 6. But this article is about the raw TFT on Bitcoin-only data. And on that data, at that size, it fails.

## Caveats

Single seed, neural network variance not captured.

Frozen weights across the test window - the model is "100 bars stale" by the time it reaches the end of the 401-bar test slice.

Future covariates are calendar-only. The model doesn't know about market regime changes, volatility spikes, or Bitcoin-specific catalysts that happen in the test period. With only cyclical time features, the model is predicting on information it can't distinguish one day from another.

Attention weights are not interpretable as causation. A bar the model attends to didn't *cause* the prediction; the model just used that input strongly to compute it. [Deep learning interpretability](https://transformer-circuits.pub/) is hard.

## Closing

The Temporal Fusion Transformer is the most dramatic failure in the series so far. 57% worse on MAE than the naive baseline. Below coin flip on direction.

The lesson: **architectural complexity without sufficient data is self-defeating.** Attention is not magic. Variable selection networks don't matter if you don't have data to select from. Gated residuals can't generalize when there's nothing to generalize.

The pattern is now clear:
- Small capacity + small data = good (ARIMA)
- Medium capacity + small data = worse (XGBoost on MAE, LSTM)
- Large capacity + small data = catastrophic (TFT)

What if we invert the problem? Instead of small data, use *pre-training*. Take a model trained on millions of time series from other domains and ask: does the prior transfer to Bitcoin?

That's next.

## How to reproduce

```bash
make 05_transformer
make 05_transformer_sweep  # note: sweep is stale on 1h
```

Results in [experiments/05_transformer/results/btc_4h_2024/](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/05_transformer/results/btc_4h_2024). Predictions in [predictions.parquet](https://github.com/jeromeetienne/transformer_bitcoin_ai/blob/HEAD/experiments/05_transformer/results/btc_4h_2024/predictions.parquet).
