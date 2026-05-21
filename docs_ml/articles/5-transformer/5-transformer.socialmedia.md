# Social media posts - Transformer

## Twitter

⚡ Transformer on Bitcoin.

Attention, variable selection, gated residuals, future covariates. The most sophisticated architecture in the series.

MAE: $813. That is **57% worse than predicting "no change"**.

Attention is not magic. 🧵

#Transformer #Bitcoin

## Bluesky

Article 5: Temporal Fusion Transformer on Bitcoin.

Tens of thousands of parameters. Attention. Variable selection networks. Gated residuals. Future covariates.

MAE: $813.10. That is 57% worse than the naive baseline. Worst in the leaderboard.

Sharpe 5.87 (decent), but directional accuracy 0.4913 (below coin flip again). 1,448 training rows cannot amortize that architecture. 📉

## LinkedIn

The Temporal Fusion Transformer is supposed to be the time-series machine-learning answer to BERT. Attention. Variable selection networks. Gated residual networks. Past and future covariate channels. The architecture is meant to extract patterns the LSTM cannot reach.

I ran it on the same 4-hour Bitcoin data, with the same train/test split as every other model in this series.

Results:

- MAE: 813.10 USD - **57% worse** than the naive baseline at 518.36
- RMSE: 1,090.78 USD - 39% worse than naive
- Directional accuracy: 0.4913 - below a coin flip
- Annualized Sharpe: 5.87 (per-bar significance ~2.51 σ, borderline)

This is the worst MAE in the entire 4h leaderboard. 51% worse than the next-worst trained model. The TFT is wrong about magnitude by a lot - and is also wrong about direction more often than right.

Why? Parameter-to-example ratio. TFT has approximately 30,000+ trainable parameters across variable selection networks, an LSTM encoder/decoder, multi-head attention (4 heads), and gated residual networks. The training set is 1,448 bars. 20:1 parameters per example. The model has more capacity to fit noise than there is signal to fit.

The architecture's promise - that attention solves what recurrence cannot - is irrelevant when there is not enough data to estimate the attention weights. The model overfits, produces wild forecasts (RMSE 1,090 vs. ARIMA's 782), and reports its disagreement with reality at full volume.

The Sharpe of 5.87 is the strangest part. It is decent because on a strongly trending slice, even a model that is wrong about magnitude will sometimes predict "up" loud enough to stay long during the rally. **Sharpe can be propped up by directional aggression even when the underlying forecast is broken.**

The honest lesson: architectural complexity without sufficient data is self-defeating. The transformer is only as good as the data you feed it.

When have you seen a fancy architecture lose to a much simpler one?

#MachineLearning #Transformer #DeepLearning #TimeSeries #Bitcoin
