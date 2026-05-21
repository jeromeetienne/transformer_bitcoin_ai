# When a Model Trained on Everything Fails on Bitcoin

After the Temporal Fusion Transformer's collapse, the pattern was clear: on 1,448 training bars, adding parameters doesn't help. You need data proportional to capacity, or you fit noise.

But what if you don't train at all? What if you take a model already trained on *hundreds of millions of time series* from other domains and ask it to predict Bitcoin without changing a single weight?

Amazon's Chronos-2 and Google's TimesFM 2.5 are foundation models for time-series forecasting. They were pre-trained on a corpus of public time-series data (electricity demand, stock prices, weather, web traffic, traffic volume, sensor data, everything). They learned the deep structure of how time-series work in general.

The promise: transfer learning. The model has a learned prior about temporal patterns. It's never seen Bitcoin, but it understands time-series.

The reality: it's worse than ARIMA on every metric.

## The foundation models

**Chronos-2 (Amazon):** An encoder-only Transformer trained on time-series data. Two parameter sizes: 28 million (small) and 120 million (large). The small version is meant for edge deployment; the large is meant for harder problems.

**TimesFM 2.5 (Google):** A decoder-only patch-transformer. 200 million parameters. Partitions the time-series into overlapping patches and treats them as tokens - a different architectural flavor from Chronos.

Both are probabilistic. Instead of outputting a single point forecast, they generate 200 stochastic samples from their learned distribution. You extract quantiles (q10 / q50 / q90) and use the median (q50) for point forecasts.

Both are zero-shot: no training. Weights are frozen. You call `fit()` and it's a no-op. The models generate forecasts using only the prior learned on their pre-training corpus.

## The experimental setup

Unlike the smaller models (ARIMA, XGBoost, LSTM), these foundation models work with longer context. The test window is the same (October 1 - December 1, 2024, 366 bars), but the train window is longer: 2024-01-01 to 2024-08-01, 3,833 bars.

The rationale: foundation models are robust to scale. Giving them more history to condition on, even if they don't "train," gives them more data to look at when making forecasts.

Three variants on the same test window:

- `chronos-2-small` (28 M parameters)
- `chronos-2-large` (120 M parameters)
- `timesfm-2.5` (200 M parameters)

## The results

| Model | MAE | RMSE | MAPE | dir_acc | Sharpe |
| --- | --- | --- | --- | --- | --- |
| ARIMA(3,1,3) | 539.15 | 808.79 | 0.6903 % | 0.5082 | 6.8559 |
| chronos-2-small | 546.66 | 817.81 | 0.7003 % | 0.4754 | 2.9722 |
| chronos-2-large | 560.59 | 837.11 | 0.7159 % | 0.4344 | 1.9004 |
| timesfm-2.5 | 570.79 | 843.75 | 0.7309 % | 0.4863 | 4.1080 |

All three lose to ARIMA on every single metric. ARIMA has 7 parameters; these have 28 million to 200 million.

Sharpe tells the story: ARIMA 6.86 vs. chronos-small 2.97. A 70% relative loss. The pretrained prior doesn't transfer.

## The paradox: bigger is worse

Chronos-2 comes in two sizes. The small one (28 M) should be the fast, edge-computation version. The large one (120 M) should handle harder problems.

On Bitcoin, the large one is uniformly worse:

| Model | MAE | dir_acc | Sharpe |
| --- | --- | --- | --- |
| chronos-small | 546.66 | 0.4754 | 2.9722 |
| chronos-large | 560.59 | 0.4344 | 1.9004 |

The difference: +$13.93 on MAE, -0.041 on directional accuracy, -1.07 on Sharpe.

The extra 92 million parameters don't help. They might even hurt. The larger model is overfitting to its pre-training corpus in ways that don't generalize to Bitcoin.

This echoes the pattern from the custom models: on a given problem, more capacity is only good if you have signal to justify it. The large Chronos was trained on diverse time series; it learned generalizations that mostly hold. But on Bitcoin, those generalizations are worse than the small version's.

## TimesFM's better Sharpe

TimesFM is a different architecture (decoder-only patch-transformer). On Sharpe, it outperforms both Chronos variants: 4.11 vs. 2.97 (chronos-small) and 1.90 (chronos-large).

That's a 38% Sharpe lift over chronos-small. But it's still 40% *below* ARIMA.

Why does TimesFM do better? The decoder-only architecture has a different inductive bias. It processes time-series as a sequence of patches, not as a full sequence. This might generalize better to Bitcoin. Or it might just be luck on this slice.

Either way, the headline is: **even the best pretrained model (TimesFM, Sharpe 4.11) loses to the 7-parameter ARIMA baseline (Sharpe 6.86).**

## Why the prior doesn't transfer

A pretrained model is only as good as the diversity of its pre-training corpus. Chronos and TimesFM were trained on time series from many domains: stock prices, electricity demand, web traffic, traffic volume, economic indicators, sensor data.

But Bitcoin is a specialized asset class. It has patterns that the general time-series corpus doesn't emphasize:

- **Directional regime clustering:** Bitcoin trends last weeks; it doesn't revert to a mean the way electricity demand or web traffic does.
- **Mempool effects:** Transaction queue dynamics, fee markets, and network congestion drive short-term price.
- **Whale behavior:** Large holders moving coins create predictable micro-patterns.
- **Sentiment shocks:** News, regulatory moves, or social media events cause regime shifts.

The pretrained prior learned the general structure (time-series are autocorrelated, volatility clusters, trends exist). But it learned to be *conservative* - to not bet too hard on any one regime because the corpus contains counter-examples from every regime.

On Bitcoin, in a strong uptrend (post-election 2024), that conservatism is wrong. The model learned "uptrends don't always continue" and hedges. ARIMA learned "recent changes predict next change" and commits.

## Directional accuracy below coin flip

Both Chronos variants produce dir_acc below 0.5. The models are *wrong more often than right* on direction.

This is a hint about the pre-training corpus. It might have contained more mean-reversion time series (stock volatility clusters, electricity demand oscillates, weather cycles) than trending time series. The prior was trained to expect reversion.

Bitcoin in the post-election regime is pure trend. The prior's learned expectation of reversion is exactly backwards.

## The probabilistic output (and why it doesn't help)

Both models generate 200 stochastic samples from their learned distribution. The median (q50) is used for the point forecast. But the distribution has q10 and q90 quantiles too - uncertainty bands.

These bands are *wider* than a simple point forecast would suggest. The model hedges. It's more uncertain, more conservative. This shows up as higher RMSE (outliers get weighted extra in RMSE calculation) and lower point precision.

The probabilistic head is a feature in other domains (where calibrated uncertainty is valuable). On this problem, where you care about point error and Sharpe, it's a liability.

## Caveats

Single seed per variant (no variance estimate). Walk-forward without retraining. Univariate only (TimesFM can't accept covariates by design; Chronos can but wasn't given them here, to keep the comparison fair). Test on one regime (post-election rally). A different regime (crash, sideways market) might flip the rankings.

## Closing

Zero-shot transfer doesn't work. The pretrained prior is general knowledge about time-series; Bitcoin is specific. The model that learned on everything loses to the model that learned ARIMA on Bitcoin specifically.

The honest reading: **a model trained on general time-series has a different belief about how the world works than what Bitcoin actually does.** On this slice, ARIMA's simpler model of momentum and mean reversion is more aligned with reality than the foundation model's learned priors.

The remaining question: can fine-tuning fix this? What if you take the 28-million-parameter Chronos and update its weights on Bitcoin data - not from scratch, but starting from the general prior?

That's the final experiment.

## How to reproduce

```bash
make 06_pretrained
# Or individual variants:
make 06_pretrained CONFIG=experiments/06_pretrained/configs/btc_4h_2024.chronos-large.config.yaml
make 06_pretrained CONFIG=experiments/06_pretrained/configs/btc_4h_2024.timesfm.config.yaml
```

Results in [experiments/06_pretrained/results/](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/06_pretrained/results).
