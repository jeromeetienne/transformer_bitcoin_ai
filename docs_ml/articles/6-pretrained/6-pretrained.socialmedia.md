# Social media posts - Pretrained foundation models

## Twitter

🌐 What happens when you point Amazon's Chronos-2 (120M params) at Bitcoin without training?

It loses to a 3-parameter ARIMA on every metric.

Bigger is worse: small Chronos > large Chronos on Sharpe, dir_acc, AND MAE.

Zero-shot is harder than the marketing. 🧵

#FoundationModels #Bitcoin

## Bluesky

Article 6: Zero-shot foundation models on Bitcoin.

Chronos-2 (28M, 120M) and TimesFM 2.5 (200M). Pretrained on hundreds of millions of time series. Never seen Bitcoin.

Best zero-shot Sharpe (TimesFM): 4.11. ARIMA's Sharpe on the same slice: 6.86.

Also: the 120M Chronos is uniformly worse than the 28M Chronos. Extra capacity ≠ extra signal. 🪙

## LinkedIn

Foundation models for time-series forecasting are this year's big story. Amazon's Chronos-2. Google's TimesFM 2.5. Models trained on hundreds of millions of time-series from every domain - electricity, weather, web traffic, stocks, sensors. The pitch is that you point them at a new problem and they just work, no training required.

Article 6 of my Bitcoin ML series points them at Bitcoin.

Three variants, all zero-shot (no training), same 4-hour test window:

- Chronos-2 small (28M parameters): MAE 546.66 USD, dir_acc 0.4754, Sharpe 2.97
- Chronos-2 large (120M parameters): MAE 560.59 USD, dir_acc 0.4344, Sharpe 1.90
- TimesFM 2.5 (200M parameters): MAE 570.79 USD, dir_acc 0.4863, Sharpe 4.11

Every variant **loses to the 7-parameter ARIMA(3,1,3) baseline** (Sharpe 6.86) on every single metric.

Three observations worth knowing:

1. **Bigger Chronos is worse, not better.** The 120M version trails the 28M version on every metric: MAE +13.93 USD, directional accuracy -0.041, Sharpe -1.07. The extra 92 million parameters of pretrained prior actively hurt on this slice. This is the "complexity does not pay" arc, now visible inside a single model family.

2. **Directional accuracy is below coin-flip on both Chronos variants.** The model is wrong about direction more often than right. The prior likely contains more mean-reversion time series than trending ones, so on a strong Bitcoin uptrend the learned expectation is exactly backwards.

3. **TimesFM is the best zero-shot variant on Sharpe but still loses to ARIMA.** A 200-million-parameter decoder-only patch-transformer pretrained on more time-series data than ARIMA has seen still cannot extract the directional signal a 7-parameter linear model finds on Bitcoin alone.

The headline: **zero-shot transfer is real but weak.** The general time-series prior does not contain Bitcoin's specific structure (regime clustering, mempool dynamics, sentiment shocks). The model has an opinion - unlike the naive baseline - but the opinion is often wrong.

The next experiment is the only thing left to try: actually fine-tune the foundation model on Bitcoin and see if that closes the gap.

What is your experience deploying zero-shot foundation models on specialized problems?

#MachineLearning #FoundationModels #Chronos #TimesFM #Bitcoin #TimeSeries
