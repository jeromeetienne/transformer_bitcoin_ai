# Outline — Pretrained Foundation Models

## Opening hook
- TFT collapsed with 30k parameters on 1448 rows
- What if the parameters come pre-trained?
- Models like Chronos-2 and TimesFM trained on millions of time series
- The question: does a model that's never seen Bitcoin have useful intuitions?

## What pretrained foundation models are
- Trained on hundreds of millions of time series from diverse domains
- Chronos-2 (Amazon): 28M or 120M parameters, encoder-only T5-style
- TimesFM 2.5 (Google): 200M parameters, decoder-only patch-transformer
- Zero-shot: no training, weights come from HuggingFace Hub
- Probabilistic output: generate 200 stochastic samples, extract quantiles (q10, q50, q90)

## The setup
- Same Bitcoin data, but longer train window (3833 bars vs. 1448)
- Test window Oct 1 - Dec 1, 2024
- Three variants tested: chronos-small, chronos-large, timesfm
- No training step: fit() is a no-op
- Walk-forward with frozen weights

## The results
- Chronos-2 small (28M): MAE 546.66, dir_acc 0.4754 (below coin flip)
- Chronos-2 large (120M): MAE 560.59 (worse!), dir_acc 0.4344
- TimesFM 2.5 (200M): MAE 570.79, dir_acc 0.4863, Sharpe 4.11
- All three lose to ARIMA(3,1,3) on every metric

## The key insight: bigger is worse
- Chronos-large has 120M parameters vs. chronos-small's 28M
- Large is uniformly worse: +$13.93 on MAE, -0.041 on dir_acc, -1.07 on Sharpe
- The extra 92M parameters of prior don't help on Bitcoin
- Bigger capacity doesn't help when you're not training

## The cross-experiment comparison
- Zero-shot foundation models lose to every non-trivial baseline
- Chronos-small Sharpe 2.97 vs. ARIMA's 6.86 (70% relative loss)
- TimesFM Sharpe 4.11 vs. ARIMA's 6.86 (40% relative loss)
- The pretrained prior alone is not enough

## The honest interpretation
- Transfer learning works when: source domain is similar to target (?)
- Bitcoin is specialized; general time-series prior is general
- The model has an opinion (unlike naive) but it's wrong-on-average
- Dir_acc below coin flip means the prior was trained on problems where uptrends are less common than Bitcoin

## What about Sharpe?
- Chronos-small Sharpe 2.97: positive only because strategy is long/flat on uptrend
- TimesFM Sharpe 4.11: 38% better than chronos-small, still trailing ARIMA
- On symmetric long/short, both would lose money

## Why the foundation model's probabilistic head doesn't help
- The models output full distributions (q10, q50, q90)
- But the strategy only uses the median (q50)
- The distributions are hedged - lower confidence
- RMSE is high: more spread, less precision per bar

## Caveats
- Single backend × size configuration, not a sweep
- Univariate only (no volume, OHLC, sentiment)
- Walk-forward without retraining
- One test regime (post-election rally)
- TimesFM can't use covariates (by design)

## Closing
- Zero-shot transfer doesn't work on this problem
- The general time-series prior was trained on other domains
- Bitcoin's specific structure (volatility clustering, mempool effects, whale behavior) isn't in the prior
- The question: can fine-tuning the prior on Bitcoin data fix this?
