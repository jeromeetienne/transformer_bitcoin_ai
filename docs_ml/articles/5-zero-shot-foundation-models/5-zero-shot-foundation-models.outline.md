# Outline: Zero-Shot Foundation Models

## 1. The transfer learning question
- What "pretrained on millions of series" actually means
- Chronos-2 and TimesFM 2.5: two architectures, two pretraining approaches
- Why zero-shot is the honest first test

## 2. The two models
- Chronos-2: T5-style encoder, trained on diverse time-series corpora
- TimesFM 2.5: decoder-only patch-transformer, 200M parameters
- What they share: a learned distribution over plausible next-step values

## 3. Why univariate only
- TimesFM does not support covariates
- The head-to-head design: isolating the prior, not feature engineering
- What this means for interpreting the results

## 4. Probabilistic forecasting as a bonus
- What QuantileRegression([0.1, 0.5, 0.9]) gives you that a point forecast doesn't
- Reading the shaded band: calibration, not just interval width
- How the median is used for the standard leaderboard metrics

## 5. The experiment setup
- fit() is a no-op; model.historical_forecasts(retrain=False)
- input_chunk_length and the "more context is better" hypothesis
- Sweeping backends and context lengths

## 6. The numbers and what they mean
- Expected result: MAE near naive, directional accuracy near 0.50
- Why this is informative rather than disappointing
- The surprise case: when a pretrained prior transfers to BTC
- Comparing the two backends: Chronos-2 vs. TimesFM on the same slice

## 7. What zero-shot cannot do
- The gap between broad priors and asset-specific calibration
- Setting up article 7: what happens when we let fit() update the weights
