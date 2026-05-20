# Social Media Posts — Article 0: Can Machine Learning Predict Bitcoin?

## Twitter

**Thread: The honest truth about predicting Bitcoin with ML**

1/ I ran 7 machine learning models on Bitcoin price data — from zero-parameter baselines to fine-tuned foundation models.

Here's what I found (and what most articles won't tell you): 🧵

2/ The target: predict the next closing price, one step ahead, no lookahead, no shuffling.

OHLCV only. No order book. No sentiment. If the signal isn't in price and volume, we need to know that first.

3/ The models, in order of complexity:
- Naive last-value predictor
- ARIMA
- XGBoost + feature engineering
- LSTM
- Transformer (Temporal Fusion Transformer)
- Zero-shot foundation models (Chronos-2, TimesFM 2.5)
- Fine-tuned foundation models

4/ Every model is benchmarked against the same naive floor on the same test slice.

No cherry-picking. No 500-run grid searches. If a model underperforms, the article says so.

5/ This is a reproducible ML study — not a trading system, not a leaderboard chase.

The GitHub repo has every config, every run script, every result.

Full series → [link]

---

## Bluesky

**Can machine learning predict Bitcoin prices?**

I ran 7 models — from a zero-parameter naive predictor to fine-tuned foundation models — on the same dataset, with the same honest evaluation.

No lookahead. No shuffled splits. No cherry-picked results.

Each model is measured against the dumbest possible floor: predict that tomorrow's price equals today's.

Spoiler: that floor is harder to beat than you'd expect.

Full reproducible study across 7 articles → [link]

---

## LinkedIn

**I spent months running machine learning models on Bitcoin price data. Here's what I actually found.**

Most ML-on-crypto articles show you the best result from 500 hyperparameter runs and call it science.

This series does the opposite.

The setup:
- One question: given OHLCV data up to bar *t*, predict the closing price of bar *t+1*
- One dataset: BTCUSDT from Binance, 2024, same train/test split for every experiment
- One rule: every model is measured against the naive last-value predictor on the same held-out test slice

Seven models, from zero parameters to fine-tuned foundation models. When a model fails to beat the naive baseline, the article says so — and explains why.

The first result is already uncomfortable: the naive predictor is hard to beat on hourly Bitcoin. That's not a bug in the methodology. It's a fact about the signal-to-noise ratio.

If you work in quantitative finance, applied ML, or time-series forecasting — or if you're just curious about what "honest benchmarking" looks like in practice — this series is for you.

Article 0 (the setup) is live: [link]
