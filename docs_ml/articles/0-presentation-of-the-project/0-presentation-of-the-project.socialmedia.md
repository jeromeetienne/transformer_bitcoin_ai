# Social Media Posts — Article 0: Can Machine Learning Predict Bitcoin?

## Twitter

**Thread: I ran 7 ML models on Bitcoin. Here's the honest version. 🧵**

1/ 🤖 I spent months running ML models on Bitcoin price data — from a zero-parameter naive predictor to fine-tuned foundation models trained on hundreds of millions of time series.

One question. Seven models. Every result published. 📊

2/ 🎯 The target: given OHLCV data up to bar *t*, predict the closing price of bar *t+1*.

No order book. No sentiment. If a model can't extract signal from raw price and volume, we need to know that first.

3/ 📏 The rules:
✅ Strict chronological train/test split — no shuffling
✅ Every model evaluated on the same 437 test bars
✅ Every model measured against the same naive floor
✅ When a model underperforms, the article says so

4/ 🏗️ The lineup:
• Naive last-value predictor (0 params)
• ARIMA
• XGBoost + feature engineering
• LSTM
• Temporal Fusion Transformer
• Zero-shot foundation models
• Fine-tuned foundation models

5/ 🔬 This isn't a leaderboard chase. It's a reproducible ML study.

Every experiment is a self-contained directory with a config, a run script, and a results folder. `make baseline` gives you the same number you'll read in the article. ⚙️

Full series → [link]

#MachineLearning #Bitcoin

---

## Bluesky

🤖 **I ran 7 ML models on Bitcoin and published every result — including the ones that failed.**

One question: given OHLCV up to bar *t*, predict the closing price of bar *t+1*. No order book, no sentiment, no shuffled splits, no cherry-picked runs. 🚫

The models go from a zero-parameter naive predictor to fine-tuned foundation models. Each one is measured against the same naive floor on the same 437 held-out test bars. 📏

The naive floor is harder to beat than it has any right to be. That's where the series begins. 👇

Full series → [link]

#MachineLearning #TimeSeries

---

## LinkedIn

🎯 **Most ML-on-crypto articles show you the best run from 500 hyperparameter searches. This series does the opposite.**

I spent months running machine learning models on Bitcoin price data — seven models, from a zero-parameter naive predictor to fine-tuned foundation models trained on hundreds of millions of time series. Every result is published, including the ones that aren't flattering. 📊

**🏗️ The setup:**
• One question: given OHLCV data up to bar *t*, predict the closing price of bar *t+1*
• One dataset: BTCUSDT from Binance, full 2024 calendar year, strict chronological split
• One rule: every model is measured against the naive last-value predictor on the same held-out test slice — and when a model fails to beat it, the article says so

**🔄 The model progression isn't just about complexity.** Each step changes one specific thing about the inductive bias: linear → nonlinear, hand-crafted features → learned representations, task-specific training → general pretraining. Understanding *what changed* at each step is the real purpose of the series — not the final leaderboard number.

⚙️ This is a reproducible ML study, not a trading system. Every experiment is a self-contained directory: config file, run script, results folder. The same `make` command that produced the numbers in the article will produce the same numbers on your machine.

If you work in applied ML, quantitative finance, or time-series forecasting — this is built for you. 👇

Article 0 (the setup) is live → [link]

#MachineLearning #TimeSeries #Bitcoin
