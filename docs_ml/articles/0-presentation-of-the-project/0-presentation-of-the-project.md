# Can Machine Learning Predict Bitcoin? Here Is What Actually Happened

You can buy Bitcoin at 3 a.m. on a Sunday. You can sell it during a holiday, a war, a bank run. No market maker goes home for the weekend; no circuit breaker pauses the tape. The result is a dataset that is simultaneously one of the richest and noisiest in all of quantitative finance - twenty-four hours a day, seven days a week, with tick-level liquidity and volatility that would make an equity trader nervous.

That combination makes Bitcoin a genuinely interesting playground for machine learning forecasting. Not because machine learning is going to unlock hidden alpha (it might not), but because the problem is hard in all the ways that reveal whether a modeling approach actually works. This series of seven articles follows a single question from the simplest possible answer to the most sophisticated one available today, and measures every step honestly against the one before it.

## The question

The target is simple: given everything we know up to the close of bar *t*, what is the closing price of bar *t+1*?

One-step-ahead. Point forecast. Closing price in USD.

This is a deliberate choice. Multi-step forecasting compounds errors in ways that obscure whether the model learned anything at the single-step level. Predicting returns instead of prices introduces its own distributional assumptions. Predicting log-returns is cleaner statistically but makes the numbers harder to interpret at a glance. The closing price is the number everyone looks at; it is the number the exchange clears; it is the right target for a first study.

The "given everything we know" part is stricter than it sounds. In a live system you might feed in order-book depth, funding rates, social sentiment, or on-chain metrics. Here, we restrict the feature set to OHLCV - open, high, low, close, volume - from a single pair. This is not a limitation of the infrastructure (the data loader is designed to accept additional series), it is a methodological choice: if a model cannot extract signal from raw price and volume, no amount of feature engineering will save it.

## The data

All experiments draw from the same source: historical OHLCV data for `BTCUSDT` from Binance, loaded via a shared data loader in [src/btc_ai/data](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/src/btc_ai/data). The loader handles caching, normalization, and the train/test split so that every experiment operates on exactly the same bytes.

Two granularities appear across the series:

- **1-hour bars** - the primary granularity for articles 1 and 2 (naive baseline and ARIMA), where the classical statistical models live. Hourly resolution gives roughly 8 760 bars per year, which is enough data for ARIMA but not enough to make a deep learning model comfortable.
- **4-hour bars** - used from article 3 onward. The 4-hour bar smooths intra-day noise that neither gradient boosting nor neural networks can usefully model, and it quadruples the effective number of "events" (trend changes, volatility clusters, weekend gaps) relative to the data volume.

The default study window runs from 2024-01-01 through 2024-12-31. That is one calendar year, which spans a Bitcoin halving (April 2024), a new all-time high (November 2024), and the full range of volatility regimes that characterize a normal BTC year. The train/test split holds out the **last 20%** of the series as the test set. On the 1-hour config that is approximately 437 bars (~18 days). On the 4-hour config it is approximately 219 bars (~36 days).

## The evaluation philosophy

The single most important methodological decision in time-series forecasting is also the easiest to get wrong: **do not shuffle the data**.

In image classification, shuffling the training set is harmless. In time-series, it is a form of cheating. A model that has seen tomorrow's price during training will appear to forecast brilliantly, and it will fail immediately in production. Every experiment in this series uses a strict time-ordered split: training observations precede test observations, with no overlap and no lookahead.

Walk-forward evaluation enforces the same discipline at inference time. Each prediction at bar *t* is computed from the *actual* values of bars *t-1*, *t-2*, …, not from previously predicted values. This is the only evaluation mode that corresponds to what you would actually experience running a model live. It is slower and less flattering than one-shot forecasting, and that is exactly the point.

Four metrics are reported for every experiment:

- **MAE** (formula: mean |actual − pred|): the average miss in USD. The primary comparison metric across experiments.
- **RMSE** (formula: sqrt(mean(actual − pred)²)): like MAE but penalizes large misses harder. The gap between RMSE and MAE reveals how spiky the errors are.
- **MAPE** (formula: mean |actual − pred| / actual): percentage error. Makes the absolute number interpretable across different price levels.
- **Directional accuracy** (formula: fraction of bars where sign(pred − prev) == sign(actual − prev)): did the model predict the *direction* correctly? 0.5 is random. Anything above 0.5 is evidence of real signal.

Directional accuracy deserves a special note. A model can have an MAE close to the naive baseline - meaning its point forecasts are not more accurate in dollar terms - and still be useful for trading, because it reliably predicts *which way* the market will move. The inverse is also possible: a model with low MAE that has no directional skill is no better than knowing where you started. Both numbers matter; neither alone is sufficient.

## The model lineup

The eight articles in this series trace a deliberate path through the model landscape:

**Article 0** (this one) - the setup. No model, no results. Just the question, the data, and the rules.

**Article 1: Baseline** - predict that the next price equals the current price. Zero parameters, zero learning. This sets the floor that every subsequent model must beat. It also establishes the full experiment architecture: data loader, config schema, metrics module, and results layout that all later experiments inherit without modification.

**Article 2: ARIMA** - the classical statistical companion. Three parameters (p, d, q), estimated from training data. ARIMA is the first model that actually learns from the series. If it cannot beat naive, the signal-to-noise ratio at this timescale is telling us something important.

**Article 3: XGBoost** - gradient-boosted trees with hand-crafted features. The model is generic; the features carry the signal. Lag features, rolling statistics, calendar effects. The article is as much about feature engineering as about the model itself.

**Article 4: LSTM** - the recurrent neural network baseline for deep learning. Takes raw sequences as input, learns to weight history without explicit feature engineering. Introduces questions of sequence length, training stability, and what exactly a recurrent cell remembers.

**Article 5: Transformer** - attention applied to time series via the Temporal Fusion Transformer, implemented in Darts. The Transformer's self-attention mechanism is theoretically well-suited to time series: it can attend to arbitrary lags rather than decaying them uniformly the way a recurrent net does.

**Article 6: Pretrained foundation models** - zero-shot forecasting with Chronos-2 and TimesFM 2.5. These models were trained on hundreds of millions of time-series observations from domains that do not include Bitcoin. The article examines what it means that a model with no Bitcoin-specific training still has an opinion about where the price is going.

**Article 7: Fine-tuned foundation models** - the same Chronos-2 and TimesFM 2.5 backbones, but `fit()` is called to update the weights against a held-out validation slice. The article studies the conditions under which fine-tuning beats zero-shot, and when it does not.

The progression is not just by complexity. Each step changes something specific about the inductive bias: from zero parameters to three, from linear to nonlinear, from hand-crafted features to learned representations, from task-specific training to general pretraining. Understanding *what changed* - not just *what the number is* - is the real purpose of the series.

## What this series is and is not

It is a **reproducible ML study**. Every experiment is a self-contained directory with a config file, a run script, and a results folder. The same `make` command that produced the numbers in the article will produce the same numbers on your machine, given the same data.

It is **not a trading system**. There is no backtester, no position sizing, no transaction-cost model, no live execution. The metrics are forecasting metrics. Whether a model with directional accuracy of 0.53 is actually profitable after fees, slippage, and tail-risk management is a separate and harder question that this series does not answer.

It is **not a leaderboard chase**. The configurations are reasonable and deliberately disclosed. There is no grid search over 500 hyperparameter combinations to find the one run where the LSTM happened to outperform the Transformer by three dollars of MAE. When an experiment underperforms, the article says so and explains why. The baseline exists precisely to prevent cherry-picking: if every model is measured against the same naive floor, the honest result is visible regardless of whether it is flattering.

The intended reader is someone who wants to understand what machine learning can and cannot do on a noisy financial time series - not someone looking for a shortcut to the numbers. Every article tries to be worth reading for its own sake: what the model does, why it was set up the way it was, what the results reveal, and what the next step in the lineup is trying to fix.

That is the setup. Article 1 establishes the floor.
