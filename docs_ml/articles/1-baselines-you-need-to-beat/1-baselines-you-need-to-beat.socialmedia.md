# Social Media Posts — Article 1: The Dumbest Model That Beats Most Others

## Twitter

**Thread: The humbling truth about Bitcoin price forecasting**

1/ I built the dumbest possible Bitcoin price predictor.

It predicts that tomorrow's price = today's price.

Zero parameters. Zero learning.

MAE: $337.89.

Most ML models can't beat it. Here's why that matters. 🧵

2/ This is the naive last-value predictor — and it works because hourly Bitcoin price is close to a random walk.

In a random walk, the best forecast of tomorrow is today. No historical pattern helps.

Efficient market hypothesis, in practice.

3/ Next up: ARIMA(3,1,3). Three parameters, estimated from training data.

The first model that actually *learns* from the series.

Result: MAE close to $337.89. Nearly indistinguishable from naive.

4/ That's not a failure of ARIMA. It's an honest signal about hourly BTC.

At this timescale, there is very little autocorrelation to exploit. Linear price history doesn't predict price changes.

5/ The metric that *will* matter: directional accuracy.

Naive has none (it always predicts "no change").

ARIMA gets a number — around 0.49–0.52. Close enough to a coin flip that you shouldn't trust it on 437 bars.

But the infrastructure is in place. Later models will show real skill. Or won't.

Full article → [link]

---

## Bluesky

**The dumbest Bitcoin predictor: MAE of $337.

Most ML models can't beat it.**

The naive last-value predictor — "next price = current price" — is the floor every model in this series must clear.

It works because hourly BTC is close to a random walk.

ARIMA(3,1,3), the first model that actually learns from data, comes in nearly identical.

That's not a bug. That's the honest signal-to-noise ratio at hourly resolution.

Article 1 also establishes the full experiment architecture — the same data loader, metrics, and results layout that every subsequent model inherits.

Full article → [link]

---

## LinkedIn

**Before you train any ML model, you need a floor. Most people skip this step.**

In this series on machine learning for Bitcoin forecasting, Article 1 establishes two baselines:

**The naive predictor**: predict that the next price equals the current price. Zero parameters. MAE of $337.89 on the 2024 BTCUSDT hourly test set.

**ARIMA(3,1,3)**: the classical statistical model. Three parameters, fitted on training data, evaluated in strict walk-forward mode. MAE: close to the naive floor.

The uncomfortable lesson: hourly Bitcoin price behaves like a near-random walk. ARIMA can't exploit it. That's not a failure of ARIMA — it's an accurate description of the signal available at this timescale.

Why does this matter for practitioners?

Because if you train an LSTM or a Transformer and it beats $337 MAE by $50, you now know that's real signal — not a benchmark artifact. And if it doesn't beat $337, you know to look harder before trusting it with anything that costs money.

The baselines aren't a preamble to the "real" results. They are the first real result.

Article 1 is live → [link]
