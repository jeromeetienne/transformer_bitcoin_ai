# Social Media Posts — Article 1: The Dumbest Model That Beats Most Others

## Twitter

**Thread: I built the dumbest Bitcoin predictor possible. Most ML models can't beat it. 🧵**

1/ The naive last-value predictor: predict that tomorrow's price = today's price.

Zero parameters. Zero learning. MAE: $337.89 on 437 hours of BTCUSDT.

Every model in this series has to clear that bar. Here's why it's harder than it sounds.

2/ Hourly Bitcoin is close to a random walk.

In a true random walk, the best forecast of the next value is the current value — because the increments are independent noise. No historical pattern helps.

That's the efficient market hypothesis, in practice.

3/ ARIMA(3,1,3): the first model that actually *learns* from the series.

Seven parameters estimated from training data. Walk-forward evaluation. No leakage.

Result: MAE nearly identical to $337.89.

4/ That's not a failure. It's an honest signal.

At hourly resolution, there is very little exploitable autocorrelation in BTC price changes. A linear model with 7 parameters finds almost nothing that a zero-parameter predictor doesn't already give you.

5/ The metric to watch later: directional accuracy.

Naive has none — it always predicts "no change."
ARIMA gets ~0.49–0.52. Close enough to a coin flip that you shouldn't trust it on 437 bars.

But the infrastructure is in place. Later models will have to earn it.

Full article → [link]

#MachineLearning #Bitcoin

---

## Bluesky

**The dumbest Bitcoin predictor has a $337 MAE. Most ML models can't beat it.**

The naive last-value predictor — "next price = current price" — is the floor every model in this series must clear. It works because hourly BTC is close to a random walk: the best single-step forecast is just the current value.

ARIMA(3,1,3), the first model that actually learns from data, comes in nearly identical. That's not a broken experiment. That's the honest signal-to-noise ratio at hourly resolution.

If a linear model with 7 parameters can't move the needle, the question for every subsequent model becomes much sharper.

Full article → [link]

#MachineLearning #TimeSeries

---

## LinkedIn

**Before you train any ML model on financial data, you need a floor. Most people skip this step — and it costs them.**

Article 1 of this Bitcoin forecasting series establishes two baselines:

**The naive predictor**: predict that the next price equals the current price. Zero parameters. MAE of $337.89 on the 2024 BTCUSDT hourly test set. This is the floor every subsequent model must beat.

**ARIMA(3,1,3)**: the first model that actually learns from the series. Seven parameters, fitted on training data, evaluated in strict walk-forward mode — each prediction uses the *actual* previous close, not the model's own forecast. MAE: close to $337.89.

The uncomfortable result: hourly Bitcoin behaves like a near-random walk. A linear model with access to the full price history finds almost no exploitable autocorrelation. That is not a flaw in the experiment — it is an accurate description of the signal available at this timescale.

**Why this matters for practitioners:**

If you later train an LSTM or a Transformer and it beats $337 MAE by $50, you now know that improvement is real — not a benchmark artifact. And if a deep model can't beat $337, you know to look harder before trusting it with anything that costs money.

The baselines are not a preamble to the "real" results. They are the first real result: that predicting hourly Bitcoin is hard, and the rest of the series will be honest about exactly how hard.

Article 1 is live → [link]

#MachineLearning #TimeSeries #QuantitativeFinance
