# The Dumbest Possible Bitcoin Predictor (And Why It Matters)

If you could predict any stock with perfect accuracy, you'd be a billionaire. But there's a question that comes first: *how hard is this really?* Not in the sense of "how much machine learning knowledge do you need," but in the sense of signal-to-noise. How much predictability even *exists* to extract?

The most honest way to answer that is with a baseline. Not a machine learning baseline - a *dumb* baseline. A model so simple it's barely a model at all. And if your sophisticated neural network with attention mechanisms can't beat it, you haven't learned anything. You've just fit noise.

This article explores the simplest possible Bitcoin price predictor: yesterday's price. It's a floor. Everything else in this series is measured against it.

## The zero-parameter model

Imagine you're asked to predict Bitcoin's 4-hour closing price. You have no historical data, no economic models, no technical indicators. You have exactly one piece of information: the price that just closed. What do you predict for the next bar?

The honest answer: you predict it stays the same.

```
close_pred[t] = close[t-1]
```

That's the entire model. No fitting, no parameters, no training. Feed it the previous close and it returns the previous close. In machine learning terms, it's the "naive random-walk" forecast - a model so simple that its only parameter is *nothing*.

Why would anyone use this? Because on financial data close to a random walk (and most trading data is), it's often the hardest baseline to beat. A price that moved up might move down next. A price that stayed flat might stay flat. The model says "I'm betting on continuity" without any intelligence about *why* continuity might hold. And on scales of hours, continuity is a surprisingly strong bet.

## The setup

To test this, we need:

- **Data:** Bitcoin BTCUSDT at 4-hour intervals from January 1, 2024 through November 30, 2024. That's 2,010 bars - about 11 months of history.
- **Split:** 80% training, 20% test, respecting time order. The test slice is 402 bars - roughly September 25 through December 1 (the post-election rally).
- **Metrics:** We measure mean absolute error (MAE - how far off we are on average), root mean squared error (RMSE - giving extra weight to outliers), mean absolute percentage error (MAPE - error as a percentage of the price), and directional accuracy (how often we called the direction right).

The walk-forward method matters. We don't train on all of 2024 and test on a future year we haven't seen. Instead, we respect causality: fit on a historical window, predict on the next window, move forward. This is how you'd actually trade it in real time.

One subtlety: the naive baseline doesn't "train" at all. It just exists. There's nothing to fit. It predicts the next price is the previous price, in one pass over the test slice.

## The numbers

From the experiment's [metrics.json](https://github.com/jeromeetienne/transformer_bitcoin_ai/blob/HEAD/experiments/01_baseline/results/btc_4h_2024/metrics.json):

| Metric | Value |
| --- | --- |
| MAE | **518.36 USD** |
| RMSE | 784.09 USD |
| MAPE | 0.6701 % |
| Directional accuracy | NaN |
| Cumulative return | — |
| Annualized Sharpe | — |

On average, the naive predictor is off by $518 per bar. On a price oscillating between $40k and $100k over the test window, that's roughly 0.67% error. Not bad for a model that knows nothing.

The RMSE is higher than the MAE (784 vs. 518) because RMSE punishes big misses more harshly. Bitcoin moved hard on some bars. The naive model gets those bars wrong by a lot, and RMSE captures that.

The directional accuracy is NaN. That's not missing data - that's intentional.

## Why directional accuracy is NaN (and that's a feature)

The directional accuracy metric asks: "Did you call the direction right?" It compares the actual price change (`actual_dir = sign(close[t] - close[t-1])`) to your predicted direction (`pred_dir = sign(close_pred[t] - close[t-1])`).

The naive baseline predicts `close_pred[t] = close[t-1]`. The difference is zero. The predicted direction is `sign(0)` - which is zero. On every single bar, the model has no opinion. It doesn't predict up or down. It predicts *no change*.

The metric's masking rule filters out bars where the model expresses no opinion (where `pred_dir == 0`). Since the naive model's direction is 0 everywhere, the mask removes all bars. The sample set is empty. The function returns NaN - not because something broke, but because the question "what percent of directional calls were right?" doesn't apply when you never make a directional call.

This is correct behavior. The naive baseline is literally expressing no opinion on direction. NaN is the honest answer.

It also means you can't compute a trading strategy from the naive model. Without a directional opinion, you can't build a "long when predicted up, flat when predicted down" rule. There's no strategy, so no cumulative return and no Sharpe ratio.

## The insight

Here's where it gets interesting. The next model in the series - ARIMA, a 3-parameter statistical model of price changes - reports:

| Experiment | MAE | RMSE | MAPE | dir_acc | Sharpe |
| --- | --- | --- | --- | --- | --- |
| 01_baseline | 518.36 | 784.09 | 0.6701 % | NaN | — |
| 02_arima (1,1,1) | 517.16 | 782.42 | 0.6686 % | **0.5547** | **6.0569** |

ARIMA beats the naive baseline on MAE by $1.20. One dollar and twenty cents. On a 402-bar test slice where a single big move can be $100, that's *statistical noise*.

But ARIMA does something the naive model can't: it expresses a directional opinion. It calls direction right 55.47% of the time - barely above a coin flip, but above. And on a post-election Bitcoin rally where the trend is strongly up, that's enough to generate a 6.06 annualized Sharpe and 33% cumulative return.

The key observation: **point error and directional skill are nearly orthogonal on this slice.**

You can be nearly right on the magnitude (naive's 0.67% error) and completely neutral on direction. You can add an enormous amount of model complexity and barely beat the magnitude while improving direction. You can fit 31 engineered features to a tree ensemble and *worse* on point error (MAE 539) while getting *better* on trading returns.

This is what a baseline teaches you. It's not just a number to beat. It's a signal about the structure of the problem itself. Bitcoin's 4-hour prices don't change much from bar to bar. But when they do, the direction matters more than the magnitude. A model that nails the point forecast but guesses wrong on direction loses money on the strategy. A model that's mediocre on magnitude but right on direction wins.

## What this means for the series

Every model in the articles that follow - ARIMA, XGBoost, LSTM, Transformer, and the foundation models - is measured against this $518 MAE floor. Some beat it. Some don't. All of them have to justify their complexity relative to this $1.20 gap.

The series doesn't pretend that the baseline is good. It's not. It's the worst model in the leaderboard on directional accuracy. But it's *instructive*. It teaches you to ask the right questions: What signal are you extracting? What are you optimizing for? Is the metric you're improving actually the one that makes money?

The naive baseline is the humble truth. Everything else either beats it or admits it doesn't.

## How to reproduce

To run this experiment yourself:

```bash
make 01_baseline
```

The code lives in [experiments/01_baseline/](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/01_baseline). The predictions are stored in [predictions.parquet](https://github.com/jeromeetienne/transformer_bitcoin_ai/blob/HEAD/experiments/01_baseline/results/btc_4h_2024/predictions.parquet). The plot is at [plot.png](https://github.com/jeromeetienne/transformer_bitcoin_ai/blob/HEAD/experiments/01_baseline/results/btc_4h_2024/plot.png).
