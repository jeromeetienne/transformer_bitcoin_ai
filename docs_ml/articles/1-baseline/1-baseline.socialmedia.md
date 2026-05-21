# Social media posts - Baseline

## Twitter

🪙 The dumbest possible Bitcoin predictor: `close_pred[t] = close[t-1]`.

Zero parameters. Zero training. MAE: $518.

The 3-parameter ARIMA in my next article beats it by exactly **$1.20**.

That gap is the whole punchline of this series. 🧵

#Bitcoin #MachineLearning

## Bluesky

I ran the dumbest possible Bitcoin predictor: predict the next price equals the current price. Zero parameters. MAE $518 per bar on 4h data.

Spoiler: the 3-parameter ARIMA that follows beats it by $1.20. On a 402-bar slice that's noise. Every fancy model in this series has to justify its complexity against that razor-thin floor. 📉

## LinkedIn

The most useful model in a machine learning project is often the one you build first - and never touch again.

I started a 7-model Bitcoin prediction series with the simplest possible baseline: `close_pred[t] = close[t-1]`. Predict next price equals current price. Zero parameters. Zero training. Pure pandas in one pass.

The result: MAE $518.36 per bar on 4-hour Bitcoin data.

Why bother? Because it sets a floor. Every model that follows - ARIMA, XGBoost, LSTM, Temporal Fusion Transformer, foundation models with hundreds of millions of parameters - has to clear that $518 floor on point error to justify its existence.

The brutal truth: ARIMA, the next experiment, beats it by **$1.20**. XGBoost is actually *worse* on MAE (539). The LSTM ties it within a few dollars. Even fine-tuned foundation models with 28 million parameters land within $10 of this number.

The naive baseline is not impressive. It is *instructive*. It teaches you that on near-random-walk financial data, point error is hard to improve and the action lives elsewhere - in directional skill, magnitude calibration, and how a model bets on what it predicts.

One more honest detail: the directional accuracy for the naive model is NaN, not zero. That is not a bug - it is the metric correctly reporting "no opinion expressed". A baseline that never picks a direction earns the right answer: NaN.

What is your favorite "embarrassingly simple" baseline that taught you more than the fancy model did?

#MachineLearning #TimeSeries #Bitcoin #DataScience
