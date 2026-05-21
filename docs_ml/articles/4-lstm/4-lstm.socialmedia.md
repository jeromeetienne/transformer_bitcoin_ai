# Social media posts - LSTM

## Twitter

🧠 Deep learning meets Bitcoin.

LSTM with thousands of parameters. 1,448 training rows. Result: directional accuracy **0.4938**.

Below coin flip. The model is wrong about direction *more often than right*.

The first clear failure in the series. 🧵

#DeepLearning #Bitcoin

## Bluesky

Article 4: LSTM on Bitcoin.

Two stacked LSTM layers. 1,448 training bars. Sharpe 4.27 - the lowest among trained models.

The kicker: directional accuracy 0.4938. Below 0.5. The model is wrong about direction more often than right. A 3-parameter ARIMA on the same data hits 0.55.

More parameters + same data = worse generalization. Deep learning failure #1. 📉

## LinkedIn

The article in my Bitcoin ML series where deep learning first stumbles.

I trained a stacked LSTM (two layers, 32 hidden units, dropout 0.1, Adam optimizer, early stopping on validation loss) on 1,448 bars of 4-hour Bitcoin data. The expectation was that a recurrent network would learn temporal patterns the simpler models miss.

Results:

- MAE: 522.29 USD - within $4 of the naive baseline
- Directional accuracy: **0.4938** - below a coin flip
- Annualized Sharpe: 4.27 - the lowest among trained models in the series

Below a coin flip. The model is *wrong* about direction more often than right. A 3-parameter ARIMA on the same data hits 0.55. A 31-feature XGBoost hits 0.54. The LSTM with orders-of-magnitude more parameters cannot recover that.

The sweep is even more telling. The smallest LSTM configuration (24-bar input, 16 hidden units, 1 layer, no dropout) leads on Sharpe. The largest leads on point error. The configured default - mid-sized - wins neither. **Capacity inversely tracks Sharpe.**

One configuration produces NaN Sharpe entirely - the strategy never goes long during the rally because the model's forecasts never exceeded the reference price. A trained model that never bets on the up direction looks identical to a missing run on the leaderboard.

The lesson: on 1,448 training rows, recurrent networks have more capacity than there is signal. They learn the noise in validation loss while the directional signal that AR(1) captures cheaply gets washed out.

Deep learning is not magic. Data-to-parameter ratio is.

When have you seen a deeper model produce worse results than a simpler one?

#MachineLearning #DeepLearning #LSTM #TimeSeries #Bitcoin
