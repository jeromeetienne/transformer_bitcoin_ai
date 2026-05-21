# Social media posts - ARIMA

## Twitter

🧮 ARIMA(1,1,1) beats the naive Bitcoin predictor by **exactly $1.20** on MAE.

But it earns a 6.06 Sharpe vs. naive's nothing.

The trick is not point error. It is the 55% directional accuracy that turns into 33% cumulative return on the post-election rally. 📈

#Bitcoin #ARIMA

## Bluesky

Article 2: ARIMA on Bitcoin.

3 parameters. MAE $517.16 - just $1.20 below the naive baseline. But it earns a Sharpe of 6.06 because it picks direction 55% of the time and the strategy compounds during the rally.

The 7-parameter ARIMA(3,1,3) leads MAE, MAPE, Sharpe AND directional accuracy all at once. The classical statisticians knew what they were doing. 📊

## LinkedIn

Why does a 3-parameter linear model from the 1970s still hold its own against million-parameter neural networks on a forecasting problem?

Article 2 of my Bitcoin machine learning series runs ARIMA against the same 4-hour BTC data as the rest. The results:

- MAE: $517.16 (vs. naive baseline at $518.36 - a $1.20 gap)
- Directional accuracy: 55.47% (naive has none)
- Annualized Sharpe: 6.06
- Cumulative return: 33% on the long/flat strategy

Three parameters. Linear. The model just says "the next change in price is a weighted sum of recent changes plus a residual correction term". That is it.

A sweep across 12 orders surfaces three things worth knowing:

1. ARIMA(0,1,0) - no AR, no MA, just differencing - returns MAE 518.358631840796 - bit-identical to the naive baseline. This is the random-walk specification. The pipeline sanity check passes.

2. ARIMA(1,0,1) - no differencing - collapses on Sharpe to 0.56. Differencing is non-optional on price data. Levels are non-stationary. Changes are.

3. ARIMA(5,1,5) overfits. AIC goes up, Sharpe drops to 4.59. More parameters do not buy more signal on noisy data.

The lesson is bigger than ARIMA. **Point error and trading metrics are nearly orthogonal on this slice.** You can be nearly right on magnitude and completely neutral on direction, or vice versa. Picking the metric you optimize for changes the model you end up choosing.

What is the simplest model that surprised you with how well it worked?

#MachineLearning #TimeSeries #ARIMA #Bitcoin #DataScience
