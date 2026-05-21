# Social media posts - XGBoost

## Twitter

🌳 31 engineered features. XGBoost trees. The result?

MAE $539 - *worse* than a 3-parameter ARIMA. *Worse* than predicting "no change".

But Sharpe 6.42 - the highest in the series so far.

How can a model be wrong on average and still make money? 🧵

#Bitcoin #XGBoost

## Bluesky

Article 3: XGBoost on Bitcoin.

31 engineered features (lagged returns, rolling moments, OHLC summaries). Gradient boosting trees. MAE $539 - worse than naive's $518. Worse than ARIMA's $517.

But the trading Sharpe is 6.42 - the highest in the series. The trees buy directional aggression at the cost of point-error fit. Strategy compounds. Magnitude lies.

Point error ≠ profitability. 📊

## LinkedIn

This is the article in my Bitcoin ML series where the leaderboard stops being intuitive.

XGBoost with 31 engineered features (lagged log-returns, rolling means and standard deviations, volume, OHLC range) reports:

- MAE: 539.39 USD (**worse** than the naive baseline at 518.36)
- Directional accuracy: 53.65 % (lower than ARIMA's 55.47 %)
- Cumulative return: 50.42 % (highest in the series so far)
- Annualized Sharpe: 6.42 (also highest)

How can a model with worse point error make more money?

The mechanism is directional aggression. XGBoost is less correct on direction than ARIMA, but when it predicts up, it predicts *further* above the reference price. The long/flat strategy stays long more often during the rally and captures more of the trend. ARIMA is calmer; its forecasts hug the reference price; the strategy goes flat more often and forgoes profit.

This is the most uncomfortable lesson in machine learning evaluation: **the metric you train on is not the metric that pays.** XGBoost trains on squared error of log-returns. ARIMA trains on residual likelihood. Neither optimizes Sharpe. Yet the model that loses on point error wins on strategy.

The sweep is even more interesting. The smallest configuration (200 trees, depth 3, lr 0.1) leads on MAE. The mid-size configuration (800 trees, depth 5, lr 0.03) leads on Sharpe. The configured default wins neither. The sweet spot on point error is not the sweet spot on Sharpe.

This article is where the series stops being a ladder and becomes a tradeoff.

What is the most counterintuitive metric-vs-metric tradeoff you have run into?

#MachineLearning #XGBoost #TimeSeries #Bitcoin #DataScience
