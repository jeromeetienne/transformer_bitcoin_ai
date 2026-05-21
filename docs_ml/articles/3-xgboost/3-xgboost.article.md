# When 31 Features Lose to 3 Parameters: The XGBoost Lesson

So far we have two models sitting in the tight band between $517 and $518 MAE: a naive baseline and a linear ARIMA. The assumption is that we're hitting a floor - that on this noisy data, point error just doesn't get better without new information.

What if we don't add new information, but we engineer the information we have? What if we take the price history and break it down into 31 different angles - lagged returns, rolling volatility, volume, OHLC range - and feed them into a model smart enough to find patterns across all of them?

The XGBoost experiment does exactly this. It's the paradigm shift: the *generic model powered by engineered features*. The genius of XGBoost is that it's almost dumb - it's just a gradient boosting ensemble, a chain of shallow decision trees. The intelligence lives in the features.

What happens? The model *loses* on point error. But it dominates on trading returns.

## The feature engineering

This is the real work. ARIMA saw only one input: the previous price. XGBoost sees thirty-one.

**Past returns (24 features):** The lagged log-returns from the last 24 bars - basically the last six days at 4-hour intervals. These capture short-term momentum. If the last six bars moved up, did they predict further up?

**Rolling statistics (4 features):** On those same returns, compute the rolling mean and standard deviation over two windows: 6 bars and 24 bars. These capture volatility regime: in high-volatility periods, does volatility persist? In calm periods, does calm persist?

```
rolling_mean_6bar = mean(r[t-6:t])
rolling_std_6bar = std(r[t-6:t])
rolling_mean_24bar = mean(r[t-24:t])
rolling_std_24bar = std(r[t-24:t])
```

**OHLC summaries (2 features):** High-low range (volatility within the bar) and open-close body (direction within the bar). These are the bar's internal structure.

**Volume (1 feature):** Log of the previous bar's volume. Does volume predict price moves?

All thirty-one features are shifted by one bar (lagged), so at prediction time `t`, you're using information that was available before time `t`. This keeps the walk-forward clean - no information leakage.

## The XGBoost model

Gradient boosting works like this: build a first shallow decision tree that predicts log-returns. It won't be perfect - some bars will be right, some wrong. Then build a second tree to predict the *residuals* (the part the first tree got wrong). Then a third tree to predict the residuals of the first two. Keep stacking until you've built enough trees to fit the data well.

The configuration:

```yaml
model.n_estimators: 400        # 400 trees in the ensemble
model.max_depth: 5             # shallow trees (5 levels max)
model.learning_rate: 0.05      # slow learning (shrink each tree's contribution)
model.subsample: 0.8           # 80% of rows per tree (row randomization)
model.colsample_bytree: 0.8    # 80% of features per tree (feature randomization)
model.reg_lambda: 1.0          # L2 regularization (penalize large weights)
```

These are conservative settings - the default values suggested by the library. But defaults aren't optima. The article on hyperparameter sweeps (not this one) will show that smaller trees do better on this data.

## The results

| Metric | Value |
| --- | --- |
| MAE | 539.39 USD |
| RMSE | 795.07 USD |
| MAPE | 0.7008 % |
| Directional accuracy | 0.5365 |
| **Cumulative return** | **0.5042** (50.42 %) |
| **Annualized Sharpe** | **6.4233** |

On point error, XGBoost is *worse* than both baselines:

| Model | MAE |
| --- | --- |
| 01_baseline | 518.36 |
| 02_arima(1,1,1) | 517.16 |
| **03_xgboost** | **539.39** |

Twenty-one dollars worse than naive. Thirty-one features and non-linearity, and you're more wrong on average.

But look at the trading metrics. Cumulative return 50.42% beats ARIMA's 33.14% - a 51% relative lift. Sharpe 6.4233 beats ARIMA's 6.0569. On Sharpe per-bar: 6.4233 / √2190 ≈ 0.1373 against SE ≈ 0.0502 gives a ratio ≈ 2.74 σ - significant at roughly the 99% level.

This is the core observation: **31 features + non-linearity win on trading returns while losing on point error.**

## Why point error gets worse

When you train a tree ensemble on noisy data with more capacity than the signal warrants, you fit noise. The thirty-one features give the model 31 degrees of freedom to find patterns - most of which are random.

On 4-hour Bitcoin, there's a tiny signal in AR(1) momentum. A linear model (ARIMA) captures it cheaply with one coefficient. A tree ensemble with 31 inputs and hundreds of trees can also capture it, but it also fits a huge amount of noise trying to extract more signal that doesn't exist.

The residual noise that AR(1) doesn't remove - the part ARIMA leaves on the table - looks predictable to a complex model. So XGBoost fits it. But that "predictability" is just overfitting: random patterns that don't generalize.

The result: XGBoost has higher MAE on the test slice because it's fit noise in the training slice.

## Why Sharpe gets better anyway

Point error is one metric. Prediction magnitude calibration - how *confident* your predictions are - is another.

When XGBoost makes a directional call, it makes it bigger. On bars it predicts "up," it predicts *further* up than ARIMA does. On a strongly uptrending slice (the post-election 2024 rally), bigger bets on up accumulate more returns.

The trading strategy is simple: long/flat. If `predicted_return > reference_return`, go long the next bar. Otherwise, stay flat.

ARIMA's average predicted return on "up" bars might be +0.5%. XGBoost's might be +1.0%. Same direction signal, more confidence. In an uptrending regime, that's worth money.

Sharpe compounds this: it's `(mean_strategy_return) / (std_strategy_return)`. If you're right more often *and* more confident when you're right, Sharpe climbs. XGBoost achieves this even though it's wrong by more on average.

## Directional accuracy

XGBoost calls direction 53.65% of the time. Better than a coin flip, but worse than ARIMA's 55.47%. Yet it still wins on Sharpe. How?

Because Sharpe isn't just directional accuracy - it's the *magnitude* of returns when you're right. XGBoost is wrong less often, but when it's right, it's more confident. The trade-off favors Sharpe.

This is a recurring pattern in the series: point error and directional accuracy and Sharpe all measure different things. A model can lose on one while winning on others.

## The feature ceiling

Here's the structural takeaway: on a near-random-walk signal, engineered features hit a hard ceiling. You can add features until you're blue in the face. The signal-to-noise ratio is fixed by the data, not the model.

ARIMA extracts the linear momentum signal efficiently (one coefficient, one difference parameter). XGBoost can find non-linear patterns if they exist, but on Bitcoin 4h, they don't. The non-linear capacity goes to overfitting.

If you had a dataset where non-linear patterns were real - say, option pricing or credit-card fraud - XGBoost would shine on point error too. On Bitcoin at this horizon, it doesn't.

## The hyperparameter sweep

A sweep of 8 configurations exists on disk, but there's a caveat: it was run on the older 1-hour data slice, not the current 4-hour one. The MAEs in that sweep (269-276 USD) are on a different scale.

Within the stale sweep, the smallest configs win on point error - `(n_estimators=200, max_depth=3, learning_rate=0.1)` posts MAE 269.05. The config `(800, 5, 0.03)` wins on Sharpe at 3.93. The configured default `(400, 5, 0.05)` is neither.

The lesson is consistent with point error theory: on noisy data, *smaller models are more honest*. The 400-tree default is bigger than the data needs.

## Caveats

This is a single deterministic fit. XGBoost with `subsample` and `colsample_bytree` randomization has run-to-run variance. The MAE 539.39 is conditional on the seed in the config (`random_state: 42`). Three to five seed runs would tighten the numbers.

The walk-forward doesn't retrain. Trees are frozen at train-time values. On a longer test window or regime shift, they'd go stale.

Features are minimal. No on-chain features (active addresses, transaction volume), no macro sentiment, no relative price of other assets. The model is blind to everything except its own history and basic price structure.

No early stopping in the baseline run. The full 400 trees are used. Holding back a validation set and stopping when validation loss plateaus would prevent overfitting and likely improve generalization. Sweeps that follow test this.

## Closing

XGBoost is the first model to add real capacity (31 features + ensembled non-linearity) and *lose* on point error. It dominates on trading returns because it trades point-error precision for directional confidence in a rallying regime.

The paradigm: **the generic model is only as good as its features.** Throw 31 features at a tree ensemble and it goes to work. But if the features are derived from noisy data with little signal, the ensemble fits noise. Sharpe still wins because magnitude calibration beats accuracy on a trending slice.

ARIMA's lesson was "linear works on this signal." XGBoost's lesson is "capacity doesn't manufacture signal you don't have - it manufactures overfitting instead."

What if we move beyond flat feature vectors entirely and use models that see *sequences*? Can a recurrent neural network that passes hidden state across bars learn patterns ARIMA and XGBoost both miss?

That's next.

## How to reproduce

```bash
make 03_xgboost
make 03_xgboost_sweep  # note: sweep is stale on 1h; do not use it to rank 4h configs
```

Results live in [experiments/03_xgboost/results/btc_4h_2024/](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/03_xgboost/results/btc_4h_2024). The predictions are in [predictions.parquet](https://github.com/jeromeetienne/transformer_bitcoin_ai/blob/HEAD/experiments/03_xgboost/results/btc_4h_2024/predictions.parquet).
