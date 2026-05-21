# How Classical Statistics Beats the Naive Model (By $1.20)

After seeing the naive baseline, you might ask: can we do better? The first instinct for time-series forecasters has always been ARIMA - AutoRegressive Integrated Moving Average. It's a model that has powered production forecasting for decades. Interest rates, inflation, airline passengers, electricity demand. It works everywhere that data is stationary.

The promise is simple: instead of predicting that price stays the same, predict that *changes* in price follow a pattern. Maybe recent gains mean more gains are coming (momentum). Maybe they mean a reversal is due (mean reversion). Maybe the surprise in the last bar suggests the next bar overshoots. ARIMA lets you tune all three.

The question: does it matter for Bitcoin?

## ARIMA(p, d, q) unpacked

ARIMA stands for three operations chained together. The names are intimidating. The intuitions are not.

**Differencing (d):** Take the price level and convert it to price *changes*. If price is $43,000 then $43,100 then $42,900, the changes are +$100 then -$200. The `d` parameter says how many times to take the difference. `d=1` (one differencing) works with changes; `d=2` would work with the *changes of changes* (the acceleration).

Why this step? Financial prices are "non-stationary" - they wander over time. The mean changes, the volatility changes. Differences are more stable: the distribution of 4-hour changes is more constant than the distribution of 4-hour prices.

**Autoregression (p):** After differencing, fit a linear regression where the change at time `t` is a weighted sum of the previous `p` changes. `p=1` says "recent change predicts the next change." `p=3` says "the last three changes all matter."

```
change[t] = c + ar1 * change[t-1] + ar2 * change[t-2] + ar3 * change[t-3]
```

If the coefficient on `change[t-1]` is positive, you have momentum: recent gains continue. If it's negative, you have mean reversion: recent gains reverse.

**Moving Average on residuals (q):** After the autoregression fits, take the leftovers (residuals - the part the AR didn't explain) and use them to predict the next residual. `q=1` says "if the last bar surprised us (the actual change was bigger than the AR predicted), expect another small surprise in the same direction."

```
change[t] = AR(...) + ma1 * residual[t-1]
```

Chained together, ARIMA(p, d, q) is: differencing, then AR on the changes, then MA on the surprises.

## The experiment

We fit ARIMA on the same 4-hour Bitcoin data as the baseline: January 1 through November 30, 2024, split 80/20 at 1,608 training bars and 402 test bars.

The headline configuration is ARIMA(1, 1, 1): one difference (work with changes), one AR coefficient (one lag), one MA coefficient (one residual lag). A tiny model by modern standards - three estimated parameters.

But we also swept a grid of 12 different (p, d, q) combinations to see which explains the data best:

- (0, 1, 0): the naive baseline in ARIMA form
- (1, 1, 0): AR only
- (0, 1, 1): MA only
- (1, 1, 1): the configured default
- (2, 1, 0), (0, 1, 2), (2, 1, 2): two-parameter variants
- (3, 1, 3): richer AR and MA
- (5, 1, 0), (0, 1, 5), (5, 1, 5): overfit shapes
- (1, 0, 1): un-differenced control

## The results

The configured ARIMA(1, 1, 1):

| Metric | Value |
| --- | --- |
| MAE | **517.16 USD** |
| RMSE | 782.42 USD |
| MAPE | 0.6686 % |
| Directional accuracy | **0.5547** |
| Cumulative return | 0.3314 (33.14 %) |
| Annualized Sharpe | **6.0569** |

Compare to naive baseline (MAE 518.36): ARIMA wins by $1.20. Within the margin of noise on a 402-bar slice.

But look at directional accuracy. The baseline was NaN - no opinion. ARIMA says up or down 55.47% of the time, beating a coin flip. And that directional edge translates to money: 33% cumulative return on a long/flat strategy, Sharpe of 6.06 (about 2.59 standard deviations above noise).

The sweep reveals something more interesting. The (3, 1, 3) configuration - richer AR and MA - leads on five columns simultaneously:

| Configuration | MAE | RMSE | MAPE | AIC | Sharpe |
| --- | --- | --- | --- | --- | --- |
| (0, 1, 0) | 518.36 | 784.09 | 0.6701 % | 25546.20 | NaN |
| (1, 1, 1) | 517.16 | 782.42 | 0.6686 % | 25549.66 | 6.0569 |
| (3, 1, 3) | 514.07 | 776.69 | 0.6659 % | 25543.88 | 6.4015 |

The (3, 1, 3) row beats everything. But notice: on MAE, the gap between (1, 1, 1) and (3, 1, 3) is $3. The difference between naive and (3, 1, 3) is only $4.29 on a 402-bar slice. You can pick a different ARIMA order and change the leaderboard by pennies.

One more crucial result: the (0, 1, 0) row - no AR, no MA, just one differencing - returns MAE 518.358631840796. Bit-identical to the naive baseline. This is the sanity check. Our pipeline works. The random walk is the random walk.

## Walking forward without retraining

The ARIMA experiment uses a specific walk-forward protocol. Fit the model on the training slice (1,608 bars) to estimate the three ARIMA(1, 1, 1) coefficients. Then, predict the test slice (402 bars) *without retraining*. The coefficients stay frozen. The model sees new observations but doesn't update its parameters.

This is realistic - you fit a model once, deploy it, and predict forward. But it's also conservative. Later test bars are predicted using coefficients fit only on the training window, which might be stale. More sophisticated approaches re-fit the model every N bars (rolling-window validation). ARIMA doesn't do that here.

The upshot: directional accuracy 0.5547 is conditional on the training window being representative. If the relationship between changes breaks down in the test regime, directional accuracy collapses.

## What the directional win really means

55.47% directional accuracy sounds good - you're right more than you're wrong. But accuracy alone doesn't predict profitability. A model that's 55% right on direction but always bets small will lose to a model that's 51% right but bets big on conviction.

ARIMA's Sharpe tells you it's not just *right more often* - it's *right in the right size*. The Sharpe metric compounds directional accuracy with prediction magnitude: on the bars ARIMA predicts "up," does it predict 0.01% up or 1% up? If it's confident in the right direction, Sharpe climbs. If it's confident in the wrong direction, Sharpe crashes.

Sharpe 6.06 on a 402-bar slice is significant. Per-bar Sharpe = 6.0569 / √2190 ≈ 0.1295. Standard error on direction ≈ 1 / √402 ≈ 0.0499. Ratio ≈ 2.59 σ - about 99% confidence that the edge is real, not luck.

## Why (1,0,1) fails

The sweep includes one "control": ARIMA(1, 0, 1) - AR and MA but *no differencing*. It fits a model directly on price levels, not changes.

It collapses. Sharpe drops to 0.5560 - barely above zero. MAE is 521.05, worse than (1, 1, 1)'s 517.16.

This is the lesson: **differencing is non-optional on price data.** Price levels are non-stationary. The distribution of prices changes as the price drifts. AR / MA models assume stationarity. Without differencing to stabilize the distribution, the model fits the wrong pattern and the strategy goes flat.

(1, 0, 1) loses because it tried to model the level directly and failed. (1, 1, 1) works because differencing forces the model to look at the structure that matters: *changes*.

## The overfit boundary

Move beyond three parameters and something interesting happens. (5, 1, 5) has 5 AR coefficients and 5 MA coefficients - plus the bias. Much more capacity.

It performs *worse* than (3, 1, 3): AIC is higher (25550.49 vs. 25543.88), MAE is worse (522.58 vs. 514.07), Sharpe is lower (4.59 vs. 6.40).

The extra AR / MA parameters don't extract signal - they fit noise. On a 1,608-bar training window, five AR and five MA coefficients are enough to overfit the training dynamics. The coefficients don't generalize to the test slice.

The sweet spot is (3, 1, 3) on this data. Adding more parameters makes things worse.

## Caveats and honesty

ARIMA assumes the residuals are roughly Gaussian, independent, and have constant variance (homoskedasticity). Bitcoin returns are none of these. They're heavy-tailed (outlier moves are more likely than a Gaussian predicts), clustered (volatile periods cluster), and heteroskedastic (volatility isn't constant).

Interval forecasts (the uncertainty bands) would be badly miscalibrated if computed from ARIMA's standard errors. Point forecasts are more robust.

ARIMA also works on univariate data: only past closes. It doesn't ingest volume, OHLC range, sentiment, or any exogenous signal. Everything the model knows comes from the price time series itself.

The walk-forward doesn't retrain. On a longer test window or a regime change, the coefficients might go stale.

## Closing

ARIMA(1, 1, 1) ties the naive baseline on point error - $517.16 vs. $518.36 - and adds directional skill on top. It's the classical-statistics answer to the problem: linear, interpretable, low-capacity. It works.

The broader lesson: you can't beat point error much without new information. ARIMA takes the same price history as the naive model and adds no new data. It extracts a little signal (momentum or mean reversion) and gets a $1.20 MAE improvement. That $1.20 doesn't sound like much, but the directional skill it buys - Sharpe 6.06 - is worth money.

Next in the series: what if we break the linearity assumption and add engineered features? Can 31 lagged returns, rolling moments, and OHLC summaries do better?

## How to reproduce

```bash
make 02_arima
make 02_arima_sweep
```

Results live in [experiments/02_arima/results/btc_4h_2024/](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/02_arima/results/btc_4h_2024). The sweep rankings are in [sweep.csv](https://github.com/jeromeetienne/transformer_bitcoin_ai/blob/HEAD/experiments/02_arima/results/btc_4h_2024/sweep.csv).
