# The Dumbest Model That Beats Most Others: Why Baselines Are Everything

Before any model runs, there is a number to beat. Not a number from the literature, not a number from a Kaggle leaderboard, but the number produced by the dumbest possible forecaster applied to this exact dataset on this exact test slice. That number is the floor. Every subsequent model in this series is measured against it, not against the void.

This article establishes two floors: the zero-parameter naive predictor and the three-parameter ARIMA model. Together they define what "learning from data" means for hourly Bitcoin price — and they reveal something uncomfortable about the problem before any serious model is trained.

## The naive baseline

The simplest conceivable forecast is:

```
close_pred[t] = close[t-1]
```

Predict that the next closing price equals the current one. No parameters, no fitting, no training data consumed. The model's entire theory of the market is that prices do not change. This is called the **naive last-value predictor**, and it is harder to beat than it has any right to be.

Why? Because Bitcoin's hourly price series is close to a **random walk**. In a true random walk, the best prediction of tomorrow's value is today's value, because the increments are independent and identically distributed noise. No information in the past helps you forecast the future. The efficient market hypothesis, in its weak form, asserts that liquid asset prices behave this way. Bitcoin, being one of the most liquid assets in the world with continuous global participation, is a reasonable candidate.

The practical consequence is that any model that learns from historical price must extract a signal that, by hypothesis, isn't there. When a model fails to beat naive, the correct interpretation is not "this model is bad." It is "this model is honest." ARIMA failing to beat naive on hourly BTC is not a bug — it is the truth about the signal-to-noise ratio at this timescale.

### How the experiment works

The experiment is in [`experiments/01_baseline/`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/01_baseline). It is deliberately minimal:

1. Load OHLCV for `BTCUSDT 1h` via the shared data loader ([`src/btc_ai/data`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/src/btc_ai/data)).
2. Hold out the last 20% of the series as the test set. No shuffling. The split is chronological.
3. For each test bar *t*, predict `close[t-1]`.
4. Compute MAE, RMSE, MAPE, and directional accuracy on the test slice.
5. Write `results/metrics.json`, `results/predictions.parquet`, `results/plot.png`.

Step 2 is worth pausing on. The 20% held-out test set on the 2024 `BTCUSDT 1h` series is approximately 437 bars — about 18 days. Every experiment in this series evaluates on the same 437 bars. That is what makes comparison valid.

```bash
make baseline
```

### What the numbers say

On the default config (`BTCUSDT 1h`, 2024-01-01 → 2024-04-01, last 20% held out):

```json
{
  "rows_total": 2184,
  "rows_test": 437,
  "mae": 337.89,
  "rmse": 471.54,
  "mape": 0.005024,
  "directional_accuracy": NaN
}
```

The MAE of **337.89 USD** is the floor. Every later model must produce a number below this on the same test slice to be considered useful. The RMSE of 471.54 is higher than the MAE — this is always true by construction, since the root-mean-square penalizes large errors more than the mean-absolute penalizes them. The gap (133 USD) tells you that hourly BTC has fat tails: most bars move modestly, but occasionally a bar moves by a thousand dollars and it shows up disproportionately in the RMSE.

The MAPE of 0.5024% sounds small. At a Bitcoin price of $65 000, half a percent is $326 per bar. On 437 bars, that compounds.

Directional accuracy is `NaN` by design. The naive predictor always predicts "no change" — it has no direction. For models that do predict a direction, 0.50 is the coin-flip baseline, and anything above that is evidence of real skill.

### What the naive baseline establishes beyond the number

The experiment's second purpose is architectural. The data loader, config schema, metrics module, and results layout used here are replicated unchanged by every later experiment. When ARIMA or the Transformer produces a `results/metrics.json`, it has the same keys, the same units, and the same test slice. The comparison is apples to apples because the machinery is identical.

## The ARIMA baseline

ARIMA — **A**uto**R**egressive **I**ntegrated **M**oving **A**verage — is the first model in this series that learns something from data. It has three parameters, selected deliberately rather than searched automatically:

```
ARIMA(p, d, q)
   p : AR  — number of autoregressive lags
   d : I   — differencing order
   q : MA  — number of moving-average lags on the residuals
```

The default configuration uses `(3, 1, 3)`. Here is what each number means, concretely.

### The three parameters

**d = 1** — differencing order. Raw BTC prices are non-stationary: they trend, they jump, they do not revert to a fixed mean. ARIMA handles this by modeling *first differences* — that is, the price change from bar to bar — rather than the price level itself. After differencing once (d=1), the series has roughly constant mean and variance, which satisfies the stationarity assumption that the AR and MA components require.

**p = 3** — autoregressive lags. The AR component models the current (differenced) value as a linear combination of its *p* most recent predecessors. With p=3, the model looks at the last three price changes and asks: given the recent direction and magnitude of movement, how much of that carries into the next bar? A positive AR coefficient says recent moves tend to continue (momentum). A negative coefficient says they tend to reverse (mean reversion). In practice, hourly BTC typically shows coefficients close to zero — there is not much exploitable autocorrelation.

**q = 3** — moving-average lags. The MA component models the current value as a linear combination of *q* past **residuals** — the prediction errors the AR component made on previous bars. This is a short-term error-correction mechanism: if the model consistently under-predicted the last three bars, the MA component nudges the next prediction upward. With q=3, the model has a memory of the last three corrections.

### Walk-forward evaluation

The experiment in [`experiments/02_arima/`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/02_arima) follows the same split as the naive baseline: fit on the training slice, evaluate on the held-out 20%. The key detail is how the walk-forward is implemented:

1. Fit `ARIMA(close_train, order=(p, d, q))` using `statsmodels`. The AR/MA coefficients are estimated once on training data — no leakage.
2. Extend the fitted model to the full series via `fit.apply(full_series, refit=False)`. This keeps the trained coefficients frozen while allowing the model to condition on actual test-set observations as they arrive.
3. Call `predict(start=split, end=last, dynamic=False)`. The `dynamic=False` flag is critical: each prediction at bar *t* uses the *actual* close of bar *t-1*, not the model's own prediction of *t-1*. This is true one-step-ahead walk-forward evaluation.

The coefficients are not re-estimated at each step. A more rigorous approach would refit the model at every new observation (rolling window or expanding window). That variant is more accurate and considerably slower. For a 437-bar test set it would not change the conclusions materially, and the simpler approach is easier to reason about.

```bash
make arima
```

To compare multiple `(p, d, q)` choices without overwriting the canonical results:

```bash
make arima-sweep
```

The sweep runs a list of ARIMA orders (defined in `sweep.py`) and writes a `results/sweep.csv` with the MAE, RMSE, MAPE, directional accuracy, AIC, and BIC for each order. The `(0, 1, 0)` row — the random walk with no AR or MA terms — should reproduce the naive baseline MAE almost exactly. If it does not, something is wrong with the pipeline.

### How to read the results

ARIMA and naive, head to head on the same 437-bar test slice:

| | Naive | ARIMA(3,1,3) |
|---|---|---|
| Model | `pred = close[t-1]` | Fitted AR + MA on first differences |
| Parameters | 0 | 7 (p + q + intercept) |
| MAE (USD) | 337.89 | *run the experiment* |
| RMSE (USD) | 471.54 | *run the experiment* |
| MAPE | 0.50% | *run the experiment* |
| Directional accuracy | NaN | *run the experiment* |

The MAE result will be close to the naive floor. This is not a flaw in the experiment — it is the expected outcome on an asset that behaves like a near-random-walk at hourly resolution.

Three possible outcomes and what they mean:

**MAE close to naive (within ~10 USD)** — the most likely outcome. ARIMA's AR and MA coefficients are small because there is little autocorrelation to exploit. The model is approximately fitting a random walk with drift, which is what `(0, 1, 0)` would give. The honest conclusion is that linear price history does not predict price changes at this timescale.

**MAE clearly below naive** — surprising, and worth investigating before celebrating. First, verify the split logic: any lookahead in the data preparation would produce artificially low error. Second, check whether the test slice happens to fall on an unusually trending period where momentum is temporarily exploitable. If the split is clean and the period is representative, a genuine ARIMA improvement on hourly BTC is a meaningful finding.

**MAE above naive** — the model is overfitting noise. The AR and MA coefficients are fitting sample-specific patterns that do not generalize. Try a simpler order: `(1, 1, 0)` is the minimal non-trivial model. Try `(0, 1, 0)` as a sanity check that it reproduces naive.

The metric that may surprise you is **directional accuracy**. Even when ARIMA's MAE matches naive to within a few dollars, it will almost certainly produce a directional accuracy number — probably in the 0.49–0.52 range. That range is close enough to 0.5 that you should not read too much into it on a 437-bar sample (the standard error of an estimated proportion on 437 observations is about ±2.4%). But the infrastructure is in place: directional accuracy will be a meaningful differentiator once deeper models start showing consistent skill.

### AIC and BIC

The experiment also reports the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) from the `statsmodels` fit. These are only useful for comparing ARIMA orders fit on the **same training data** — they penalize model complexity and reward goodness of fit. They are useless for comparing ARIMA against XGBoost or LSTM. Use the sweep to pick an order; use the MAE on the test set to know if that order is actually better at forecasting.

One counterintuitive result worth watching for: the order that minimizes AIC on the training set is often not the order that minimizes test MAE. AIC measures in-sample fit; test MAE measures out-of-sample predictive accuracy. They disagree precisely because ARIMA on hourly BTC is fitting noise. The sweep table makes this visible.

## What both baselines tell us

The naive predictor tells us the price of doing nothing. ARIMA tells us the price of doing something linear. Together, they define the landscape:

- If a nonlinear model (XGBoost, LSTM, Transformer) cannot beat naive, it is finding no signal.
- If it beats naive but not by much, it is finding a small, noisy signal — potentially real but requiring careful backtest validation before trusting it.
- If it beats ARIMA while ARIMA itself cannot beat naive, the nonlinear model is doing something qualitatively different from ARIMA — but what, exactly, is the interesting question.

At hourly resolution, the signal-to-noise ratio is low enough that the baseline articles are themselves informative. They are not a preamble to the "real" results. They are the first real result: that predicting hourly Bitcoin price is hard, and the subsequent articles will be honest about how hard.

The 4-hour granularity used from article 3 onward smooths some of the intra-day noise. It does not make the problem easy — it makes it tractable. The baseline numbers will look different on 4-hour bars, because trend persistence is longer and autocorrelation structure is slightly richer. But the methodology is identical: same split, same metrics, same honest accounting against the floor.

Article 3 introduces the first nonlinear model. Bring features.
