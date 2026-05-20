# Forecasting Bitcoin with machine learning, Part 1: the baselines you need to beat

The single most useful thing a forecasting paper can do — and the thing the bad ones never do — is show you the dumbest possible predictor's number on the same data slice, in the same units, before introducing the clever one. This article is that number, plus the simplest non-trivial model that beats it. It is the floor for the rest of the series. Every model in articles 2 through 5 will have to clear what shows up here.

The headline: on 366 test bars at a 4-hour cadence on Bitcoin spot, the **naive predictor `close_pred[t] = close[t-1]` reports MAE 540.96 USD**. A linear ARIMA(3, 1, 3) — three autoregressive parameters, three moving-average parameters, one differencing step — reports **MAE 539.15 USD**. The gap is one dollar eighty-one. Every model later in this series will sit within $11 of those numbers in either direction, except one (the Temporal Fusion Transformer in article 4, which is $350 *worse* than naive). That gap is the entire story of articles 1 through 5.

## Methodology, in one paragraph

Target: one-step-ahead log-return `r_T = log(close_T / close_{T-1})`, reconstructed back to price as `close_pred = close_{T-1} * exp(r_pred)`. Data: `BTCUSDT` 4-hour bars from Binance Vision, pulled by [`scripts/fetch_data.py`](scripts/fetch_data.py). Split: train `2023-01-01` → `2024-08-01` UTC (≈ 3 834 bars), validation `2024-08-01` → `2024-10-01` UTC (used only by the deep-learning models in later articles), test `2024-10-01` → `2024-12-01` UTC (**366 bars**). All intervals half-open in UTC. Walk-forward, one step ahead, weights frozen across the test window. Metrics computed by a single shared module — [`src/btc_ai/eval/metrics.py`](src/btc_ai/eval/metrics.py) — so an "MAE" in one experiment means the same thing in every other. Reproduce: `make 01_baseline`, `make 02_arima`, `make 02_arima_sweep`. This is the same paragraph that opens articles 2, 3, 4, and 5, with the model line swapped.

## The naive last-value baseline

`close_pred[t] = close[t-1]`. No fit, no parameters, no walk-forward in the sense of fitting at every step — the entire experiment is one pass over the test bars. The actual prediction code is three lines:

```python
# experiments/01_baseline/run.py
close = df['close']
test = close.iloc[split:]
ref = close.iloc[split - 1:-1]
ref.index = test.index
y_pred = ref
```

That is the whole model. The reason 01_baseline exists is not the prediction — it is the architecture. Its `run.py`, its dataset config, its `results/` layout, and the metric module it calls are reused unchanged by every later experiment. By the time article 5 wraps up, six successor experiments will have inherited this scaffold and the only thing that differs row-to-row is the line that produces `y_pred`. That is what makes the leaderboard apples-to-apples.

Numbers, taken verbatim from [`experiments/01_baseline/results/btc_4h_2024/metrics.json`](experiments/01_baseline/results/btc_4h_2024/metrics.json):

| Metric | Value |
|---|---|
| MAE | **540.96 USD** |
| RMSE | 813.14 USD |
| MAPE | 0.6922 % |
| Directional accuracy | **NaN** |
| Cumulative return | — (omitted) |
| Annualized Sharpe | — (omitted) |

The NaN is not a bug. It is the metric module honestly reporting that the model expressed no opinion. From [`src/btc_ai/eval/metrics.py`](src/btc_ai/eval/metrics.py):

```python
def directional_accuracy(y_true: pd.Series, y_pred: pd.Series, ref: pd.Series) -> float:
        true_dir = np.sign(y_true.to_numpy() - ref.to_numpy())
        pred_dir = np.sign(y_pred.to_numpy() - ref.to_numpy())
        mask = (true_dir != 0) & (pred_dir != 0)
        if not mask.any():
                return float('nan')
        return float(np.mean(true_dir[mask] == pred_dir[mask]))
```

Because the naive predictor returns `close_pred[t] = close[t-1]` and the reference price `ref` is `close[t-1]`, `pred − ref` is exactly zero for every bar. The mask `(true_dir != 0) & (pred_dir != 0)` is empty. The function short-circuits to `NaN`. Cumulative return and Sharpe never get written to `metrics.json` for the same reason — the long-flat strategy never goes long, so there is nothing to compound.

It would be reasonable to call this a "design tax" on the naive baseline. I would argue the opposite: any forecasting harness whose metric module doesn't distinguish between "model said zero" and "model is missing" is hiding information that the leaderboard reader is going to need. We will see this `NaN` shape twice more in the series — once in the ARIMA sweep, where the `(0, 1, 0)` order is mathematically the random walk and lights up the same NaN pair; and once in the LSTM sweep, where a particular hyperparameter combination produces a model that never expresses a long position and degenerates the Sharpe to `NaN`. Recognize them as "no opinion" rather than "missing run."

## ARIMA in 90 seconds

ARIMA stands for AutoRegressive Integrated Moving Average. Three integers — `(p, d, q)` — pin down the architecture:

- **`p` (autoregressive lags).** Number of past values of the *differenced* series the model uses as inputs. Positive AR coefficients capture short-term momentum (recent up-moves tend to continue); negative AR coefficients capture mean reversion. For Bitcoin at the 4-hour horizon, AR is most useful as a noise filter — the linear AR(1) coefficient is small, but it is not zero.
- **`d` (differencing order).** Number of times to difference the series before modelling. `d = 1` means we model price *changes*, not price levels. This matters because the price level is non-stationary while one-step price changes are roughly stationary.
- **`q` (moving-average lags).** Number of past *residuals* (shocks) the model uses to clean up the current prediction. MA captures structure in the noise that AR can't.

On `BTCUSDT` at this slice, the model that performs best in the sweep is ARIMA(3, 1, 3) — three AR lags, one difference, three MA lags. Seven estimated parameters total. The whole thing is a linear fit on a differenced series.

The library is `statsmodels`. The walk-forward shape worth dwelling on:

```python
# 02_arima/run.py (sketch)
fitted = ARIMA(train['close'], order=(3, 1, 3)).fit()
extended = fitted.apply(full_series, refit=False)
y_pred = extended.predict(start=split, end=last, dynamic=False)
```

Two flags here are the ones that make this an honest walk-forward forecast:

- **`refit=False`** in `apply()`. The trained `(p, d, q)` coefficients are frozen at train-time values. The model sees the new observations as they arrive, but it does not update its parameters. This matches the protocol every later trained model in the series uses.
- **`dynamic=False`** in `predict()`. Each one-step-ahead prediction at bar `t` is computed from the *actual* past values, not from the model's own previous predictions. With `dynamic=True`, the predictions would chain into a multi-step forecast that compounds the model's earlier errors; that is a multi-horizon problem with a much harder shape.

Together, those two flags are the difference between a forecaster you can trust on a leaderboard and an unfalsifiable one. The same idea recurs in every later article: Darts' `historical_forecasts(retrain=False, last_points_only=True)` is the same protocol for the LSTM and the Temporal Fusion Transformer; the foundation models in article 5 use it too. The whole series shares one walk-forward shape.

## The sweep

ARIMA orders are not the kind of thing you handpick; you sweep them. The repo includes a sweep over twelve `(p, d, q)` orders along the canonical Box-Jenkins ramp, plus endpoints chosen to surface known pathologies. Source: [`experiments/02_arima/results/btc_4h_2024/sweep.csv`](experiments/02_arima/results/btc_4h_2024/sweep.csv).

| order | AIC | MAE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|
| (0, 1, 0) | 25 546.20 | 518.36 | NaN | 0.0000 | NaN |
| (1, 1, 0) | 25 547.86 | 517.58 | 0.5174 | 0.2785 | 5.20 |
| (0, 1, 1) | 25 547.87 | 517.61 | 0.5124 | 0.2619 | 4.91 |
| (1, 1, 1) | 25 549.66 | 517.16 | 0.5547 | 0.3314 | 6.06 |
| (2, 1, 0) | 25 549.35 | 516.87 | 0.5224 | 0.2316 | 3.97 |
| (0, 1, 2) | 25 549.35 | 516.88 | 0.5224 | 0.2321 | 3.98 |
| (2, 1, 2) | 25 553.35 | 516.87 | 0.5249 | 0.2335 | 4.00 |
| **(3, 1, 3)** | **25 543.88** | **514.07** | 0.5274 | **0.5101** | **6.40** |
| (5, 1, 0) | 25 554.83 | 517.06 | 0.5075 | 0.3577 | 5.49 |
| (0, 1, 5) | 25 554.89 | 517.02 | 0.5000 | 0.3500 | 5.14 |
| (5, 1, 5) | 25 550.49 | 522.58 | 0.5224 | 0.3312 | 4.59 |
| (1, 0, 1) | 25 571.25 | 521.05 | 0.4726 | 0.0087 | 0.56 |

(The sweep MAE values are on the older 401-bar slice that the sweep CSV was last regenerated against; the *single-fit canonical* MAE for ARIMA(3, 1, 3) on the current 366-bar slice is 539.15, taken from `metrics.json`. The relative ranking is unchanged by the slice rotation.)

Three things the sweep makes visible at once.

**First — the `(0, 1, 0)` row is the random walk, and it returns the naive baseline exactly.** MAE 518.36 — bit-identical to the older naive number on the same slice. That row is the sanity check: the random-walk specification, with no AR and no MA and one difference, is mathematically equivalent to "predict that next-bar's price equals this bar's price." If the pipeline were broken, those two numbers would diverge. They don't. The data loader, the metric module, and the prediction reconstruction agree across the two experiments.

**Second — the `(1, 0, 1)` row collapses on Sharpe.** It is the only un-differenced row in the sweep, and its Sharpe is 0.56 against the cluster's 4–6. The lesson is that **differencing is structural for Bitcoin price**; an ARMA model fitted on the price *level* rather than the differenced series is fitting non-stationarity instead of signal. `d ≥ 1` is non-optional. Most ARIMA introductions tell you this but few of them show what happens when you don't.

**Third — AIC and the out-of-sample Sharpe agree, here, on ARIMA(3, 1, 3).** Lowest AIC (25 543.88), lowest MAE (514.07), highest Sharpe (6.40), highest cumulative return (51 %). Five out of six leaderboard columns. In-sample model-selection criteria and out-of-sample trading metrics do not always agree (the global report has a separate, sharper case where they don't); on this slice, they do. Hyperparameter selection by AIC on this experiment would have landed on the same order as hyperparameter selection by Sharpe.

## The number

Canonical numbers for the article — taken verbatim from [`experiments/02_arima/results/btc_4h_2024/metrics.json`](experiments/02_arima/results/btc_4h_2024/metrics.json):

| Metric | ARIMA(3, 1, 3) |
|---|---|
| AIC | 58 271.35 |
| BIC | 58 315.11 |
| MAE | **539.15 USD** |
| RMSE | 808.79 USD |
| MAPE | 0.6903 % |
| Directional accuracy | 0.5082 |
| Cumulative return | 0.5269 |
| Annualized Sharpe | **6.8559** |

The MAE win over naive is $1.81. Three AR parameters and three MA parameters moved the prediction $1.81 closer to the price on average. Stated like that it sounds tiny. Read in the context of the series it is the difference between "model is the naive predictor" and "model is at the top of the leaderboard on five out of six trading metrics" — including beating every deep-learning model in articles 3 and 4 on Sharpe.

A per-bar Sharpe significance back-of-the-envelope: 4h has 2 190 bars per year, so per-bar Sharpe = 6.8559 / √2190 = 0.1465. The standard error on the per-bar mean, treating the test bars as i.i.d., is roughly 1 / √366 = 0.0523. Ratio ≈ **2.80 σ** — significant at about the 99 % level. The i.i.d. assumption is optimistic on 4h crypto (volatility clustering inflates the true standard error), so read 2.80 σ as an upper bound on real significance rather than a clean number. The right interpretation is "this Sharpe is not zero on this slice"; the wrong interpretation is "this strategy will produce 6.86 Sharpe out of sample."

## What this article tells us about the model class

Three load-bearing observations.

**Linear AR / MA on differenced Bitcoin price is a remarkably competitive baseline at this horizon.** It is *the* baseline every later model will be compared against. The MAE gap from naive is small, but it is real, and on the trading metrics it opens up — Sharpe goes from "undefined" to 6.86 (or 6.06 at the original (1, 1, 1) order). Articles 3 and 4 will train models with thousands of parameters on the same slice and not match this Sharpe.

**The right differencing order matters more than the AR / MA depth.** The `(1, 0, 1)` row of the sweep is the proof: a richer linear model on the wrong (un-differenced) representation collapses to Sharpe 0.56. The `(5, 1, 5)` row is the converse: too rich on the right representation and you start overfitting (Sharpe 4.59, worse than the cheaper (3, 1, 3)). The lesson generalizes well past ARIMA — *what* the model is asked to predict and *what* representation it is asked to predict from are the load-bearing choices, more than the parameter count.

**AIC and Sharpe sometimes agree, and sometimes they don't.** Here they agree on `(3, 1, 3)`. The point is not "use AIC to pick your model"; the point is that any selection rule you use is one of several you could have used, and they can rank differently. A sweep that produces six columns of metrics tells you that the column you cared about is one column out of six. The series treats this as a recurring methodology beat: in article 2 the XGBoost sweep is on a stale slice but the *configured default* is not the sweep's Sharpe winner; in article 3 the LSTM sweep's MAE winner and Sharpe winner are at *opposite ends* of the capacity axis. Defaults set by `config.yaml` are starting points, not optima.

## Regime caveat

The 6.86 Sharpe is conditional on a test window — `2024-10-01` → `2024-12-01` UTC — that lands precisely on the post-election Bitcoin rally. A long-flat strategy with mild directional skill (`dir_acc` = 0.508, only one point above coin-flip) compounds favourably on a strongly trending up-regime because the bars it correctly stays long collect the drift, and the bars it goes flat forgo profit rather than realize a loss. Run the same model on a sideways or down-trending window and the Sharpe figure will be very different, possibly negative. Nothing in this article (and nothing in the series) has been tested out-of-regime. Read every positive Sharpe in articles 1 through 5 as *"skill conditional on this regime"* — and watch for the moment in article 4 where a Sharpe of 2.5 still passes the significance threshold purely because the slice is so favourable.

## Reproduce

```
make 01_baseline
make 02_arima
make 02_arima_sweep
```

The first writes [`experiments/01_baseline/results/btc_4h_2024/metrics.json`](experiments/01_baseline/results/btc_4h_2024/metrics.json) — the naive numbers above. The second writes [`experiments/02_arima/results/btc_4h_2024/metrics.json`](experiments/02_arima/results/btc_4h_2024/metrics.json) — the ARIMA(3, 1, 3) numbers above (the default config in the YAML is `order: (3, 1, 3)`). The third writes [`experiments/02_arima/results/btc_4h_2024/sweep.csv`](experiments/02_arima/results/btc_4h_2024/sweep.csv) — the table earlier in this article.

Article 2 introduces the first non-linear model and the first one with engineered features. It is also the first one that loses to ARIMA on a metric that the series cares about. Onwards.
