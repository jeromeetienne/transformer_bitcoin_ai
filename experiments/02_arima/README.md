# 02_arima

First "real" model in the lineup. Fits an **ARIMA(p, d, q)** to the training slice and produces walk-forward one-step-ahead forecasts on the held-out test slice.

```
ARIMA(p, d, q)
   p : AR — number of autoregressive lags
   d : I  — differencing order (d=1 → model the price *changes*, not the price)
   q : MA — number of moving-average lags (on the residuals)
```

ARIMA is essentially a linear, stationary model of price *increments*. It can capture short-term momentum (positive AR coefficient → recent up-moves expected to continue) or mean-reversion (negative AR), plus a short residual-correction term via MA. With `d=1` it sidesteps the non-stationarity of the price level itself.

## Why this baseline

Naive sets the floor on point error. MA gave us a directional signal to compare. ARIMA is the first model that actually *estimates parameters from data* — if it can't beat naive on MAE, no fancier sequence model is going to help, because the dynamics are too noisy to learn from price alone.

It is also the cleanest place to see how the architecture handles model fitting + walk-forward evaluation, which every later experiment will replicate.

## How it's done

1. Load OHLCV via the shared loader ([src/btc_ai/data](../../src/btc_ai/data)).
2. Time-split: hold out the last `test_fraction` (default 20%) as the test set.
3. Fit `ARIMA(close_train, order=(p, d, q))` on the training slice — parameters estimated **on training data only**, no leakage.
4. Extend the fitted model with the test slice via `fit.apply(full_series, refit=False)` — keeps the trained `(p, d, q)` parameters, just lets the model see the new observations.
5. Call `predict(start=split, end=last, dynamic=False)` — `dynamic=False` is the key flag: each prediction at time `t` is computed from **actual** past values, not from previously predicted values. That's true 1-step-ahead walk-forward.
6. Compute MAE, RMSE, MAPE, directional accuracy. Also report the model's `AIC` and `BIC` for cross-order comparison.
7. Write `results/metrics.json`, `results/predictions.parquet`, `results/plot.png`.

Library: **`statsmodels`** (`statsmodels.tsa.arima.model.ARIMA`). Chosen for explicit `(p, d, q)` control and straightforward walk-forward semantics. Future variants could swap in `pmdarima` (auto-ARIMA) or Nixtla's `statsforecast` (faster, multi-series).

## How to run

```
make arima
```

Try other orders by editing `config.yaml`:

```yaml
order: [1, 1, 0]   # pure AR(1) on first differences
order: [1, 1, 1]   # default — Box-Jenkins classic
order: [2, 1, 2]   # richer; risks overfitting on short samples
```

## How to interpret

Compare `mae` against the previous experiments **on the same data slice**:

| | Naive | MA(24) | ARIMA(p,d,q) |
|---|---|---|---|
| What | `pred = close[t-1]` | `pred = mean(prev 24)` | learned from data |
| MAE on default slice | 337.89 (the floor) | 1012.86 | *this experiment* |
| Directional accuracy | NaN (no opinion) | ~0.48 (slight anti-trend) | *this experiment* |

What the numbers tell you:

- **`mae` close to naive's `mae`** — this is the typical outcome on hourly BTC. ARIMA is a linear model, hourly BTC is near-random-walk, so the best ARIMA can do is approximate naive. Don't be disappointed; this is the *truth* about the signal-to-noise ratio at this timescale.
- **`mae` clearly below naive** — surprising and worth investigating. Suspect leakage first (verify split logic), then check whether the test slice is unusually trending.
- **`mae` above naive** — the chosen `(p, d, q)` is overfitting noise. Try `[1, 1, 0]` or larger `d`.
- **`directional_accuracy > 0.50`** — ARIMA is finding *some* short-term momentum/mean-reversion signal. Even small wins (51–53%) compound in a backtest.
- **`aic` / `bic`** — only useful for comparing different `(p, d, q)` choices on the *same* training data. Lower is better; they penalise extra parameters. Use them when sweeping orders, not for cross-experiment comparison.

### Caveats baked into this v1

- Single fit, no order search. No AutoARIMA. We pick `(p, d, q)` deliberately.
- Walk-forward without re-estimation. Each prediction uses real past data but the AR/MA coefficients are frozen at the train-time values. A more rigorous version would re-fit at every step (slower, marginally more accurate).
- No exogenous regressors yet (ARIMA**X**). Volume, on-chain features, and sentiment will be added in later experiments.
- ARIMA assumes residuals are roughly Gaussian and homoskedastic. BTC returns are neither. Heavy tails and volatility clustering mean point forecasts may look reasonable while interval forecasts (not produced here) would be badly miscalibrated.

## Sweeping orders

For comparing multiple `(p, d, q)` choices on the same data slice without re-running the full experiment each time:

```
make arima-sweep
```

Edit the `ORDERS` list at the top of [sweep.py](sweep.py) to change which orders are tried. Output: a printed table to stdout (sorted in source order, easy to scan) and a `results/sweep.csv` for downstream analysis. The sweep does not touch `metrics.json` / `predictions.parquet` / `plot.png` — those reflect the single order in `config.yaml` set via `make arima`.

What to look at in the sweep:

- **MAE on the same test slice** — the only number that says "this order predicts better than that one." Differences within a few dollars on a $338 baseline are noise on a 437-bar test set.
- **AIC / BIC** — useful for picking between orders fit on the *same* training data, not for cross-experiment comparison. **Watch out:** AIC's favorite order is often *not* the test-MAE winner. AIC rewards in-sample fit; MAE measures out-of-sample. They disagree on small samples.
- **The (0, 1, 0) row should match naive exactly** — it's the random walk, mathematically equivalent to last-value. Acts as a pipeline sanity check.

## Files

```
experiments/02_arima/
├── README.md
├── config.yaml         # data slice + test_fraction + order=(p,d,q)
├── run.py              # entry point — fits the single order in config.yaml
├── sweep.py            # sweeps multiple (p,d,q) orders, writes results/sweep.csv
└── results/
    ├── metrics.json        # produced by run.py
    ├── predictions.parquet # produced by run.py
    ├── plot.png            # produced by run.py
    └── sweep.csv           # produced by sweep.py
```
