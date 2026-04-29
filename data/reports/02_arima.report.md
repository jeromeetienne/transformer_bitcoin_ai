# Report — `02_arima`

**Run date:** 2026-04-29
**Status:** completed (single fit + 12-order sweep)

## What this experiment is

The first model that *learns parameters from data*. Fits an `ARIMA(p, d, q)` on the training slice, then produces walk-forward one-step-ahead forecasts on the held-out test slice.

```
ARIMA(p, d, q)
   p : AR — autoregressive lags
   d : I  — differencing order (d=1 → model price changes, not the price)
   q : MA — moving-average lags on residuals
```

Library: `statsmodels.tsa.arima.model.ARIMA`. Walk-forward without re-fitting: parameters are estimated **on training data only**; the test slice is then fed via `fit.apply(full_series, refit=False)` and `predict(start=split, end=last, dynamic=False)` — each prediction at time `t` is computed from **actual** past values, not from previously predicted ones. See [experiments/02_arima/README.md](../../experiments/02_arima/README.md) for the full description.

## Configuration

From [experiments/02_arima/config.yaml](../../experiments/02_arima/config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `1h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-04-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| order | **`[1, 1, 1]`** (Box-Jenkins default) |

Total bars: **2 184**. Train: **1 747**. Test: **437**.

## Results — single fit, ARIMA(1, 1, 1)

From [experiments/02_arima/results/metrics.json](../../experiments/02_arima/results/metrics.json):

| Metric | Value |
|---|---|
| AIC | 24 870.62 |
| BIC | 24 887.02 |
| MAE | **338.15** USD |
| RMSE | 471.49 USD |
| MAPE | 0.5028 % |
| Directional accuracy | **0.4920** |

## Cross-experiment comparison

Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | vs naive |
|---|---|---|---|---|---|
| 01_baseline_naive | **337.89** | 471.54 | 0.502 % | NaN | — |
| 01b_moving_average (window=24) | 1012.86 | 1356.63 | 1.506 % | 0.483 | +674.97 |
| **02_arima (1, 1, 1)** | **338.15** | **471.49** | 0.503 % | 0.492 | **+0.27** |

ARIMA ties naive within $0.30 on MAE — within sample noise on a 437-bar test set.

## Order sweep

From [experiments/02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv) — same data slice, 12 orders:

| order | what | AIC | BIC | MAE | RMSE | MAPE | dir_acc |
|---|---|---|---|---|---|---|---|
| (3, 1, 3) | richer ARMA | 24871.7 | 24910.0 | **337.02** | 471.71 | 0.501 % | 0.508 |
| (0, 1, 0) | random walk = naive | 24867.4 | 24872.8 | 337.89 | 471.54 | 0.502 % | NaN |
| (1, 1, 0) | AR(1) on diffs | 24869.0 | 24879.9 | 338.10 | 471.57 | 0.503 % | 0.510 |
| (0, 1, 1) | MA(1) on diffs (exp smooth) | 24868.9 | 24879.9 | 338.11 | 471.57 | 0.503 % | **0.513** |
| (1, 1, 1) | Box-Jenkins default | 24870.6 | 24887.0 | 338.15 | **471.49** | 0.503 % | 0.492 |
| (1, 0, 1) | ARMA on **raw** price (no diff) | 24895.2 | 24917.0 | 338.21 | 471.49 | 0.503 % | 0.503 |
| (0, 1, 2) | MA(2) | 24870.1 | 24886.5 | 338.23 | **471.42** | 0.503 % | 0.497 |
| (2, 1, 0) | AR(2) | 24870.1 | 24886.5 | 338.26 | 471.44 | 0.503 % | 0.503 |
| (5, 1, 0) | AR(5) | 24872.2 | 24905.0 | 339.75 | 472.20 | 0.505 % | 0.503 |
| (0, 1, 5) | MA(5) | 24871.9 | 24904.7 | 339.91 | 472.19 | 0.505 % | 0.501 |
| (5, 1, 5) | rich both | 24869.7 | 24929.8 | 341.11 | 476.14 | 0.507 % | 0.499 |
| (2, 1, 2) | AIC favourite | **24863.2** | 24890.5 | 345.79 | 481.21 | 0.514 % | 0.478 |

## Interpretation

1. **All twelve orders sit within a $9 MAE band** ($337–346) around naive's $337.89. The signal isn't in `(p, d, q)` — the dynamics genuinely aren't there to extract from price alone at hourly resolution.

2. **(0, 1, 0) ≡ naive.** ARIMA with no AR, no MA, and first-differencing is mathematically a random walk. MAE matches naive exactly (337.89). Pipeline sanity check passed.

3. **AIC and out-of-sample MAE disagree, sharply.** (2, 1, 2) has the **best AIC** of the lot (24 863.2) but the **worst MAE** (345.79). Classic overfitting tell — AIC rewards in-sample fit, MAE measures what actually matters out-of-sample. **Don't pick orders by AIC alone on small samples.**

4. **(3, 1, 3) "wins" MAE by $0.87** vs naive — within noise on 437 test bars. Treat as a tie.

5. **Directional accuracy hovers at ~0.50 ± 0.02** across all orders. (0, 1, 1) hits 0.513 which sounds like skill but is well inside the standard error on this sample size.

6. **(1, 0, 1) — ARMA on raw price, no differencing** — didn't explode. The AR coefficient just absorbed the non-stationarity (probably converged near 1.0, locally indistinguishable from a random walk). AIC is markedly worse (+25) so the diagnostic *did* flag it; MAE didn't.

### Bottom line

**No ARIMA order on close-price-alone meaningfully beats naive on hourly BTC.** That isn't a bug in any model — it's a statement about signal-to-noise at this timescale. ARIMA's job in this lineup is to confirm the baseline, not to surpass it.

The next experiment that should actually win needs either:

- **Richer features** — volume, returns of correlated assets, on-chain metrics, basis to perp futures. ARIMA**X** (with exogenous regressors) is a near-zero-effort next step.
- **A non-linear model with much more data** — weekly/daily horizons or multi-symbol training to get enough samples for a transformer-style model to find weak signals.

## Caveats baked into v1

- Single fit, no order search — we pick `(p, d, q)` deliberately. AutoARIMA could be added as `02b_arima_auto` if useful.
- Walk-forward without re-estimation — each prediction uses real past data but the AR/MA coefficients are frozen at train-time values. Re-fitting at every step would be marginally more accurate, much slower.
- No exogenous regressors (ARIMA, not ARIMAX).
- ARIMA assumes residuals are roughly Gaussian and homoskedastic. BTC returns are neither — heavy tails and volatility clustering. Point forecasts may look reasonable while interval forecasts (not produced here) would be miscalibrated.

## Files produced

- [experiments/02_arima/results/metrics.json](../../experiments/02_arima/results/metrics.json) — single fit
- [experiments/02_arima/results/predictions.parquet](../../experiments/02_arima/results/predictions.parquet)
- [experiments/02_arima/results/plot.png](../../experiments/02_arima/results/plot.png)
- [experiments/02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv) — 12-order sweep

## How to reproduce

```
make 02_arima         # single fit using order in config.yaml
make 02_arima_sweep   # 12-order sweep (~30 s)
```
