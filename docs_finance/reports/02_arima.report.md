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
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| order | **`[1, 1, 1]`** (Box-Jenkins default) |

Total bars: **8 040**. Train: **6 432**. Test: **1 608**.

## Results — single fit, ARIMA(1, 1, 1)

From [experiments/02_arima/results/metrics.json](../../experiments/02_arima/results/metrics.json):

| Metric | Value |
|---|---|
| AIC | 93 381.54 |
| BIC | 93 401.85 |
| MAE | **260.50** USD |
| RMSE | 396.97 USD |
| MAPE | 0.3407 % |
| Directional accuracy | **0.5336** |
| Cumulative return | **+53.02 %** |
| Annualized Sharpe | **+7.28** |

## Cross-experiment comparison

Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| 01_baseline | **260.50** | 396.97 | 0.341 % | NaN | — | — |
| **02_arima (1, 1, 1)** | **260.50** | **396.97** | 0.341 % | **0.5336** | **+53.02 %** | **+7.28** |
| 03_gradient_boosting (default) | 270.73 | 414.07 | 0.354 % | 0.4872 | +8.09 % | +1.45 |
| 04_lstm (Darts BlockRNN-LSTM) | 274.02 | 409.66 | 0.360 % | 0.5196 | +50.34 % | +4.95 |

ARIMA ties naive **to the cent** on MAE and **leads the leaderboard** on every directional metric over Jan–Nov 2024. This is a sharp inversion of the Q1-only result, where ARIMA was a draw with naive across the board (dir_acc 0.492, Sharpe +0.45). Same model, same order; the difference is the test window covers more regime variation.

## Order sweep

From [experiments/02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv) — same data slice, 12 orders:

| order | what | AIC | BIC | MAE | RMSE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|---|
| **(1, 1, 0)** | AR(1) on diffs | 93 379.5 | 93 393.1 | **260.49** | **396.96** | **0.5373** | **+53.50 %** | **+7.52** |
| **(0, 1, 1)** | MA(1) on diffs (exp smooth) | 93 379.5 | 93 393.1 | **260.49** | **396.96** | **0.5373** | **+53.50 %** | **+7.52** |
| (1, 1, 1) | Box-Jenkins default | 93 381.5 | 93 401.8 | 260.50 | 396.97 | 0.5336 | +53.02 % | +7.28 |
| (0, 1, 0) | random walk = naive | **93 377.5** | **93 384.3** | 260.50 | 396.97 | NaN | 0.000 % | NaN |
| (2, 1, 0) | AR(2) | 93 379.5 | 93 399.8 | 260.59 | 397.20 | 0.5019 | +10.09 % | +1.73 |
| (0, 1, 2) | MA(2) | 93 379.5 | 93 399.8 | 260.59 | 397.19 | 0.5025 | +10.76 % | +1.83 |
| (2, 1, 2) | richer ARMA | 93 383.3 | 93 417.2 | 260.67 | 397.25 | 0.4876 | +15.16 % | +2.38 |
| (5, 1, 0) | AR(5) | 93 385.0 | 93 425.6 | 260.68 | 397.21 | 0.4956 | +17.87 % | +2.89 |
| (0, 1, 5) | MA(5) | 93 385.1 | 93 425.7 | 260.67 | 397.21 | 0.4975 | +18.38 % | +2.96 |
| (3, 1, 3) | richer ARMA | 93 386.8 | 93 434.2 | 260.77 | 397.28 | 0.4882 | +18.48 % | +2.89 |
| (5, 1, 5) | rich both | 93 379.1 | 93 453.6 | 261.24 | 397.92 | 0.4826 | +8.72 % | +1.46 |
| (1, 0, 1) | ARMA on **raw** price (no diff) | 93 404.0 | 93 431.0 | 261.47 | 398.41 | 0.4820 | +0.99 % | +2.88 |

## Interpretation

1. **AR(1) on diffs and MA(1) on diffs are tied for the lead** — `(1, 1, 0)` and `(0, 1, 1)` produce identical AIC/BIC, identical MAE, identical dir_acc 0.5373, and identical Sharpe 7.52. With one parameter on differenced returns, the simplest possible learned signal is also the strongest one. They beat naive's MAE by 1¢ — within noise on the level metric — but produce a sharply non-trivial directional edge.

2. **(0, 1, 0) ≡ naive.** ARIMA with no AR, no MA, and first-differencing is mathematically a random walk. MAE matches naive exactly to two decimals; Sharpe is NaN because the model takes no positions. Pipeline sanity check passed.

3. **AIC and out-of-sample Sharpe disagree.** AIC's favourite is `(0, 1, 0)` (naive!) at 93 377.5 — pure parsimony. Out-of-sample, the order with the best AIC takes no positions and earns nothing, while the orders with one extra parameter — `(1, 1, 0)` / `(0, 1, 1)` — extract the directional edge. Don't pick orders by AIC alone when the goal is a trading signal.

4. **Adding AR/MA terms beyond 1 *hurts* directional accuracy.** Every order with AR or MA ≥ 2 is at or below 0.50 dir_acc except `(2, 1, 0)` and `(0, 1, 2)` (each barely above coin-flip). The richer orders ((2, 1, 2), (3, 1, 3), (5, 1, 5)) are anti-correct. Reading: on hourly BTC, only the most recent lag carries directional information; adding more terms fits noise.

5. **MAE bandwidth across orders is ~$1**, all sitting on top of naive's $260.50. The level-prediction problem is essentially solved by "predict the last price"; the *only* place models meaningfully separate is direction.

6. **(1, 0, 1) — ARMA on raw price, no differencing** — didn't explode. The AR coefficient absorbed the non-stationarity (probably converged near 1.0). AIC is markedly worse (+27) so the diagnostic *did* flag it; MAE didn't.

### Bottom line

**An AR(1)-on-differences signal is the leaderboard's leading directional edge over Jan–Nov 2024.** It ties naive on MAE (as expected — random-walk floor) and produces dir_acc 0.5373 / Sharpe +7.52. That said, the test window (1 608 bars, ~67 days) includes the Sep–Nov post-halving + election rally, which is structurally favourable to any "go-with-momentum" rule. Per-bar Sharpe ≈ 0.080 with SE ≈ √(1/1 608) ≈ 0.025 — about 3σ above zero. Real evidence, but not yet falsified against a bear-leg test window.

Compare to the Q1-only result (Jan–Mar 2024 only, 437 test bars), where the ARIMA leaderboard had `(0, 1, 1)` peaking at dir_acc 0.513 and Sharpe ~0.5. Same family of models, totally different rank order. The lesson is about test-window length and regime mix, not model choice.

The next experiment that should win the leaderboard structurally needs:

- **A test window that includes a real bear leg** (e.g. mid-2022) so we can confirm the directional edge holds out of regime.
- **Richer features** — funding rate, perp basis, related-asset returns, on-chain. ARIMAX is a near-zero-effort step. (`03_gradient_boosting` adds engineered features and *underperforms* on this window — see [03_gradient_boosting.report.md](03_gradient_boosting.report.md).)
- **Sequence models** (LSTM, transformer) — `04_lstm` lands at 0.5196 / +4.95, behind ARIMA on this window. See [04_lstm.report.md](04_lstm.report.md).

## Caveats

- Single split, no rolling-origin CV. Hyperparameter rankings could shift on a different test window — and we already know they do, comparing Q1-only vs Jan–Nov.
- Walk-forward without re-estimation — parameters are frozen at train-time values. Re-fitting at every step would be marginally more accurate, much slower.
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
