# Report — `01_baseline_naive`

**Run date:** 2026-04-29
**Status:** completed

## What this experiment is

A naive last-value forecaster: predicts that the next price equals the current price.

```
close_pred[t] = close[t-1]
```

No training, no parameters, no tuning. The point is the architecture, not the model — every later experiment reuses the same shared loader, metrics, and `results/` layout. Its job is to set the **floor** that every later model must beat to be worth using. See [experiments/01_baseline_naive/README.md](../../experiments/01_baseline_naive/README.md) for the full description.

## Configuration

From [experiments/01_baseline_naive/config.yaml](../../experiments/01_baseline_naive/config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `1h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-04-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |

Total bars loaded: **2 184** (91 days × 24 h).
Test slice (last 20%): **437 bars** (~18 days, mid-March → end of March 2024).

## Results

From [experiments/01_baseline_naive/results/metrics.json](../../experiments/01_baseline_naive/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | **337.89** USD |
| RMSE | 471.54 USD |
| MAPE | 0.5024 % |
| Directional accuracy | **NaN** (by design — see below) |

## Cross-experiment comparison

Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | vs naive (MAE) |
|---|---|---|---|---|---|
| **01_baseline_naive** | **337.89** | 471.54 | 0.502 % | NaN | — |
| 01b_moving_average (window=24) | 1012.86 | 1356.63 | 1.506 % | 0.483 | +674.97 |
| 02_arima (1, 1, 1) | 338.15 | 471.49 | 0.503 % | 0.492 | +0.27 |
| 03_gradient_boosting (default) | 354.60 | 493.14 | 0.528 % | **0.519** | +16.71 |

Naive is still **the floor on MAE**; nothing has beaten it on this slice yet. But 03 is the first model with `dir_acc > 0.5` — the leaderboard is now two-axis.

## Interpretation

- **`mae` = 337.89** — average absolute error per bar, in USD. With BTC at ~$60–70k during the test slice, that's roughly half a percent. Sounds tiny, but it's the bar every later experiment must clear.
- **`rmse` (471.54) > `mae` (337.89)** — normal. RMSE penalises big misses harder, so the gap reflects volatile hours where errors were larger than average.
- **`mape` = 0.50 %** — same error expressed as a percentage of price. On hourly BTC, naive is typically in the 0.4–0.6 % band.
- **`directional_accuracy` = NaN** — the metric returns `NaN` when the model never expresses a direction (predicted move is exactly flat). Naive predicts `close[t] = close[t-1]`, so the predicted *direction* is always zero. The honest answer is "no opinion." This is by design (see [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py)) — a 0% reading would have suggested the model was *anti-correct*, which it isn't; it's silent.

### The non-obvious bit

BTC hourly is *very close to a random walk*. Beating MAE 337.89 with a model that only sees price is genuinely hard — that's not a flaw in any subsequent model, it's a statement about signal-to-noise at this timescale. Subsequent experiments confirm this: ARIMA(1,1,1) ties naive within $0.30 (see [02_arima.report.md](02_arima.report.md)) and gradient boosting on 31 engineered features (lags, rolling stats, volume, OHLC) is ~$17 *worse* than naive on MAE while being the first model to clear `dir_acc > 0.5` (see [03_gradient_boosting.report.md](03_gradient_boosting.report.md)). The next productive direction is **richer features that aren't price-derived** (on-chain, related-asset returns, funding/basis) or **non-linear models with much more data**.

## Files produced

- [experiments/01_baseline_naive/results/metrics.json](../../experiments/01_baseline_naive/results/metrics.json)
- [experiments/01_baseline_naive/results/predictions.parquet](../../experiments/01_baseline_naive/results/predictions.parquet)
- [experiments/01_baseline_naive/results/plot.png](../../experiments/01_baseline_naive/results/plot.png)

## How to reproduce

```
make 01_baseline_naive
```
