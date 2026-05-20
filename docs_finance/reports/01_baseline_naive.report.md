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
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |

Total bars loaded: **8 040** (335 days × 24 h).
Test slice (last 20%): **1 608 bars** (~67 days, late Sep → end of Nov 2024).

## Results

From [experiments/01_baseline_naive/results/metrics.json](../../experiments/01_baseline_naive/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | **260.50** USD |
| RMSE | 396.97 USD |
| MAPE | 0.3407 % |
| Directional accuracy | **NaN** (by design — see below) |

A naive baseline cannot produce a strategy P&L (no opinion → no positions taken), so `cumulative_return` and `annualized_sharpe` are not reported.

## Cross-experiment comparison

Same data slice, same split, same metrics. The four price-aware models all sit within $14 of naive on MAE — at 1h resolution, **MAE is essentially measuring volatility and the random walk is the floor.** What separates the leaderboard now is directional accuracy and the strategy P&L derived from it.

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| **01_baseline_naive** | **260.50** | 396.97 | 0.341 % | NaN | — | — |
| **02_arima (1, 1, 1)** | 260.50 | **396.97** | 0.341 % | **0.5336** | **+53.02 %** | **+7.28** |
| 03_gradient_boosting (default) | 270.73 | 414.07 | 0.354 % | 0.4872 | +8.09 % | +1.45 |
| 04_lstm (Darts BlockRNN-LSTM) | 274.02 | 409.66 | 0.360 % | 0.5196 | +50.34 % | +4.95 |

Naive is still **the floor on MAE**, tied with ARIMA(1,1,1). Two models — ARIMA and the LSTM — have built non-trivial directional edges. XGBoost dropped below 50% directional accuracy on this wider window after looking like the leader on a Q1-only test slice; see [03_gradient_boosting.report.md](03_gradient_boosting.report.md) for the regime-fit analysis.

## Interpretation

- **`mae` = 260.50** — average absolute error per bar, in USD. With BTC averaging ~$60-70k across the test window, that's ~0.34 %. The bar every later experiment must clear.
- **`rmse` (396.97) > `mae` (260.50)** — normal. RMSE penalises big misses harder, so the gap reflects volatile hours where errors were larger than average.
- **`mape` = 0.34 %** — same error as a percentage of price. The lower number vs the earlier Q1-only window partly reflects a less volatile mid-2024 stretch in the data.
- **`directional_accuracy` = NaN** — the metric returns `NaN` when the model never expresses a direction (predicted move is exactly flat). Naive predicts `close[t] = close[t-1]`, so the predicted *direction* is always zero. The honest answer is "no opinion." This is by design (see [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py)) — a 0% reading would have suggested the model was *anti-correct*, which it isn't; it's silent.

### The non-obvious bit

BTC hourly is *very close to a random walk* on price. Beating MAE 260.50 with a price-only model is genuinely hard — that's not a flaw in any subsequent model, it's a statement about signal-to-noise at this timescale. ARIMA(1,1,1) ties naive's MAE to the cent (see [02_arima.report.md](02_arima.report.md)). Where models do separate is on **direction**: at this window's resolution, two of them (ARIMA, LSTM) produce a meaningfully positive trading edge while XGBoost lands below coin-flip.

The regime mix matters as much as model choice. The earlier Q1-only window had XGBoost looking like the leader; widening to Jan–Nov 2024 — which includes the post-halving rally and the Sep–Nov election rally — flipped the order. None of the dir_acc / Sharpe numbers should be trusted as forward estimates yet.

## Files produced

- [experiments/01_baseline_naive/results/metrics.json](../../experiments/01_baseline_naive/results/metrics.json)
- [experiments/01_baseline_naive/results/predictions.parquet](../../experiments/01_baseline_naive/results/predictions.parquet)
- [experiments/01_baseline_naive/results/plot.png](../../experiments/01_baseline_naive/results/plot.png)

## How to reproduce

```
make 01_baseline_naive
```
