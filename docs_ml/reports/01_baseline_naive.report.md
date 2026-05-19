# Report — `01_baseline_naive`

**Run date:** 2026-05-19
**Status:** completed (single fit, no sweep — the experiment has no parameters to sweep)

## What this experiment is

The dumbest possible forecaster: predict that the next price equals the current price. Exists to set the floor every later model on the same data slice must beat — if 02 / 03 / 04 / 05 / 06 cannot lower MAE relative to this, they aren't earning their complexity. The experiment also doubles as the architecture probe: the data loader, config schema, metrics module, and `results/` layout used here are reused unchanged by every later experiment.

```
close_pred[t] = close[t-1]
```

No training, no parameters, no walk-forward in the model-fitting sense — predictions are computed in a single pass over the test slice. Library: pandas. See [experiments/01_baseline_naive/README.md](../../experiments/01_baseline_naive/README.md) for the full narrative.

## Configuration

From [experiments/01_baseline_naive/configs/btc_4h_2024.config.yaml](../../experiments/01_baseline_naive/configs/btc_4h_2024.config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `4h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |

Total bars: **2 010**. Test: **402**. (No train slice — there is nothing to fit.)

## Results — single fit

From [experiments/01_baseline_naive/results/btc_4h_2024/metrics.json](../../experiments/01_baseline_naive/results/btc_4h_2024/metrics.json):

| Metric | Value |
|---|---|
| MAE | **518.36** USD |
| RMSE | 784.09 USD |
| MAPE | 0.6701 % |
| Directional accuracy | **NaN** (by design — see Interpretation) |
| Cumulative return | — (not produced; the naive predictor never expresses a direction) |
| Annualized Sharpe | — |

## Cross-experiment comparison

4h-slice leaderboard. Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| **01_baseline_naive** | 518.36 | 784.09 | 0.6701 % | NaN | — | — |
| [01b_moving_average (window=24)](01b_moving_average.report.md) | 1 905.27 | 2 580.61 | 2.4782 % | 0.4801 | 0.0511 | 1.3003 |
| [02_arima (1, 1, 1)](02_arima.report.md) | **517.16** | **782.42** | **0.6686 %** | **0.5547** | 0.3314 | 6.0569 |
| [03_gradient_boosting (n_feat=31)](03_gradient_boosting.report.md) | 539.39 | 795.07 | 0.7008 % | 0.5365 | **0.5042** | **6.4233** |
| [04_lstm](04_lstm.report.md) | 522.29 | 785.45 | 0.6761 % | 0.4938 | 0.2596 | 4.2660 |

01_baseline_naive sets the floor on point error: MAE 518.36, RMSE 784.09. Only 02_arima clears it on MAE, and only by $1.20 — within sweep-to-sweep noise on a 402-bar test slice. Every other 4h row — including the two with the highest Sharpe — is *worse* on MAE than this naive predictor. The take-away: on 4h BTC, point-error and directional-skill are nearly orthogonal targets. (Experiments 05 and 06 currently report on a stale 1h slice — see [05_transformer.report.md](05_transformer.report.md) and [06_pretrained.report.md](06_pretrained.report.md).)

## Interpretation

1. **Directional accuracy is `NaN` and that is a feature, not a bug.** The naive predictor returns `close_pred[t] = close[t-1]`, which equals the reference price the metric uses. The metric's mask is `(true_dir != 0) & (pred_dir != 0)`; with `pred_dir = sign(0) = 0` everywhere, the mask is empty and the function short-circuits to `float('nan')` — see [src/btc_ai/eval/metrics.py:23-34](../../src/btc_ai/eval/metrics.py). A naive baseline expresses no opinion on direction by construction.
2. **MAE / RMSE / MAPE are the leaderboard's floor.** Every later 4h model is measured against this row. The gap between this MAE and 02_arima's MAE is *one dollar twenty* — that is the headline of the series so far.
3. **No Sharpe, no cumulative return.** Without a directional opinion there is no long/flat strategy to evaluate. The keys simply don't appear in this experiment's `metrics.json` — they don't fail, they don't exist.
4. **The (0, 1, 0) row of 02_arima's sweep returns this MAE exactly** (518.358631840796). That row is the random walk; mathematically equivalent to last-value. It is the pipeline-sanity check for the whole architecture.

### Bottom line

The naive baseline reports **MAE 518.36 / RMSE 784.09 / NaN dir_acc** on a 402-bar 4h test slice covering approximately Sep 25 → Dec 1 2024 (the post-election BTC rally). It produces no Sharpe — there is no strategy. The interesting facts in this report are not its numbers; they are the *shape* of the metric set (NaN-by-design) and the absurdly narrow gap between this MAE and the best-fit ARIMA's MAE in the next experiment.

## Caveats

- Single split, no rolling-origin CV.
- No model — no parameters to fit, nothing to refit, no leakage to introduce.
- Regime sensitivity is inherited from the test slice (post-election BTC rally); a different test window would produce a different MAE floor.

## Files produced

- [experiments/01_baseline_naive/results/btc_4h_2024/metrics.json](../../experiments/01_baseline_naive/results/btc_4h_2024/metrics.json)
- [experiments/01_baseline_naive/results/btc_4h_2024/predictions.parquet](../../experiments/01_baseline_naive/results/btc_4h_2024/predictions.parquet)
- [experiments/01_baseline_naive/results/btc_4h_2024/plot.png](../../experiments/01_baseline_naive/results/btc_4h_2024/plot.png)

## How to reproduce

```
make 01_baseline_naive
```
