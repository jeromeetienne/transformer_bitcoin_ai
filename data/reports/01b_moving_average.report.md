# Report — `01b_moving_average`

**Run date:** 2026-04-29
**Status:** completed

## What this experiment is

Sibling baseline to `01_baseline_naive`. Predicts the next price as the **mean of the previous `window` closes**:

```
close_pred[t] = mean(close[t-window : t])
```

The `b` suffix marks this as a *companion* to `01`, not a new generation. With `window: 1` it degenerates to naive last-value (sanity check). The headline reason for keeping it: naive returns `directional_accuracy: NaN` because it never expresses a direction. MA *does* — when prices have been rising, `MA(window) < close[t-1]`, so the prediction reads as "predicted to go down." That makes MA mechanically a **mean-reverter**, and the directional metric becomes meaningful for the first time. See [experiments/01b_moving_average/README.md](../../experiments/01b_moving_average/README.md) for the full description.

## Configuration

From [experiments/01b_moving_average/config.yaml](../../experiments/01b_moving_average/config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `1h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| window | **24** (= one day for hourly data) |

Total bars: **8 040**. Test slice: **1 608 bars**.

## Results

From [experiments/01b_moving_average/results/metrics.json](../../experiments/01b_moving_average/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | **756.19** USD |
| RMSE | 1 100.93 USD |
| MAPE | 0.9897 % |
| Directional accuracy | **0.5143** |
| Cumulative return | **+25.67 %** |
| Annualized Sharpe | **+4.33** |

## Cross-experiment comparison

Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| 01_baseline_naive | **260.50** | 396.97 | 0.341 % | NaN | — | — |
| **01b_moving_average (window=24)** | **756.19** | 1 100.93 | 0.990 % | **0.5143** | **+25.67 %** | **+4.33** |
| 02_arima (1, 1, 1) | 260.50 | 396.97 | 0.341 % | 0.5336 | +53.02 % | +7.28 |
| 03_gradient_boosting (default) | 270.73 | 414.07 | 0.354 % | 0.4872 | +8.09 % | +1.45 |
| 04_lstm (Darts BlockRNN-LSTM) | 274.02 | 409.66 | 0.360 % | 0.5196 | +50.34 % | +4.95 |

MA(24) is **~3× worse than naive on MAE** but lands clearly on the *positive* side of directional accuracy with a strategy Sharpe of +4.33 — a reversal of the Q1-only result, which had MA at 0.4828 / Sharpe -3.08. Same model, same window size, same test_fraction; what changed is the regime mix in the test slice.

## Interpretation

- **`mae` = 756.19, ~3× naive's 260.50.** Exactly the predicted outcome. A 24-bar (one-day) window is a long lag on a series whose hourly moves are roughly random walk. The lag dominates. Smaller windows would be closer to naive; larger windows worse still.
- **`rmse` = 1100.93.** Same story, with the gap-vs-MAE indicating that on volatile hours the lag-induced miss is even bigger.
- **`mape` = 0.99 %** — about one percent average error per bar. Roughly the cost of using the wrong tool for the level forecast.
- **`directional_accuracy` = 0.5143 > 0.5.** On Q1-only this was 0.483 (anti-correct); over Jan–Nov 2024 it inverted. MA(24) is structurally a **mean-reverter**: it bets "down" after a run-up and "up" after a sell-off. That bet *loses* in trends and *wins* in chop. The Q1 window was a clean uptrend, the wider Jan–Nov window mixes that with the Apr–Jul drawdown and the Sep–Nov rally — so the mean-reversion bet recovers a small directional edge.
- **`annualized_sharpe` = +4.33** on 1 608 test bars. Per-bar Sharpe ≈ 0.046, SE ≈ √(1/1 608) ≈ 0.025 — about 2σ from zero. Real but modest evidence; the annualization (× √8 760) is what makes the headline number look impressive.

### What changed from the Q1-only window

| Metric | Q1 (432 bars) | Jan–Nov (1 608 bars) |
|---|---|---|
| MAE | 1 012.86 | 756.19 |
| dir_acc | 0.4828 | **0.5143** |
| cum_ret | -8.90 % | **+25.67 %** |
| sharpe | -3.08 | **+4.33** |

Sign-flip on Sharpe is regime, not model. A pure-MA forecaster has zero learned parameters; nothing in this experiment "improved." The wider window *included* the Q1 drawdown for mean-reversion **and** several other regimes that happened to favour the MA bet. Don't read this as MA(24) being a useful predictor — read it as a reminder that **dir_acc and Sharpe over a single test slice are regime-dependent observations, not model-quality measurements.**

### Why it still earns its place

Even with MAE far above naive, the experiment **fixed** the directional-accuracy axis. Without MA, `02_arima` has only naive (NaN) and itself to compare against on direction. With MA, every later model has a structurally-naive directional baseline to compare against — a model that *did* take positions, and slightly won this time around.

## Files produced

- [experiments/01b_moving_average/results/metrics.json](../../experiments/01b_moving_average/results/metrics.json)
- [experiments/01b_moving_average/results/predictions.parquet](../../experiments/01b_moving_average/results/predictions.parquet)
- [experiments/01b_moving_average/results/plot.png](../../experiments/01b_moving_average/results/plot.png)

## How to reproduce

```
make 01b_moving_average
```
