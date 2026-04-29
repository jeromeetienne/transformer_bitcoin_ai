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
| end | `2024-04-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| window | **24** (= one day for hourly data) |

Total bars: **2 184**. Test slice: **437 bars**.

## Results

From [experiments/01b_moving_average/results/metrics.json](../../experiments/01b_moving_average/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | **1 012.86** USD |
| RMSE | 1 356.63 USD |
| MAPE | 1.5064 % |
| Directional accuracy | **0.4828** |

## Cross-experiment comparison

Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc |
|---|---|---|---|---|
| 01_baseline_naive | **337.89** | 471.54 | 0.502 % | NaN |
| **01b_moving_average (window=24)** | **1012.86** | 1356.63 | 1.506 % | 0.483 |
| 02_arima (1, 1, 1) | 338.15 | 471.49 | 0.503 % | 0.492 |

MA(24) is **3× worse than naive on MAE** and **0.483 on directional accuracy** — slightly *anti-correct*.

## Interpretation

- **`mae` = 1012.86, ~3× naive's 337.89.** Exactly the predicted outcome. A 24-bar (one-day) window is a long lag on a series whose hourly moves are roughly random walk. The lag dominates. Smaller windows would be closer to naive; larger windows worse still.
- **`rmse` = 1356.63.** Same story, with the gap-vs-MAE indicating that on volatile hours the lag-induced miss is even bigger.
- **`mape` = 1.51 %** — a multiple-percent average error per bar. Roughly the cost of using the wrong tool.
- **`directional_accuracy` = 0.4828 < 0.5.** The mean-reversion bet *loses* on this slice. When MA(24) said "I think it'll go down" (because recent prices were above the 24-bar mean), the actual move was up more often than down. Reading: hourly BTC in this Q1 2024 sample shows mild **momentum**, not mean-reversion. The deficit (1.7 %) is small enough that on 437 bars it's borderline noise — but the sign is clearly < 0.5, not above.

### Choosing `window`

If you re-ran the experiment over a sweep of windows, the picture would be:

- Small (3, 5) → MAE close to naive, directional signal near zero.
- Medium (24) → today's run; clear lag, meaningful directional reading.
- Large (168 = one week) → MAE blows up further; useful only as a "any model worse than this is broken" lower bound.

## Why it still earns its place

Even with MAE far above naive, the experiment **fixed** the directional-accuracy axis. Without MA, `02_arima` has only naive (NaN) and itself to compare against on direction. With MA, we know that "predict-the-recent-mean" gives 0.483 on this slice, so any later model claiming directional skill (>0.5) has a meaningful baseline to beat — not just zero, but a model that *did* take positions, and slightly lost.

## Files produced

- [experiments/01b_moving_average/results/metrics.json](../../experiments/01b_moving_average/results/metrics.json)
- [experiments/01b_moving_average/results/predictions.parquet](../../experiments/01b_moving_average/results/predictions.parquet)
- [experiments/01b_moving_average/results/plot.png](../../experiments/01b_moving_average/results/plot.png)

## How to reproduce

```
make 01b_moving_average
```
