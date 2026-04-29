# 01b_moving_average

Sibling baseline to [`01_baseline_naive`](../01_baseline_naive/). Predicts the next price as the **mean of the previous `window` closes**:

```
close_pred[t] = mean(close[t-window : t])
```

The `b` suffix marks this as a *companion* to `01`, not a new generation: same data slice, same metrics, just a different (still trivially simple) predictor. With `window: 1` it degenerates to naive last-value — useful as a sanity check.

## Why it exists

Two reasons:

1. **A second floor for ARIMA et al. to clear.** Naive sets the trivial floor; MA sets a slightly less trivial one. ARIMA without seasonal terms is essentially a fancy weighted MA, so if `02_arima` doesn't beat MA, the model isn't earning its complexity.
2. **Directional skill, not just point error.** Naive returns `directional_accuracy: NaN` because it never expresses a direction. MA *does* express a direction: when prices have been rising, `MA(window) < close[t-1]`, so the prediction is *below* the reference and reads as "predicted to go down." This makes MA mechanically a **mean-reverter** — it bets against the recent trend. On crypto's hourly noise, that's often a coin-flip, but the metric is meaningful here, unlike for naive.

## How it's done

Same pipeline as `01_baseline_naive`:

1. Load OHLCV via the shared loader ([src/btc_ai/data](../../src/btc_ai/data)).
2. Time-split: hold out the last `test_fraction` (default 20%) as the test set.
3. For each test bar `t`, predict `mean(close[t-window:t])` (computed once via `close.shift(1).rolling(window).mean()` so each prediction uses only data strictly before `t`).
4. Compute MAE, RMSE, MAPE, directional accuracy.
5. Write `results/metrics.json`, `results/predictions.parquet`, `results/plot.png`.

## How to run

From the repo root:

```
make moving-average
```

Override the window without editing the YAML:

```
uv run python experiments/01b_moving_average/run.py --config experiments/01b_moving_average/config.yaml
```

(Currently `window` lives in YAML only; no CLI override yet — keep configs as the single source of truth.)

## How to interpret

Compare the metrics against [`01_baseline_naive`](../01_baseline_naive/results/metrics.json) on the **same data slice**:

| Comparison | Reading |
|---|---|
| `mae(MA) < mae(naive)` | MA is smoothing out hourly noise — the moves it averages over are smaller than the noise. Possible on calm regimes; rare on crypto. |
| `mae(MA) > mae(naive)` | The MA lag is hurting more than it helps. Expected for crypto on small-to-medium windows because price moves are persistent on short horizons. |
| `directional_accuracy ≈ 0.5` | MA is a coin-flip on direction. Likely on hourly BTC. |
| `directional_accuracy < 0.5` | Mean-reversion bet is wrong: the recent trend keeps going. Suggests momentum, not mean-reversion. |
| `directional_accuracy > 0.5` | Mean-reversion bet is paying off: prices are oscillating around a level. |

### Choosing `window`

- **Small (e.g. 3, 5)** — barely smooths, behaves close to naive, lower MAE but near-zero directional signal.
- **Medium (e.g. 24 = one day for 1h data)** — the default. Strong enough to lag visibly; directional accuracy becomes a real number to read.
- **Large (e.g. 168 = one week)** — the lag dominates; MAE blows up. Useful as an anti-baseline (a model that's *worse* than this is doing something pathological).

## Files

```
experiments/01b_moving_average/
├── README.md           # this file
├── config.yaml         # data slice + test_fraction + window
├── run.py              # entry point
└── results/
    ├── metrics.json
    ├── predictions.parquet  # close + pred + ref on the test slice
    └── plot.png
```
