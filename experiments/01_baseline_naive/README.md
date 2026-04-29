# 01_baseline_naive

The dumbest possible forecaster: **predict that the next price equals the current price.**

```
close_pred[t] = close[t-1]
```

This experiment exists to set the *floor* every later model must beat. If `02_arima`, `03_lstm`, `04_transformer_*` can't produce a lower MAE than this on the same test slice, they aren't doing anything useful. It also doubles as the architecture probe for the project: the data loader, config schema, metrics, and `results/` layout used here are reused unchanged by every later experiment.

## How it's done

1. Load OHLCV via the shared loader ([src/btc_ai/data](../../src/btc_ai/data)) using the slice in `config.yaml` (`symbol`, `interval`, `start`, `end`, `period`).
2. Time-split: hold out the **last `test_fraction` (default 20%)** of the series as the test set. No random shuffle — order matters in time series.
3. For each test bar `t`, the prediction is the previous bar's close, `close[t-1]`.
4. Compute MAE, RMSE, MAPE, and directional accuracy on the test slice via [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py).
5. Write `results/metrics.json`, `results/predictions.parquet`, `results/plot.png`.

There is no training, no parameters, no hyperparameter tuning. That's the point.

## How to run

From the repo root:

```
make baseline
```

Or with a different config (e.g. a longer range, different symbol):

```
uv run python experiments/01_baseline_naive/run.py --config <path-to-yaml>
```

## How to interpret the results

Example output for the default config (`BTCUSDT 1h`, 2024-01-01 → 2024-04-01, test = last 20%):

```json
{
  "rows_total": 2184,
  "rows_test": 437,
  "mae": 337.89,
  "rmse": 471.54,
  "mape": 0.005024,
  "directional_accuracy": NaN
}
```

| Metric | What it means | How to read it |
|---|---|---|
| `mae` | Mean absolute error in price units (USD here) | The *floor*. Every later experiment on this same config must beat this number. |
| `rmse` | Root mean squared error, also in USD | `rmse > mae` is normal — it weights big misses harder, so the gap reveals how spiky the volatile hours were. |
| `mape` | Mean absolute % error | Naive on hourly BTC is typically 0.4–0.6%. Sounds tiny; at $65k that's still hundreds of dollars per bar. |
| `directional_accuracy` | Fraction of bars where the predicted move sign matches the actual move sign | `NaN` here by design — naive predicts "no change", so it never expresses a direction. Real models will return a number; 0.5 = coin-flip, >0.5 = real skill. |

### The non-obvious bit

BTC hourly is *very close to a random walk*, so beating MAE 337.89 is genuinely hard. A "good" later model isn't one with low MAE in absolute terms — it's one whose MAE on the **same test slice** is meaningfully below this baseline.

A model that ties naive on MAE but achieves directional accuracy > 0.5 may still be useful for trading even if its forecast error isn't lower — directional skill is what drives PnL, and the future backtester (`src/btc_ai/eval/backtest.py`, not yet implemented) will measure that separately.

## Files

```
experiments/01_baseline_naive/
├── README.md           # this file
├── config.yaml         # data slice + test_fraction
├── run.py              # entry point
└── results/
    ├── metrics.json        # produced by run.py
    ├── predictions.parquet # close + pred on the test slice
    └── plot.png            # close vs naive pred (test slice)
```
