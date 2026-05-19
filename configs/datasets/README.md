# Dataset configs

Centralized "historical CSV" specs. Each `*.yaml` here defines:
- A Binance Vision **source** (market / symbol / interval / period), and
- Three **splits** — `train`, `validation`, `test` — as explicit UTC `[start, end)` date ranges.

Experiment configs (`experiments/<N>_<model>/configs/*.yaml`) reference a dataset by name:

```yaml
dataset: btc_4h_2024     # resolves to configs/datasets/btc_4h_2024.yaml
model: { ... }            # rest of the experiment config: model / covariates / optuna
```

## Two valid shapes

**Shape A — single source, three time slices** (common case):

```yaml
source:
  market: spot
  symbol: BTCUSDT
  interval: 4h
  period: monthly
splits:
  train:      { start: 2024-01-01, end: 2024-09-01 }
  validation: { start: 2024-09-01, end: 2024-10-01 }
  test:       { start: 2024-10-01, end: 2024-12-01 }
```

The loader does one Binance Vision pull over `[min(splits.start), max(splits.end))` and slices.

**Shape B — per-split source** (e.g. BTC train → ETH test, out-of-distribution evaluation):

```yaml
splits:
  train:      { market: spot, symbol: BTCUSDT, interval: 4h, period: monthly, start: 2023-01-01, end: 2023-12-01 }
  validation: { market: spot, symbol: BTCUSDT, interval: 4h, period: monthly, start: 2024-01-01, end: 2024-06-01 }
  test:       { market: spot, symbol: ETHUSDT, interval: 4h, period: monthly, start: 2024-06-01, end: 2024-12-01 }
```

Per-split sources are required when there is no top-level `source:`. All three splits must share `interval` (metrics depend on a fixed bar size). Mixing top-level `source:` with per-split source fields is rejected.

## Rules

- `validation:` is **always required**. Experiments that don't use a validation slice (ARIMA, naive, MA, XGBoost, foundation models) concatenate `train + validation` for fitting.
- Splits must be `train.end <= validation.start` and `validation.end <= test.start`. Gaps between splits are allowed; overlaps are not.
- Dates are UTC. Use ISO `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS` (UTC implied).
