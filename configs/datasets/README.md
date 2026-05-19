# Dataset configs

Centralized "historical CSV" specs. Each `*.yaml` here defines:
- A Binance Vision **source** (market / symbol / interval / period), and
- Three **splits** — `train`, `validation`, `test` — as explicit UTC `[start, end)` date ranges.

Experiment configs (`experiments/<N>_<model>/configs/*.config.yaml`) reference a dataset by name:

```yaml
dataset: btc_4h_2024     # resolves to configs/datasets/btc_4h_2024.dataset.yaml
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
  train:      { start: 2024-01-01, end: 2024-08-01 }
  validation: { start: 2024-08-01, end: 2024-10-01 }
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

- `validation:` is **always required** by the loader, but **not every experiment uses it**:
  - `04_lstm`, `05_transformer` — fit on `train`, early-stop on `validation`, evaluate on `test`.
  - `01_baseline_naive`, `01b_moving_average`, `02_arima`, `03_gradient_boosting`, `06_pretrained` — concatenate `train + validation` for fitting; the validation slice acts as extra training data, not held-out signal.
  - Pick `validation` boundaries with the neural models in mind (enough bars for a stable early-stopping signal); the other experiments are insensitive to where `train` ends and `validation` begins.
- Splits must be `train.end <= validation.start` and `validation.end <= test.start`. Gaps between splits are allowed; overlaps are not.
- Dates are UTC. Use ISO `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS` (UTC implied).
- Lag and rolling features at the first bar of `validation` / `test` legitimately read history from the prior slice — that is walk-forward evaluation, not leakage. Bars themselves never cross; only the rolling-window feature inputs do.
