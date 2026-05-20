# Dataset configs

Centralized "historical CSV" specs. Each `*.yaml` here defines three top-level arrays — `training`, `validation`, `test` — and each array element is a CSV selector that pins one Binance Vision pull (market / symbol / interval / period) over a UTC `[start, end)` date range.

Experiment configs (`experiments/<N>_<model>/configs/*.config.yaml`) reference a dataset by name:

```yaml
dataset: btc_4h_2024     # resolves to configs/datasets/btc_4h_2024.dataset.yaml
model: { ... }            # rest of the experiment config: model / covariates / optuna
```

## Shape

```yaml
training:
  - market: spot
    symbol: BTCUSDT
    interval: 4h
    period: monthly
    start: 2023-01-01
    end:   2024-08-01
validation:
  - market: spot
    symbol: BTCUSDT
    interval: 4h
    period: monthly
    start: 2024-08-01
    end:   2024-10-01
test:
  - market: spot
    symbol: BTCUSDT
    interval: 4h
    period: monthly
    start: 2024-10-01
    end:   2024-12-01
```

Every selector must specify all six fields (`market`, `symbol`, `interval`, `period`, `start`, `end`); the loader rejects partial selectors. Each of the three arrays must be non-empty.

## Multi-symbol training

Appending a second selector to `training` (and optionally to `validation` / `test`) widens the data the model sees. Two coexisting experiment families read multi-symbol arrays differently:

- **Sequence models** (`04_lstm`, `05_transformer`, `06_pretrained_direct`, `07_pretrained_finetuned`) consume training selectors natively as `list[TimeSeries]` and call `model.fit(series=[...], val_series=[...])`. At evaluation, walk-forward runs once per test selector and the resulting per-symbol metrics are aggregated by an unweighted macro mean.
- **Per-series models** (`01_baseline`, `02_arima`, `03_gradient_boosting`) fit one model per `(market, symbol)` present in `test`, matched against same-symbol training selectors (sorted by start). A training selector whose `(market, symbol)` does not appear in `test` is ignored (with a warning); a test `(market, symbol)` with no training match raises an error.

`metrics.json` always carries an `aggregate` block (macro means) and a `per_symbol` array (one entry per test selector). With a single test selector the two are equivalent.

## Rules

- `validation:` is **always required** by the loader, but **not every experiment uses it**:
  - `04_lstm`, `05_transformer`, `07_pretrained_finetuned` — fit on `train`, early-stop on `validation`, evaluate on `test`.
  - `01_baseline`, `02_arima`, `03_gradient_boosting`, `06_pretrained_direct` — concatenate `train + validation` (within a symbol) for fitting; the validation slice acts as extra training data, not held-out signal.
- Every selector across every array must share `interval`. The loader rejects mixed intervals because metrics (`periods_per_year`) assume one bar size per run.
- For each `(market, symbol)` that appears in `test`, the per-symbol envelope must satisfy `train.end <= validation.start <= test.start` (gaps allowed; overlaps not). Selectors for the same symbol may be split across multiple entries — the envelope check uses the per-symbol min-start / max-end.
- Dates are UTC. Use ISO `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS` (UTC implied).
- Lag and rolling features at the first bar of `validation` / `test` legitimately read history from the prior slice — that is walk-forward evaluation, not leakage. Bars themselves never cross; only the rolling-window feature inputs do.
