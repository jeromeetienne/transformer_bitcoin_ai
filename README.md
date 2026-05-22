# transformer_bitcoin_ai

A collection of Python experiments exploring whether Bitcoin trading signals can be predicted from price time series. The repo serves as a sandbox for trying different modeling approaches (transformers, classical sequence models, baselines) against historical BTC price data, comparing their forecasting accuracy and trading-strategy performance.

## How to install

1. Install [`uv`](https://docs.astral.sh/uv/) — the project uses it for env, deps, and runs:
   ```
   brew install uv          # macOS
   ```
   Other platforms: see the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/).

2. Sync dependencies (creates `.venv/` from `pyproject.toml` + `uv.lock`):
   ```
   uv sync
   ```

3. Verify by listing Make targets:
   ```
   make help
   ```

## Quickstart

The Makefile is the canonical command surface — every target wraps `uv run`.

```
make fetch                                         # pre-warm data cache
make 01_baseline                             # naive last-value baseline
make 02_arima                                      # ARIMA(p, d, q) baseline
make 03_xgboost                          # XGBoost on engineered features
make 04_lstm                                       # Darts BlockRNN-LSTM
make 05_transformer                                # Darts Temporal Fusion Transformer
make 06_pretrained                                 # zero-shot Chronos-2 / TimesFM 2.5
make help                                          # list all targets (incl. *_sweep variants)
```

Override the active experiment YAML for `make fetch`:
```
make fetch CONFIG=experiments/02_arima/config.yaml
```

## Repository Layout

Shared infrastructure lives at the root; each experiment is a self-contained folder so runs stay independent and comparable.

```
transformer_bitcoin_ai/
├── Makefile                    # canonical command surface (wraps uv run)
├── README.md
├── pyproject.toml              # deps + tooling (ruff, pytest), managed by uv
├── uv.lock                     # committed
├── .python-version             # 3.11, managed by uv
├── data/
│   ├── raw/                    # downloaded zips (gitignored)
│   ├── processed/              # reserved for future feature caches (gitignored)
│   └── README.md
├── src/btc_ai/                 # shared library (importable)
│   ├── config.py               # YAML + KlineRequest construction
│   ├── data/
│   │   ├── schema.py           # KlineRequest dataclass
│   │   ├── cache.py            # HTTP download with on-disk cache
│   │   └── binance_vision.py   # data.binance.vision loader
│   └── eval/
│       └── metrics.py          # MAE, RMSE, MAPE, directional accuracy
├── experiments/                # one folder per experiment
│   ├── 01_baseline/      # last-value forecast (the architecture probe)
│   │   ├── run.py
│   │   ├── config.yaml
│   │   └── results/            # metrics.json, predictions.parquet, plot.png
│   ├── 02_arima/               # ARIMA(p, d, q) + sweep over orders
│   ├── 03_xgboost/   # XGBoost on engineered features (+ sweep)
│   ├── 04_lstm/                # Darts BlockRNN-LSTM (+ sweep)
│   ├── 05_transformer/         # Darts Temporal Fusion Transformer (+ sweep)
│   └── 06_pretrained/          # zero-shot Chronos-2 / TimesFM 2.5 foundation models (+ sweep)
└── scripts/
    └── fetch_data.py           # CLI that pre-warms data cache from a config.yaml
```

## Reports & Articles

Each experiment has a detailed Markdown report in [docs_ml/reports/](docs_ml/reports/) and a companion article in [docs_ml/articles/](docs_ml/articles/). The reports are the raw source material drawn from `results/`; the articles are the polished write-ups for readers. See [docs_ml/README.md](docs_ml/README.md) for the series principles and [docs_ml/STYLE.md](docs_ml/STYLE.md) for writing conventions.

| # | Experiment | Report | Article |
|---|------------|--------|---------|
| 0 | Project presentation | — | [0-presentation-of-the-project](docs_ml/articles/0-presentation-of-the-project/0-presentation-of-the-project.article.md) |
| 1 | Baseline | [01_baseline.report.md](docs_ml/reports/01_baseline.report.md) | [1-baseline](docs_ml/articles/1-baseline/1-baseline.article.md) |
| 2 | ARIMA | [02_arima.report.md](docs_ml/reports/02_arima.report.md) | [2-arima](docs_ml/articles/2-arima/2-arima.article.md) |
| 3 | XGBoost | [03_xgboost.report.md](docs_ml/reports/03_xgboost.report.md) | [3-xgboost](docs_ml/articles/3-xgboost/3-xgboost.article.md) |
| 4 | LSTM | [04_lstm.report.md](docs_ml/reports/04_lstm.report.md) | [4-lstm](docs_ml/articles/4-lstm/4-lstm.article.md) |
| 5 | Transformer | [05_transformer.report.md](docs_ml/reports/05_transformer.report.md) | [5-transformer](docs_ml/articles/5-transformer/5-transformer.article.md) |
| 6 | Pretrained foundation models | [06_pretrained.report.md](docs_ml/reports/06_pretrained.report.md) | [6-pretrained](docs_ml/articles/6-pretrained/6-pretrained.article.md) |
| 7 | Fine-tuned foundation models | [07_finetuned.report.md](docs_ml/reports/07_finetuned.report.md) | [7-finetuned](docs_ml/articles/7-finetuned/7-finetuned.article.md) |
| — | Cross-experiment synthesis | [XX_global.report.md](docs_ml/reports/XX_global.report.md) | — |

## Conventions

- **Tooling is `uv`.** `uv init`, `uv add`, `uv run`. `uv.lock` is committed; `.venv/` is not.
- **All commands go through the Makefile.** Don't invoke `python` directly.
- **Each experiment is self-contained**: `run.py` + `config.yaml` + `results/`. Never edit a past experiment — copy it to the next number (`cp -r experiments/01_baseline experiments/02_arima`).
- **The YAML is the single source of truth** for both data selection and model params. `run.py` and `scripts/fetch_data.py` both consume the same `--config`.
- **Shared code lives in `src/btc_ai/`** so every experiment is evaluated identically.
- **Numbered prefixes** (`01_`, `02_`) preserve chronology and make leaderboards readable.
- **`period` is required and explicit** (`monthly` or `daily`) — no auto-pick. Use `daily` for ranges that include a not-yet-complete month.
- **Data is gitignored**; `make fetch` makes it reproducible.
