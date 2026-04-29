# transformer_bitcoin_ai

A collection of Python experiments exploring whether Bitcoin trading signals can be predicted from price time series. The repo serves as a sandbox for trying different modeling approaches (transformers, classical sequence models, baselines) against historical BTC price data, comparing their forecasting accuracy and trading-strategy performance.

## Repository Layout

Shared infrastructure lives at the root; each experiment is a self-contained folder so runs stay independent and comparable.

```
transformer_bitcoin_ai/
├── README.md
├── pyproject.toml              # deps + tooling (ruff, pytest)
├── .python-version
├── data/
│   ├── raw/                    # downloaded OHLCV (gitignored)
│   ├── processed/              # cleaned/resampled parquet (gitignored)
│   └── README.md               # how to fetch data
├── src/btc_ai/                 # shared library (importable)
│   ├── __init__.py
│   ├── data/
│   │   ├── fetch.py            # exchange/API downloaders
│   │   ├── loaders.py          # train/val/test splits, windowing
│   │   └── features.py         # returns, log-returns, TA features
│   ├── models/                 # reusable model building blocks
│   ├── eval/
│   │   ├── metrics.py          # MAE, directional accuracy, Sharpe
│   │   └── backtest.py         # turn predictions into PnL
│   └── viz.py
├── experiments/                # one folder per experiment
│   ├── 01_baseline_naive/      # last-value, moving average
│   │   ├── run.py
│   │   ├── config.yaml
│   │   └── results/            # metrics, plots, saved model
│   ├── 02_arima/
│   ├── 03_lstm/
│   ├── 04_transformer_vanilla/
│   ├── 05_transformer_multivariate/   # + volume, on-chain, sentiment
│   └── 06_informer_or_patchtst/
├── notebooks/                  # exploration, not the source of truth
│   └── eda_btc_returns.ipynb
├── scripts/
│   ├── fetch_data.py
│   └── compare_experiments.py  # leaderboard across experiments/
└── tests/
```

## Conventions

- **Each experiment is self-contained**: `run.py` + `config.yaml` + `results/`. Never edit a past experiment — copy it to the next number. This keeps results reproducible.
- **Shared code lives in `src/btc_ai/`**, not duplicated per experiment. Data loaders, metrics, and the backtester especially — every model is evaluated identically.
- **Numbered prefixes** (`01_`, `02_`) preserve chronology and make leaderboards readable.
- **Data is gitignored**; `scripts/fetch_data.py` makes it reproducible.
