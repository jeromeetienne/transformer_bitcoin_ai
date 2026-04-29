# data/

Historical klines fetched from [data.binance.vision](https://data.binance.vision/)
on demand by `scripts/fetch_data.py`. Nothing in `raw/` or `processed/` is
committed — re-fetch with `make fetch` (or `uv run python scripts/fetch_data.py
--config <experiment>/config.yaml`).

## Layout

```
data/
├── raw/         # downloaded zips, mirroring data.binance.vision URL structure
│   └── binance/{market}/{period}/klines/{SYMBOL}/{INTERVAL}/*.zip
└── processed/   # reserved for future parquet/feature caches (unused for now)
```
