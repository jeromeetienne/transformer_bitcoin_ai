# Canonical command surface. Every target wraps `uv run`; do not invoke python directly.
#
# Override the active experiment YAML on the command line:
#   make fetch CONFIG=experiments/02_arima/config.yaml

CONFIG ?= experiments/01_baseline_naive/config.yaml

.PHONY: help install fetch lint test clean \
	01_baseline_naive 01b_moving_average 02_arima 02_arima_sweep \
	03_gradient_boosting 03_gradient_boosting_sweep 04_lstm 04_lstm_sweep \
	05_transformer 05_transformer_sweep

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-28s %s\n", $$1, $$2}'

install:        ## install deps via uv (creates .venv on first run)
	uv sync

fetch:          ## pre-warm the data cache for $(CONFIG)
	uv run python scripts/fetch_data.py --config $(CONFIG)

# ---

lint:           ## ruff check
	uv run ruff check .

test:           ## pytest
	uv run pytest

clean:          ## remove cached data and experiment results
	rm -rf data/raw data/processed
	find experiments -path '*/results/*' ! -name '.gitkeep' -delete

# ---

01_baseline_naive:  ## run experiments/01_baseline_naive
	uv run python experiments/01_baseline_naive/run.py

01b_moving_average: ## run experiments/01b_moving_average
	uv run python experiments/01b_moving_average/run.py

02_arima:          ## run experiments/02_arima
	uv run python experiments/02_arima/run.py

02_arima_sweep:    ## sweep ARIMA (p, d, q) orders on the same data slice
	uv run python experiments/02_arima/sweep.py

03_gradient_boosting:       ## run experiments/03_gradient_boosting (XGBoost)
	uv run python experiments/03_gradient_boosting/run.py

03_gradient_boosting_sweep: ## sweep XGBoost hyperparams on the same data slice
	uv run python experiments/03_gradient_boosting/sweep.py

04_lstm:           ## run experiments/04_lstm (Darts BlockRNN-LSTM)
	uv run python experiments/04_lstm/run.py

04_lstm_sweep:     ## sweep LSTM hyperparams on the same data slice
	uv run python experiments/04_lstm/sweep.py

05_transformer:        ## run experiments/05_transformer (Darts TFT)
	uv run python experiments/05_transformer/run.py

05_transformer_sweep:  ## sweep TFT hyperparams on the same data slice
	uv run python experiments/05_transformer/sweep.py
