# Canonical command surface. Every target wraps `uv run`; do not invoke python directly.
#
# Override the active experiment YAML on the command line:
#   make fetch CONFIG=experiments/02_arima/config.yaml

CONFIG ?= experiments/01_baseline_naive/config.yaml

.PHONY: help install fetch baseline lint test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install:        ## install deps via uv (creates .venv on first run)
	uv sync

fetch:          ## pre-warm the data cache for $(CONFIG)
	uv run python scripts/fetch_data.py --config $(CONFIG)

baseline:       ## run experiments/01_baseline_naive
	uv run python experiments/01_baseline_naive/run.py

lint:           ## ruff check
	uv run ruff check .

test:           ## pytest
	uv run pytest

clean:          ## remove cached data and experiment results
	rm -rf data/raw data/processed
	find experiments -path '*/results/*' ! -name '.gitkeep' -delete
