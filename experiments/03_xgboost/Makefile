# All paths in this file are relative to experiments/03_xgboost/.
# Invoked from the root via `$(MAKE) -C experiments/03_xgboost <target>`.

CONFIG ?= configs/btc_4h_2024.config.yaml

.PHONY: help run sweep

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-20s %s\n", $$1, $$2}'

run:    ## run with $(CONFIG)
	uv run python run.py --config $(CONFIG)

sweep:  ## sweep XGBoost hyperparams on the data slice in $(CONFIG)
	uv run python sweep.py --config $(CONFIG)
