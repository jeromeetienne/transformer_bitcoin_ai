# All paths in this file are relative to experiments/04_lstm/.
# Invoked from the root via `$(MAKE) -C experiments/04_lstm <target>`.

CONFIG ?= configs/btc_4h_2024.config.yaml
TRIAL  := $(notdir $(basename $(CONFIG)))

# Silence Lightning/Torch noise (PL pytree deprecation, MPS pin_memory notice).
# Set in env so it survives Lightning's in-process filter resets and reaches DataLoader workers.
export PYTHONWARNINGS := ignore::UserWarning,ignore::DeprecationWarning

.PHONY: help run sweep optuna optuna_dashboard

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-20s %s\n", $$1, $$2}'

run:    ## run with $(CONFIG)
	uv run python run.py --config $(CONFIG)

sweep:  ## sweep LSTM hyperparams on the data slice in $(CONFIG)
	uv run python sweep.py --config $(CONFIG)

optuna: ## adaptive TPE search with $(CONFIG)
	uv run python optuna_sweep.py --config $(CONFIG)

optuna_dashboard: ## launch optuna-dashboard against $(TRIAL)'s study (Ctrl-C to stop)
	uv run optuna-dashboard sqlite:///results/$(TRIAL)/optuna_study.db
