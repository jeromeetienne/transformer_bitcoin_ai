# All paths in this file are relative to experiments/06_pretrained_direct/.
# Invoked from the root via `$(MAKE) -C experiments/06_pretrained_direct <target>`.

CONFIG ?= configs/btc_4h_2024.chronos-small.config.yaml

# Silence Lightning/Torch noise (PL pytree deprecation, MPS pin_memory notice).
# Set in env so it survives Lightning's in-process filter resets and reaches DataLoader workers.
export PYTHONWARNINGS := ignore::UserWarning,ignore::DeprecationWarning

.PHONY: help run sweep timesfm chronos-large

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-20s %s\n", $$1, $$2}'

run:           ## run with $(CONFIG)
	uv run python run.py --config $(CONFIG)

sweep:         ## sweep foundation-model backends + context lengths on the data slice in $(CONFIG)
	uv run python sweep.py --config $(CONFIG)

run_timesfm:       ## run with configs/btc_4h_2024.timesfm.config.yaml
	$(MAKE) run CONFIG=configs/btc_4h_2024.timesfm.config.yaml

run_chronos_large: ## run with configs/btc_4h_2024.chronos-large.config.yaml
	$(MAKE) run CONFIG=configs/btc_4h_2024.chronos-large.config.yaml

run_chronos_small: ## run with configs/btc_4h_2024.chronos-small.config.yaml
	$(MAKE) run CONFIG=configs/btc_4h_2024.chronos-small.config.yaml
