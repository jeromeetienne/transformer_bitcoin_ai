# All paths in this file are relative to experiments/07_pretrained_finetune/.
# Invoked from the root via `$(MAKE) -C experiments/07_pretrained_finetune <target>`.

CONFIG ?= configs/btc_4h_2024.config.yaml

# Silence Lightning and Torch noise (PyTorch Lightning pytree deprecation, Apple Metal Performance Shaders pin_memory notice).
# Set in env so it survives Lightning's in-process filter resets and reaches DataLoader workers.
export PYTHONWARNINGS := ignore::UserWarning,ignore::DeprecationWarning

.PHONY: help run sweep clean_checkpoints

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-20s %s\n", $$1, $$2}'

run:    ## run with $(CONFIG)
	uv run python run.py --config $(CONFIG)

sweep:  ## sweep fine-tuning recipes (fine-tuning mode, learning_rate, n_epochs) on the data slice in $(CONFIG)
	uv run python sweep.py --config $(CONFIG)

clean_checkpoints:  ## remove results/darts_checkpoints/ written by save_checkpoints=True
	rm -rf results/darts_checkpoints/
