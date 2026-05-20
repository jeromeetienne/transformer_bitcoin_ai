# All paths in this file are relative to experiments/07_pretrained_finetune/.
# Invoked from the root via `$(MAKE) -C experiments/07_pretrained_finetune <target>`.

CONFIG ?= configs/btc_4h_2024.config.yaml

# Silence Lightning and Torch noise (PyTorch Lightning pytree deprecation, Apple Metal Performance Shaders pin_memory notice).
# Set in env so it survives Lightning's in-process filter resets and reaches DataLoader workers.
export PYTHONWARNINGS := ignore::UserWarning,ignore::DeprecationWarning

.PHONY: help run sweep run_chronos_2_large run_chronos_2_small run_timesfm clean_checkpoints

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-28s %s\n", $$1, $$2}'

run:    ## run with $(CONFIG)
	uv run python run.py --config $(CONFIG)

sweep:  ## sweep fine-tuning recipes (fine-tuning mode, learning_rate, n_epochs) on the data slice in $(CONFIG)
	uv run python sweep.py --config $(CONFIG)

run_chronos_2_large:  ## run encoder-only fine-tuning on btc_4h_2020_2024 with chronos-2-large backend
	uv run python run.py --config configs/btc_4h_2020_2024_encoder_only.chronos-2-large.config.yaml

run_chronos_2_small:  ## run encoder-only fine-tuning on btc_4h_2020_2024 with chronos-2-small backend
	uv run python run.py --config configs/btc_4h_2020_2024_encoder_only.chronos-2-small.config.yaml

run_timesfm:  ## run encoder-only fine-tuning on btc_4h_2020_2024 with timesfm backend
	uv run python run.py --config configs/btc_4h_2020_2024_encoder_only.timesfm.config.yaml

clean_checkpoints:  ## remove results/darts_checkpoints/ written by save_checkpoints=True
	rm -rf results/darts_checkpoints/
