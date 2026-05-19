# Canonical command surface. The root Makefile is a thin orchestrator that
# delegates each experiment-level target to that experiment's own Makefile via
# `$(MAKE) -C experiments/<id> <target>`. Per-experiment defaults (CONFIG) live
# in the sub-Makefiles. The root never assumes a default CONFIG for per-experiment
# work, but CLI overrides (`make 02_arima CONFIG=foo`) recurse automatically.
#
# Override the active config for `make fetch` on the command line:
#   make fetch CONFIG=experiments/02_arima/configs/btc_4h_2024.config.yaml

CONFIG ?= experiments/01_baseline_naive/configs/btc_4h_2024.config.yaml

.PHONY: help fetch lint test clean \
	01_baseline_naive 01b_moving_average 02_arima 02_arima_sweep \
	03_gradient_boosting 03_gradient_boosting_sweep \
	04_lstm 04_lstm_sweep 04_lstm_optuna 04_lstm_optuna_dashboard \
	05_transformer 05_transformer_sweep 05_transformer_optuna 05_transformer_optuna_dashboard \
	06_pretrained 06_pretrained_sweep \
	07_pretrained_finetune 07_pretrained_finetune_sweep 07_pretrained_finetune_clean_checkpoints

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-28s %s\n", $$1, $$2}'

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
	$(MAKE) -C experiments/01_baseline_naive run

01b_moving_average: ## run experiments/01b_moving_average
	$(MAKE) -C experiments/01b_moving_average run

02_arima:          ## run experiments/02_arima
	$(MAKE) -C experiments/02_arima run

02_arima_sweep:    ## sweep ARIMA (p, d, q) orders on the same data slice
	$(MAKE) -C experiments/02_arima sweep

03_gradient_boosting:       ## run experiments/03_gradient_boosting (XGBoost)
	$(MAKE) -C experiments/03_gradient_boosting run

03_gradient_boosting_sweep: ## sweep XGBoost hyperparams on the same data slice
	$(MAKE) -C experiments/03_gradient_boosting sweep

04_lstm:           ## run experiments/04_lstm (Darts BlockRNN-LSTM)
	$(MAKE) -C experiments/04_lstm run

04_lstm_sweep:     ## sweep LSTM hyperparams on the same data slice
	$(MAKE) -C experiments/04_lstm sweep

04_lstm_optuna:    ## optuna-driven hyperparameter search for 04_lstm
	$(MAKE) -C experiments/04_lstm optuna

04_lstm_optuna_dashboard: ## launch optuna-dashboard against the 04_lstm study (Ctrl-C to stop)
	$(MAKE) -C experiments/04_lstm optuna_dashboard

05_transformer:        ## run experiments/05_transformer (Darts TFT)
	$(MAKE) -C experiments/05_transformer run

05_transformer_sweep:  ## sweep TFT hyperparams on the same data slice
	$(MAKE) -C experiments/05_transformer sweep

05_transformer_optuna: ## optuna-driven hyperparameter search for 05_transformer
	$(MAKE) -C experiments/05_transformer optuna

05_transformer_optuna_dashboard: ## launch optuna-dashboard against the 05_transformer study (Ctrl-C to stop)
	$(MAKE) -C experiments/05_transformer optuna_dashboard

06_pretrained:         ## run experiments/06_pretrained (zero-shot Chronos-2 / TimesFM 2.5)
	$(MAKE) -C experiments/06_pretrained run

06_pretrained_sweep:   ## sweep foundation-model backends + context lengths on the same data slice
	$(MAKE) -C experiments/06_pretrained sweep

07_pretrained_finetune:        ## run experiments/07_pretrained_finetune (fine-tuned Chronos-2 / TimesFM 2.5)
	$(MAKE) -C experiments/07_pretrained_finetune run

07_pretrained_finetune_sweep:  ## sweep fine-tuning recipes (fine-tuning mode, learning_rate, n_epochs) on the same data slice
	$(MAKE) -C experiments/07_pretrained_finetune sweep

07_pretrained_finetune_clean_checkpoints:  ## remove experiments/07_pretrained_finetune/results/darts_checkpoints/
	$(MAKE) -C experiments/07_pretrained_finetune clean_checkpoints
