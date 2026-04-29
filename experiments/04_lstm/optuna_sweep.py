import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from run import train_and_evaluate

from btc_ai.config import load_yaml

EXPERIMENT_DIR = Path(__file__).parent
DEFAULT_CONFIG = EXPERIMENT_DIR / 'config.yaml'
RESULTS_DIR = EXPERIMENT_DIR / 'results'
STUDY_NAME = '04_lstm'

VALID_OBJECTIVES = {
	'mae', 'rmse', 'mape',
	'directional_accuracy', 'cumulative_return', 'annualized_sharpe',
}
VALID_DIRECTIONS = {'minimize', 'maximize'}

logger = logging.getLogger(__name__)


def require(d: dict[str, Any], key: str, ctx: str) -> Any:
	# Strict lookup: missing keys raise instead of silently defaulting. Aligns with the
	# project's "no auto-pick defaults" rule for ambiguous knobs.
	if key not in d:
		raise KeyError(f'missing required key {ctx!r}.{key!r} in config')
	return d[key]


def suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
	# Dispatches on spec['type'] to the matching trial.suggest_* call. Keeps the YAML
	# search-space declarative so adding a knob is a config edit, not a code edit.
	t = require(spec, 'type', f'optuna.search_space.{name}')
	if t == 'categorical':
		return trial.suggest_categorical(name, require(spec, 'choices', f'optuna.search_space.{name}'))
	if t == 'int':
		return trial.suggest_int(
			name,
			int(require(spec, 'low', f'optuna.search_space.{name}')),
			int(require(spec, 'high', f'optuna.search_space.{name}')),
			step=int(spec.get('step', 1)),
			log=bool(spec.get('log', False)),
		)
	if t == 'float':
		return trial.suggest_float(
			name,
			float(require(spec, 'low', f'optuna.search_space.{name}')),
			float(require(spec, 'high', f'optuna.search_space.{name}')),
			log=bool(spec.get('log', False)),
		)
	raise ValueError(f'unknown search-space type {t!r} for {name!r}; expected categorical/int/float')


def build_objective(
	cfg: dict[str, Any],
	opt_cfg: dict[str, Any],
) -> Callable[[optuna.Trial], float]:
	objective_key = require(opt_cfg, 'objective', 'optuna')
	if objective_key not in VALID_OBJECTIVES:
		raise ValueError(
			f'optuna.objective={objective_key!r} not in {sorted(VALID_OBJECTIVES)}',
		)
	search_space = require(opt_cfg, 'search_space', 'optuna')

	def objective(trial: optuna.Trial) -> float:
		overrides = {name: suggest(trial, name, spec) for name, spec in search_space.items()}
		pruning_cb = PyTorchLightningPruningCallback(trial, monitor='val_loss')
		try:
			metrics, _ = train_and_evaluate(
				cfg,
				model_overrides=overrides,
				extra_pl_callbacks=[pruning_cb],
			)
		except optuna.TrialPruned:
			raise
		except Exception as exc:  # noqa: BLE001
			# Treat training failures as pruned so the study keeps moving instead of crashing.
			# The trial's params are still recorded in the storage with state=PRUNED.
			logger.warning('trial %d failed (%s); marking pruned', trial.number, exc)
			raise optuna.TrialPruned() from exc
		value = float(metrics[objective_key])
		print(
			f'trial {trial.number:>3d} {objective_key}={value:.6f} '
			f'params={overrides}',
		)
		return value

	return objective


def main() -> None:
	logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

	parser = argparse.ArgumentParser(
		description='Optuna hyperparameter search for BlockRNN-LSTM on the same data slice as run.py.',
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	args = parser.parse_args()

	cfg = load_yaml(args.config)
	opt_cfg = require(cfg, 'optuna', 'config')

	n_trials = int(require(opt_cfg, 'n_trials', 'optuna'))
	timeout_seconds = opt_cfg.get('timeout_seconds')
	if timeout_seconds is not None:
		timeout_seconds = float(timeout_seconds)
	direction = require(opt_cfg, 'direction', 'optuna')
	if direction not in VALID_DIRECTIONS:
		raise ValueError(f'optuna.direction={direction!r} must be one of {sorted(VALID_DIRECTIONS)}')
	seed = int(require(opt_cfg, 'seed', 'optuna'))

	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	storage_path = RESULTS_DIR / 'optuna_study.db'
	storage_url = f'sqlite:///{storage_path}'

	print(f'study={STUDY_NAME!r} storage={storage_url}')
	print(f'n_trials={n_trials} timeout={timeout_seconds} direction={direction} seed={seed}')

	study = optuna.create_study(
		study_name=STUDY_NAME,
		storage=storage_url,
		load_if_exists=True,
		direction=direction,
		sampler=optuna.samplers.TPESampler(seed=seed),
		pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
	)

	study.optimize(
		build_objective(cfg, opt_cfg),
		n_trials=n_trials,
		timeout=timeout_seconds,
	)

	trials_csv = RESULTS_DIR / 'optuna_trials.csv'
	study.trials_dataframe().to_csv(trials_csv, index=False)

	pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
	complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
	best_json: dict[str, Any] = {
		'study_name': STUDY_NAME,
		'storage_url': storage_url,
		'objective': opt_cfg['objective'],
		'direction': direction,
		'n_trials_total': len(study.trials),
		'n_trials_complete': complete,
		'n_trials_pruned': pruned,
		'best_value': float(study.best_value) if complete > 0 else None,
		'best_params': study.best_params if complete > 0 else None,
		'best_trial_number': study.best_trial.number if complete > 0 else None,
	}
	(RESULTS_DIR / 'optuna_best.json').write_text(json.dumps(best_json, indent=2) + '\n')

	print(json.dumps(best_json, indent=2))
	print(f'\nwrote {trials_csv}')
	print(f'wrote {RESULTS_DIR / "optuna_best.json"}')
	print(f'\nlaunch dashboard: make 04_lstm_optuna_dashboard')


if __name__ == '__main__':
	main()
