import argparse
import csv
import logging
from pathlib import Path

from run import train_and_evaluate

from btc_ai.config import load_yaml

EXPERIMENT_DIR = Path(__file__).parent

# Edit this list to change which fine-tuning recipes the sweep evaluates. All
# rows share the same data slice / split defined in config.yaml; the model
# constants (output_chunk_length, num_samples, quantiles, random_state,
# batch_size, early_stopping_patience) come from config.yaml too unless a row
# overrides them explicitly. Metrics shown here are the aggregate macro means
# across test symbols (single-symbol configs report the same value either way).
CHRONOS_SMALL = 'autogluon/chronos-2-small'
CHRONOS = 'amazon/chronos-2'
TIMESFM = 'google/timesfm-2.5-200m-pytorch'

# Glob patterns are fnmatch-style and match against model.named_parameters().
CHRONOS_HEAD_ONLY: dict[str, list[str]] = {'unfreeze': ['*output_patch_embedding*']}

GRID: list[dict[str, object]] = [
	{
		'backend': 'chronos', 'hub_model_name': CHRONOS_SMALL, 'input_chunk_length': 256,
		'learning_rate': 1e-4, 'n_epochs': 10, 'enable_finetuning': CHRONOS_HEAD_ONLY,
	},
	{
		'backend': 'chronos', 'hub_model_name': CHRONOS, 'input_chunk_length': 256,
		'learning_rate': 1e-4, 'n_epochs': 10, 'enable_finetuning': CHRONOS_HEAD_ONLY,
	},
	{
		'backend': 'chronos', 'hub_model_name': CHRONOS, 'input_chunk_length': 256,
		'learning_rate': 1e-4, 'n_epochs': 20, 'enable_finetuning': True,
	},
	{
		'backend': 'chronos', 'hub_model_name': CHRONOS, 'input_chunk_length': 256,
		'learning_rate': 3e-5, 'n_epochs': 20, 'enable_finetuning': True,
	},
	{
		'backend': 'timesfm', 'hub_model_name': TIMESFM, 'input_chunk_length': 256,
		'learning_rate': 1e-4, 'n_epochs': 10, 'enable_finetuning': True,
	},
]

logger = logging.getLogger(__name__)


def _finetuning_label(finetuning: object) -> str:
	# Compact string form of enable_finetuning suitable for a CSV cell or stdout column.
	if isinstance(finetuning, bool):
		return 'full' if finetuning else 'frozen'
	if isinstance(finetuning, dict):
		mode, patterns = next(iter(finetuning.items()))
		return f'{mode}={",".join(patterns)}'
	return str(finetuning)


def main() -> None:
	logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

	parser = argparse.ArgumentParser(
		description='Sweep fine-tuning recipes on the same data slice as run.py.',
	)
	parser.add_argument('--config', type=Path, required=True)
	args = parser.parse_args()

	results_dir = EXPERIMENT_DIR / 'results' / args.config.stem.removesuffix('.config')
	results_dir.mkdir(parents=True, exist_ok=True)

	cfg = load_yaml(args.config)

	header = [
		'backend', 'hub_model_name', 'input_chunk_length',
		'enable_finetuning', 'learning_rate', 'n_epochs', 'epochs_trained',
		'mae', 'rmse', 'mape',
		'directional_accuracy', 'cumulative_return', 'annualized_sharpe',
	]
	print(
		f'{"backend":>8} {"hub_model_name":>34} {"input_chunk_length":>20} '
		f'{"enable_finetuning":>32} {"learning_rate":>14} '
		f'{"n_epochs":>10} {"epochs_trained":>16} '
		f'{"mae":>9} {"rmse":>9} {"mape":>9} '
		f'{"directional_accuracy":>22} '
		f'{"cumulative_return":>20} {"annualized_sharpe":>20}',
	)
	print('-' * 240)

	rows: list[dict[str, object]] = []
	for params in GRID:
		finetuning_str = _finetuning_label(params['enable_finetuning'])
		try:
			# train_and_evaluate consumes a *copy* of params and pops finetune/training
			# keys out — sweep.py keeps the original dict intact for logging.
			out = train_and_evaluate(cfg, model_overrides=dict(params))
			row = {
				'backend': str(params['backend']),
				'hub_model_name': str(params['hub_model_name']),
				'input_chunk_length': int(params['input_chunk_length']),  # type: ignore[arg-type]
				'enable_finetuning': finetuning_str,
				'learning_rate': float(params['learning_rate']),  # type: ignore[arg-type]
				'n_epochs': int(params['n_epochs']),  # type: ignore[arg-type]
				'epochs_trained': int(out.epochs_trained),
				'mae': out.aggregate['mae'],
				'rmse': out.aggregate['rmse'],
				'mape': out.aggregate['mape'],
				'directional_accuracy': out.aggregate['directional_accuracy'],
				'cumulative_return': out.aggregate['cumulative_return'],
				'annualized_sharpe': out.aggregate['annualized_sharpe'],
			}
			print(
				f'{row["backend"]:>8} {row["hub_model_name"]:>34} '
				f'{row["input_chunk_length"]:>20d} '
				f'{finetuning_str:>32} {row["learning_rate"]:>14.0e} '
				f'{row["n_epochs"]:>10d} {row["epochs_trained"]:>16d} '
				f'{row["mae"]:>9.2f} {row["rmse"]:>9.2f} '
				f'{row["mape"] * 100:>8.3f}% '
				f'{row["directional_accuracy"]:>22.4f} '
				f'{row["cumulative_return"] * 100:>19.3f}% '
				f'{row["annualized_sharpe"]:>20.3f}',
			)
			rows.append(row)
		except Exception as exc:  # noqa: BLE001
			print(
				f'{str(params["backend"]):>8} {str(params["hub_model_name"]):>34} '
				f'{int(params["input_chunk_length"]):>20d} '  # type: ignore[arg-type]
				f'{finetuning_str:>32} FAILED: {exc}',
			)

	out_path = results_dir / 'sweep.csv'
	with open(out_path, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=header)
		writer.writeheader()
		writer.writerows(rows)
	print(f'\nwrote {out_path}')


if __name__ == '__main__':
	main()
