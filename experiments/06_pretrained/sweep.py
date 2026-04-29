import argparse
import csv
import logging
from pathlib import Path

from run import train_and_evaluate

from btc_ai.config import load_yaml

EXPERIMENT_DIR = Path(__file__).parent
DEFAULT_CONFIG = EXPERIMENT_DIR / 'config.yaml'
RESULTS_DIR = EXPERIMENT_DIR / 'results'

# Edit this list to change which (backend, hub_model_name, input_chunk_length)
# combos the sweep evaluates. All combos use the same data slice / split
# defined in config.yaml; other model params (output_chunk_length, num_samples,
# quantiles) come from config.yaml too.
CHRONOS_SMALL = 'autogluon/chronos-2-small'
CHRONOS = 'amazon/chronos-2'
TIMESFM = 'google/timesfm-2.5-200m-pytorch'

GRID: list[dict[str, str | int]] = [
	{'backend': 'chronos', 'hub_model_name': CHRONOS_SMALL, 'input_chunk_length': 64},
	{'backend': 'chronos', 'hub_model_name': CHRONOS,       'input_chunk_length': 64},
	{'backend': 'chronos', 'hub_model_name': CHRONOS,       'input_chunk_length': 256},
	{'backend': 'chronos', 'hub_model_name': CHRONOS,       'input_chunk_length': 1024},
	{'backend': 'timesfm', 'hub_model_name': TIMESFM,       'input_chunk_length': 64},
	{'backend': 'timesfm', 'hub_model_name': TIMESFM,       'input_chunk_length': 256},
	{'backend': 'timesfm', 'hub_model_name': TIMESFM,       'input_chunk_length': 1024},
]

logger = logging.getLogger(__name__)


def main() -> None:
	logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

	parser = argparse.ArgumentParser(
		description='Sweep zero-shot foundation models on the same data slice as run.py.',
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	args = parser.parse_args()

	cfg = load_yaml(args.config)

	header = [
		'backend', 'hub_model_name', 'input_chunk_length',
		'mae', 'rmse', 'mape', 'dir_acc', 'cum_ret', 'sharpe',
	]
	print(
		f'{"backend":>8} {"hub_model_name":>34} {"icl":>5} '
		f'{"mae":>9} {"rmse":>9} {"mape":>9} {"dir_acc":>9} '
		f'{"cum_ret":>10} {"sharpe":>8}',
	)
	print('-' * 116)

	rows: list[dict[str, object]] = []
	for params in GRID:
		try:
			metrics, _ = train_and_evaluate(cfg, model_overrides=params)
			row = {
				'backend': str(params['backend']),
				'hub_model_name': str(params['hub_model_name']),
				'input_chunk_length': int(params['input_chunk_length']),
				'mae': metrics['mae'],
				'rmse': metrics['rmse'],
				'mape': metrics['mape'],
				'dir_acc': metrics['directional_accuracy'],
				'cum_ret': metrics['cumulative_return'],
				'sharpe': metrics['annualized_sharpe'],
			}
			print(
				f'{row["backend"]:>8} {row["hub_model_name"]:>34} '
				f'{row["input_chunk_length"]:>5d} '
				f'{row["mae"]:>9.2f} {row["rmse"]:>9.2f} '
				f'{row["mape"] * 100:>8.3f}% {row["dir_acc"]:>9.4f} '
				f'{row["cum_ret"] * 100:>9.3f}% {row["sharpe"]:>8.3f}',
			)
			rows.append(row)
		except Exception as exc:  # noqa: BLE001
			print(
				f'{str(params["backend"]):>8} {str(params["hub_model_name"]):>34} '
				f'{int(params["input_chunk_length"]):>5d} FAILED: {exc}',
			)

	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	out = RESULTS_DIR / 'sweep.csv'
	with open(out, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=header)
		writer.writeheader()
		writer.writerows(rows)
	print(f'\nwrote {out}')


if __name__ == '__main__':
	main()
