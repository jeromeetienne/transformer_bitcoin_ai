import argparse
import csv
import logging
from pathlib import Path

from run import train_and_evaluate

from btc_ai.config import load_yaml

EXPERIMENT_DIR = Path(__file__).parent
# results_dir is derived inside main() from the config filename stem
# (e.g. configs/btc_4h_2024.yaml → results/btc_4h_2024/).

# Edit this list to change which (input_chunk_length, hidden_dim, n_rnn_layers, dropout)
# combos the sweep evaluates. All combos use the same data slice / split / covariates
# defined in config.yaml; other model params (batch_size, n_epochs, learning_rate, RNG)
# come from config.yaml too.
GRID: list[dict[str, float | int]] = [
	{'input_chunk_length': 24, 'hidden_dim': 16, 'n_rnn_layers': 1, 'dropout': 0.0},
	{'input_chunk_length': 24, 'hidden_dim': 32, 'n_rnn_layers': 2, 'dropout': 0.1},
	{'input_chunk_length': 48, 'hidden_dim': 32, 'n_rnn_layers': 2, 'dropout': 0.1},
	{'input_chunk_length': 48, 'hidden_dim': 64, 'n_rnn_layers': 2, 'dropout': 0.2},
	{'input_chunk_length': 96, 'hidden_dim': 32, 'n_rnn_layers': 2, 'dropout': 0.1},
	{'input_chunk_length': 96, 'hidden_dim': 64, 'n_rnn_layers': 3, 'dropout': 0.2},
]

logger = logging.getLogger(__name__)


def main() -> None:
	logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

	parser = argparse.ArgumentParser(
		description='Sweep BlockRNN-LSTM hyperparameters on the same data slice as run.py.',
	)
	parser.add_argument('--config', type=Path, required=True)
	args = parser.parse_args()

	results_dir = EXPERIMENT_DIR / 'results' / args.config.stem
	results_dir.mkdir(parents=True, exist_ok=True)

	cfg = load_yaml(args.config)

	header = [
		'input_chunk_length', 'hidden_dim', 'n_rnn_layers', 'dropout',
		'mae', 'rmse', 'mape', 'dir_acc', 'cum_ret', 'sharpe',
	]
	print(
		f'{"icl":>4} {"h":>4} {"L":>3} {"drop":>5} '
		f'{"mae":>9} {"rmse":>9} {"mape":>9} {"dir_acc":>9} '
		f'{"cum_ret":>10} {"sharpe":>8}',
	)
	print('-' * 88)

	rows: list[dict[str, object]] = []
	for params in GRID:
		try:
			metrics, _ = train_and_evaluate(cfg, model_overrides=params)
			row = {
				'input_chunk_length': int(params['input_chunk_length']),
				'hidden_dim': int(params['hidden_dim']),
				'n_rnn_layers': int(params['n_rnn_layers']),
				'dropout': float(params['dropout']),
				'mae': metrics['mae'],
				'rmse': metrics['rmse'],
				'mape': metrics['mape'],
				'dir_acc': metrics['directional_accuracy'],
				'cum_ret': metrics['cumulative_return'],
				'sharpe': metrics['annualized_sharpe'],
			}
			print(
				f'{row["input_chunk_length"]:>4d} {row["hidden_dim"]:>4d} '
				f'{row["n_rnn_layers"]:>3d} {row["dropout"]:>5.2f} '
				f'{row["mae"]:>9.2f} {row["rmse"]:>9.2f} '
				f'{row["mape"] * 100:>8.3f}% {row["dir_acc"]:>9.4f} '
				f'{row["cum_ret"] * 100:>9.3f}% {row["sharpe"]:>8.3f}',
			)
			rows.append(row)
		except Exception as exc:  # noqa: BLE001
			print(
				f'{int(params["input_chunk_length"]):>4d} '
				f'{int(params["hidden_dim"]):>4d} '
				f'{int(params["n_rnn_layers"]):>3d} '
				f'{float(params["dropout"]):>5.2f} FAILED: {exc}',
			)

	out = results_dir / 'sweep.csv'
	with open(out, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=header)
		writer.writeheader()
		writer.writerows(rows)
	print(f'\nwrote {out}')


if __name__ == '__main__':
	main()
