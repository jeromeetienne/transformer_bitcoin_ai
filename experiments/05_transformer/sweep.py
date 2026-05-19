import argparse
import csv
import logging
from pathlib import Path

from run import train_and_evaluate

from btc_ai.config import load_yaml

EXPERIMENT_DIR = Path(__file__).parent
# results_dir is derived inside main() from the config filename stem
# (e.g. configs/btc_4h_2024.config.yaml → results/btc_4h_2024/).

# Edit this list to change which (input_chunk_length, hidden_size, num_attention_heads,
# lstm_layers, dropout) combos the sweep evaluates. All combos use the same data slice /
# split / covariates defined in config.yaml; other model params (batch_size, n_epochs,
# learning_rate, RNG, hidden_continuous_size, full_attention, add_relative_index) come
# from config.yaml too.
GRID: list[dict[str, float | int]] = [
	{
		'input_chunk_length': 24, 'hidden_size': 16,
		'num_attention_heads': 2, 'lstm_layers': 1, 'dropout': 0.1,
	},
	{
		'input_chunk_length': 48, 'hidden_size': 32,
		'num_attention_heads': 4, 'lstm_layers': 1, 'dropout': 0.1,
	},
	{
		'input_chunk_length': 48, 'hidden_size': 32,
		'num_attention_heads': 4, 'lstm_layers': 2, 'dropout': 0.2,
	},
	{
		'input_chunk_length': 48, 'hidden_size': 64,
		'num_attention_heads': 8, 'lstm_layers': 1, 'dropout': 0.1,
	},
	{
		'input_chunk_length': 96, 'hidden_size': 32,
		'num_attention_heads': 4, 'lstm_layers': 1, 'dropout': 0.1,
	},
	{
		'input_chunk_length': 96, 'hidden_size': 64,
		'num_attention_heads': 8, 'lstm_layers': 2, 'dropout': 0.2,
	},
]

logger = logging.getLogger(__name__)


def main() -> None:
	logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

	parser = argparse.ArgumentParser(
		description='Sweep TFT hyperparameters on the same data slice as run.py.',
	)
	parser.add_argument('--config', type=Path, required=True)
	args = parser.parse_args()

	results_dir = EXPERIMENT_DIR / 'results' / args.config.stem.removesuffix('.config')
	results_dir.mkdir(parents=True, exist_ok=True)

	cfg = load_yaml(args.config)

	header = [
		'input_chunk_length', 'hidden_size', 'num_attention_heads', 'lstm_layers', 'dropout',
		'mae', 'rmse', 'mape', 'dir_acc', 'cum_ret', 'sharpe',
	]
	print(
		f'{"icl":>4} {"h":>4} {"heads":>5} {"L":>3} {"drop":>5} '
		f'{"mae":>9} {"rmse":>9} {"mape":>9} {"dir_acc":>9} '
		f'{"cum_ret":>10} {"sharpe":>8}',
	)
	print('-' * 94)

	rows: list[dict[str, object]] = []
	for params in GRID:
		try:
			metrics, _ = train_and_evaluate(cfg, model_overrides=params)
			row = {
				'input_chunk_length': int(params['input_chunk_length']),
				'hidden_size': int(params['hidden_size']),
				'num_attention_heads': int(params['num_attention_heads']),
				'lstm_layers': int(params['lstm_layers']),
				'dropout': float(params['dropout']),
				'mae': metrics['mae'],
				'rmse': metrics['rmse'],
				'mape': metrics['mape'],
				'dir_acc': metrics['directional_accuracy'],
				'cum_ret': metrics['cumulative_return'],
				'sharpe': metrics['annualized_sharpe'],
			}
			print(
				f'{row["input_chunk_length"]:>4d} {row["hidden_size"]:>4d} '
				f'{row["num_attention_heads"]:>5d} {row["lstm_layers"]:>3d} '
				f'{row["dropout"]:>5.2f} '
				f'{row["mae"]:>9.2f} {row["rmse"]:>9.2f} '
				f'{row["mape"] * 100:>8.3f}% {row["dir_acc"]:>9.4f} '
				f'{row["cum_ret"] * 100:>9.3f}% {row["sharpe"]:>8.3f}',
			)
			rows.append(row)
		except Exception as exc:  # noqa: BLE001
			print(
				f'{int(params["input_chunk_length"]):>4d} '
				f'{int(params["hidden_size"]):>4d} '
				f'{int(params["num_attention_heads"]):>5d} '
				f'{int(params["lstm_layers"]):>3d} '
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
