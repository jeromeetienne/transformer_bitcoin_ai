import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import BlockRNNModel
from pytorch_lightning.callbacks import EarlyStopping

from btc_ai.config import load_yaml
from btc_ai.data import load_dataset_from_experiment_cfg
from btc_ai.eval.metrics import (
	annualized_sharpe,
	cumulative_return,
	directional_accuracy,
	mae,
	mape,
	periods_per_year,
	rmse,
	strategy_returns,
)

EXPERIMENT_DIR = Path(__file__).parent
# results_dir is derived inside main() from the config filename stem
# (e.g. configs/btc_4h_2024.yaml → results/btc_4h_2024/).
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / 'data' / 'raw'

logger = logging.getLogger(__name__)


def build_target_and_covariates(
	df: pd.DataFrame,
	cov_cfg: dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame | None, pd.Series]:
	# target r_T = log(close_T) - log(close_{T-1}); ref_T = close_{T-1} for
	# downstream price reconstruction. Covariate columns use bar-T OHLCV here, but
	# darts only feeds past_covariates strictly before each forecast step, so no
	# current-bar leakage at evaluation time.
	close = df['close'].astype('float64')
	target = np.log(close).diff().rename('r_target')
	ref = close.shift(1).rename('ref_close')

	cov_cols: dict[str, pd.Series] = {}
	if cov_cfg.get('use_volume') is True:
		cov_cols['log_volume'] = np.log1p(df['volume'].astype('float64'))
	if cov_cfg.get('use_ohlc') is True:
		cov_cols['hl_range'] = (df['high'] - df['low']).astype('float64')
		cov_cols['oc_body'] = (df['close'] - df['open']).astype('float64')

	if cov_cols:
		cov_df = pd.DataFrame(cov_cols, index=df.index)
		joined = pd.concat([target, ref, cov_df], axis=1).dropna()
		return joined['r_target'], joined[list(cov_cols.keys())], joined['ref_close']

	joined = pd.concat([target, ref], axis=1).dropna()
	return joined['r_target'], None, joined['ref_close']


def train_and_evaluate(
	cfg: dict[str, Any],
	model_overrides: dict[str, Any] | None = None,
	extra_pl_callbacks: list[Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
	# End-to-end pipeline shared by run.py and sweep.py: load data, build target +
	# past covariates, fit Scaler on train only, train BlockRNN-LSTM with val-based
	# early stopping, walk-forward 1-step-ahead predict on the test slice, reconstruct
	# prices, compute price + strategy metrics. Returns (metrics_dict, predictions_df).
	# `model_overrides` lets sweep.py poke individual hyperparams without rewriting cfg.
	# `extra_pl_callbacks` lets optuna_sweep.py inject a PyTorchLightningPruningCallback
	# without forcing the early-stopping callback to be reconstructed in callers.
	cov_cfg = cfg.get('covariates', {})
	model_cfg = dict(cfg['model'])
	if model_overrides is not None:
		model_cfg.update(model_overrides)
	train_cfg = cfg['training']

	splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)
	df = pd.concat([splits.train, splits.validation, splits.test])
	pre_val_rows = len(splits.train)
	pre_test_rows = len(splits.train) + len(splits.validation)
	# Drop tz so darts' time indexing is unambiguous (matches 02_arima's convention).
	df.index = df.index.tz_localize(None)
	logger.info('loaded %d rows from %s to %s', len(df), df.index[0], df.index[-1])

	target_pd, cov_pd, ref_pd = build_target_and_covariates(df, cov_cfg)
	# build_target_and_covariates drops leading rows that lack history (diff/lag NaNs).
	# Translate the bar-level boundary indices to target-row indices via timestamps.
	train_end = target_pd.index.get_indexer([df.index[pre_val_rows]])[0]
	val_end = target_pd.index.get_indexer([df.index[pre_test_rows]])[0]
	n = len(target_pd)
	logger.info(
		'split: train=[0:%d] val=[%d:%d] test=[%d:%d]',
		train_end, train_end, val_end, val_end, n,
	)

	# Cast to float32 for the torch model: MPS does not support float64 tensors,
	# and float32 is plenty for log-returns at this magnitude. Price reconstruction
	# downstream uses the float64 ref / target_pd, so accuracy of MAE/MAPE is preserved.
	target_full_ts = TimeSeries.from_series(target_pd.astype('float32'))
	cov_full_ts = (
		TimeSeries.from_dataframe(cov_pd.astype('float32'))
		if cov_pd is not None and not cov_pd.empty
		else None
	)

	target_train = target_full_ts[:train_end]
	target_val = target_full_ts[train_end:val_end]

	scaler_target = Scaler()
	scaler_target.fit(target_train)
	target_train_s = scaler_target.transform(target_train)
	target_val_s = scaler_target.transform(target_val)
	target_full_s = scaler_target.transform(target_full_ts)

	cov_train_s = None
	cov_val_s = None
	cov_full_s = None
	if cov_full_ts is not None:
		scaler_cov = Scaler()
		cov_train = cov_full_ts[:train_end]
		cov_val = cov_full_ts[train_end:val_end]
		scaler_cov.fit(cov_train)
		cov_train_s = scaler_cov.transform(cov_train)
		cov_val_s = scaler_cov.transform(cov_val)
		cov_full_s = scaler_cov.transform(cov_full_ts)

	early_stop = EarlyStopping(
		monitor='val_loss',
		patience=int(train_cfg['early_stopping_patience']),
		mode='min',
	)
	callbacks: list[Any] = [early_stop]
	if extra_pl_callbacks is not None:
		callbacks.extend(extra_pl_callbacks)
	pl_trainer_kwargs: dict[str, Any] = {
		'callbacks': callbacks,
		'accelerator': 'auto',
		'enable_progress_bar': False,
		'enable_model_summary': False,
		'logger': False,
	}
	model = BlockRNNModel(
		model='LSTM',
		input_chunk_length=int(model_cfg['input_chunk_length']),
		output_chunk_length=int(model_cfg['output_chunk_length']),
		hidden_dim=int(model_cfg['hidden_dim']),
		n_rnn_layers=int(model_cfg['n_rnn_layers']),
		dropout=float(model_cfg['dropout']),
		batch_size=int(model_cfg['batch_size']),
		n_epochs=int(model_cfg['n_epochs']),
		optimizer_kwargs={
			'lr': float(model_cfg['learning_rate']),
			'weight_decay': float(model_cfg['weight_decay']),
		},
		random_state=int(model_cfg['random_state']),
		pl_trainer_kwargs=pl_trainer_kwargs,
		save_checkpoints=False,
		force_reset=True,
	)
	logger.info('fitting BlockRNN-LSTM on %d training rows', len(target_train_s))
	model.fit(
		series=target_train_s,
		past_covariates=cov_train_s,
		val_series=target_val_s,
		val_past_covariates=cov_val_s,
	)

	# Walk-forward 1-step prediction across the test window. retrain=False reuses the
	# parameters fit above; the model just slides its input window across test.
	test_start = target_full_s.time_index[val_end]
	preds_s = model.historical_forecasts(
		series=target_full_s,
		past_covariates=cov_full_s,
		start=test_start,
		forecast_horizon=1,
		retrain=False,
		last_points_only=True,
		verbose=False,
	)
	preds = scaler_target.inverse_transform(preds_s).to_series().rename('r_pred')

	target_test = target_pd.iloc[val_end:]
	ref_test = ref_pd.iloc[val_end:]
	common = preds.index.intersection(target_test.index)
	target_test = target_test.loc[common]
	ref_test = ref_test.loc[common]
	preds = preds.loc[common]

	close_test = ref_test * np.exp(target_test)
	pred_close = ref_test * np.exp(preds)

	ppy = periods_per_year(splits.interval)
	strat = strategy_returns(close_test, pred_close, ref_test)

	metrics: dict[str, Any] = {
		'experiment': '04_lstm',
		'dataset': cfg['dataset'],
		'symbol': splits.symbol_test,
		'interval': splits.interval,
		'rows_total': int(n),
		'rows_train': int(train_end),
		'rows_val': int(val_end - train_end),
		'rows_test': int(len(common)),
		'mae': mae(close_test, pred_close),
		'rmse': rmse(close_test, pred_close),
		'mape': mape(close_test, pred_close),
		'directional_accuracy': directional_accuracy(close_test, pred_close, ref_test),
		'cumulative_return': cumulative_return(strat),
		'annualized_sharpe': annualized_sharpe(strat, ppy),
	}
	predictions = pd.DataFrame({
		'close': close_test,
		'pred': pred_close,
		'ref': ref_test,
		'strategy_return': strat,
	})
	return metrics, predictions


def main() -> None:
	logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

	parser = argparse.ArgumentParser(description='Darts BlockRNN-LSTM on 1h log-returns.')
	parser.add_argument('--config', type=Path, required=True)
	args = parser.parse_args()

	results_dir = EXPERIMENT_DIR / 'results' / args.config.stem
	results_dir.mkdir(parents=True, exist_ok=True)

	cfg = load_yaml(args.config)
	metrics, predictions = train_and_evaluate(cfg)

	(results_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
	predictions.to_parquet(results_dir / 'predictions.parquet')

	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(predictions.index, predictions['close'].values, label='close', linewidth=1)
	ax.plot(
		predictions.index,
		predictions['pred'].values,
		label='LSTM pred',
		linewidth=1,
		alpha=0.7,
	)
	ax.set_title(
		f'{metrics["symbol"]} {metrics["interval"]} — BlockRNN-LSTM (test set)',
	)
	ax.set_ylabel('price')
	ax.legend()
	fig.tight_layout()
	fig.savefig(results_dir / 'plot.png', dpi=120)
	plt.close(fig)

	print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
	main()
