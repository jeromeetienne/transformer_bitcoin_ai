import argparse
import json
import logging
from dataclasses import dataclass
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
from btc_ai.data import (
	PerSymbolFrame,
	build_per_symbol_frames,
	load_dataset_from_experiment_cfg,
)
from btc_ai.eval.aggregate import (
	PerSymbolResult,
	aggregate_metrics,
	concat_predictions,
	write_aggregated_metrics_json,
)
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


@dataclass
class _PreparedSymbol:
	frame: PerSymbolFrame
	target_pd: pd.Series
	cov_pd: pd.DataFrame | None
	ref_pd: pd.Series
	train_end: int
	val_end: int
	scaler_target: Scaler
	scaler_cov: Scaler | None
	target_train_s: TimeSeries
	target_val_s: TimeSeries
	target_full_s: TimeSeries
	cov_train_s: TimeSeries | None
	cov_val_s: TimeSeries | None
	cov_full_s: TimeSeries | None


def _prepare_symbol(frame: PerSymbolFrame, cov_cfg: dict[str, Any]) -> _PreparedSymbol:
	df = frame.df
	target_pd, cov_pd, ref_pd = build_target_and_covariates(df, cov_cfg)
	if frame.val_start_ts is None:
		raise ValueError(
			f'{frame.market} {frame.symbol}: validation selector required for '
			f'BlockRNN-LSTM early stopping (val_loss patience)',
		)
	train_end = target_pd.index.get_indexer([frame.val_start_ts])[0]
	val_end = target_pd.index.get_indexer([frame.test_start_ts])[0]

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

	scaler_cov: Scaler | None = None
	cov_train_s: TimeSeries | None = None
	cov_val_s: TimeSeries | None = None
	cov_full_s: TimeSeries | None = None
	if cov_full_ts is not None:
		scaler_cov = Scaler()
		cov_train = cov_full_ts[:train_end]
		cov_val = cov_full_ts[train_end:val_end]
		scaler_cov.fit(cov_train)
		cov_train_s = scaler_cov.transform(cov_train)
		cov_val_s = scaler_cov.transform(cov_val)
		cov_full_s = scaler_cov.transform(cov_full_ts)

	return _PreparedSymbol(
		frame=frame,
		target_pd=target_pd,
		cov_pd=cov_pd,
		ref_pd=ref_pd,
		train_end=train_end,
		val_end=val_end,
		scaler_target=scaler_target,
		scaler_cov=scaler_cov,
		target_train_s=target_train_s,
		target_val_s=target_val_s,
		target_full_s=target_full_s,
		cov_train_s=cov_train_s,
		cov_val_s=cov_val_s,
		cov_full_s=cov_full_s,
	)


def train_and_evaluate(
	cfg: dict[str, Any],
	model_overrides: dict[str, Any] | None = None,
	extra_pl_callbacks: list[Any] | None = None,
) -> tuple[dict[str, float], list[PerSymbolResult], str]:
	# End-to-end pipeline shared by run.py and sweep.py. Trains one BlockRNN-LSTM
	# over a list of per-symbol target series (multi-series fit), then walk-forwards
	# per test symbol and aggregates metrics. Returns (payload, per-symbol results, interval).
	# `model_overrides` lets sweep.py poke individual hyperparams without rewriting cfg.
	# `extra_pl_callbacks` lets optuna_sweep.py inject a PyTorchLightningPruningCallback.
	cov_cfg = cfg.get('covariates', {})
	model_cfg = dict(cfg['model'])
	if model_overrides is not None:
		model_cfg.update(model_overrides)
	train_cfg = cfg['training']

	splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)
	frames = build_per_symbol_frames(splits)
	logger.info(
		'loaded %d symbol frame(s); interval=%s', len(frames), splits.interval,
	)

	prepared = [_prepare_symbol(f, cov_cfg) for f in frames]

	target_train_list = [p.target_train_s for p in prepared]
	target_val_list = [p.target_val_s for p in prepared]
	cov_train_list = (
		[p.cov_train_s for p in prepared]
		if prepared and prepared[0].cov_train_s is not None else None
	)
	cov_val_list = (
		[p.cov_val_s for p in prepared]
		if prepared and prepared[0].cov_val_s is not None else None
	)

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
	logger.info(
		'fitting BlockRNN-LSTM on %d series, %d train rows total',
		len(target_train_list), sum(len(s) for s in target_train_list),
	)
	model.fit(
		series=target_train_list,
		past_covariates=cov_train_list,
		val_series=target_val_list,
		val_past_covariates=cov_val_list,
	)

	results = [_evaluate_symbol(p, model, splits.interval) for p in prepared]
	return aggregate_metrics(results), results, splits.interval


def _evaluate_symbol(
	p: _PreparedSymbol, model: BlockRNNModel, interval: str,
) -> PerSymbolResult:
	frame = p.frame
	# Walk-forward 1-step prediction across the test window. retrain=False reuses
	# the parameters fit above; the model just slides its input window across test.
	test_start = p.target_full_s.time_index[p.val_end]
	preds_s = model.historical_forecasts(
		series=p.target_full_s,
		past_covariates=p.cov_full_s,
		start=test_start,
		forecast_horizon=1,
		retrain=False,
		last_points_only=True,
		verbose=False,
	)
	preds = p.scaler_target.inverse_transform(preds_s).to_series().rename('r_pred')

	target_test = p.target_pd.iloc[p.val_end:]
	ref_test = p.ref_pd.iloc[p.val_end:]
	common = preds.index.intersection(target_test.index)
	target_test = target_test.loc[common]
	ref_test = ref_test.loc[common]
	preds = preds.loc[common]

	close_test = ref_test * np.exp(target_test)
	pred_close = ref_test * np.exp(preds)

	ppy = periods_per_year(interval)
	strat = strategy_returns(close_test, pred_close, ref_test)

	predictions = pd.DataFrame({
		'close': close_test,
		'pred': pred_close,
		'ref': ref_test,
		'strategy_return': strat,
	})
	return PerSymbolResult(
		symbol=frame.symbol,
		market=frame.market,
		n_test_rows=int(len(common)),
		metrics={
			'mae': mae(close_test, pred_close),
			'rmse': rmse(close_test, pred_close),
			'mape': mape(close_test, pred_close),
			'directional_accuracy': directional_accuracy(close_test, pred_close, ref_test),
			'cumulative_return': cumulative_return(strat),
			'annualized_sharpe': annualized_sharpe(strat, ppy),
		},
		predictions=predictions,
		extras={
			'rows_train': int(p.train_end),
			'rows_val': int(p.val_end - p.train_end),
		},
	)


def main() -> None:
	logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

	parser = argparse.ArgumentParser(description='Darts BlockRNN-LSTM on 1h log-returns.')
	parser.add_argument('--config', type=Path, required=True)
	args = parser.parse_args()

	results_dir = EXPERIMENT_DIR / 'results' / args.config.stem.removesuffix('.config')
	results_dir.mkdir(parents=True, exist_ok=True)

	cfg = load_yaml(args.config)
	_aggregate, results, interval = train_and_evaluate(cfg)

	payload = write_aggregated_metrics_json(
		results,
		results_dir / 'metrics.json',
		experiment='04_lstm',
		dataset=cfg['dataset'],
		interval=interval,
	)

	concat_predictions(results).to_parquet(results_dir / 'predictions.parquet')

	_plot(results, interval, results_dir / 'plot.png')

	print(json.dumps(payload, indent=2))


def _plot(results: list[PerSymbolResult], interval: str, out_path: Path) -> None:
	n = len(results)
	fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)
	for ax, r in zip(axes[:, 0], results, strict=True):
		close_test = r.predictions['close']
		pred_close = r.predictions['pred']
		ax.plot(close_test.index, close_test.values, label='close', linewidth=1)
		ax.plot(pred_close.index, pred_close.values, label='LSTM pred', linewidth=1, alpha=0.7)
		ax.set_title(f'{r.symbol} {interval} — BlockRNN-LSTM (test set)')
		ax.set_ylabel('price')
		ax.legend()
	fig.tight_layout()
	fig.savefig(out_path, dpi=120)
	plt.close(fig)


if __name__ == '__main__':
	main()
