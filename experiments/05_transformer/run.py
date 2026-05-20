import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

# Quiet noisy third-party imports before darts/lightning load:
# - UserWarning covers PL's _pytree deprecation + torch's pin_memory MPS notice
# - darts emits a WARNING at import about missing statsforecast (we don't use it)
warnings.filterwarnings('ignore')
logging.getLogger('darts').setLevel(logging.ERROR)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from darts import TimeSeries  # noqa: E402
from darts.dataprocessing.transformers import Scaler  # noqa: E402
from darts.models import TFTModel  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping  # noqa: E402

from btc_ai.config import load_yaml  # noqa: E402
from btc_ai.data import (  # noqa: E402
	PerSymbolFrame,
	build_per_symbol_frames,
	load_dataset_from_experiment_cfg,
)
from btc_ai.eval.aggregate import (  # noqa: E402
	PerSymbolResult,
	aggregate_metrics,
	concat_predictions,
	write_aggregated_metrics_json,
)
from btc_ai.eval.metrics import (  # noqa: E402
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
) -> tuple[pd.Series, pd.DataFrame | None, pd.DataFrame | None, pd.Series]:
	# target r_T = log(close_T) - log(close_{T-1}); ref_T = close_{T-1} for downstream
	# price reconstruction. Past covariates use bar-T OHLCV; darts feeds them strictly
	# before each forecast step so there is no current-bar leakage. Future covariates
	# are cyclical encodings of hour-of-day and day-of-week — deterministic functions
	# of the timestamp, so they are legitimately known at any future bar.
	close = df['close'].astype('float64')
	target = np.log(close).diff().rename('r_target')
	ref = close.shift(1).rename('ref_close')

	past_cols: dict[str, pd.Series] = {}
	if cov_cfg.get('use_volume') is True:
		past_cols['log_volume'] = np.log1p(df['volume'].astype('float64'))
	if cov_cfg.get('use_ohlc') is True:
		past_cols['hl_range'] = (df['high'] - df['low']).astype('float64')
		past_cols['oc_body'] = (df['close'] - df['open']).astype('float64')

	future_cols: dict[str, pd.Series] = {}
	if cov_cfg.get('use_time_features') is True:
		idx = df.index
		hour = idx.hour.to_numpy(dtype='float64')
		dow = idx.dayofweek.to_numpy(dtype='float64')
		future_cols['hour_sin'] = pd.Series(np.sin(2 * np.pi * hour / 24), index=idx)
		future_cols['hour_cos'] = pd.Series(np.cos(2 * np.pi * hour / 24), index=idx)
		future_cols['dow_sin'] = pd.Series(np.sin(2 * np.pi * dow / 7), index=idx)
		future_cols['dow_cos'] = pd.Series(np.cos(2 * np.pi * dow / 7), index=idx)

	frames = [target, ref]
	if past_cols:
		frames.append(pd.DataFrame(past_cols, index=df.index))
	if future_cols:
		frames.append(pd.DataFrame(future_cols, index=df.index))
	joined = pd.concat(frames, axis=1).dropna()

	past_df = joined[list(past_cols.keys())] if past_cols else None
	future_df = joined[list(future_cols.keys())] if future_cols else None
	return joined['r_target'], past_df, future_df, joined['ref_close']


@dataclass
class _PreparedSymbol:
	frame: PerSymbolFrame
	target_pd: pd.Series
	ref_pd: pd.Series
	train_end: int
	val_end: int
	scaler_target: Scaler
	target_train_s: TimeSeries
	target_val_s: TimeSeries
	target_full_s: TimeSeries
	past_train_s: TimeSeries | None
	past_val_s: TimeSeries | None
	past_full_s: TimeSeries | None
	future_train_s: TimeSeries | None
	future_val_s: TimeSeries | None
	future_full_s: TimeSeries | None


def _prepare_symbol(frame: PerSymbolFrame, cov_cfg: dict[str, Any]) -> _PreparedSymbol:
	df = frame.df
	target_pd, past_pd, future_pd, ref_pd = build_target_and_covariates(df, cov_cfg)
	if frame.val_start_ts is None:
		raise ValueError(
			f'{frame.market} {frame.symbol}: validation selector required for TFT early stopping',
		)
	train_end = target_pd.index.get_indexer([frame.val_start_ts])[0]
	val_end = target_pd.index.get_indexer([frame.test_start_ts])[0]

	target_full_ts = TimeSeries.from_series(target_pd.astype('float32'))
	past_full_ts = (
		TimeSeries.from_dataframe(past_pd.astype('float32'))
		if past_pd is not None and not past_pd.empty
		else None
	)
	future_full_ts = (
		TimeSeries.from_dataframe(future_pd.astype('float32'))
		if future_pd is not None and not future_pd.empty
		else None
	)

	target_train = target_full_ts[:train_end]
	target_val = target_full_ts[train_end:val_end]

	scaler_target = Scaler()
	scaler_target.fit(target_train)
	target_train_s = scaler_target.transform(target_train)
	target_val_s = scaler_target.transform(target_val)
	target_full_s = scaler_target.transform(target_full_ts)

	past_train_s = past_val_s = past_full_s = None
	if past_full_ts is not None:
		scaler_past = Scaler()
		past_train = past_full_ts[:train_end]
		past_val = past_full_ts[train_end:val_end]
		scaler_past.fit(past_train)
		past_train_s = scaler_past.transform(past_train)
		past_val_s = scaler_past.transform(past_val)
		past_full_s = scaler_past.transform(past_full_ts)

	# Future covariates are already bounded in [-1, 1] (sin/cos), but scaling on the
	# train slice keeps the pipeline uniform and lets darts treat them like any other
	# real-valued covariate. Skip if the user opted into add_relative_index instead.
	future_train_s = future_val_s = future_full_s = None
	if future_full_ts is not None:
		scaler_future = Scaler()
		future_train = future_full_ts[:train_end]
		future_val = future_full_ts[train_end:val_end]
		scaler_future.fit(future_train)
		future_train_s = scaler_future.transform(future_train)
		future_val_s = scaler_future.transform(future_val)
		future_full_s = scaler_future.transform(future_full_ts)

	return _PreparedSymbol(
		frame=frame,
		target_pd=target_pd,
		ref_pd=ref_pd,
		train_end=train_end,
		val_end=val_end,
		scaler_target=scaler_target,
		target_train_s=target_train_s,
		target_val_s=target_val_s,
		target_full_s=target_full_s,
		past_train_s=past_train_s,
		past_val_s=past_val_s,
		past_full_s=past_full_s,
		future_train_s=future_train_s,
		future_val_s=future_val_s,
		future_full_s=future_full_s,
	)


def train_and_evaluate(
	cfg: dict[str, Any],
	model_overrides: dict[str, Any] | None = None,
	extra_pl_callbacks: list[Any] | None = None,
) -> tuple[dict[str, float], list[PerSymbolResult], str]:
	# End-to-end pipeline shared by run.py, sweep.py, and optuna_sweep.py. Trains
	# one TFT over a list of per-symbol target series (multi-series fit), then
	# walk-forwards per test symbol and aggregates metrics across symbols.
	# Returns (aggregate, per-symbol results, interval).
	cov_cfg = cfg.get('covariates', {})
	model_cfg = dict(cfg['model'])
	if model_overrides is not None:
		model_cfg.update(model_overrides)
	train_cfg = cfg['training']

	# TFT requires either real future_covariates or add_relative_index=True. Surface
	# misconfiguration eagerly rather than letting darts raise downstream.
	use_time_features = cov_cfg.get('use_time_features') is True
	add_relative_index = bool(model_cfg['add_relative_index'])
	if use_time_features is False and add_relative_index is False:
		raise ValueError(
			'TFT requires future covariates: set covariates.use_time_features=true '
			'or model.add_relative_index=true in the config.',
		)

	splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)
	frames = build_per_symbol_frames(splits)
	logger.info('loaded %d symbol frame(s); interval=%s', len(frames), splits.interval)

	prepared = [_prepare_symbol(f, cov_cfg) for f in frames]

	target_train_list = [p.target_train_s for p in prepared]
	target_val_list = [p.target_val_s for p in prepared]
	past_train_list = (
		[p.past_train_s for p in prepared]
		if prepared and prepared[0].past_train_s is not None else None
	)
	past_val_list = (
		[p.past_val_s for p in prepared]
		if prepared and prepared[0].past_val_s is not None else None
	)
	future_train_list = (
		[p.future_train_s for p in prepared]
		if prepared and prepared[0].future_train_s is not None else None
	)
	future_val_list = (
		[p.future_val_s for p in prepared]
		if prepared and prepared[0].future_val_s is not None else None
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
	model = TFTModel(
		input_chunk_length=int(model_cfg['input_chunk_length']),
		output_chunk_length=int(model_cfg['output_chunk_length']),
		hidden_size=int(model_cfg['hidden_size']),
		lstm_layers=int(model_cfg['lstm_layers']),
		num_attention_heads=int(model_cfg['num_attention_heads']),
		dropout=float(model_cfg['dropout']),
		hidden_continuous_size=int(model_cfg['hidden_continuous_size']),
		add_relative_index=add_relative_index,
		full_attention=bool(model_cfg['full_attention']),
		batch_size=int(model_cfg['batch_size']),
		n_epochs=int(model_cfg['n_epochs']),
		optimizer_kwargs={'lr': float(model_cfg['learning_rate'])},
		random_state=int(model_cfg['random_state']),
		pl_trainer_kwargs=pl_trainer_kwargs,
		save_checkpoints=False,
		force_reset=True,
	)
	logger.info(
		'fitting TFT on %d series, %d train rows total (max %d epochs, patience=%d)',
		len(target_train_list),
		sum(len(s) for s in target_train_list),
		int(model_cfg['n_epochs']),
		int(train_cfg['early_stopping_patience']),
	)
	model.fit(
		series=target_train_list,
		past_covariates=past_train_list,
		future_covariates=future_train_list,
		val_series=target_val_list,
		val_past_covariates=past_val_list,
		val_future_covariates=future_val_list,
	)
	logger.info('training complete')

	results = [_evaluate_symbol(p, model, splits.interval) for p in prepared]
	return aggregate_metrics(results), results, splits.interval


def _evaluate_symbol(
	p: _PreparedSymbol, model: TFTModel, interval: str,
) -> PerSymbolResult:
	frame = p.frame
	test_start = p.target_full_s.time_index[p.val_end]
	logger.info(
		'walk-forward 1-step from %s over %d test bars (%s %s)',
		test_start, len(p.target_full_s) - p.val_end, frame.market, frame.symbol,
	)
	preds_s = model.historical_forecasts(
		series=p.target_full_s,
		past_covariates=p.past_full_s,
		future_covariates=p.future_full_s,
		start=test_start,
		forecast_horizon=1,
		retrain=False,
		last_points_only=True,
		verbose=True,
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
	# Drop Lightning's "GPU/TPU/HPU available" startup banner without losing our INFO logs.
	logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

	parser = argparse.ArgumentParser(description='Darts TFT on 1h log-returns.')
	parser.add_argument('--config', type=Path, required=True)
	args = parser.parse_args()

	results_dir = EXPERIMENT_DIR / 'results' / args.config.stem.removesuffix('.config')
	results_dir.mkdir(parents=True, exist_ok=True)

	cfg = load_yaml(args.config)
	_aggregate, results, interval = train_and_evaluate(cfg)

	payload = write_aggregated_metrics_json(
		results,
		results_dir / 'metrics.json',
		experiment='05_transformer',
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
		ax.plot(pred_close.index, pred_close.values, label='TFT pred', linewidth=1, alpha=0.7)
		ax.set_title(f'{r.symbol} {interval} — TFT (test set)')
		ax.set_ylabel('price')
		ax.legend()
	fig.tight_layout()
	fig.savefig(out_path, dpi=120)
	plt.close(fig)


if __name__ == '__main__':
	main()
