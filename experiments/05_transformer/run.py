import argparse
import json
import logging
import warnings
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

from btc_ai.config import kline_request_from_config, load_yaml  # noqa: E402
from btc_ai.data import BinanceVisionLoader  # noqa: E402
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
# results_dir is derived inside main() from the config filename stem
# (e.g. configs/btc_4h_2024.yaml → results/btc_4h_2024/).
CACHE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw'

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


def split_three_way(
	n: int,
	test_fraction: float,
	val_fraction: float,
) -> tuple[int, int]:
	# Returns (train_end, val_end). test = [val_end:n]. test_fraction is taken from
	# the full series; val_fraction is then taken from the *remaining* train tail so
	# val never overlaps test.
	test_size = int(n * test_fraction)
	val_end = n - test_size
	val_size = int(val_end * val_fraction)
	train_end = val_end - val_size
	return train_end, val_end


def train_and_evaluate(
	cfg: dict[str, Any],
	model_overrides: dict[str, Any] | None = None,
	extra_pl_callbacks: list[Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
	# End-to-end pipeline shared by run.py, sweep.py, and optuna_sweep.py: load data,
	# build target + past + future covariates, fit Scaler on train only, train TFT
	# with val-based early stopping, walk-forward 1-step-ahead predict on the test
	# slice, reconstruct prices, compute price + strategy metrics.
	# Returns (metrics_dict, predictions_df). `model_overrides` lets callers poke
	# individual hyperparams without rewriting cfg; `extra_pl_callbacks` lets
	# optuna_sweep.py inject a per-trial PruningCallback alongside EarlyStopping.
	req = kline_request_from_config(cfg)
	test_fraction = float(cfg['test_fraction'])
	cov_cfg = cfg.get('covariates', {})
	model_cfg = dict(cfg['model'])
	if model_overrides is not None:
		model_cfg.update(model_overrides)
	train_cfg = cfg['training']
	val_fraction = float(train_cfg['val_fraction'])

	# TFT requires either real future_covariates or add_relative_index=True. Surface
	# misconfiguration eagerly rather than letting darts raise downstream.
	use_time_features = cov_cfg.get('use_time_features') is True
	add_relative_index = bool(model_cfg['add_relative_index'])
	if use_time_features is False and add_relative_index is False:
		raise ValueError(
			'TFT requires future covariates: set covariates.use_time_features=true '
			'or model.add_relative_index=true in the config.',
		)

	loader = BinanceVisionLoader(cache_dir=CACHE_DIR)
	df = loader.load(req)
	# Drop tz so darts' time indexing is unambiguous (matches 02_arima/04_lstm).
	df.index = df.index.tz_localize(None)
	logger.info('loaded %d rows from %s to %s', len(df), df.index[0], df.index[-1])

	target_pd, past_pd, future_pd, ref_pd = build_target_and_covariates(df, cov_cfg)
	n = len(target_pd)
	train_end, val_end = split_three_way(n, test_fraction, val_fraction)
	logger.info(
		'split: train=[0:%d] val=[%d:%d] test=[%d:%d]',
		train_end, train_end, val_end, val_end, n,
	)

	# Cast to float32 for the torch model: MPS does not support float64 tensors,
	# and float32 is plenty for log-returns at this magnitude. Price reconstruction
	# downstream uses the float64 ref / target_pd, so accuracy of MAE/MAPE is preserved.
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

	past_train_s = None
	past_val_s = None
	past_full_s = None
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
	future_train_s = None
	future_val_s = None
	future_full_s = None
	if future_full_ts is not None:
		scaler_future = Scaler()
		future_train = future_full_ts[:train_end]
		future_val = future_full_ts[train_end:val_end]
		scaler_future.fit(future_train)
		future_train_s = scaler_future.transform(future_train)
		future_val_s = scaler_future.transform(future_val)
		future_full_s = scaler_future.transform(future_full_ts)

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
		'fitting TFT on %d training rows for up to %d epochs '
		'(early-stopping patience=%d on val_loss; PL progress bar suppressed)',
		len(target_train_s),
		int(model_cfg['n_epochs']),
		int(train_cfg['early_stopping_patience']),
	)
	model.fit(
		series=target_train_s,
		past_covariates=past_train_s,
		future_covariates=future_train_s,
		val_series=target_val_s,
		val_past_covariates=past_val_s,
		val_future_covariates=future_val_s,
	)
	logger.info('training complete')

	# Walk-forward 1-step prediction across the test window. retrain=False reuses the
	# parameters fit above; the model just slides its input window across test.
	test_start = target_full_s.time_index[val_end]
	test_bars = len(target_full_s) - val_end
	logger.info(
		'walk-forward 1-step from %s over %d test bars',
		test_start, test_bars,
	)
	# verbose=True surfaces darts' tqdm bar across the test window — without
	# it the walk-forward sits silent and looks hung on slower hardware.
	preds_s = model.historical_forecasts(
		series=target_full_s,
		past_covariates=past_full_s,
		future_covariates=future_full_s,
		start=test_start,
		forecast_horizon=1,
		retrain=False,
		last_points_only=True,
		verbose=True,
	)
	logger.info('walk-forward complete; reconstructing prices')
	preds = scaler_target.inverse_transform(preds_s).to_series().rename('r_pred')

	target_test = target_pd.iloc[val_end:]
	ref_test = ref_pd.iloc[val_end:]
	common = preds.index.intersection(target_test.index)
	target_test = target_test.loc[common]
	ref_test = ref_test.loc[common]
	preds = preds.loc[common]

	close_test = ref_test * np.exp(target_test)
	pred_close = ref_test * np.exp(preds)

	ppy = periods_per_year(req.interval)
	strat = strategy_returns(close_test, pred_close, ref_test)

	metrics: dict[str, Any] = {
		'experiment': '05_transformer',
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
	# Drop Lightning's "GPU/TPU/HPU available" startup banner without losing our INFO logs.
	logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

	parser = argparse.ArgumentParser(description='Darts TFT on 1h log-returns.')
	parser.add_argument('--config', type=Path, required=True)
	args = parser.parse_args()

	results_dir = EXPERIMENT_DIR / 'results' / args.config.stem
	results_dir.mkdir(parents=True, exist_ok=True)

	cfg = load_yaml(args.config)
	metrics, predictions = train_and_evaluate(cfg)

	(results_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
	predictions.to_parquet(results_dir / 'predictions.parquet')

	req = kline_request_from_config(cfg)
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(predictions.index, predictions['close'].values, label='close', linewidth=1)
	ax.plot(
		predictions.index,
		predictions['pred'].values,
		label='TFT pred',
		linewidth=1,
		alpha=0.7,
	)
	ax.set_title(f'{req.symbol} {req.interval} — TFT (test set)')
	ax.set_ylabel('price')
	ax.legend()
	fig.tight_layout()
	fig.savefig(results_dir / 'plot.png', dpi=120)
	plt.close(fig)

	print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
	main()
