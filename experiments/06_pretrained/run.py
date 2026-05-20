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
from darts.models import Chronos2Model, TimesFM2p5Model
from darts.utils.likelihood_models.torch import QuantileRegression

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

VALID_BACKENDS: tuple[str, ...] = ('chronos', 'timesfm')

logger = logging.getLogger(__name__)


def build_target(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
	# target r_T = log(close_T) - log(close_{T-1}); ref_T = close_{T-1} for
	# downstream price reconstruction. Univariate by design — neither backend
	# is given covariates here (TimesFM 2.5 doesn't support them; we keep
	# Chronos-2 univariate too so the head-to-head comparison is on the same
	# inputs).
	close = df['close'].astype('float64')
	target = np.log(close).diff().rename('r_target')
	ref = close.shift(1).rename('ref_close')
	joined = pd.concat([target, ref], axis=1).dropna()
	return joined['r_target'], joined['ref_close']


def build_model(
	backend: str,
	hub_model_name: str,
	model_cfg: dict[str, Any],
	quantiles: list[float],
) -> Chronos2Model | TimesFM2p5Model:
	pl_trainer_kwargs: dict[str, Any] = {
		'accelerator': 'auto',
		'enable_progress_bar': False,
		'enable_model_summary': False,
		'logger': False,
	}
	likelihood = QuantileRegression(quantiles=quantiles)
	common_kwargs: dict[str, Any] = {
		'input_chunk_length': int(model_cfg['input_chunk_length']),
		'output_chunk_length': int(model_cfg['output_chunk_length']),
		'hub_model_name': hub_model_name,
		'likelihood': likelihood,
		'pl_trainer_kwargs': pl_trainer_kwargs,
	}
	if backend == 'chronos':
		return Chronos2Model(**common_kwargs)
	if backend == 'timesfm':
		return TimesFM2p5Model(**common_kwargs)
	raise ValueError(f'unknown backend={backend!r}; expected one of {VALID_BACKENDS}')


@dataclass
class _PreparedSymbol:
	frame: PerSymbolFrame
	target_pd: pd.Series
	ref_pd: pd.Series
	train_end: int                  # fold val into training, so train_end == val_end of legacy code
	scaler_target: Scaler
	target_train_s: TimeSeries
	target_full_s: TimeSeries


def _prepare_symbol(frame: PerSymbolFrame) -> _PreparedSymbol:
	# Zero-shot: fold validation (if present) into the training segment, matching
	# the legacy single-split behavior. train_end is the boundary between
	# train+validation and test.
	df = frame.df
	target_pd, ref_pd = build_target(df)
	train_end = target_pd.index.get_indexer([frame.test_start_ts])[0]

	target_full_ts = TimeSeries.from_series(target_pd.astype('float32'))
	target_train = target_full_ts[:train_end]

	scaler_target = Scaler()
	scaler_target.fit(target_train)
	target_train_s = scaler_target.transform(target_train)
	target_full_s = scaler_target.transform(target_full_ts)

	return _PreparedSymbol(
		frame=frame,
		target_pd=target_pd,
		ref_pd=ref_pd,
		train_end=train_end,
		scaler_target=scaler_target,
		target_train_s=target_train_s,
		target_full_s=target_full_s,
	)


def train_and_evaluate(
	cfg: dict[str, Any],
	model_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, float], list[PerSymbolResult], str]:
	# End-to-end pipeline shared by run.py and sweep.py. Fits (no-op for zero-shot)
	# the requested foundation model on the per-symbol training series, then
	# walk-forwards per test symbol and aggregates metrics.
	merged: dict[str, Any] = {
		'backend': cfg['backend'],
		'hub_model_name': cfg['hub_model_name'],
		**dict(cfg['model']),
	}
	if model_overrides is not None:
		merged.update(model_overrides)

	backend = str(merged['backend'])
	if backend not in VALID_BACKENDS:
		raise ValueError(f'backend must be one of {VALID_BACKENDS}; got {backend!r}')
	hub_model_name = str(merged['hub_model_name'])
	quantiles = [float(q) for q in merged['quantiles']]
	if len(quantiles) != 3:
		raise ValueError(
			f'expected exactly 3 quantiles (low, median, high); got {quantiles}',
		)
	if quantiles[1] != 0.5:
		raise ValueError(
			f'middle quantile must be 0.5 (used for point metrics); got {quantiles}',
		)
	num_samples = int(merged['num_samples'])

	splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)
	frames = build_per_symbol_frames(splits)
	logger.info('loaded %d symbol frame(s); interval=%s', len(frames), splits.interval)

	prepared = [_prepare_symbol(f) for f in frames]

	logger.info('instantiating %s (%s)', backend, hub_model_name)
	model = build_model(backend, hub_model_name, merged, quantiles)
	logger.info(
		'fit %s on %d series (zero-shot, no weight updates). '
		'First run downloads HuggingFace weights (chronos-2-small ≈120 MB, '
		'chronos-2 ≈300 MB, timesfm-2.5 ≈800 MB), cached at ~/.cache/huggingface/hub/.',
		backend, len(prepared),
	)
	# fit() is required by the darts API but performs no weight updates for
	# foundation models. It triggers the HuggingFace weight download on first
	# invocation. Multi-series fit is supported and equally no-op per series.
	train_list = [p.target_train_s for p in prepared]
	model.fit(train_list if len(train_list) > 1 else train_list[0])
	logger.info('model ready')

	results = [
		_evaluate_symbol(p, model, splits.interval, quantiles, num_samples)
		for p in prepared
	]
	return aggregate_metrics(results), results, splits.interval


def _evaluate_symbol(
	p: _PreparedSymbol,
	model: Chronos2Model | TimesFM2p5Model,
	interval: str,
	quantiles: list[float],
	num_samples: int,
) -> PerSymbolResult:
	frame = p.frame
	test_start = p.target_full_s.time_index[p.train_end]
	logger.info(
		'walk-forward 1-step from %s over %d test bars (%s %s, num_samples=%d)',
		test_start, len(p.target_full_s) - p.train_end,
		frame.market, frame.symbol, num_samples,
	)
	preds_s = model.historical_forecasts(
		series=p.target_full_s,
		start=test_start,
		forecast_horizon=1,
		retrain=False,
		last_points_only=True,
		num_samples=num_samples,
		verbose=True,
	)
	# Stochastic series — inverse-scale on the full distribution, then extract
	# quantiles. Linear scaling preserves quantile order, so inverse-scaling first
	# keeps everything on log-return scale before we slice.
	preds_unscaled = p.scaler_target.inverse_transform(preds_s)

	r_q_lo = preds_unscaled.quantile(quantiles[0]).to_series().rename('r_q_lo')
	r_q_med = preds_unscaled.quantile(quantiles[1]).to_series().rename('r_q_med')
	r_q_hi = preds_unscaled.quantile(quantiles[2]).to_series().rename('r_q_hi')

	target_test = p.target_pd.iloc[p.train_end:]
	ref_test = p.ref_pd.iloc[p.train_end:]
	common = r_q_med.index.intersection(target_test.index)
	target_test = target_test.loc[common]
	ref_test = ref_test.loc[common]
	r_q_lo = r_q_lo.loc[common]
	r_q_med = r_q_med.loc[common]
	r_q_hi = r_q_hi.loc[common]

	close_test = ref_test * np.exp(target_test)
	pred_close = ref_test * np.exp(r_q_med)
	pred_close_lo = ref_test * np.exp(r_q_lo)
	pred_close_hi = ref_test * np.exp(r_q_hi)

	ppy = periods_per_year(interval)
	strat = strategy_returns(close_test, pred_close, ref_test)

	predictions = pd.DataFrame({
		'close': close_test,
		'pred': pred_close,
		'pred_lo': pred_close_lo,
		'pred_hi': pred_close_hi,
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
		extras={'rows_train': int(p.train_end)},
	)


def main() -> None:
	logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

	parser = argparse.ArgumentParser(
		description='Zero-shot Chronos-2 / TimesFM 2.5 on 1h log-returns.',
	)
	parser.add_argument('--config', type=Path, required=True)
	args = parser.parse_args()

	results_dir = EXPERIMENT_DIR / 'results' / args.config.stem.removesuffix('.config')
	results_dir.mkdir(parents=True, exist_ok=True)

	cfg = load_yaml(args.config)
	_aggregate, results, interval = train_and_evaluate(cfg)

	backend = str(cfg['backend'])
	hub_model_name = str(cfg['hub_model_name'])
	quantiles = [float(q) for q in cfg['model']['quantiles']]

	payload = write_aggregated_metrics_json(
		results,
		results_dir / 'metrics.json',
		experiment='06_pretrained',
		dataset=cfg['dataset'],
		interval=interval,
		extra_run_fields={
			'backend': backend,
			'hub_model_name': hub_model_name,
			'input_chunk_length': int(cfg['model']['input_chunk_length']),
			'num_samples': int(cfg['model']['num_samples']),
			'quantiles': quantiles,
		},
	)

	concat_predictions(results).to_parquet(results_dir / 'predictions.parquet')

	_plot(results, interval, backend, hub_model_name, quantiles, results_dir / 'plot.png')

	print(json.dumps(payload, indent=2))


def _plot(
	results: list[PerSymbolResult],
	interval: str,
	backend: str,
	hub_model_name: str,
	quantiles: list[float],
	out_path: Path,
) -> None:
	label = f'{backend} ({hub_model_name})'
	q_lo, _, q_hi = quantiles
	n = len(results)
	fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)
	for ax, r in zip(axes[:, 0], results, strict=True):
		preds = r.predictions
		ax.plot(preds.index, preds['close'].values, label='close', linewidth=1)
		ax.plot(
			preds.index, preds['pred'].values,
			label=f'{label} median', linewidth=1, alpha=0.7,
		)
		ax.fill_between(
			preds.index, preds['pred_lo'].values, preds['pred_hi'].values,
			alpha=0.2, label=f'q{q_lo:g}–q{q_hi:g} band',
		)
		ax.set_title(f'{r.symbol} {interval} — {label} (test set, zero-shot)')
		ax.set_ylabel('price')
		ax.legend()
	fig.tight_layout()
	fig.savefig(out_path, dpi=120)
	plt.close(fig)


if __name__ == '__main__':
	main()
