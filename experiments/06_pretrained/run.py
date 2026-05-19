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
from darts.models import Chronos2Model, TimesFM2p5Model
from darts.utils.likelihood_models.torch import QuantileRegression

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
# (e.g. configs/btc_4h_2024.config.yaml → results/btc_4h_2024/).
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
	# Both classes inherit from darts FoundationModel and share the same surface
	# we use here (input_chunk_length, output_chunk_length, hub_model_name,
	# likelihood, pl_trainer_kwargs). The only divergence is the class itself.
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


def train_and_evaluate(
	cfg: dict[str, Any],
	model_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
	# End-to-end pipeline shared by run.py and sweep.py: load data, build target,
	# fit a Scaler on the train slice, instantiate the requested foundation model
	# (chronos or timesfm), fit() it (no-op for these zero-shot models), then
	# walk-forward 1-step probabilistic forecasts on the test slice, reconstruct
	# prices on the median, and compute the standard metric suite.
	# `model_overrides` lets sweep.py poke individual config values (backend,
	# hub_model_name, input_chunk_length, num_samples) without rewriting cfg.
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
	# Zero-shot models don't use a validation slice: fold val into the training
	# segment so behavior matches the legacy single-split path.
	df = pd.concat([splits.train, splits.validation, splits.test])
	pre_test_rows = len(splits.train) + len(splits.validation)
	# Drop tz so darts' time indexing is unambiguous (matches 02_arima/04_lstm/05_transformer).
	df.index = df.index.tz_localize(None)
	logger.info('loaded %d rows from %s to %s', len(df), df.index[0], df.index[-1])

	target_pd, ref_pd = build_target(df)
	# build_target drops the leading row (.diff() yields NaN at index 0). Translate
	# the bar-level pre_test_rows index into a target-row index via the timestamp.
	train_end = target_pd.index.get_indexer([df.index[pre_test_rows]])[0]
	n = len(target_pd)
	logger.info('split: train=[0:%d] test=[%d:%d]', train_end, train_end, n)

	# Cast to float32 for the torch model: MPS does not support float64 tensors,
	# and float32 is plenty for log-returns at this magnitude. Price reconstruction
	# downstream uses the float64 ref / target_pd so MAE/MAPE accuracy is preserved.
	target_full_ts = TimeSeries.from_series(target_pd.astype('float32'))
	target_train = target_full_ts[:train_end]

	scaler_target = Scaler()
	scaler_target.fit(target_train)
	target_full_s = scaler_target.transform(target_full_ts)

	logger.info('instantiating %s (%s)', backend, hub_model_name)
	model = build_model(backend, hub_model_name, merged, quantiles)
	logger.info(
		'fit %s on %d training rows — zero-shot, no weight updates. '
		'First run downloads HuggingFace weights (chronos-2-small ≈120 MB, '
		'chronos-2 ≈300 MB, timesfm-2.5 ≈800 MB), cached at ~/.cache/huggingface/hub/.',
		backend, len(target_train),
	)
	# fit() is required by the darts API but performs no weight updates for
	# foundation models. It does, however, trigger the HuggingFace weight
	# download on first invocation (via _create_model → hf_connector.load_model).
	model.fit(scaler_target.transform(target_train))
	logger.info('model ready')

	test_start = target_full_s.time_index[train_end]
	test_bars = len(target_full_s) - train_end
	logger.info(
		'walk-forward 1-step from %s over %d test bars (num_samples=%d)',
		test_start, test_bars, num_samples,
	)
	# verbose=True surfaces darts' tqdm bar across the test window — without
	# it the model sits silent for the full sweep, which is several minutes.
	preds_s = model.historical_forecasts(
		series=target_full_s,
		start=test_start,
		forecast_horizon=1,
		retrain=False,
		last_points_only=True,
		num_samples=num_samples,
		verbose=True,
	)
	logger.info('walk-forward complete; extracting quantiles and reconstructing prices')
	# Stochastic series — inverse-scale on the full distribution, then extract
	# quantiles. (Linear scaling preserves quantile order, so this and
	# scale->quantile->inverse-scale would agree, but inverse-scaling first
	# keeps everything on log-return scale before we slice.)
	preds_unscaled = scaler_target.inverse_transform(preds_s)

	r_q_lo = preds_unscaled.quantile(quantiles[0]).to_series().rename('r_q_lo')
	r_q_med = preds_unscaled.quantile(quantiles[1]).to_series().rename('r_q_med')
	r_q_hi = preds_unscaled.quantile(quantiles[2]).to_series().rename('r_q_hi')

	target_test = target_pd.iloc[train_end:]
	ref_test = ref_pd.iloc[train_end:]
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

	ppy = periods_per_year(splits.interval)
	strat = strategy_returns(close_test, pred_close, ref_test)

	metrics: dict[str, Any] = {
		'experiment': '06_pretrained',
		'dataset': cfg['dataset'],
		'backend': backend,
		'hub_model_name': hub_model_name,
		'symbol': splits.symbol_test,
		'interval': splits.interval,
		'input_chunk_length': int(merged['input_chunk_length']),
		'num_samples': num_samples,
		'quantiles': quantiles,
		'rows_total': int(n),
		'rows_train': int(train_end),
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
		'pred_lo': pred_close_lo,
		'pred_hi': pred_close_hi,
		'ref': ref_test,
		'strategy_return': strat,
	})
	return metrics, predictions


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
	metrics, predictions = train_and_evaluate(cfg)

	(results_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
	predictions.to_parquet(results_dir / 'predictions.parquet')

	label = f'{metrics["backend"]} ({metrics["hub_model_name"]})'
	q_lo, _, q_hi = metrics['quantiles']

	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(predictions.index, predictions['close'].values, label='close', linewidth=1)
	ax.plot(
		predictions.index,
		predictions['pred'].values,
		label=f'{label} median',
		linewidth=1,
		alpha=0.7,
	)
	ax.fill_between(
		predictions.index,
		predictions['pred_lo'].values,
		predictions['pred_hi'].values,
		alpha=0.2,
		label=f'q{q_lo:g}–q{q_hi:g} band',
	)
	ax.set_title(
		f'{metrics["symbol"]} {metrics["interval"]} — {label} (test set, zero-shot)',
	)
	ax.set_ylabel('price')
	ax.legend()
	fig.tight_layout()
	fig.savefig(results_dir / 'plot.png', dpi=120)
	plt.close(fig)

	print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
	main()
