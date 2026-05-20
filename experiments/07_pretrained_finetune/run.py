import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

# Quiet noisy third-party imports before darts and PyTorch Lightning load:
# - warnings.filterwarnings catches PyTorch Lightning's _pytree deprecation,
#   torch's Apple Metal Performance Shaders pin_memory notice, and the
#   "StatsForecast could not be imported" print.
# - darts logger drops the "Train dataset contains N samples." / "Time series
#   values are 32-bits; casting model to float32." chatter.
# - httpx logger drops the per-request HTTP HEAD lines emitted when HuggingFace
#   resolves chronos / timesfm config + safetensors URLs.
warnings.filterwarnings('ignore')
logging.getLogger('darts').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from darts import TimeSeries  # noqa: E402
from darts.dataprocessing.transformers import Scaler  # noqa: E402
from darts.models import Chronos2Model, TimesFM2p5Model  # noqa: E402
from darts.utils.likelihood_models.torch import QuantileRegression  # noqa: E402
from pytorch_lightning.callbacks import Callback, EarlyStopping  # noqa: E402

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
# darts writes checkpoints under {CHECKPOINT_DIR}/darts_logs/{model_name}/.
CHECKPOINT_DIR = EXPERIMENT_DIR / 'results' / 'darts_checkpoints'

VALID_BACKENDS: tuple[str, ...] = ('chronos', 'timesfm')

logger = logging.getLogger(__name__)


class LossHistory(Callback):
	# Captures per-epoch train_loss and val_loss from the PyTorch Lightning trainer
	# so we can plot the fine-tuning curve afterwards. darts logs `{train,val}_loss`
	# to trainer.callback_metrics from PLForecastingModule._train_val_step. The
	# training-step log uses PyTorch Lightning's defaults `on_step=True,
	# on_epoch=False`, so `callback_metrics['train_loss']` at epoch end is just
	# the LAST batch's loss — too noisy to plot. We instead accumulate per-batch
	# losses ourselves and average them per epoch. Validation uses `on_step=False,
	# on_epoch=True` by default so `callback_metrics['val_loss']` already is the
	# epoch mean.
	def __init__(self) -> None:
		self.train_loss: list[float] = []
		self.val_loss: list[float] = []
		self._train_batch_losses: list[float] = []

	def on_fit_start(self, trainer: Any, pl_module: Any) -> None:  # noqa: ARG002
		# One-shot diagnostic: count trainable parameters after _setup_finetuning
		# has set requires_grad on each tensor. If the unfreeze pattern misses,
		# this prints `trainable=0` and the operator knows to fix the pattern
		# rather than wait through a no-op training run.
		trainable = [(n, p.numel()) for n, p in pl_module.named_parameters() if p.requires_grad]
		n_trainable = sum(numel for _, numel in trainable)
		n_total = sum(p.numel() for p in pl_module.parameters())
		preview = ', '.join(n for n, _ in trainable[:5])
		if len(trainable) > 5:
			preview += f', ... (+{len(trainable) - 5} more)'
		logger.info(
			'trainable parameters: %d / %d (%.3f%%); names: [%s]',
			n_trainable, n_total,
			100.0 * n_trainable / max(n_total, 1),
			preview if preview else '<none — unfreeze pattern matched nothing>',
		)

	def on_train_batch_end(
		self, trainer: Any, pl_module: Any,  # noqa: ARG002
		outputs: Any, batch: Any, batch_idx: int,  # noqa: ARG002
	) -> None:
		loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
		if loss is not None:
			self._train_batch_losses.append(float(loss.detach()))

	def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:  # noqa: ARG002
		if len(self._train_batch_losses) > 0:
			mean = sum(self._train_batch_losses) / len(self._train_batch_losses)
			self.train_loss.append(mean)
			self._train_batch_losses.clear()

	def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:  # noqa: ARG002
		# PyTorch Lightning runs one sanity-check validation before training;
		# skip it so the val_loss list aligns with epoch index.
		if getattr(trainer, 'sanity_checking', False):
			return
		metric = trainer.callback_metrics.get('val_loss')
		if metric is not None:
			self.val_loss.append(float(metric))


def _pandas_freq(interval: str) -> str:
	# Translate a binance interval string ('4h', '15m', '1d') to the pandas
	# frequency string accepted by DataFrame.asfreq(). binance uses 'm' for
	# minutes; pandas needs 'min' because 'M' is month-end.
	interval = interval.strip().lower()
	if interval.endswith('m'):
		return interval[:-1] + 'min'
	return interval


def build_target(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
	# target r_T = log(close_T) - log(close_{T-1}); ref_T = close_{T-1} for
	# downstream price reconstruction. Univariate by design — neither backend
	# is given covariates here.
	close = df['close'].astype('float64')
	target = np.log(close).diff().rename('r_target')
	ref = close.shift(1).rename('ref_close')
	joined = pd.concat([target, ref], axis=1).dropna()
	return joined['r_target'], joined['ref_close']


def build_model(
	backend: str,
	hub_model_name: str,
	model_cfg: dict[str, Any],
	finetune_cfg: dict[str, Any],
	train_cfg: dict[str, Any],
	quantiles: list[float],
	extra_callbacks: list[Any] | None = None,
	work_dir: str | None = None,
	model_name: str | None = None,
) -> Chronos2Model | TimesFM2p5Model:
	early_stop = EarlyStopping(
		monitor='val_loss',
		patience=int(train_cfg['early_stopping_patience']),
		mode='min',
	)
	callbacks: list[Any] = [early_stop]
	if extra_callbacks is not None:
		callbacks.extend(extra_callbacks)
	pl_trainer_kwargs: dict[str, Any] = {
		'callbacks': callbacks,
		'accelerator': 'auto',
		'enable_progress_bar': True,
		'enable_model_summary': False,
		'logger': False,
		'num_sanity_val_steps': 0,
	}
	likelihood = QuantileRegression(quantiles=quantiles)
	common_kwargs: dict[str, Any] = {
		'input_chunk_length': int(model_cfg['input_chunk_length']),
		'output_chunk_length': int(model_cfg['output_chunk_length']),
		'hub_model_name': hub_model_name,
		'likelihood': likelihood,
		'pl_trainer_kwargs': pl_trainer_kwargs,
		'enable_finetuning': finetune_cfg['enable_finetuning'],
		'n_epochs': int(model_cfg['n_epochs']),
		'batch_size': int(model_cfg['batch_size']),
		'optimizer_kwargs': {'lr': float(model_cfg['learning_rate'])},
		'random_state': int(model_cfg['random_state']),
		'save_checkpoints': True,
		'force_reset': True,
	}
	if work_dir is not None:
		common_kwargs['work_dir'] = work_dir
	if model_name is not None:
		common_kwargs['model_name'] = model_name
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
	train_end: int
	val_end: int
	scaler_target: Scaler
	target_train_s: TimeSeries
	target_val_s: TimeSeries
	target_full_s: TimeSeries


def _prepare_symbol(frame: PerSymbolFrame, interval: str) -> _PreparedSymbol:
	if frame.val_start_ts is None:
		raise ValueError(
			f'{frame.market} {frame.symbol}: validation selector required for fine-tuning',
		)
	# Reindex onto a regular grid for the dataset's interval. Forward-fills any
	# gaps (e.g. the 2020-02-19 Binance outage on BTCUSDT 4h) so the log-return
	# at the gap boundary is exactly 0 — no fake price movement — and darts
	# can infer the frequency on the resulting TimeSeries downstream.
	df = frame.df.asfreq(_pandas_freq(interval), method='ffill')

	target_pd, ref_pd = build_target(df)
	# build_target drops the leading row (.diff() yields NaN at index 0).
	# Translate the bar-level boundary timestamps to target-row indices.
	train_end = target_pd.index.get_indexer([frame.val_start_ts])[0]
	val_end = target_pd.index.get_indexer([frame.test_start_ts])[0]

	target_full_ts = TimeSeries.from_series(target_pd.astype('float32'))
	target_train = target_full_ts[:train_end]
	target_val = target_full_ts[train_end:val_end]

	scaler_target = Scaler()
	scaler_target.fit(target_train)
	target_train_s = scaler_target.transform(target_train)
	target_val_s = scaler_target.transform(target_val)
	target_full_s = scaler_target.transform(target_full_ts)

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
	)


@dataclass
class TrainOutput:
	aggregate: dict[str, float]
	results: list[PerSymbolResult]
	interval: str
	backend: str
	hub_model_name: str
	quantiles: list[float]
	input_chunk_length: int
	num_samples: int
	enable_finetuning: object
	n_epochs: int
	epochs_trained: int
	restored_best_checkpoint: bool
	best_val_loss: float | None
	batch_size: int
	learning_rate: float
	early_stopping_patience: int
	train_loss_history: list[float]
	val_loss_history: list[float]


def train_and_evaluate(
	cfg: dict[str, Any],
	model_overrides: dict[str, Any] | None = None,
) -> TrainOutput:
	# End-to-end pipeline shared by run.py and sweep.py: load data, build per-symbol
	# target series, fine-tune one foundation model on the list of training series
	# (multi-series fit), restore best-val_loss checkpoint, walk-forward 1-step
	# probabilistic forecasts per test symbol, aggregate metrics.
	merged: dict[str, Any] = {
		'backend': cfg['backend'],
		'hub_model_name': cfg['hub_model_name'],
		**dict(cfg['model']),
	}
	finetune_cfg: dict[str, Any] = dict(cfg['finetune'])
	train_cfg: dict[str, Any] = dict(cfg['training'])
	if model_overrides is not None:
		for key in ('enable_finetuning',):
			if key in model_overrides:
				finetune_cfg[key] = model_overrides.pop(key)
		for key in ('early_stopping_patience',):
			if key in model_overrides:
				train_cfg[key] = model_overrides.pop(key)
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

	prepared = [_prepare_symbol(f, splits.interval) for f in frames]

	target_train_list = [p.target_train_s for p in prepared]
	target_val_list = [p.target_val_s for p in prepared]

	logger.info('instantiating %s (%s)', backend, hub_model_name)
	loss_history = LossHistory()
	# Naming by dataset + backend keeps consecutive runs on the same configuration
	# overwriting each other (via force_reset=True) while keeping parallel
	# datasets / backends in separate directories.
	model_name = f'{cfg["dataset"]}_{backend}'
	CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
	model = build_model(
		backend, hub_model_name, merged, finetune_cfg, train_cfg, quantiles,
		extra_callbacks=[loss_history],
		work_dir=str(CHECKPOINT_DIR),
		model_name=model_name,
	)
	logger.info(
		'fine-tuning %s on %d series, %d train rows total, up to %d epochs '
		'(patience=%d on val_loss, enable_finetuning=%r). '
		'First run downloads HuggingFace weights, cached at ~/.cache/huggingface/hub/.',
		backend, len(target_train_list),
		sum(len(s) for s in target_train_list),
		int(merged['n_epochs']),
		int(train_cfg['early_stopping_patience']),
		finetune_cfg['enable_finetuning'],
	)
	model.fit(series=target_train_list, val_series=target_val_list)
	epochs_trained = len(loss_history.val_loss)
	logger.info('fine-tuning complete (epochs_trained=%d)', epochs_trained)

	# Restore the best-val_loss checkpoint into the in-memory model.
	restored_best_checkpoint = False
	best_val_loss: float | None = None
	if len(loss_history.val_loss) > 0:
		try:
			model.load_weights_from_checkpoint(
				model_name=model_name,
				work_dir=str(CHECKPOINT_DIR),
				best=True,
			)
			best_epoch = int(np.argmin(loss_history.val_loss)) + 1
			best_val_loss = float(loss_history.val_loss[best_epoch - 1])
			restored_best_checkpoint = True
			logger.info(
				'restored best-val_loss checkpoint (epoch %d, val_loss=%.6f) for inference',
				best_epoch, best_val_loss,
			)
		except (FileNotFoundError, RuntimeError) as exc:
			logger.warning(
				'could not restore best-val_loss checkpoint (%s); '
				'inference will use last-epoch weights',
				exc,
			)

	results = [
		_evaluate_symbol(p, model, splits.interval, quantiles, num_samples)
		for p in prepared
	]

	return TrainOutput(
		aggregate=aggregate_metrics(results),
		results=results,
		interval=splits.interval,
		backend=backend,
		hub_model_name=hub_model_name,
		quantiles=quantiles,
		input_chunk_length=int(merged['input_chunk_length']),
		num_samples=num_samples,
		enable_finetuning=finetune_cfg['enable_finetuning'],
		n_epochs=int(merged['n_epochs']),
		epochs_trained=epochs_trained,
		restored_best_checkpoint=restored_best_checkpoint,
		best_val_loss=best_val_loss,
		batch_size=int(merged['batch_size']),
		learning_rate=float(merged['learning_rate']),
		early_stopping_patience=int(train_cfg['early_stopping_patience']),
		train_loss_history=loss_history.train_loss,
		val_loss_history=loss_history.val_loss,
	)


def _evaluate_symbol(
	p: _PreparedSymbol,
	model: Chronos2Model | TimesFM2p5Model,
	interval: str,
	quantiles: list[float],
	num_samples: int,
) -> PerSymbolResult:
	frame = p.frame
	test_start = p.target_full_s.time_index[p.val_end]
	logger.info(
		'walk-forward 1-step from %s over %d test bars (%s %s, num_samples=%d)',
		test_start, len(p.target_full_s) - p.val_end,
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
	preds_unscaled = p.scaler_target.inverse_transform(preds_s)

	r_q_lo = preds_unscaled.quantile(quantiles[0]).to_series().rename('r_q_lo')
	r_q_med = preds_unscaled.quantile(quantiles[1]).to_series().rename('r_q_med')
	r_q_hi = preds_unscaled.quantile(quantiles[2]).to_series().rename('r_q_hi')

	target_test = p.target_pd.iloc[p.val_end:]
	ref_test = p.ref_pd.iloc[p.val_end:]
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
		extras={
			'rows_train': int(p.train_end),
			'rows_val': int(p.val_end - p.train_end),
		},
	)


def main() -> None:
	logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
	logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
	logging.getLogger('lightning.pytorch').setLevel(logging.WARNING)

	parser = argparse.ArgumentParser(
		description='Fine-tuned Chronos-2 / TimesFM 2.5 on BTC log-returns.',
	)
	parser.add_argument('--config', type=Path, required=True)
	args = parser.parse_args()

	results_dir = EXPERIMENT_DIR / 'results' / args.config.stem.removesuffix('.config')
	results_dir.mkdir(parents=True, exist_ok=True)

	cfg = load_yaml(args.config)
	out = train_and_evaluate(cfg)

	payload = write_aggregated_metrics_json(
		out.results,
		results_dir / 'metrics.json',
		experiment='07_pretrained_finetune',
		dataset=cfg['dataset'],
		interval=out.interval,
		extra_run_fields={
			'backend': out.backend,
			'hub_model_name': out.hub_model_name,
			'input_chunk_length': out.input_chunk_length,
			'num_samples': out.num_samples,
			'quantiles': out.quantiles,
			'enable_finetuning': out.enable_finetuning,
			'n_epochs': out.n_epochs,
			'epochs_trained': out.epochs_trained,
			'restored_best_checkpoint': out.restored_best_checkpoint,
			'best_val_loss': out.best_val_loss,
			'batch_size': out.batch_size,
			'learning_rate': out.learning_rate,
			'early_stopping_patience': out.early_stopping_patience,
			'train_loss_history': out.train_loss_history,
			'val_loss_history': out.val_loss_history,
		},
	)

	concat_predictions(out.results).to_parquet(results_dir / 'predictions.parquet')

	_plot_test(out, results_dir / 'plot.png')
	_plot_training_curves(out, results_dir / 'training_curves.png')

	print(json.dumps(payload, indent=2))


def _plot_test(out: TrainOutput, path: Path) -> None:
	label = f'{out.backend} ({out.hub_model_name})'
	q_lo, _, q_hi = out.quantiles
	weights_source = 'best-val_loss' if out.restored_best_checkpoint else 'last-epoch'
	n = len(out.results)
	fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)
	for ax, r in zip(axes[:, 0], out.results):
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
		ax.set_title(
			f'{r.symbol} {out.interval} — {label} '
			f'(test set, fine-tuned, epochs_trained={out.epochs_trained}, '
			f'weights={weights_source})',
		)
		ax.set_ylabel('price')
		ax.legend()
	fig.tight_layout()
	fig.savefig(path, dpi=120)
	plt.close(fig)


def _plot_training_curves(out: TrainOutput, path: Path) -> None:
	train_loss = out.train_loss_history
	val_loss = out.val_loss_history
	if len(train_loss) == 0 and len(val_loss) == 0:
		return
	label = f'{out.backend} ({out.hub_model_name})'
	fig, ax = plt.subplots(figsize=(8, 4))
	if len(train_loss) > 0:
		ax.plot(
			range(1, len(train_loss) + 1), train_loss,
			label='train_loss', marker='o', linewidth=1,
		)
	if len(val_loss) > 0:
		ax.plot(
			range(1, len(val_loss) + 1), val_loss,
			label='val_loss', marker='s', linewidth=1,
		)
		best_epoch = int(np.argmin(val_loss)) + 1
		ax.axvline(
			best_epoch, color='gray', linestyle='--', alpha=0.5,
			label=f'best val_loss @ epoch {best_epoch}',
		)
	ax.set_xlabel('epoch')
	ax.set_ylabel('loss')
	ax.set_title(
		f'{out.interval} — {label} fine-tuning curves '
		f'(epochs_trained={out.epochs_trained})',
	)
	ax.legend()
	ax.grid(alpha=0.3)
	fig.tight_layout()
	fig.savefig(path, dpi=120)
	plt.close(fig)


if __name__ == '__main__':
	main()
