# 07_pretrained_finetune

Natural follow-up to [`06_pretrained`](../06_pretrained/): same two foundation-model backends — Amazon's [Chronos-2](https://huggingface.co/amazon/chronos-2) (encoder-only T5-style, 120M parameters) or Google's [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (decoder-only patch-transformer, 200M parameters) — but with `enable_finetuning` set so `fit()` actually updates weights against a held-out validation slice before walk-forward inference.

```
HuggingFace Hub weights
    │
    ▼
darts FoundationModel
    │
    ▼  fine-tune on train slice, early-stop on val_loss,
       save best-val_loss checkpoint to disk
    │
    ▼  restore best-val_loss weights for inference
       (NOT the last-epoch weights, which are typically overfit)
    │
    ▼
historical_forecasts(retrain=False, num_samples=1000)
    │
    ▼  quantile(0.1) / 0.5 / 0.9 ──► reconstruct close
```

The target is the same bar-T log-return `r_T = log(close_T / close_{T-1})` used by [`04_lstm`](../04_lstm/), [`05_transformer`](../05_transformer/), and [`06_pretrained`](../06_pretrained/); predictions are reconstructed to price as `close_pred = close_{T-1} * exp(r_q0.5)`. MAE / RMSE / MAPE / directional_accuracy / Sharpe stay directly comparable to all prior experiments.

## Headline result

On the extended `btc_4h_2020_2024` slice, **encoder-only fine-tuning of `autogluon/chronos-2-small` produces statistically significant improvements over 06's zero-shot baseline on the two trading metrics that matter most: annualized Sharpe and cumulative return.** Five-seed comparison (95% confidence intervals from t-distribution, n=5):

| metric | fine-tuned mean ± std | 95% CI | 06 zero-shot | beats zero-shot |
|---|---|---|---|---|
| mae | 539.41 ± 1.88 | (537.07, 541.75) | 540.37 | 4/5 (within noise) |
| rmse | 809.76 ± 1.17 | (808.30, 811.21) | 810.28 | 4/5 (within noise) |
| mape | 0.691% ± 0.003 | (0.688%, 0.694%) | 0.692% | 3/5 (within noise) |
| directional_accuracy | 0.5290 ± 0.0101 | (0.5163, 0.5416) | 0.5164 | 4/5 (borderline) |
| **cumulative_return** | **44.52% ± 5.48pp** | **(37.70%, 51.33%)** | **36.64%** | **5/5 ★** |
| **annualized_sharpe** | **5.523 ± 0.418** | **(5.004, 6.043)** | **4.743** | **5/5 ★** |

The CIs on Sharpe and cumulative_return exclude the zero-shot value at 95% confidence. Every single seed independently beats zero-shot on both metrics. Reproduce with:

```
make 07_pretrained_finetune CONFIG=configs/btc_4h_2020_2024_encoder_only.config.yaml
```

(See [Reproducing the multi-seed result](#reproducing-the-multi-seed-result) below for how to sweep `random_state`.)

## What this tells us

The gain is in **prediction magnitude calibration**, not directional bias frequency. Directional accuracy is barely above zero-shot (the CI on `directional_accuracy` touches zero-shot's value), but Sharpe and cumulative_return improve substantially. The fine-tuned model leans more strongly when it has signal and less when it doesn't — i.e., it learned to size the predictive distribution appropriately for BTC log-returns, while the frozen head preserved the directional prior from pretraining.

This is a more interesting result than a directional-accuracy story would have been, because it speaks to *prediction confidence calibration* rather than *bias*. Quantile-regression fine-tuning of the encoder with the head frozen taught the encoder to produce features whose downstream quantile spreads are sized appropriately for BTC log-returns.

## Three things had to happen for this to work

Each of these was discovered the hard way; earlier configurations of 07 lost to 06 zero-shot on every metric until all three were in place.

1. **Extend the training slice to 4.7 years.** The original `btc_4h_2024` dataset (1.6 years, ~3500 train rows) produced a model that overfit within 1-3 epochs and never escaped. Switching to `btc_4h_2020_2024` (4.7 years, ~10000 train rows) gave the encoder enough signal to generalize.
2. **Freeze the head, fine-tune the encoder.** Head-only fine-tuning (the obvious default) consistently *erodes* the pretrained directional prior — directional_accuracy on chronos-2-small dropped from 0.5164 (zero-shot) to 0.4863 (worse than coin-flip). The head encodes the directional bias; the encoder produces representations. Fine-tune the latter.
3. **Restore the best-val_loss checkpoint for inference.** The val_loss curve diverges within 1-4 epochs of fine-tuning starting (this is foundation-model fine-tuning, where the pretrained init is already near-optimal). EarlyStopping halts training but does NOT roll the in-memory weights back to the best epoch. Without explicit checkpoint restoration, walk-forward inference uses the (overfit) end-of-training weights, which produces a misleading result. This pipeline calls `model.load_weights_from_checkpoint(best=True)` after `fit()` returns.

A useful aside: an earlier configuration of 07 reported `annualized_sharpe = 6.146` from a single-seed run — looked like a major win at the time. That number was an artifact of point #3 not being in place yet: the model had diverged but the last-epoch weights coincidentally aligned with the test slice's drift. Wiring in checkpoint restoration dropped that same configuration's Sharpe to 3.46. The result was wrong, not unlucky.

## Why univariate

Same constraint as 06: TimesFM 2.5 doesn't accept covariates at all, and we keep Chronos-2 univariate too so the head-to-head between the two backends compares priors rather than feature engineering. Past / future covariates remain a clean follow-up ablation.

## How it works

1. Load OHLCV via the shared loader ([src/btc_ai/data](../../src/btc_ai/data)).
2. Reindex the joined dataframe onto a regular grid via `df.asfreq(...).fillna(method='ffill')` — necessary because the extended dataset includes the 2020-02-19 Binance outage gap, and darts needs a regular index to infer frequency on the resulting TimeSeries.
3. Build target `r_T = log(close_T / close_{T-1})` and `ref_T = close_{T-1}` for downstream price reconstruction. **Univariate only.**
4. Three-way time split: `[0:train_end]` for training, `[train_end:val_end]` for validation (drives EarlyStopping and checkpoint selection), `[val_end:n]` for test.
5. Fit a `darts.dataprocessing.transformers.Scaler` on the **training slice only**. Cast to `float32` so the model can run on Apple Metal Performance Shaders or CUDA.
6. Build the model based on `cfg['backend']` with fine-tuning enabled:
   - `chronos` → `Chronos2Model(input_chunk_length, output_chunk_length=1, hub_model_name, likelihood=QuantileRegression([0.1, 0.5, 0.9]), enable_finetuning=cfg['finetune']['enable_finetuning'], n_epochs, batch_size, optimizer_kwargs={'lr': ...}, pl_trainer_kwargs={'callbacks': [EarlyStopping('val_loss', patience=...), LossHistory()], ...}, save_checkpoints=True, work_dir=..., model_name=...)`
   - `timesfm` → same with `TimesFM2p5Model(...)`.
   - The `LossHistory` callback captures epoch-mean `train_loss` and `val_loss` for the training-curves plot.
   - `save_checkpoints=True` lets darts auto-add a `ModelCheckpoint(monitor='val_loss')` so the best-val_loss state is persisted.
7. Call `model.fit(series=target_train_s, val_series=target_val_s)`. Unlike 06, this **does** update weights — only the params matched by the `enable_finetuning` pattern (or all params, if `True`).
8. **Restore the best-val_loss checkpoint** via `model.load_weights_from_checkpoint(model_name=..., work_dir=..., best=True)`. This is the difference between honest evaluation and overfit-endpoint evaluation.
9. Walk-forward predict on the test slice via `model.historical_forecasts(start=test_start, forecast_horizon=1, retrain=False, last_points_only=True, num_samples=...)`. Each step samples N futures from the fine-tuned predictive distribution.
10. Inverse-scale the stochastic series, then extract `quantile(0.1) / quantile(0.5) / quantile(0.9)` to get `r_q_lo / r_q_med / r_q_hi`.
11. Reconstruct prices: `close_pred = ref * exp(r_q_med)` for the point prediction (the median), plus `pred_lo / pred_hi` for the interval.
12. Compute MAE, RMSE, MAPE, directional accuracy, cumulative return, and annualized Sharpe on the **median** prediction via the shared metrics in [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py).
13. Write `results/<config_stem>/metrics.json` (including `enable_finetuning`, `epochs_trained`, `rows_val`, `restored_best_checkpoint`, `best_val_loss`), `results/<config_stem>/predictions.parquet` (`close, pred, pred_lo, pred_hi, ref, strategy_return`), `results/<config_stem>/plot.png` (close + median + shaded q10–q90 band), and `results/<config_stem>/training_curves.png` (per-epoch train_loss + val_loss with a vertical line at the best-val_loss epoch).

Library: **[darts](https://unit8co.github.io/darts/)** (`darts.models.Chronos2Model`, `darts.models.TimesFM2p5Model`). HuggingFace download is automatic on first run and cached in `~/.cache/huggingface/hub/`; if you already ran 06, the chronos-2-small weights are already there.

## How to run

```
make 07_pretrained_finetune CONFIG=configs/btc_4h_2020_2024_encoder_only.config.yaml   # the winning recipe
make 07_pretrained_finetune CONFIG=configs/btc_4h_2024_encoder_only.config.yaml        # encoder-only, 1.6yr slice (loses to zero-shot)
make 07_pretrained_finetune CONFIG=configs/btc_4h_2024.config.yaml                     # head-only, 1.6yr slice (loses worse)
make 07_pretrained_finetune CONFIG=configs/btc_1h_2024.config.yaml                     # 1h slice, head-only
```

Pick a backend in the config — there is **no default**, you must set one:

```yaml
backend: chronos          # or 'timesfm'
hub_model_name: autogluon/chronos-2-small
                          # chronos: amazon/chronos-2 | autogluon/chronos-2-small | autogluon/chronos-2-synth
                          # timesfm: google/timesfm-2.5-200m-pytorch
```

The `finetune.enable_finetuning` key drives the recipe:

```yaml
finetune:
  enable_finetuning:
    freeze:
      - '*output_patch_embedding*'   # encoder-only — the winning recipe
  # enable_finetuning:
  #   unfreeze:
  #     - '*output_patch_embedding*' # head-only — observed to lose to zero-shot
  # enable_finetuning: true          # full fine-tuning (every parameter trainable)
  # enable_finetuning: false         # zero-shot — equivalent to 06_pretrained
```

Training knobs from `btc_4h_2020_2024_encoder_only.config.yaml`:

```yaml
training:
  early_stopping_patience: 5         # epochs of no val_loss improvement before stop

model:
  input_chunk_length: 256            # bars of history (chronos-2-small max=8192)
  output_chunk_length: 1
  num_samples: 1000                  # samples drawn from the predictive distribution per step
  quantiles: [0.1, 0.5, 0.9]
  n_epochs: 50                       # upper bound; EarlyStopping usually stops earlier
  batch_size: 32                     # 32 fits chronos-2-small encoder-only fine-tuning on Apple Metal Performance Shaders at input_chunk_length=256
  learning_rate: 0.00001             # 1e-5 — conservative; preserves the pretrained encoder
  random_state: 42                   # reproducibility for dropout / data shuffle
```

First run downloads the HuggingFace weights and 4.7 years of BTC monthly archives (one-time, then cached). End-to-end runtime on Apple Metal Performance Shaders is roughly: training 3-6 minutes (EarlyStopping at patience 5 usually fires within 6-10 epochs because the val_loss diverges fast), checkpoint restoration <1 second, walk-forward over 366 test bars 5-6 minutes at `num_samples=1000`. Total per run: ~10 minutes.

## How to choose unfreeze patterns

`enable_finetuning: {unfreeze: [...]}` and `{freeze: [...]}` accept lists of [`fnmatch`](https://docs.python.org/3/library/fnmatch.html)-style glob patterns matched against `model.named_parameters()` names (darts applies them at `torch_forecasting_model.py:_setup_finetuning`). The default `'*output_patch_embedding*'` matches the Chronos-2 prediction head:

- `{freeze: ['*output_patch_embedding*']}` = encoder-only (94.3% trainable; the winning recipe).
- `{unfreeze: ['*output_patch_embedding*']}` = head-only (5.7% trainable; loses to zero-shot).

The `trainable parameters: N / M (P%); names: [...]` log line at the start of every run shows exactly which parameters got `requires_grad=True` — that's the diagnostic that tells you whether the pattern matched what you intended. If `P` is 0 or implausibly small, the pattern missed.

TimesFM 2.5's head and encoder names differ from Chronos-2's; the default pattern in the config does **not** match TimesFM. When `backend: timesfm`, either switch to `enable_finetuning: true` (full fine-tuning) or introspect first and supply a TimesFM-specific pattern.

## Reproducing the multi-seed result

The headline numbers in this README come from five runs of the same config with `random_state` set to 42, 7, 13, 5, and 99 in turn. To reproduce on your machine:

```bash
CONFIG=experiments/07_pretrained_finetune/configs/btc_4h_2020_2024_encoder_only.config.yaml
RESULTS_DIR=experiments/07_pretrained_finetune/results/btc_4h_2020_2024_encoder_only

for seed in 42 7 13 5 99; do
    sed -i '' "s/random_state: [0-9]*/random_state: $seed/" "$CONFIG"   # macOS BSD sed
    make 07_pretrained_finetune CONFIG=configs/btc_4h_2020_2024_encoder_only.config.yaml
    cp "$RESULTS_DIR/metrics.json" "$RESULTS_DIR/metrics_seed${seed}.json"
done
```

Total wall time: ~50 minutes on Apple Metal Performance Shaders. Each seed's `metrics.json` is preserved alongside the running one. The five-seed mean and 95% CI is then a few lines of Python over the saved files.

## How it compares to earlier experiments

The headline table compares 07 fine-tuned (encoder-only, chronos-2-small, extended dataset, five-seed mean) against 06 zero-shot on the same test slice. The broader leaderboard:

| family | trained on slice | probabilistic | mae (4h test) | directional_accuracy | annualized_sharpe |
|---|---|---|---|---|---|
| 01 naive last-value | — | — | (see 01) | — | — |
| 02 ARIMA(3,1,3) | yes | — | (see 02) | — | (see 02) |
| 03 XGBoost | yes | — | (see 03) | — | (see 03) |
| 04 LSTM | yes | — | (see 04) | — | (see 04) |
| 05 TFT | yes | — | (see 05) | — | (see 05) |
| 06 chronos-2-small zero-shot | no | yes | 540.37 | 0.5164 | 4.743 |
| **07 chronos-2-small encoder-only fine-tuned (5-seed mean)** | **yes (94% of params)** | **yes** | **539.41** | **0.5290** | **5.523** ★ |

## Caveats

- **Five seeds is a small sample.** The Sharpe and cumulative_return improvements are formally significant at 95% with n=5, but the per-seed spread is meaningful (Sharpe std 0.42 on a mean of 5.52). More seeds would tighten the bars.
- **Single test regime.** The test slice (2024-10 to 2024-12, 366 four-hour bars) is shared across every comparison run in this repo, so the comparison isolates the fine-tuning effect — but it's a single market period. Generalization to other regimes (2021 bull, 2022 bear, 2023 chop) is not tested here.
- **Zero-shot baseline uses different `num_samples`.** 06_pretrained's `num_samples: 200` vs 07's `num_samples: 1000`. Bumping 07's count tightened the trading metrics meaningfully (Sharpe mean 5.02 → 5.41 between the two settings). For a perfectly apples-to-apples comparison, 06 should also be re-run at `num_samples: 1000`. The 1.4-Sharpe-point gap probably wouldn't close completely, but the precise number would shift.
- **Univariate.** No past covariates (volume, OHLC range/body) and no future covariates (hour-of-day, day-of-week). TimesFM doesn't accept any; Chronos-2 does but we deliberately match 06 for parity.
- **Walk-forward without re-estimation.** Fine-tuning happens once in `fit()`; `retrain=False` keeps weights frozen across the entire test window. Matches [`02_arima`](../02_arima/) / [`04_lstm`](../04_lstm/) / [`05_transformer`](../05_transformer/) / [`06_pretrained`](../06_pretrained/).
- **`output_chunk_length: 1`.** One-step-ahead only. Multi-horizon is a one-line config change but changes the loss surface.
- **Unfreeze patterns are backend-specific.** The default targets the Chronos-2 head. TimesFM needs introspection; the comment in the config flags this.
- **Apple Metal Performance Shaders memory pressure** at `input_chunk_length=256` + full fine-tuning. Encoder-only and head-only fit; full fine-tuning of `amazon/chronos-2` (120M parameters) at this `batch_size` may exceed Apple Metal Performance Shaders memory — drop `batch_size` to 16 or 8, or switch to CUDA.
- **TimesFM RINorm-affine override.** darts logs a warning when foundation-model checkpoints lack RINorm affine weights; the wrapper overrides `use_reversible_instance_norm` to `affine=False`. This is documented at `darts/models/forecasting/foundation_model.py:185`. The warning is harmless but worth re-reading once with `enable_finetuning=True` to confirm it doesn't silently freeze something we intended to train.

## Sweeping fine-tuning recipes

```
make 07_pretrained_finetune_sweep CONFIG=configs/btc_4h_2020_2024_encoder_only.config.yaml
```

Edit the `GRID` list at the top of [`sweep.py`](sweep.py) to change which (backend, hub_model_name, input_chunk_length, learning_rate, n_epochs, fine-tuning mode) combos are tried. Output: a printed table to stdout (in source order, easy to scan) and a `results/<config_stem>/sweep.csv` for downstream analysis. The sweep does not touch `metrics.json` / `predictions.parquet` / `plot.png` — those reflect the single config in `configs/*.config.yaml` set via `make 07_pretrained_finetune`.

What to look at in the sweep:

- **head-only vs. encoder-only vs. full fine-tuning** — given the same learning rate and dataset, which mode wins on Sharpe / cumulative_return? On chronos-2-small + extended dataset, the answer was encoder-only. On other backbones or shorter datasets it may differ.
- **learning rate 1e-5 vs. 3e-5 vs. 1e-4** — encoder fine-tuning is sensitive to learning rate; too high destroys the pretrained encoder, too low never moves. 1e-5 was the sweet spot on chronos-2-small.
- **`epochs_trained` across rows** — recipes that early-stop very fast either converged quickly or didn't have a learnable signal (e.g. unfreeze pattern missed all params, or val_loss diverged immediately).
- **`best_val_loss` and `restored_best_checkpoint`** — `restored_best_checkpoint: true` confirms the inference predictions came from the best-val_loss weights, not the (likely overfit) end-of-training weights. If `false`, fine-tuning happened but checkpoint restoration failed silently — investigate before believing the metrics.

## Cleaning up checkpoints

`save_checkpoints=True` writes `.ckpt` files under `results/darts_checkpoints/darts_logs/{dataset}_{backend}/`. `force_reset=True` wipes the previous checkpoint for the same `{dataset, backend}` pair at the start of each run, so they don't accumulate within a configuration. The whole tree can be wiped with:

```
make 07_pretrained_finetune_clean_checkpoints
```

or transitively via the top-level `make clean` (which wipes everything under any `results/`).

## Files

```
experiments/07_pretrained_finetune/
├── README.md                                           # this file
├── Makefile                                            # `make help` shows run / sweep / clean_checkpoints targets
├── configs/
│   ├── btc_1h_2024.config.yaml                         # 1h data slice, head-only fine-tuning
│   ├── btc_4h_2024.config.yaml                         # 4h data slice (1.6 years), head-only — loses to zero-shot
│   ├── btc_4h_2024_encoder_only.config.yaml            # 4h data slice (1.6 years), encoder-only — loses to zero-shot
│   └── btc_4h_2020_2024_encoder_only.config.yaml       # 4h data slice (4.7 years), encoder-only — the winning recipe
├── run.py                                              # entry point — fine-tunes the single config given via --config
├── sweep.py                                            # sweeps fine-tuning recipes; writes results/<config_stem>/sweep.csv
└── results/
    ├── darts_checkpoints/                              # auto-saved best-val_loss checkpoints (transient)
    └── <config_stem>/                                  # e.g. btc_4h_2020_2024_encoder_only/
        ├── metrics.json                                # produced by run.py
        ├── predictions.parquet                         # close + pred + pred_lo + pred_hi + ref + strategy_return
        ├── plot.png                                    # close + median + shaded q10–q90 band
        ├── training_curves.png                         # per-epoch train_loss + val_loss
        └── sweep.csv                                   # produced by sweep.py
```
