# 07_pretrained_finetune

Natural follow-up to [`06_pretrained`](../06_pretrained/): same two foundation-model backends — Amazon's [Chronos-2](https://huggingface.co/amazon/chronos-2) (encoder-only T5-style, 120M params) or Google's [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (decoder-only patch-transformer, 200M params) — but with `enable_finetuning` set so `fit()` actually updates weights against a held-out validation slice before walk-forward inference.

```
HuggingFace Hub weights ──► darts FoundationModel ──► fine-tune on train, early-stop on val_loss
                                              │
                                              └──► historical_forecasts(retrain=False, num_samples=200)
                                                    │
                                                    └──► quantile(0.1) / 0.5 / 0.9 ──► reconstruct close
```

The target is the same bar-T log-return `r_T = log(close_T / close_{T-1})` used by [`04_lstm`](../04_lstm/), [`05_transformer`](../05_transformer/), and [`06_pretrained`](../06_pretrained/); predictions are reconstructed to price as `close_pred = close_{T-1} * exp(r_q0.5)`. MAE / RMSE / MAPE / directional_accuracy / Sharpe stay directly comparable to all prior experiments.

## Why this experiment

[`06_pretrained`](../06_pretrained/) answers "does a generic time-series prior transfer to BTC zero-shot?" — its `fit()` is a no-op and the weights come straight from HuggingFace. The realistic outcome there is MAE near the naive floor with `directional_accuracy ≈ 0.50`: the prior is symmetric, the signal is in the data, and the parameter count buys nothing extra off the shelf.

07 asks a different question: **if we let those weights move toward this asset and timescale, does the pretrained init give us a better starting point than [`05_transformer`](../05_transformer/) trained from scratch?** Concretely:

- **MAE / Sharpe materially below 05** → the pretrained init is a real advantage. The foundation model learned a useful inductive bias about scale and noise structure that a fresh-from-scratch TFT had to (and failed to) discover from a few thousand bars alone.
- **MAE / Sharpe match 05** → the asset / timescale carries enough signal that *any* sufficient-capacity model lands in the same place. Pretraining buys convergence speed, not accuracy.
- **MAE / Sharpe match 06's zero-shot** → fine-tuning didn't move the needle. Either we under-trained (early-stopping too aggressive), or the prior is already as good as it gets for this slice.
- **MAE / Sharpe worse than 06** → catastrophic forgetting. The fine-tuning recipe (learning rate too high, too many parameters unfrozen, too few epochs to recover) destroyed the prior. Lower the learning rate or tighten the unfreeze pattern.

This is *not* the same experiment as 06 with `enable_finetuning=True` toggled — 07 also holds out the validation slice (06 folds it into train, because zero-shot has nothing to early-stop on) and routes EarlyStopping(`monitor=val_loss`) the same way as 05.

## Why univariate

Same constraint as 06: TimesFM 2.5 doesn't accept covariates at all, and we keep Chronos-2 univariate too so the head-to-head between the two backends compares priors rather than feature engineering. Past / future covariates remain a clean follow-up ablation.

## How it's done

1. Load OHLCV via the shared loader ([src/btc_ai/data](../../src/btc_ai/data)).
2. Build target `r_T = log(close_T / close_{T-1})` and `ref_T = close_{T-1}` for downstream price reconstruction. **Univariate only.**
3. Three-way time split: `[0:train_end]` for training, `[train_end:val_end]` for validation (drives EarlyStopping), `[val_end:n]` for test.
4. Fit a `darts.dataprocessing.transformers.Scaler` on the **training slice only**. Cast to `float32` so the model can run on Apple MPS or CUDA.
5. Build the model based on `cfg['backend']` with fine-tuning enabled:
   - `chronos` → `Chronos2Model(input_chunk_length, output_chunk_length=1, hub_model_name, likelihood=QuantileRegression([0.1, 0.5, 0.9]), enable_finetuning=cfg['finetune']['enable_finetuning'], n_epochs, batch_size, optimizer_kwargs={'lr': ...}, pl_trainer_kwargs={'callbacks': [EarlyStopping('val_loss', patience=...)], ...})`
   - `timesfm` → same with `TimesFM2p5Model(...)`.
6. Call `model.fit(series=target_train_s, val_series=target_val_s)`. Unlike 06, this **does** update weights — only the params matched by the `enable_finetuning` pattern (or all params, if `True`).
7. Walk-forward predict on the test slice via `model.historical_forecasts(start=test_start, forecast_horizon=1, retrain=False, last_points_only=True, num_samples=200)`. Each step samples 200 futures from the fine-tuned predictive distribution.
8. Inverse-scale the stochastic series, then extract `quantile(0.1) / quantile(0.5) / quantile(0.9)` to get `r_q_lo / r_q_med / r_q_hi`.
9. Reconstruct prices: `close_pred = ref * exp(r_q_med)` for the point prediction (the median), plus `pred_lo / pred_hi` for the interval.
10. Compute MAE, RMSE, MAPE, directional accuracy, cumulative return, and annualized Sharpe on the **median** prediction via the shared metrics in [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py).
11. Write `results/<config_stem>/metrics.json` (including `enable_finetuning`, `epochs_trained`, `rows_val`), `results/<config_stem>/predictions.parquet` (`close, pred, pred_lo, pred_hi, ref, strategy_return`), `results/<config_stem>/plot.png`.

Library: **[darts](https://unit8co.github.io/darts/)** (`darts.models.Chronos2Model`, `darts.models.TimesFM2p5Model`). HuggingFace download is automatic on first run and cached in `~/.cache/huggingface/hub/`; if you already ran 06, the chronos-2 weights are already there.

## How to run

```
make 07_pretrained_finetune                                                  # default config (4h)
make 07_pretrained_finetune CONFIG=configs/btc_1h_2024.config.yaml           # 1h variant
```

Pick a backend in the config — there is **no default**, you must set one:

```yaml
backend: chronos          # or 'timesfm'
hub_model_name: amazon/chronos-2
                          # chronos: amazon/chronos-2 | autogluon/chronos-2-small | autogluon/chronos-2-synth
                          # timesfm: google/timesfm-2.5-200m-pytorch
```

The `finetune.enable_finetuning` key drives the recipe:

```yaml
finetune:
  enable_finetuning:
    unfreeze:
      - '*output_patch_embedding*'   # partial fine-tuning, head only (default — light on Apple Metal Performance Shaders)
  # enable_finetuning: true          # full fine-tuning (every parameter trainable; heavy on Apple Metal Performance Shaders)
  # enable_finetuning: false         # zero-shot — equivalent to 06_pretrained
```

Training knobs:

```yaml
training:
  early_stopping_patience: 5         # epochs of no val_loss improvement before stop

model:
  input_chunk_length: 256            # bars of history (chronos-2 max=8192, timesfm-2.5 max=16384)
  output_chunk_length: 1
  num_samples: 200
  quantiles: [0.1, 0.5, 0.9]
  n_epochs: 20                       # upper bound; EarlyStopping usually stops earlier
  batch_size: 32                     # 32 fits chronos-2 head-only fine-tuning on Apple Metal Performance Shaders at input_chunk_length=256
  learning_rate: 0.0001              # 1e-4 is conventional for fine-tuning; 1e-3 risks destroying the prior
  random_state: 42
```

First run downloads the HuggingFace weights (one-time, then cached). Head-only fine-tuning + walk-forward over ~1100 test bars takes single-digit minutes on Apple Metal Performance Shaders or a recent CUDA GPU. Full fine-tuning is several times slower and may exceed Apple Metal Performance Shaders memory at `input_chunk_length=256` + `batch_size=32` — drop `batch_size` to 16 or move to CUDA.

## How to choose unfreeze patterns

`enable_finetuning: {unfreeze: [...]}` accepts a list of [`fnmatch`](https://docs.python.org/3/library/fnmatch.html)-style glob patterns matched against `model.named_parameters()` names (darts applies them at `torch_forecasting_model.py:_setup_finetuning`). The default `'*output_patch_embedding*'` matches the Chronos-2 prediction head — it's the cheapest possible fine-tune and the safest baseline.

To verify the pattern matches anything (or to extend it to e.g. the last encoder block), inspect the names after `model.fit()` has run once:

```python
# from inside an interactive session:
for name, param in model.model.named_parameters():
	print(f'{param.requires_grad}\t{name}')
```

Anything with `requires_grad=True` is being trained. If no parameters match your pattern, EarlyStopping will fire at the patience threshold with no learning signal and metrics will land within float noise of 06's zero-shot — that's the diagnostic that tells you the pattern missed.

TimesFM 2.5's head and encoder names differ from Chronos-2's; the default pattern in the config does **not** match TimesFM. When `backend: timesfm`, either switch to `enable_finetuning: true` (full fine-tuning) or introspect first and supply a TimesFM-specific pattern.

## How to interpret

Compare against earlier experiments **on the same data slice**:

| | Naive | MA(24) | ARIMA(1,1,1) | XGBoost | LSTM | TFT | Chronos-2 / TimesFM 2.5 (zero-shot) | **Chronos-2 / TimesFM 2.5 (fine-tuned)** |
|---|---|---|---|---|---|---|---|---|
| Family | last-value | rolling mean | linear | tabular GBT | recurrent NN | attention NN | pretrained foundation | **pretrained + fine-tuned** |
| Trained on this slice | — | — | yes | yes | yes | yes | no | **yes (head ± last block)** |
| Probabilistic | — | — | — | — | — | — | yes | **yes (q10 / q50 / q90)** |

What the numbers tell you:

- **MAE materially below 06** — the prior was useful but generic; specializing it to BTC scale + noise picks up additional signal. Cross-check with directional_accuracy and Sharpe.
- **MAE matches 06** — the prior is already as good as it can be for this asset/timescale, or the unfreeze pattern is too narrow. Try `enable_finetuning: true` and a lower lr.
- **MAE worse than 06** — catastrophic forgetting. The learning rate is too high or too many parameters moved. Drop the learning rate to 3e-5 or tighten the unfreeze pattern.
- **Sharpe matches or beats ARIMA's `+7.52`** — the fine-tuned foundation model caught up on the metric that matters for trading. The pretrained init + a small specialization budget delivered something a from-scratch 05 didn't.
- **`epochs_trained` ≈ `n_epochs`** — EarlyStopping never fired; val_loss kept improving. Bump `n_epochs` and rerun to see if there's more headroom.
- **`epochs_trained` ≪ `n_epochs`** — EarlyStopping fired early, the model converged or stalled. Healthy.

### Caveats baked into this v1

- **Univariate.** No past covariates (volume, OHLC range/body) and no future covariates (hour-of-day, day-of-week). TimesFM doesn't accept any; Chronos-2 does but we deliberately match 06 for parity.
- **Walk-forward without re-estimation.** Fine-tuning happens once in `fit()`; `retrain=False` keeps weights frozen across the entire test window. Matches [`02_arima`](../02_arima/) / [`04_lstm`](../04_lstm/) / [`05_transformer`](../05_transformer/) / [`06_pretrained`](../06_pretrained/).
- **`output_chunk_length: 1`.** One-step-ahead only. Multi-horizon is a one-line config change but changes the loss surface.
- **Unfreeze patterns are backend-specific.** The default targets the Chronos-2 head. TimesFM needs introspection; the comment in the config flags this.
- **Apple Metal Performance Shaders memory pressure** at `input_chunk_length=256` + full fine-tuning. Head-only fine-tuning is the default for that reason. If chronos-2 / timesfm-2.5 runs out of memory, drop `batch_size` to 16 or 8, or switch to CUDA.
- **TimesFM RINorm-affine override.** darts logs a warning when foundation-model checkpoints lack RINorm affine weights; the wrapper overrides `use_reversible_instance_norm` to `affine=False`. This is documented at `darts/models/forecasting/foundation_model.py:185`. The warning is harmless but worth re-reading once with `enable_finetuning=True` to confirm it doesn't silently freeze something we intended to train.
- **Single sampling seed.** `num_samples=200` gives a reasonable distribution but two runs may produce slightly different quantile bands. Average over a few seeds for a publication-grade number.

## Sweeping fine-tuning recipes

```
make 07_pretrained_finetune_sweep                                              # default config (4h)
make 07_pretrained_finetune_sweep CONFIG=configs/btc_1h_2024.config.yaml       # 1h variant
```

Edit the `GRID` list at the top of [`sweep.py`](sweep.py) to change which (backend, hub_model_name, input_chunk_length, learning_rate, n_epochs, fine-tuning mode) combos are tried. Output: a printed table to stdout (in source order, easy to scan) and a `results/<config_stem>/sweep.csv` for downstream analysis. The sweep does not touch `metrics.json` / `predictions.parquet` / `plot.png` — those reflect the single config in `configs/*.config.yaml` set via `make 07_pretrained_finetune`.

What to look at in the sweep:

- **head-only vs. full fine-tuning** — does unfreezing everything beat unfreezing only the head, given the same learning rate and `n_epochs`? If not, head-only is the cheaper default.
- **learning rate 1e-4 vs. 3e-5** — full fine-tuning runs are sensitive to learning rate; a too-high learning rate destroys the prior. The sweep includes both so the symptom is visible.
- **`epochs_trained` across rows** — recipes that early-stop very fast either converged quickly or didn't have a learnable signal (e.g. unfreeze pattern missed all params).

## Files

```
experiments/07_pretrained_finetune/
├── README.md                     # this file
├── Makefile                      # `make help` shows run / sweep targets
├── configs/
│   ├── btc_1h_2024.config.yaml   # 1h data slice + fine-tuning recipe + model knobs
│   └── btc_4h_2024.config.yaml   # 4h data slice + fine-tuning recipe + model knobs
├── run.py                        # entry point — fine-tunes the single config given via --config
├── sweep.py                      # sweeps fine-tuning recipes; writes results/<config_stem>/sweep.csv
└── results/
    └── <config_stem>/            # e.g. btc_4h_2024/
        ├── metrics.json          # produced by run.py
        ├── predictions.parquet   # close + pred + pred_lo + pred_hi + ref + strategy_return
        ├── plot.png              # close + median + shaded q10–q90 band
        └── sweep.csv             # produced by sweep.py
```
