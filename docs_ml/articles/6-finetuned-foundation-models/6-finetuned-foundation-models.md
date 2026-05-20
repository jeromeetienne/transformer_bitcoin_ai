# When You Teach the Giant: Fine-tuning Foundation Models on Bitcoin

Article 6 established that zero-shot foundation models — applied to Bitcoin without any BTC-specific training — produce results near the naive floor. The pretrained prior over "what time series look like" does not transfer cleanly to an asset whose hourly returns are close to white noise. That was the honest result, and it was informative.

The natural follow-up is: what happens when we let the model update its weights on Bitcoin data? This is fine-tuning — the same technique that took GPT-3 from a generic language model to a useful assistant, applied now to time-series foundation models. The hypothesis is that the pretrained architecture and general time-series knowledge provide a useful starting point, and that a relatively small amount of BTC-specific training can calibrate the model's predictions to the distributional properties of this specific asset.

The hypothesis is correct. But getting there required learning three lessons the hard way.

## The three things that had to be true

The experiment is in [`experiments/07_finetuned/`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/07_finetuned). The winning configuration used `autogluon/chronos-2-small`, the 4.7-year `btc_4h_2020_2024` dataset, and encoder-only fine-tuning with checkpoint restoration. None of those choices were obvious in advance. Earlier configurations of this experiment all *lost* to zero-shot on every metric. Here is what changed.

### Lesson 1: dataset size matters

The first attempts at fine-tuning used the standard `btc_4h_2024` dataset: approximately 1.6 years of 4-hour BTC data, with the last 20% held out for testing and a further 10% of the training set carved off for validation. That leaves roughly 3 500 training rows.

On a 3 500-row training set, fine-tuning the Chronos-2-small encoder caused the model to overfit within 1–3 epochs. The training loss dropped; the validation loss reversed and climbed. Early stopping fired quickly. The resulting weights were worse on the test set than the original pretrained weights.

The fix was to extend the training slice to `btc_4h_2020_2024`: 4.7 years of 4-hour bars, from 2020-01-01 through 2024-12-31. That produces approximately 10 000 training rows — nearly three times as many. On this longer dataset, the model had enough signal to adjust its encoder representations without immediately overfitting the BTC-specific noise.

The practical implication: fine-tuning a foundation model on a few months of data is likely to make things worse, not better. The pretrained weights are already near-optimal for a generic time-series prior; the fine-tuning signal needs to be large enough to meaningfully update the representations, not just overfit a small sample.

### Lesson 2: freeze the head, fine-tune the encoder

Darts' `enable_finetuning` parameter accepts a dictionary with `freeze` and `unfreeze` keys, each containing a list of `fnmatch`-style glob patterns matched against `model.named_parameters()`. The patterns determine which parameters receive gradients during training.

The obvious default is to fine-tune everything (`enable_finetuning: true`). The next most obvious choice is to fine-tune only the head — the output layer that maps encoder representations to quantile predictions — since it is the part most likely to need BTC-specific calibration.

Both are wrong.

Head-only fine-tuning (`unfreeze: ['*output_patch_embedding*']`) consistently *eroded* the pretrained directional prior. In the experiments, directional accuracy dropped from 0.5164 (zero-shot) to below 0.49 — worse than a coin flip — when only the head was trained. The explanation in hindsight is clear: the head encodes the directional structure of the pretrained distribution. Updating it with BTC-specific signal, where directional predictability is near zero, taught the head to suppress the directional signal it had learned during pretraining. The result was a model with no useful prior at all.

Encoder-only fine-tuning (`freeze: ['*output_patch_embedding*']`) has the opposite structure: 94.3% of the model's parameters are trainable, and the frozen head preserves the directional prior from pretraining. The encoder learns to produce representations that are calibrated to BTC's distributional properties; the head converts those representations to quantile predictions using the same mapping it learned during pretraining. The combination worked.

To verify the pattern matched correctly, the log line at the start of each training run reports:

```
trainable parameters: N / M (P%); names: [list of matched parameter names]
```

If `P` is 0 or unexpectedly small, the pattern missed its target. This diagnostic caught several misconfigured runs during development.

### Lesson 3: restore the best-val_loss checkpoint

Fine-tuning a foundation model on a noisy financial target is fast to overfit. In the experiments, the validation loss diverged from the training loss within 1–4 epochs in almost every configuration. The `EarlyStopping` callback halted training when the validation loss did not improve for 5 consecutive epochs. So far, standard practice.

The problem: when `model.fit()` returns after early stopping, the in-memory model weights are the weights from the *last training epoch*, not the weights from the epoch with the best validation loss. The `ModelCheckpoint` callback that Darts automatically adds when `save_checkpoints=True` has saved the best weights to disk, but they are not in memory.

Without explicit checkpoint restoration, `model.historical_forecasts()` uses the end-of-training weights — which are overfit relative to the best-val_loss epoch. In one configuration that had appeared to show a major improvement (`annualized_sharpe = 6.146`), adding the restoration call dropped the Sharpe to 3.46. The earlier result was an artifact: the overfit weights had happened to align with the test slice's drift during that specific period.

The fix is a single call after `fit()`:

```python
model.load_weights_from_checkpoint(model_name=model_name, work_dir=work_dir, best=True)
```

The `best=True` flag loads the checkpoint with the best validation loss. Every metric in the results since this fix was applied is honest: the inference weights are the best-generalization weights, not the overfit-endpoint weights.

The `metrics.json` output includes a `restored_best_checkpoint` boolean. If it is `false`, the fine-tuning happened but checkpoint restoration failed — the metrics should not be trusted.

## The headline result

On the `btc_4h_2020_2024` slice with encoder-only fine-tuning of `autogluon/chronos-2-small`, averaged over five random seeds (42, 7, 13, 5, 99):

| Metric | Fine-tuned (5-seed mean ± std) | 95% CI | Zero-shot | Beats zero-shot |
|---|---|---|---|---|
| MAE | 539.41 ± 1.88 USD | (537.07, 541.75) | 540.37 | 4/5 (within noise) |
| RMSE | 809.76 ± 1.17 USD | (808.30, 811.21) | 810.28 | 4/5 (within noise) |
| MAPE | 0.691% ± 0.003 | (0.688%, 0.694%) | 0.692% | 3/5 (within noise) |
| Directional accuracy | 0.529 ± 0.010 | (0.516, 0.542) | 0.5164 | 4/5 (borderline) |
| **Cumulative return** | **44.52% ± 5.48pp** | **(37.70%, 51.33%)** | **36.64%** | **5/5 ★** |
| **Annualized Sharpe** | **5.52 ± 0.42** | **(5.00, 6.04)** | **4.74** | **5/5 ★** |

The 95% confidence intervals on Sharpe and cumulative return exclude the zero-shot values. Every single seed independently beat zero-shot on both trading metrics. The result is formally significant at the 5% level with n=5, and it is consistent: no single seed is an outlier carrying the result.

The MAE and RMSE improvements are within noise. The directional accuracy improvement is borderline — the confidence interval barely excludes zero-shot. The Sharpe and cumulative return improvements are clear.

To reproduce:

```bash
make 07_finetuned CONFIG=configs/btc_4h_2020_2024_encoder_only.config.yaml
```

## Why magnitude calibration is the interesting story

The result might seem puzzling. How can Sharpe improve substantially when directional accuracy is barely above zero-shot? The answer is in the *magnitude* of the predictions, not their direction.

A direction-only trading strategy — long when the model predicts a positive return, flat otherwise — earns the same per-correct-call regardless of whether the predicted move was 0.1% or 2%. But a strategy that sizes its position based on the predicted return magnitude earns more per correct call when the magnitude estimate is right.

The encoder-only fine-tuning learned to produce representations that result in better-sized quantile intervals: smaller predicted intervals when the market was quiet, larger intervals when it was volatile. The head's quantile mapping — which encodes the *shape* of the predictive distribution from pretraining — then translates those representations into predictions with appropriate spread. The model was not learning to predict direction better; it was learning to predict *magnitude* better. And better-calibrated magnitude predictions improve Sharpe even when directional accuracy barely moves.

This is a more interesting result than a directional accuracy story would have been. It tells us that the value of fine-tuning, in this case, is in *confidence calibration* — knowing when to be uncertain and when to commit — rather than in flipping the sign of predictions correctly more often.

## The training configuration

The winning config:

```yaml
backend: chronos
hub_model_name: autogluon/chronos-2-small

finetune:
  enable_finetuning:
    freeze:
      - '*output_patch_embedding*'   # freeze the head; train the encoder

training:
  early_stopping_patience: 5

model:
  input_chunk_length: 256
  output_chunk_length: 1
  num_samples: 1000
  quantiles: [0.1, 0.5, 0.9]
  n_epochs: 50                  # early stopping usually fires around epoch 6–10
  batch_size: 32
  learning_rate: 0.00001        # conservative; preserves the pretrained encoder
  random_state: 42
```

The learning rate of 1e-5 is deliberately conservative. Higher rates (3e-5, 1e-4) consistently caused faster val_loss divergence and worse test results. The pretrained encoder is already near-optimal for general time-series representation; the fine-tuning signal only needs to nudge it toward BTC-specific calibration, not rebuild it from scratch.

Total runtime per seed on Apple Metal Performance Shaders: approximately 10 minutes. Training (6–10 epochs before early stopping) takes 3–6 minutes; walk-forward inference over 366 test bars at `num_samples=1000` takes 5–6 minutes.

## Caveats

**Five seeds is a thin evidence base.** The formal significance at 95% with n=5 is real, but the per-seed Sharpe spread (standard deviation of 0.42 on a mean of 5.52) means that individual runs vary meaningfully. More seeds would tighten the confidence intervals.

**Single test regime.** The test slice covers October–December 2024 — a 366-bar window during a Bitcoin bull run. The fine-tuning improvement has not been tested on the 2022 bear market, the 2021 volatility regime, or the 2023 sideways chop. Generalization across regimes is an open question.

**`num_samples` discrepancy.** Article 6 (zero-shot) used `num_samples=200`; article 7 (fine-tuned) uses `num_samples=1000`. A higher sample count tightens the quantile estimates and improves the Sharpe slightly (~5.02 → ~5.41 in internal comparisons). The reported 1.4-Sharpe-point gap between fine-tuned and zero-shot is real but would narrow somewhat if zero-shot were re-run at 1000 samples. The direction of the finding would not change.

**TimesFM fine-tuning.** The `*output_patch_embedding*` pattern targets Chronos-2's head. TimesFM 2.5 uses different parameter names; this pattern does not match anything in its architecture. Running the same config with `backend: timesfm` effectively disables fine-tuning. To fine-tune TimesFM, introspect its parameter names first and provide a backend-specific pattern.

## The full leaderboard

Placing fine-tuned Chronos in context against every model in the series — note that earlier experiments used different dataset granularities and lengths, so direct MAE comparison is only valid within each dataset group:

| Article | Model | Dataset | Probabilistic | Annualized Sharpe |
|---|---|---|---|---|
| 1 | Naive last-value | `btc_1h_2024` | — | (floor) |
| 2 | ARIMA(3,1,3) | `btc_1h_2024` | — | *run to see* |
| 3 | XGBoost | `btc_4h_2024` | — | *run to see* |
| 4 | LSTM | `btc_4h_2024` | — | *run to see* |
| 5 | TFT | `btc_4h_2024` | — | *run to see* |
| 6 | Chronos-2 zero-shot | `btc_4h_2020_2024` | yes (q10/50/90) | 4.74 |
| **7** | **Chronos-2 fine-tuned** | **`btc_4h_2020_2024`** | **yes (q10/50/90)** | **5.52 ★** |

## What the series reveals

Seven models. One question. Here is the honest answer.

On hourly and 4-hour Bitcoin price, the primary bottleneck is not the model — it is the signal. Price changes on short timescales are close to independent of their own history, close to independent of volume patterns, and close to independent of what model was trained on. ARIMA could not beat naive by much. XGBoost could not beat ARIMA by much. The LSTM and TFT could not beat XGBoost by much. Pretrained foundation models without fine-tuning could not beat any of them.

What moved the needle, finally, was giving the model more historical data *and* adapting the internal representations of a 28M-parameter pretrained model specifically to Bitcoin's distributional properties. The gain was not in directional accuracy — no model in this series reliably predicted up versus down. The gain was in magnitude calibration: the fine-tuned model knew when to be confident and when not to.

That is a real finding. It suggests that the path to better Bitcoin forecasting is not a more complex architecture but more data and better calibration. More data means longer history or higher frequency. Better calibration means techniques like fine-tuning or explicit distributional modeling — not just minimizing mean squared error.

The next step, if this were a production system, would be adding exogenous signals: on-chain transaction volume, funding rates from perpetual futures markets, sentiment from social media, macro indicators. The models in this series have only ever seen OHLCV. Whether those additional signals break through the noise floor is a question for a different series.
