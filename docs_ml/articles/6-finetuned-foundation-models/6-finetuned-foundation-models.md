# When fine-tuning beats zero-shot — and the three conditions that make it happen

The previous article ended with three zero-shot foundation models — Amazon's Chronos-2 in two sizes, Google's TimesFM 2.5 — losing to a 3-parameter linear ARIMA on every leaderboard metric. Two of three were below coin-flip on directional accuracy. The "borrowing pretrained parameters wins by default" framing the foundation-model literature occasionally implies did not survive contact with `BTCUSDT` 4-hour log-returns. This article is the follow-up. The same backbones, *fine-tuned* on Bitcoin instead of run zero-shot. The question is whether fine-tuning closes the gap article 5 opened.

The honest answer is *yes — on three metrics, under three conditions, on a favourable seed*. Single-seed on disk, the canonical 07 row reports **annualized Sharpe 8.20** — the highest in the entire series. Five seeds averaged per the experiment's README report **Sharpe 5.523 ± 0.418, 95 % confidence interval (5.00, 6.04)**. Both numbers belong in the article. The CI lower bound 5.00 still clears the 06 zero-shot benchmark; the single-seed 8.20 is a favourable seed. The lift is real, and it requires the right train-slice length, the right architectural surface to fine-tune, and a workflow detail (restoring the best-validation-loss checkpoint before walk-forward inference) without which the Sharpe collapses from 6.15 to 3.46 on the same model.

Mean absolute error stays with ARIMA. Across seven experiments the MAE band for six of seven models is $539–$571; ARIMA(3, 1, 3) is at the floor of that band; the fine-tuned foundation model is at $549.85, $10.70 above ARIMA. The fine-tuned model is *better at direction and better at predicted magnitude calibration on the bars it gets right*. ARIMA is *closer to the price on average*. Two different victory conditions, on the same 366 test bars.

## Methodology, in one paragraph

Target: one-step-ahead log-return `r_T = log(close_T / close_{T-1})`, reconstructed back to price as `close_pred = close_{T-1} * exp(r_pred)`. Data: `BTCUSDT` 4-hour bars from Binance Vision. Split: train `2020-01-01` → `2024-08-01` UTC (**10 043 rows** — extended from 1.6 years to 4.7 years), validation `2024-08-01` → `2024-10-01` UTC (366 rows, used for early stopping *and* for selecting the best-loss checkpoint), test `2024-10-01` → `2024-12-01` UTC (**366 bars** — unchanged from articles 1 through 5). Walk-forward, one step ahead, weights frozen across the test window after fine-tuning. Metrics from the shared module — [`src/btc_ai/eval/metrics.py`](src/btc_ai/eval/metrics.py). Reproduce: `make 07_finetuned`. **New this article:** `fit()` actually updates weights now (it was a no-op in article 5), the train slice extends back to 2020-01-01 (3× more rows than article 5's slice), and the best-validation-loss checkpoint is restored before walk-forward inference.

## What changed from article 5

Three architectural surfaces are different from the zero-shot setup in article 5. All three matter; swapping any one of them out collapses the lift.

**The train slice is 4.7 years instead of 1.6 years.** The dataset config moved from [`btc_4h_2024.dataset.yaml`](configs/datasets/btc_4h_2024.dataset.yaml) (train `2023-01-01` → `2024-08-01`) to [`btc_4h_2020_2024.dataset.yaml`](configs/datasets/btc_4h_2020_2024.dataset.yaml) (train `2020-01-01` → `2024-08-01`). Training rows went from 3 833 to 10 043 — roughly 2.6× more bars. The dataset config's own comment names the motivation directly: at 1.6 years of training data, every fine-tuning recipe (head-only, encoder-only, full) consistently overfit within ~3 epochs (train loss collapsed, validation loss diverged), and every fine-tuned variant landed *worse* than zero-shot on the trading metrics. The hypothesis — that the 1.6-year slice was the bottleneck — turned out to be correct.

**The fine-tuning recipe is encoder-only.** From [`btc_4h_2020_2024_encoder_only.chronos-2-small.config.yaml`](experiments/07_finetuned/configs/btc_4h_2020_2024_encoder_only.chronos-2-small.config.yaml):

```yaml
finetune:
  enable_finetuning:
    freeze:
      - '*output_patch_embedding*'   # keep the head static
```

The `freeze: ['*output_patch_embedding*']` pattern is an fnmatch glob against `model.named_parameters()`. Everything matching the pattern is frozen; everything else becomes trainable. The pattern matches Chronos-2's output prediction head — the layer that maps the encoded representation to a distribution over next-token IDs. By freezing the head, the recipe **preserves the directional prior the pretrained head already carries**, and fine-tunes the encoder beneath it. Approximately 94.3 % of Chronos-2 small's 28 million parameters become trainable; the head's roughly 1.7 million parameters do not.

The alternative recipes are documented:
- `enable_finetuning: false` — zero-shot (article 5's setup).
- `enable_finetuning: true` — full fine-tuning, every parameter updates.
- `enable_finetuning: {unfreeze: [...]}` — head-only (the previous canonical, demoted to a sibling row).
- `enable_finetuning: {freeze: [...]}` — encoder-only (the current canonical).

**The best-validation-loss checkpoint is restored before walk-forward inference.** The `metrics.json` field `restored_best_checkpoint: true` confirms it for the canonical row. The mechanism: PyTorch Lightning's `ModelCheckpoint(best=val_loss)` callback saves the model state at the epoch where validation loss is minimized; after training finishes (early-stopped at epoch 10 on the canonical, with best validation loss at epoch 5), `model.load_weights_from_checkpoint(best=True)` swaps the end-of-training weights for the best-validation-loss weights before walk-forward inference. Without this step, validation-loss divergence within 1–4 epochs silently feeds overfit end-of-training weights into the test slice. The experiment's README documents one such incident directly: **the same configuration posted Sharpe 6.146 before checkpoint restoration was wired in, and 3.46 after** — a 2.7-point swing from a single workflow detail. The 3.46 is the honest number; the 6.146 was overfit weights making the model look better than it was.

The third condition is the kind of thing it is easy to skip silently. It is also the kind of thing that, skipped silently, makes leaderboard rows lie. Watching for `restored_best_checkpoint: true` (or `false`) in `metrics.json` is methodology hygiene visible in the artefact, in the same shape as ARIMA's `dynamic=False` flag from article 1.

## The configuration, end to end

```yaml
# experiments/07_finetuned/configs/btc_4h_2020_2024_encoder_only.chronos-2-small.config.yaml
backend: chronos
hub_model_name: autogluon/chronos-2-small    # 28 M parameters

finetune:
  enable_finetuning:
    freeze:
      - '*output_patch_embedding*'

training:
  early_stopping_patience: 3

model:
  input_chunk_length: 256          # ≈ 42 days at 4h
  output_chunk_length: 1
  num_samples: 1000                # raised from 200 in article 5
  quantiles: [0.1, 0.5, 0.9]
  n_epochs: 50                     # upper bound; early-stopped
  batch_size: 32
  learning_rate: 1e-5              # conservative — 8× more parameters in motion than head-only
  random_state: 42
```

Two of those numbers are worth a sentence each. **`num_samples: 1000`** is up from article 5's `200`. Each forecast at each test bar is now 1 000 stochastic draws from the model's predictive distribution; the three quantiles are computed across those samples. Five times more samples cuts the per-seed quantile-estimate noise by approximately √5 — a real component of the lift, though not the primary one (recipe and train-slice length matter more). **`learning_rate: 1e-5`** is conservative on purpose: the encoder-only recipe puts roughly eight times more parameters in motion than the previous head-only recipe, and a small learning rate preserves more of the pretrained encoder structure. The configuration's own comment makes this trade-off explicit.

Training reported: 10 epochs trained, early-stopped on `val_loss` patience 3, best `val_loss` 0.01579 at epoch 5. The 4.7-year slice's best `val_loss` is roughly 2.3× lower than the 1.6-year sibling's `val_loss` of 0.03598 for the same recipe and learning rate — more training data lowers the validation surface, which is the headline mechanism behind the lift.

## The numbers

From [`experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/metrics.json`](experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/metrics.json):

| Metric | Chronos-2 small, encoder-only fine-tune, 4.7-yr train slice |
|---|---|
| MAE | 549.85 USD |
| RMSE | 813.80 USD |
| MAPE | 0.7020 % |
| Directional accuracy | **0.5683** |
| Cumulative return | **0.7329** |
| Annualized Sharpe | **8.1987** |

The complete leaderboard across all seven experiments, on the same 366-bar test slice:

| Model | MAE (USD) | RMSE (USD) | dir_acc | cum_ret | Sharpe |
|---|---|---|---|---|---|
| Naive last-value | 540.96 | 813.14 | NaN | — | — |
| ARIMA(3, 1, 3) | **539.15** | 808.79 | 0.5082 | 0.5269 | 6.8559 |
| XGBoost (31 features) | 550.85 | **808.65** | 0.5082 | 0.4149 | 6.1388 |
| LSTM | 539.23 | 808.29 | 0.5301 | 0.4284 | 4.8221 |
| Temporal Fusion Transformer | 891.09 | 1 271.26 | 0.5055 | 0.1219 | 2.5390 |
| Chronos-2 small (zero-shot) | 546.66 | 817.81 | 0.4754 | 0.2225 | 2.9722 |
| **Chronos-2 small (fine-tuned, canonical)** | 549.85 | 813.80 | **0.5683** | **0.7329** | **8.1987** |

The 07 row leads directional accuracy, cumulative return, and annualized Sharpe. Directional accuracy 0.5683 is 0.038 above the next best (the LSTM at 0.5301) — about 14 more correctly-directed bars in 366. Sharpe 8.1987 is 1.34 above ARIMA's long-standing 6.8559. The MAE row stays with ARIMA — 539.15 vs. 549.85, a $10.70 gap.

But the number on disk is one seed. The right way to read it is what comes next.

## Single-seed versus five-seed — the honest framing

The canonical `metrics.json` reports `random_state: 42`, a single deterministic seed. The experiment's README ([`experiments/07_finetuned/README.md`](experiments/07_finetuned/README.md)) ships a five-seed comparison for the same recipe and reports:

- **Sharpe 5.523 ± 0.418** (95 % CI 5.004 – 6.043)
- **Cumulative return 44.52 % ± 5.48 pp** (95 % CI 37.70 – 51.33 %)

The single-seed on disk (Sharpe 8.20, cum_ret 0.7329) sits **above the upper bound of the 95 % confidence interval** — it is a favourable seed, not the reproducible centre of the distribution. The honest read is *"the seed distribution clears 06 zero-shot at 95 % confidence,"* not *"a model reproducibly posts Sharpe 8.20."* Both numbers belong in the article: the 8.20 on disk is what `metrics.json` says today, and the 5-seed CI is what the headline finding should be cited against.

A per-bar Sharpe significance back-of-envelope, computed three ways:

| Citation | Sharpe | per-bar | SE | ratio (σ) |
|---|---|---|---|---|
| Single seed on disk | 8.1987 | 0.1752 | 0.0523 | **3.35** |
| 5-seed mean | 5.5230 | 0.1180 | 0.0523 | **2.26** |
| 5-seed CI lower bound | 5.0040 | 0.1069 | 0.0523 | **2.04** |

Under the same i.i.d. assumption that gave ARIMA(3, 1, 3) a per-bar ratio of 2.80 σ from article 1, the fine-tuned foundation model is between 2.04 σ and 3.35 σ depending on which number you cite. The CI lower bound (2.04 σ) is borderline; the on-disk single seed (3.35 σ) is clean; the mean (2.26 σ) is borderline. The seed *distribution* clears the 06 zero-shot mean at 95 % confidence — that is the load-bearing claim. The 8.20 single-seed value is one favourable draw.

## The three conditions, walked through the variants table

The experiment ships three sibling variants on disk that exercise the conditions one at a time. The full table from [`docs_ml/reports/07_finetuned.report.md`](docs_ml/reports/07_finetuned.report.md):

| Variant | train | recipe | backend | dir_acc | cum_ret | Sharpe |
|---|---|---|---|---|---|---|
| Canonical | 4.7 yr | encoder-only | chronos-2-small | **0.5683** | **0.7329** | **8.1987** |
| Swap backend | 4.7 yr | encoder-only | timesfm-2.5 | 0.5246 | 0.4512 | 5.3910 |
| Swap train slice & recipe | 1.6 yr | head-only | chronos-2-small | 0.5328 | 0.4162 | 5.6022 |

**Swap the backend.** Keep the train slice (4.7 years) and the recipe (encoder-only) constant. Swap Chronos-2 small for TimesFM 2.5. Sharpe drops 8.20 → 5.39 (single seed each). The mechanism: the `freeze: ['*output_patch_embedding*']` recipe freezes the *prediction head* and fine-tunes the *encoder*. On Chronos-2's encoder-only T5-style architecture, the encoder is a much larger architectural surface than the head; freezing the head and fine-tuning the encoder updates a lot of parameters while preserving the head's directional prior. On TimesFM 2.5's decoder-only patch-transformer, the same recipe updates a different (and smaller) effective surface, with less of the architecture's bias preserved. The "encoder-only" recipe is recipe-architecture-specific in a way the name does not advertise.

**Swap the train slice (and necessarily the recipe).** The previous canonical was head-only fine-tuning on the 1.6-year slice. Sharpe 5.60 single seed. Moving to encoder-only on the 4.7-year slice lifts Sharpe by 2.60. The README documents directly that on the 1.6-year slice, every fine-tuning recipe overfits within 1–3 epochs regardless of recipe — the validation loss diverges before the encoder has had time to extract anything generalizable. More train data delays overfitting; the encoder-only recipe needs that delay to land its lift. Both conditions are required; neither alone is sufficient.

**Skip the best-checkpoint restoration.** The README documents a prior incident: same configuration, Sharpe 6.146 before `model.load_weights_from_checkpoint(best=True)` was wired in, 3.46 after. The 3.46 is the honest number; the 6.146 was end-of-training weights that the validation-loss curve had already diverged past. The `restored_best_checkpoint: true` field in `metrics.json` is the artefact that confirms the right thing happened. If a future run flips it to `false` — by editing the config, by a regression in the training harness — every downstream number is suspect until investigated.

## The sweep — a different lesson

The repository includes a ten-row sweep over `(recipe, learning_rate)` configurations at [`experiments/07_finetuned/results/btc_4h_2024/sweep.csv`](experiments/07_finetuned/results/btc_4h_2024/sweep.csv). The sweep was run on the **former canonical's 1.6-year slice**, not the current canonical's 4.7-year slice — its rankings tell you which recipe and learning rate win at 1.6 years of training data, not whether those rankings hold at 4.7 years. A 4.7-year sweep does not exist on disk; that is an operational follow-up.

Two observations the sweep makes load-bearing.

**The Sharpe-leader row of the sweep — `encoder-only × learning_rate 1e-5`, Sharpe 6.87 on 1.6 years — is the same recipe / learning-rate corner as the canonical at 4.7 years (Sharpe 8.20).** Train-slice length is the lift; recipe and learning rate are the corner that gets the lift. The +1.33 Sharpe move from extending the slice at this corner is clean; the same recipe / learning rate on the 1.6-year slice already pointed at the right answer, just couldn't extract as much from it.

**The lowest-`val_loss` row of the sweep — `full × 1e-5`, val_loss 0.03479 — is *not* the Sharpe leader.** It posts Sharpe 5.73. The Sharpe leader (`encoder-only × 1e-5`) posts val_loss 0.03505 — slightly higher. *In-sample validation loss does not pick the out-of-sample Sharpe winner*. Same lesson as ARIMA's AIC ↔ Sharpe disagreement from article 1, now visible inside a foundation-model fine-tuning sweep. If you select your fine-tuning recipe by validation loss alone, you miss the trading-metric winner by half a Sharpe point. The seven-experiment series has now produced this disagreement three times (ARIMA sweep, LSTM sweep where the smallest config won Sharpe, and now 07's sweep) — the lesson generalizes.

## What this article tells us about the model class

**Fine-tuning a foundation model on a small target is not free.** Three conditions had to hold simultaneously — the right train-slice length, the right architectural surface to fine-tune, and the right checkpoint restoration. Without any one of them, the lift collapses. The 1.6-year train slice produced models that landed *below zero-shot* on the trading metrics regardless of recipe. The wrong recipe (head-only) on the right train slice landed at Sharpe 5.60 — clears zero-shot, loses to ARIMA. The wrong checkpoint (end-of-training instead of best-val_loss) on the right recipe and right slice landed at Sharpe 3.46 — does not clear zero-shot. Only the combination produces the canonical row.

**The architectural surface matters as much as the parameter count.** Encoder-only on Chronos-2 small (94 % of 28 million parameters trainable on top of a frozen head) lands at Sharpe 8.20 single-seed. The same encoder-only recipe on TimesFM 2.5 (200 million parameters total) lands at 5.39. Five times more parameters, *worse* result. The mechanism that makes Chronos-2 + encoder-only work is the directional prior in the prediction head — frozen by the recipe, preserved across fine-tuning, and re-paired with an encoder that has been updated on Bitcoin-specific structure. TimesFM's decoder-only architecture does not separate "head" and "encoder" in the same way, and the equivalent operation does not produce the same result. The lift is recipe-architecture-specific.

**The gain is in direction and magnitude calibration, not in point error.** Mean absolute error stays with ARIMA across the entire series. The 07 row's MAE is $10.70 above ARIMA's; the directional accuracy and cumulative return are above every other row. The mechanism that makes Sharpe move is what the experiment's README calls "prediction magnitude calibration" — when the fine-tuned model gets direction right, it predicts the magnitude of the up-move closer to the actual magnitude than ARIMA does, so the long-flat strategy sizes its participation in a way that compounds favourably. The dir_acc-vs-Sharpe split from articles 2 and 3 is the same mechanism in reverse: where the LSTM beat ARIMA on direction and lost on Sharpe because it under-predicted magnitudes, the fine-tuned foundation model beats ARIMA on both direction *and* magnitude calibration.

**Single-seed numbers can mislead.** The on-disk Sharpe of 8.20 is the strongest single-seed result in the series. It sits above the upper bound of the README's five-seed 95 % confidence interval (6.04). The reproducible benchmark is the CI itself — 5.00 to 6.04 — and the lower bound clears 06 zero-shot at 95 % confidence. Citing "Sharpe 8.20" as the result of this experiment is technically accurate and substantively misleading; citing the CI is honest. A repository that only ever shows the best seed it ever drew is a repository that lies; one that shows the CI alongside is one that does not. The methodology lesson generalizes well past this experiment.

## What ARIMA keeps

Mean absolute error, and the point-error crown that comes with it. After seven experiments — a 3-parameter linear baseline, a 31-feature gradient-boosted tree ensemble, two thousand-parameter deep-learning models, a thirty-thousand-parameter Temporal Fusion Transformer, three zero-shot foundation models with 28 to 200 million pretrained parameters, and one fine-tuned foundation model with 26 million trainable parameters on top of 4.7 years of Bitcoin — the lowest MAE on the leaderboard is still ARIMA(3, 1, 3)'s $539.15. The 07 canonical at $549.85 is $10.70 worse. The cluster of MAE values around $539–$571 covers six of seven models in the leaderboard; only the Temporal Fusion Transformer at $891 falls outside it. That cluster is the noise floor of one-step-ahead forecasting on this slice at this cadence — and seven different model classes have now agreed on where it is, within $32.

ARIMA's role across the series has been the *floor* that everything else has to clear. The fine-tuned foundation model clears it on three metrics. The MAE floor stays where it was. Read that the way the series has been recommending — pick the model for the metric you care about. There is no single-row winner across all six leaderboard columns. The 07 row wins three; ARIMA wins two and a half. The LSTM, the Temporal Fusion Transformer, and XGBoost each contribute something specific (RMSE for LSTM and XGBoost; nothing top-of-leaderboard for the Temporal Fusion Transformer) without dominating the others. The series is a structured way to see that.

## Regime caveat

The test slice — `2024-10-01` → `2024-12-01` UTC — is the post-election Bitcoin rally. Every positive Sharpe figure on the leaderboard is *conditional* on a strongly trending up-regime. The fine-tuned foundation model has not been tested out-of-regime; neither have any of the other six experiments. The README's five-seed comparison is *seed* variance, not *regime* variance — running the same recipe on a sideways slice or a down-trending slice would tell whether the directional skill the canonical model shows generalizes beyond the training regimes the 2020–2024 slice contains. The 5.00 CI lower bound is conditional on this regime; an out-of-regime test is the operational follow-up that would turn the 07 row from "skill conditional on regime" into "skill in general."

## Reproduce

```
make 07_finetuned                                                                                                       # canonical: 4.7-yr encoder-only, chronos-2-small
make 07_finetuned CONFIG=experiments/07_finetuned/configs/btc_4h_2020_2024_encoder_only.timesfm.config.yaml             # variant: 4.7-yr encoder-only, timesfm-2.5
make 07_finetuned CONFIG=experiments/07_finetuned/configs/btc_4h_2024.config.yaml                                       # former canonical: 1.6-yr head-only, chronos-2-small
make 07_finetuned_sweep                                                                                                 # 10-row recipe × learning-rate sweep
make 07_finetuned_clean_checkpoints                                                                                     # wipe results/darts_checkpoints/
```

The first writes [`experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/metrics.json`](experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/metrics.json) — the canonical numbers in this article. The sweep currently runs against the 1.6-year sibling slice on disk; a 4.7-year sweep does not yet exist and is on the operational follow-up list. The full per-experiment report at [`docs_ml/reports/07_finetuned.report.md`](docs_ml/reports/07_finetuned.report.md) ships the multi-seed comparison and the full caveat list.

The article series is current as of the canonical `metrics.json` files on disk. It is not closed — the `chronos-2-large` variant config has a filename / `hub_model_name` mismatch that needs fixing, the 4.7-year sweep has not been generated, and no row has been tested out-of-regime. The cross-experiment synthesis at [`docs_ml/reports/XX_global.report.md`](docs_ml/reports/XX_global.report.md) keeps the full leaderboard up to date alongside the operational follow-ups. Thank you for reading.
