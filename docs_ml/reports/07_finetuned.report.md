# Report — `07_finetuned`

**Run date:** 2026-05-20
**Status:** completed (canonical trial: `btc_4h_2020_2024_encoder_only.chronos-2-small` — encoder-only fine-tuning on 4.7-yr train slice; recipe × learning-rate sweep on the 1.6-yr sibling slice; two more variants on disk)

## What this experiment is

Fine-tuning follow-up to [06_pretrained](06_pretrained.report.md): same two foundation-model backends (Amazon's Chronos-2 encoder-only T5-style; Google's TimesFM 2.5 decoder-only patch-transformer), but `enable_finetuning` is set so `fit()` actually updates weights against a held-out validation slice with `EarlyStopping(monitor='val_loss')`. The pipeline **restores the best-val_loss checkpoint** via `model.load_weights_from_checkpoint(best=True)` before walk-forward inference — without that step, val_loss divergence within 1–4 epochs silently feeds overfit end-of-training weights into the test slice and produces misleading numbers (the README documents one such artefact: the same configuration posted Sharpe 6.146 before checkpoint restoration was wired in, 3.46 after).

```
HF Hub weights ──► darts FoundationModel
                       │
                       ▼  fit(train, val_series=val)  ──►  EarlyStopping(monitor='val_loss')
                       │                                   ModelCheckpoint(best=val_loss)
                       ▼
                   load_weights_from_checkpoint(best=True)
                       │
                       ▼
                   historical_forecasts(retrain=False, last_points_only=True, num_samples=1000)
                       │
                       └──► quantile(0.1) / 0.5 / 0.9 ──► reconstruct close
```

Target and price reconstruction unchanged from 04 / 05 / 06: `r_T = log(close_T / close_{T-1})`, `close_pred = close_{T-1} * exp(r_q0.5)`. Library: **darts** (`Chronos2Model`, `TimesFM2p5Model`) on PyTorch Lightning. Walk-forward shape matches 02 / 04 / 05 / 06 (`retrain=False`). See [experiments/07_finetuned/README.md](../../experiments/07_finetuned/README.md) for the full narrative — including the *three things that had to happen* for fine-tuning to outperform zero-shot at significance: (a) extend the train slice from 1.6 to 4.7 years; (b) freeze the prediction head and fine-tune the encoder; (c) restore the best-val_loss checkpoint.

## Configuration

From [experiments/07_finetuned/configs/btc_4h_2020_2024_encoder_only.chronos-2-small.config.yaml](../../experiments/07_finetuned/configs/btc_4h_2020_2024_encoder_only.chronos-2-small.config.yaml) (the canonical trial; matches the Makefile's default `CONFIG ?=`):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `4h` |
| dataset | `btc_4h_2020_2024` ([configs/datasets/btc_4h_2020_2024.dataset.yaml](../../configs/datasets/btc_4h_2020_2024.dataset.yaml)) |
| train window | `2020-01-01` → `2024-08-01` UTC (≈ 4.7 years, ~10 000 bars) |
| val window | `2024-08-01` → `2024-10-01` UTC (early-stopping signal + best-checkpoint selector) |
| test window | `2024-10-01` → `2024-12-01` UTC (unchanged across all 07 / 06 / 04 / 05 / 03 / 02 / 01 canonical rows) |
| backend | `chronos` |
| hub_model_name | `autogluon/chronos-2-small` (28 M parameters) |
| finetune.enable_finetuning | `{freeze: ['*output_patch_embedding*']}` (encoder-only, ≈ 94.3 % trainable) |
| training.early_stopping_patience | `3` |
| model.input_chunk_length | `256` (≈ 42 days at 4h) |
| model.output_chunk_length | `1` |
| model.num_samples | `1000` (raised from 200 to tighten per-seed quantile noise) |
| model.quantiles | `[0.1, 0.5, 0.9]` |
| model.n_epochs | `50` (upper bound) |
| model.batch_size | `32` |
| model.learning_rate | `1e-5` |
| model.random_state | `42` |

Train: **10 043** rows. Val: **366**. Test: **366**. `epochs_trained: 10` (EarlyStopping triggered; best val_loss at epoch 5). `best_val_loss: 0.01579` (≈ 2.3× lower than the 1.6-yr sibling's 0.03598 — more train data → lower validation surface for the same recipe). `restored_best_checkpoint: true`.

## Results — single fit (chronos-2-small, encoder-only fine-tuning, 4.7-year train slice)

From [experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/metrics.json](../../experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/metrics.json):

| Metric | Value |
|---|---|
| MAE | 549.85 USD |
| RMSE | **813.80** USD |
| MAPE | 0.7020 % |
| Directional accuracy | **0.5683** |
| Cumulative return | **0.7329** |
| Annualized Sharpe | **8.1987** |

**Single-seed caveat upfront:** the README ships a 5-seed comparison for this exact recipe; the seed-mean Sharpe is **5.523 ± 0.418 (95 % CI 5.004 – 6.043)** and seed-mean cum_ret is **44.52 % ± 5.48 pp (CI 37.70 – 51.33 %)**. The canonical row currently on disk (`random_state: 42`, Sharpe 8.20, cum_ret 0.733) sits *above* the upper CI bound — it is a favourable seed, not the reproducible centre of the distribution. Read the headline as evidence the seed *distribution* clears 06 zero-shot at 95 % confidence, **not** as a single reproducible 8.20 number.

## Cross-experiment comparison

4h-slice leaderboard. Same `2024-10-01` → `2024-12-01` test window across all rows; numbers verbatim from each experiment's canonical `metrics.json` on disk. Older per-experiment reports cite a prior slice (rows_test ≈ 401) and have not been re-edited per the doc's cross-report-consistency rule.

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| [01_baseline](01_baseline.report.md) | 540.96 | 813.14 | 0.6922 % | NaN | — | — |
| [02_arima (3, 1, 3)](02_arima.report.md) | **539.15** | 808.79 | **0.6903 %** | 0.5082 | 0.5269 | 6.8559 |
| [03_xgboost (n_feat=31)](03_xgboost.report.md) | 550.85 | 808.65 | 0.7096 % | 0.5082 | 0.4149 | 6.1388 |
| [04_lstm](04_lstm.report.md) | 539.23 | **808.29** | 0.6909 % | 0.5301 | 0.4284 | 4.8221 |
| [05_transformer](05_transformer.report.md) | 891.09 | 1 271.26 | 1.1499 % | 0.5055 | 0.1219 | 2.5390 |
| [06_pretrained (chronos-2 small, zero-shot)](06_pretrained.report.md) | 546.66 | 817.81 | 0.7003 % | 0.4754 | 0.2225 | 2.9722 |
| **07_finetuned (encoder-only, chronos-2 small, 4.7-yr)** | **549.85** | **813.80** | **0.7020 %** | **0.5683** | **0.7329** | **8.1987** |

**07 canonical now leads dir_acc, cum_ret, and Sharpe.** dir_acc 0.5683 is 0.038 above the next best (04_lstm 0.5301) — about 14 additional correctly-directed bars out of 366. cum_ret 0.7329 is 0.206 above ARIMA(3, 1, 3)'s 0.5269 and 73 % above 06 zero-shot. Sharpe 8.20 is the first row to clear ARIMA's long-standing 6.86 lead. ARIMA still leads MAE ($539.15, $10.70 below 07) and MAPE; 04_lstm still leads RMSE by a hair ($808.29 vs. $813.80). **Read the Sharpe lead through the 5-seed CI lower bound** (5.00) rather than the on-disk single-seed value (8.20) — even the lower bound clears 06 zero-shot (4.74) and the lower bound's per-bar ratio is ≈ 2.04 σ, borderline but real.

## Variants

Three additional configs in `experiments/07_finetuned/configs/` produce results on disk on the same test window. Two axes are exercised: **train-slice length** (`btc_4h_2024` 1.6 yr vs. `btc_4h_2020_2024` 4.7 yr) and **fine-tuning recipe** (head-only `{unfreeze: ['*output_patch_embedding*']}` vs. encoder-only `{freeze: ['*output_patch_embedding*']}`). The 1.6-yr head-only variant was the prior canonical, now demoted to a sibling row.

| Variant | train | recipe | backend | num_samples | MAE | RMSE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|---|---|
| `btc_4h_2020_2024_encoder_only.chronos-2-small` (canonical) | 4.7 yr | encoder-only | chronos-2-small | 1000 | 549.85 | **813.80** | **0.5683** | **0.7329** | **8.1987** |
| `btc_4h_2020_2024_encoder_only.timesfm` | 4.7 yr | encoder-only | timesfm-2.5 | 1000 | **543.92** | 815.39 | 0.5246 | 0.4512 | 5.3910 |
| `btc_4h_2020_2024_encoder_only.chronos-2-large` ⚠ | 4.7 yr | encoder-only | chronos-2-small (filename misnamed) | 1000 | 554.55 | 824.10 | 0.5437 | 0.5199 | 6.0544 |
| `btc_4h_2024` (former canonical) | 1.6 yr | head-only | chronos-2-small | 200 | 542.98 | 815.12 | 0.5328 | 0.4162 | 5.6022 |

⚠ **Filename / `hub_model_name` mismatch.** [`btc_4h_2020_2024_encoder_only.chronos-2-large.config.yaml`](../../experiments/07_finetuned/configs/btc_4h_2020_2024_encoder_only.chronos-2-large.config.yaml) is named "large" but sets `hub_model_name: autogluon/chronos-2-small`. On-disk metrics confirm: this row is a chronos-small re-run, not a true `amazon/chronos-2` (120 M) run. The 4.7-yr × encoder-only × 120 M cell in the head-to-head is therefore *missing*; fixing the config and re-running fills it.

Three observations:

- **Backend matters as much as recipe.** Holding train slice (4.7 yr) and recipe (encoder-only) constant, swapping chronos-2-small for timesfm drops Sharpe from 8.20 to 5.39 and dir_acc from 0.568 to 0.525. The "encoder-only" recipe pairs better with the Chronos T5-style encoder than with the TimesFM decoder — consistent with what *encoder-only* means architecturally: freezing the *decoding head* and fine-tuning the *encoder*, which is a much larger architectural surface on T5 than on the decoder-only TimesFM.
- **Train-slice length is load-bearing.** Same backend (chronos-2-small), same architectural surface (head-only ⇆ encoder-only differs by recipe, but compare the canonical 4.7-yr encoder-only against the former-canonical 1.6-yr head-only): Sharpe 8.20 vs. 5.60 — a 46 % lift; dir_acc 0.568 vs. 0.533. The headline finding requires *both* 4.7 yr of train data and the encoder-only recipe; the 1.6-yr slice produces a model that overfits within 1–3 epochs regardless of recipe (the README documents this directly).
- **The TimesFM variant on the extended slice (Sharpe 5.39) is roughly tied with the chronos-2-small head-only 1.6-yr variant (Sharpe 5.60).** Adding 2.9× train data to TimesFM lands at about the same skill ceiling as freezing the head and using less data on Chronos — different paths to the same mediocre plateau. The Chronos + encoder-only + 4.7-yr corner is genuinely doing something the others aren't.

## Sweep — recipe × learning rate on the 1.6-yr sibling slice

From [experiments/07_finetuned/results/btc_4h_2024/sweep.csv](../../experiments/07_finetuned/results/btc_4h_2024/sweep.csv). **Sweep was run against the former canonical (`btc_4h_2024`, 1.6-yr train slice), not the current canonical's 4.7-yr slice** — its rankings tell you which recipe / lr wins on 1.6 yr of train data, *not* whether those rankings hold on 4.7 yr. Test window is identical, so the metrics live on the same scale. 10 rows, chronos-2-small backend except one timesfm row, `input_chunk_length = 256`, `n_epochs = 20` max. Sorted by annualized Sharpe descending; **bold the column leaders**:

| backend | recipe | lr | epochs | best_val_loss | MAE | RMSE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|---|---|
| chronos | encoder-only | 1e-5 | 7 | 0.03505 | 541.93 | 812.46 | **0.5464** | 0.5520 | **6.8664** |
| chronos | head-only | 1e-4 | 9 | 0.03555 | **540.46** | 811.65 | 0.5383 | **0.5626** | 6.7701 |
| chronos | full | 3e-5 | 7 | 0.03481 | 543.36 | 811.37 | 0.5355 | 0.5938 | 6.3428 |
| chronos | head-only | 3e-5 | 9 | 0.03580 | 541.36 | 813.35 | 0.5410 | 0.4568 | 5.9435 |
| chronos | full | 1e-4 | 7 | 0.03550 | 547.91 | 813.37 | 0.5246 | 0.5373 | 5.9431 |
| chronos | full | 1e-5 | 9 | **0.03479** | 540.22 | **806.89** | 0.5164 | 0.4099 | 5.7251 |
| chronos | head-only | 1e-5 | 9 | 0.03598 | 542.98 | 815.12 | 0.5328 | 0.4162 | 5.6022 |
| chronos | encoder-only | 3e-5 | 8 | 0.03536 | 541.39 | 811.96 | 0.5219 | 0.3126 | 5.3956 |
| chronos | encoder-only | 1e-4 | 9 | 0.03498 | 543.53 | 809.24 | 0.5137 | 0.4552 | 5.0740 |
| timesfm | full | 1e-5 | 6 | 0.03627 | 547.54 | 820.88 | 0.5000 | 0.2792 | 4.4660 |

**Sharpe leader is `encoder-only × 1e-5`** (6.87) on the 1.6-yr slice — the same recipe / lr direction that wins on the canonical 4.7-yr slice (8.20). The lift from 1.6 → 4.7 yr at that fixed (recipe, lr) corner is +1.33 Sharpe, +0.022 dir_acc, +0.18 cum_ret — train-slice length pays off cleanly when the recipe is right.

The *lowest val_loss* row (`full × 1e-5`, val_loss 0.03479) posts Sharpe 5.73, **not** the Sharpe leader. The Sharpe leader posts val_loss 0.03505. **In-sample val_loss does not pick the out-of-sample Sharpe winner** — same family of disagreement as ARIMA's AIC ↔ Sharpe lesson, now visible inside a single foundation-model fine-tuning sweep. If you pick the fine-tuning recipe by val_loss alone you miss the trading-metric winner.

A 4.7-yr-slice sweep does not exist on disk; the canonical's row in this sweep is the *former* canonical (`head-only × 1e-5`, Sharpe 5.60). Re-running `make 07_finetuned_sweep` now (with the Makefile pointing at the 4.7-yr config) would produce a sweep that ranks recipes on the canonical slice — currently a real gap.

## Interpretation

1. **The 4h leaderboard's Sharpe / dir_acc / cum_ret leadership all sit on the 07 canonical row.** Single-seed Sharpe 8.20 / dir_acc 0.5683 / cum_ret 0.7329 are loud, but the load-bearing number is the README's 5-seed CI lower bound: **Sharpe lower bound 5.004 vs. 06 zero-shot mean 4.743 — the seed distribution clears zero-shot at 95 % confidence**. Read the canonical as "fine-tuning beats zero-shot at significance on this recipe" rather than "a model posts 8.20."
2. **Per-bar Sharpe ratio depends on which number you cite.** Single-seed on-disk: `8.1987 / √2190 = 0.1752`; SE = `1 / √366 = 0.0523`; ratio **3.35 σ** — clean. 5-seed mean: `5.523 / √2190 = 0.1180`; ratio **2.26 σ** — borderline. 5-seed CI lower bound: `5.004 / √2190 = 0.1069`; ratio **≈ 2.04 σ**. The honest range is *borderline to clean*, not "the strongest result in the series at 3.35 σ" — the prior canonical (head-only, 1.6 yr) at 2.29 σ was in the same neighbourhood.
3. **The canonical now embodies all three of the load-bearing conditions** the README enumerates: 4.7-yr train slice (vs. 1.6 yr in the sibling head-only variant), encoder-only recipe (`freeze: ['*output_patch_embedding*']` — keep the head static, fine-tune the encoder), and `restored_best_checkpoint: true` confirmed in `metrics.json`. Swap any one out and the lift collapses: the 1.6-yr sibling at the same recipe / lr (sweep's `encoder-only × 1e-5`) drops to Sharpe 6.87; the 4.7-yr TimesFM variant drops to 5.39; an inadvertent `restored_best_checkpoint: false` would (per the README's documented incident) flip honest numbers to overfit ones.
4. **MAE leadership stays with ARIMA(3, 1, 3).** The canonical posts MAE 549.85, $10.70 above ARIMA's 539.15. Encoder-only fine-tuning trades small-MAE precision for direction-getting-right and predicted-magnitude calibration — exactly the trade the README's "What this tells us" section names: the gain is in *prediction magnitude calibration*, not point-error reduction. The sized-distribution-on-correct-signal property shows up cleanly on Sharpe and cum_ret.
5. **val_loss does not rank Sharpe.** Sweep's lowest `best_val_loss` is on `full × 1e-5` (Sharpe 5.73); the Sharpe leader posts a higher val_loss. Inside the canonical itself, `best_val_loss: 0.01579` is 2.3× *lower* than the 1.6-yr sibling's 0.03598, but Sharpe is also higher — so the in-sample / out-of-sample disagreement is recipe-level, not slice-level. Selecting recipes by val_loss alone misses the trading-metric winner.
6. **NaN by design — same surface as earlier experiments.** 07 itself produces no NaN cells (every fine-tuned variant expresses direction); the relevant rule still holds for 01's NaN dir_acc and the global report's NaN-by-design section, per [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py)'s "no opinion expressed" semantics.

### Bottom line

**Fine-tuning beats zero-shot at significance on the right recipe and enough data — and 07's canonical row is that combination.** Headline numbers (single-seed on disk, `random_state: 42`): Sharpe **8.1987** / dir_acc **0.5683** / cum_ret **0.7329** / MAE 549.85 / RMSE 813.80. Headline numbers (5-seed mean, the README's reproducible benchmark): Sharpe **5.523 ± 0.418** (CI 5.004 – 6.043) / cum_ret **44.52 % ± 5.48 pp** (CI 37.70 – 51.33 %). The CI lower bound clears 06 zero-shot's mean (4.743) on Sharpe; every one of the README's five seeds independently beats 06 on both Sharpe and cum_ret. Per-bar Sharpe ratio range: **2.04 σ (CI lower bound) to 3.35 σ (single seed on disk)**. Test slice covers `2024-10-01` → `2024-12-01` (post-election BTC rally, 366 bars at 4h); no row has been tested out-of-regime, and every positive Sharpe — including the canonical — is conditional on this regime.

## Caveats

- **Single deterministic seed in `metrics.json`.** `random_state = 42` is fixed; the README ships the 5-seed CI separately. Read the canonical's Sharpe / cum_ret through the 5-seed CI, not the on-disk single-seed value.
- **`chronos-2-large` variant filename is misleading.** Filename promises `amazon/chronos-2` (120 M); config and metrics deliver `autogluon/chronos-2-small` (28 M). The leaderboard does not currently contain a true chronos-2-large fine-tuned row.
- **`num_samples` differs across variants.** Canonical and extended siblings use 1000; the 1.6-yr head-only sibling uses 200. Five times more stochastic draws tightens per-seed quantile-estimate noise on the canonical (~√5 reduction). Not the main driver of the Sharpe lift — recipe and train-slice length are — but a real component.
- **Sweep is on the wrong slice for the current canonical.** The 10-row sweep was run on the 1.6-yr sibling slice (former canonical). No 4.7-yr sweep exists on disk; the canonical's recipe / lr corner can only be ranked indirectly via the 1.6-yr sibling sweep.
- **timesfm sweep coverage is one row.** `full × 1e-5` only; encoder-only and head-only timesfm sweep rows are missing.
- **Walk-forward without re-estimation.** Weights frozen across the entire test window after fine-tuning. Matches 02 / 04 / 05 / 06.
- **Univariate only.** TimesFM 2.5 doesn't accept covariates; Chronos-2 kept univariate so the head-to-head stays fair on identical inputs.
- `output_chunk_length: 1`. One-step-ahead only.
- **Checkpoint restoration is load-bearing.** `restored_best_checkpoint: true` in every current `metrics.json` confirms inference used best-val_loss weights. If this ever flips to `false`, metrics will silently reflect overfit end-of-training weights — investigate before believing them. Documented prior incident: same configuration posted Sharpe 6.146 before this was wired in, 3.46 after.
- **Single test regime.** Post-election BTC rally; no row has been tested out-of-regime. Every positive Sharpe — including the canonical — is conditional on this regime.
- Long/flat strategy on the median (q50) forecast, no shorting, no transaction costs. q10 / q90 bands are stored in `predictions.parquet` but not used in the strategy.

## Files produced

- [experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/metrics.json](../../experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/metrics.json) (canonical)
- [experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/predictions.parquet](../../experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/predictions.parquet)
- [experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/plot.png](../../experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/plot.png)
- [experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/training_curves.png](../../experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-small/training_curves.png)
- Sweep artefact lives in the **sibling** results dir, not the canonical: [experiments/07_finetuned/results/btc_4h_2024/sweep.csv](../../experiments/07_finetuned/results/btc_4h_2024/sweep.csv)
- Variant artefacts (same set minus `sweep.csv`): [`btc_4h_2020_2024_encoder_only.chronos-2-large/`](../../experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.chronos-2-large/), [`btc_4h_2020_2024_encoder_only.timesfm/`](../../experiments/07_finetuned/results/btc_4h_2020_2024_encoder_only.timesfm/), [`btc_4h_2024/`](../../experiments/07_finetuned/results/btc_4h_2024/) (former canonical)
- Transient checkpoints under `experiments/07_finetuned/results/darts_checkpoints/` (wipe via `make 07_finetuned_clean_checkpoints`)

## How to reproduce

```
make 07_finetuned                                                                                                       # canonical: 4.7-yr encoder-only, chronos-2-small
make 07_finetuned CONFIG=experiments/07_finetuned/configs/btc_4h_2020_2024_encoder_only.timesfm.config.yaml             # 4.7-yr encoder-only, timesfm-2.5
make 07_finetuned CONFIG=experiments/07_finetuned/configs/btc_4h_2020_2024_encoder_only.chronos-2-large.config.yaml     # filename misnamed — see Variants ⚠ note
make 07_finetuned CONFIG=experiments/07_finetuned/configs/btc_4h_2024.config.yaml                                       # 1.6-yr head-only, chronos-2-small (former canonical)
make 07_finetuned CONFIG=experiments/07_finetuned/configs/btc_4h_2024_encoder_only.config.yaml                          # 1.6-yr encoder-only (no results dir on disk yet)
make 07_finetuned_sweep                                                                                                 # 10-row recipe × lr sweep — currently runs against $(CONFIG), i.e. the canonical 4.7-yr slice if not overridden
make 07_finetuned_clean_checkpoints                                                                                     # wipe results/darts_checkpoints/
```

(The `CONFIG=` override is required for any non-canonical variant.)
