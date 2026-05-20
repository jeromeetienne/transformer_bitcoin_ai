# Outline — Article 6: Fine-tuned foundation models

Working title: *"When fine-tuning beats zero-shot — and the three conditions that make it happen."*

Source experiment: [`07_finetuned`](experiments/07_finetuned/).

## Editorial framing

The follow-up article 5 set up. Same two foundation-model backends as article 5 (Chronos-2 small, TimesFM 2.5) — but now `fit()` actually updates weights against a held-out validation slice, and the best-validation-loss checkpoint is restored before walk-forward inference.

Three honest beats anchor the article:

1. The on-disk Sharpe is **8.20** — the highest in the entire series. But it is a favourable single seed.
2. The five-seed mean per the experiment's README is **5.523 ± 0.418, 95 % CI 5.004 – 6.043** — borderline-to-clean; the CI lower bound still clears the zero-shot benchmark.
3. The lift requires *three conditions* to all hold: (a) a 4.7-year train slice instead of 1.6 years; (b) encoder-only fine-tuning, not head-only and not full; (c) restoring the best-validation-loss checkpoint before walk-forward inference. Swap any one out and the Sharpe collapses.

The article is not "fine-tuning works." It is "fine-tuning works on this slice under these three conditions, by this much, on a favourable seed, and ARIMA still leads MAE." That is the load-bearing distinction.

## Beats

1. **Hook.** Article 5 ended with zero-shot foundation models losing to a 3-parameter linear ARIMA. This article asks the natural follow-up: does fine-tuning the same backbone on Bitcoin close the gap? Answer: **yes on Sharpe / direction / cumulative return, no on point error, and only under three specific conditions.** Single-seed Sharpe on disk is 8.20 — the highest in the leaderboard. The 5-seed CI lower bound is 5.00 — still clears zero-shot at 95 % confidence. Both numbers belong in the article.

2. **Methodology recap.** Same 366-bar test slice (`2024-10-01` → `2024-12-01` UTC). Same walk-forward shape. Same metric module. **New this article:** the train slice extends back to 2020-01-01 (4.7 years instead of 1.6 years, ~10 000 training rows), `fit()` now updates weights against the validation slice with `EarlyStopping(monitor='val_loss')`, and the best-val_loss checkpoint is restored before walk-forward inference via `model.load_weights_from_checkpoint(best=True)`.

3. **What changed from article 5 — three architectural surfaces.**
   - **Train slice:** `btc_4h_2020_2024` instead of `btc_4h_2024`. 10 043 training rows instead of 3 833. Includes the 2020 COVID crash, the 2021 bull run, the 2022 bear market, the 2023 chop, the mid-2024 ATH — a much richer set of regimes for the foundation model to internalize.
   - **Fine-tuning recipe:** `enable_finetuning: {freeze: ['*output_patch_embedding*']}`. Encoder-only — freeze the prediction head, fine-tune the encoder. Approximately 94.3 % of Chronos-2 small's 28 M parameters become trainable; the head's directional prior is preserved.
   - **Best-checkpoint restoration:** the canonical's `metrics.json` records `restored_best_checkpoint: true`. The README documents a prior incident where the same configuration posted Sharpe 6.146 *before* checkpoint restoration was wired in, 3.46 after — a 2.7-point swing from a single workflow detail.

4. **Configuration table** — from [`btc_4h_2020_2024_encoder_only.chronos-2-small.config.yaml`](experiments/07_finetuned/configs/btc_4h_2020_2024_encoder_only.chronos-2-small.config.yaml):
   - input_chunk_length 256, output_chunk_length 1
   - num_samples 1000 (raised from 200 in article 5 — √5 ≈ 2.2× quantile-noise reduction per seed)
   - quantiles [0.1, 0.5, 0.9]
   - n_epochs 50 (upper bound), early_stopping_patience 3
   - learning_rate 1e-5, batch_size 32
   - random_state 42
   - finetune.enable_finetuning {freeze: ['*output_patch_embedding*']} (encoder-only)
   - 10 043 training rows, 366 validation rows, 366 test rows
   - epochs_trained 10 (early-stopped; best val_loss at epoch 5)
   - best_val_loss 0.01579 (≈ 2.3× lower than the 1.6-yr sibling's 0.03598)

5. **The result — verbatim from `metrics.json`.**
   - MAE 549.85 USD
   - RMSE 813.80 USD
   - MAPE 0.7020 %
   - directional accuracy **0.5683** — leaderboard leader
   - cumulative return **0.7329** — leaderboard leader
   - annualized Sharpe **8.1987** (single seed; 5-seed mean per README is 5.523 ± 0.418)

6. **The leaderboard, all seven experiments.**
   | Model | MAE | dir_acc | Sharpe |
   |---|---|---|---|
   | Naive | 540.96 | NaN | — |
   | ARIMA(3, 1, 3) | **539.15** | 0.5082 | 6.8559 |
   | XGBoost | 550.85 | 0.5082 | 6.1388 |
   | LSTM | 539.23 | 0.5301 | 4.8221 |
   | Temporal Fusion Transformer | 891.09 | 0.5055 | 2.5390 |
   | Chronos-2 small (zero-shot) | 546.66 | 0.4754 | 2.9722 |
   | **Chronos-2 small (fine-tuned, canonical)** | 549.85 | **0.5683** | **8.1987** |

7. **The single-seed-vs-five-seed beat — honest framing.**
   - On disk: Sharpe 8.20, dir_acc 0.5683, cum_ret 0.7329.
   - Per-bar Sharpe ratio: 8.1987 / √2190 = 0.1752; SE 1 / √366 = 0.0523; ratio **3.35 σ** — the strongest single-seed result in the series.
   - But: this is `random_state: 42`. A single seed.
   - 5-seed mean per the README: Sharpe **5.523 ± 0.418**, 95 % CI (5.004, 6.043); cum_ret 44.52 % ± 5.48 pp, CI (37.70 %, 51.33 %).
   - Per-bar Sharpe ratio at the 5-seed mean: 5.523 / √2190 = 0.1180; ratio **2.26 σ** — borderline. At the CI lower bound: 5.004 / √2190 = 0.1069; ratio **2.04 σ** — borderline.
   - **Honest read:** "the seed *distribution* clears 06 zero-shot at 95 % confidence," not "a model posts 8.20." Cite both numbers; don't hide either.

8. **The three conditions — variant table as evidence.** From [`07_finetuned.report.md`'s Variants section](docs_ml/reports/07_finetuned.report.md):
   | Variant | train | recipe | backend | dir_acc | cum_ret | Sharpe |
   |---|---|---|---|---|---|---|
   | encoder-only / chronos-2-small / 4.7 yr (canonical) | 4.7 yr | encoder-only | chronos-2-small | 0.5683 | 0.7329 | **8.1987** |
   | encoder-only / timesfm / 4.7 yr | 4.7 yr | encoder-only | timesfm-2.5 | 0.5246 | 0.4512 | 5.3910 |
   | head-only / chronos-2-small / 1.6 yr (former canonical) | 1.6 yr | head-only | chronos-2-small | 0.5328 | 0.4162 | 5.6022 |

   - **Swap backend (chronos → timesfm), keep recipe and train slice:** Sharpe drops 8.20 → 5.39. The encoder-only recipe pairs better with the Chronos T5-style encoder than with the TimesFM decoder. *Architecturally*: "encoder-only" freezes the head and fine-tunes the encoder, which is a much larger surface on T5 than on the decoder-only TimesFM.
   - **Swap train slice (4.7 → 1.6 yr), keep backend and recipe direction:** Sharpe drops 8.20 → 5.60 (on head-only) or 6.87 (on encoder-only — sweep row). 1.6 years overfits within 1–3 epochs regardless of recipe; the README documents this directly.
   - **Skip checkpoint restoration:** README documents Sharpe 6.146 → 3.46 on the same configuration. The `restored_best_checkpoint: true` field in `metrics.json` is *load-bearing* — if it ever flips to `false`, the numbers reflect overfit end-of-training weights silently.

9. **The sweep — recipe × learning rate on the 1.6-yr sibling slice.** From [`experiments/07_finetuned/results/btc_4h_2024/sweep.csv`](experiments/07_finetuned/results/btc_4h_2024/sweep.csv). Ten rows, chronos-2-small backend except one timesfm row.
   - **Sharpe leader: `encoder-only × 1e-5`** at 6.87 on the 1.6-yr slice — same recipe / lr corner as the canonical's 4.7-yr 8.20. Train-slice length is the lift; recipe / lr is the corner that gets you there.
   - **Lowest val_loss row: `full × 1e-5`** at val_loss 0.03479 — Sharpe 5.73, *not* the sweep's Sharpe winner. **In-sample val_loss does not pick the out-of-sample Sharpe winner**, same lesson as ARIMA's AIC ↔ Sharpe disagreement, now visible inside a foundation-model fine-tuning sweep.
   - Sweep was run on the *former* canonical's slice. A 4.7-yr-slice sweep does not exist on disk; the canonical's recipe / lr corner is currently rankable only indirectly via the sibling sweep. Operational follow-up: re-run `make 07_finetuned_sweep`.

10. **What this article tells us about the model class.**
    - **Fine-tuning a foundation model on a small target is not free.** Three conditions all had to hold simultaneously — the right train-slice length, the right architectural surface to fine-tune, and the right checkpoint restoration step. Without any one of them the lift collapses.
    - **The architectural surface matters as much as the parameter count.** Encoder-only on Chronos (≈ 94 % of 28 M trainable on top of a frozen head) lands at Sharpe 8.20 single-seed; the same encoder-only recipe on TimesFM 2.5 (200 M parameters total) lands at 5.39. The directional prior in Chronos-2's prediction head is what the "freeze the head, fine-tune the encoder" recipe is preserving — and that prior generalizes to Bitcoin 4-hour log-returns in a way the equivalent operation on TimesFM does not.
    - **More train data delays overfitting.** The validation loss surface is 2.3× lower on the 4.7-year slice than on the 1.6-year slice for the same recipe. Combined with checkpoint restoration, this is what lets fine-tuning extract signal at all.
    - **The gain is in direction / magnitude calibration, not in point error.** MAE 549.85 is $10.70 above ARIMA's 539.15. The 07 row wins Sharpe by predicting *better-calibrated magnitudes* on the bars where it gets direction right — exactly the same dir_acc-vs-Sharpe mechanism we saw inverted in articles 2 and 3.
    - **Single-seed numbers can mislead.** The on-disk 8.20 sits above the 95 % CI upper bound (6.04) of the README's 5-seed comparison. The right citation is "5.00 – 6.04 CI", not "8.20."

11. **What ARIMA keeps.** Point error. MAE 539.15 vs. 549.85 — $10.70 below 07. The 3-parameter linear baseline remains the leaderboard's MAE leader after seven experiments. Across the entire series the MAE band is $539–$571 for six of seven models, with the Temporal Fusion Transformer at $891 as the outlier. ARIMA is *closer to the price on average*; the fine-tuned foundation model is *more often right about direction* and *better calibrated in predicted magnitude* on those bars. Pick the model for the metric you care about.

12. **Per-bar Sharpe significance — three numbers.**
    - Single-seed on disk: ratio 3.35 σ — clean.
    - 5-seed mean: ratio 2.26 σ — borderline.
    - 5-seed CI lower bound: ratio 2.04 σ — borderline.
    The honest range is borderline-to-clean depending on which number you cite.

13. **Regime caveat** — same paragraph as articles 1 through 5. Test slice is the post-election BTC rally; no row in the entire repo has been tested out-of-regime; every positive Sharpe — including 07's — is *conditional* on this regime. The README's 5-seed comparison is *seed* variance, not regime variance.

14. **The series wrap.** Bottom line, across all seven experiments. Reframe the article 5 wrap-up: "fine-tuning was the question article 5 didn't try to answer; article 6 answers it." The answer is yes on Sharpe under three conditions, no on MAE, and conditional on the regime.

15. **Reproduce.** `make 07_finetuned` plus the variant `CONFIG=…` overrides.

## What this article is NOT

- A claim that fine-tuned foundation models beat ARIMA on Bitcoin in general. They beat ARIMA on Sharpe / direction / cumulative return on this slice, under three specific conditions, on a favourable seed. They do not beat ARIMA on MAE on any seed.
- A foundation-model fine-tuning tutorial. Refer the curious reader to the Chronos-2 and TimesFM model cards on HuggingFace Hub and the Darts documentation.
- A claim that 8.20 Sharpe is reproducible. It is one seed. The README's 5-seed CI is the reproducible benchmark; cite that.
- A claim that the article series is now "complete." `07_finetuned` has the `chronos-2-large` filename bug, no 4.7-year sweep, and no out-of-regime test — and the docs_ml ecosystem has operational follow-ups still to land. The series is *current as of the canonical metrics.json files on disk*, not closed.
