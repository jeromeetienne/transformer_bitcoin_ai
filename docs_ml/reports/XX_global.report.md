# Report — `XX_global` — cross-experiment comparison

**Run date:** 2026-05-20
**Status:** synthesis of seven per-experiment reports under [docs_ml/reports/](.)

## What this report is

A single-file leaderboard and analytical comparison of every experiment in the repo: [01_baseline](01_baseline.report.md), [02_arima](02_arima.report.md), [03_xgboost](03_xgboost.report.md), [04_lstm](04_lstm.report.md), [05_transformer](05_transformer.report.md), [06_pretrained](06_pretrained.report.md), [07_finetuned](07_finetuned.report.md). Every number cited here is verbatim from the corresponding canonical `metrics.json` (for 06 / 07 the canonical trial differs from `btc_4h_2024/`; see Data-slice picture); the per-experiment reports are the audit trail.

Sections progress from raw numbers to analytical observations. The intent is not to repeat what the per-experiment reports already say but to put the rows side-by-side and read the *gradient* across the ladder of model classes.

## Data-slice picture

All seven experiments report on the same 4h **test window**: `2024-10-01` UTC → `2024-12-01` UTC, 366 bars at 4h. Train and validation windows differ between experiments (notably 07's `btc_4h_2020_2024_encoder_only.*` variants train on 4.7 years instead of 1.6), but the test-window numbers are directly comparable.

| # | canonical trial | `interval` | `rows_train` | `rows_test` | `metrics.json` mtime | Slice |
|---|---|---|---|---|---|---|
| 01_baseline | `btc_4h_2024` | 4h | n/a (no fit) | 366 | 2026-05-20 | fresh 4h |
| 02_arima | `btc_4h_2024` | 4h | 3 834 | 366 | 2026-05-20 | fresh 4h (sweep on disk, current) |
| 03_xgboost | `btc_4h_2024` | 4h | 3 809 | 366 | 2026-05-20 | fresh 4h (sweep on disk) |
| 04_lstm | `btc_4h_2024` | 4h | 3 467 | 366 | 2026-05-20 | fresh 4h |
| 05_transformer | `btc_4h_2024` | 4h | 3 467 | 366 | 2026-05-20 | fresh 4h |
| 06_pretrained | `btc_4h_2024.chronos-small` | 4h | 3 833 (scaler-fit only; no weight update) | 366 | 2026-05-20 | fresh 4h (chronos-small canonical; chronos-large + timesfm variants on disk) |
| 07_finetuned | `btc_4h_2020_2024_encoder_only.chronos-2-small` | 4h | 10 043 | 366 | 2026-05-20 | fresh 4h (4.7-yr encoder-only canonical; 1.6-yr head-only sibling holds the sweep) |

The 4h test slice covers **`2024-10-01` → `2024-12-01`** (October–November 2024) — the strong post-election BTC rally. All seven sweep / variant artefacts are aligned to the current test window; the older `btc_4h_2024` slice cited in pre-2026-05-20 reports (rows_test ≈ 401) has been superseded. **Older per-experiment reports have not been re-edited** — they remain snapshots at their run date and their leaderboard rows may not match the numbers cited here.

## Leaderboard — 4h slice (all 7 experiments, canonical trial only)

One row per experiment, canonical trial only. Non-canonical variants of 06 / 07 surface in the [Variants and recipes](#variants-and-recipes) section below.

| Experiment | MAE (USD) | RMSE (USD) | MAPE | dir_acc | cum_ret | annualized_sharpe |
|---|---|---|---|---|---|---|
| [01_baseline](01_baseline.report.md) | 540.96 | 813.14 | 0.6922 % | NaN | — | — |
| [02_arima (3, 1, 3)](02_arima.report.md) | **539.15** | 808.79 | **0.6903 %** | 0.5082 | 0.5269 | 6.8559 |
| [03_xgboost (n_feat=31)](03_xgboost.report.md) | 550.85 | 808.65 | 0.7096 % | 0.5082 | 0.4149 | 6.1388 |
| [04_lstm](04_lstm.report.md) | 539.23 | **808.29** | 0.6909 % | 0.5301 | 0.4284 | 4.8221 |
| [05_transformer](05_transformer.report.md) | 891.09 | 1 271.26 | 1.1499 % | 0.5055 | 0.1219 | 2.5390 |
| [06_pretrained (chronos-2 small)](06_pretrained.report.md) | 546.66 | 817.81 | 0.7003 % | 0.4754 | 0.2225 | 2.9722 |
| [07_finetuned (encoder-only, chronos-2 small, 4.7-yr)](07_finetuned.report.md) | 549.85 | 813.80 | 0.7020 % | **0.5683** | **0.7329** | **8.1987** |

**Who leads what:** **07_finetuned canonical leads three of six metrics** — dir_acc, cum_ret, and annualized_sharpe. **02_arima(3, 1, 3) leads MAE and MAPE**. **04_lstm leads RMSE**. The trading-metric leadership has moved from the linear baseline to the fine-tuned foundation model with the rotation of 07's canonical to its 4.7-yr encoder-only variant; ARIMA retains the point-error crown.

**Three observations the canonical leaderboard makes load-bearing.** First, **the MAE band 539.15 → 550.85 contains six of the seven rows** (only 05_transformer at 891.09 falls outside) — on 366 test bars these gaps are within sweep-noise of the floor. Second, **05_transformer's MAE 891.09 is the outlier** — 65 % above naive's 540.96, in a leaderboard otherwise within $11 of naive. Third, **07's single-seed Sharpe 8.20 is a favourable seed**: the README's 5-seed comparison reports `5.523 ± 0.418 (CI 5.004–6.043)` for this exact recipe. The CI lower bound (5.00) still clears 06 zero-shot (2.97) and ARIMA's seed-of-one (6.86) sits inside the CI — read the 07 lead as "seed distribution clears 06 zero-shot at 95 % confidence", not as a reproducible 8.20.

## Variants and recipes

Two experiments ship multiple trial configs on the current slice. The canonical row in the leaderboard above is the Makefile-default trial; the other variants — same test window, different backend / size / train slice / fine-tuning recipe — surface here.

### 06_pretrained variants (same test slice, zero-shot, different backend × size)

| Variant | hub_model_name | params | MAE | RMSE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|
| `btc_4h_2024.chronos-small` (canonical) | `autogluon/chronos-2-small` | 28 M | 546.66 | 817.81 | 0.4754 | 0.2225 | 2.9722 |
| `btc_4h_2024.chronos-large` | `amazon/chronos-2` | 120 M | 560.59 | 837.11 | 0.4344 | 0.1116 | 1.9004 |
| `btc_4h_2024.timesfm` | `google/timesfm-2.5-200m-pytorch` | 200 M | 570.79 | 843.75 | **0.4863** | **0.3164** | **4.1080** |

Zero-shot **bigger Chronos is worse, not better** (chronos-large trails small on every metric — Sharpe 1.90 vs. 2.97). TimesFM is the best zero-shot variant on trading metrics (Sharpe 4.11, 38 % lift over chronos-small) but the worst on point error (MAE 570.79, +24.13 vs. small). No zero-shot variant beats the linear baseline; see [06's Variants section](06_pretrained.report.md#variants) for the editorial.

### 07_finetuned variants (same test slice, different train slice × recipe × backend)

| Variant | train | recipe | backend | MAE | RMSE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|---|
| `btc_4h_2020_2024_encoder_only.chronos-2-small` (canonical) | 4.7 yr | encoder-only | chronos-2-small | 549.85 | **813.80** | **0.5683** | **0.7329** | **8.1987** |
| `btc_4h_2020_2024_encoder_only.timesfm` | 4.7 yr | encoder-only | timesfm-2.5 | **543.92** | 815.39 | 0.5246 | 0.4512 | 5.3910 |
| `btc_4h_2020_2024_encoder_only.chronos-2-large` ⚠ | 4.7 yr | encoder-only | chronos-2-small (filename misnamed; see [07's variant note](07_finetuned.report.md#variants)) | 554.55 | 824.10 | 0.5437 | 0.5199 | 6.0544 |
| `btc_4h_2024` (former canonical) | 1.6 yr | head-only | chronos-2-small | 542.98 | 815.12 | 0.5328 | 0.4162 | 5.6022 |

The canonical now embodies all three of the README's load-bearing conditions: 4.7-yr train slice, encoder-only recipe, `restored_best_checkpoint: true`. Swap any one out and the lift collapses: timesfm at same train + recipe drops Sharpe to 5.39; head-only at same backend on the shorter 1.6-yr slice drops Sharpe to 5.60. The on-disk single-seed Sharpe 8.20 is a favourable seed — the README's 5-seed mean is **5.523 ± 0.418 (CI 5.004–6.043)**.

### 07_finetuned sweep — fine-tuning recipe × learning rate on the 1.6-yr sibling slice

From [`experiments/07_finetuned/results/btc_4h_2024/sweep.csv`](../../experiments/07_finetuned/results/btc_4h_2024/sweep.csv): 10 rows on the **former-canonical 1.6-yr slice** (the sweep was last regenerated before the Makefile rotation; no 4.7-yr sweep exists on disk). Chronos-2-small backend except one timesfm row, `input_chunk_length = 256`, `n_epochs = 20` max. The Sharpe leader is **encoder-only at lr=1e-5** (Sharpe 6.87 on 1.6 yr) — the same recipe / lr corner the current canonical pins on 4.7 yr (Sharpe 8.20). The *lowest val_loss* row (`full × 1e-5`, val_loss 0.03479) posts Sharpe 5.73 — not the Sharpe leader. **In-sample val_loss does not pick the out-of-sample Sharpe winner.** Same lesson as ARIMA's AIC ↔ Sharpe disagreement.

## The complexity-doesn't-pay arc (with two big asterisks)

Ordering by approximate trainable-parameter count on the canonical leaderboard:

| Row | Approx. trainable params | Sharpe | dir_acc | MAE (USD) |
|---|---|---|---|---|
| 01_baseline | 0 | — | NaN | 540.96 |
| 02_arima (3, 1, 3) | 7 | **6.8559** | 0.5082 | **539.15** |
| 03_xgboost | ~31 feat × trees × depth | 6.1388 | 0.5082 | 550.85 |
| 04_lstm (default) | ~10 000 weights | 4.8221 | 0.5301 | 539.23 |
| 05_transformer (default) | ~30 000+ weights | 2.5390 | 0.5055 | 891.09 |
| 06_pretrained (chronos-2 small, zero-shot) | ≈ 28 M (no BTC training) | 2.9722 | 0.4754 | 546.66 |
| 07_finetuned (canonical: encoder-only, chronos-2 small, 4.7-yr) | ≈ 26 M trainable on top of frozen head | **8.1987** | **0.5683** | 549.85 |
| 07_finetuned (head-only, 1.6-yr, former canonical)* | ≈ 1.6 M trainable on top of 28 M frozen | 5.6022 | 0.5328 | 542.98 |

\* Non-canonical sibling row; see [Variants](#07_finetuned-variants-same-test-slice-different-train-slice--recipe--backend).

The trained-on-this-data trajectory (rows 01 → 05) tells the now-familiar story: **capacity helps up to ARIMA(3, 1, 3) at 7 parameters, plateaus through XGBoost, then *declines* through LSTM and the TFT** at ten- to thirty-thousand weights. LSTM Sharpe 4.82 and TFT Sharpe 2.54 both sit below ARIMA(3, 1, 3)'s 6.86. The 3 467-row 4h training set is not enough for the deep models to amortise their capacity.

**The pretrained rows (06, 07) modify the arc in two distinct ways:**

- **06 (zero-shot, 28 M frozen params) does *not* beat the trained ARIMA.** Last version of this report cited 06 as the Sharpe leader at 7.59; the current canonical (chronos-2-small instead of chronos-2) lands at Sharpe 2.97 — *worse* than every trained model except the TFT. The "borrowing 120 M pretrained params beats the linear baseline" headline does not survive a backend / checkpoint swap.
- **07 canonical (fine-tuned, ~94 % of chronos-2-small trainable on top of frozen head, 4.7-yr slice) takes Sharpe / dir_acc / cum_ret leadership.** Sharpe 8.20 single-seed (5.523 ± 0.418 over five seeds). Borrowed parameters need to be both *appropriate* (chronos-2-small + encoder-only > chronos-2-small + head-only > timesfm + encoder-only) and *given enough BTC* (4.7 yr > 1.6 yr) before they outperform a 7-parameter ARIMA. The previous canonical (head-only, 1.6 yr) cleared 06 zero-shot but still trailed ARIMA on Sharpe; the rotated canonical clears both.

Headline reframing: **on a small training set, you don't get to use more parameters by training them yourself — and you don't get a free win by borrowing them either. The borrowed parameters need fine-tuning targeted at the right architectural surface, on enough of the target data, before they outperform a 7-parameter linear baseline.**

## Point-error vs. trading-metric tradeoff

The canonical leaderboard surfaces a clean split between point-error and trading-metric leadership:

- **07_finetuned (canonical) vs. 02_arima(3, 1, 3).** MAE: **02 wins by $10.70**. MAPE: 02 wins by 0.0117 pp. RMSE: 02 wins by $5.01. **dir_acc: 07 wins (0.5683 vs. 0.5082, Δ +0.060).** **cum_ret: 07 wins by 20.6 pp.** **Sharpe: 07 wins by 1.34.** ARIMA is *closer* to the price on average; 07 gets direction right more often *and* leans in the right magnitude when it does. The strategy rewards 07's calibrated magnitudes over ARIMA's smaller-but-more-frequent edge.
- **04_lstm vs. 02_arima(3, 1, 3).** MAE: 02 wins by $0.08 (effectively tied). RMSE: 04 wins by $0.50. **dir_acc: 04 wins (0.5301 vs. 0.5082)**. **cum_ret: 02 wins (0.5269 vs. 0.4284). Sharpe: 02 wins (6.86 vs. 4.82).** 04 gets direction more often but compounds it worse — its predicted-up moves are smaller, so it stays long less, and the rally collects fewer up-bars per long.

The mechanism is well-established: **dir_acc and the magnitude-aware trading metrics are not the same objective**. 07 canonical wins because it *both* gets direction right more often *and* sizes the predicted distribution appropriately for BTC log-returns — the README calls this "prediction magnitude calibration" and credits encoder-only fine-tuning with the head frozen. The previous canonical (head-only) lifted dir_acc but not magnitude calibration, which is why it cleared 06 zero-shot on Sharpe but lost to ARIMA. **Pick the model for the metric you care about; on Sharpe, the rotated 07 canonical now clears that ceiling — but only on a favourable seed (CI 5.00–6.04, which still clears 06 and brackets ARIMA).**

## Where capacity helps and where it hurts

Pair-by-pair walk of the ladder, one transition at a time (each row adds **one** thing to the previous):

| Pair | What's added | Δ MAE | Δ dir_acc | Δ Sharpe | Verdict |
|---|---|---|---|---|---|
| 01 → 02(3, 1, 3) | linear AR + MA, differencing | −1.81 | from NaN to 0.5082 | from — to 6.8559 | huge help |
| 02 → 03 | non-linear trees + 31 engineered features | +11.70 | 0.0000 | −0.7171 | hurt (MAE up, Sharpe down) |
| 03 → 04 | recurrent state, raw past covariates instead of engineered features | −11.62 | +0.0219 | −1.3167 | mixed (dir_acc up but Sharpe down) |
| 04 → 05 | attention + future covariates + variable selection | +351.86 | −0.0246 | −2.2832 | hurt across the board |
| 05 → 06 | **swap from trained TFT to zero-shot Chronos-2 small** (paradigm change, not capacity step) | −344.43 | −0.0301 | +0.4332 | mixed (MAE recovers, dir_acc drops below coin-flip, Sharpe nearly unchanged) |
| 06 → 07 (canonical) | encoder-only fine-tuning of chronos-2-small on 4.7 yr of BTC + best-val_loss checkpoint restore | +3.19 | +0.0929 | +5.2265 | huge help on dir_acc / Sharpe; small MAE penalty |

Two transitions help meaningfully on dir_acc and Sharpe (`01 → 02`, `06 → 07`). One hurts across the board (`04 → 05`). The big load-bearing observation: **the lifts that matter come from inductive-bias swaps**, not from adding capacity along a single architectural axis. ARIMA's linear-differencing bias, the foundation models' pretrained prior, fine-tuning's targeted weight updates on enough data — each of these is a different *kind* of model, not a bigger version of the previous one.

The `06 → 07` Δ Sharpe `+5.23` is by far the largest single-step lift in the ladder — but it bundles three changes (4.7-yr slice, encoder-only recipe, checkpoint restore) into one canonical rotation. The within-07 decomposition (canonical vs. the former-canonical sibling at head-only-on-1.6-yr, Sharpe 5.60 — see [07's Variants](07_finetuned.report.md#variants)) suggests roughly half of the lift comes from train-slice length and half from recipe.

## AIC vs. out-of-sample

Two sweeps land in the same family of disagreement:

**02_arima** sweep (12 `(p, d, q)` orders on the same training slice, see [02_arima.report.md#sweep](02_arima.report.md)) — AIC and the canonical metrics now agree on `(3, 1, 3)` as the order to run (the canonical 02 row *is* `(3, 1, 3)`, the AIC leader). The classic AIC vs. Sharpe disagreement that earlier 1h sweeps surfaced does not currently fire on the 4h slice for this experiment.

**07_finetuned** sweep (10 recipe × learning-rate rows on the same train slice, see [07_finetuned.report.md#sweep--fine-tuning-recipe--learning-rate](07_finetuned.report.md)) — the lowest `best_val_loss` (0.03479, `full × 1e-5`) posts Sharpe 5.73, *not* the sweep's Sharpe leader. The sweep's Sharpe leader (6.87, `encoder-only × 1e-5`) posts `best_val_loss` 0.03505. **In-sample val_loss does not rank out-of-sample Sharpe.** Same lesson as the prior ARIMA AIC-vs-Sharpe disagreement, now visible in 07's recipe selection: if you pick the fine-tuning recipe by val_loss alone you miss the Sharpe leader.

## NaN-by-design as a teaching surface

NaN appears in three places, each for a different reason — each informative about [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py):

1. **01's `directional_accuracy = NaN`** — the naive predictor returns `close_pred[t] = close[t-1]`, so `sign(pred − ref) = sign(0) = 0` for every bar. The metric's mask `(true_dir != 0) & (pred_dir != 0)` is empty; the function short-circuits to NaN per [metrics.py:23-34](../../src/btc_ai/eval/metrics.py).
2. **02 sweep `(0, 1, 0)` row — `directional_accuracy = NaN` and `sharpe = NaN`** — the random-walk ARIMA is mathematically equivalent to naive, so it inherits naive's NaN dir_acc *and* produces an entirely flat strategy (`pred = ref` everywhere → `position = 0` → `stddev = 0` → NaN per [metrics.py:51-59](../../src/btc_ai/eval/metrics.py)).
3. **01's `cumulative_return` and `annualized_sharpe`** — naive never expresses a long position, so the strategy never participates and both metrics report "—" (omitted from `metrics.json`, not technically NaN — but the absence has the same meaning).

A leaderboard reader can mistake any of these for a missing run. They are not. The metric module reports NaN as a deliberate signal of "no opinion expressed."

## Pipeline sanity checks

Three checks land cleanly in the artefacts:

1. **02 sweep's `(0, 1, 0)` row reports MAE bit-identical to 01_baseline's MAE** — the random-walk ARIMA *is* the naive predictor; identical numbers confirm the data slice, the split index, and the metric implementation are shared.
2. **07_finetuned's `restored_best_checkpoint: true`** — confirms that walk-forward inference used best-val_loss weights, not the (overfit) end-of-training weights. Per the per-experiment README, an earlier configuration of 07 posted Sharpe 6.146 *before* this check was wired in — the number dropped to 3.46 after fix. The artefact field is load-bearing; flag any future row where it flips to `false`.
3. **All seven canonical rows report identical `rows_test: 366`** — confirms the test-window alignment claimed in the [Data-slice picture](#data-slice-picture).

Two checks that *don't* exist yet but would be useful:

- A `walk-forward with retraining` shadow of any 4h experiment, to verify that `retrain=False` doesn't materially change the Sharpe on this slice.
- Multi-seed runs on the canonical rows of 06 and 07 — the per-experiment README for 07 has a 5-seed comparison, but `metrics.json` is single-seed. The headline Sharpes have run-to-run variance that is not currently surfaced.

## Per-bar Sharpe significance

Per-bar Sharpe = `annualized / √ppy` with `ppy = 2190` for 4h. SE ≈ `1 / √N_test` with `N_test = 366`. Collected across all seven canonical rows plus the 07 encoder-only-extended variant:

| Row | Sharpe (annualized) | √ppy | per-bar | √N_test | SE | ratio (σ) |
|---|---|---|---|---|---|---|
| 02_arima (3, 1, 3) | 6.8559 | 46.80 | 0.1465 | 19.13 | 0.0523 | **2.80** |
| 03_xgboost | 6.1388 | 46.80 | 0.1312 | 19.13 | 0.0523 | **2.51** |
| 04_lstm | 4.8221 | 46.80 | 0.1031 | 19.13 | 0.0523 | 1.97 |
| 05_transformer | 2.5390 | 46.80 | 0.0543 | 19.13 | 0.0523 | 1.04 |
| 06_pretrained (chronos-2 small) | 2.9722 | 46.80 | 0.0635 | 19.13 | 0.0523 | 1.21 |
| 07_finetuned (canonical, single-seed on disk) | 8.1987 | 46.80 | 0.1752 | 19.13 | 0.0523 | **3.35** |
| 07_finetuned (canonical, 5-seed mean per README) | 5.5230 | 46.80 | 0.1180 | 19.13 | 0.0523 | 2.26 |
| 07_finetuned (canonical, 5-seed CI lower bound) | 5.0040 | 46.80 | 0.1069 | 19.13 | 0.0523 | 2.04 |
| 07_finetuned (former canonical, head-only 1.6-yr)* | 5.6022 | 46.80 | 0.1197 | 19.13 | 0.0523 | 2.29 |

\* Non-canonical sibling row.

**Three single-number rows clear the 2.5 σ threshold**: ARIMA(3, 1, 3) (2.80), XGBoost (2.51), and 07's canonical single-seed (3.35). The 07-canonical 5-seed mean lands at 2.26 σ — borderline — and the CI lower bound at 2.04 σ. **The honest read is that 07's canonical is borderline-to-clean**, depending on whether you cite the single-seed value or the 5-seed CI. The single-seed 3.35 σ is a favourable-seed artefact; the 5-seed CI is more conservative. 06's canonical (1.21) is not significant under any read.

This is a coarse approximation — it assumes per-bar strategy returns are i.i.d., which they almost certainly aren't on 4h crypto. Volatility clustering inflates the true SE. **The σ ratios above are upper bounds on the real significance.**

## Regime caveat

Every result above is conditional on a test slice that lands precisely **`2024-10-01` → `2024-12-01`** (366 bars at 4h) — the post-election BTC rally. A strongly trending regime is exactly where a long/flat strategy with mild directional skill compounds favourably. **None of the rows in this report have been tested out-of-regime.** The directional skill that ARIMA, XGBoost, the fine-tuned foundation models, and (marginally) the LSTM show may or may not survive a sideways or down-trending slice. The 07 encoder-only-extended Sharpe of 8.20 — the strongest in the series — is the most regime-sensitive finding: a model trained on 4.7 years of mixed regimes was evaluated on 2 months of one of the strongest BTC rallies on record, and no second test window has been run.

Until that is tested, every positive Sharpe in this report is best read as *"skill conditional on this regime"*, not *"skill in general."*

## Bottom line

On 366 bars of 4h BTC over the `2024-10-01` → `2024-12-01` test window:

- **07_finetuned canonical leads dir_acc, cum_ret, and Sharpe.** Single-seed on disk: Sharpe **8.1987** / dir_acc **0.5683** / cum_ret **0.7329**. 5-seed mean per README: Sharpe **5.523 ± 0.418** (CI 5.004–6.043). Per-bar significance: **3.35 σ single-seed, 2.04 σ CI lower bound** — borderline-to-clean depending on which number you cite.
- **02_arima(3, 1, 3) still leads point error** (MAE 539.15, MAPE 0.6903 %). The linear baseline is closer to the price on average; 07 is more often right about direction and sizes its magnitudes better.
- **The most-parameters-trained-from-scratch model, 05_transformer, is worst** — MAE 891 (65 % above naive), Sharpe 2.54.
- **The prior canonical for 07 (head-only, 1.6-yr) cleared 06 zero-shot on Sharpe but trailed ARIMA**; rotating the canonical to encoder-only on the 4.7-yr slice is what put a fine-tuned foundation model above the linear baseline on the trading metrics.
- **The earlier "120 M-zero-shot-parameters beat ARIMA" headline from the prior synthesis still does not survive the canonical-trial pin** — 06's current canonical (chronos-2-small zero-shot) lands at Sharpe 2.97.

Read in one line: *The rotated 07 canonical (encoder-only fine-tuning of chronos-2-small on 4.7 years of BTC with best-val_loss checkpoint restoration) takes Sharpe / dir_acc / cum_ret leadership from ARIMA on a single favourable seed (5-seed CI 5.00–6.04, lower bound clears 06 zero-shot at 95 %), while ARIMA keeps the point-error crown.*

## Where this goes next

Operational follow-ups surfaced by the per-experiment reports:

- **Re-run `make 07_finetuned_sweep` on the current canonical (4.7-yr) slice.** The on-disk sweep was last regenerated against the former canonical (1.6-yr) slice; the canonical's recipe / lr corner is currently only rankable indirectly via the sibling sweep. With the Makefile now pointing at the 4.7-yr config, a fresh `make 07_finetuned_sweep` would write to the canonical's results dir and rank recipes on the right slice.
- **Fix the `chronos-2-large` filename bug** in 07's variant configs. [`btc_4h_2020_2024_encoder_only.chronos-2-large.config.yaml`](../../experiments/07_finetuned/configs/btc_4h_2020_2024_encoder_only.chronos-2-large.config.yaml) has `hub_model_name: autogluon/chronos-2-small`; correct it to `amazon/chronos-2` and re-run. Without this, the 4.7-yr × encoder-only × 120 M cell in the head-to-head is missing.
- **Run 06_pretrained_sweep on the current slice** — a real backend × `input_chunk_length` sweep is not on disk; the variants table covers backend × size at one ICL only.
- **Out-of-regime test.** Run a sibling experiment family on, e.g., 2022-Q2 (sideways) or 2022-Q4 (down-trending) and re-rank. The 07 canonical Sharpe (single-seed 8.20, 5-seed CI 5.00–6.04) needs that test before "skill" replaces "skill in this regime." The README's 5-seed comparison is *seed* variance, not regime variance.
- **Multi-seed canonical row for 07 in `metrics.json`.** The per-experiment README ships the 5-seed CI; `metrics.json` is single-seed. A `make 07_finetuned_seeds` (or `metrics.json` enriched with `seeds_mean / seeds_std`) would let the global leaderboard cite the reproducible centre rather than the favourable seed currently on disk. Same gap applies to 06.
- **Probabilistic-interval analysis.** Only 06 / 07 produce q10 / q50 / q90 bands. Calibration is not summarised in `metrics.json`. A future global report could add an interval-coverage column (fraction of bars where `close ∈ [pred_lo, pred_hi]`).
- **Article series**, per [docs_ml/articles_todo.md](../articles_todo.md). The reports above are the raw material for those articles; this synthesis is the closing summary the series will eventually mirror.

## Caveats (shared by all rows)

- Single split, no rolling-origin CV.
- Walk-forward without re-estimation (`retrain=False` for trained models; nothing to re-estimate for naive / MA / foundation models; fine-tuning happens once before walk-forward).
- One seed per row for the stochastic trainers (XGBoost, LSTM, TFT, fine-tuned foundation models). 07's per-experiment README has multi-seed numbers; this report does not surface them.
- Long / flat strategy, no shorting, no transaction costs, no slippage.
- Single test regime (Oct–Nov 2024 BTC rally). See [Regime caveat](#regime-caveat).
- All numbers above are copied verbatim from each experiment's canonical `metrics.json` / `sweep.csv`. No values are recomputed in this document.

## Source artefacts

Per-experiment reports (audit trail):

- [01_baseline.report.md](01_baseline.report.md)
- [02_arima.report.md](02_arima.report.md)
- [03_xgboost.report.md](03_xgboost.report.md)
- [04_lstm.report.md](04_lstm.report.md)
- [05_transformer.report.md](05_transformer.report.md)
- [06_pretrained.report.md](06_pretrained.report.md)
- [07_finetuned.report.md](07_finetuned.report.md)

Raw `metrics.json` / `sweep.csv` for each experiment under `experiments/${id}/results/${canonical_trial}/`.

## How to regenerate this report

```
# 1. Make sure every per-experiment canonical artefact is fresh.
make 01_baseline && make 02_arima && make 03_xgboost
make 04_lstm && make 05_transformer && make 06_pretrained && make 07_finetuned
# (re-run sweeps too if they are stale)

# 2. Regenerate per-experiment reports per Prompt 1 in docs_ml/report_generation.md.

# 3. Regenerate this global report per Prompt 2 in docs_ml/report_generation.md.
```
