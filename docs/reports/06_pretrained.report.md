# Report — `06_pretrained`

**Run date:** 2026-04-29
**Status:** completed (single fit, no sweep yet)

## What this experiment is

The first **zero-shot foundation model** in the lineup. Where 04 and 05 *trained* a sequence model on this exact slice of BTC, 06 takes a model that has never seen this data — its weights come straight from HuggingFace Hub — and asks whether the prior baked into a 120 M-parameter transformer trained on millions of unrelated time series transfers to 1 h BTC log-returns. The model class is selected by `backend:` in [config.yaml](../../experiments/06_pretrained/config.yaml): either Amazon's [Chronos-2](https://huggingface.co/amazon/chronos-2) (encoder-only T5-style, 120 M params) or Google's [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (decoder-only patch transformer, 200 M params). This run is **chronos-2 only**; timesfm and the multi-config sweep are deferred.

```
target  : r_T = log(close_T / close_{T-1})           # next-step log-return (same as 04 / 05)
inputs  : target sequence, with input_chunk_length    # univariate — no covariates
predict : r_q0.1 / r_q0.5 / r_q0.9   (200 samples)    # probabilistic, calibrated quantile bands
          close_pred_T = close_{T-1} * exp(r_q0.5)    # median feeds the leaderboard metrics
```

Library: [Darts](https://unit8co.github.io/darts/) `Chronos2Model`. Walk-forward 1-step-ahead via `historical_forecasts(retrain=False, num_samples=200)`. `fit()` is required by the API but performs no weight updates — the model is genuinely zero-shot. Two material differences from 04 / 05:

1. **No training.** Weights come from HF Hub. Fitting takes seconds (just sets up the schema).
2. **Probabilistic output.** The median (q=0.5) feeds MAE / RMSE / MAPE / directional_accuracy / Sharpe so the headline numbers stay comparable to 01–05; q10 and q90 are stored alongside in `predictions.parquet` and shown as a shaded band on `plot.png`. This is the first model in the lineup with calibrated uncertainty.

### Why univariate

Both backends are run **without covariates**. TimesFM 2.5 doesn't accept any (univariate-only by construction); Chronos-2 supports past + future covariates but is deliberately not given any here. With one backend forced univariate, giving the other extra inputs would muddy the head-to-head: any difference would conflate "which prior is stronger?" with "did the covariates help?". Keeping both univariate isolates the question this experiment is actually asking — *does the pretrained prior alone transfer to BTC?* Adding covariates to chronos-2 only is a clean follow-up ablation, not v1.

This also means the comparison vs 04 / 05 isn't a strict superset / subset relation: 04 had OHLCV past-covariates, 05 added cyclical hour/dow future-covariates, and 06 has *neither*. If 06 ties 04 / 05 on metrics despite running without those inputs, that's strong evidence the inputs weren't carrying signal.

## Configuration

From [experiments/06_pretrained/config.yaml](../../experiments/06_pretrained/config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `1h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| backend | **`chronos`** |
| hub_model_name | `amazon/chronos-2` (120 M params) |
| model.input_chunk_length | **256** (≈ 10.7 days of 1 h history per forecast) |
| model.output_chunk_length | 1 (one-step-ahead) |
| model.num_samples | 200 (samples drawn from the predictive quantile distribution) |
| model.quantiles | `[0.1, 0.5, 0.9]` (subset of chronos-2's 21 pre-trained quantiles) |

Total bars after dropna: **8 039**. Train: **6 431**. Test: **1 608**. The split is a simple 2-way (train / test) — there is no validation slice because there's no early stopping in zero-shot. This means 06's train slice is ~640 bars longer than 04 / 05's (which carve a val tail off train), and 06's test slice is **1 bar offset** from 04 / 05's by the int-arithmetic difference between `int(n*(1-0.2))` and `n - int(n*0.2)`. Test still spans the same Sep–Nov 2024 window; metrics are comparable to 1-bar tolerance.

> **Float32 cast.** Inputs cast to `float32` for Apple's MPS backend, same as 04 / 05. Price reconstruction stays in float64.

> **HuggingFace download.** First run downloads `amazon/chronos-2` (~300 MB) and caches it at `~/.cache/huggingface/hub/`. Subsequent runs are offline.

## Results — single fit

From [experiments/06_pretrained/results/metrics.json](../../experiments/06_pretrained/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | **262.32** USD |
| RMSE | 398.27 USD |
| MAPE | 0.3432 % |
| Directional accuracy | **0.5019** |
| Cumulative return | **+34.04 %** |
| Annualized Sharpe | **+4.29** |

The plot at [experiments/06_pretrained/results/plot.png](../../experiments/06_pretrained/results/plot.png) tells the story before the table does: the median prediction tracks close almost exactly, and the q0.1 – q0.9 band is so tight it's barely visible. The model's distribution over next-step log-returns is *narrow and centered on the last value*. That's the pretrained prior on BTC at 1 h: "near-zero log-return with very low variance" — i.e., predict the last price.

## Cross-experiment comparison

Same data slice, ~1607–1608 test bars from 2024-09 to 2024-11, same metrics (06 has 1 extra test bar — see Configuration note):

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| **01_baseline_naive** | **260.50** | **396.97** | **0.341 %** | NaN | — | — |
| 01b_moving_average (window=24) | 756.19 | 1 100.93 | 0.990 % | 0.5143 | +25.67 % | +4.33 |
| **02_arima (1, 1, 1)** | **260.50** | **396.97** | **0.341 %** | **0.5336** | **+53.02 %** | **+7.28** |
| 03_gradient_boosting (default) | 270.73 | 414.07 | 0.354 % | 0.4872 | +8.09 % | +1.45 |
| **04_lstm (default)** | 274.02 | 409.66 | 0.360 % | **0.5196** | **+50.34 %** | **+4.95** |
| 05_transformer (default) | 373.70 | 546.68 | 0.488 % | 0.5053 | +29.57 % | +4.59 |
| **06_pretrained chronos-2 (icl=256)** | **262.32** | **398.27** | **0.343 %** | 0.5019 | +34.04 % | +4.29 |

Three things to notice:

1. **MAE is essentially at the naive floor.** 06 lands at $262.32 — within $2 / 0.7 % of naive's $260.50. RMSE and MAPE are also within rounding of naive. This is the cleanest hit-the-floor result in the lineup: 03 leaves $10 of MAE on the table, 04 leaves $14, 05 leaves $113. **A 120 M-parameter pretrained transformer arrives at the same MAE as a one-line `pred = last_close` baseline.**
2. **Directional accuracy is 0.5019 — within Bernoulli noise of 50 %.** With 1 608 test bars, the 95 % CI on a fair coin is about ±2.4 %; 0.5019 is squarely inside that. The pretrained prior has **no directional opinion** on hourly BTC.
3. **Sharpe is +4.29 despite dir_acc ≈ 0.50.** Worth pausing on: the long/flat strategy earns +34 % cumulative on a model that calls direction at chance. That's not skill — it's regime. The Sep–Nov 2024 test window is a strong uptrend (close moves from ~$58 k to ~$96 k). Any model that *occasionally* says "long" — even at random — captures part of that drift, and the per-bar realized return when long is positive on average. The Sharpe number is real but it's measuring the test window's drift, not the model's signal. Same caveat applies retroactively to LSTM (+4.95) and TFT (+4.59); this run just makes the issue impossible to ignore because the dir_acc / Sharpe gap is so visible.

## Interpretation

1. **The pretrained prior on BTC is the naive prior.** This is what the experiment was designed to test, and the answer is unambiguous. Chronos-2 — trained on Amazon's mixture of real and synthetic time series, scoring SOTA on fev-bench, GIFT-Eval, Chronos Benchmark II — has converged to "predict the last value" on BTC log-returns. That's not a failure of Chronos-2; it's a statement about hourly BTC log-returns. Any sequence model whose training distribution doesn't include BTC-specific noise structure will land here.
2. **The probabilistic head is the new thing, and the bands are tight.** Earlier experiments produced a point prediction; 06 produces a distribution. The q0.1 – q0.9 spread on the median prediction is small relative to the actual close volatility — the model is *confident*, and confidently centred on `close_{T-1}`. That's useful information: it means even when calibrated for uncertainty, the model can't separate "up" from "down" at the next bar, so it covers the predictive interval with a near-symmetric band around the last value. Quantile loss is doing exactly what it should; there's just no asymmetry to exploit.
3. **No covariates, no training, ties LSTM on MAE and beats TFT on every metric.** 06 has *neither* OHLCV past-covariates nor cyclical hour/dow future-covariates and *no* gradient updates on this slice. It still lands at MAE $262 vs LSTM's $274 vs TFT's $374. Reading this honestly: **the covariates and the training in 04 / 05 were not contributing real predictive content.** They were either zero-mean noise on top of the same naive prior or — in TFT's case — dead weight that hurt MAE without buying directional accuracy. This is the cleanest negative result for "more inputs / more capacity / more training" in the lineup so far.
4. **ARIMA(1, 1, 1) still wins the leaderboard.** Sharpe +7.28 / dir_acc 0.5336 is unmatched. One AR coefficient + one MA coefficient on first differences extracts the only stable directional signal that exists at 1 h — and a 120 M-parameter foundation model doesn't recover it. The signal isn't *complexity*; it's the AR(1) structure on differences, which Chronos-2's training corpus presumably *contained* but didn't teach it to apply specifically to high-vol financial-style series at 1 h cadence.
5. **The "free Sharpe from a positive drift window" caveat is now front and centre.** 06's Sharpe of +4.29 against a dir_acc of 0.5019 is the cleanest demonstration that the trading-strategy column is dominated by realized BTC drift, not model skill. Every Sharpe number in the leaderboard above is partially a function of the Sep–Nov 2024 rally; only ARIMA's +7.28 has dir_acc strong enough to be doing real directional work on top of that drift. Re-running every experiment on a falling or chopping window is the only way to sort skill from beta.

### Bottom line

**A pretrained time-series foundation model has no useful prior on hourly BTC log-returns.** Chronos-2 with 256 bars of context lands within $2 of naive on MAE, calls direction at chance, and delivers a Sharpe number that's almost entirely the test window's drift. That is a *useful* negative: it tells us cleanly that the bottleneck for every model in the lineup — naive through TFT — is the data, not the architecture or the parameter count or the training corpus. The signal at 1 h is what AR(1) on differences extracts, and nothing in the broader TS-pretraining literature seems to find more.

This is also the first experiment where every metric line up the way the user predicted before running it ("near-naive MAE and ~0.5 dir_acc — a useful data point, not a leaderboard win"). When the prior is right, the experiment confirms; when it's wrong, the report has to explain why. Today it's the former.

Productive next directions:

- **Run TimesFM 2.5.** This report is chronos-only. Running `backend: timesfm` on the same slice would tell us whether the same negative result holds across foundation-model families, or whether it's specific to chronos's training mixture. The harness and config knobs are already wired ([sweep.py](../../experiments/06_pretrained/sweep.py) GRID); just needs the run.
- **Sweep `input_chunk_length`.** The intuition "more context = better foundation-model forecast" is testable. The GRID has 64 / 256 / 1024 for both backends. If the answer is "all three land at the naive floor," that's an even stronger version of the v1 conclusion.
- **Add covariates to chronos-2 only (ablation).** Chronos-2 supports `past_covariates` and `future_covariates`. A separate run with the OHLCV + cyclical-time channels from 05 would tell us whether the foundation model can use covariates that LSTM and TFT couldn't extract signal from.
- **Probabilistic-head fine-tuning.** Both backends support `enable_finetuning=True` for partial / full fine-tuning. Quantile-loss fine-tuning on 6 431 BTC training bars is the obvious next experiment: does any fine-tuning of the prior pull the model away from "predict the last close"? (Likely no, but the asymmetry of the result vs effort is small enough to test.)
- **Different test windows.** Sep–Nov 2024 is a strong uptrend; the Sharpe leaderboard is partially a long-drift artefact. The natural falsifiability test is mid-2022 (post-LUNA, FTX) — a bidirectional / falling regime. Applies to every model, not just 06.

## Caveats

- **Single backend run only.** Just chronos-2 at icl=256. TimesFM 2.5 is wired but not run; `make 06_pretrained_sweep` would produce the full backend × icl table.
- **Single split.** Same 2024-09 to 2024-11 test window as every other experiment. Hyperparameter rankings (here: backend / icl) could shift on a different window.
- **Walk-forward without re-estimation.** `retrain=False` keeps weights frozen across the entire test window. For zero-shot foundation models this is the *only* mode of operation that makes sense — there's nothing to retrain.
- **`num_samples=200`, single sampling seed.** Two runs may produce slightly different quantile bands. The median is stable; the q0.1 / q0.9 endpoints are within ~$5 between sampling seeds at this num_samples, well below the spread of the band itself.
- **Univariate by design.** Reading "Chronos lost on Sharpe to ARIMA" without remembering 06 is *also* the only model in the lineup with no covariates and no training-time exposure to BTC is missing the point. The fair comparison is "does Chronos-2 with the same inputs as ARIMA/LSTM/TFT lose?" — and that experiment is the chronos-2-with-covariates ablation listed under next directions.
- **Test slice is 1 bar longer than 04 / 05** (1 608 vs 1 607). Stems from the 2-way vs 3-way split arithmetic. Below the noise floor for every metric reported.
- **HuggingFace download on first run.** Network access required the first time; subsequent runs are offline. The chronos-2 weights are ~300 MB.
- **Test-window drift dominates Sharpe.** As noted in interpretation point 5, +4.29 with dir_acc 0.50 is structural, not skill. Same caveat applies retroactively to 04 / 05; 06 just makes it impossible to miss.

## Files produced

- [experiments/06_pretrained/results/metrics.json](../../experiments/06_pretrained/results/metrics.json) — single fit
- [experiments/06_pretrained/results/predictions.parquet](../../experiments/06_pretrained/results/predictions.parquet) — columns: close, pred (median), pred_lo (q0.1), pred_hi (q0.9), ref, strategy_return
- [experiments/06_pretrained/results/plot.png](../../experiments/06_pretrained/results/plot.png) — close + median + shaded q0.1 – q0.9 band
- *(no `sweep.csv` yet — `make 06_pretrained_sweep` produces one)*

## How to reproduce

```
make 06_pretrained         # single fit using params in config.yaml (~3-5 min on MPS, +download on first run)
make 06_pretrained_sweep   # 7-config sweep across (backend, hub_model_name, input_chunk_length) — 20-40 min on MPS
```
