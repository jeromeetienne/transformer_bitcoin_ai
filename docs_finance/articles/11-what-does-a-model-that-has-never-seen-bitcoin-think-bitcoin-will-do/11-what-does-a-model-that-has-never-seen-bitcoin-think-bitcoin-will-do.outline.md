# Outline — What does a model that has never seen Bitcoin think Bitcoin will do?

## One-line pitch
Walkthrough of [06_pretrained](../../experiments/06_pretrained/) using Amazon's Chronos-2 (120 M params), zero-shot, no fine-tuning, on the same 1h BTCUSDT log-returns slice as 04 / 05. Honest framing: if a pretrained prior over millions of unrelated series transfers, that's the result; if MAE sits near the naive floor (~$260) and dir_acc ≈ 0.50, **the data — not the architecture, not the parameter count — is the bottleneck**. Chronos-2 lands at MAE $262 with dir_acc 0.5019 (chance), confidently centred on `close_{T-1}`. The first time the lineup answers "is BTC unusually hard, or are we just bad at it?" *(TimesFM 2.5 is also wired in the harness; the article will discuss the recent TimesFM rerun with dir_acc 0.4677 / Sharpe +2.44 to round out the foundation-model story.)*

## Audience
Anyone who's read a "foundation models for time series" announcement (Chronos, TimesFM, Lag-Llama, Moirai) and wondered if the prior generalizes to crypto. Series readers who watched LSTM and TFT lose to ARIMA in articles 8/9 and want to know if a pretrained model finally gets past the floor.

## Thesis
Chronos-2 was trained on millions of unrelated time series and has a calibrated prior over plausible next-step distributions. On hourly BTC log-returns, that prior transfers to the *level* (MAE $262 — within $2 of naive) but not to the *direction* (dir_acc 0.5019 — chance). The model has learned that "the next bar will be near the last bar"; it has not learned anything about which direction the next bar will move. That's not a critique of Chronos-2. **It's the cleanest data point this lab will produce for the claim "the bottleneck on hourly BTC is the data, not the model".**

## Structure

### 1. The hook — a 120 M parameter model that has never seen Bitcoin
- Chronos-2 was pretrained on millions of unrelated time series (electricity, retail, weather, traffic, …).
- It has *never* been fine-tuned on BTCUSDT.
- We hand it the same hourly slice the rest of the lab uses, ask for one-step-ahead probabilistic forecasts, and ride the same harness.
- The provocation: if the pretrained prior generalizes, that's a striking result. If it lands at the naive floor, *that's also* a striking result — it would mean architecture and parameter count don't break the floor.

### 2. The original Chronos-2 result (the canonical illustration)
From the first 06_pretrained run with `backend: chronos, hub_model_name: amazon/chronos-2`:
```json
{
  "backend": "chronos",
  "hub_model_name": "amazon/chronos-2",
  "input_chunk_length": 256,
  "rows_test": 1608,
  "mae":  262.27,
  "rmse": 397.63,
  "mape": 0.00342,
  "directional_accuracy": 0.5019,
  "cumulative_return":   0.169,
  "annualized_sharpe":   4.292
}
```
Within ~$2 of naive's MAE. dir_acc statistically indistinguishable from coin-flip. Sharpe at the drift floor (Article 10).

### 3. The follow-up TimesFM 2.5 result
- The currently-committed `metrics.json` reflects a TimesFM 2.5 run (200 M params, decoder-only) on the same slice:
  - MAE $268.89, dir_acc 0.4677, Sharpe +2.44.
- Slightly worse on every column than Chronos. dir_acc *below* chance (0.4677 means the model's directional opinions are anti-correlated with reality more than half the time on the bars where it expressed an opinion).
- Both backends are zero-shot, both inherit the same harness, both sit near the naive MAE floor.
- The pattern is consistent: foundation models trained on millions of unrelated series produce *level-coherent but direction-blind* predictions on hourly BTC.

### 4. What "level-coherent but direction-blind" means
- The model's median prediction is `close_pred ≈ close_{T-1}`. It has learned the right scale and the right neighborhood.
- The predicted distribution (q10 / q50 / q90) is centered on the last value with a calibrated width.
- But the *median* sign of `pred - ref` is essentially random across test bars.
- The strategy gate (`pred > ref`) flips roughly half the time, with no skill in which half.
- That's the cleanest possible illustration that "predicting the level well" and "predicting the direction" are different tasks. The naive baseline does the first by definition; foundation models extend that with calibrated uncertainty bands; nobody so far has done the second.

### 5. The architecture of the experiment
- **darts foundation-model wrapper.** `Chronos2Model` and `TimesFM2p5Model` both inherit from `darts.models.forecasting.foundation_model.FoundationModel`, so the rest of the harness is identical.
- **Univariate by design.** Both backends are run *without* covariates. TimesFM 2.5 doesn't accept any; Chronos-2 supports past+future covariates but we keep it univariate to make the head-to-head fair. (The "Why univariate" section in the README is the canonical statement.)
- **`fit()` is a no-op.** Both classes require `model.fit(target_train_scaled)` for API consistency, but no weight updates happen. The fit() call serves to lock the schema; weights come straight from HuggingFace Hub.
- **Walk-forward evaluation.** Same `historical_forecasts(retrain=False)` harness as LSTM/TFT.
- **Probabilistic head.** `QuantileRegression([0.1, 0.5, 0.9])` with `num_samples=200`. The median feeds the leaderboard metrics; q10 / q90 produce the prediction band — Article 12 is about that.

Code excerpt — the entire foundation-model fit + predict from [06_pretrained/run.py](../../experiments/06_pretrained/run.py):

```python
model = build_model(backend, hub_model_name, merged, quantiles)
model.fit(scaler_target.transform(target_train))   # no-op for foundation models
preds_s = model.historical_forecasts(
    series=target_full_s,
    start=test_start,
    forecast_horizon=1,
    retrain=False,
    last_points_only=True,
    num_samples=200,
)
```

### 6. Why this is the cleanest data point in the series
- Capacity: 120 M – 200 M parameters. Way above LSTM's 22k or TFT's ~50k.
- Training data: millions of time series. Way above the ~6,400 training bars we have for in-domain models.
- Architecture: state-of-the-art for general time-series forecasting (encoder-only T5 / decoder-only patch-transformer).
- Result: same MAE as a one-line predictor. Same dir_acc as coin-flip. Same Sharpe as the drift floor.
- **If Chronos-2 / TimesFM 2.5 can't extract directional signal, the bottleneck is unambiguously the data.** This is the experiment that makes that claim with a straight face.

### 7. What "the data is the bottleneck" actually means
A short subsection.
- It does *not* mean BTC is unpredictable. It means *the conditional mean of `r_T` given the past 256 bars of `r_T` alone is statistically indistinguishable from zero*.
- The signal that exists (the small AR(1) ARIMA captures) is small enough that 120 M parameters and millions of unrelated training series do not extract it.
- A model that wants to do better has to either:
  - Use *different inputs* — funding rate, perp basis, on-chain flow, news, order-book microstructure. ARIMAX with a single covariate that has even modest signal might beat ARIMA.
  - Use a *different output* — directional / quantile loss instead of squared error.
  - Look at *different timescales* — 1d or 1w log-returns might have richer structure.
- Article 13's roadmap will mention all three.

### 8. The probabilistic forecasts that come for free
- Both backends produce calibrated quantile bands (q10 / q50 / q90).
- This is the first experiment in the lineup with probabilistic outputs.
- Article 12 will be the methods post specifically about what those bands give you that point predictions don't.
- For this article, the band's *narrowness and centeredness* is itself a finding: the model is saying "I'm 80 % confident the next bar's return will be inside a small band centred on zero". That's a high-confidence claim of "no information about direction".

### 9. The leaderboard, with foundation models
| Model | dir_acc | Sharpe | MAE | params |
|---|---:|---:|---:|---|
| ARIMA(1,1,1) | 0.5336 | +7.28 | $260.50 | 3 |
| LSTM | 0.5196 | +4.95 | $274.02 | ~22k |
| MA(24) | 0.5143 | +4.33 | $756.19 | 0 |
| TFT | 0.5053 | +4.59 | $373.70 | ~50k |
| **Chronos-2** | **0.5019** | **+4.29** | **$262.27** | **120M** |
| XGBoost | 0.4872 | +1.45 | $270.73 | 31 features × 400 trees |
| **TimesFM 2.5** | **0.4677** | **+2.44** | **$268.89** | **200M** |
| Naive | NaN | — | $260.50 | 0 |

(All on the same Sep–Nov 2024 window.)

The 120-200 M-parameter foundation models are *better than XGBoost on MAE* but still at chance directional accuracy. Capacity bought them nothing on the leaderboard column we care about.

### 10. Caveats baked into v1
A short list, mostly pulled from the README:
- **Univariate.** No covariates for either backend. Future work: turn on Chronos-2's covariate channel.
- **No fine-tuning.** Both classes support `enable_finetuning`. Zero-shot is a clean baseline; fine-tuning is a separate experiment.
- **`output_chunk_length: 1`.** One-step-ahead. Multi-horizon is a one-line edit but changes the loss surface.
- **Single sampling seed.** `num_samples=200` gives a reasonable distribution; multi-seed would give a confidence interval on the metrics.
- **`amazon/chronos-2` is 120 M; `chronos-2-small` is 28 M; `timesfm-2.5-200m` is 200 M.** The small variant is faster; useful for sanity checks. Article 13 will mention the size sweep.

### 11. Reproducing
```
make 06_pretrained         # uses config.yaml's backend setting
make 06_pretrained_sweep   # tries multiple (backend, hub_model_name, icl) combos
```
First run downloads HF weights (one-time). Single-digit minutes on Apple MPS / CUDA per backend.

### 12. Closing — the central thesis of the lab, restated
- 120 M – 200 M parameters trained on millions of time series. dir_acc 0.5019 / 0.4677.
- Three numbers fit on 6,432 BTC bars by `statsmodels`. dir_acc 0.5336.
- It's the data.
- Tease Article 12 (probabilistic forecasts) and Article 13 (the roadmap).

## Key code/file references
- [experiments/06_pretrained/run.py](../../experiments/06_pretrained/run.py) — the harness
- [experiments/06_pretrained/config.yaml](../../experiments/06_pretrained/config.yaml) — backend selection
- [experiments/06_pretrained/README.md](../../experiments/06_pretrained/README.md) — extensive interpretive guide

## Tone notes
- Genuinely curious tone, not gloating. The result is striking either way.
- Make the point about "data is the bottleneck" with the leaderboard, not adjectives.
- Hold off on probabilistic forecasts — that's Article 12.

## Length target
~1,800–2,200 words.
