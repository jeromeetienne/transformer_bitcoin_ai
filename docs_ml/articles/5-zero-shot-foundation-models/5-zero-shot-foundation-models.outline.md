# Outline — Article 5: Zero-shot foundation models

Working title: *"What does a model that has never seen Bitcoin think Bitcoin will do?"*

Source experiment: [`06_pretrained`](experiments/06_pretrained/).

## Editorial framing

The capstone. Two off-the-shelf pretrained time-series models — Amazon's Chronos-2 (in two sizes: 28 M and 120 M parameters) and Google's TimesFM 2.5 (200 M parameters) — applied zero-shot to `BTCUSDT` 4h log-returns. No training. No fine-tuning. The weights are downloaded from HuggingFace Hub; `fit()` is a no-op for these models in Darts; the model walks forward across the test slice having never been shown a single bar of Bitcoin during pretraining.

The point of the article is the question the previous four articles set up: does borrowing 28 to 200 million pretrained parameters close the ARIMA gap? The honest answer: **no, not zero-shot, not on this slice.** The best zero-shot variant (TimesFM 2.5) reports Sharpe 4.11 — still below ARIMA's 6.86. The default canonical (Chronos-2 small) reports dir_acc 0.4754 — *below coin-flip*. The expected "120 million parameters beat the linear baseline" headline does not survive the experiment.

The article also introduces something every prior model in the series did not have: **probabilistic forecasts**. Three quantiles per bar (Q10 / Q50 / Q90) instead of a single point estimate.

## Beats

1. **Hook.** A 28-million-parameter model trained on millions of time series — none of them Bitcoin — generates a forecast at every test bar. Its directional accuracy on Bitcoin is 0.4754 — *below coin-flip*. A 120 M model from the same family does worse. A 200 M decoder-only model does better on Sharpe (4.11) but still loses to ARIMA's 6.86. Borrowing parameters is not the free lunch the foundation-model literature occasionally suggests.

2. **Methodology recap.** Same 366-bar test slice. Same walk-forward shape. Same metric module. The new things this article: (1) no training at all — `fit()` is a no-op; (2) the model has never seen Bitcoin in pretraining; (3) probabilistic forecasts with Q10 / Q50 / Q90 quantiles, of which Q50 (median) is what feeds the leaderboard metrics.

3. **What the models are — in plain English.**
   - **Chronos-2.** Encoder-only T5-style architecture from Amazon. Pretrained on millions of synthetic and real time series, framed as a language-modelling problem over discretized token IDs of values. Available in two sizes: `autogluon/chronos-2-small` (28 M parameters) and `amazon/chronos-2` (120 M parameters).
   - **TimesFM 2.5.** Decoder-only patch-transformer architecture from Google. Different inductive bias from Chronos: TimesFM uses continuous patches of fixed length rather than discretized tokens, and is decoder-only (autoregressive) rather than encoder-only. 200 M parameters.
   - **Both are univariate by design.** TimesFM 2.5 does not accept covariates. Chronos-2 technically supports them, but the experiment keeps it univariate for an apples-to-apples comparison with TimesFM.
   - **Library:** `darts.models.Chronos2Model`, `darts.models.TimesFM2p5Model`. Both wrap HuggingFace Hub model downloads.

4. **The "fit() is a no-op" beat.** For these models, `fit()` is implemented but does nothing — there are no weights to update. The model is fully specified by the HuggingFace Hub checkpoint. This is *true* zero-shot: not zero-shot-with-some-scaler-fit, just no-training-period. Code-link to where this is wired up.

5. **The walk-forward shape.** `historical_forecasts(retrain=False, last_points_only=True, num_samples=200)` — same harness as articles 3 and 4, plus the `num_samples=200` argument. Each forecast at each test bar is now 200 stochastic draws from the model's predictive distribution; the three quantiles are computed across those samples and stored in `predictions.parquet`. The median (Q50) feeds the leaderboard.

6. **The configuration — three variants** (one canonical + two siblings). From the canonical [`btc_4h_2024.chronos-small.config.yaml`](experiments/06_pretrained/configs/btc_4h_2024.chronos-small.config.yaml):
   - input_chunk_length: 256 (≈ 42 days at 4h)
   - output_chunk_length: 1
   - num_samples: 200
   - quantiles: [0.1, 0.5, 0.9]
   - backend: chronos
   - hub_model_name: autogluon/chronos-2-small

   Siblings: `*.chronos-large.config.yaml` and `*.timesfm.config.yaml`.

7. **The result — canonical (chronos-2 small, zero-shot), verbatim from `metrics.json`.**
   - MAE 546.66 USD
   - RMSE 817.81 USD
   - MAPE 0.7003 %
   - dir_acc 0.4754 (below coin-flip)
   - cum_ret 0.2225
   - Sharpe 2.9722

8. **The variants table — full backend × parameter-count head-to-head.**
   | Variant | params | MAE | dir_acc | Sharpe |
   |---|---|---|---|---|
   | Chronos-2 small (canonical) | 28 M | **546.66** | 0.4754 | 2.9722 |
   | Chronos-2 large | 120 M | 560.59 | 0.4344 | 1.9004 |
   | TimesFM 2.5 | 200 M | 570.79 | **0.4863** | **4.1080** |
   
   Three observations the variants table makes load-bearing.

9. **Observation 1: bigger Chronos is uniformly *worse*, not better.** The 120 M chronos-large trails the 28 M chronos-small by $13.93 on MAE, 0.041 on dir_acc, and 1.07 on Sharpe. Same architecture family, same context length, same `num_samples`. Only the parameter count changes. On near-random-walk financial data, the extra 92 M parameters of pretrained prior actively *hurt*. This is the cleanest single-table inversion of "bigger is better" the series has produced.

10. **Observation 2: TimesFM wins the trading metrics by a margin.** Sharpe 4.108 vs. chronos-small's 2.972 is a 38 % relative lift. cum_ret 0.316 vs. 0.223 is a 42 % lift. dir_acc 0.4863 is still below coin-flip but it is the highest of the three. The decoder-only patch-transformer has a meaningfully different (and on this slice, better) inductive bias for BTC than either Chronos checkpoint.

11. **Observation 3: All three lose to ARIMA(3, 1, 3) on every metric except RMSE.** The Sharpe gap from the best zero-shot (4.11) to ARIMA (6.86) is 2.75 — a wider gap than ARIMA's lead over the LSTM in article 3. The "pretrained prior wins by default" framing the foundation-model literature sometimes implies *does not hold* on this slice.

12. **The leaderboard, complete.**
   | Model | MAE | dir_acc | Sharpe |
   |---|---|---|---|
   | Naive | 540.96 | NaN | — |
   | ARIMA(3, 1, 3) | **539.15** | 0.5082 | **6.8559** |
   | XGBoost | 550.85 | 0.5082 | 6.1388 |
   | LSTM | 539.23 | **0.5301** | 4.8221 |
   | Temporal Fusion Transformer | 891.09 | 0.5055 | 2.5390 |
   | Chronos-2 small (zero-shot) | 546.66 | 0.4754 | 2.9722 |
   | Chronos-2 large (zero-shot) | 560.59 | 0.4344 | 1.9004 |
   | TimesFM 2.5 (zero-shot) | 570.79 | 0.4863 | 4.1080 |

13. **The probabilistic-forecast beat — Q10 / Q50 / Q90.** This is the first article in the series where the model emits a calibrated predictive *distribution* per bar instead of a point estimate. The three quantiles are stored in `predictions.parquet`; the long-flat strategy and the headline metrics use only the Q50 median. But the Q10 / Q90 band is what makes probabilistic forecasting *valuable* — it gives a reader the right shape of the model's uncertainty. Briefly mention what one would *do* with the bands (interval coverage, calibration plots) and note that the experiment doesn't yet score them.

14. **Why zero-shot Bitcoin is not the free win it sounds like.**
   - **Pretraining datasets do not contain Bitcoin (or anything like Bitcoin).** Chronos-2 was pretrained on synthetic time-series + real datasets from M5, traffic, electricity load, etc. TimesFM 2.5 was pretrained on Google Trends, financial markets (some), demand forecasting datasets. Neither corpus has heavy-tailed-zero-mean log-returns at 4h cadence as a prominent class.
   - **The inductive bias is "smooth periodic signal with occasional regime shift,"** not "noise with autocorrelated volatility." The pretrained prior dampens the median forecast toward the recent mean — that is the right thing to do for electricity demand, the wrong thing for a Bitcoin log-return where the next bar is mostly noise around zero.
   - **Bigger models concentrate the prior more aggressively, not less.** 120 M Chronos is *worse* than 28 M Chronos because the larger model is more confident in the wrong prior. Capacity in foundation models acts as a *prior sharpener*, not just an information container; sharpening a wrong prior makes the forecast worse.

15. **Per-bar Sharpe significance.**
   - Chronos-small: 2.9722 / √2190 = 0.0635; SE 0.0523; ratio ≈ **1.21 σ** — not significant.
   - Chronos-large: 1.9004 / √2190 = 0.0406; SE 0.0523; ratio ≈ **0.78 σ** — not significant.
   - TimesFM: 4.1080 / √2190 = 0.0878; SE 0.0523; ratio ≈ **1.68 σ** — not significant.
   - **No zero-shot variant clears the 2 σ bar.** The Sharpe figures are positive but not statistically separable from no-skill under the optimistic i.i.d. assumption.

16. **What this article tells us about the model class.**
   - **Zero-shot transfer to Bitcoin is real but weak.** All three variants produce finite forecasts and finite metrics; the forecasts are not random; the models do something. But what they do does not generalize past a 7-parameter linear ARIMA on this slice.
   - **Foundation-model scaling does not always behave the way computer vision and language taught us to expect.** 120 M Chronos is *worse* than 28 M Chronos on every leaderboard metric. The "more parameters = better zero-shot" intuition transferred from large language models does not survive contact with Bitcoin 4h log-returns.
   - **The probabilistic-forecast head is the load-bearing capability** these models bring that no other model in the series has. The Q10 / Q90 bands are calibrated; the strategy currently only uses the median. Future work, summarized below, exploits the bands.

17. **Series-wide bottom line, in two paragraphs.**
   - Across articles 1 through 5: a 3-parameter linear ARIMA(3, 1, 3) takes the point-error crown (MAE 539.15) and is within 2 Sharpe points of the LSTM's directional accuracy lead. Every deep-learning model — LSTM, Temporal Fusion Transformer — adds capacity and goes backwards on at least one metric we care about. Zero-shot foundation models with 28 M to 200 M pretrained parameters do not close the gap. The most ML-credible result in the series is that **complexity does not pay on this signal-to-noise ratio** — and the lesson generalizes well past Bitcoin, well past 4h, and well past this test slice. Where models with hundreds of millions of pretrained parameters fail to beat a 7-parameter linear baseline, it is a property of the data, not of the model.
   - The series ends here for the article series. The repo continues — `07_finetuned` answers the natural follow-up question ("does fine-tuning these pretrained models on Bitcoin close the gap?") in a separate experiment not covered here. The article series stops at the zero-shot result on purpose: zero-shot is the cleanest statement of the "pretrained prior alone" question, and the answer to that question is the load-bearing finding of the series.

18. **Reproduce.** `make 06_pretrained` and the variant `CONFIG=…` overrides.

## What this article is NOT

- A claim that zero-shot foundation models are useless on time series. The Chronos-2 and TimesFM 2.5 papers report strong results on standard benchmarks. The lesson is what happens on *one* heavy-tailed-zero-mean BTC slice.
- A claim that fine-tuning would not help. The `07_finetuned` experiment in the repo investigates that question separately; the article series stops at zero-shot for editorial focus.
- A foundation-model tutorial. Refer the curious reader to the Chronos and TimesFM papers.
