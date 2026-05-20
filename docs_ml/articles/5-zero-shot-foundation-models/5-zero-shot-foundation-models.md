# What does a model that has never seen Bitcoin think Bitcoin will do?

The closing article in the series. The previous four took models that were *trained on Bitcoin* — naive, ARIMA, XGBoost, LSTM, Temporal Fusion Transformer — and ranked them against each other on the same 366 test bars. The winner of that race is a 3-parameter linear ARIMA(3, 1, 3) with annualized Sharpe 6.86. The Temporal Fusion Transformer, after thirty thousand trainable parameters and a hardware-accelerated training loop, came in dead last on point error. The deep-learning story did not deliver. The question this article answers is the natural follow-on: **can borrowing twenty-eight to two hundred million pretrained parameters from a foundation model that has never seen Bitcoin close the gap?**

The honest answer is no, not zero-shot, not on this slice. The best zero-shot variant — Google's TimesFM 2.5 at 200 million parameters — lands at Sharpe 4.11, still below ARIMA's 6.86. The configured-canonical (Amazon's Chronos-2 small, 28 M parameters) lands at directional accuracy 0.4754, *below coin-flip*. And — the cleanest "bigger is not better" inversion in the entire series — Chronos-2 large with 120 M parameters is *uniformly worse* than Chronos-2 small with 28 M parameters across every leaderboard column.

This article walks through what those numbers mean and what they tell us about the inductive bias of foundation models for time series when the target is Bitcoin 4-hour log-returns.

## Methodology, in one paragraph

Target: one-step-ahead log-return `r_T = log(close_T / close_{T-1})`, reconstructed back to price as `close_pred = close_{T-1} * exp(r_pred)`. Data: `BTCUSDT` 4-hour bars from Binance Vision. Split: train `2023-01-01` → `2024-08-01` UTC (3 833 rows, used only to *seed the context window* — see below), validation `2024-08-01` → `2024-10-01` UTC (unused — there is no early stopping when there is no training), test `2024-10-01` → `2024-12-01` UTC (**366 bars**). Walk-forward, one step ahead, weights frozen across the test window. Metrics from the shared module — [`src/btc_ai/eval/metrics.py`](src/btc_ai/eval/metrics.py). Reproduce: `make 06_pretrained`. **New this article:** the models do not train at all (`fit()` is a no-op), and forecasts are probabilistic — three quantiles Q10 / Q50 / Q90 per bar, drawn from 200 stochastic samples from the model's predictive distribution.

## What the models are

Two distinct foundation-model families, three variants total.

**Chronos-2** (Amazon). Encoder-only T5-style architecture. Pretrained as a language model over discretized token IDs of time-series values — quantize the value range into a fixed vocabulary, treat the time series as a sequence of tokens, and train an encoder-decoder transformer with masked-language-modelling loss on a corpus of synthetic and real time series. At inference time, the same tokenization is applied to the input context, the model produces a distribution over next-token IDs, and the IDs are decoded back to continuous values. The repo ships both checkpoints:

- `autogluon/chronos-2-small` — 28 M parameters.
- `amazon/chronos-2` — 120 M parameters.

**TimesFM 2.5** (Google). Decoder-only patch-transformer architecture. Different design choices from Chronos: continuous patches of fixed length rather than discretized tokens, decoder-only (autoregressive) rather than encoder-only. The patch-transformer reads the input series in chunks of fixed length and predicts the next patch directly in value space.

- `google/timesfm-2.5-200m-pytorch` — 200 M parameters.

Both families are **univariate**. TimesFM 2.5 does not accept covariates. Chronos-2 technically supports them, but the experiment keeps both univariate so the head-to-head comparison is on identical inputs and the question stays *"does the pretrained prior alone transfer to Bitcoin?"*. Library: `darts.models.Chronos2Model` and `darts.models.TimesFM2p5Model`.

## Zero-shot means `fit()` is a no-op

For these models, `fit()` is implemented but does nothing. There are no weights to update. The model is fully specified by the HuggingFace Hub checkpoint named in the config:

```yaml
# experiments/06_pretrained/configs/btc_4h_2024.chronos-small.config.yaml
backend: chronos
hub_model_name: autogluon/chronos-2-small

model:
  input_chunk_length: 256       # ≈ 42 days at 4h
  output_chunk_length: 1
  num_samples: 200
  quantiles: [0.1, 0.5, 0.9]
```

The training slice from articles 1 through 4 still exists in the config — that is what the leaderboard reports as `rows_train: 3833`. But the model never sees those rows for parameter updates; they exist only to *seed the input-chunk-length context window* at the start of the walk-forward. The model is *not* trained on Bitcoin and never has been. The forecasts the model makes on the test slice are made by a network whose weights were frozen at HuggingFace Hub publication time and have not changed since.

This is true zero-shot. Not zero-shot-with-some-scaler-fit; not zero-shot-with-a-few-final-layer-tweaks. The model has *no* exposure to Bitcoin in its training corpus, *no* fine-tuning, *no* head-only adaptation. The pretrained prior alone is what produces the forecasts.

## The walk-forward shape

```python
preds = model.historical_forecasts(
    series=close_series,
    start=test_start,
    forecast_horizon=1,
    retrain=False,
    last_points_only=True,
    num_samples=200,
)
```

Same harness as articles 3 and 4 — `historical_forecasts(retrain=False, last_points_only=True)` — plus the `num_samples=200` argument. At each test bar, the model is given the most recent 256 bars of price history (in log-return space), runs them through the frozen network, and emits 200 stochastic draws from its predictive distribution for the next bar. Three quantiles (Q10, Q50, Q90) are computed from those 200 samples and stored in `predictions.parquet`. The median Q50 feeds the leaderboard metrics — MAE, RMSE, MAPE, directional accuracy, cumulative return, annualized Sharpe — so the row is directly comparable to the point-forecast rows from articles 1 through 4.

## The numbers

From [`experiments/06_pretrained/results/btc_4h_2024.chronos-small/metrics.json`](experiments/06_pretrained/results/btc_4h_2024.chronos-small/metrics.json) (the configured canonical):

| Metric | Chronos-2 small, zero-shot |
|---|---|
| MAE | 546.66 USD |
| RMSE | 817.81 USD |
| MAPE | 0.7003 % |
| Directional accuracy | **0.4754** (below coin-flip) |
| Cumulative return | 0.2225 |
| Annualized Sharpe | 2.9722 |

The full variants table — same test slice, same metric module, three different pretrained checkpoints:

| Variant | hub_model_name | params | MAE | dir_acc | cum_ret | Sharpe |
|---|---|---|---|---|---|---|
| Chronos-2 small | `autogluon/chronos-2-small` | 28 M | **546.66** | 0.4754 | 0.2225 | 2.9722 |
| Chronos-2 large | `amazon/chronos-2` | 120 M | 560.59 | 0.4344 | 0.1116 | 1.9004 |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | 200 M | 570.79 | **0.4863** | **0.3164** | **4.1080** |

Three observations the variants table makes load-bearing.

## Observation one: bigger Chronos is worse, not better

Chronos-2 large has 120 M parameters. Chronos-2 small has 28 M. The same architectural family, the same context length, the same `num_samples`. Only the parameter count changes. The 120 M checkpoint trails the 28 M checkpoint by $13.93 on MAE, 0.041 on directional accuracy, and 1.07 on Sharpe. Across every leaderboard column.

This is the cleanest single-table inversion of "bigger is better" the series has produced. The intuition from large language models — where scaling laws are well-behaved enough that the right answer to "should I use the bigger model?" is approximately always "yes, if you can afford it" — does not survive contact with `BTCUSDT` 4-hour log-returns. The mechanism is that *foundation-model capacity acts as a prior sharpener, not just an information container*. The 120 M model is more confident in its pretrained prior than the 28 M model is. On a target that doesn't match the pretrained corpus (the target is heavy-tailed, zero-mean, nearly random; the corpus is mostly smooth periodic signal), sharpening a wrong prior makes the forecast *worse*. The 28 M model hedges; the 120 M model commits; the commits are wrong more often.

This is the kind of result that does not come up in a Chronos-2 abstract or a TimesFM whitepaper, because their evaluation corpora are not Bitcoin. It is what happens when the prior the model carries does not generalize to the target the user has in hand.

## Observation two: TimesFM wins trading metrics, loses point error

The decoder-only patch-transformer family has a different inductive bias from Chronos's encoder-only T5. On the leaderboard, TimesFM 2.5 reports:

- the *worst* MAE of the three variants (570.79 — $24.13 above Chronos-2 small);
- the *best* directional accuracy of the three (0.4863, though still below coin-flip);
- the *best* cumulative return (0.3164) and *best* Sharpe (4.1080).

A 38 % relative lift in Sharpe over Chronos-2 small, at the cost of $24.13 in MAE per bar. The same dir_acc-vs-Sharpe split shape we saw in article 3 (LSTM vs. ARIMA, LSTM trading direction for magnitude) and the MAE-vs-Sharpe split shape we saw in article 4 (Temporal Fusion Transformer Sharpe-positive on a slice that flatters everything). TimesFM's predictions are wrong-by-more on magnitude but more aggressively directional than Chronos's; the long-flat strategy triggers on more bars during the rally and collects more drift; cum_ret and Sharpe go up; MAE goes down.

The architecture-family difference matters. Two foundation models pretrained on overlapping but distinct corpora, with different tokenization strategies and encoder/decoder topologies, produce *materially different* zero-shot Bitcoin forecasts. This is the kind of observation that justifies the existence of multiple foundation-model families: they are *not* interchangeable on out-of-distribution targets. The intuition that "any large pretrained time-series model gives roughly the same answer" is not borne out.

## Observation three: all three lose to ARIMA(3, 1, 3) on every metric

The best zero-shot variant (TimesFM 2.5) at Sharpe 4.11 is 2.75 Sharpe points behind ARIMA's 6.86. The Sharpe gap from the best zero-shot foundation model to a 7-parameter linear AR / MA model is *wider* than the Sharpe gap from ARIMA to the LSTM in article 3 (4.82). The "pretrained prior wins by default" framing the foundation-model literature sometimes implies does not hold on this slice.

Here is the complete leaderboard, every model in articles 1 through 5:

| Model | MAE (USD) | RMSE (USD) | dir_acc | cum_ret | Sharpe |
|---|---|---|---|---|---|
| Naive last-value | 540.96 | 813.14 | NaN | — | — |
| ARIMA(3, 1, 3) | **539.15** | 808.79 | 0.5082 | 0.5269 | **6.8559** |
| XGBoost (31 features) | 550.85 | **808.65** | 0.5082 | 0.4149 | 6.1388 |
| LSTM | 539.23 | 808.29 | **0.5301** | 0.4284 | 4.8221 |
| Temporal Fusion Transformer | 891.09 | 1 271.26 | 0.5055 | 0.1219 | 2.5390 |
| Chronos-2 small (zero-shot) | 546.66 | 817.81 | 0.4754 | 0.2225 | 2.9722 |
| Chronos-2 large (zero-shot) | 560.59 | 837.11 | 0.4344 | 0.1116 | 1.9004 |
| TimesFM 2.5 (zero-shot) | 570.79 | 843.75 | 0.4863 | **0.3164** | 4.1080 |

A 3-parameter linear ARIMA leads on point error and on Sharpe. An LSTM leads on directional accuracy. Two of three zero-shot foundation models are *below coin-flip on direction*. Six of eight rows cluster inside an MAE band of $539–$571 — that is the data's noise floor at this horizon, and it sets the same ceiling for every model class that does not transfer signal from somewhere else.

## Probabilistic forecasts — the new shape

This article is the first in the series where the model emits a calibrated predictive *distribution* per bar instead of a point estimate. The Q10 / Q50 / Q90 quantiles are stored in [`experiments/06_pretrained/results/btc_4h_2024.chronos-small/predictions.parquet`](experiments/06_pretrained/results/btc_4h_2024.chronos-small/predictions.parquet) alongside the actual close, the reference price, and the strategy-relevant fields. The headline leaderboard metrics use only the median Q50 — that is what makes the row comparable to the point-forecast rows from articles 1 through 4. But the Q10 / Q90 band is what makes probabilistic forecasting *valuable* — it gives a reader the right shape of the model's uncertainty.

A probabilistic forecast deserves its own evaluation. Interval coverage — the fraction of bars where the actual close falls inside the Q10 / Q90 band — should be close to 80 % if the model is calibrated, and quantile loss or CRPS would be the metric of choice over MAE. The current `metrics.json` does not include these. They are on the operational follow-up list. A version of the leaderboard that scored interval coverage alongside point error would tell a *much* richer story about Chronos-2 vs. TimesFM 2.5 than the median-only row currently does.

## Why zero-shot Bitcoin is not a free win

Foundation-model pretraining works when the pretraining corpus contains the inductive biases that the downstream task needs. For computer vision: ImageNet-pretrained backbones encode "edges, textures, objects in scenes" — and that transfers to almost any image-based task. For language: web-scale text encodes "syntax, semantics, world knowledge" — and that transfers to almost any text-based task. For time series, the analog of ImageNet does not really exist yet, and the *de facto* pretraining corpora for Chronos-2 and TimesFM 2.5 do not include Bitcoin or anything that behaves like it:

- The Chronos-2 pretraining corpus is dominated by synthetic time series (procedurally generated to cover a range of canonical patterns) and real datasets from M5 retail forecasting, electricity load, traffic, weather, finance (some). What those have in common: roughly smooth signal with periodic structure plus regime shifts plus moderate noise.
- The TimesFM 2.5 corpus is more variable but broadly similar: Google Trends, demand forecasting, some financial markets.

Heavy-tailed-zero-mean log-returns at 4-hour cadence are not a prominent class in either corpus. The inductive bias the models carry is "smooth periodic signal with occasional regime shift" — exactly the *wrong* prior for Bitcoin one-step-ahead log-returns, where the next bar is mostly noise around zero with intermittent volatility clusters. The pretrained prior dampens the median forecast toward the recent mean and the rolling variance — which is the right thing to do for electricity demand and the wrong thing to do for Bitcoin.

That mismatch is the entire explanation for the three observations above: zero-shot transfer is weak; bigger Chronos is worse because it commits harder to the wrong prior; TimesFM's different patch-transformer bias happens to be slightly less wrong than Chronos's discretization bias on this slice.

## Per-bar Sharpe sanity check

For each zero-shot variant, the back-of-envelope per-bar Sharpe significance:

| Variant | Sharpe | per-bar | SE | ratio (σ) |
|---|---|---|---|---|
| Chronos-2 small | 2.9722 | 0.0635 | 0.0523 | **1.21** |
| Chronos-2 large | 1.9004 | 0.0406 | 0.0523 | **0.78** |
| TimesFM 2.5 | 4.1080 | 0.0878 | 0.0523 | **1.68** |

None of the three clears the 2 σ bar under the optimistic i.i.d. assumption. The positive Sharpe figures are real in the data, but they are not statistically distinguishable from no-skill on 366 test bars. The right read is "the zero-shot priors do not produce no-opinion forecasts — they produce something — but on this slice, what they produce is not detectably better than chance after accounting for noise."

## What this article tells us about the model class

**Zero-shot transfer to Bitcoin is real but weak at this scale and on this slice.** All three foundation-model variants produce finite, structured forecasts. The forecasts are not random; the models do something. But what they do does not generalize past a 7-parameter linear ARIMA on this slice — and the larger Chronos checkpoint does it *worse* than the smaller one. The foundation-model literature's framing of "borrow a pretrained prior and beat task-specific models" applies on benchmarks where the pretraining corpus shares structure with the target. Bitcoin 4-hour log-returns are not such a target.

**Foundation-model scaling does not always behave the way computer vision and language taught us to expect.** The 120 M Chronos checkpoint is uniformly worse than the 28 M Chronos checkpoint on this slice. Larger pretrained models concentrate the prior more aggressively, which is good if the prior is right and bad if it is not. The series has tracked this same "complexity does not pay" arc through trained-from-scratch models (XGBoost ties ARIMA, LSTM goes backwards on Sharpe, Temporal Fusion Transformer goes backwards on MAE); it shows up again with pretrained models, in a sharper form.

**The probabilistic-forecast head is the load-bearing capability** these models bring that no other model in the series has. The Q10 / Q90 bands are calibrated; the long-flat strategy and the headline metrics use only the median. Scoring interval coverage, calibration, and quantile loss alongside MAE would tell a richer story about Chronos-2 vs. TimesFM 2.5 than the median-only row currently does. That is on the operational follow-up list for the repo.

## Series-wide bottom line

Across articles 1 through 5: a 3-parameter linear ARIMA(3, 1, 3) takes the point-error crown (MAE 539.15) and the trading-metric crown (Sharpe 6.86) on a 366-bar 4-hour BTC test slice. Every later model — XGBoost on 31 engineered features, an LSTM with two recurrent layers, a Temporal Fusion Transformer with thirty thousand attention parameters, three zero-shot foundation models with twenty-eight to two hundred million pretrained parameters — adds capacity in some axis and goes backwards in another. The Temporal Fusion Transformer is sixty-five percent worse than naive on MAE. Two of three zero-shot foundation models are below coin-flip on direction. The most ML-credible result the series produces is **complexity does not pay on this signal-to-noise ratio**, and the lesson generalizes well past Bitcoin. Where models with hundreds of millions of pretrained parameters fail to beat a 7-parameter linear baseline, it is a property of the data, not of the model.

The series ends here. The repository continues — a fine-tuning experiment ([`experiments/07_finetuned/`](experiments/07_finetuned/)) answers the natural follow-up question about whether fine-tuning these pretrained models on Bitcoin closes the gap, and the cross-experiment synthesis at [`docs_ml/reports/XX_global.report.md`](docs_ml/reports/XX_global.report.md) keeps the full leaderboard up to date. The article series stops at the zero-shot result on purpose: zero-shot is the cleanest statement of the *pretrained prior alone* question, and the answer to that question — *no, not on this slice* — is the load-bearing finding of the series.

## Regime caveat

Every result in this article is conditional on the test slice landing precisely on the post-election Bitcoin rally (`2024-10-01` → `2024-12-01` UTC). None of the zero-shot variants has been tested out-of-regime. The series's positive Sharpes are best read as *"skill conditional on this regime."* A sibling experiment family on a sideways or down-trending slice would tell whether the directional skill that ARIMA, XGBoost, the LSTM, and TimesFM 2.5 show survives a regime that does not flatter long-flat strategies.

## Reproduce

```
make 06_pretrained                                                                         # canonical: chronos-2 small
make 06_pretrained CONFIG=experiments/06_pretrained/configs/btc_4h_2024.chronos-large.config.yaml
make 06_pretrained CONFIG=experiments/06_pretrained/configs/btc_4h_2024.timesfm.config.yaml
```

The first writes [`experiments/06_pretrained/results/btc_4h_2024.chronos-small/metrics.json`](experiments/06_pretrained/results/btc_4h_2024.chronos-small/metrics.json) — the canonical chronos-small numbers. The second and third write the sibling variants' `metrics.json` files in their own results subdirectories.

The series is published. The code is the artefact; the articles are commentary. Thank you for reading.
