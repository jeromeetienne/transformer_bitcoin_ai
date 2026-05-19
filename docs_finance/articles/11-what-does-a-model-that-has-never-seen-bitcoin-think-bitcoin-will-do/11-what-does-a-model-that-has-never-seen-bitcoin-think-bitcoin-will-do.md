# What does a model that has never seen Bitcoin think Bitcoin will do?

*Article 11 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

[06_pretrained](../../experiments/06_pretrained/) is the experiment in the lineup with the most flattering paragraph in the README and the least flattering numbers on the leaderboard. It runs Amazon's [Chronos-2](https://huggingface.co/amazon/chronos-2) (120 M parameters, encoder-only T5-style) and Google's [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (200 M parameters, decoder-only patch-transformer) — two state-of-the-art **foundation models for time series**. Pretrained on millions of unrelated time series. *Zero-shot* on hourly BTCUSDT — no fine-tuning, no training, weights downloaded straight from HuggingFace Hub.

If a generic prior over plausible time-series futures transfers to crypto, this is the experiment that will show it. If it doesn't, this is the experiment that says "the bottleneck on hourly BTC is the data, not the architecture, not the parameter count" — about as clearly as the lab can ever say it.

The result was the second one. This article is about why that's actually the most informative outcome the foundation-model experiment could have produced.

---

## The numbers

### Chronos-2 — the canonical run

The first 06_pretrained run, with `backend: chronos, hub_model_name: amazon/chronos-2, input_chunk_length: 256`:

```json
{
  "backend": "chronos",
  "hub_model_name": "amazon/chronos-2",
  "input_chunk_length": 256,
  "rows_test": 1608,
  "mae":   262.27,
  "rmse":  397.63,
  "mape":  0.00342,
  "directional_accuracy": 0.5019,
  "cumulative_return":    0.169,
  "annualized_sharpe":    4.292
}
```

**MAE within $2 of the naive last-value floor of $260.50** — Chronos-2 is essentially predicting "the next bar will be near the last bar", with a tiny MAE improvement that's well within sampling noise. **Directional accuracy 0.5019** — statistically indistinguishable from a coin-flip on the bars where the model expressed an opinion. **Sharpe +4.29** — sitting at the drift floor Article 10 quantified, consistent with "no skill, ride the rally".

### TimesFM 2.5 — the follow-up rerun (committed `metrics.json`)

After re-pointing `backend: timesfm, hub_model_name: google/timesfm-2.5-200m-pytorch, input_chunk_length: 256`:

```json
{
  "backend": "timesfm",
  "rows_test": 1608,
  "mae":   268.89,
  "rmse":  405.14,
  "directional_accuracy": 0.4677,
  "cumulative_return":    0.169,
  "annualized_sharpe":    2.44
}
```

A 200 M-parameter decoder-only transformer trained on a different (also vast) collection of time series. *Slightly worse* than Chronos-2 on every metric. dir_acc *below* chance — 0.4677 means the model's directional opinions are anti-correlated with reality on the bars where it expressed one, just enough to drag the strategy below the drift floor. MAE still within $9 of naive.

Both backends are univariate (no covariates), zero-shot (no fine-tuning), running on the same 1,608 test bars under the same harness. The leaderboard, including the foundation models:

| Model | dir_acc | Sharpe | MAE | params |
|---|---:|---:|---:|---|
| ARIMA(1,1,1) | 0.5336 | +7.28 | $260.50 | 3 |
| LSTM | 0.5196 | +4.95 | $274.02 | ~22k |
| MA(24) | 0.5143 | +4.33 | $756.19 | 0 |
| TFT | 0.5053 | +4.59 | $373.70 | ~50k |
| **Chronos-2** | **0.5019** | **+4.29** | **$262.27** | **120 M** |
| XGBoost | 0.4872 | +1.45 | $270.73 | 31 feats × 400 trees |
| **TimesFM 2.5** | **0.4677** | **+2.44** | **$268.89** | **200 M** |
| Naive | NaN | — | $260.50 | 0 |

(Sep–Nov 2024 window, all numbers from committed `metrics.json` files except the Chronos-2 row, which is from the previous backend pointing on the same harness.)

The 120 M – 200 M-parameter foundation models are *better than XGBoost on MAE* and worse than naive on Sharpe-above-drift-floor. Capacity bought them a slightly tighter MAE; it bought them no directional skill at all.

---

## What "level-coherent but direction-blind" actually means

The mechanistic story is clean. Both foundation models output a probabilistic forecast — three quantiles (q10, q50, q90) drawn from `num_samples=200` future trajectories sampled from the model's predictive distribution. The *median* (q50) prediction is what feeds the point-metric leaderboard.

Two empirical observations from the predictions:

1. **The median is centered on `close_{T-1}`.** Across 1,608 test bars, the predicted log-return median is essentially zero, with tiny per-bar variation. The model has learned that the right scale and the right neighborhood for the next bar are *adjacent* to the current bar — exactly what naive does for free. The MAE of $262 is the consequence: the prediction is close enough to `close_{T-1}` that the per-bar absolute error is dominated by `|close_T - close_{T-1}|`, which is the asset's hourly volatility (Article 1's whole point).
2. **The median's *sign* relative to `close_{T-1}` is essentially random.** That's where dir_acc 0.5019 comes from. The model's median prediction is sometimes a hair above the last close, sometimes a hair below, with no detectable directional pattern.

The strategy gate (`pred > ref`) flips on that sign. If the sign is essentially random, the strategy is roughly long half the time, and the Sharpe lands at the drift floor — no skill required. Article 10 walked through this lens in detail; this article is its cleanest illustration.

The reason this is informative: **predicting the level well and predicting the direction well are different tasks.** Naive does the first by definition. Foundation models extend that with calibrated uncertainty bands, which Article 12 will be about. *Nobody so far in this lineup has done the second*, except ARIMA — and ARIMA's edge is small (3 percentage points above chance).

---

## Why "the data is the bottleneck" is finally a defensible claim

Up to this point in the series, every losing model could be explained away. XGBoost (Article 4) lost because of the wrong inductive bias for low SNR. LSTM (Article 8) lost because capacity hurt at low SNR. TFT (Article 9) lost because the cyclical seasonality TFT was built for doesn't exist in hourly BTC. Each loss had a model-side excuse.

Chronos-2 and TimesFM 2.5 are the cleanest possible removal of those excuses:

- **Capacity**: 120 M – 200 M parameters. Way above LSTM's 22k or TFT's ~50k. Way above the *training data size* available in-domain (~6,400 bars) — these models were trained on millions of series; the prior is over a much larger distribution than anything we could have learned from BTC alone.
- **Training data**: millions of unrelated series. Electricity, retail, weather, traffic, manufacturing, web logs, biological signals. If there's a *general* time-series forecasting prior to be had, this is what holds it.
- **Architecture**: state-of-the-art for generic time-series forecasting. Chronos-2 is the encoder-only T5-style approach optimized for probabilistic forecasting; TimesFM is the decoder-only patch-transformer approach. Two distinct successful design philosophies.
- **Inductive bias**: built to handle non-stationary series, multiple scales, long-range dependence, calibrated quantile distributions.

If those models can't extract a directional signal from hourly BTC log-returns, the model side has done its job. The remaining bottleneck is on the data side: **on this asset, at this resolution, with these inputs, the conditional mean of the next-bar return given the past is statistically indistinguishable from zero**. The signal that does exist (the small AR(1) ARIMA captures) is below the noise floor of any prior more general than "tiny linear thing on differences".

That's not a critique of Chronos-2 or TimesFM. They're doing the right thing — predicting a tight, calibrated band centered on the last value, with no false-confidence claims about direction. They're announcing, *correctly*, that they have no information about which side the next bar will land on. The directional accuracy of 0.50 is the model being honest about its own uncertainty.

---

## How the experiment is wired

Worth a short walkthrough because the architecture of `06_pretrained` is unusual relative to `04_lstm` / `05_transformer`.

### Both backends inherit one Darts surface

From [06_pretrained/run.py](../../experiments/06_pretrained/run.py):

```python
def build_model(backend, hub_model_name, model_cfg, quantiles):
    likelihood = QuantileRegression(quantiles=quantiles)
    common_kwargs = {
        'input_chunk_length': int(model_cfg['input_chunk_length']),
        'output_chunk_length': int(model_cfg['output_chunk_length']),
        'hub_model_name': hub_model_name,
        'likelihood': likelihood,
        'pl_trainer_kwargs': pl_trainer_kwargs,
    }
    if backend == 'chronos':
        return Chronos2Model(**common_kwargs)
    if backend == 'timesfm':
        return TimesFM2p5Model(**common_kwargs)
```

Both classes inherit from `darts.models.forecasting.foundation_model.FoundationModel`. The only divergence is the class itself; the rest of the pipeline (Scaler, walk-forward eval, quantile extraction, price reconstruction) is identical. That's the whole point of using `darts` here — Chronos-2 vs TimesFM 2.5 is a `backend:` field in [config.yaml](../../experiments/06_pretrained/config.yaml), not two separate experiments.

### `fit()` is a no-op

```python
model.fit(scaler_target.transform(target_train))
```

This call is required by the Darts API but performs *no weight updates* for foundation models. It locks the schema, triggers the HuggingFace weight download on first invocation (cached at `~/.cache/huggingface/hub/`), and returns. The model that comes out is parameter-identical to the model that went in. Zero-shot, by construction.

### Walk-forward + probabilistic head

```python
preds_s = model.historical_forecasts(
    series=target_full_s,
    start=test_start,
    forecast_horizon=1,
    retrain=False,
    last_points_only=True,
    num_samples=200,
)
```

Same `historical_forecasts(retrain=False)` harness as LSTM and TFT (Article 7). The new piece is `num_samples=200`: at every test bar, the model samples 200 future trajectories from its predictive distribution. We then extract the q0.1, q0.5, q0.9 quantiles of those samples to produce the point prediction (median) and the prediction band (10/90 percentiles).

### Univariate by design

Both backends are run *without* covariates. The 06_pretrained README has the full justification, but the short version: TimesFM 2.5 doesn't accept covariates at all, and we keep Chronos-2 univariate to match. With one backend forced to univariate, giving the other extra inputs would muddy the head-to-head — any difference in metrics would conflate "which prior is stronger?" with "did the covariates help?". Adding covariates to Chronos-2 only is a clean follow-up ablation, listed in Article 13's roadmap.

---

## What this article is *not* saying

A short list because the temptations are real:

- **Not "foundation models are useless".** They aren't. On in-domain non-crypto series with structural seasonality and well-behaved noise, both Chronos-2 and TimesFM 2.5 are state-of-the-art zero-shot forecasters. The [TSMixer / GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) benchmarks show this repeatedly. *On hourly BTC log-returns specifically* the prior doesn't transfer to direction. That's a domain claim.
- **Not "BTC is unpredictable".** It says *the conditional mean of `r_T` given the past 256 bars of `r_T` alone* is statistically indistinguishable from zero. There may be predictable structure in other inputs (funding rate, perp basis, on-chain flow, news, order-book microstructure) or at other resolutions (1d, 1w aggregates). The lab hasn't tried those yet.
- **Not "scale doesn't matter".** It says scale didn't help *on this task with these inputs*. A foundation model with three orders of magnitude more parameters might extract something a 22k-parameter LSTM doesn't — but the foundation models we ran already are at that scale, and they didn't.
- **Not "fine-tuning would change this".** It might. Both `Chronos2Model` and `TimesFM2p5Model` support `enable_finetuning`. Article 13's roadmap will list "fine-tune the foundation models on BTC" as an explicit follow-up. Whether 67 days of training data and a single asset are enough to fine-tune through is an open question; my prior is "no, not at this signal-to-noise ratio", but the experiment hasn't been run.

---

## What survives the foundation-model lens

Combining Article 4 (XGBoost lost), Article 8 (LSTM lost less), Article 9 (TFT lost similarly), and this article (foundation models tied chance), the converged story is:

1. **The directional signal in hourly BTC log-returns is small.** ARIMA's AR(1) coefficient on differenced returns produces dir_acc 0.5336 — three percentage points above chance. That's the size of the signal.
2. **Models with the right inductive bias for the signal extract some of it.** ARIMA bakes "tiny linear thing on differences" into its parameterization. LSTM does too (less directly).
3. **Models without the right inductive bias don't extract more.** Trees that look at unordered features, transformers that look for cyclicality, foundation models that have a generic prior — none of them did better on directional accuracy than ARIMA.
4. **More capacity on the wrong inductive bias produces no improvement and sometimes regression.** Chronos-2 (120 M) ties chance. TimesFM 2.5 (200 M) lands below chance. XGBoost (the deepest config, 800 trees × 7 deep) lands below MA(24).

That's the data-is-the-bottleneck claim, made with the cleanest possible evidence. The next models worth trying are not bigger; they're *differently informed*. ARIMAX with a real exogenous covariate (funding rate, perp basis), or a model on a different time scale where the signal-to-noise might be more favorable. Article 13's roadmap will lay this out.

---

## Reproducing

Pick a backend explicitly in [config.yaml](../../experiments/06_pretrained/config.yaml) — there is no default:

```yaml
backend: chronos          # or 'timesfm'
hub_model_name: amazon/chronos-2
                          # chronos: amazon/chronos-2 | autogluon/chronos-2-small | autogluon/chronos-2-synth
                          # timesfm: google/timesfm-2.5-200m-pytorch
```

Then:

```
make 06_pretrained         # the headline run
make 06_pretrained_sweep   # backends × input_chunk_lengths
```

First run downloads HF weights (~120–800 MB depending on backend, cached at `~/.cache/huggingface/hub/`). Walk-forward over 1,600 test bars takes single-digit minutes on Apple MPS or a recent CUDA GPU; CPU is several times slower.

The sweep is configurable in [sweep.py](../../experiments/06_pretrained/sweep.py)'s `GRID` — useful for asking "does Chronos-2's `chronos-2-small` (28 M) match `chronos-2` (120 M)?" (often yes, on hourly BTC), or "does `input_chunk_length=512` improve over `256`?" (rarely, on hourly BTC).

---

## What's next

Article 12 — *Probabilistic forecasts, finally: what a q10/q50/q90 band changes* — is the methods post about what the calibrated quantile bands give you that point predictions don't. The negative finding worth showing: the q10–q90 spread on hourly BTC is *narrow and centred on the last value*. The quantile loss is doing exactly what it should; there's just no asymmetry to exploit. Foundation models *announce* their uncertainty cleanly, which is informative even when the announcement is "I have no useful asymmetric opinion about the next bar".

Article 13 — *Where this repo goes next* — is the roadmap, and it's mostly built around what *this* article concludes: if the data is the bottleneck, the next experiments are about different inputs, different timescales, and stress-testing what we have rather than reaching for more architectures.

---

*Code: [experiments/06_pretrained/](../../experiments/06_pretrained/) · [06_pretrained/README.md](../../experiments/06_pretrained/README.md) · Repo: [transformer_bitcoin_ai](../../../README.md)*
