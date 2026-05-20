# I built the transformer my repo is named after. ARIMA still won.

*Article 9 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

The repository is called `transformer_bitcoin_ai`. This is the post about the transformer.

In [05_transformer](../../experiments/05_transformer/) I fit a Temporal Fusion Transformer — Darts' [`TFTModel`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html). 4 attention heads, 48-bar input chunk, hidden size 32, 1 LSTM layer in encoder/decoder, dropout 0.1. Multivariate past covariates (volume, OHLC range and body). Cyclical *future* covariates: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` — deterministic time features the model can read at any future bar. ~50k+ trainable parameters. The architecture I'd reach for if I genuinely thought hourly BTC had intraday or weekly seasonality and wanted a model that could find it.

`make 05_transformer` produces ([05_transformer/results/metrics.json](../../experiments/05_transformer/results/metrics.json)):

```json
{
  "experiment": "05_transformer",
  "rows_test": 1607,
  "mae":  373.70,
  "rmse": 546.68,
  "directional_accuracy": 0.5053,
  "cumulative_return":   0.296,
  "annualized_sharpe":   4.586
}
```

That number lands in this leaderboard position:

| Model | dir_acc | Sharpe | cum_ret | MAE |
|---|---:|---:|---:|---:|
| ARIMA(1,1,1) | 0.5336 | +7.28 | +53.0 % | $260.50 |
| LSTM | 0.5196 | +4.95 | +50.3 % | $274.02 |
| **TFT** | **0.5053** | **+4.59** | **+29.6 %** | **$373.70** |
| XGBoost | 0.4872 | +1.45 | +8.1 % | $270.73 |
| Naive | NaN | — | — | $260.50 |

(Sep–Nov 2024 window, same as every other article in this series.)

The transformer the repo was nominally built around finishes third out of four trained models, behind the LSTM and well behind a linear model from 1970. That is the most honest thing this lab could possibly publish, so this is the post that publishes it.

---

## What TFT can do that LSTM can't

TFT is not a vanilla transformer. It is the architecture you'd choose if your problem had real time-series structure — seasonality, interpretable feature importance, multi-horizon support, mixed continuous and categorical inputs. Concretely, four mechanisms beyond what the LSTM in Article 8 had:

1. **Attention** — at every forecast step, the model can look at any prior bar in the input chunk with learned weights, rather than relying on hidden-state recurrence alone. Long-range dependencies become as cheap as short-range ones.
2. **Future covariates** — TFT consumes deterministic time features at *future* bars. The model is told what hour-of-day and day-of-week it's predicting *for*, not just what's in the input history. This is the killer feature for any series with cyclical structure (e.g. hourly electricity demand).
3. **Variable selection networks** — per-input gating decides which features matter at each step, learned end-to-end. The model can effectively zero out a covariate it finds useless.
4. **Interpretable masked attention** — temporal locality is preserved; future bars are masked; attention weights are inspectable as feature-importance proxies.

If hourly BTC log-returns had any of: a meaningful intraday cycle, a meaningful weekly cycle, a meaningful long-range dependency, or a covariate-driven structure, TFT is the architecture that would find it. The lab's null hypothesis going into Article 9 was: *some of those exist; let's see how much.*

---

## The result: the future covariates didn't pay off

Compare the LSTM and the TFT on the same data slice:

| | dir_acc | Sharpe | MAE | future cov? | attention? |
|---|---:|---:|---:|:-:|:-:|
| LSTM (Article 8) | **0.5196** | **+4.95** | $274.02 | — | — |
| TFT (this article) | 0.5053 | +4.59 | $373.70 | hour/dow cyc | yes |

TFT has every mechanism the LSTM has *and* future covariates *and* attention. Its directional accuracy is lower. Its Sharpe is lower. Its MAE is meaningfully worse — $99 per bar of *additional* per-bar error vs LSTM, on a $260 floor. The model is more confidently wrong about price level *and* worse at calling direction.

That's not a scenario where the architecture is doing better than the LSTM and being penalized by some other factor. It's the architecture *not* doing better. Adding attention and future covariates over LSTM did not produce more directional skill. The TFT is the architecturally richer model and the leaderboard column it controls is the worst directional metric of any trained model in the lineup.

---

## The sweep — capacity vs. Sharpe is flat

`make 05_transformer_sweep` writes [05_transformer/results/sweep.csv](../../experiments/05_transformer/results/sweep.csv). Six configurations:

| icl | hidden | heads | layers | dropout | dir_acc | Sharpe | MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 16 | 2 | 1 | 0.1 | 0.5028 | +4.37 | $398.59 |
| 48 | 32 | 4 | 1 | 0.1 | 0.5053 | +4.59 | $373.70 |
| 48 | 32 | 4 | 2 | 0.2 | 0.4984 | +4.12 | $420.61 |
| **48** | **64** | **8** | **1** | **0.1** | **0.5109** | **+4.83** | $392.46 |
| 96 | 32 | 4 | 1 | 0.1 | 0.4984 | **+3.45** | $409.96 |
| 96 | 64 | 8 | 2 | 0.2 | 0.5047 | +4.47 | $395.20 |

Across six configurations, the **Sharpe range is +3.45 to +4.83** — a spread of 1.4 in annualized Sharpe units, or about 0.015 in per-bar Sharpe. The directional accuracy range is 0.4984 to 0.5109. Both ranges are within what we'd expect from sampling noise alone on 1,608 bars. **Capacity vs. Sharpe is essentially flat.**

The configurations are not the same. The smallest (`24, 16, 2, 1, 0.1`) has tens of thousands of fewer parameters than the largest (`96, 64, 8, 2, 0.2`). The largest has more attention heads, a wider hidden state, more LSTM layers, longer context. If TFT's mechanisms had something to extract from this asset, the larger configurations would pull away. They don't.

---

## The calendar-blindness story

This is the most informative piece of evidence in the article, so I want to land it slowly.

TFT's distinctive capability over LSTM is **future covariates** — `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`. Those four columns are deterministic functions of the timestamp; the model can compute them for any future bar without leakage. They are how the model would learn "at 9 AM UTC volatility is X" or "Sunday afternoons return Y on average".

If hourly BTC log-returns had meaningful intraday or weekly seasonality, you would expect the following pattern in the sweep:

- The configurations with more parameters dedicated to processing covariates (wider `hidden_continuous_size`, wider `hidden_size`, more attention heads) should *systematically* outperform the configurations with less.
- The relationship between capacity and Sharpe should be monotonically positive within the sweep range.
- The wider configurations should pull away on directional accuracy specifically, because the seasonality is a directional bias (e.g. "tend to go up at weekend opens").

What the sweep actually shows:

- The wider config (`hidden=64, heads=8`) at `(48, 64, 8, 1, 0.1)` is best — Sharpe +4.83. Better than the narrower `(48, 32, 4, 1, 0.1)` at +4.59. Improvement: 0.24 in annualized Sharpe.
- But the second wider config `(96, 64, 8, 2, 0.2)` is *worse* than the smaller `(48, 32, 4, 1, 0.1)`. Improvement: -0.12.
- The deepest config `(48, 32, 4, 2, 0.2)` is worse than its single-layer counterpart by 0.47 in Sharpe.
- The `icl=96` configurations are systematically worse than their `icl=48` counterparts — *more historical context decreases performance.*

That isn't the pattern you'd see if the model were exploiting cyclical inputs. It's the pattern you'd see if the future covariates were essentially noise — and the small Sharpe variation across configurations were tracking sampling-noise differences in optimization.

**Conclusion: at the hourly resolution, there is no detectable cyclical seasonality in BTC log-returns that a TFT can exploit.** That's a substantive empirical claim about hourly BTC, supported by the sweep — not a claim about TFT.

There may be cyclicality at *other* resolutions (Asian-session vs US-session daily, weekend-vs-weekday weekly aggregates), but the strategy gate at hourly resolution doesn't see it. The lab is structured to make a daily-resolution rerun a one-line config edit (`interval: 1d`); whether it would change this finding is an open question for Article 13's roadmap.

---

## Width beat depth

A second clean signal in the sweep:

- `(48, 64, 8, 1, 0.1)` — wider, single-layer — best Sharpe (+4.83).
- `(48, 32, 4, 2, 0.2)` — narrower but deeper — worse (+4.12).
- `(96, 64, 8, 2, 0.2)` — wider *and* deeper — worse than the wider-single-layer (+4.47).

This is consistent with Article 8's LSTM finding (`n_rnn_layers=2` regressed at icl=96). Across both deep models, **extra capacity along the time dimension** — deeper recurrent stacks, longer input chunks — *doesn't help and sometimes hurts*. Extra capacity across features (more attention heads, wider channel) helps a tiny amount.

The clean reading: the model is not learning a long-range pattern. It is learning something close to "weakly continue what `r_{T-1}` was doing" with a small per-feature adjustment. The recurrence depth is unused; the attention range is unused. The width of the inference at the *current* step pays a marginal benefit, presumably because it gives the variable-selection network slightly more room to ignore the inputs that don't matter.

In a sentence: **on this asset, TFT's capacity to model time-extended structure is structurally underutilized, because the time-extended structure isn't there.**

---

## Why TFT loses to LSTM, mechanistically

A short version:

- LSTM's predictions are produced by a 22k-parameter recurrent stack, fit with MSE loss on log-returns. Predictions are typically near zero with mild magnitude reflecting recent return variability.
- TFT's predictions are produced by a 50k+-parameter network with attention, gating, and future covariate ingestion. With more capacity to express larger predictions, TFT's predictions are larger in magnitude — MAE $373.70 vs $274.02 for LSTM is the diagnostic. The model is making *bigger* predictions, more confidently, and being wrong about magnitude more often.
- The strategy gate (`pred > ref`) flips on prediction sign. TFT's predictions, being larger and more variable, flip sign on noise more often than LSTM's. Directional accuracy lands lower.

This is the same mechanism Articles 4 and 8 surfaced: capacity hurts at low SNR. TFT is the most capacity-rich trained model in this series and it produces the worst directional accuracy of any trained model — exactly the picture the mechanism predicts.

---

## The moral discomfort of the title

I want to land the discomfort honestly because it's the article's point.

I did not write a paper claiming TFT beats ARIMA on BTC. I wrote a lab notebook showing TFT loses to ARIMA on BTC. The transformer in this repo is *not bad* — it converges, it survives the harness, the predictions are sensible, the implementation is competent. It's just *outclassed* by a 1970 linear model with fewer than ten parameters on a near-random-walk asset.

That is the data point you cannot get from a vendor blog post. The vendor blog post — and the literature, by-and-large — describes TFT's architectural strengths in detail and benchmarks them on series where those strengths pay off (UCI retail demand, energy, weather). On those series, TFT genuinely beats ARIMA. On hourly BTC log-returns it doesn't. The honest framing is: **transformer architectures earn their keep where the structure they're built to model exists**, and on this asset most of that structure is absent.

The repo is named `transformer_bitcoin_ai` because that was the headline experiment when I started. Now that it's run, the appropriate response is to publish the loss, document the mechanism, and refuse the temptation to redefine the metric. ARIMA(1,1,1) wins on the leaderboard column we set up in Article 1 and refused to move thereafter. The transformer is the most expensive way to lose to it.

---

## What this result is *not* saying

- **Not "transformers don't work on time series."** TFT is excellent on retail demand, energy, weather. Most series where industry forecasting matters have temporal structure that TFT was built for. Hourly BTC log-returns is one of the cases where that structure is absent.
- **Not "future covariates are useless."** They're not. They're useless *here* because the underlying data has no cyclical structure to exploit. Toss the same architecture at hourly electricity demand and `hour_sin/cos` will pay off massively.
- **Not "more capacity always hurts."** It says more capacity didn't help, which is a weaker and supported claim. The same architecture trained on a series with stronger signal would benefit from the same capacity.
- **Not the end of the deep-learning case.** Article 11's foundation models (Chronos-2, TimesFM 2.5) are the next escalation: 120 M / 200 M parameters trained on millions of unrelated series, dropped onto this slice zero-shot. The thesis is the same as this article's: *the data is the floor*. If the foundation models also fail to beat ARIMA (and they do), the conclusion isn't "TFT was bad" — it's "the signal isn't reachable through any prior we've tried".

---

## Reproducing

```
make 05_transformer
make 05_transformer_sweep   # 6-config sweep -> results/sweep.csv
```

Single-digit minutes on Apple MPS for the headline run; under an hour for the full sweep. No HuggingFace download — the TFT is trained from scratch, like the LSTM.

The architecture knobs all live in [05_transformer/config.yaml](../../experiments/05_transformer/config.yaml):

```yaml
model:
  input_chunk_length: 48        # bars of history fed to the transformer
  hidden_size: 32               # transformer hidden dim
  lstm_layers: 1                # depth of TFT's encoder/decoder LSTM stack
  num_attention_heads: 4
  dropout: 0.1
  hidden_continuous_size: 8     # GRN size for continuous covariate processing
  add_relative_index: false     # we provide real future_covariates instead
  full_attention: false         # use TFT's interpretable masked attention
```

The sweep is the `GRID` constant at the top of [sweep.py](../../experiments/05_transformer/sweep.py); editing it is one line, and the harness handles the rest.

---

## What's next

Two posts that deflate the +4.59 number from different directions:

**Article 10 — *Your Sharpe is mostly drift*** is the cross-cutting lens that will show how much of every Sharpe number in this series is BTC's own rally rather than the model's signal. The +4.59 TFT Sharpe is, on inspection, mostly the long-during-rally floor. That doesn't make TFT *worse*; it makes the absolute Sharpe number *smaller* in the sense that mattered for trading-relevance. Article 10 will quantify it.

**Article 11 — *What does a model that has never seen Bitcoin think Bitcoin will do?*** is the foundation-model post. Chronos-2 and TimesFM 2.5, dropped onto this slice zero-shot, with no fine-tuning. The first time the lineup answers the question "is BTC unusually hard, or are we just bad at it?". The result is the same as this article's. The data is the floor.

---

*Code: [experiments/05_transformer/](../../experiments/05_transformer/) · [05_transformer/results/sweep.csv](../../experiments/05_transformer/results/sweep.csv) · Repo: [transformer_bitcoin_ai](../../../README.md)*
