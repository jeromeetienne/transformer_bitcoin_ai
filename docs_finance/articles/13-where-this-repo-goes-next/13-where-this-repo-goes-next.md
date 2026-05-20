# Where this repo goes next

*Article 13 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

After twelve articles, here's what the lab knows:

- **ARIMA(1,1,1) wins.** dir_acc 0.5336, Sharpe +7.28, on the Sep–Nov 2024 window. Three numbers, fitted by `statsmodels` in less code than a tweet.
- **Every other trained model — XGBoost, LSTM, TFT, foundation models — is at or near the drift floor on directional accuracy.** Some are below it. Capacity scaled across four orders of magnitude (3 → 200 M parameters) didn't escape the floor.
- **The signal in the inputs we used (past log-returns ± OHLCV ± hour/day-of-week cyclicality) is small.** Roughly the size of an AR(1) coefficient on differenced returns.
- **Most of the Sharpe headlines in this series are partly drift.** Article 10's lens deflates them; only ARIMA's number has a meaningful skill component on top.

That's the v1 lab notebook. The honest next phase is *not* "try a bigger model". It's a mix of (a) tightening the methodology so the leaderboard rows say more than one number each, and (b) reaching for *different inputs* — funding rate, on-chain flow, calendar/macro events — because the data has been the bottleneck, not the architecture. This article is the roadmap, with cost and likely-impact estimates so readers can see which items would actually change a verdict.

---

## Tier 1 — Methodology fixes (tighten the lab, then re-run)

These don't depend on finding new signal. They make the existing leaderboard more honest.

### a. Multi-seed runs on the deep models

**Cost: ~1 day.** **Impact: methodology-tightening; doesn't change the headline.**

Single-seed deep-model results carry several Sharpe units of run-to-run variance. The LSTM's +4.95 and TFT's +4.59 are point estimates with no error bars; rerunning each at 5 seeds and reporting median ± IQR would let the series stop pretending those numbers are precise. Architecturally, `random_state` is already a config field in [04_lstm/config.yaml](../../experiments/04_lstm/config.yaml) and [05_transformer/config.yaml](../../experiments/05_transformer/config.yaml); the work is a small loop in `sweep.py` and a leaderboard column.

### b. Drift-floor reporting

**Cost: ~half-day.** **Impact: changes the way every Sharpe in the series should be read.**

Article 10's commitment. A `drift_floor.py` script does a Monte Carlo of N random-direction long/flat strategies on the same test slice, computes the distribution of resulting Sharpes, and writes a `drift_floor.json` next to each experiment's `metrics.json`. The leaderboard then gets a `sharpe_above_floor` column — the Sharpe that's *not* explained by drift. *That* is the column that ought to rank the leaderboard, and it's currently missing.

Lives in [src/btc_ai/eval/](../../src/btc_ai/eval/) so every experiment shares it.

### c. Multi-window evaluation

**Cost: 1–2 days.** **Impact: Article 6's commitment. Could meaningfully change the ranking.**

Re-run every model on at least:
- **Q1 2024 (Jan–Mar)** — choppy, no rally. The "horror story" window from Article 6.
- **Apr–Aug 2024** — sideways, mid-vol.
- **Sep–Nov 2024** — rally (the canonical window the series uses).
- **Dec 2024 onward** — once data is in cache.

Report median ± IQR of leaderboard metrics across windows. Article 6 set up the framing; this is the doing. The work is a small wrapper around the existing `make NN_*` targets that re-points the YAML's `start:` / `end:` for each window.

### d. Quantile heads on every model

**Cost: 1–2 days.** **Impact: Article 12's commitment. Adds uncertainty as a first-class metric.**

LSTM (`BlockRNNModel`) and TFT (`TFTModel`) both accept `QuantileRegression` likelihoods in Darts; switching from MSE to a quantile head is a one-line change that requires retraining. ARIMA's `statsmodels` fit carries predictive variance — `extended.get_prediction(start=split, end=...).conf_int(alpha=0.2)` gives a Gaussian 80 % band for free.

Once every experiment has q10/q50/q90, the leaderboard adds:
- `coverage_80` (target 0.80) — calibration check.
- `band_width_mean` — average uncertainty width.
- `band_width_corr_realized_error` — does predicted band tightness correlate with realized accuracy?

### e. Per-bar Sharpe and sampling SE columns

**Cost: ~half-day.** **Impact: Article 14's commitment. Makes the size of differences visible.**

Annualized Sharpe is shaped by `√8760`. Per-bar Sharpe with a sampling SE makes the size of differences honest — a per-bar Sharpe of 0.080 on 1,608 bars has SE ≈ 0.025, so the typical "+5 vs +6" annualized comparison is well within sampling noise. The leaderboard should show *both* the annualized number (for marketing-style comparison) and the per-bar number plus SE (for honest comparison).

### Tier 1 summary

After these five items the leaderboard becomes:

| Model | dir_acc ± SE | Sharpe / yr | Sharpe-above-floor / yr | per-bar Sharpe ± SE | coverage_80 | windows |
|---|---|---|---|---|---|---|

Each row would be median across seeds, summarised across windows. That's a much higher-signal table than what the series currently quotes.

---

## Tier 2 — Architecture follow-ups (probably won't move the floor)

Things worth running for completeness. Lab's prior is that they don't change the headline.

### f. ARIMA prediction interval extraction

**Cost: 1 hour.** **Impact: free, completes Article 12's leaderboard.**

Already noted under (d). Mentioned separately because it's a one-line edit in [02_arima/run.py](../../experiments/02_arima/run.py).

### g. Fine-tuning Chronos-2 / TimesFM 2.5 on BTC

**Cost: ~1 week + GPU time.** **Impact: prior is "small".**

Both Darts foundation-model classes support `enable_finetuning`. With ~6,400 training bars and a single asset, my prior is fine-tuning won't break the floor — there isn't enough signal in the BTC slice to fine-tune *toward*. But it's a clean experiment to commit, and the result is informative either way:

- Fine-tuning beats zero-shot meaningfully → there *is* in-domain signal that the generic prior missed.
- Fine-tuning ties zero-shot → the generic prior already extracted everything available.
- Fine-tuning under-performs zero-shot → over-fitting to the small in-domain slice.

Any of those is publishable.

### h. Chronos-2 *with* covariates

**Cost: ~half-day.** **Impact: prior is "small".**

TimesFM 2.5 doesn't accept covariates; Chronos-2 does. The lab deliberately keeps both univariate for a fair head-to-head (Article 11). A separate Chronos-2 + past covariates (volume, OHLC) + future covariates (hour/day cyclical) run is a clean ablation. Compare to univariate Chronos-2 to isolate "do covariates help at the foundation-model scale?".

### i. ARIMAX with one explicit covariate

**Cost: ~half-day per candidate.** **Impact: could matter; lowest-cost signal-side bet using existing data.**

`statsmodels.ARIMA` accepts `exog=` for ARIMAX. One-line lift from [02_arima/run.py](../../experiments/02_arima/run.py). Candidates from cheapest to most ambitious:

- **Volume z-score** (already in OHLCV). `(volume[t-1] - rolling_mean) / rolling_std`. Tests "do volume bursts predict next-bar return?"
- **Realized vol over `[T-24, T-1]`** as a regressor on the mean. Tests "does mean shift with vol regime?"
- **Hour-of-day dummies** (24 binaries). Same input as TFT's cyclical encoding, but in ARIMAX's parametric form.

If any of these add dir_acc above ARIMA's 0.5336, that's the first signal-side win the lab finds.

---

## Tier 3 — Inputs that aren't in the lab yet (the actual signal-side bets)

The data-is-the-bottleneck thesis says the next signal must come from *different inputs*. These are the experiments most likely to change a verdict, in approximate cost-to-experiment order.

### j. Funding rate (Binance perp)

**Cost: ~1 day + data work.** **Impact: could matter a lot.**

Hourly funding rate from Binance's perpetual swap on BTCUSDT. Free from the public API. A long-standing proxy for crowded longs / shorts in crypto: high positive funding → longs are paying shorts → markets are crowded long → reversal risk. ARIMAX with `funding_rate` as a regressor is the smallest version. If it adds dir_acc materially above ARIMA's 0.5336, that's a real signal-side win.

### k. Perpetual basis (perp price - spot price)

**Cost: ~1 day + data work.** **Impact: could matter a lot.**

Free from the same Binance API. A second proxy for positioning. Often correlated with funding rate but not identical — perp basis is a *price* divergence, funding is the *premium* paid to maintain that divergence. ARIMAX with `(perp_close - spot_close) / spot_close` as a regressor.

These two (j and k) are the lab's highest-priority signal-side experiments.

### l. On-chain flow (Glassnode, Coinmetrics)

**Cost: ~1 week + data work + API costs.** **Impact: TBD.**

Hourly / daily aggregates of exchange inflows, miner outflows, active addresses, realized cap, etc. Most likely to help on *daily* resolution rather than hourly, where most on-chain data is too sparse. Probably Article 13's first follow-up after (j) and (k).

### m. Order-book microstructure

**Cost: ~1 month + data work.** **Impact: new regime entirely.**

Bid-ask imbalance, queue depth, recent trade aggression. Sub-minute timescale. The current harness runs hourly; this would be its own experiment, probably its own repo. Probably the highest-information / hardest-to-overfit signal source on this asset, but it's also a very different ML problem.

### n. News / sentiment

**Cost: ~1 month + data work.** **Impact: high variance, easy to overfit.**

Either an LLM-derived sentiment score over X / Reddit / news feeds, or a structured news-events feed (FOMC, ETF approvals, exchange hacks). Hard to do well. Easy to overfit. I'd want to see (j), (k), (l) results before committing here — if those don't move the floor, news/sentiment probably won't either.

---

## Tier 4 — Different timescales

The lab is hourly. The signal-to-noise ratio at *different timescales* is different.

### Daily

**Cost: ~1 day.** **Impact: likely best ROI in the entire roadmap.**

`interval: 1d` in every YAML. Same harness, completely different signal regime. Daily BTC has known momentum and mean-reversion patterns; daily forecasting models routinely have dir_acc materially above 0.55. The lab's ARIMA / LSTM / TFT / foundation harness is interval-agnostic by construction — `periods_per_year_by_interval` (in [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py)) already includes `'1d': 365` for the Sharpe annualization.

This is the lab's #1 follow-up. If hourly says "no signal", the right response is "try a different timescale", not "try a bigger model".

### Weekly

**Cost: ~1 day.** **Impact: probably easy, but sample size is the problem.**

Even more autocorrelated. Sample size is the issue (52 bars/year × few years of training data ≈ a couple hundred bars). A weekly leaderboard is more useful for understanding the structural autocorrelation than for trading.

### Sub-hourly (5m / 15m)

**Cost: TBD.** **Impact: probably its own repo.**

Microstructure dominates. Different problem from "predicting the bar". Probably out of scope for v1 of this lab.

---

## Open methodology questions

Things the lab is *not* sure about and would welcome reader pushback on.

- **Are 1,608 test bars enough?** Short answer: marginally. Article 14 quantifies — 3σ from zero on the canonical Sharpe but with overlapping CIs across most of the leaderboard.
- **Is `pred > ref` the right strategy gate?** Deliberately the dumbest. A tighter `pred > ref + epsilon` gate or a band-driven gate (Article 12) might better isolate signal from noise — at the cost of fewer trades.
- **Should we report a signed `n_long_bars` column?** Some strategies under-trade; some over-trade. The current single-number Sharpe doesn't reveal frequency. Adding `n_long_bars` and `mean_return_when_long` would be informative.
- **Should the harness retrain at week boundaries?** Article 7's "honest cheat" is `retrain=False` for the entire test window. A weekly-refit harness would be a different and more realistic experiment for deployment claims; it would also add 60–80 % more compute per deep-model run.

---

## What's *not* on the roadmap

Be explicit about deferrals:

- **A trading bot.** This is a research repo, not a service. Live trading requires execution infra, slippage modeling, latency budgets, monitoring — none of which is the lab's purpose.
- **Multi-asset.** Adding ETH, SOL, etc., would dilute focus. Single-asset until the BTC story has a settled methodology. *Then* port to other assets to test generalization.
- **Closer-to-real-time data.** The Binance vision archive lags by a day or two. Fine for backtesting; not fine for live. Out of scope for v1.
- **Reinforcement learning.** Lots of crypto-RL pitches; very few that survive the regime tests Article 6 calls for. Not a planned experiment.
- **Hyperparameter sweeps over more models.** The Optuna setup in [04_lstm/optuna_sweep.py](../../experiments/04_lstm/optuna_sweep.py) shows that sweeping doesn't change the headline. More sweeps would be expensive and unlikely to surface a model that beats ARIMA.

---

## Cost / impact table

A rough decision matrix. Cost is "person-time, ignoring already-committed compute"; impact is the lab's prior on whether the experiment changes a verdict.

| # | Item | Cost | Likely impact |
|---|---|---|---|
| a | Multi-seed runs | 1 day | Methodology |
| b | Drift-floor reporting | half-day | Methodology, *high signal-to-noise gain* |
| c | Multi-window eval | 1–2 days | Article 6's commitment |
| d | Quantile heads on all models | 1–2 days | Probabilistic leaderboard |
| e | Per-bar Sharpe + SE | half-day | Article 14's commitment |
| f | ARIMA prediction intervals | 1 hour | Free completion of (d) |
| g | Foundation fine-tuning | 1 week + GPU | Prior: small |
| h | Chronos-2 with covariates | half-day | Prior: small |
| i | ARIMAX, single covariate | half-day per candidate | **Could matter** |
| j | Funding rate ARIMAX | 1 day + data work | **Could matter a lot** |
| k | Perp basis ARIMAX | 1 day + data work | **Could matter a lot** |
| l | On-chain flow | 1 week + data work | TBD |
| m | Order-book microstructure | 1 month + data work | New regime, different scope |
| n | News / sentiment | 1 month + data work | High variance |
| — | Daily timescale rerun | 1 day | **Likely best ROI overall** |

If I had a week, I'd do `b`, `e`, `f`, `c`, and the daily rerun. Two weeks: add `d`, `i`, `j`. A month: add `k`, `l`, multi-seed `a`.

---

## The honest summary

The lab has done what it set out to do for v1. Every model from naive through 200 M-parameter foundation models is in the leaderboard, comparable, reproducible. The headline result — *ARIMA wins, capacity does not escape the floor on this asset and timescale* — is real on the Sep–Nov 2024 window.

The next year of work is **methodological tightening** (b–e), then **signal-side experiments on different inputs** (i–k), then a **daily timescale rerun**. In that order. The deep-architecture frontier is, for this problem, exhausted; the data frontier is wide open.

If a year from now the leaderboard still has ARIMA on top after a multi-window, multi-seed, drift-floor-aware audit and a funding-rate ARIMAX, that's *still* a meaningful result. The lab would have published the most thoroughly-audited "your transformer doesn't beat ARIMA on hourly BTC" claim in the field.

---

## Open invitation

The repo is reproducible: `brew install uv && uv sync && make 06_pretrained_direct`. The structure is set up to make extension cheap (Article 2). Any item from this roadmap that someone wants to add is welcome as a PR — the experiment numbering convention (`07_arimax_funding`, `08_daily_rerun`, …) makes it easy to slot new work in without disturbing the existing leaderboard.

The next post in the series ([Article 14 — *Stop annualizing your Sharpe*](../../articles_todo.md)) is the optional methodological piece prompted by the +7.28 Sharpe headlines. Per-bar Sharpe ≈ 0.080, SE ≈ 0.025 on 1,608 bars — about 3σ from zero, which is the right size of claim to make. The `√8760` multiplier makes the number look like a hedge fund. Useful to write because every BTC ML post on the internet does this and never says it.

After Article 14, the series concludes; the *lab* keeps going.

---

*Code: [README.md](../../README.md) · all `experiments/` · [src/btc_ai/](../../src/btc_ai/) · Repo: [transformer_bitcoin_ai](../../../README.md)*
