# Outline — Where this repo goes next

## One-line pitch
Roadmap post — what's left now that the headline architecture is in. Open threads: TimesFM 2.5 zero-shot to confirm or break the Chronos result, ARIMAX with funding rate / perp basis, fine-tuning the foundation models instead of zero-shot, multi-seed runs to turn point-estimate Sharpe gaps into distributions, and re-running every model on a regime that *isn't* the Sep–Nov 2024 rally. Honest about which of these are "fix the methodology" vs. "chase signal that may not exist."

## Audience
Series readers who've gone all the way through. ML practitioners who want to know what would change a verdict and what wouldn't. Anyone considering forking the repo to extend it.

## Thesis
After twelve articles, the core lab notebook is in place: every model from naive through 200 M-parameter foundation models is in the leaderboard, comparable, reproducible. The honest next phase is *not* "try a bigger model"; it's (a) tightening methodology so the leaderboard rows say more than one number each, and (b) reaching for *different inputs* — funding rate, on-chain flow, calendar/macro events — because the data has been the bottleneck, not the architecture.

## Structure

### 1. The hook — what we know
The series's converged thesis, one paragraph:
- ARIMA(1,1,1) wins. dir_acc 0.5336, Sharpe +7.28.
- Every other trained model — XGBoost, LSTM, TFT, foundation models — is at or near the drift floor on directional accuracy, sometimes below.
- Capacity scaled across four orders of magnitude (3 → 200 M parameters) didn't escape the floor.
- The signal in the inputs we used (past log-returns ± OHLCV ± hour/dow cyclicality) is small. Roughly the size of an AR(1) coefficient on differenced returns.
- That's what we know. The roadmap below is what's left to find out.

### 2. Methodological follow-ups (fix the lab, then re-run)
Things that don't depend on finding new signal — they depend on the lab being more rigorous.

**a. Multi-seed runs** (deep models only). Single-seed deep-model results have several Sharpe-units of seed variance. Right now the LSTM's +4.95 and TFT's +4.59 are point estimates. Re-running each at 5 seeds and reporting median ± IQR would let the series stop pretending those numbers are precise.

**b. Drift-floor reporting** (Article 10's commitment). A `drift_floor.py` Monte Carlo (N random-direction strategies on the same test slice, 95 % CI of resulting Sharpe distribution) would add a `sharpe_above_floor` column to every `metrics.json`. *That* is the Sharpe column that ought to rank the leaderboard.

**c. Multi-window evaluation** (Article 6's commitment). Re-run every model on at least:
- Q1 2024 (Jan–Mar) — choppy, no rally.
- Apr–Aug 2024 — sideways, mid-vol.
- Sep–Nov 2024 — rally (the canonical window).
- A 2025-and-later window once data is in cache.
Report median ± IQR of the leaderboard metrics across windows. Article 6 set up the framing; this is the doing.

**d. Quantile heads on every model** (Article 12's commitment). LSTM and TFT can take `QuantileRegression` likelihoods; ARIMA's `statsmodels` fit carries predictive variance for free. Adding `coverage_80`, `crps`, and `band_width_correlation` columns to the leaderboard makes uncertainty a first-class metric.

**e. Single-bar Sharpe and SE columns** (Article 14's commitment). Annualized Sharpe is shaped by `√8760`. Per-bar Sharpe with a sampling SE makes the size of differences honest. The leaderboard should show both.

### 3. Architecture follow-ups (probably won't move the floor)
Things worth trying for completeness, but the lab's prior is they won't change much.

**f. ARIMA prediction interval extraction.** One-line `extended.get_prediction(...).conf_int(alpha=0.2)`. Free in `statsmodels`.

**g. Fine-tuning Chronos-2 / TimesFM 2.5 on BTC.** Both classes support `enable_finetuning`. With ~6,400 training bars, my prior is fine-tuning won't break the floor — there isn't enough signal to fine-tune *toward*. But it's a clean experiment to commit and the result is informative either way.

**h. Chronos-2 *with* covariates.** TimesFM 2.5 doesn't accept any; Chronos-2 does. A Chronos-2 + past covariates + future covariates run is a clean ablation. If it pulls away from univariate Chronos-2, that says past-covariates carry signal even at this scale; if it doesn't, that's another data point for "no covariate signal in OHLCV".

**i. ARIMAX with one explicit covariate.** Single-feature ARIMAX is the smallest possible test of "does any covariate help on top of AR(1)?". Candidates from cheapest to most ambitious:
   - Volume z-score (already in OHLCV).
   - Realized vol over `[T-24, T-1]` as a regressor on the mean.
   - Hour-of-day dummies (24 binaries).
   These are all data we already have.

### 4. Inputs that aren't in the lab yet (the actual signal-side bets)
The data-is-the-bottleneck thesis says the next signal must come from *different inputs*. Roadmap, in approximate order of cost-to-experiment:

**j. Funding rate** (Binance perp). Hourly. Free from Binance API. A long-standing proxy for crowded longs / shorts in crypto. ARIMAX with funding_rate as a regressor is the smallest version. If it adds dir_acc above ARIMA, that's the first real signal-side win the lab finds.

**k. Perpetual basis** (perp price - spot price). Hourly. Free. A second proxy for positioning. Often correlated with funding but not identical.

**l. On-chain flow** (Glassnode, Coinmetrics). Daily / hourly aggregates of exchange inflows, miner outflows, active addresses, etc. Costs depend on the API tier. Most likely to help on *daily* resolution than hourly, where a lot of on-chain data is too sparse.

**m. Order-book microstructure**. Bid-ask imbalance, queue depth, recent trade aggression. Shorter timescale (sub-minute) than the lab currently runs. Probably its own experiment; this isn't going to slot into the existing harness.

**n. News / sentiment**. Either an LLM-derived sentiment score over X (Twitter) / Reddit, or a structured news-events feed (FOMC, ETF approvals, exchange hacks). Hard to do well; easy to overfit.

### 5. Different timescales
The lab is hourly. The signal-to-noise ratio at *different timescales* is different.

- **Daily.** `interval: 1d` in the YAML. Same harness, completely different signal regime. Daily BTC has known momentum and mean-reversion patterns; daily models routinely have dir_acc above 0.55.
- **Weekly.** Even more autocorrelated. Probably easy. Sample size is the problem (52 bars/year × number of years of training data).
- **Sub-hourly.** 5m / 15m. Microstructure dominates. Different problem. Probably its own repo.

The roadmap should commit to a `interval: 1d` rerun of the existing leaderboard as the lowest-hanging fruit. It's a config-edit; the harness handles the rest.

### 6. Open methodology questions
Things the lab is *not* sure about and would welcome reader pushback on.

- **Are 1,608 test bars enough?** Short answer: marginally. Article 14 will quantify.
- **Is the long/flat strategy gate the right execution rule?** It's deliberately the dumbest one. A tighter `pred > ref + epsilon` gate or a band-driven gate (Article 12) might better isolate signal from noise.
- **Should we report a signed Sharpe?** Some strategies under-trade; some over-trade. The current single-number Sharpe doesn't reveal frequency. A `n_long_bars` column would.
- **Should the harness retrain at week boundaries?** Article 7's "honest cheat" is `retrain=False` for the entire test window. A weekly-refit harness would be a different and more realistic experiment for deployment claims; it would also be 60–80 % more compute per deep-model run.

### 7. What's *not* on the roadmap
Be explicit about deferrals.

- **A trading bot.** This is a research repo, not a service.
- **Multi-asset.** Adding ETH, SOL, etc., would dilute focus. Keep the lab single-asset until the BTC story has a settled methodology.
- **Closer to real-time data.** The Binance vision archive lags by a day or two. That's fine for backtesting; it's not fine for live trading. Out of scope.
- **Reinforcement learning.** Lots of crypto-RL pitches; very few that survive the regime tests Article 6 calls for. Not a planned experiment.

### 8. The cost/value table for the roadmap
Quick decision matrix readers can use.

| Item | Cost | Likely Impact |
|---|---|---|
| Multi-seed runs (a) | 1 day | Methodology-tightening |
| Drift-floor reporting (b) | half-day | Methodology-tightening |
| Multi-window eval (c) | 1–2 days | Article 6's commitment |
| Quantile heads on all models (d) | 1–2 days | Probabilistic leaderboard |
| Per-bar Sharpe + SE (e) | half-day | Article 14's commitment |
| ARIMA prediction intervals (f) | 1 hour | Free |
| Foundation fine-tuning (g) | 1 week + GPU | Likely small |
| Chronos-2 with covariates (h) | half-day | Likely small |
| ARIMAX, single covariate (i) | half-day | **Could matter** |
| Funding rate ARIMAX (j) | 1 day + data work | **Could matter a lot** |
| Perp basis ARIMAX (k) | 1 day + data work | **Could matter a lot** |
| On-chain flow (l) | 1 week + data work | TBD |
| Order-book microstructure (m) | 1 month + data work | New regime, different scope |
| News / sentiment (n) | 1 month + data work | High variance, easy to overfit |
| Daily timescale rerun | 1 day | **Likely best ROI** |
| Weekly / sub-hourly | TBD | Out of scope for v1 |

### 9. The honest summary
- The lab has done what it set out to do for v1.
- The architecture progression (naive → MA → ARIMA → XGBoost → LSTM → TFT → foundation) is in. The leaderboard is real on the canonical window.
- The next year of work is *methodological tightening* (b–e) and *signal-side experiments on different inputs* (i–k) and *a daily timescale rerun*. In that order.
- If a year from now the leaderboard still has ARIMA on top, that's still a meaningful result.

### 10. Closing — open invitation
- The repo is reproducible: `brew install uv && uv sync && make 06_pretrained_direct`.
- Anything in this roadmap that someone wants to add is welcome as a PR.
- The lab is structured to make extension cheap (Article 2). Use it.

## Key code/file references
- All `experiments/*/config.yaml` for the multi-window and timescale reruns.
- [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py) — where new metrics (`coverage_80`, `sharpe_above_floor`, per-bar Sharpe + SE) would land.

## Tone notes
- Concrete. Each item has cost and likely impact.
- Honest about which are "tighten the lab" vs "chase signal that may not exist". The reader should leave knowing exactly what would change a verdict.

## Length target
~1,500–1,800 words.
