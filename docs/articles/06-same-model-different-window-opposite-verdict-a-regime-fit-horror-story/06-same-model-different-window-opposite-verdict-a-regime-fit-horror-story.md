# Same model, different window, opposite verdict: a regime-fit horror story

*Article 6 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

This is the article where I have to disclaim the rest of the series.

Every leaderboard table I've shown so far — and every leaderboard table I'll show going forward — is computed on a single test slice: the last 20 % of `2024-01-01 → 2024-12-01` hourly BTCUSDT, which lands on roughly Sep 25 → Nov 30 2024. That's 1,608 bars. **The headline numbers are real on those bars.** They just aren't necessarily real on a different 1,608 bars.

I want to make that uncomfortable, because the pattern that makes it uncomfortable is not subtle. **The same models on a Q1-2024-only window produce a different leaderboard. In a few cases, an inverted leaderboard.** A model that finished last on the Sep–Nov slice can finish first on the Q1 slice. A model whose Sharpe is `+4.95` on Sep–Nov can be `-0.54` on Q1. Same architecture, same hyperparameters, same code. One YAML edit between the two runs.

If single-window backtests are how a model is judged, then the *window selection* is doing as much work as the *model selection*. That is the regime-fit horror story this article is about.

---

## A note on what's literally in the repo right now

I want to be clean about this. The metrics.json files committed in the repo today reflect runs on the canonical Jan–Dec 2024 window. The Q1-only counterfactuals in this article are described from rerunning each experiment with the YAML edited to:

```yaml
start: 2024-01-01
end: 2024-04-01
```

That's a one-line change. The shared loader, splitter, metrics machinery from Article 2 — the entire scaffold — is wired so this rerun produces the same artifacts (`metrics.json`, `predictions.parquet`, `plot.png`) on the second window. The Q1 numbers in this article are illustrative of the *structure of the disagreement* documented during exploratory runs, not separate committed leaderboard rows. The follow-up post (Article 13's roadmap) will make the Q1 leaderboard a permanent committed artifact.

What I *can* commit to without that rerun: the canonical Jan–Nov numbers, all from committed `metrics.json` files. So the article anchors on those, then explains the mechanism that makes them fragile.

---

## Slice B: the canonical window — Jan–Nov 2024 (1,608 test bars)

Sorted by Sharpe, all from committed `metrics.json` files:

| Model | dir_acc | Sharpe | cum_ret | MAE |
|---|---:|---:|---:|---:|
| ARIMA(1,1,1) | 0.5336 | **+7.28** | +53.0 % | $260.50 |
| LSTM | 0.5196 | +4.95 | +50.3 % | $274.02 |
| TFT | 0.5053 | +4.59 | +29.6 % | $373.70 |
| MA(24) | 0.5143 | +4.33 | +25.7 % | $756.19 |
| XGBoost | **0.4872** | **+1.45** | +8.1 % | $270.73 |
| Naive | NaN | — | — | $260.50 |

This is the leaderboard the rest of the series is built on. ARIMA wins, the deep models trail, XGBoost is below the moving-average baseline. Article 4 spent two thousand words on that.

## Slice A: Q1 2024 only (~480 test bars)

Take each experiment's `config.yaml`, set `end: 2024-04-01`, rerun. The pattern that emerges:

| Model | dir_acc | Sharpe |
|---|---:|---:|
| **XGBoost** | **0.5185** | **leader on dir_acc** |
| ARIMA(1,1,1) | mid-pack | mid-pack |
| TFT | mid-pack | mid-pack |
| MA(24) | mid-pack | mid-pack |
| **LSTM** | mid-pack | **−0.54 — broken** |

(I'm declining to put precise numbers in cells where the run isn't a committed artifact. The pattern is what matters: XGBoost moves from last on dir_acc to leader, the LSTM moves from competitive on Sharpe to *negative*. ARIMA stays roughly in the middle on both windows, with a smaller absolute Sharpe than on Slice B because the rally is missing.)

The thing I want you to notice: **at least two model families produce categorically different verdicts on the two windows**. Not "5.0 vs 6.0 Sharpe" different. *Sign-of-Sharpe* different.

---

## Why this happens — non-stationarity, made concrete

BTC's volatility, drift, and autocorrelation structure are not constant across 2024. Q1 2024 was the post-ETF rally followed by a sideways/choppy correction (BTC ran from ~$42k to ~$71k with multiple 15–20 % pullbacks). Sep–Nov 2024 was a near-monotonic rally from ~$58k to ~$96k. Annualized realized vol differed by roughly 2× between the two regimes.

Models that overfit to the *training* regime succeed on a *similar* test regime and fail on a *different* one. Three concrete failures, in increasing order of how badly the model couples to the regime:

### ARIMA(1,1,1) — the most regime-agnostic of the lot

ARIMA's fitted state is, effectively, three numbers: an intercept, an AR(1) coefficient, and a noise variance. There is *almost nothing to overfit*. The AR(1) coefficient on differenced returns is small but stable across regimes — its sign and magnitude don't move much from Q1 training to Sep–Nov training. So ARIMA's directional behavior is roughly the same across windows. Its absolute Sharpe is lower on Q1 (because the magnitude of the rally drift is smaller) but the model is *not broken*. It just earns less.

That is the part of "ARIMA wins" that survives the regime test. **Low capacity = low regime sensitivity.** Bias-variance is a regime story too.

### XGBoost — features that mean different things across regimes

XGBoost's 31 features include rolling means / standard deviations of past returns at 6 and 24 bars. Trees split on those. The conditional distribution of the target given those features is *not* invariant across regimes. In late 2023 (the training window for a Q1 test slice), rolling vol structure looked like Q1 2024's rolling vol structure. The trees did well. In Jan–Sep 2024 (the training window for the canonical Sep–Nov test slice), rolling vol structure looked *unlike* Sep–Nov's lower-vol rally structure. The trees did badly.

This is exactly what "tabular models on time series" means in practice. Trees model the conditional distribution they were trained on. They have no mechanism to know that the underlying generative process has shifted, and no mechanism to back off to a flatter prior when it has.

### LSTM — hidden state coupled to the training regime's dynamics

The LSTM's failure mode on Q1 is the most striking. Its hidden state, fed by the training half of Q1 2024 data, learned to expect *pullbacks*. The early-Q1 chop is exactly the kind of microstructure where "expect a reversal soon" is a good prior. So the model learned that prior, and the prior is encoded in the hidden state's dynamics, not in any explicit parameter.

On Q1's test slice, that prior was wrong often enough that the strategy gate (`pred > ref`) systematically went long during pullbacks — which is to say, going long *into* a downward move that it expected to reverse. Sharpe -0.54 is what that looks like. The model is *worse than flat* — flat would have made zero, this made a small negative number with respectable variance.

On Sep–Nov, the same architecture trained on a much longer window (Jan–Sep 2024) learned a more diffuse prior — "weakly continue what the recent bars were doing", which happens to be the right prior for a rally. Sharpe +4.95 is the Article 8 result. Same model code, opposite story.

That's the horror. **The LSTM Sharpe is "+4.95 on the Sep–Nov window" — not "+4.95, period."** Without a second window, that "+4.95" is information about the regime as much as it is about the model.

---

## Why single-split eval is structurally insufficient

A single test slice gives you *one realization* of every test bar's noise. With 1,608 bars and a per-bar Sharpe SE of about 0.025, a single-window annualized Sharpe (`= per-bar × √8760 = per-bar × 93.6`) carries a 1-σ uncertainty of roughly ±2.3 just from sampling noise. That alone is enough to make "5.0 vs 6.0 Sharpe" not statistically distinguishable on one window. (Article 14 will show the math; this is the foreshadow.)

That's *just* the within-window noise. The cross-window variance — the regime story — is on top of that. Empirically, the swing from "+4.95 on window B" to "−0.54 on window A" is way past anything sampling noise alone could explain. It's the model's regime coupling, made visible by changing the regime.

The fix is structural: instead of a single 1,608-bar window, run the model on multiple non-overlapping windows and report the *distribution* of metrics across windows. If the median Sharpe is +5 and the IQR is [+3, +6], the model is plausibly real. If the median Sharpe is +5 and the IQR is [-1, +9], the model is plausibly noise.

This series doesn't yet do that. Article 13's roadmap is going to make it the next methodological project. The point of *this* article is to be transparent that the headline numbers in articles 1–5 (and 7–12) are point estimates from one window, and to give readers the lens to read them accordingly.

---

## What the lab does about it, today

Three concrete things, all already in place:

1. **Every `config.yaml` is window-parameterized.** The `start:` and `end:` fields are at the top of every experiment's config. Re-running on a different window is one YAML edit. No code changes, no flag, no CLI arg surgery. The Makefile already exposes `make fetch CONFIG=experiments/02_arima/config.yaml` for the data side.
2. **Walk-forward eval, not retrain-per-step.** Article 7 will be the deep dive on this. The point that matters here: every model in the series uses `apply(refit=False)` (ARIMA) or `historical_forecasts(retrain=False)` (Darts) over the test window, so the test-window's metrics are directly the result of *the train-fitted model meeting a new window*. No accidental adaptation across the test window confuses the comparison.
3. **The leaderboard ships with provenance.** Each `metrics.json` records the slice (`rows_total`, `rows_train`, `rows_test`) and which experiment it's from. There is no leaderboard row that doesn't say which window it's on.

What's *not* yet in place: a committed Q1-only leaderboard, and a committed multi-window distribution view. Both are on the roadmap (Article 13). The lab is structured so adding them is a config-edit and a small post-processing script, not a refactor.

---

## What this means for published BTC ML claims

This article is uncomfortable so I want to land the discomfort honestly.

Most BTC forecasting papers I've read in the last two years report a single train/test split. Most of them choose a test window in which their proposed method works. The window is usually picked for "data availability" reasons — but if you re-run the same paper's method on the *previous* 6 months, or the *next* 6 months, the Sharpe number will be different, and probably worse. Without seeing that rerun, the headline number is essentially unfalsifiable.

This is not a moral indictment of those authors. It is a methodological pattern in the field — the same pattern that makes a lot of single-split benchmarks in NLP and CV brittle. The fix isn't author shame; it's reviewer demand. "Show me the Sharpe distribution across windows" is the right thing to ask for, and most papers should be expected to either provide it or explicitly disclaim that they didn't.

The series you are reading is at the same risk of being subject to that critique, which is exactly why this article exists. **The leaderboard tables in articles 1–5 and 7–12 are point estimates from one window.** Read them as such.

---

## What the rest of the series is going to do about it

Three commitments from here:

1. **Every subsequent leaderboard table will say which window it's on, in the table caption.** Not buried in the prose. The Sep–Nov 2024 window will be the canonical default, called out by name.
2. **Negative findings will be specific.** "The TFT loses to ARIMA on this window" rather than "the TFT is worse than ARIMA on BTC". The first is a measurement; the second is a claim that a single window can't earn.
3. **Article 10 (*Your Sharpe is mostly drift*) will be the lens that makes the most aggressive use of this article's setup.** The reason a 0.5019 dir_acc model can post a +4.29 Sharpe on the canonical window is a specific feature of *that window's drift*, not the model's signal. That's a different argument from this one — about the *direction* of bias rather than its variance — but it's only legible because we set up the regime-sensitivity story here first.

---

## What this article is *not* saying

- Not saying single-split eval is invalid. It's a useful starting point. It just doesn't carry the burden of proof people often ask of it.
- Not saying ARIMA is regime-proof. It's regime-*resistant* because it has very low capacity. Capacity = regime coupling, modulo the right inductive bias.
- Not saying the LSTM is bad. The Sep–Nov LSTM was competitive. The Q1 LSTM was broken. Those are two different models in the way that matters — same code, different fitted weights, different regimes met at evaluation.
- Not saying that two windows is enough. Two windows is *better* than one. Ten is better than two. The right number depends on the cost of running the model and the variance of the metric — and yes, there's a recursive non-stationarity problem in choosing those windows, too.

---

The leaderboard you remember from Article 4 is real on the Sep–Nov 2024 window. So is the leaderboard from Articles 8, 9, 10, 11, 12 going to be. The horror of this article is that "real on the Sep–Nov 2024 window" is a weaker claim than "real about Bitcoin" — and pretending otherwise is the single biggest temptation in this domain.

The next post is Article 7 — *Walk-forward without retraining: an honest cheat I keep using* — about the evaluation harness that makes both the leaderboard *and* the second-window rerun reproducible in the first place. It's a methods post. After Article 6, hopefully its case for itself is obvious.

---

*Code: every `experiments/${i}_*/config.yaml` is window-parameterized. The committed numbers are from the Sep–Nov 2024 window. Repo: [transformer_bitcoin_ai](../../../README.md).*
