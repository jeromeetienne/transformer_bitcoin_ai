# Outline — Your Sharpe is mostly drift: separating skill from beta in a rising market

## One-line pitch
The cleanest cross-cutting insight, made unmissable by [06_pretrained](../../experiments/06_pretrained/): a `dir_acc 0.5019` zero-shot model posts `Sharpe +4.29` on this window. **A chance-level directional signal produced a number that looks like skill.** Any long/flat strategy that *occasionally* says "long" — even at random — captures part of BTC's $58k → $96k rally. Apply the same lens retroactively to LSTM (+4.95) and TFT (+4.59). Only ARIMA's +7.28 has a directional signal strong enough to be doing real work on top of the drift.

## Audience
Anyone who has ever read a "AI predicts BTC, +X % Sharpe" headline and not paused. Series readers who've watched the Sharpe column accumulate across articles 3, 4, 8, 9 and want to know how much of any of it is real.

## Thesis
Sharpe in a rising market is **noisy times signal plus a drift baseline**. The drift baseline equals "the rally's contribution to a long-flat strategy that goes long occasionally". If the directional accuracy is at chance, the entire Sharpe is drift — and the model has not earned it. Only models whose dir_acc is *materially* above chance are doing skill-driven work on top of the drift.

## Structure

### 1. The hook — a 0.5019 dir_acc model with +4.29 Sharpe
- Quote the [06_pretrained](../../experiments/06_pretrained/) Chronos-2 result as described in the original articles_todo.md description: dir_acc 0.5019, Sharpe +4.29.
- (The currently-committed metrics.json reflects the more recent TimesFM run with dir_acc 0.4677 / Sharpe +2.44 — the article will note both runs and use the Chronos-2 numbers as the canonical illustration of the "chance-level dir_acc with a real Sharpe number" pattern.)
- The provocation: *this is what beta in a rising market looks like in a Sharpe column*.

### 2. The mechanism — what the strategy does at chance directional accuracy
- The strategy gate: long whenever `pred > ref`, else flat. No costs.
- If `pred > ref` is hit roughly half the time *at random*, the strategy is long for ~50 % of bars.
- During Sep–Nov 2024 — a rally — those random long-bars on average have positive realized return.
- Mean of strategy returns is positive. Sharpe is positive. **No skill required.**
- This is *exactly* what a long-flat strategy with no signal looks like in a trending market.

### 3. Quantifying the floor
- Estimate the "long-during-rally" Sharpe from a synthetic random-direction baseline.
- The math: per-bar realized return is `r_T = log(close_T / close_{T-1})` ≈ small positive on average over Sep–Nov 2024.
- A strategy that goes long randomly with probability p captures `p * mean(r_T) - 0` per bar, with stddev `sqrt(p * (1-p)) * stddev(r_T)`.
- For p = 0.5, mean(r) ≈ small positive, the per-bar Sharpe contribution is fixed by the rally's drift, not the model.
- Annualizing × √8760: the floor lands around **Sharpe +3 to +5** for any reasonable random-direction model on this window. (Exact number depends on the exact realized returns and which bars random happens to be long on; Article 14 will be more careful about uncertainty.)
- That's the *baseline*. Any model with `dir_acc ≈ 0.50` and `Sharpe ∈ [+3, +5]` is sitting on this baseline.

### 4. Apply the lens — the leaderboard, deflated
| Model | dir_acc | raw Sharpe | drift floor | skill component |
|---|---:|---:|---:|---:|
| ARIMA(1,1,1) | 0.5336 | +7.28 | ~+4 | **substantial** |
| LSTM | 0.5196 | +4.95 | ~+4 | small |
| TFT | 0.5053 | +4.59 | ~+4 | indistinguishable from drift |
| Chronos-2 | 0.5019 | +4.29 | ~+4 | indistinguishable from drift |

(The "drift floor" column is illustrative; the real estimate would come from a Monte Carlo simulation of random-direction strategies on the same test slice. The article will be transparent about that.)

### 5. The retrospective humbling
- LSTM's +4.95 — the deep-learning headline from Article 8 — is mostly drift. The skill component is small (~+1 in Sharpe units).
- TFT's +4.59 (Article 9) is *indistinguishable from drift*. The transformer's contribution to the Sharpe headline is approximately zero; the +4.59 is what a chance-level directional model gets on a rallying window.
- The number that survives this lens is **ARIMA(1,1,1) +7.28**. A dir_acc of 0.5336 — three percentage points above chance — produces a Sharpe well above the drift floor. *That* is real signal.

### 6. Why this matters for paper-reading
- A lot of published BTC ML claims are point-Sharpe-on-one-window numbers in [+3, +5]. By this lens, those claims are *consistent with the model having no skill at all on a rising window*.
- The fix is not "use a better metric than Sharpe". Sharpe is fine. The fix is **always pair Sharpe with directional accuracy**, and treat any Sharpe >0 with dir_acc ≈ 0.50 as "consistent with the regime".
- Or: report `Sharpe - drift_floor` as the skill-attributable Sharpe. The lab will start doing this in follow-up posts.

### 7. The Q1-2024 sanity check (forward-reference Article 6)
- The `drift floor` is *regime-specific*. On a chopping or falling regime, "go long randomly" loses money on average and the floor is *negative*.
- That's why Article 6's Q1 rerun is the punchline: the LSTM that posts +4.95 on the rally is *broken* (-0.54) on Q1, because there's no rally for the chance-direction component to ride.
- ARIMA's directional skill, by contrast, should mostly survive the regime change — its Sharpe will be lower on Q1 (less rally to amplify the small positive expected return per long-bar) but still meaningfully positive.

### 8. The honest reporting standard
A short prescriptive section.
- **Always report `dir_acc` next to Sharpe.** A Sharpe headline without a dir_acc is unreadable.
- **Always state the test regime.** "Sharpe +4 on Sep–Nov 2024" is informative; "Sharpe +4 on BTC" is not.
- **Report on multiple regimes when possible.** Multi-window distributions are how you separate signal from beta.
- **Estimate the drift floor.** Synthetic random-direction baselines are cheap; running 1000 of them on the test slice and taking the 95 % CI gives you an honest "what should we have expected at zero skill?" number.

### 9. The deeper point — Sharpe is not portable across regimes
- Sharpe was built for stationary return distributions. Hourly BTC log-returns aren't stationary.
- A "Sharpe of 5" is a different claim on a rising window than on a falling one. The number doesn't carry the regime metadata, so the claim becomes ambiguous as soon as it leaves the test window.
- This isn't a Sharpe critique; it's a reminder that the metric needs the context.

### 10. Closing — what survives
- ARIMA(1,1,1) is the only model in the trained lineup whose Sharpe is detectably above the drift floor. Three numbers, one positive AR(1) coefficient, and a handful of percentage points of directional skill above chance.
- Every other trained model in this lab is, on this window, indistinguishable from "long-flat with no signal".
- That doesn't mean those models are bad. It means *we don't have evidence they're doing something the rally isn't already doing for free*.
- Tease Article 11 (Chronos-2 zero-shot is the cleanest illustration of the chance-direction-with-nominal-Sharpe pattern, used here because that pattern is the entire article).

## Key code/file references
- [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py) `strategy_returns`, `annualized_sharpe`
- [experiments/06_pretrained/results/metrics.json](../../experiments/06_pretrained/results/metrics.json) — the canonical "chance-direction with positive Sharpe" data point (in either Chronos-2 or TimesFM run; both qualify)
- All of `experiments/*/results/metrics.json` — the leaderboard the lens is applied to

## Tone notes
- This is the article that tells the reader to be uncomfortable with most of the numbers in the rest of the series. Don't soften it.
- Don't be smug about ARIMA winning. The point isn't that ARIMA is amazing; it's that the bar most models clear is *the rally itself*, not skill.

## Length target
~1,800–2,200 words.
