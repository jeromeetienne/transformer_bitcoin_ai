# Outline — Stop annualizing your Sharpe

## One-line pitch
Short methodological piece prompted by the +7.28 Sharpe headlines. Per-bar Sharpe ≈ 0.080 with SE ≈ 0.025 on 1,608 bars — about 3σ from zero. The `√8760` multiplier makes the number look like a hedge fund. Useful to write because every BTC ML post on the internet does this and never says it.

## Audience
Anyone who has ever read a "Sharpe of 5" headline on a hourly-bar backtest and felt like that was a lot. Series readers who want to see what ARIMA's +7.28 actually means in raw units.

## Thesis
Annualized Sharpe on hourly bars multiplies the per-bar number by 93.6 (`√8760`). That makes legitimate but small per-bar effects look gigantic. Per-bar Sharpe with a sampling-noise SE is the honest unit: it tells you *how many standard errors above zero* the strategy's per-bar return is, on the bars you actually have. ARIMA's "+7.28 Sharpe" translates to "per-bar Sharpe 0.078 ± 0.025 on 1,608 bars, ~3.1σ from zero". That's *real* — but it's a different shape of claim from the annualized number.

## Structure

### 1. The hook — ARIMA's +7.28 in different units
- ARIMA(1,1,1) on the Sep–Nov 2024 window:
  - per-bar mean strategy return ≈ +0.039 % (estimated from cum_ret +53 % over 1,608 bars)
  - per-bar stddev of strategy returns ≈ ~0.50 % (rough)
  - per-bar Sharpe ≈ +0.078
  - SE on per-bar Sharpe ≈ 1 / √1608 ≈ 0.025
  - z-score ≈ 0.078 / 0.025 ≈ **3.1**
  - Annualized: × √8760 ≈ × 93.6 → +7.28
- Same model. Two different presentations. The annualized one is the marketing version; the per-bar + SE one is the honest version.

### 2. The annualization formula and what it assumes
- `Sharpe_annualized = sqrt(periods_per_year) * mean(r) / stddev(r)`.
- Implicit assumption: per-bar returns are i.i.d. with constant mean and variance.
- Hourly BTC returns *aren't* i.i.d. — there's volatility clustering, regime drift, seasonality (or, per Article 9, lack thereof).
- For low autocorrelation in the strategy returns, the formula is approximately correct. For higher autocorrelation, it overstates the annualized number.
- The lab uses the formula as a convention because everyone does. The article is about being honest that the convention has assumptions.

### 3. The sampling SE on per-bar Sharpe
- For a sample of N bars with per-bar Sharpe `s`, the SE is approximately `sqrt((1 + s^2/2) / N)` (rough; Lo 2002).
- For `s ≈ 0.078, N = 1608`: SE ≈ `sqrt((1 + 0.003) / 1608)` ≈ 0.025. Round to 0.025.
- That's what makes ARIMA's per-bar Sharpe of 0.078 a roughly 3σ claim. Real, but at the edge of "publishable signal" not "obviously skill".

### 4. The claim "Sharpe ≈ 5" on hourly data, deflated
- Per-bar Sharpe of 0.05 / 93.6 = 0.0005, on 1,608 bars: z ≈ 0.0005 / 0.025 ≈ 0.02. *Not statistically distinguishable from zero.*
- Wait — that's wrong. Let me redo: annualized 5 means per-bar = 5 / 93.6 = 0.0535. z = 0.0535 / 0.025 ≈ 2.1. So a single-window annualized Sharpe of 5 is about 2σ from zero on 1,608 bars — at the edge of "real but marginal".
- An annualized Sharpe of 7 (ARIMA) is about 3σ. Annualized Sharpe of 4 is about 1.6σ — *not even one-tail significant*. That's a real result on every Sharpe headline in [+3, +5] in this lab and most of the literature.

### 5. The 1,608-bar problem
- 1,608 hourly bars is roughly 67 days. A reasonably-sized backtest by hourly-data standards.
- It's still small for SE-on-Sharpe purposes. SE ≈ 0.025 doesn't shrink fast — to halve it you need 4× the bars, i.e. ~270 days.
- That's why Article 6 cares about multi-window: more bars across non-overlapping regimes give you both a better point estimate *and* a regime-noise estimate.

### 6. The right reporting standard
A short prescriptive section.
- **Report per-bar Sharpe alongside annualized Sharpe.** Both numbers are useful; only one is misleading by itself.
- **Report the SE on per-bar Sharpe.** ~0.025 for 1,600 bars is a useful number for readers to internalize.
- **Report the z-score.** "Sharpe = 7.28 (per-bar 0.078 ± 0.025, z ≈ 3.1)" is informative; "Sharpe = 7.28" is dramatic.
- **Note the autocorrelation in the strategy returns.** If significant, the SE formula above understates uncertainty; corrections like Newey-West are appropriate.

### 7. Reapply the lens to the leaderboard
| Model | Sharpe (annual) | per-bar | per-bar SE | z |
|---|---:|---:|---:|---:|
| ARIMA(1,1,1) | +7.28 | 0.078 | 0.025 | ~3.1 |
| LSTM | +4.95 | 0.053 | 0.025 | ~2.1 |
| TFT | +4.59 | 0.049 | 0.025 | ~2.0 |
| MA(24) | +4.33 | 0.046 | 0.025 | ~1.8 |
| Chronos-2 | +4.29 | 0.046 | 0.025 | ~1.8 |
| XGBoost | +1.45 | 0.015 | 0.025 | ~0.6 |
| TimesFM 2.5 | +2.44 | 0.026 | 0.025 | ~1.0 |

(SE is approximate; z is "per-bar Sharpe / per-bar SE".)

The lens is brutal: most of the deep-model rows are at ~1.8–2.1σ — at the edge of "real, but marginal". XGBoost and TimesFM 2.5 are sub-1σ — indistinguishable from zero. **Only ARIMA's row is unambiguously a real signal.**

### 8. Why the convention exists
A short, honest defense.
- Annualized Sharpe is comparable across timescales (1h vs 1d vs 1w strategies). Per-bar Sharpe isn't — a per-bar Sharpe of 0.05 means very different things at hourly vs daily resolution.
- It's a *convention*. The convention is fine. The dishonesty is using it without the SE / z context.

### 9. The deeper lesson
- A backtest's annualized Sharpe is a marketing summary. It's the right number to *quote* but the wrong number to *trust*.
- Trust is built by: per-bar Sharpe + SE, multi-window distribution, drift-floor adjustment (Article 10), regime sensitivity (Article 6), calibration of any uncertainty bands (Article 12).
- The lab will start reporting this every time, alongside the annualized number it currently quotes.

### 10. Closing — the series concludes
- This was the optional fourteenth piece. The lab notebook is now thirteen substantive articles plus this methodological footnote.
- The headline ARIMA win is real: per-bar Sharpe 0.078 ± 0.025, ~3σ from zero, on this window. Smaller than +7.28 made it sound. Still real.
- The deep-model wins are mostly noise.
- The lab continues; the series ends.

## Key code/file references
- [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py) `annualized_sharpe` — the function in question
- All `experiments/*/results/metrics.json` — the reported numbers

## Tone notes
- Short, sharp, pedagogical. Don't be condescending.
- Show the math. The article should be 1,000–1,400 words tops.

## Length target
~1,000–1,400 words.
