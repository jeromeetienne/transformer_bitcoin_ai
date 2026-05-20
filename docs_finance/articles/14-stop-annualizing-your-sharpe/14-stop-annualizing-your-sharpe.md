# Stop annualizing your Sharpe

*Article 14 (the optional methodological footnote) of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

ARIMA's headline result in this series is **annualized Sharpe +7.28**. That's the number I've been quoting since Article 4. It's the number that beats the LSTM (+4.95), the TFT (+4.59), the moving-average baseline (+4.33), and the foundation models (+4.29 Chronos-2, +2.44 TimesFM). It's a hedge-fund-shaped number.

Translated to per-bar units, that "+7.28" is a per-bar Sharpe of about **0.078** with a sampling-noise standard error of about **0.025** on 1,608 test bars. That's a z-score of roughly **3.1** above zero. *Real, but a different shape of claim than "+7.28" suggests.* This article is the math for why those two numbers are the same number and why the lab should report both.

---

## The annualization formula

From [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py):

```python
def annualized_sharpe(strat_ret, periods_per_year):
    sd = strat_ret.std(ddof=1)
    return float(np.sqrt(periods_per_year) * strat_ret.mean() / sd)
```

For 1h bars, `periods_per_year = 8760`, so the annualization multiplier is `√8760 ≈ 93.6`. The mean and stddev are computed on per-bar strategy returns over the test window.

That formula assumes per-bar returns are i.i.d. with constant mean and variance. Hourly BTC returns aren't fully i.i.d. — there's volatility clustering, regime drift, slight autocorrelation. For low autocorrelation in the *strategy* returns, the formula is approximately right; for higher autocorrelation, it overstates the annualized number relative to its true uncertainty. The lab uses the convention because everyone does. This article is about being honest that the convention has assumptions, and that the resulting number is bigger than the underlying signal.

---

## The per-bar Sharpe and its sampling SE

The per-bar Sharpe is just `mean / stddev` on the per-bar strategy returns — no `√8760`. For ARIMA(1,1,1) on the canonical window, with cum_ret = +53 % over 1,608 long-or-flat bars:

```
mean per-bar return  ≈ ((1 + 0.530)^(1/1608) - 1) ≈ 0.000264 (i.e. +0.0264 %)
stddev per-bar return ≈ ~0.0034 (rough; from predictions.parquet's strategy_return column)
per-bar Sharpe ≈ 0.000264 / 0.0034 ≈ 0.078
annualized: × √8760 = × 93.6 → 7.28 ✓
```

(That checks out — the lab's `+7.28` is consistent with a per-bar Sharpe near 0.078.)

The sampling SE on a per-bar Sharpe estimate, for N i.i.d. bars, is approximately ([Lo 2002](https://www.tandfonline.com/doi/abs/10.1080/0015198X.2002.10481158)):

```
SE(s) ≈ sqrt((1 + s^2 / 2) / N)
```

For `s ≈ 0.078, N = 1608`:

```
SE ≈ sqrt((1 + 0.003) / 1608) ≈ sqrt(0.000624) ≈ 0.025
```

So **per-bar Sharpe = 0.078 ± 0.025** on this test slice. The z-score against the null of zero per-bar Sharpe is `0.078 / 0.025 ≈ 3.1`. That's a "real but marginal" finding by typical statistical standards — passes a one-tailed 99 % test but isn't a slam-dunk.

The annualization preserves the ratio (`Sharpe_annual / SE_annual` = `Sharpe_per_bar / SE_per_bar`), so the *z-score is the same in both units*. What changes is the magnitude. `+7.28` looks like a hedge fund. `0.078 ± 0.025` looks like a careful empirical claim. Same data, same evidence, different presentation.

---

## What "Sharpe of 5" on hourly bars actually means

Apply the lens to the rest of the leaderboard. Per-bar Sharpe = `Sharpe_annual / 93.6`. Standard error ~0.025 for 1,600 bars (the per-bar SE is roughly the same across the leaderboard because it depends on N and on `s²` — for small `s`, it's approximately `1/√N ≈ 0.025`).

| Model | Sharpe (annual) | per-bar | per-bar SE | z |
|---|---:|---:|---:|---:|
| **ARIMA(1,1,1)** | **+7.28** | **0.078** | 0.025 | **~3.1** |
| LSTM | +4.95 | 0.053 | 0.025 | ~2.1 |
| TFT | +4.59 | 0.049 | 0.025 | ~2.0 |
| Chronos-2 | +4.29 | 0.046 | 0.025 | ~1.8 |
| TimesFM 2.5 | +2.44 | 0.026 | 0.025 | ~1.0 |
| XGBoost | +1.45 | 0.015 | 0.025 | ~0.6 |

What the z-column actually says:

- **ARIMA at z ≈ 3.1** — a real, ~99.9 %-confidence finding that per-bar Sharpe is positive, on this window.
- **LSTM, TFT, Chronos-2 at z ≈ 1.8–2.1** — at the edge of one-tailed 95 %. *Real but marginal*. A second test window would either confirm or wash these out (Article 6's regime test would do exactly that).
- **TimesFM 2.5 at z ≈ 1.0** — about one standard error above zero. Could easily be noise.
- **XGBoost at z ≈ 0.6** — indistinguishable from zero. The +1.45 annualized number is *literally consistent with a no-skill model getting unlucky in one direction or another*.

The picture this lens paints is *much* less dramatic than the annualized column. Most of the leaderboard is at the edge of statistical significance on a single window. **Only ARIMA is unambiguously a real signal.** Article 10's drift-floor lens deflates it further (skill-attributable Sharpe ≈ +3, vs +4 drift floor); Article 6's regime lens would deflate it further still (or possibly confirm it). After all of those, the lab's most defensible claim is "ARIMA(1,1,1) extracts a small but real per-bar directional signal on hourly BTC, ~3σ above zero on the Sep–Nov 2024 window".

That's a different, smaller, more honest claim than "Sharpe = 7.28".

---

## The 1,608-bar problem

1,608 hourly bars is roughly 67 days — a reasonably sized backtest by hourly-crypto standards. It's still small for SE-on-Sharpe purposes. The SE shrinks like `1/√N`, so:

- 1,608 bars → SE ≈ 0.025
- 6,432 bars (a full year) → SE ≈ 0.0125
- 26,280 bars (3 years) → SE ≈ 0.006

To halve the SE you need 4× the bars. To bring the LSTM (+4.95) into "unambiguously above zero" territory at z ≈ 3, you'd need its per-bar Sharpe of 0.053 to clear three SEs — which means SE of 0.018, which means ~3,000 bars (~125 days). That's a longer test window.

This is the right unit for thinking about "do I have enough data to make this claim?". The lab's 67-day test slice is enough for ARIMA's z ≈ 3 finding to be real. It's *not* enough to distinguish the LSTM's ~+5 Sharpe from the moving-average baseline's ~+4.33 at any reasonable significance. The ranking on the leaderboard between rows 2–5 is mostly within sampling noise on this window.

The fix isn't bigger Sharpe numbers; it's longer or more numerous test windows. Article 13's roadmap commits to multi-window evaluation precisely so this kind of comparison stops being noise-dominated.

---

## Why the convention exists, in defense

A short, honest defense of annualized Sharpe:

- It's **comparable across timescales**. A daily strategy with per-bar Sharpe 0.5 and an hourly strategy with per-bar Sharpe 0.05 have the same annualized Sharpe (≈ 9.5 and ≈ 4.7 respectively, with the daily one looking better — which it should, on a per-unit-time basis). Per-bar Sharpe is not comparable across timescales without context.
- It's a **convention** the entire industry uses. Reporting only per-bar Sharpe makes it harder for a reader to compare your strategy to others.
- For very small `s` and large N, the formula's i.i.d. assumption is approximately fine. The dishonesty is using the convention *without the SE context*, not the convention itself.

The lab will keep reporting annualized Sharpe — it'll just always pair it with per-bar Sharpe and the SE. Article 13's roadmap commits to adding both columns to every leaderboard.

---

## The reporting standard, going forward

Three small changes to the lab's `metrics.json` schema, all committed in Article 13's roadmap:

1. **`per_bar_sharpe`** — `mean / stddev` on the strategy returns, no annualization.
2. **`per_bar_sharpe_se`** — `sqrt((1 + per_bar_sharpe^2 / 2) / rows_test)`, the rough Lo-style SE.
3. **`per_bar_sharpe_z`** — `per_bar_sharpe / per_bar_sharpe_se`, the z-score against zero.

The leaderboard then becomes:

| Model | dir_acc | Sharpe (annual) | per-bar Sharpe | SE | z |
|---|---:|---:|---:|---:|---:|

Suddenly "ARIMA at z ≈ 3.1" reads differently from "LSTM at z ≈ 2.1", and both read very differently from "+7.28 vs +4.95". Same evidence, more honest summary.

---

## The deeper lesson

A backtest's annualized Sharpe is a **marketing summary**. It's the right number to *quote* but the wrong number to *trust* by itself. Trust is built by:

1. **Per-bar Sharpe + SE + z** (this article).
2. **Multi-window distribution** (Article 6).
3. **Drift-floor adjustment** (Article 10).
4. **Regime sensitivity** (Article 6 again).
5. **Calibration of any uncertainty bands** (Article 12).

The lab's headline ARIMA result survives lens (1) at z ≈ 3.1, survives lens (3) at skill-attributable Sharpe ≈ +3, and would *probably* survive lens (4) on a Q1 rerun (low-capacity AR(1) is regime-resistant). It's a small claim, three numbers and a coefficient, and it's *actually real*. The deep-model wins in this series are mostly at the edge of significance and mostly drift; Article 13's follow-ups will tell us which of them survive the same audit.

That's not a critique of the deep models. It's the right way to read backtest numbers in a low-SNR domain. And it's the lens every BTC ML post on the internet should adopt. Most don't.

---

## Closing — the series concludes

This was the optional fourteenth piece. The series closes here: thirteen substantive articles, one methodological footnote, one settled but small headline.

ARIMA(1,1,1) on hourly BTCUSDT, Sep–Nov 2024:
- annualized Sharpe **+7.28**
- per-bar Sharpe **0.078 ± 0.025**, z ≈ **3.1**
- skill-attributable component, after drift-floor (Article 10): roughly **+3 annualized**, z ≈ 1.5 — at the edge of "real on top of the rally".

Smaller than +7.28 made it sound. Still real. The single defensible claim of this lab.

The deep models — LSTM, TFT, foundation models — were mostly noise on this window in this lens. Not bad models; just not models that produced a measurable signal above the rally on the inputs we tried at the resolution we tried.

The lab continues. The series ends.

---

*Code: [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py) · Repo: [transformer_bitcoin_ai](../../../README.md)*

*Reading order: [Article 1](../01-the-floor-why-a-one-line-btc-predictor-is-so-hard-to-beat/01-the-floor-why-a-one-line-btc-predictor-is-so-hard-to-beat.md) → [Article 13](../13-where-this-repo-goes-next/13-where-this-repo-goes-next.md), then back to this one when somebody quotes a Sharpe number on hourly bars.*
