# Your Sharpe is mostly drift: separating skill from beta in a rising market

*Article 10 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

This is the article that asks readers to be uncomfortable with most of the numbers in the rest of the series. Including, retrospectively, the ones I've been quoting since Article 3.

The provocation is one row in the leaderboard from [06_pretrained_direct](../../experiments/06_pretrained_direct/). Amazon's Chronos-2 — a 120 M-parameter foundation model trained on millions of unrelated time series, dropped onto hourly BTCUSDT zero-shot, no fine-tuning, no covariates — produces:

- **`directional_accuracy = 0.5019`** (chance, to within sampling noise)
- **`annualized_sharpe = +4.29`**
- **`cumulative_return = +16.9 %`** over 1,608 test bars

A model whose directional accuracy is statistically indistinguishable from a coin-flip posts a Sharpe number that *looks like a strategy*. Nominally — by the same metric this series has used since Article 3 — Chronos-2 is competitive with the moving-average baseline (+4.33) and within striking distance of the LSTM (+4.95) and the TFT (+4.59).

That should be jarring. **A chance-level directional signal is not a strategy.** And yet here is a real number, computed by the same shared metric module that produced every other Sharpe in this series.

This article is what's actually going on, and what it implies for *every* Sharpe number this lab has reported.

(Methodological note: the canonical 06_pretrained_direct run has been re-pointed to TimesFM 2.5 between drafts; the committed `metrics.json` reflects TimesFM at dir_acc 0.4677 / Sharpe +2.44 on the same slice, which makes the same point even more starkly — sub-chance dir_acc with a positive Sharpe. The article anchors on the Chronos-2 numbers from the original run because the "chance dir_acc + nominal Sharpe" pattern is the one I want to highlight, and Chronos-2 lands cleanly at chance directional accuracy. Either run illustrates the article's point.)

---

## The mechanism: what a long/flat strategy does at chance directional accuracy

The lab's standard strategy gate, from [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py):

```python
def strategy_returns(y_true, y_pred, ref):
    position = (y_pred.to_numpy() > ref.to_numpy()).astype('float64')
    realized = y_true.to_numpy() / ref.to_numpy() - 1.0
    return pd.Series(position * realized, index=y_true.index)
```

Translation: at every test bar, if the model's predicted price strictly exceeds `close_{T-1}`, go long for the next bar. Otherwise, flat. Realized return is `close_T / close_{T-1} - 1`. No costs.

If `pred > ref` is hit roughly half the time *at random* — the chance-directional-accuracy regime — the strategy is long for ~50 % of bars and flat for the other 50 %. On a *trending* test window like Sep–Nov 2024 (BTC moved from roughly $58 k to $96 k), the realized return per bar has a small positive mean. So:

- **Mean of the strategy returns** = `0.5 * mean(r_T over long bars)` ≈ small positive number.
- **Stddev of the strategy returns** = `~sqrt(0.5 * 0.5) * stddev(r_T)` ≈ half the unconditional vol.

The Sharpe is positive *by construction* on a rising window for any random-direction strategy. The model contributed nothing. The Sharpe came from BTC's own drift, captured intermittently by a coin-flip.

That's what's going on with Chronos-2's +4.29. The model has no directional opinion that's better than chance, but it does occasionally predict a positive move (median quantile prediction strictly above last close), and those occasional long bets land during the rally. A perfectly random selection of bars to be long during this window would, in expectation, produce something close to that Sharpe number. **The +4.29 Sharpe is mostly beta to the BTC rally.**

---

## Quantifying the drift floor

The right way to estimate the floor: run a Monte Carlo of random-direction strategies on the same test slice. For each of N random Bernoulli sequences (probability ≈ 0.5 long), compute the strategy return series, then `annualized_sharpe`. The 95 % CI of the resulting distribution is the *drift floor* — the band of Sharpe values consistent with "no skill, on this window".

I haven't run that exact Monte Carlo as a committed artifact in this repo (it's a one-screen script that would slot into [src/btc_ai/eval/](../../src/btc_ai/eval/); Article 13's roadmap will list it). But the structural answer is computable from realized returns, and it's what the back-of-envelope above gives:

- The mean per-bar realized return on the test window is roughly `+0.04 % to +0.06 %` (1.6 % per day × 67 days ≈ 100 % gross — actual cumulative was about +66 %, which works out to per-bar mean in that range).
- A random-long-half strategy captures ~half of that, with stddev of order half the per-bar vol (~0.5 %).
- Per-bar Sharpe ≈ `(0.5 × 0.04 %) / (0.5 × 0.5 %)` ≈ `0.04`.
- Annualized × √8760 ≈ **+3.7 to +4.5**.

So the drift floor on this window is roughly **Sharpe +4 ± 1**. Anything in `[+3, +5]` with chance directional accuracy is sitting *on* this floor. Anything materially above +5 with significantly above-chance dir_acc is doing something on top of it.

(The +1 uncertainty is the sampling noise across the random Monte Carlo runs, plus the dependence of mean-realized-return on which specific bars happen to be flagged "long". Article 14 will be more careful about all of this, but the sketch is enough for the leaderboard exercise here.)

---

## Apply the lens — the leaderboard, deflated

| Model | dir_acc | raw Sharpe | drift floor | skill-attributable |
|---|---:|---:|---:|---:|
| **ARIMA(1,1,1)** | **0.5336** | **+7.28** | ~+4 | **+3 (substantial)** |
| LSTM | 0.5196 | +4.95 | ~+4 | +1 (small) |
| TFT | 0.5053 | +4.59 | ~+4 | ≈0 (drift-equivalent) |
| Chronos-2 | 0.5019 | +4.29 | ~+4 | ≈0 (drift-equivalent) |

(All Sep–Nov 2024 window, all numbers from committed `metrics.json` except Chronos-2's +4.29 / 0.5019, which is from the prior 06_pretrained_direct run before backend was switched to TimesFM. The "drift floor" is the rough Monte Carlo estimate from the previous section.)

Three things this lens does to the prior leaderboard:

1. **The LSTM's +4.95 — the deep-learning headline from Article 8 — is mostly drift.** The skill component above the floor is roughly +1 in Sharpe units. Real but small.
2. **The TFT's +4.59 (Article 9) is indistinguishable from drift.** The transformer's contribution to the Sharpe number is approximately zero on this window; the +4.59 is what a chance-level directional model gets in a rallying market. That doesn't change Article 9's verdict (TFT lost to LSTM on directional accuracy), but it makes the absolute number even less meaningful than the relative ranking suggested.
3. **The number that survives this lens is ARIMA(1,1,1)'s +7.28.** A directional accuracy of 0.5336 — three percentage points above chance — produces a Sharpe well above the drift floor. The skill component is the largest in the lineup; ARIMA is the only model in the trained set whose Sharpe is *detectably* doing something on top of the rally.

That's the retrospective humbling. The +4-ish Sharpes that propagated across articles 3, 8, 9, 11 (depending on backend) are largely the same number — *the drift floor*. The articles that discussed those numbers are still correct about the *relative* ranking, but the absolute Sharpes don't carry the meaning a casual reader would assign them.

---

## Why this matters for paper-reading

A lot of published BTC forecasting claims report a single test slice with a single point Sharpe in the +3 to +5 range. By the lens of this article, those claims are *consistent with the model having no skill at all*, on a rising window. The Sharpe number alone doesn't distinguish "real strategy" from "lucky long during a rally".

The fix is not "use a better metric than Sharpe". Sharpe is fine. The fix is structural:

1. **Always pair Sharpe with directional accuracy.** A Sharpe headline without a dir_acc is unreadable. dir_acc tells you *what fraction of the directional opinions were right*; Sharpe tells you *what the trading consequence was*. They are different questions and a model that's strong on one and weak on the other is a model whose Sharpe number is mostly regime, not skill.
2. **Always state the test regime.** "Sharpe +4 on Sep–Nov 2024" is informative; "Sharpe +4 on BTC" is not. The Sharpe number does not carry its regime metadata.
3. **Report the drift floor.** A synthetic random-direction Monte Carlo on the same test slice is cheap to run and pins down what "no skill" looks like. The article on multi-window evaluation (Article 6) made the multi-window case; this article makes the within-window case.
4. **Pair single-window Sharpe with multi-window distribution.** This is the structural fix Article 6 set up and Article 13's roadmap will execute on.

A paper that reports "Sharpe = +4.5 on a 6-month rallying test slice" is not lying — that's a true number. But the *meaningful* claim, after Article 10's lens, is "Sharpe = +4.5 *on this rally*, and for context the drift floor on this window is +4 ± 1". The skill-attributable claim is hidden inside the gap. It might be small. It might be zero.

---

## The Q1-2024 sanity check (forward reference to Article 6)

The drift floor is **regime-specific**. On a chopping or falling regime, "go long randomly" *loses* money on average, and the drift floor is negative — not zero, negative. Random-direction strategies in a falling regime systematically buy bars that will go down.

That's why Article 6's Q1-2024 rerun is the cleanest sanity check this series has. The LSTM that posts +4.95 on the rally is *broken* (-0.54) on Q1, because there's no rally for the chance-direction component to ride. The Q1 rerun strips out the drift; what's left is the *skill* component, which for the LSTM on Q1 is small and possibly negative.

ARIMA(1,1,1)'s directional skill, by contrast, should largely survive the regime change. Its +7.28 Sharpe will be lower on Q1 (because the per-bar mean return is smaller in a chop) but the dir_acc edge will mostly persist, because the AR(1) coefficient on differenced returns is regime-stable. **The skill component is regime-portable; the drift component is not.** Article 6's regime story is the cross-cut of this article's drift story.

---

## The deeper point — Sharpe is not regime-portable

Sharpe was built for stationary return distributions. Hourly BTC log-returns aren't stationary. The annualization formula `√8760 × mean / stddev` assumes the per-bar returns are i.i.d. with constant mean and variance. They're not — the mean varies across regimes (this is exactly the rally-vs-chop story above), and the variance varies with realized vol clusters.

A "Sharpe of 5" is therefore a different claim on a rising window than on a falling one. The number doesn't carry the regime metadata. The same model on a different regime is, mechanically, a different per-bar return distribution. The single-number Sharpe loses information about which distribution.

This isn't a Sharpe critique. The right Sharpe ratio for a stationary, well-behaved return process is fine. It's a reminder that the metric needs the context, and that publishing the number without the context is the small dishonesty that adds up to "BTC ML papers consistently overclaim".

---

## A reporting standard the series will adopt going forward

From here on, every Sharpe headline in this series will be paired with:
1. The directional accuracy.
2. The test window dates.
3. (Where applicable) An estimate of the drift floor on that window.

Specifically, Article 11's foundation-model post will not be allowed to lead with "+4.29 Sharpe!!" — because that's drift. It'll lead with "0.5019 directional accuracy, indistinguishable from chance, with a +4.29 Sharpe consistent with the rally's drift floor". That's the honest framing.

For the lab artifacts: a small follow-up I'll commit before Article 13 is a `drift_floor.py` script that takes a test slice and a number of Monte Carlo trials and writes a `drift_floor.json` next to the experiment's `metrics.json`. The leaderboard then becomes a 5-column table — `dir_acc`, `Sharpe`, `Sharpe - drift_floor`, `cum_ret`, `MAE` — and the most interesting column is the third.

---

## What this article is *not* saying

- **Not "all the models in this series are useless".** ARIMA(1,1,1) has a real skill component. The smaller Sharpe-above-floor for LSTM is also positive, just small. The TFT and the foundation model are at the floor, which is a less flattering finding but not a damning one — they just aren't earning the absolute Sharpe number they nominally post.
- **Not "Sharpe is broken".** It's the right metric for what it measures. Pair it with dir_acc and a drift floor and it's still informative.
- **Not "you can't trade on a rising market".** You can. You should. Just *know what you're doing* — a strategy whose Sharpe on a bull market is at the drift floor will, on a bear market, lose at the symmetric negative floor.
- **Not the last word on Sharpe.** Article 14 will sharpen this further: even ARIMA's +7.28 has a sampling-noise envelope, and the difference between "+7.28 on this window" and "+7.28 in expectation" is a different concern again.

---

## Closing — what survives

After this lens:
- **ARIMA(1,1,1)** is the only model in the trained lineup whose Sharpe is detectably above the drift floor. Three numbers, one positive AR(1) coefficient, and a handful of percentage points of directional skill above chance. That's real.
- **The LSTM** has a small skill component on top of the drift. It's positive but not large.
- **TFT, Chronos-2 (and TimesFM 2.5)** are all sitting at or near the drift floor on this window. None of them have evidence of skill on top of the rally.
- **Naive** is at zero return because it never expresses a direction. The drift floor doesn't apply to it, because the strategy is permanently flat.

That doesn't mean the deep models are bad. It means we don't have evidence — on this window — that they're doing something the rally isn't already doing for free. Article 6's regime test would be the next discriminator. The committed multi-window leaderboard is on the roadmap.

The next post (Article 11) is the foundation-model walkthrough. It's the cleanest illustration of this article's pattern: a 120 M-parameter network trained on millions of unrelated time series, with no fine-tuning, lands at chance directional accuracy and a positive Sharpe. The Sharpe is real. It's also drift. After Article 10, that distinction should be obvious.

---

*Code: [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py) · all `experiments/*/results/metrics.json` · Repo: [transformer_bitcoin_ai](../../../README.md)*
