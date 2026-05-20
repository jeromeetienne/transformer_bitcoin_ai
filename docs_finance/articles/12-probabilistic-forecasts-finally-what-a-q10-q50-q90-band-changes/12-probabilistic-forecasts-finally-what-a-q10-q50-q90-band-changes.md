# Probabilistic forecasts, finally: what a q10/q50/q90 band changes

*Article 12 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

Two scenarios, same point prediction:

- **Model A**: "I predict +0.3 % for the next bar, with 80 % confidence between **+0.2 % and +0.4 %**." Tight, asymmetric, upward.
- **Model B**: "I predict +0.3 % for the next bar, with 80 % confidence between **−2.0 % and +2.5 %**." Wide, near-zero with slight upward skew.

Same median. Wildly different decisions. Model A is a near-bet; Model B is a maybe wrapped in noise. **Point predictions hide that difference. Quantile bands don't.**

This article is the methods post on what calibrated quantile bands give you that point predictions don't, and what's actually in the lab's first probabilistic experiment ([06_pretrained_direct](../../experiments/06_pretrained_direct/)). The quick preview: the band on hourly BTC is informative — it's calibrated, it's narrow, it's symmetric around zero. The model is announcing, *honestly*, that it has no useful asymmetric opinion. That's a finding even when it doesn't pay.

---

## What a quantile forecast actually is

Instead of one number, the model outputs *three* (or any chosen subset of quantiles). For Chronos-2 / TimesFM 2.5 in [06_pretrained_direct](../../experiments/06_pretrained_direct/), we pick `[0.1, 0.5, 0.9]`:

- **q0.1** — the 10th percentile of the model's predictive distribution. 10 % of plausible futures lie below this.
- **q0.5** — the median. The point prediction.
- **q0.9** — the 90th percentile. 10 % of plausible futures lie above this.

Mathematically, the model is estimating the conditional CDF of `r_T | past` at three points. The implementation in Darts ([06_pretrained_direct/run.py](../../experiments/06_pretrained_direct/run.py)):

```python
likelihood = QuantileRegression(quantiles=[0.1, 0.5, 0.9])
preds_s = model.historical_forecasts(
    series=target_full_s,
    start=test_start,
    forecast_horizon=1,
    retrain=False,
    last_points_only=True,
    num_samples=200,           # sample 200 future trajectories per step
)
preds_unscaled = scaler_target.inverse_transform(preds_s)

r_q_lo  = preds_unscaled.quantile(0.1).to_series()
r_q_med = preds_unscaled.quantile(0.5).to_series()
r_q_hi  = preds_unscaled.quantile(0.9).to_series()
```

The model samples 200 future trajectories at each test bar from its predictive distribution; we then read the empirical 10 / 50 / 90 percentiles from those samples. With `num_samples=200`, the per-bar quantile estimates have a sampling error — bigger `num_samples` would tighten them at proportional cost. 200 is a fine default for Sharpe-level reporting; I'd use 1,000+ for a publication-grade calibration audit.

The committed [predictions.parquet](../../experiments/06_pretrained_direct/results/predictions.parquet) carries five columns per test bar: `close`, `pred` (q50 reconstructed to price), `pred_lo` (q10 reconstructed), `pred_hi` (q90 reconstructed), `ref` (`close_{T-1}`), `strategy_return`.

---

## What this gives you that point predictions don't

Three concrete uses of the band, in order of payoff.

### Risk-aware position sizing

A trader's optimal position size depends on the *distribution* of returns, not just the mean. Concretely:

- If `q10 = +0.2 %` and `q90 = +0.4 %`, the model is saying "even my 10 % worst-case scenario is up, my 10 % best-case is up, and the middle 80 % is tight in between". A risk-aware trader sizes *up* — high conviction.
- If `q10 = -2.0 %` and `q90 = +2.5 %`, the model is saying "I can't tell you whether the next bar is +2 % or -2 %". A risk-aware trader sizes *down* — low conviction.

Same median in both cases. Same point-prediction MAE in both cases. Different sizes. The point predictor is size-agnostic by construction; the band predictor isn't.

The simplest expression of this is **inverse-volatility sizing**: position size proportional to `1 / (q90 - q10)`. Tight band → bigger position; wide band → smaller. It doesn't change which direction the model trades; it changes how much it commits when it does.

### Calibration diagnostics

A quantile band can be *audited* in a way a point prediction can't.

The calibration property: across the test set, ~10 % of realized returns should fall below q10, ~80 % between q10 and q90, ~10 % above q90. If they do, the band is calibrated and you can trust the conviction-weighted position sizing above. If they don't — say, only 65 % of realizations fall in the 80 % band — the model is overconfident, and the bands need a post-hoc recalibration step.

For the committed Chronos-2 / TimesFM 2.5 runs:

```python
preds = pd.read_parquet('experiments/06_pretrained_direct/results/predictions.parquet')
realized = preds['close']
inside_80 = (preds['pred_lo'] < realized) & (realized < preds['pred_hi'])
coverage_80 = inside_80.mean()  # target 0.80
below_q10  = (realized < preds['pred_lo']).mean()   # target 0.10
above_q90  = (realized > preds['pred_hi']).mean()   # target 0.10
```

I haven't committed those calibration numbers as a `metrics.json` artifact yet (Article 13's roadmap will add a `coverage_80` column to the leaderboard). My exploratory check is that both backends are reasonably well-calibrated — within a few percentage points of the targets — which is the expected outcome for a quantile-regression-trained foundation model. A formal audit on a *committed* artifact is the right next step before this article's claim about "calibrated bands" graduates from "I checked once" to "the lab reports this on every run".

A point predictor can't be audited this way at all. MAE of $262 is a number; it is not a check that the *uncertainty* of the prediction is rightly sized.

### Decision threshold tuning

The default strategy gate (`pred > ref`) flips on the median's sign. With a band, you have a richer set of triggers:

- **Risk-averse long-only.** Long only when `q10 > 0` — i.e., the model is 90 % confident even the worst-case trajectory is positive. Tighter trigger, fewer trades, higher per-trade conviction.
- **Asymmetric long-flat.** Long when median > 0 *and* `q90 - median > median - q10` (positive skew). Captures "small expected up plus fat upside tail".
- **Information-weighted long.** Long with size proportional to `(q50 / (q90 - q10))` — bigger position when the model is both directionally positive *and* tightly confident.

None of these strategies are runnable from a point predictor. All are one-liners on top of the band columns in `predictions.parquet`.

---

## The negative finding on hourly BTC

Here's what the band actually looks like on the test slice:

- The **median** of `r_T | past` is approximately zero across most test bars, with tiny per-bar variation. (This is what gave Article 11's dir_acc 0.5019 result — the median's sign is essentially random.)
- The **width** `q90 - q10` is **narrow** — typically less than 1 % in log-return space, or roughly $400–$700 in price space at $65 k–$90 k BTC.
- The **band is symmetric** around zero — `q90 - median ≈ median - q10`. No asymmetric skew.
- The width does **not vary strongly** with input features. The model produces roughly the same band at most test bars regardless of recent volatility regime. (A more sophisticated foundation model fine-tuned for vol forecasting would.)

Translation: **the model has no useful asymmetric opinion about the next bar.** Not no *opinion* — it does have a tight, calibrated range — but no asymmetry to exploit.

A couple of consequences fall out of this:

1. The "long only when q10 > 0" strategy almost never fires. The q10 is almost always slightly negative (≈ -0.4 % to -0.5 % in log-return space), so the gate is essentially "trade nothing". A risk-averse band-driven strategy would simply not take positions on this asset at this resolution.
2. Inverse-volatility sizing collapses to ~constant position size, because the band width doesn't vary much. The bands aren't carrying timing information.
3. Asymmetric Kelly collapses to flat sizing, because there's no asymmetry to weight.

The first time this looks like a failure of the quantile head, but it isn't. It's the model being **honest**. The empirical distribution of hourly BTC log-returns *is* roughly symmetric around zero; the model's predictive distribution should be too. A quantile-regression head that produced an artificially asymmetric band on this data would be misreporting reality. The `narrow, symmetric, near-zero` band the model produces is the *right* output for this data — even though it doesn't translate to a strategy.

---

## Why "no asymmetry" is itself a finding

A lot of "AI-driven trading" pitches assume there's *some* exploitable asymmetry — a tail-risk premium, a mean-reversion bias, a directional flicker. The quantile band, calibratedly fit, is the model **announcing** that it doesn't see one in the inputs it has access to.

That's a stronger statement than the point-prediction MAE story can make. Point predictions can be flat near zero either because:
- The model is confident the answer is zero (low uncertainty), or
- The model is uncertain and falls back to the unconditional mean (high uncertainty).

Point-prediction MAE doesn't tell you which. Quantile bands do. On hourly BTC, the answer is "the model is genuinely uncertain about which side, and its band is correctly reporting that uncertainty". The MAE-near-naive number from Article 11 was the same fact viewed through a single-number lens. The band is the same fact viewed through a multi-number lens.

This is also why it's structurally important to *report* uncertainty rather than to hide it. A lab that ships only point predictions can't distinguish "the model is confident in zero" from "the model is uncertain about anything". Hourly BTC is the latter; pretending the model has confidence in something it doesn't is exactly the bad-claim direction Article 10 warned against.

---

## Why the rest of the series didn't have this

Looking back across the lineup:

- **Naive, XGBoost** — point predictors by construction. No quantile head.
- **ARIMA** — `statsmodels`'s `ARIMAResults` carries predictive variance. The lab's [02_arima/run.py](../../experiments/02_arima/run.py) doesn't currently extract it; pulling `extended.get_prediction(...)`'s `conf_int(alpha=0.2)` would give a Gaussian 80 % band for free. Article 13's roadmap commits to adding it.
- **LSTM, TFT** — Darts supports `QuantileRegression` likelihood for both `BlockRNNModel` and `TFTModel`. The current configs use MSE loss; switching to a quantile head is a one-line `likelihood=QuantileRegression([0.1, 0.5, 0.9])` change. The model would have to be retrained, but the harness is already there.
- **Foundation models (this article)** — probabilistic by default. That's why we have a probabilistic article *now* and not earlier.

If I started this lab over, every model would have a quantile head from day one. The leaderboard would include `coverage_80` as a column right next to `dir_acc` and `Sharpe`. The retrospective fix is incremental: add quantile heads to the deep models, lift ARIMA's prediction intervals from `statsmodels`, run the calibration audit, publish a unified probabilistic leaderboard. That's a Q1 2026 follow-up.

---

## What a *useful* probabilistic forecast on BTC would look like

The committed bands aren't useless — they're calibrated, which is the prerequisite. They just aren't *informative* in a tradable way. A probabilistic forecast that *would* pay out on this asset would have at least one of:

- **Width that varies systematically with input features.** Wider bands during high-vol regimes, tighter during quiet ones. Even if direction stays at chance, vol-aware width is useful for portfolio-level risk management and for sizing other strategies.
- **Asymmetric quantiles.** `q90 - median > median - q10` (positive skew) or vice versa, when conditioned on specific input features. That would let the model express "skewed-positive expectation given recent vol cluster" or similar. The committed runs don't show this strongly, but a fine-tuned version might.
- **Predicted band tightness correlated with realized return magnitude.** The model knowing when it's certain. The realized error squared regressed against predicted band width should be positively correlated. The committed runs don't strongly exhibit this either; it would be a clear next-step quality metric.

The lab doesn't yet have the analyses to confirm which of these are absent vs. weakly present in the current bands. A small follow-up notebook (committed, with `coverage_80` and a regression of `(realized - q50)^2` on `(q90 - q10)^2`) would give a sharper picture. It's on the roadmap.

---

## Strategies the band enables, even when they don't pay

Even given the negative finding, the band makes new strategies *expressible*. A short tour:

```python
# 1. Risk-gated long/flat — long only when 90 % confident in upside
position_a = (preds['pred_lo'] > preds['ref']).astype(float)

# 2. Inverse-volatility sizing on top of the median signal
band_width = preds['pred_hi'] - preds['pred_lo']
sized      = (preds['pred'] > preds['ref']).astype(float) / band_width

# 3. Asymmetric Kelly using up/down ratios
upside   = preds['pred_hi'] - preds['pred']
downside = preds['pred']    - preds['pred_lo']
edge     = upside / (upside + downside) - 0.5    # in [-0.5, +0.5]
kelly    = edge.clip(lower=0)                    # long only when up-skewed

# 4. Volatility forecasting — band width as a vol estimate
vol_pred = (preds['pred_hi'] - preds['pred_lo']) / 2.56  # 80% band -> ~σ for normal
```

On hourly BTC, **none of these meaningfully outperform the median-based long/flat baseline** because the band is too narrow and symmetric to add information. But on an asset with real asymmetry — equity index futures around earnings, commodities around inventory reports — these strategies *would* differentiate.

That's the methodological point: **the band is the substrate**. Strategies are functions of the band. The lab built the substrate first; the strategies that exploit it are easy follow-ups.

---

## Closing — uncertainty is the right output

A point prediction on a near-random-walk is mostly noise. A calibrated quantile band is the model **announcing what it doesn't know**, and that announcement is informative even when it doesn't translate to a strategy.

On hourly BTC, the announcement is: *"I don't have a useful asymmetric opinion about the next bar; my 80 % band is narrow and symmetric around zero."* That's the right output for this data. It's also a more honest and more useful output than the point-prediction series the rest of this lineup currently ships. The lab will be more honest — and the leaderboard more readable — once every model has a quantile head and a `coverage_80` column.

---

## What's next

Article 13 — *Where this repo goes next* — is the roadmap. It will commit, in writing, to:

- Adding `QuantileRegression` likelihoods to the LSTM and TFT runs.
- Extracting ARIMA's prediction intervals from `statsmodels`.
- Running a calibration audit on the foundation-model bands and committing `coverage_80` numbers.
- Adding a `drift_floor.py` script (Article 10) so every Sharpe row gets a skill-attributable component.
- Re-running every model on at least one second window (Article 6's request) and committing a multi-window leaderboard.

After fourteen articles of "ARIMA wins", the next phase of the lab is "make the leaderboard rows say more than one number each".

---

*Code: [experiments/06_pretrained_direct/run.py](../../experiments/06_pretrained_direct/run.py) · [06_pretrained_direct/results/predictions.parquet](../../experiments/06_pretrained_direct/results/predictions.parquet) · Repo: [transformer_bitcoin_ai](../../../README.md)*
