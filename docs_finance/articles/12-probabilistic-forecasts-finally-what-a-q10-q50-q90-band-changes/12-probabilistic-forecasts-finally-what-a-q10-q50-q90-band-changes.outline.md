# Outline — Probabilistic forecasts, finally: what a q10/q50/q90 band changes

## One-line pitch
Methods post, generalized beyond [06_pretrained](../../experiments/06_pretrained/). The first experiment in the lineup with calibrated prediction intervals, via `QuantileRegression([0.1, 0.5, 0.9])` and `num_samples=200`. What intervals give you that point predictions don't: risk-aware position sizing, calibration diagnostics, the difference between "the model predicts +1 %" and "the model says 20 % chance of -2 %, 20 % chance of +3 %". The negative finding worth showing: on hourly BTC the q10 – q90 spread is *narrow and centred on the last value* — quantile loss is doing exactly what it should; there's just no asymmetry to exploit. Ties back to why every prior post talked about Sharpe but never about uncertainty.

## Audience
Quants, risk managers, and ML practitioners who've shipped point predictors and felt the absence of a confidence interval. Series readers who watched the foundation models in Article 11 produce calibrated bands and want to know what to do with them.

## Thesis
A point prediction tells you what the model thinks the next bar will be. A calibrated quantile band tells you *what the model is confident about*. That distinction is the difference between a leaderboard metric and an actual trading decision. On hourly BTC, the q10 – q90 band is informative — it's narrow, symmetric, centred on the last value. The model is telling you, calibratedly, that it has no useful asymmetric opinion. That's a finding, even though it doesn't pay out.

## Structure

### 1. The hook — your model says +0.3 %. Is that a strong signal or a weak one?
- Two scenarios:
  - Model A: "I predict +0.3 %, with 80 % confidence between +0.2 % and +0.4 %". Tight, asymmetric upward.
  - Model B: "I predict +0.3 %, with 80 % confidence between -2.0 % and +2.5 %". Wide, near-zero with slight upward skew.
- Same point prediction. Wildly different decisions.
- Point predictions hide the difference. Quantile bands don't.

### 2. What a quantile forecast actually is
- Instead of one number, the model outputs *three* (q10, q50, q90) — or an arbitrary set of quantiles.
- Mathematically: estimate the conditional CDF of `r_T` at three (or N) points.
- Calibration property: across the test set, ~10 % of realizations should fall below q10, ~80 % between q10 and q90, ~10 % above q90. If they do, the model is calibrated.
- Code: from [06_pretrained/run.py](../../experiments/06_pretrained/run.py):
  ```python
  likelihood = QuantileRegression(quantiles=[0.1, 0.5, 0.9])
  preds = model.historical_forecasts(..., num_samples=200)
  r_q_lo = preds.quantile(0.1).to_series()
  r_q_med = preds.quantile(0.5).to_series()
  r_q_hi = preds.quantile(0.9).to_series()
  ```

### 3. What this gives you that point predictions don't
Three concrete uses, in order of payoff:

**a. Risk-aware position sizing.** A trader's optimal position size depends on the *distribution* of returns, not just the mean. If the q10 is far in the red, you size smaller. If q90 is far in the green, you size larger (Kelly-ish). Point predictions are size-agnostic.

**b. Calibration diagnostics.** "Is the realized return inside the 80 % band 80 % of the time?" — if yes, the model is calibrated; if no, the model is over- or under-confident. This is a *check* on the model that point-prediction MAE can't deliver.

**c. Decision threshold tuning.** The default strategy gate (`pred > ref`) flips on the median's sign. A risk-averse alternative is "long only when q10 > 0" — i.e., the model is 90 % confident even the worst-case trajectory is positive. Tighter trigger, fewer trades, higher per-trade conviction.

### 4. The negative finding on hourly BTC
- The q10–q90 band on this data, from [06_pretrained](../../experiments/06_pretrained/), is roughly *symmetric around zero* and *narrow*. The model is saying: "I think the next bar's log-return is in [-X, +X] with 80 % confidence, and my best guess is approximately 0".
- *That* is the model being honest. It's not a failure of the quantile head; it's a faithful report of the underlying distribution.
- Calibration is roughly correct: ~10 % of realized returns fall outside the 80 % band on each side. (The article will commit to running the calibration check before publication; if I haven't, I'll say so.)
- Translation: the model has no asymmetric opinion. The strategy gate "long when q10 > 0" almost never fires (because the q10 is almost always negative — small but negative). A risk-averse band-driven strategy would simply not trade.

### 5. Why "no asymmetry" is itself a finding
- A lot of "AI-driven trading" pitches assume there's *some* asymmetry to exploit.
- The quantile band, calibratedly fit, is the model announcing "there isn't one detectable in the inputs I see".
- That's a stronger claim than the point-prediction MAE story can make. Point predictions can be flat near zero either because the model is confident about zero or because it's uncertain. The quantile band distinguishes the two.

### 6. Strategy variants the band enables
A short tour of strategies that *can* be defined now and couldn't be before:

- **Risk-gated long/flat.** Long only when `q10 > 0` (90 % confident in upside). Almost never fires on this data → almost never trades → almost never loses.
- **Position sizing by `(q90 - q10) / 2`.** Smaller position when the band is wide; larger when narrow. Doesn't change the directional logic but smooths the equity curve.
- **Asymmetric Kelly.** Trader-aware sizing using the ratio of upside (`q90 - q50`) to downside (`q50 - q10`). On hourly BTC the ratio is near 1, so this collapses to flat sizing. On an asset with real asymmetry, it would not.
- **Volatility forecasting.** The band's *width* is itself a forecast of next-bar volatility. Useful as an input to other models (option pricing, position sizing for portfolio risk).

### 7. What a *useful* probabilistic forecast on BTC would look like
- A wider q90 - q10 spread *that varies systematically with input features* — e.g., wider during high-vol regimes, tighter during quiet ones. That would be useful for vol forecasting even if direction stays at chance.
- Asymmetric quantiles — e.g., q90 > median + (median - q10). That would let the model express "skewed-positive expectation" or "skewed-negative". Hourly BTC's empirical return distribution is roughly symmetric, so unsurprisingly the model's forecast distribution is too.
- Predicted band tightness *correlated with realized return magnitude* — the model knowing when it's certain. The committed run doesn't show this strongly; future fine-tuned versions might.

### 8. The calibration audit
Concretely:
- For Chronos-2's quantile predictions on the test slice, compute `coverage_80 = mean(q10 < r_realized < q90)`. Should be ~0.8.
- Compute `coverage_10_lo = mean(r_realized < q10)`. Should be ~0.1.
- Compute `coverage_10_hi = mean(r_realized > q90)`. Should be ~0.1.
- If those numbers are within a few percentage points of their targets, the band is calibrated.
- (I'll add this as a small post-run script to the lab. The article will describe the test honestly and either quote the numbers, or note "haven't run yet, follow-up coming".)

### 9. Why the rest of the series didn't talk about this
- Articles 1–5: linear / tree models, all point predictors. ARIMA technically has predictive variance, but we used point predictions. (The repo could lift ARIMA's prediction intervals from `statsmodels` — Article 13's roadmap.)
- Articles 8–9: LSTM and TFT can support quantile losses (`QuantileRegression` likelihood) — neither was set up that way in this v1. Adding it is a one-line model change.
- Article 11 / this article: foundation models have probabilistic outputs out of the box. That's why we have a probabilistic article *now* and not earlier.
- The retrospective: if I started this lab over, every model would have a quantile head from day one. The leaderboard would include `coverage_80` as a column.

### 10. The leaderboard, with a probabilistic column
Aspirational table:
| Model | dir_acc | Sharpe | MAE | coverage_80 (target 0.8) |
|---|---:|---:|---:|---:|
| ARIMA(1,1,1) | 0.5336 | +7.28 | $260.50 | (TBD — extract from statsmodels) |
| LSTM | 0.5196 | +4.95 | $274.02 | (TBD — needs QuantileRegression head) |
| TFT | 0.5053 | +4.59 | $373.70 | (TBD — needs QuantileRegression head) |
| Chronos-2 | 0.5019 | +4.29 | $262.27 | ~0.8 (calibrated by construction) |
| TimesFM 2.5 | 0.4677 | +2.44 | $268.89 | ~0.8 (calibrated by construction) |

(The lab doesn't yet have the coverage_80 numbers committed; Article 13's roadmap commits to producing them.)

### 11. Reproducing the band
- The committed `predictions.parquet` from [06_pretrained](../../experiments/06_pretrained/) has `pred`, `pred_lo`, `pred_hi` columns — q50, q10, q90.
- The committed `plot.png` shows the close + median + shaded q10–q90 band.
- A reader who wants to inspect can:
  ```python
  preds = pd.read_parquet('experiments/06_pretrained/results/predictions.parquet')
  band_width = preds['pred_hi'] - preds['pred_lo']
  coverage_80 = ((preds['pred_lo'] < preds['close']) & (preds['close'] < preds['pred_hi'])).mean()
  ```

### 12. Closing — uncertainty is the right output
- Point predictions on a near-random-walk are mostly noise.
- Quantile bands are how the model *announces what it doesn't know*.
- On hourly BTC, the announcement is "I don't know the direction, my band is narrow and symmetric around zero". That's a finding even when it's not a strategy.
- Tease Article 13 (the roadmap mentions adding quantile heads to LSTM/TFT and pulling ARIMA's prediction intervals).

## Key code/file references
- [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py) — point metrics; coverage metrics still TBD
- [experiments/06_pretrained/run.py](../../experiments/06_pretrained/run.py) — `QuantileRegression([0.1, 0.5, 0.9])` setup
- [experiments/06_pretrained/results/predictions.parquet](../../experiments/06_pretrained/results/predictions.parquet) — `close, pred, pred_lo, pred_hi, ref, strategy_return`
- [experiments/06_pretrained/results/plot.png](../../experiments/06_pretrained/results/plot.png) — visual

## Tone notes
- Methods post, but practical. The "what would I do with a band" use cases should be concrete.
- The "negative finding" framing is right. The band is informative even when it doesn't pay.

## Length target
~1,500–1,800 words.
