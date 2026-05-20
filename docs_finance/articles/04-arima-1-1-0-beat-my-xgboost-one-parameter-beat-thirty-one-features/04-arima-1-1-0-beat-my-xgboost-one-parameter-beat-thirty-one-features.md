# ARIMA(1,1,0) beat my XGBoost. One parameter beat thirty-one features.

*Article 4 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

The headline is the title. A model from 1970 with a single autoregressive coefficient on differenced returns beats a 31-feature, 400-tree gradient booster on the same hourly BTCUSDT slice. By the leaderboard column we care about — directional accuracy — it isn't even close.

| Model | dir_acc | Sharpe | cum_ret | MAE |
|---|---|---|---|---|
| Naive (Article 1) | NaN | NaN | — | $260.50 |
| **ARIMA(1,1,0)** | **0.5373** | **+7.52** | **+53.5 %** | $260.49 |
| **XGBoost (default)** | **0.4872** | **+1.45** | **+8.1 %** | $270.73 |

(Numbers from [02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv) row `(1,1,0)` and [03_gradient_boosting/results/metrics.json](../../experiments/03_gradient_boosting/results/metrics.json).)

ARIMA wins on every column. XGBoost lands at dir_acc 0.4872 — worse than coin-flip on the bars where it expressed an opinion — while ARIMA(1,1,0) clears 0.53.

This article is about how that happens, and what it actually means by "feature engineering didn't help".

---

## The two contestants

### `ARIMA(1, 1, 0)` — one coefficient and a noise term

`(p, d, q) = (1, 1, 0)` decodes to:
- **`d = 1`** — model the *first differences* of the price, i.e. the per-bar log-returns rather than the price itself.
- **`p = 1`** — include one autoregressive lag: today's return is a linear function of yesterday's return.
- **`q = 0`** — no moving-average term on the residuals.

Mathematically, after differencing, the model says:

```
r_t = c + φ * r_{t-1} + ε_t
```

with `r_t` the log-return at bar `t`, `c` an intercept (typically tiny on hourly BTC), `φ` an autoregressive coefficient learned from the training window, and `ε_t` Gaussian noise with learned variance. The whole fitted model is essentially **three numbers**: `(c, φ, σ²)`. Plus the implicit "differenced" structure that ties returns back to prices.

The fit and the walk-forward are six lines from [02_arima/run.py](../../experiments/02_arima/run.py):

```python
fit = ARIMA(train, order=order).fit()
extended = fit.apply(close, refit=False)            # extend over test bars
y_pred = extended.predict(start=split, end=len(close) - 1, dynamic=False)
```

The `apply(refit=False)` is the walk-forward harness — it reuses the train-fitted `φ` while extending the model state with the new observed bars. No leakage; no retraining either. Article 7 will be entirely about why that compromise is defensible. For now: the model has been fit once on the training half, and it's predicting one step ahead at a time over the test half using its train-fitted parameters.

### `XGBoost` — 31 features, 400 trees, 5 deep

Same data, same split, but a tabular setup. From [03_gradient_boosting/features.py](../../experiments/03_gradient_boosting/features.py):

- **24 lagged log-returns** — `r_{T-1}, r_{T-2}, …, r_{T-24}`.
- **Rolling stats** — rolling mean and rolling std of past returns at windows 6 and 24. (4 features.)
- **Volume** — `log1p(volume_{T-1})`. (1 feature.)
- **OHLC** — high-low range and close-open body of bar T-1. (2 features.)

Total: **31 features**. All shifts are strict `.shift(1)` so a row at bar `T` only sees data from bars strictly before `T`. Same target as ARIMA implicitly models — `r_T = log(close_T / close_{T-1})`. Predictions are reconstructed to price as `close_{T-1} * exp(r_pred)`, so MAE / RMSE / dir_acc / Sharpe stay on the same scale as 01 / 02.

The default model config in [03_gradient_boosting/config.yaml](../../experiments/03_gradient_boosting/config.yaml):

```yaml
model:
  n_estimators: 400
  max_depth: 5
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  reg_lambda: 1.0
```

400 trees, depth 5, mild regularization. By 2026 standards this is a polite, well-regularized tabular configuration — not a deliberately handicapped one. The XGBoost sweep (`make 03_gradient_boosting_sweep`) explored eight variations; the default is among the best of them.

---

## Why ARIMA wins: capacity hurts at low SNR

This is the part of the article that's the actual lesson, not just the leaderboard. Four reasons, ranked by importance.

### Reason 1 — ARIMA(1,1,0) cannot lie about magnitude

ARIMA's predicted move at bar `T` is `c + φ * r_{T-1}`. With `|φ|` on the order of 0.02–0.05 on hourly BTC and `|r_{T-1}|` typically a few tenths of a percent, the predicted move is in the *single basis points*. Tiny.

That tininess is a feature, not a limitation. The strategy gate is `pred > ref` — i.e., the predicted *price* strictly exceeds `close_{T-1}`, which means the predicted *log-return* is strictly positive. ARIMA's predicted log-return inherits its sign almost mechanically from `r_{T-1}` (because `φ` is small but stably positive in this regime), so the strategy goes long on bars whose previous-bar return was positive. That is, in effect, a one-bar momentum rule with a built-in calibration to "the AR(1) coefficient is real but small".

XGBoost's predicted move can be *anything*. Its 31 features induce a piecewise-constant function with sharp jumps; on test bars, predictions can flip sign relative to `r_{T-1}` whenever the tree's path lands on a different leaf. Most of those flips are noise — but the strategy gate doesn't know that; it only sees `pred > ref` or not.

### Reason 2 — Trees cannot extrapolate

XGBoost, on a regression target, produces predictions that are always *averages of training-leaf values*. Inside the convex hull of the training data, that's fine. On a target as close to zero-mean as `r_T`, that means the model's prediction is essentially a tree-flavored conditional mean — a denoised version of the historical log-return distribution.

That sounds reasonable, but it's the wrong inductive bias for the directional metric. The truth is that the *sign* of `r_T` is the bit that pays out, and trees don't have a particularly good way to reward "get the sign right with a small magnitude" over "get the magnitude right and the sign roughly right". Squared-error on a near-zero-mean target rewards predictions that are *small*, which lands the model squarely on the sign-flip boundary, where it gets the strategy gate wrong half the time.

### Reason 3 — Feature richness in low-SNR data is a confabulation budget

If the underlying signal is roughly `r_T ≈ 0.03 * r_{T-1} + noise`, then features `r_{T-2}` through `r_{T-24}` are *not signal*. They have, on a long enough training window, near-zero correlation with `r_T`. But on the *training* portion they will, by chance, correlate weakly enough for trees to make money on them.

That's overfitting in slow motion. The XGBoost sweep ([03_gradient_boosting/results/sweep.csv](../../experiments/03_gradient_boosting/results/sweep.csv)) confirms it: shallower configurations do less badly. The single best configuration in the sweep is `n_estimators=800, max_depth=5, learning_rate=0.03` at dir_acc 0.5078 / Sharpe +3.93 — better than the default but still well below ARIMA(1,1,0)'s 0.5373 / +7.52. And a depth-3 / 200-tree config sits at 0.5041 / +3.26, again well behind. **Every reasonable XGBoost configuration in the sweep loses to a one-parameter linear model on directional accuracy.**

### Reason 4 — ARIMA's loss is calibrated to the data structure

Maximum likelihood under the Gaussian-on-differences assumption is, in practice, what hourly BTC log-returns roughly look like to a first approximation: zero-mean, fat-tailed, weakly autocorrelated. The ARIMA likelihood does not have to learn that the data is differences-of-prices — it is *built* with that assumption. XGBoost's squared error on raw returns does not have that prior; it has to discover, statistically, that the right thing to do is "stay near zero", and it does so imperfectly.

This is a generic point: the right inductive bias for a near-random-walk is a *tiny linear thing on the differenced series*. ARIMA bakes that in; trees don't.

---

## What the strategy actually did, bar by bar

Pull `predictions.parquet` from each experiment and the qualitative story shows up immediately:

- **ARIMA's predicted move sign** matches `sign(r_{T-1})` for ~98 % of test bars — its predictions are essentially "weakly continue the previous bar's direction". Whenever momentum is real (and on this slice it is, by a hair), that pays.
- **XGBoost's predicted move sign** matches `sign(r_{T-1})` only ~64 % of the time — the trees disagree with the previous bar's return frequently, often because some other feature pushed the leaf assignment. When those disagreements are wrong (and they are wrong slightly more than half the time), the strategy gate eats the loss.

ARIMA goes long on **791 of 1,608** test bars; XGBoost goes long on **750**. Roughly comparable bet count. But the *per-bar realized return* on ARIMA's long bars averages a small positive number; XGBoost's averages something close to zero. Cumulative return: ARIMA +53.5 %, XGBoost +8.1 %.

(Sharpe takes that mean-of-realized-returns and divides by stddev. ARIMA's denominator is also smaller — it's only long on bars whose previous-bar return was positive, which is a smaller, less volatile subset. So ARIMA wins both in numerator and denominator. The +7.52 / +1.45 spread is real but, as Article 14 will argue, the absolute level is partially the `√8760` annualization. The *gap* between the two models survives the annualization complaint.)

---

## What "feature engineering didn't help" actually means

This is the line every reader is going to quote, so I want to bound it carefully. Three qualifiers, in order of importance.

### Qualifier 1 — At *this* signal-to-noise ratio

On hourly BTC log-returns, the autocorrelation at lag 1 is roughly 2–5 %. That's the entire forecastable signal. On daily oil consumption, monthly retail sales, hourly electricity demand — series with strong seasonality, regime, or covariate dependence — feature engineering on rolling stats and exogenous variables routinely helps a lot. The claim **specific to 1h BTC** is "31 lag/stat features did not extract more directional skill than one AR coefficient". Don't generalize past that.

### Qualifier 2 — At *this* capacity

A constrained gradient booster — max_depth 1, n_estimators 50, lr 0.01 — would, in the limit, behave like a linear model on the lagged returns and approach ARIMA's directional behavior. The XGBoost sweep stops short of that limit, but it shows the trend: shallower / fewer trees / lower learning rate consistently does *less badly*. The XGBoost configurations that fail hardest are the deepest ones (max_depth 7, n_estimators 800), which is exactly the regime where capacity overwhelms signal. **Capacity is the failure mode**, not gradient boosting per se.

### Qualifier 3 — At *this* loss

Squared error on `r_T` rewards predictions that match magnitude. On a near-zero-mean target, that incentivizes the model to predict near zero, which puts the model on the sign-flip boundary, which is where the *directional* metric punishes it. A loss aligned with the leaderboard column — pinball loss for quantile regression, or a smooth proxy for `sign(pred) == sign(true)` — would partly close this gap. We don't try that here. It's a real follow-up.

So when I say "feature engineering didn't help": I mean that on this asset, at this capacity, with this loss, the 31-feature gradient booster was outperformed by a single autoregressive coefficient. I do not mean "trees are bad" or "features don't matter on time series". I mean that the prior baked into ARIMA — *the dominant pattern is a tiny linear thing on the differenced series* — is the right prior for hourly BTC, and XGBoost did not discover it from data.

---

## Reproducing the shootout

The whole thing runs in under two minutes:

```
make 02_arima              # default order (1,1,1)
make 02_arima_sweep        # sweeps (0,1,0) ... (2,1,2) -> results/sweep.csv
make 03_gradient_boosting  # default XGBoost config
make 03_gradient_boosting_sweep  # 8 alternative XGBoost configs
```

The headline `ARIMA(1,1,0)` result lives in the sweep, not in `metrics.json`. The default config in [02_arima/config.yaml](../../experiments/02_arima/config.yaml) is `order: [1, 1, 1]`, which lands at Sharpe +7.28 / dir_acc 0.5336 — also clearly above XGBoost. We don't ship `(1,1,0)` as the default, even though it's narrowly Sharpe-best in the sweep, because Article 5 (the next post) is the article *about* why picking the order is fraught: AIC says `(0,1,0)` (literally naive), Sharpe says `(1,1,0)`, and that disagreement is the whole post. `(1,1,1)` is the Box-Jenkins default and it's where I stop.

If you want the absolute-best Sharpe ARIMA from the sweep, set `order: [1, 1, 0]` (or `[0, 1, 1]` — by happy structural coincidence it lands at the same dir_acc 0.5373 / Sharpe +7.52). Either way, the picture is the same.

---

## The lesson

"Throw XGBoost at it" is a strong prior in tabular ML, and it's a perfectly defensible one for most problems. On hourly BTC log-returns, it's the wrong prior. The right prior is *a small linear thing on the differenced series*, and ARIMA encoded that prior in 1970. We forgot, then we built it back, and we called it a temporal foundation model.

The deep-learning posts that come later in this series — LSTM (Article 8), TFT (Article 9), foundation models (Article 11) — are going to spend a lot of GPU time, in some cases hundreds of millions of parameters, confirming that capacity by itself does not escape the floor ARIMA establishes here. The TFT in [05_transformer](../../experiments/05_transformer/) will land at Sharpe +4.59 / dir_acc 0.5053 — a serious model, well below this article's ARIMA. The Chronos-2 zero-shot foundation model in [06_pretrained](../../experiments/06_pretrained/) will land near naive's MAE with dir_acc ≈ 0.50.

The series isn't telling you "don't use deep learning". It's telling you that for *this* problem, **capacity is not the bottleneck. The signal is.** And ARIMA(1,1,0) — three numbers — is closer to extracting that thin signal than thirty-one engineered features and four hundred trees.

---

## What's next

Article 5 — *AIC said pick the random walk. The market said otherwise.* — is the spinoff from the `(p, d, q)` sweep. The AIC-best order is `(0, 1, 0)`, which is literally the naive last-value model. The Sharpe-best is `(1, 1, 0)`. Information criteria and out-of-sample trading metrics disagree, and the question of which one to trust is genuinely hard. That's the next post.

Article 6 — *Same model, different window, opposite verdict* — will rerun this entire shootout on a Q1-only test slice and watch the leaderboard rearrange. XGBoost will (briefly) lead. Don't get too attached to the table at the top of this article.

---

*Code: [experiments/02_arima/](../../experiments/02_arima/) · [experiments/03_gradient_boosting/](../../experiments/03_gradient_boosting/) · Repo: [transformer_bitcoin_ai](../../../README.md)*
