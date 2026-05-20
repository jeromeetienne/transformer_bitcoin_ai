# Outline — ARIMA(1,1,0) beat my XGBoost. One parameter beat thirty-one features.

## One-line pitch
A single AR(1) coefficient on differenced returns produces `dir_acc 0.5373 / Sharpe +7.52` ([02_arima sweep](../../experiments/02_arima/results/sweep.csv) row `(1,1,0)`), while XGBoost with 24 lagged returns + rolling stats + volume + OHLC lands at `0.4872 / +1.45` (default config from [03_gradient_boosting](../../experiments/03_gradient_boosting/results/metrics.json)). One parameter beat thirty-one features.

## Audience
Quant-curious ML engineers who default to "throw a gradient booster at it" on any tabular-feeling forecasting problem, plus anyone who has ever heard "ARIMA is dead, use deep learning". Series readers who saw Article 3 set up the dir_acc / Sharpe leaderboard and want to see it used.

## Thesis
On a near-random-walk like 1h BTC, the dominant signal is a *very* mild autoregressive return — exactly what `ARIMA(1,1,0)` is built to capture. A flexible non-linear model with thirty-one engineered features doesn't know to keep its predictions *small enough*; it confabulates structure that isn't there. Capacity hurts when signal-to-noise is this low. **Feature engineering didn't help** is the headline.

## Structure

### 1. The hook — David vs Goliath, with both numbers
- Two contestants, same data slice (Article 1's slice), same metrics module.
- ARIMA(1,1,0): one autoregressive coefficient on first differences. Closed-form fit. ~30 seconds.
- XGBoost: 31 features, 400 trees, depth 5, regularization, the works. ~1 minute fit.
- The headline numbers, side-by-side. The reader should be a little uncomfortable.

### 2. The leaderboard
| Model | dir_acc | Sharpe | MAE | rows |
|---|---|---|---|---|
| Naive | NaN | NaN | $260.50 | 1608 |
| **ARIMA(1,1,0)** | **0.5373** | **+7.52** | $260.49 | 1608 |
| **XGBoost** | **0.4872** | **+1.45** | $270.73 | 1603 |

(Numbers from [02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv) and [03_gradient_boosting/results/metrics.json](../../experiments/03_gradient_boosting/results/metrics.json).)

ARIMA wins on every leaderboard column. XGBoost is *below 0.50* on dir_acc — worse than chance on the bars where it expresses an opinion.

### 3. What ARIMA(1,1,0) actually is
- `(p, d, q) = (1, 1, 0)`. Translation:
  - `d=1`: model first differences (price changes), not the price itself.
  - `p=1`: include one autoregressive term — `r_t = φ * r_{t-1} + ε_t`.
  - `q=0`: no moving-average residual term.
- The fit produces *one* coefficient `φ`. Plus a small intercept and a noise variance.
- Show the few lines of code from [run.py](../../experiments/02_arima/run.py): `ARIMA(train, order=order).fit()`, then walk-forward via `apply(refit=False)` + `predict(dynamic=False)`.

### 4. What XGBoost is being asked to do
- Target: the same log-return as ARIMA models implicitly.
- Features ([features.py](../../experiments/03_gradient_boosting/features.py)):
  - 24 lagged log-returns: r_{T-1}, ..., r_{T-24}
  - Rolling mean + rolling std of returns at windows 6 and 24
  - log1p(volume) of bar T-1
  - high-low range and close-open body of bar T-1
  - = **31 features**
- Default model config: 400 trees, depth 5, lr 0.05, subsample 0.8.
- All shifts are strict (`.shift(1)`) — no current-bar leakage. The pipeline is identical to ARIMA's apart from the model.

### 5. Why ARIMA wins: capacity hurts on low SNR
This is the meat of the article.

**Reason 1 — ARIMA(1,1,0) cannot lie about return magnitude.** Its prediction at bar T is `φ * r_{T-1}`. With `|φ|` typically very small (the AR(1) coefficient on hourly BTC returns is something like 0.02–0.05), the predicted move is *tiny*. The strategy gate is `pred > ref` — i.e., `predicted move > 0`. ARIMA's predicted move is small but *consistently signed* with the prior bar's return; XGBoost's predicted move is big enough to flip sign on noise.

**Reason 2 — Trees cannot extrapolate.** XGBoost on a regression target produces a piecewise-constant function inside the convex hull of the training data. The training data's mean log-return is roughly zero with a fat tail. The model's predictions are therefore a tree-flavored *mean*, with sharp jumps. Reconstructing price as `close_{T-1} * exp(r_pred)` then turns small but *systematically wrong-signed* return predictions into a wandering price track that gets the direction wrong 51 % of the time.

**Reason 3 — Feature richness in low-SNR data is a confabulation budget.** The 31 features each give the trees a degree of freedom to "explain" patterns. When the underlying signal is `r_t ≈ 0.03 * r_{t-1} + noise`, every feature past `r_{t-1}` is a route to overfitting. `r_{t-2}` through `r_{t-24}` look like signal in-sample because by chance some of them correlate weakly; out of sample they're noise.

**Reason 4 — ARIMA's loss is calibrated to the data structure.** Maximum likelihood under the differenced-Gaussian assumption matches what hourly BTC log-returns roughly look like. XGBoost's squared error on raw returns is fine, but its splitting decisions are made on individual feature thresholds, which is the wrong inductive bias for "the dominant pattern is `φ * r_{t-1}`".

### 6. What the strategy did, bar by bar
- Pull `predictions.parquet` from each experiment.
- ARIMA: when its tiny predicted return aligns with the previous bar's return, it goes long; that lines up with very mild momentum in the test slice and pays Sharpe +7.52.
- XGBoost: predicted return frequently changes sign vs the previous bar's return because the tree partitions don't preserve sign. The strategy goes long at the wrong moments — long when the next bar drops, flat when it rises.
- Concrete: cumulative return ARIMA = +53 %, XGBoost = +8 %. Even on cum_ret, the difference is decisive.

### 7. What "feature engineering didn't help" actually means
Be careful — this is the line every reader will quote. Three honest qualifiers:
1. **At this signal-to-noise ratio.** On daily oil prices or monthly retail sales, feature engineering on rolling stats and exogenous variables routinely helps. The claim is specific to 1h BTC log-returns.
2. **At this capacity.** A *constrained* gradient booster (max_depth 1, n_estimators 50, lr 0.01) — basically, hand-built linear regression — would land closer to ARIMA. The XGBoost sweep ([sweep.csv](../../experiments/03_gradient_boosting/results/sweep.csv)) confirms shallower trees do less badly. Capacity is the issue.
3. **At this loss.** Squared error on r_T means the model is rewarded for matching magnitude. On a near-zero-mean target, that's a recipe for a model that's right on the L2 metric but wrong on the directional one. A directional / quantile loss would partly fix this.

### 8. Reproducing it
```
make 02_arima_sweep   # populates results/sweep.csv with (0,1,0)..(2,1,2)
make 03_gradient_boosting
```
- For ARIMA, the default config is `order: [1, 1, 1]` (Sharpe +7.28 / dir_acc 0.5336). The (1,1,0) result quoted above lives in the sweep — Article 5 will explore why we don't ship (1,1,0) as the default.
- For XGBoost, default config is `n_estimators: 400, max_depth: 5, learning_rate: 0.05`.

### 9. The lesson
- "Throw XGBoost at it" is a prior. On a near-random-walk it is the wrong prior.
- The right prior on hourly crypto returns is *small*: differenced returns, one AR coefficient, calibrated likelihood. ARIMA encoded that prior in 1970 and we forgot.
- Subsequent posts in this series — LSTM (Article 8), TFT (Article 9), foundation models (Article 11) — will spend a lot of GPU time confirming that more capacity does not, by itself, escape the floor ARIMA establishes.

### 10. Closing — what to read next
- Article 5 is the spin-off: the ARIMA order sweep also reveals that AIC picks `(0,1,0)` (literally naive), while Sharpe picks `(1,1,0)`. Information criteria and out-of-sample trading metrics disagree.
- Article 6 will rerun this entire shootout on a different test window. The verdict will not survive cleanly.

## Key code/file references
- [experiments/02_arima/run.py](../../experiments/02_arima/run.py) — the 100-line ARIMA pipeline
- [experiments/02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv) — `(1,1,0)` row is the headline
- [experiments/03_gradient_boosting/features.py](../../experiments/03_gradient_boosting/features.py) — 31 features
- [experiments/03_gradient_boosting/run.py](../../experiments/03_gradient_boosting/run.py) — XGBoost wrapper
- [experiments/03_gradient_boosting/results/metrics.json](../../experiments/03_gradient_boosting/results/metrics.json) — the headline 0.4872 / +1.45

## Tone notes
- Lab-notebook honest. Don't gloat — XGBoost is a fine tool, this domain is just hostile.
- Land the headline early, defend it carefully, and qualify it.

## Length target
~1,800–2,200 words.
