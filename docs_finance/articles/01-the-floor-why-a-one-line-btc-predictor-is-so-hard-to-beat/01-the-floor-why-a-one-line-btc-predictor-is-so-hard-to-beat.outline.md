# Outline — The Floor: why a one-line BTC predictor is so hard to beat

## One-line pitch
The naive last-value baseline (`close_pred[t] = close[t-1]`) clocks **MAE = $260.50** on 1h BTCUSDT. Frame why "predict the last price" is a mathematical floor on a near-random-walk, why MAE at this resolution measures volatility rather than skill, and why every later experiment is judged by **direction**, not error.

## Audience
ML practitioners and quant-curious engineers who have built a "BTC predictor", seen a small MAPE, and wondered why their PnL is flat. Series opener — readers may not have seen the rest of the lab.

## Thesis (the one thing the reader should leave with)
On a near-random-walk like 1h BTC, MAE measures **volatility, not skill**. A one-line model with no parameters is a hard floor; beating it requires directional information that USD-error metrics cannot see.

## Structure

### 1. The hook — "I built the dumbest BTC predictor and it's hard to beat"
- The model: `close_pred[t] = close[t-1]`. No training, no parameters.
- The number on 1h BTCUSDT, 2024-01-01 → 2024-12-01, last 20% as test:
  - 1,608 test bars
  - **MAE = $260.50**
  - **RMSE = $396.97**
  - **MAPE = 0.34 %**
  - **Directional accuracy = NaN** (by design — explained later)
- The provocation: 0.34 % sounds tiny. Show why "tiny" is a trap.

### 2. Why this works — the random-walk floor
- Quick refresher: random walk = `close[t] = close[t-1] + ε[t]`, with `ε[t]` zero-mean noise.
- For a true random walk, the **best constant-functional point predictor in MAE is the last observation**.
- Show the math intuition without proofs: any bounded function of past prices fights the noise; the optimal in expectation is "do nothing".
- BTC at 1h is *not* a perfect random walk, but it is close enough that the leftover signal is small relative to the noise.

### 3. What MAE is actually measuring
- The naive predictor's per-bar error is `close[t] - close[t-1]` — the **bar's actual move**.
- So MAE-of-naive = mean absolute hourly move = a volatility number.
- $260 on a $65k asset = ~0.4 % typical hourly move. That's just BTC.
- Concrete consequence: a "better MAE" means *less than the asset's natural volatility per bar* — a strong claim.
- Side note: a MAPE of 0.34 % looks like an A+ on a school exam. On BTC it's literally "the price didn't move much that hour". Don't ship MAPE as a headline.

### 4. The directional accuracy gotcha
- Naive returns `directional_accuracy = NaN`. Show the metric ([src/btc_ai/eval/metrics.py:23-34](../../../src/btc_ai/eval/metrics.py#L23-L34)).
- Why: predicted move = `pred - ref = close[t-1] - close[t-1] = 0`. `sign(0) = 0`. The bar is excluded from the mask. NaN.
- This is not a bug. The naive baseline **expresses no opinion** about direction — it is informational about *level*, not about *the next move*.
- Implication: any model that beats naive on MAE but stays at dir_acc ≈ 0.50 is not better than naive *for trading*. MAE and PnL are different goals.

### 5. The lab convention this forces
- Every later experiment in this repo (ARIMA, XGBoost, LSTM, TFT, Chronos-2, ...) is evaluated against this exact baseline on the same test slice.
- Headline metric is **directional accuracy + Sharpe**, not MAE.
- MAE stays in the report so we can see if a model also wins as a denoiser, but it is not the leaderboard column.
- Foreshadow: in [02_arima](../../experiments/02_arima/), an `ARIMA(1,1,0)` improves dir_acc to 0.5373 / Sharpe +7.52 with a single coefficient, while the naive MAE is barely beaten. That's the lesson — small directional improvements matter more than small MAE improvements.

### 6. Reproducing the floor (5-line repo demo)
- Show the actual run: `make 01_baseline`.
- Pull the relevant snippet from [run.py](../../experiments/01_baseline/run.py) — the prediction is literally `close.shift(1)` on the test slice.
- Show the produced `metrics.json`.
- Note the loader, splitter, and metrics module are shared across every experiment in [src/btc_ai/](../../src/btc_ai/) — so when later posts say "ARIMA beats this", it really is the same number, same slice, same file.

### 7. What this floor doesn't say
- It doesn't say BTC is unpredictable — it says **its 1h conditional mean is hard to beat with point forecasts**.
- It doesn't say models are useless — it says the **right output isn't a price, it's a directional opinion or a quantile band**.
- It doesn't say the floor is constant — change the slice, change the floor. Volatility is regime-dependent (see Article 6 on regime-fit).

### 8. Closing — what the rest of the series is about
- The naive baseline is the floor.
- The rest of the series asks: *given this floor, what evidence would convince us a model is doing real directional work?*
- Tease: ARIMA(1,1,0) does, XGBoost mostly doesn't, the transformer the repo is named after... well, see Article 9.

## Key code/file references
- [experiments/01_baseline/run.py](../../experiments/01_baseline/run.py) — 70-line implementation
- [experiments/01_baseline/config.yaml](../../experiments/01_baseline/config.yaml) — data slice
- [experiments/01_baseline/results/metrics.json](../../experiments/01_baseline/results/metrics.json) — the headline numbers
- [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py) — shared MAE / RMSE / MAPE / directional accuracy
- [experiments/01_baseline/README.md](../../experiments/01_baseline/README.md) — README this article popularizes

## Tone notes
- Lab-notebook honest, not hype. The naive baseline is presented as a *worthy adversary*, not a punchline.
- No equations beyond the random-walk identity. One small block of Python is enough.
- Numbers must match `metrics.json` exactly — the post is grounded in the actual run, not approximations.

## Length target
~1,400–1,800 words. Long enough to set up the series, short enough that a reader leaves with one crisp idea: *the floor is real, and MAE is the wrong metric to fight it with*.
