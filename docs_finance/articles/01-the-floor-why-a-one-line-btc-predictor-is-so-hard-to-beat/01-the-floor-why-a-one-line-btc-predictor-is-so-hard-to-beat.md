# The Floor: why a one-line BTC predictor is so hard to beat

*Article 1 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

I built the dumbest possible Bitcoin predictor. It is one line:

```python
close_pred[t] = close[t-1]
```

That's it. No training. No parameters. No features. No GPU. The model says: *whatever the price was at the last bar, that's my prediction for the next bar.*

On hourly BTCUSDT from 2024-01-01 to 2024-12-01, evaluated on the last 20 % as a holdout, this one-liner clocks:

```json
{
  "rows_test": 1608,
  "mae":  260.50,
  "rmse": 396.97,
  "mape": 0.003407,
  "directional_accuracy": NaN
}
```

A mean absolute error of $260 on a ~$65 k asset. A mean absolute percentage error of **0.34 %**. Those are the numbers I now have to beat with every transformer, LSTM, and gradient-booster I'm going to spend the next twelve articles building. And they're a real fight. This first post is about *why* — and why I think the only honest takeaway is that **MAE on a near-random-walk doesn't measure what you think it measures.**

This is the opening post of a series on whether BTC is actually predictable. Every later experiment in the repo gets compared against this baseline. So before I show you a transformer, I have to show you the wall the transformer hits.

---

## The model, the data, the run

The setup lives in [experiments/01_baseline_naive/](../../experiments/01_baseline_naive/). The whole thing is ~70 lines. The forecasting bit is four:

```python
close = df['close']
split = int(len(close) * (1.0 - test_fraction))
test  = close.iloc[split:]
y_pred = close.iloc[split - 1:-1]   # shifted one bar
```

Time-ordered split — no shuffling, because order matters in time series — last 20 % held out. The prediction at every test bar is the close of the previous bar. Then the shared metrics module ([src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py)) computes MAE, RMSE, MAPE, and a directional-accuracy score that I'll come back to.

To reproduce:

```
make 01_baseline_naive
```

You get a `metrics.json`, a `predictions.parquet`, and a plot. That same loader, splitter, and metrics module is what every later experiment uses. When I later say "ARIMA beat naive" or "the transformer didn't", it really is the same data slice, same test window, same `mae()` function.

---

## Why this works: the random-walk floor

The reason a one-line predictor isn't a punchline is that 1-hour BTC is *very close* to a random walk:

```
close[t] = close[t-1] + ε[t]
```

with `ε[t]` a noise term that is roughly zero-mean. For a true random walk, the conditional mean of `close[t]` given everything you know up to bar `t-1` is exactly `close[t-1]`. There is nothing else to use. The optimal point forecast under squared error is the last observation. Under absolute error it's the median of the noise — also `close[t-1]` if the noise is symmetric.

In other words: when a series is genuinely a random walk, **the naive forecaster is the conditional mean**. It is not "the dumb baseline" — it is the optimal point forecast, full stop. Anything fancier is fitting noise.

BTC at 1 h is not a *perfect* random walk. There's a smidge of mean reversion, some volatility clustering, and some calendar effects buried in there. (Subsequent articles will hunt for them.) But the leftover signal is small relative to the noise, and that small ratio is what makes naive so hard to beat in MAE terms.

---

## What MAE is actually measuring

Here is the trick that took me longer than it should have to internalize. The naive predictor's per-bar error is:

```
err[t] = close[t] - close[t-1]
```

That's literally **the bar's actual move**. So:

```
MAE_naive = mean(|close[t] - close[t-1]|) = mean absolute hourly move
```

`MAE_naive` is not a model error in any meaningful sense. It is **a summary statistic of BTC's hourly volatility on the test window**. $260.50 mean absolute hourly move on a ~$65 k asset works out to roughly 0.4 % per bar. That is just what BTC was doing in late 2024.

This has two consequences I want every subsequent post in this series to inherit:

1. **A model that beats `MAE_naive` is one whose forecast errors are smaller than the asset's natural per-bar volatility.** That is a strong claim. In practice almost nothing does it on hourly BTC by more than a hair.
2. **A 0.34 % MAPE is not a brag.** It is "the price didn't move much per bar". Anyone shipping a Medium post that headlines a 0.4 % MAPE on hourly crypto is reporting volatility, not skill. Including, briefly, me. Don't.

If you take exactly one thing from this article, take this: **MAE on hourly BTC measures the asset, not the model.** The floor is a volatility number.

---

## The `NaN` is the point

Look at the directional accuracy: `NaN`. That is not a bug — it is the most honest output naive can produce. The metric is defined like this ([src/btc_ai/eval/metrics.py:23-34](../../src/btc_ai/eval/metrics.py#L23-L34)):

```python
def directional_accuracy(y_true, y_pred, ref):
    true_dir = np.sign(y_true.to_numpy() - ref.to_numpy())
    pred_dir = np.sign(y_pred.to_numpy() - ref.to_numpy())
    mask = (true_dir != 0) & (pred_dir != 0)
    if not mask.any():
        return float('nan')
    return float(np.mean(true_dir[mask] == pred_dir[mask]))
```

For naive, `y_pred[t] = ref[t] = close[t-1]`, so `pred_dir` is identically zero. The mask is empty. The function returns `NaN`. The model **never expressed a direction**. It refused to take a side.

That is the right behavior. Naive is informational about *level*, not about *the next move*. It says "the price will be near where it is now" — which on hourly BTC is true and almost useless for trading. A long/flat strategy driven by naive's prediction is permanently flat: it never sees a positive predicted move, never goes long, never makes or loses anything.

So the `NaN` is the line in the sand. **Any model that wants to claim it has learned something useful for trading must produce a non-NaN directional accuracy and beat 0.50.** A model that ties naive on MAE but sits at dir_acc ≈ 0.50 is not better than naive for any decision you care about. MAE and PnL are different goals on this data, and the rest of the series will keep proving it.

---

## The lab convention this forces

Every other experiment in [transformer_bitcoin_ai](../../) inherits this baseline as the floor:

- [02_arima](../../experiments/02_arima/) — `ARIMA(1,1,0)` lifts dir_acc to 0.5373 with Sharpe +7.52, while barely shaving naive's MAE. Article 4 will dig into this.
- [03_gradient_boosting](../../experiments/03_gradient_boosting/) — XGBoost with 31 engineered features lands at dir_acc 0.4872 (worse than chance) on the same window. Same article.
- [04_lstm](../../experiments/04_lstm/), [05_transformer](../../experiments/05_transformer/), [06_pretrained](../../experiments/06_pretrained/) — same story, judged on direction and Sharpe, with MAE kept on the report card so we can also see who is a better denoiser.

The leaderboard column is **directional accuracy**, with Sharpe as the trading-weighted refinement. MAE stays visible — it's still a useful diagnostic for "did this model at least learn the level?" — but it is not the headline. That decision falls out of this article. If MAE_naive is mostly volatility, ranking models by MAE on top of that floor is ranking them by how well they tracked the noise, which is not what we hired them for.

---

## Reproducing the floor

If you want the experience first-hand, the whole thing is six commands from a clean clone:

```
brew install uv
uv sync
make fetch
make 01_baseline_naive
```

You'll see the same `metrics.json` content quoted at the top of this post. The implementation worth reading is here:

```python
# experiments/01_baseline_naive/run.py — the actual prediction
close = df['close']
split = int(len(close) * (1.0 - test_fraction))
test  = close.iloc[split:]
ref   = close.iloc[split - 1:-1]
ref.index = test.index
y_pred = ref

metrics = {
    'mae':  mae(test, y_pred),
    'rmse': rmse(test, y_pred),
    'mape': mape(test, y_pred),
    'directional_accuracy': directional_accuracy(test, y_pred, ref),
}
```

That's the entirety of the model. The rest of the file is config loading, plotting, and writing the report.

---

## What the floor does *not* say

I want to be careful with what this baseline actually proves, because the obvious takeaway ("BTC is unpredictable, go home") is too strong.

- **It doesn't say BTC is unpredictable.** It says *its 1-hour conditional mean is hard to beat with point forecasts*. Predictability could live in direction, in volatility, in the tails, or at a different timescale.
- **It doesn't say models are useless.** It says the *right output* probably isn't a price. It might be a directional opinion (Article 3 onward), a quantile band (Article 12), or a position size derived from a probabilistic forecast.
- **It doesn't say the floor is constant.** Change the slice, change the floor — Q1 2024 has a different volatility than Sep–Nov 2024. Every later experiment is run on multiple windows for exactly this reason (Article 6).
- **It doesn't say MAE is useless.** It says MAE on a near-random-walk *as a leaderboard metric* is misleading. On a less noisy series — daily oil consumption, monthly retail sales — MAE is fine. On hourly BTC, it's measuring the wrong thing.

---

## What the rest of this series is about

The naive baseline is the floor. The series from here is one long answer to the question:

> Given this floor, what evidence would convince us a model is doing real directional work?

The shape of that evidence matters a lot. It has to be:
- A **directional accuracy** materially above 0.50, on a **specified test window**, after a **specified train cutoff**, with the **same loader and splitter** as the baseline.
- Stable across at least two non-overlapping windows (Article 6 on regime-fit will be ruthless about this).
- Robust to the test window's drift (Article 10 on Sharpe-vs-beta will show how easy it is to fake skill in a rising market).

If a model can't clear that bar, it didn't beat naive in any way that matters, no matter what its MAE column says.

The next post in the series ([Article 2 — *How I structured a forecasting lab so nothing rots*](../../articles_todo.md)) is about the engineering: how the repo enforces "same data, same metrics, same report" so that every leaderboard row in this series is honestly comparable. Then we get into the models.

The tease for Article 4: a single AR(1) coefficient on differenced log-returns, fit by `statsmodels` in less code than this article's title, beats a 31-feature XGBoost on directional accuracy by five percentage points. The floor is real. The path through it is narrower than I expected.

---

*Code: [experiments/01_baseline_naive/](../../experiments/01_baseline_naive/) · Repo: [transformer_bitcoin_ai](../../../README.md)*
