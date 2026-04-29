# The mean-reverter that fixed my metrics

*Article 3 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

I built a worse baseline on purpose.

[01b_moving_average](../../experiments/01b_moving_average/) — a 24-bar rolling mean — is **2.9× worse than naive on MAE** ($756.19 vs $260.50) and 2.9× worse on MAPE (0.99 % vs 0.34 %). By every standard "predictor accuracy" headline, this thing is awful. It's also the single most useful baseline in this lab, because it's the first model in the lineup that *expresses an opinion about direction*, and that's the metric every later post in this series is judged on.

This is the article where the leaderboard gets a soul.

---

## The numbers

`make 01b_moving_average`, on the same hourly BTCUSDT slice as Article 1 (2024-01-01 → 2024-12-01, last 20 % held out), produces:

```json
{
  "experiment": "01b_moving_average",
  "window": 24,
  "rows_total": 8040,
  "rows_test": 1608,
  "mae":  756.19,
  "rmse": 1100.93,
  "mape": 0.0099,
  "directional_accuracy": 0.5143,
  "cumulative_return":   0.2567,
  "annualized_sharpe":   4.33
}
```

Side-by-side with Article 1's naive result:

| | MAE | MAPE | dir_acc | cum_return | Sharpe |
|---|---|---|---|---|---|
| **Naive** | $260.50 | 0.34 % | NaN | — | — |
| **MA(24)** | $756.19 | 0.99 % | 0.5143 | +25.67 % | +4.33 |

By the MAE column, naive obliterates this. By every column to the right of it, naive doesn't show up at all.

---

## Why MA(24) is mechanically worse on MAE

Two lines of intuition. Article 1 showed that the naive predictor's per-bar error is exactly *the bar's actual move*:

```
err_naive[t] = close[t] - close[t-1]   ≈ ±$260 typical
```

The moving-average predictor's per-bar error is something different:

```
err_MA[t]    = close[t] - mean(close[t-24:t])
```

On a series with a non-zero local trend, the rolling mean *lags*. The ~24 hours leading up to bar `t` had, on average, lower prices than bar `t` itself if BTC was rallying. Hourly BTC in late 2024 — moving from ~$58 k in early September to ~$96 k by late November — gives the MA exactly the kind of trending environment where it's structurally pulled toward older, lower closes.

That lag is the entire MAE story. **MAE_MA ≈ trend magnitude over the test window**, where MAE_naive ≈ noise magnitude. The MA isn't worse at predicting; it's measuring something else, the same way naive's MAE was measuring volatility, not skill. (If you skipped Article 1, that's the lens for both numbers — MAE on a near-random-walk doesn't measure what you think it measures.)

---

## Why MA(24) suddenly *has* a directional opinion

Here's the part that turns this from a punchline into a useful baseline. Compare predicted *moves*:

- **Naive.** Predicted move is `pred[t] - ref[t] = close[t-1] - close[t-1] = 0`. Always. No opinion. `directional_accuracy` returns `NaN` because the prediction is never strictly above or below the reference. The model refuses to take a side.
- **MA(24).** Predicted move is `pred[t] - ref[t] = mean(close[t-24:t]) - close[t-1]`. This is *negative* when the last bar is above the rolling mean (the MA is implicitly saying "you've gotten ahead of yourself, expect mean reversion down") and *positive* when the last bar is below it ("you've gotten behind, expect mean reversion up").

That's a **mean-reversion signal**, baked in by construction. Whether the signal is right or wrong is what `directional_accuracy = 0.5143` measures: across 1,608 test bars, the predicted move sign matched the realized move sign 51.4 % of the time. That's only 1.4 percentage points above coin-flip on the bars where the model expressed an opinion — barely above chance, exactly what you'd expect from a structural rule that has no actual market knowledge. But it's *measurable*. It's a number. And every later model in this lab now has something to be compared against on the metric we care about.

The whole forecasting code is one line:

```python
ma = close.shift(1).rolling(window).mean()
```

The `.shift(1)` is doing the heavy lifting on leakage — at bar `t`, the rolling mean only sees data strictly before `t`. The rest is loader / splitter / metrics from the shared module ([src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py)).

---

## The four metrics this baseline introduces

This is the article that introduces the trading-aware vocabulary the rest of the series uses. All four come from [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py).

### `directional_accuracy(y_true, y_pred, ref)`

Fraction of bars where the predicted move sign matches the realized move sign, masked to bars where both sides expressed an opinion. The naive baseline's identically-zero predicted move is what makes its `directional_accuracy` a `NaN`. MA(24) is the first row in the leaderboard that returns a real number. **MA(24): 0.5143.**

### `strategy_returns(y_true, y_pred, ref)`

The long-flat trading rule the lab uses for everything:

```python
def strategy_returns(y_true, y_pred, ref):
    position = (y_pred.to_numpy() > ref.to_numpy()).astype('float64')
    realized = y_true.to_numpy() / ref.to_numpy() - 1.0
    return pd.Series(position * realized, index=y_true.index)
```

Translated: go long for the next bar whenever the model predicts a *strictly positive* move (`pred > ref`), else flat. Realized return is `close[t] / close[t-1] - 1`. No transaction costs, no slippage, no leverage. It's deliberately the dumbest possible execution of a directional opinion, because we want metrics that respond cleanly to the prediction quality, not to a clever overlay.

### `cumulative_return`

The product of `(1 + r_t)` minus 1 across the test window. **MA(24): +25.67 %** over 1,608 hourly bars. Sounds like a great strategy. Article 10 (*Your Sharpe is mostly drift*) will be ruthless about how much of this is BTC's rally and how little is the model. Foreshadow only; not the discussion for this post.

### `annualized_sharpe`

```python
def annualized_sharpe(strat_ret, periods_per_year):
    sd = strat_ret.std(ddof=1)
    return float(np.sqrt(periods_per_year) * strat_ret.mean() / sd)
```

For 1h bars, `periods_per_year = 8760`. **MA(24): +4.33.** That number is shaped like a hedge-fund report. Don't get attached to it yet — Article 14 will argue that the per-bar Sharpe is the more honest unit and that the `√8760` multiplier is doing most of the marketing.

The headline metric for the leaderboard is **directional accuracy first, Sharpe second**. MAE stays in the report card as a diagnostic ("did the model learn the level?") but it is not what we rank by. Article 1 explained why; this article is where that ranking convention starts paying out.

---

## The 24-bar window is opinionated

`window: 24` ≈ "yesterday's average". On 1h data, 24 bars is one trading day, which is the smallest window a human trader can plausibly reason about as a regime. The YAML even spells out the degenerate case:

```yaml
window: 24            # MA window size, in bars. 24 = one day for 1h data.
```

A window of 1 collapses the MA back to naive last-value (sanity check). Bigger windows (48, 96, 168 = a week) trade lag for smoothing; smaller windows trade smoothing for noise. We don't sweep here — **this is a baseline, not a tuned strategy**. Sweeping the window on the test set would be exactly the kind of leaderboard fraud the rest of the lab is set up to prevent. 24 was picked because it's the most defensible default on 1h crypto, and that's the only freedom this baseline gets.

---

## Why a "bad" baseline is structurally necessary

Here's the punchline.

The leaderboard column the rest of the series cares about is **directional accuracy**, with Sharpe as the trading-weighted refinement. If the *only* baseline in the lab is naive — MAE $260, dir_acc `NaN`, Sharpe `NaN` — there is nothing for a future model to be ranked against on those columns. The first non-NaN row in the leaderboard *creates the column*.

MA(24) is the lab's null hypothesis for direction. It's the answer to the question "what does a model with no real market knowledge but a willingness to commit to a direction look like?". Specifically:
- 0.5143 dir_acc is what you get from a structural mean-reversion rule with no information.
- +4.33 Sharpe is what you get from any long/flat rule that is occasionally long during a major rally — Article 10's central insight, previewed.
- +25.7 % cumulative return is what 1,608 bars of "mostly correct sign about half the time, in a trending market" looks like.

That gives every later post in this series something concrete to beat. Article 4 is going to land XGBoost at `directional_accuracy = 0.4872` — *worse than MA(24)*. That comparison only matters because MA(24) is in the leaderboard. Without it, "0.4872" is a number floating in space; with it, "the 31-feature gradient booster lost to a 24-bar rolling mean" is a finding.

In other words: **a baseline whose job is to lose plausibly is what makes "winning" a meaningful claim about anything else.**

---

## Reproducing it

The whole experiment runs in a few seconds:

```
make 01b_moving_average
```

The forecasting code, in full:

```python
# experiments/01b_moving_average/run.py — the actual prediction
ma = close.shift(1).rolling(window).mean()

split = int(len(close) * (1.0 - test_fraction))
test  = close.iloc[split:]
ref   = close.iloc[split - 1:-1]
ref.index = test.index
y_pred = ma.iloc[split:]

strat = strategy_returns(test, y_pred, ref)
metrics = {
    'mae':  mae(test, y_pred),
    'directional_accuracy': directional_accuracy(test, y_pred, ref),
    'cumulative_return':    cumulative_return(strat),
    'annualized_sharpe':    annualized_sharpe(strat, periods_per_year(req.interval)),
}
```

Same loader, same splitter, same metric module, same test slice. When Article 4 says "ARIMA at +7.52 Sharpe beats this", it really is the same data, the same Sharpe denominator, the same 1,608 test bars.

---

## What the +4.33 Sharpe doesn't say

I want to set the boundary of what this number proves, because the obvious takeaway — "a moving-average strategy makes 25 % over two months at Sharpe 4.3, ship it" — is wrong, and Article 10 is going to spend two thousand words explaining why.

In short: the test window covers Sep–Nov 2024, when BTC moved from roughly $58 k to roughly $96 k. Any long-flat strategy that *occasionally* says "long" — even at random — captures part of that rally. The Sharpe number is real but most of what it's measuring is the regime, not the model. Article 10 will apply the same lens retroactively to LSTM (+4.95), TFT (+4.59), and the foundation model lineup, and only ARIMA's +7.28 will survive as something that's plausibly doing more than ride the drift.

For now, the `+4.33` is in the leaderboard as the *bar to clear*. A model that produces a Sharpe close to it without doing materially better on `directional_accuracy` has not earned anything that this baseline didn't already earn for free.

---

## The leaderboard, after Article 3

We now have two floors:

| Floor | What it bounds | The number |
|---|---|---|
| Naive (Article 1) | MAE — point-forecast accuracy | $260.50 |
| MA(24) (this article) | dir_acc / Sharpe — trading-relevant signal | 0.5143 / +4.33 |

A model from here on has to clear *both*. Beating naive on MAE without beating MA(24) on directional accuracy is a denoiser, not a forecaster. Beating MA(24) on Sharpe without beating naive's MAE by much is fine — it just means the model traded better, not that it tracked the price better. Both columns are kept on the report card; only one of them gets to rank the leaderboard.

The next post in the series ([Article 4 — *ARIMA(1,1,0) beat my XGBoost*](../../articles_todo.md)) is where the floors get cleared for the first time. A single autoregressive coefficient on differenced returns, fit by `statsmodels` in less code than the title of this paragraph, lands at dir_acc 0.5373 / Sharpe +7.52. A 31-feature XGBoost on the same slice lands at 0.4872 / +1.45 — below MA(24) on both columns. *Feature engineering didn't help* is the sentence that paragraph is structured to land. This article is the one that built the bar it has to clear.

---

*Code: [experiments/01b_moving_average/](../../experiments/01b_moving_average/) · Repo: [transformer_bitcoin_ai](../../../README.md)*
