# Outline — The mean-reverter that fixed my metrics

## One-line pitch
A 24-bar moving average is **3× worse than naive on MAE** but unlocks `directional_accuracy = 0.5143`, `Sharpe = +4.33`, and `cumulative_return = +25.7 %`. Use it to introduce dir_acc, cumulative return, Sharpe, and why **a bad baseline is structurally necessary**.

## Audience
Same as Article 1: ML practitioners and quant-curious engineers. Bonus utility for anyone who has ever benchmarked a model and felt mildly suspicious of their own metric choice.

## Thesis
The moving-average baseline is *worse* than naive on MAE — and that's the entire point. It's the first model in the lineup that *expresses an opinion about direction*. That makes it the first row in the leaderboard with a non-NaN dir_acc, and that means every later row finally has something to compare against on the metric that actually matters for trading.

## Structure

### 1. The hook — "I built a worse baseline on purpose"
- The pitch: a moving-average forecaster is structurally worse than naive at predicting *level*. Yet here it is, in the lab.
- The numbers (from `experiments/01b_moving_average/results/metrics.json`):
  - `MAE = $756.19` (vs naive's $260.50 — ~2.9× worse)
  - `MAPE = 0.99 %` (vs naive's 0.34 %)
  - `directional_accuracy = 0.5143` (vs naive's NaN)
  - `cumulative_return = +25.7 %`
  - `annualized_sharpe = +4.33`
- The provocation: by every "predictor accuracy" metric we usually quote, this thing is awful. By the metrics that decide whether you'd run it as a trading rule, it is *the first model in this series that's allowed to play*.

### 2. Why the moving average is mechanically worse on MAE
- Naive's per-bar error is `close[t] - close[t-1]` — the bar's actual move. Mean magnitude ~$260.
- MA(24)'s per-bar error is `close[t] - mean(close[t-24:t])`. On a trending market that's a *much bigger* number, because the rolling mean lags the trend.
- Demonstrate why: when BTC rallies from $58k to $96k between Sep and Nov 2024, a 24-bar mean is structurally pulled toward the older, lower prices. Lag = error.
- So MAE_MA ≈ trend-magnitude over the test window, not noise magnitude. *This is what MAE measures here, just like in Article 1*.

### 3. Why the moving average suddenly has a directional opinion
- Compare predicted moves:
  - Naive: `pred[t] - ref[t] = close[t-1] - close[t-1] = 0`. No opinion. NaN.
  - MA(24): `pred[t] - ref[t] = mean(close[t-24:t]) - close[t-1]`. Negative if the last bar is *above* the rolling mean (predict mean-revert down), positive if below (predict mean-revert up).
- This is a **mean-reversion signal**, baked in by construction. The model is saying: "the price has gotten ahead of its trailing average; it'll snap back."
- That signal can be wrong (and on a trending series, it's wrong a lot — note that 0.5143 is barely above coin-flip), but it is *expressed*. It can be measured. It can be beaten.

### 4. The metrics that this baseline introduces
For each, give the formula, the intuition, and the value MA(24) clocked. All from [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py).

- **`directional_accuracy`** — fraction of bars where predicted move sign matches realized move sign. MA(24): 0.5143. Above coin-flip but barely; this is the bar everything else in the series will have to clear materially.
- **`strategy_returns`** — long-flat rule: long for the next bar when `pred > ref`, else flat. No costs. Mean reversion: long whenever the price is below its rolling average.
- **`cumulative_return`** — product of `(1 + r_t)` over the test window. MA(24): +25.7 % across ~1,608 hourly bars. Sounds great. We'll deflate it in Article 10.
- **`annualized_sharpe`** — `sqrt(8760) * mean / stddev` of the per-bar strategy returns. MA(24): +4.33. Same caveat — Article 14 will be sceptical of how 8760 makes the number look like a hedge fund.

Note: `mae` is still in the report for diagnostic value, but the leaderboard column is `directional_accuracy + sharpe`. This article is the moment we cement that.

### 5. The 24-bar window is opinionated
- `window: 24` ≈ "yesterday's average". On 1h data, 24 bars = 1 trading day.
- A window of 1 degenerates to naive (the YAML comment makes this explicit).
- Bigger windows (48, 96, 168 = 1 week) trade lag for smoothing; smaller windows trade smoothing for noise. We don't sweep here — this is a baseline, not a tuned strategy.
- The honest framing: 24 was picked because it's the most defensible default on 1h data. Any window-tuning on the *test set* is leaderboard fraud.

### 6. Why "a bad baseline is structurally necessary"
This is the punchline. The leaderboard column we care about is dir_acc / Sharpe. If the *only* baseline is naive (MAE $260, dir_acc NaN), there is nothing for a future model to be compared against on dir_acc / Sharpe. The first non-NaN dir_acc value in the lineup *creates* the leaderboard column.

The MA(24) is the lab's null hypothesis for direction:
- "A model that has no real signal but is willing to take a side."
- Future articles benchmark against this. Article 4's XGBoost lands at dir_acc 0.4872 — *worse than MA(24)*. That comparison is only possible because MA(24) exists.

### 7. Reproducing it
- `make 01b_moving_average`
- The whole forecasting bit is one line:
  ```python
  ma = close.shift(1).rolling(window).mean()
  ```
- Same data slice as 01, same metrics module, comparable Sharpe.

### 8. What the +4.33 Sharpe doesn't say
A short, careful paragraph because Article 10 will eviscerate this number. MA(24)'s Sharpe is real but it's mostly capturing the test window's rally as BTC moved from ~$58k to ~$96k. A long-flat strategy that occasionally says "long" — even on a near-coin-flip directional signal — captures part of that rally. The Sharpe is not the model's skill, it's BTC's drift. Set up Article 10's lens.

### 9. Closing — the leaderboard now has a soul
- Naive: floor on MAE (Article 1).
- MA(24): floor on direction (Article 3).
- Every later experiment in the series gets compared against both.
- Tease: Article 4 — ARIMA(1,1,0) at dir_acc 0.5373 / Sharpe +7.52 versus XGBoost at 0.4872 / +1.45. *One AR coefficient versus thirty-one features.* This baseline is the bar both have to clear.

## Key code/file references
- [experiments/01b_moving_average/run.py](../../experiments/01b_moving_average/run.py) — ~100 lines
- [experiments/01b_moving_average/config.yaml](../../experiments/01b_moving_average/config.yaml) — `window: 24`
- [experiments/01b_moving_average/results/metrics.json](../../experiments/01b_moving_average/results/metrics.json) — the headline numbers
- [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py) — `strategy_returns`, `cumulative_return`, `annualized_sharpe`

## Tone notes
- Honest discomfort: a "worse" model is in the leaderboard on purpose.
- Set up the dir_acc / Sharpe vocabulary that the rest of the series leans on.
- Tease but don't over-claim — Article 10 is the corrective.

## Length target
~1,400–1,700 words.
