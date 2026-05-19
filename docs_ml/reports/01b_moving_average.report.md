# Report — `01b_moving_average`

**Run date:** 2026-05-19
**Status:** completed (single fit, no sweep)

## What this experiment is

Sibling baseline to [01_baseline_naive](01_baseline_naive.report.md). Predicts the next price as the mean of the previous `window` closes; with `window=1` it degenerates to naive last-value. The `b` suffix marks this as a companion to 01, not a new generation — same data slice, same metrics, just a different (still trivially simple) predictor. Two purposes: a second floor for ARIMA et al. to clear, and the first predictor in the ladder that expresses a *direction* — MA mechanically bets against the recent trend, so `directional_accuracy` becomes meaningful (unlike for naive, where it is NaN by design).

```
close_pred[t] = mean(close[t-window : t])
```

No training, no parameters fit, no walk-forward in the model-fitting sense — the rolling mean is computed once via `close.shift(1).rolling(window).mean()` and read off on the test slice. Library: pandas. See [experiments/01b_moving_average/README.md](../../experiments/01b_moving_average/README.md) for the full narrative.

## Configuration

From [experiments/01b_moving_average/configs/btc_4h_2024.yaml](../../experiments/01b_moving_average/configs/btc_4h_2024.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `4h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| window | `24` bars (= 4 days at 4h) |

Total bars: **2 010**. Test: **402**. (No train slice in the fitting sense.)

## Results — single fit

From [experiments/01b_moving_average/results/btc_4h_2024/metrics.json](../../experiments/01b_moving_average/results/btc_4h_2024/metrics.json):

| Metric | Value |
|---|---|
| MAE | 1 905.27 USD |
| RMSE | 2 580.61 USD |
| MAPE | 2.4782 % |
| Directional accuracy | 0.4801 |
| Cumulative return | 0.0511 |
| Annualized Sharpe | 1.3003 |

## Cross-experiment comparison

4h-slice leaderboard. Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| [01_baseline_naive](01_baseline_naive.report.md) | 518.36 | 784.09 | 0.6701 % | NaN | — | — |
| **01b_moving_average (window=24)** | 1 905.27 | 2 580.61 | 2.4782 % | 0.4801 | 0.0511 | 1.3003 |
| [02_arima (1, 1, 1)](02_arima.report.md) | **517.16** | **782.42** | **0.6686 %** | **0.5547** | 0.3314 | 6.0569 |
| [03_gradient_boosting (n_feat=31)](03_gradient_boosting.report.md) | 539.39 | 795.07 | 0.7008 % | 0.5365 | **0.5042** | **6.4233** |
| [04_lstm](04_lstm.report.md) | 522.29 | 785.45 | 0.6761 % | 0.4938 | 0.2596 | 4.2660 |

01b is the *anti-baseline* — the row everything else must clear. MAE 1 905.27 is **3.68× worse than naive** (518.36) and **3.69× worse than ARIMA(1,1,1)**. The 24-bar lag is enormous on a rallying test slice: when prices trend up, the trailing mean stays below the most recent close, so the predicted price is consistently below the true price, and MAE blows up linearly in the trend. The strategy column saves the row from total embarrassment — long/flat occasionally catches the trend — but a 1.30 annualized Sharpe with `dir_acc < 0.5` is statistical noise at this sample size (see Bottom line). (Experiments 05 and 06 currently report on a stale 1h slice — see [05_transformer.report.md](05_transformer.report.md) and [06_pretrained.report.md](06_pretrained.report.md).)

## Interpretation

1. **MA(24) trails the trend by design.** On an uptrending slice, the trailing mean is mechanically below `close[t-1]`, so the predicted "next close" is *below* the reference. The long/flat rule (`pred > ref → long`) goes flat. Most of the test slice ends up *not* long during a rally — hence the modest 5.11 % cumulative return on a window where simply buy-and-hold would have made many multiples of that.
2. **dir_acc 0.4801** — below coin-flip, but only barely. Reading: MA is correctly betting against the trend in the rare down-bars and is wrong on most of the up-bars. The rolling mean is a *mean-reverter*; this slice is *trending*; the bet loses on most bars.
3. **Sharpe 1.3003 is within noise.** Per-bar Sharpe ≈ 1.3003 / √2190 = 0.0278. Standard error ≈ 1 / √402 = 0.0499. Ratio ≈ 0.56 σ — well inside the noise band. Treat as "no edge demonstrated."
4. **No formal sanity check against window=1.** A `window=1` MA collapses to naive (MAE 518.36); 01b at `window=24` gives 1 905.27. The 24-bar MA is the *anti-baseline*: a model that scores worse than this is doing something pathological. ARIMA, XGBoost, LSTM all clear it by a factor of 3.5+ — which is the bare minimum expected.

### Bottom line

MA(24) lags trends mechanically and **breaks naive's MAE floor by a factor of 3.68×** on this rally-heavy 4h slice. Its Sharpe (1.30) is within one standard error of zero (per-bar 0.0278 vs. SE 0.0499; ratio ≈ 0.56 σ). The test window covers approximately Sep 25 → Dec 1 2024 (post-election BTC rally), exactly the regime in which a mean-reverter is most punished. This row is useful only as the *anti-baseline*; it is the floor every parametric model must clear by more than 3×.

## Caveats

- Single split, no rolling-origin CV.
- No parameter fit — `window` is set once in `config.yaml`. A `window=1` sanity check is not part of this run but would degenerate to naive.
- Performance is highly regime-sensitive: on a sideways or oscillating slice MAE would shrink dramatically and dir_acc might climb above 0.5. This regime is *not* sideways.
- Strategy is long/flat, no shorting, no transaction costs.

## Files produced

- [experiments/01b_moving_average/results/btc_4h_2024/metrics.json](../../experiments/01b_moving_average/results/btc_4h_2024/metrics.json)
- [experiments/01b_moving_average/results/btc_4h_2024/predictions.parquet](../../experiments/01b_moving_average/results/btc_4h_2024/predictions.parquet)
- [experiments/01b_moving_average/results/btc_4h_2024/plot.png](../../experiments/01b_moving_average/results/btc_4h_2024/plot.png)

## How to reproduce

```
make 01b_moving_average
```
