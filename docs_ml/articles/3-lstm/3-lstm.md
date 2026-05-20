# What an LSTM learns from a window of Bitcoin history

This article is about the first deep-learning model in the series — a stacked long-short-term-memory recurrent network — and the first model in the series whose extra architectural capacity helps on one metric and *hurts* on another. The LSTM is more often right about direction than the ARIMA(3, 1, 3) from article 1 (directional accuracy 0.5301 vs. 0.5082 on the same 366 test bars). And its annualized Sharpe is *lower* than ARIMA's by 2.03 (4.82 vs. 6.86). That split is the interesting beat of this article: how a model can get direction right more often *and* compound it worse.

Mean absolute error lands at 539.23 USD — within $0.08 of ARIMA. So on point error the LSTM is tied with the linear baseline; on direction it is meaningfully ahead; on the trading metric it is meaningfully behind. Three answers, three different stories. The article walks through why.

## Methodology, in one paragraph

Target: one-step-ahead log-return `r_T = log(close_T / close_{T-1})`, reconstructed back to price as `close_pred = close_{T-1} * exp(r_pred)`. Data: `BTCUSDT` 4-hour bars from Binance Vision. Split: train `2023-01-01` → `2024-08-01` UTC (3 467 rows after the 48-bar input window and validation trim), validation `2024-08-01` → `2024-10-01` UTC (366 rows, used for early stopping), test `2024-10-01` → `2024-12-01` UTC (**366 bars**). Walk-forward, one step ahead, weights frozen across the test window. Metrics from the shared module — [`src/btc_ai/eval/metrics.py`](src/btc_ai/eval/metrics.py). Reproduce: `make 04_lstm`. **New this article:** the validation slice is used (Lightning's `EarlyStopping(monitor='val_loss')`, patience 5), and the model is trained with PyTorch / PyTorch Lightning rather than statsmodels or XGBoost.

## What the model is

The model is a stacked LSTM with a linear regression head:

```yaml
# experiments/04_lstm/configs/btc_4h_2024.config.yaml — model section
model:
  input_chunk_length: 48     # 8 days at 4h
  output_chunk_length: 1
  hidden_dim: 32
  n_rnn_layers: 2
  dropout: 0.1
  batch_size: 64
  n_epochs: 30                # early-stopped on val_loss
  learning_rate: 0.001        # Adam
  random_state: 42
```

The class is `darts.models.BlockRNNModel(model='LSTM', ...)`. At each time step in the input window, the LSTM cell takes the current bar's feature vector and the previous step's hidden state, runs them through input / forget / output gates, and emits a new hidden state. After the full 48-step window is consumed, the final hidden state is passed through a linear head and produces a single number — the predicted log-return for the next bar. Two layers means the second LSTM cell consumes the output of the first; dropout 0.1 is applied between layers as regularization.

About 10 000 trainable weights, all told. By neural-net standards this is small; by Bitcoin-4h-data standards (3 467 training rows) it is still a lot of capacity to amortize. That mismatch is going to show up in the sweep.

## The input handoff

This is the first model in the series that does not see a hand-engineered feature vector. The LSTM is given a window of 48 past bars and each bar contributes:

- the log-return target itself, as past data — so the LSTM has access to the same 48 lagged returns that XGBoost saw in article 2;
- `log_volume`, `hl_range`, `oc_body` — the three past covariates from article 2's feature set, fed in unchanged.

What the LSTM gets to do that XGBoost did not: see the temporal *order* of those values, and carry a recurrent state across the 48 steps. What XGBoost did that the LSTM does not: see rolling moments (`r_mean_6`, `r_std_6`, etc.) precomputed for it. In principle the LSTM can learn rolling means and standard deviations from the raw lags inside its hidden state. In practice, with 3 467 training rows, it does not get many opportunities to learn them well.

No future covariates here. `BlockRNNModel` is past-covariate-only — the model gets to look back, not forward. Article 4 lifts that restriction with the Temporal Fusion Transformer (calendar-cyclical features as future covariates).

## The walk-forward shape

```python
preds = model.historical_forecasts(
    series=close_series,
    past_covariates=past_cov,
    start=test_start,
    forecast_horizon=1,
    retrain=False,
    last_points_only=True,
)
```

`retrain=False` is the load-bearing flag. The LSTM is trained once (on `2023-01-01` → `2024-08-01` UTC, with early stopping on the validation slice) and then walks forward through the test slice with its weights frozen. At each test bar the model takes the most recent 48 bars (including ones in the test slice it has already "seen" via the sliding window), runs them through the frozen network, and emits one log-return. The model sees new observations as they arrive but does not update parameters from them — same protocol as article 1's ARIMA, same protocol as article 4's Temporal Fusion Transformer. The shared `historical_forecasts` machinery means the harness across articles 3 and 4 is bit-identical except for the model class.

## The numbers

From [`experiments/04_lstm/results/btc_4h_2024/metrics.json`](experiments/04_lstm/results/btc_4h_2024/metrics.json):

| Metric | LSTM |
|---|---|
| MAE | 539.23 USD |
| RMSE | 808.29 USD |
| MAPE | 0.6909 % |
| Directional accuracy | **0.5301** |
| Cumulative return | 0.4284 |
| Annualized Sharpe | 4.8221 |

Side-by-side with everything in the leaderboard so far, on the same 366-bar test slice:

| Model | MAE | dir_acc | Sharpe |
|---|---|---|---|
| Naive last-value | 540.96 | NaN | — |
| ARIMA(3, 1, 3) | **539.15** | 0.5082 | **6.8559** |
| XGBoost (31 features) | 550.85 | 0.5082 | 6.1388 |
| **LSTM** | 539.23 | **0.5301** | 4.8221 |

The LSTM is the first model in the leaderboard to clear ARIMA on directional accuracy. Sixty-five bars out of 366 right vs. ARIMA's 59 right out of 366 above the no-direction line — a gain of two-and-a-bit percentage points. *And* it is also the first trained model in the leaderboard whose Sharpe is meaningfully below the ARIMA's. The same split shape as article 2's XGBoost, in the other direction: there, XGBoost tied direction and lost Sharpe; here, the LSTM beats direction and loses Sharpe by more.

## The direction-versus-Sharpe split

This is where the article spends its time, because it is the load-bearing methodology beat of the series. The mechanism:

The long-flat strategy rule is "go long for the next bar when `pred > ref`, else flat." Two models with different MAE / dir_acc / Sharpe profiles can be analyzed by the *fraction of bars they spend long* and the *average up-bar return collected per long bar*.

- ARIMA(3, 1, 3) predicts above the reference price on enough bars to spend a large fraction of the test slice long. Its predicted positive log-returns are calibrated to the data — when it is right, the magnitude of the predicted up-move roughly matches the actual up-move. The Sharpe is the bars-long fraction times the average actual up-bar return, normalized by the strategy's return standard deviation.
- The LSTM gets direction right slightly more often (0.5301), but the regression head's MSE loss against log-returns trains the model to *underpredict magnitudes* — log-returns are heavy-tailed and zero-mean, and MSE pulls the prediction toward zero. So the LSTM's `pred > ref` triggers on fewer bars (the model is more often correctly directional but in a smaller, less confident way), and the strategy spends less of the test slice long. It collects more correct calls per call but fewer up-bar returns in total. Sharpe goes down.

So the LSTM's hidden state is finding some structure that ARIMA's linear AR / MA isn't — there is *something* in the sequence of 48 past bars that improves the direction call. But the MSE-against-log-returns objective the network was trained on does not translate that direction skill into well-sized predicted moves. The trading metric is magnitude-aware; direction skill alone doesn't compound.

This split is *not* a defect in the LSTM. It is what happens when you train a sequence model with an MSE loss on a heavy-tailed zero-mean target. There are remedies — quantile loss, asymmetric loss, the Temporal Fusion Transformer's per-quantile head from article 4 — but at this slice, with these covariates and these 3 467 training rows, the default MSE objective is what the model has, and the price is paid in Sharpe.

## The sweep

The repo includes a six-row sweep over `(input_chunk_length, hidden_dim, n_rnn_layers, dropout)` configurations: [`experiments/04_lstm/results/btc_4h_2024/sweep.csv`](experiments/04_lstm/results/btc_4h_2024/sweep.csv). Two extremes worth highlighting:

- The *smallest* sweep configuration, `(input_chunk_length=24, hidden_dim=16, n_rnn_layers=1, dropout=0.0)` — basically one shallow LSTM layer on a 24-bar (4-day) window — was the Sharpe winner of the sweep, comfortably ahead of the configured default.
- The *largest* sweep configuration, `(96, 64, 3, 0.2)` — three layers, hidden state of 64, on a 96-bar (16-day) window — was the MAE / dir_acc / cumulative-return winner.

The configured default `(48, 32, 2, 0.1)` was the sweep winner of *neither* MAE nor Sharpe. The two extremes point in opposite directions on capacity: smaller wins Sharpe; larger wins direction and point error. The default splits the difference and is on neither extreme.

The load-bearing observation: *less recurrent capacity is more honest on this signal-to-noise ratio*. A single-layer LSTM with a hidden state of 16, looking at 24 bars, lands a better Sharpe than two layers of 32 looking at 48 bars. The deep model is more confident in the wrong direction on the bars where the small model would have stayed flat. Recurrent capacity here is buying overconfidence rather than skill.

## The NaN-Sharpe row

One row of the sweep — `(input_chunk_length=48, hidden_dim=64, n_rnn_layers=2, dropout=0.2)` — reports `Sharpe = NaN`. Not zero. Not negative. NaN. The mechanism, from [`src/btc_ai/eval/metrics.py:51-59`](src/btc_ai/eval/metrics.py):

```python
def annualized_sharpe(strat_ret: pd.Series, periods_per_year: int) -> float:
        r = strat_ret.to_numpy()
        sd = r.std(ddof=1)
        if not np.isfinite(sd) or sd == 0.0:
                return float('nan')
        return float(np.sqrt(periods_per_year) * r.mean() / sd)
```

The strategy went entirely flat: `pred ≤ ref` on every test bar, position = 0 everywhere, `strat_ret` was a vector of zeros, the standard deviation collapsed to zero, and Sharpe degenerated. This is the third NaN-by-design surface the series has seen — first the naive baseline's `dir_acc`, then the ARIMA sweep's `(0, 1, 0)` random-walk row, now this. *"Model that never expresses a long position"* reads identical to a missing run on the leaderboard at a glance. Recognize the shape — a model that produces a finite `dir_acc` but a NaN Sharpe is a model that was correctly directional somewhere but never *strongly enough* to trigger the long-flat rule.

## What this article tells us about the model class

**Recurrent capacity helps on direction but hurts on magnitude calibration at this signal-to-noise ratio.** The hidden state is picking up something the linear AR / MA structure misses. The MSE-against-log-returns objective is sanding the predicted magnitudes down toward zero. The trading metric is magnitude-aware; direction is necessary but not sufficient. This is the first model in the series that makes the dir_acc ≠ Sharpe split unambiguous.

**3 467 training rows are not enough for the configured default to amortize.** The sweep's Sharpe winner is the smallest configuration. The configured default has too much capacity for what the data slice supports. This recurs throughout deep learning on small datasets — the right answer is often to *shrink* the model, not grow it — and it will recur in article 4 on a bigger and more expressive architecture.

**Early stopping on `val_loss` is honest but does not pick the Sharpe leader.** The validation slice — `2024-08-01` → `2024-10-01` UTC — is a different two-month chunk than the test slice and behaves differently regime-wise. A model whose `val_loss` is lowest is the model whose log-return MSE is lowest on those two months. That does not have to be the model whose long-flat Sharpe is highest on the next two months. Same lesson as article 1's ARIMA AIC discussion: in-sample selection criterion is one column of six in the metrics.json; the column you care about is one of them.

## Per-bar Sharpe sanity check

4.8221 / √2190 = 0.1031 per bar; standard error on the per-bar mean ≈ 1 / √366 = 0.0523; ratio ≈ **1.97 σ** — only weakly significant under the optimistic i.i.d. assumption, the lowest in the leaderboard so far ex the transformer from article 4. The Sharpe is positive; the evidence that it is positive on this slice is thin.

## Regime caveat

The test slice is `2024-10-01` → `2024-12-01` UTC — the post-election Bitcoin rally. A long-flat strategy with mild directional skill compounds favourably on a strongly trending up-regime. The 4.82 Sharpe figure is conditional on that. Nothing in this article (and nothing in the series so far) has been tested out-of-regime; read every positive Sharpe figure as *"skill conditional on this regime."* Article 4 will reach a Sharpe of 2.5 with a model whose MAE is *worse than naive by 65 %* — purely because the regime is so favourable that even a wrong-about-magnitude model trips into Sharpe-positive territory.

## Reproduce

```
make 04_lstm
make 04_lstm_sweep
```

The first writes [`experiments/04_lstm/results/btc_4h_2024/metrics.json`](experiments/04_lstm/results/btc_4h_2024/metrics.json) — the canonical LSTM numbers above. The second writes [`experiments/04_lstm/results/btc_4h_2024/sweep.csv`](experiments/04_lstm/results/btc_4h_2024/sweep.csv) — the six-row sweep summarized above.

Article 4 is the Temporal Fusion Transformer — the model the repo is named after. Attention, variable-selection networks, future covariates. It is also the model that loses the hardest in the entire series, and the lesson is *not* "transformers don't work on time series." Onwards.
