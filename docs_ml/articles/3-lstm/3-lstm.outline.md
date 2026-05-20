# Outline — Article 3: LSTM

Working title: *"What an LSTM learns from a window of Bitcoin history."*

Source experiment: [`04_lstm`](experiments/04_lstm/).

## Editorial framing

First **deep-learning** article. Recurrent network on a sliding window of past bars. The contract reverses from article 2 (XGBoost): the model is now richer (stacked LSTM + dropout + linear head) and the input is now *raw* (sequence of past bars + a few past covariates) rather than a flat feature vector. The point of this article is what the LSTM does and does not learn from that handoff on this slice.

Note on the numbers used in this article: the per-experiment report file lists slightly different numbers than the canonical `metrics.json` file — I cite the canonical `metrics.json` (366 test bars) since that is the source of truth for the leaderboard. The shape of the lesson is the same.

## Beats

1. **Hook.** First deep-learning model in the series. Stacked LSTM with dropout, sliding 48-bar window (8 days at 4h), past covariates (volume, OHLC range, OHLC body). MAE 539.23 — essentially identical to ARIMA. dir_acc **0.5301** — *higher* than ARIMA's 0.5082 (the LSTM is the first model that beats ARIMA on direction). Sharpe **4.82** — *lower* than ARIMA's 6.86. First model in the series where extra capacity helps on direction but hurts on the trading metric. Interesting and worth dwelling on.

2. **Methodology recap.** Standard 1-paragraph block. Same target, same 366-bar test slice. New thing this article: three-way time split, val slice used for early stopping (`EarlyStopping(monitor='val_loss')`), and a `BlockRNNModel` from Darts.

3. **What the model is — LSTM in plain English.**
   - Recurrent neural network: a hidden state that gets updated as the network reads each bar in the window.
   - LSTM = long short-term memory. Gated cells that decide what to remember, what to forget, and what to output at each step. The architecture solves the vanishing-gradient problem that plain RNNs have on long sequences. Brief mention only; not an LSTM tutorial.
   - In code: `darts.models.BlockRNNModel(model='LSTM', input_chunk_length=48, hidden_dim=32, n_rnn_layers=2, dropout=0.1, ...)`. Stacked two layers deep, hidden state of 32, dropout between layers. Linear head on top regresses to a single log-return.

4. **The input handoff.** The model is given a **window of 48 past bars** (each containing the log-return target and the three past covariates: `log_volume`, `hl_range`, `oc_body`) and asked to predict the next bar's log-return. Past covariates feed into the encoder; output is one number. No future covariates — `BlockRNN` is past-only. Mention this gap; article 4 fills it with the Temporal Fusion Transformer.

5. **The training shape.**
   - Train slice: 3 467 rows (after the dataset trims for the window length and val split).
   - Validation slice: 366 rows (the 2024-08-01 → 2024-10-01 window). Used by Lightning's `EarlyStopping` callback on `val_loss` with patience 5.
   - n_epochs 30, batch size 64, Adam optimizer, `learning_rate` 0.001.
   - `random_state` 42 — single seed for the canonical row. Acknowledge that neural net training has run-to-run variance from MPS / CUDA non-determinism beyond the `random_state` seed.

6. **The walk-forward shape.** `model.historical_forecasts(start=test_start, forecast_horizon=1, retrain=False, last_points_only=True)`. Same shape as ARIMA's walk-forward in article 1, same shape as the Temporal Fusion Transformer in article 4 — the harness is shared. The model slides a 48-bar window across the test slice with its weights frozen at train-time values; at each test bar it predicts one step ahead.

7. **The result — verbatim from `metrics.json`.**
   - MAE 539.23 USD
   - RMSE 808.29 USD
   - MAPE 0.6909 %
   - dir_acc 0.5301
   - cum_ret 0.4284
   - Sharpe 4.8221

8. **The leaderboard, as of article 3.**
   | Model | MAE | dir_acc | Sharpe |
   |---|---|---|---|
   | Naive | 540.96 | NaN | — |
   | ARIMA(3, 1, 3) | **539.15** | 0.5082 | **6.8559** |
   | XGBoost | 550.85 | 0.5082 | 6.1388 |
   | **LSTM** | 539.23 | **0.5301** | 4.8221 |

9. **The dir_acc-vs-Sharpe split — first time it shows up in the series.** LSTM is more often right about direction than ARIMA (0.5301 vs. 0.5082) — that is two more correct calls in 366 bars. *And* the LSTM's Sharpe is lower than ARIMA's by 2.03. How does that happen? Same mechanism as article 2 in reverse: the LSTM gets direction right slightly more often but sizes its predicted up-moves smaller; the long-flat strategy stays long less often during the rally and collects fewer up-bars per correct call. dir_acc and magnitude-aware Sharpe are not the same objective. This is a load-bearing observation for the rest of the series.

10. **The sweep — six (input_chunk_length, hidden_dim, n_rnn_layers, dropout) configs.** Note that the per-experiment report's sweep table cites values from a previous slice. The structural observation that matters: the *smallest* sweep config `(icl=24, hidden=16, n_rnn_layers=1, dropout=0.0)` was the Sharpe leader; the *largest* `(96, 64, 3, 0.2)` was the MAE / dir_acc / cum_ret leader. The configured default `(48, 32, 2, 0.1)` was Sharpe winner on *neither*. Same lesson as articles 1 and 2: defaults are starting points.

11. **The NaN-Sharpe sweep row.** One sweep row — `(48, 64, 2, 0.2)` — produces Sharpe `NaN`. The strategy went entirely flat (`pred ≤ ref` on every test bar) → `cum_ret = 0`, `stddev = 0` → Sharpe degenerates per [`src/btc_ai/eval/metrics.py:51-59`](src/btc_ai/eval/metrics.py). This is a useful failure mode to recognize on a leaderboard: "model that never goes long" reads identical to a missing run. Recall the analogous NaN in article 1 from the random-walk ARIMA.

12. **What this article tells us about the model class.**
   - **Recurrent capacity helps on direction but hurts on magnitude calibration at this signal-to-noise ratio.** The LSTM's hidden-state mechanism picks up a directional pattern that linear AR / MA on differenced prices misses, but it also damps down its predicted log-return magnitudes through the regression head's MSE loss, which is what costs Sharpe.
   - **3 467 training rows are *not enough* for ~10 000 LSTM weights to amortize.** This is a small-data deep-learning model on noisy data. The fact that the smallest sweep config wins Sharpe is the same observation: less recurrent capacity is more honest on this signal.
   - **The validation slice's early-stopping is honest, but it pins the model to val_loss — which doesn't pick the Sharpe winner.** Same lesson as articles 1 and 2: in-sample selection criterion ≠ out-of-sample trading-metric leader.

13. **Per-bar Sharpe significance.** 4.8221 / √2190 = 0.1031. SE 1 / √366 = 0.0523. Ratio ≈ **1.97 σ** — borderline, the lowest in the leaderboard ex-transformer.

14. **Regime caveat** — same paragraph as articles 1 / 2.

15. **Reproduce.** `make 04_lstm` + `make 04_lstm_sweep`.

## What this article is NOT

- An LSTM tutorial. Refer the curious reader to Goodfellow / Bengio / Courville or the Darts documentation.
- A "deep learning doesn't work on Bitcoin" article. The lesson is more specific: 3 467 rows at 4h with this much noise don't let an LSTM amortize its recurrent capacity past a 3-parameter ARIMA on Sharpe.
