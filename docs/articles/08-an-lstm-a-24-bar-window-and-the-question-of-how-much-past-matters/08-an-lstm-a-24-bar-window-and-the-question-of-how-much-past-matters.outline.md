# Outline — An LSTM, a 24-bar window, and the question of how much past matters

## One-line pitch
Walkthrough of [04_lstm](../../experiments/04_lstm/) using Darts' `BlockRNNModel`: why log-returns as the target, why past-covariates only, and the sweep finding that `input_chunk_length=24` (one day) **fails** while 48–96h works. The interesting bit: longer context didn't beat ARIMA's one lag — discuss what that says about where the signal lives.

## Audience
Deep-learning practitioners and quant-curious engineers. Series readers who've watched the linear baselines beat XGBoost (Articles 3, 4) and want to see whether deep capacity changes the picture.

## Thesis
On hourly BTC log-returns, the LSTM lands at `dir_acc 0.5196 / Sharpe +4.95` ([04_lstm/results/metrics.json](../../experiments/04_lstm/results/metrics.json)) — competitive but well behind ARIMA's +7.28. The sweep tells a clean story: *short context (24h) actively hurts* because the LSTM only learns mean reversion in a noisy regime; *medium context (48–96h)* works; *longer context doesn't help*. The signal is concentrated in the last few bars and the LSTM is, in effect, a slightly more flexible version of the AR(1) ARIMA encodes — without the AR(1)'s parsimony.

## Structure

### 1. The hook — first deep model in the lineup
- "Now we run a real model."
- LSTM with 32-dim hidden state, 2 stacked layers, dropout, past covariates (volume, OHLC range/body), 48-bar input chunk. ~22k parameters.
- The committed result:
  ```json
  {
    "directional_accuracy": 0.5196,
    "cumulative_return":   0.5034,
    "annualized_sharpe":   4.945,
    "mae":  274.02
  }
  ```
- That's *behind* ARIMA(1,1,1) on every column except the third decimal of MAE. The deep model didn't beat the model from 1970.

### 2. The leaderboard, post-Article 7
| Model | dir_acc | Sharpe | cum_ret | MAE |
|---|---:|---:|---:|---:|
| ARIMA(1,1,1) | 0.5336 | +7.28 | +53.0 % | $260.50 |
| **LSTM** | **0.5196** | **+4.95** | **+50.3 %** | **$274.02** |
| MA(24) | 0.5143 | +4.33 | +25.7 % | $756.19 |
| XGBoost | 0.4872 | +1.45 | +8.1 % | $270.73 |
| Naive | NaN | — | — | $260.50 |

### 3. What the LSTM is being given
- **Target.** `r_T = log(close_T / close_{T-1})` — same as XGBoost's. Stationary.
- **Past covariates.** `log_volume`, `hl_range`, `oc_body` of bar `T-1` and earlier. Fed as parallel time series, not flattened scalars.
- **No future covariates.** BlockRNN doesn't accept them — that's TFT's job (Article 9).
- **Architecture.** 2 LSTM layers, hidden dim 32, dropout 0.1, 48-bar input chunk, single-step output, MSE loss. Trained with early stopping on a 10 % val tail.

### 4. The sweep — the actual interesting bit
From [04_lstm/results/sweep.csv](../../experiments/04_lstm/results/sweep.csv):

| icl | hidden | layers | dropout | dir_acc | Sharpe |
|---:|---:|---:|---:|---:|---:|
| **24** | 16 | 1 | 0.0 | 0.4798 | **+0.54** |
| **24** | 32 | 2 | 0.1 | 0.4810 | **+2.15** |
| 48 | 32 | 2 | 0.1 | **0.5196** | **+4.95** |
| 48 | 64 | 2 | 0.2 | 0.5178 | +4.59 |
| 96 | 32 | 2 | 0.1 | 0.4916 | +3.06 |
| 96 | 64 | 3 | 0.2 | **0.5196** | **+4.95** |

Two clean observations:

**`icl=24` fails** — both 24-bar configs land at dir_acc < 0.50, *worse than coin-flip*. With only 24 bars (one trading day) of context, the LSTM has nothing but recent vol structure to chew on; it learns a mean-reversion-ish prior that's wrong on the rallying test slice. Sharpe is positive (+0.54, +2.15) only because of the long-during-rally floor (Article 10).

**`icl=48` is the sweet spot** — both 48-bar configs land at dir_acc ≈ 0.518–0.520 and Sharpe +4.59 / +4.95. Doubling the lookback from one day to two unlocks a usable signal.

**`icl=96` doesn't reliably help** — the larger 96-bar config (`hidden=64, layers=3`) ties the best 48-bar config. The smaller 96-bar config (`hidden=32, layers=2`) regresses to dir_acc 0.4916. More context isn't doing structural work; it's just averaging out the per-bar lag-1 signal.

### 5. What this says about where the signal lives
- ARIMA(1,1,1) wins on 1-bar of context. LSTM ties itself with 96 bars. The marginal value of bars 2–96 is approximately zero on this signal.
- That's exactly the picture you'd expect if the signal is a small AR(1) on differenced returns: bar `T-1` matters, bar `T-2` matters slightly less, bars `T-3 … T-96` are noise.
- The LSTM is, in effect, learning a fancier version of `r_T ≈ φ * r_{T-1} + noise` and incurring extra parameters for the privilege.
- The 48-bar context floors out at the right number because (a) BlockRNN needs a few bars of warmup to populate hidden state usefully, and (b) the past covariates (volume, OHLC) might add a little — but this is tested in the next subsection.

### 6. Ablation: does the covariate channel help?
- Re-run [04_lstm](../../experiments/04_lstm/) with `use_volume: false, use_ohlc: false` (target only). The committed sweep doesn't include this ablation explicitly; the article will narrate the expected outcome and link to a follow-up issue rather than fabricate numbers.
- The honest answer based on prior posts: probably very little. Article 4's XGBoost had volume and OHLC features and lost to ARIMA.
- The `use_volume / use_ohlc` toggles in [config.yaml](../../experiments/04_lstm/config.yaml) make this ablation a one-line edit.

### 7. Why the LSTM lost to ARIMA, mechanistically
- ARIMA's predicted move is `c + φ * r_{T-1}` — tiny, signed, calibrated.
- LSTM's predicted move is whatever the recurrent dynamics produce — bigger in magnitude (the model has capacity to "predict" a real number rather than just a tiny perturbation), and the strategy gate (`pred > ref`) is more easily flipped by noise in the recurrent layers.
- The MAE numbers tell the same story: ARIMA's $260.50 is essentially the volatility floor; LSTM's $274.02 is slightly above it, meaning LSTM's predictions are wrong by *more* than the per-bar volatility. That's a sign the model is producing predictions whose magnitude is overstated.
- Same picture as Article 4's XGBoost: capacity hurts when SNR is low.

### 8. Why the LSTM still beat MA(24) and XGBoost
- The two structural improvements over MA(24): the LSTM's predicted move can be small (closer to ARIMA's signed-but-tiny output) and signed in the right direction more often than mean-reversion suggests. dir_acc 0.5196 vs MA(24)'s 0.5143 is small but real.
- vs XGBoost: the LSTM models the *order* of past bars; XGBoost saw an unordered bag of 31 features. On a near-random-walk, "order" is a weak prior — but it's the right one.

### 9. The optuna run that didn't change the verdict
- [optuna_sweep.py](../../experiments/04_lstm/optuna_sweep.py) is wired in. The TPE sampler explores the same 4-D space (`icl`, `hidden_dim`, `n_rnn_layers`, `dropout`).
- A 30-trial Optuna search will pick a config near the best sweep row (dir_acc ≈ 0.520, Sharpe ≈ +4.95).
- The lesson: optuna can find the *best within the family*, but the family ceiling is below ARIMA. Hyperparameter search is not a path past the floor.

### 10. The walk-forward harness, in case you skipped Article 7
- `historical_forecasts(retrain=False, last_points_only=True, forecast_horizon=1)` over the test window.
- Same harness as TFT (Article 9) and the foundation models (Article 11). All deep-model leaderboard rows are computed identically.

### 11. Reproducing
```
make 04_lstm
make 04_lstm_sweep   # writes results/sweep.csv
make 04_lstm_optuna  # 30 TPE trials, writes optuna_best.json
make 04_lstm_optuna_dashboard  # interactive parallel coords / param importance
```
First run downloads no weights (this is a from-scratch model); single-digit minutes on Apple MPS.

### 12. Closing — the question for Article 9
- LSTM with capacity didn't beat ARIMA. Does *attention* with capacity beat ARIMA?
- Article 9's TFT result (Sharpe +4.59 / dir_acc 0.5053) is going to answer no. Worse than LSTM, in fact.
- The series is converging on a single thesis: *the floor is the data, not the architecture*.

## Key code/file references
- [experiments/04_lstm/run.py](../../experiments/04_lstm/run.py) — the harness
- [experiments/04_lstm/config.yaml](../../experiments/04_lstm/config.yaml) — the architecture knobs
- [experiments/04_lstm/results/sweep.csv](../../experiments/04_lstm/results/sweep.csv) — the icl story
- [experiments/04_lstm/results/metrics.json](../../experiments/04_lstm/results/metrics.json) — the headline +4.95 Sharpe

## Tone notes
- Honest engineering. Don't oversell deep capacity; don't dismiss it.
- The icl=24 finding is the article's most interesting empirical fact — give it room.

## Length target
~1,800–2,200 words.
