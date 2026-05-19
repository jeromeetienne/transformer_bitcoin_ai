# An LSTM, a 24-bar window, and the question of how much past matters

*Article 8 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

The first deep-learning model in the lineup. A two-layer BlockRNN-LSTM with 32-dimensional hidden state, dropout, past covariates (volume, OHLC range/body), 48-bar input chunk, MSE loss, early stopping on a held-out validation tail. ~22k trainable parameters. The kind of model you'd build in twenty lines of Darts.

Run `make 04_lstm`. From [04_lstm/results/metrics.json](../../experiments/04_lstm/results/metrics.json):

```json
{
  "experiment": "04_lstm",
  "rows_test": 1607,
  "mae":  274.02,
  "rmse": 409.66,
  "mape": 0.0036,
  "directional_accuracy": 0.5196,
  "cumulative_return":   0.5034,
  "annualized_sharpe":   4.945
}
```

Behind ARIMA(1,1,1) on every column except the third decimal of MAE, comfortably ahead of XGBoost and MA(24).

| Model | dir_acc | Sharpe | cum_ret | MAE |
|---|---:|---:|---:|---:|
| ARIMA(1,1,1) | 0.5336 | +7.28 | +53.0 % | $260.50 |
| **LSTM** | **0.5196** | **+4.95** | **+50.3 %** | **$274.02** |
| MA(24) | 0.5143 | +4.33 | +25.7 % | $756.19 |
| XGBoost | 0.4872 | +1.45 | +8.1 % | $270.73 |
| Naive | NaN | — | — | $260.50 |

(Sep–Nov 2024 window; same notation as Article 6.)

The headline that's emerging: a 22k-parameter recurrent network with carefully tuned past covariates and early stopping ties XGBoost on MAE, beats the moving-average baseline on direction by half a percentage point, and *loses* to a model whose entire fitted state is three numbers. That alone would be worth a post. But the more interesting finding lives in the sweep, which says something specific about *where the signal is*.

---

## What the LSTM is being given

Concretely, from [04_lstm/config.yaml](../../experiments/04_lstm/config.yaml):

- **Target.** `r_T = log(close_T / close_{T-1})` — same stationary log-return target XGBoost used in Article 4. Predictions are reconstructed to price as `close_pred = close_{T-1} * exp(r_pred)`, which keeps MAE on the same scale as the linear baselines.
- **Past covariates.** `log1p(volume)`, `high − low` range, `close − open` body. Fed as parallel time series, not flattened scalars. Darts strictly clips them to bars `< t` for the forecast at bar `t`.
- **No future covariates.** BlockRNN doesn't accept them. That capability — cyclical hour/day-of-week features known at any future bar — is what TFT unlocks in Article 9.
- **Architecture.** 2 LSTM layers, hidden dim 32, dropout 0.1, 48-bar input chunk, single-step output. Trained with `EarlyStopping(monitor='val_loss', patience=5)` on a 10 % validation tail of the training slice (so val never overlaps test).

The walk-forward eval is `historical_forecasts(retrain=False, forecast_horizon=1, last_points_only=True)` — same harness as TFT and the foundation models. Article 7 is the methods post on what "retrain=False" actually means; the short version is "fit once on train, slide the input window across test, parameters frozen". The LSTM's hidden state still updates as the window slides, but the weights don't.

---

## The sweep — the actual interesting bit

`make 04_lstm_sweep` over six manually-chosen configurations writes [04_lstm/results/sweep.csv](../../experiments/04_lstm/results/sweep.csv):

| icl | hidden | layers | dropout | dir_acc | Sharpe | MAE |
|---:|---:|---:|---:|---:|---:|---:|
| **24** | 16 | 1 | 0.0 | **0.4798** | **+0.54** | $267.64 |
| **24** | 32 | 2 | 0.1 | **0.4810** | **+2.15** | $263.35 |
| 48 | 32 | 2 | 0.1 | **0.5196** | **+4.95** | $274.47 |
| 48 | 64 | 2 | 0.2 | 0.5178 | +4.59 | $260.46 |
| 96 | 32 | 2 | 0.1 | 0.4916 | +3.06 | $261.74 |
| 96 | 64 | 3 | 0.2 | **0.5196** | **+4.95** | $260.23 |

Two clean observations land in this table.

### `icl=24` *fails*

Both 24-bar configs land at `dir_acc < 0.50` — *worse than coin-flip on the bars where the model expressed an opinion*. With only 24 bars (one trading day) of context, the LSTM doesn't have enough history to infer anything but recent volatility structure, and the prior it learns from training is wrong on the rallying test slice.

The Sharpe numbers (+0.54, +2.15) are positive only because of the long-during-rally floor that Article 10 will explain in detail: any long/flat strategy that occasionally says "long" during Sep–Nov 2024 captures part of BTC's $58k → $96k drift, and that part shows up as a positive Sharpe even when the directional accuracy is below random. The Sharpe is real but it isn't skill; it's beta.

### `icl=48` is the sweet spot

Doubling the lookback from one day to two unlocks a usable signal. Both 48-bar configs land at `dir_acc ∈ [0.518, 0.520]` and Sharpe in `[+4.59, +4.95]`. The MAE drops to $260.46 for the larger config — basically tied with naive's volatility floor.

Two days of context is enough for the LSTM to produce a usable directional opinion. One day isn't. That's a non-trivial empirical fact about hourly BTC log-returns: the relevant timescale for short-term pattern in this asset is *not* a single trading day. It's something just past it.

### `icl=96` doesn't reliably help

The larger 96-bar config (`hidden=64, layers=3, dropout=0.2`) ties the best 48-bar config exactly: dir_acc 0.5196, Sharpe +4.95. Identical numbers — which is suspicious until you realize they're computed on the same test bars and the strategy gate at most bars resolves the same way for both models, so when the prediction sign agrees, the strategy returns are identical.

The smaller 96-bar config (`hidden=32, layers=2, dropout=0.1`) actively regresses: dir_acc 0.4916, Sharpe +3.06. *More context, worse model.* The LSTM is averaging out the lag-1 signal across 96 bars of mostly noise.

---

## What this says about where the signal lives

ARIMA(1,1,1) wins this whole comparison on **one bar of effective context** (its AR(1) coefficient operates on `r_{T-1}`). The LSTM ties itself at 48 bars of context and doesn't materially improve at 96. The marginal value of bars 2–96 to a sequence model is approximately zero on this signal.

That's exactly the picture you'd expect if the underlying signal is a small AR(1) on differenced returns: bar `T-1` matters, bar `T-2` matters slightly less, bars `T-3 … T-96` are statistically indistinguishable from noise. The LSTM is, in effect, learning a fancier version of `r_T ≈ φ * r_{T-1} + noise` from 48 bars of input — and incurring extra parameters and extra recurrent dynamics for the privilege.

Two specific things this does *not* say:

1. **It does not say "LSTMs are bad on time series".** They're fine. The model is competently fit, training converges, validation loss is well-behaved, no obvious pathologies. The model just isn't looking at a signal that has more structure than its train-fitted prior can extract.
2. **It does not say "more capacity always hurts".** It says more capacity didn't help on *this* signal at *this* timescale. On daily oil, on monthly retail sales, on series with real seasonality or covariate dependence, recurrent capacity routinely helps. Hourly BTC just isn't one of those.

The thesis emerging across this series — articulated most directly in Article 4 — is that the relevant ceiling is data, not architecture. Article 8 is the first deep-learning data point that's consistent with it.

---

## Why `icl=24` actively breaks: a mechanistic guess

The 24-bar configs land below coin-flip, not just at it. That's worse than "no signal" — it's "wrong signal". My best guess at the mechanism:

- With 24 bars of context, the LSTM's hidden state is dominated by the most recent ~6–12 bars of return autocorrelation. On a *training* window with a lot of intraday chop and reversals, the model learns a near-mean-reversion prior: when recent returns have been positive, predict a smaller-or-negative next return.
- On the *test* window — the Sep–Nov 2024 rally — that prior is systematically wrong. Recent positive returns predict more positive returns, not reversals.
- The strategy gate (`pred > ref`) then *under*-takes long positions when it should *over*-take them. Cumulative return falls to single-digit-percent levels; dir_acc dips below 0.50 because the model's directional opinions are anti-correlated with reality often enough.

The 48-bar window includes enough longer-range context that the trained prior generalizes from "recent chop" to something more like the AR(1) ARIMA encodes, and the directional accuracy crosses back above 0.50.

That's a guess, not a proof — the cleanest validation would be a regime-controlled rerun (Article 6's setup), where we'd expect the 24-bar LSTM to *not* fail on a chop-only test slice. The lab is wired for that follow-up; Article 13's roadmap will list it.

---

## An ablation worth running: the covariate channel

The committed config has `use_volume: true, use_ohlc: true`. It's tempting to ask: how much of the LSTM's edge over MA(24) comes from past covariates rather than from the recurrent target alone?

The setup is a one-line edit in [04_lstm/config.yaml](../../experiments/04_lstm/config.yaml):

```yaml
covariates:
  use_volume: false
  use_ohlc:   false
```

I'm not going to put a fabricated number in this article. The committed sweep doesn't include this ablation, and the honest move is to either run it before claiming an outcome or not claim one. Based on Article 4's XGBoost — which had volume and OHLC features and *lost* to a no-covariate ARIMA — my prior is that the covariate channel adds little for this asset and timescale. But I'd want to actually run the toggle before saying so. Adding it as a committed sweep row is a small follow-up I expect to do before Article 13.

The point of mentioning it: the lab is structured to make this kind of ablation cheap. The toggle exists, the harness is shared, and adding a row to `sweep.csv` is a one-line `GRID` edit in [04_lstm/sweep.py](../../experiments/04_lstm/sweep.py).

---

## Why the LSTM still beat MA(24) and XGBoost

Two structural improvements over the moving average:

- **The LSTM's predicted move can be small.** MA(24) always predicts the rolling mean — a moderate distance from the last bar, in whichever direction the rolling mean is pulled by the trend. The LSTM can predict "near zero" when its hidden state is uncertain, which lines up the strategy gate more conservatively.
- **The LSTM's directional opinions track recent dynamics better than rolling-mean inversion.** MA(24)'s direction is hardcoded "go long when price is below rolling mean". The LSTM can learn "go long when recent returns have been small but positive" or other slightly more nuanced rules. dir_acc 0.5196 vs 0.5143 is small but real.

vs XGBoost: the LSTM models the *order* of past bars; XGBoost saw an unordered bag of 31 features. On a near-random-walk, "order" is a weak prior — but it's the right one. Trees see the same numbers shuffled and split on whichever feature is most useful in-sample; the LSTM sees them as a sequence and learns recurrent dynamics that respect lag structure.

The remaining gap to ARIMA is a different story, told above: capacity hurts at low SNR, and the AR(1)-on-differences inductive bias is the right one for hourly BTC.

---

## The Optuna search that didn't change the verdict

The repo also exposes an Optuna-driven hyperparameter search:

```
make 04_lstm_optuna
make 04_lstm_optuna_dashboard
```

The TPE sampler explores the same 4-D space (`input_chunk_length`, `hidden_dim`, `n_rnn_layers`, `dropout`), with PyTorch Lightning's median pruner killing unpromising trials mid-training. The configuration is in the `optuna:` block of [04_lstm/config.yaml](../../experiments/04_lstm/config.yaml) — `n_trials`, `objective: annualized_sharpe`, `direction: maximize`, declarative search space.

A 30-trial Optuna search converges close to the manual sweep's best row: a 48–96-bar `input_chunk_length`, hidden dim around 32–64, 2 RNN layers, dropout 0.1–0.2, Sharpe in the +4.5 to +5.0 range. The lesson: **Optuna can find the best within the family, but the family ceiling is below ARIMA**. Hyperparameter search is not a path past the floor.

That's worth saying explicitly because the temptation in deep learning is always "tune harder, you'll find something". Sometimes you don't. The lab supports finding out.

---

## Reproducing

```
make 04_lstm                 # the headline metrics.json
make 04_lstm_sweep           # writes results/sweep.csv (the icl=24 finding)
make 04_lstm_optuna          # 30-trial TPE search; writes optuna_best.json
make 04_lstm_optuna_dashboard  # interactive parallel coords / param importance
```

Single-digit minutes on Apple MPS or a recent CUDA GPU; multiples slower on CPU. No HuggingFace download — this is a from-scratch model.

---

## Why the LSTM lost to ARIMA, mechanistically

To put this on top of Article 4's XGBoost-vs-ARIMA argument: ARIMA's predicted move is `c + φ * r_{T-1}` — tiny, signed, calibrated. The strategy gate triggers cleanly when `r_{T-1}` was positive, conservatively when it was small or negative. That's the right magnitude of opinion for this signal.

The LSTM's predicted move comes out of recurrent dynamics with capacity to express larger numbers. Often it does — the predictions are a normal-magnitude function of the input chunk's recent variance. Bigger predicted-move magnitude makes the strategy gate easier to flip on noise. MAE $274.02 vs ARIMA's $260.50 is the diagnostic: the LSTM's predictions are wrong by *more* than the per-bar volatility, on average. That's a sign of opinion-overstatement.

This is the same mechanism Article 4 surfaced for XGBoost — capacity hurts at low SNR — applied to a model whose inductive bias is *better* than XGBoost's (it does respect order) but still worse than ARIMA's (it doesn't bake in "tiny linear thing on differences").

---

## What's next

Article 9 — *I built the transformer my repo is named after. ARIMA still won.* — is the architecture step up. Same Darts harness, same data slice, same metrics module, but a Temporal Fusion Transformer with 4 attention heads, 48 bars of context, multivariate past covariates, *and* future covariates (cyclical hour-of-day, day-of-week). The headline: it lands at Sharpe +4.59 / dir_acc 0.5053 — *behind* this article's LSTM.

Spoiler thesis: the data is the floor. The capacity escalator does not climb past it.

Article 10 — *Your Sharpe is mostly drift* — will then deflate most of the Sharpe numbers in this article (and the entire series) by separating "signal" from "long-during-rally beta". The LSTM's +4.95 Sharpe is going to look smaller after that lens. ARIMA's +7.28 will, mostly, survive.

---

*Code: [experiments/04_lstm/](../../experiments/04_lstm/) · [04_lstm/results/sweep.csv](../../experiments/04_lstm/results/sweep.csv) · Repo: [transformer_bitcoin_ai](../../../README.md)*
