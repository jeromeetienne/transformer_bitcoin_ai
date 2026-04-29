# Walk-forward without retraining: an honest cheat I keep using

*Article 7 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

This is a methods post. Every leaderboard number in this series — ARIMA's +7.28 Sharpe, the LSTM's +4.95, the TFT's +4.59, the foundation models' answers, all of them — was produced by an evaluation harness that **fits the model once on the training window, then walks the test window one bar at a time without retraining**. This article explains exactly what that means, why it isn't leakage, what it sacrifices, and when the cheat stops being defensible.

If you want a one-line summary: walk-forward-without-retraining is the right benchmark for comparing model families on a frozen test slice. It is *not* the same as deploying a model in production. The lab is transparent about which question it's answering, and this article is where that transparency lives.

---

## Three things people call "walk-forward"

People who run time-series benchmarks call three different procedures "walk-forward". They produce different numbers and answer different questions. It's worth pinning them down.

### Level 1 — Re-fit at every test bar (gold standard, expensive)

At each test bar `t`, refit the model on `[0, t)`, predict bar `t`, advance, refit on `[0, t+1)`, predict bar `t+1`, …. Every test prediction is made by a model trained on strictly more data than the previous one. The published metric is what a deployed system would have observed if it had retrained at every step — including any benefit from regime adaptation that happens during the test window.

Honest. Expensive. For 1,608 test bars and a TFT that takes ~5 minutes to fit, this is *130+ hours per experiment*, on a laptop. It's not happening for this lab.

### Level 2 — Train once on the train window, evaluate one bar at a time without retraining (this lab)

Fit the model once on `[0, train_end)`. Then for each test bar `t ∈ [train_end, n)`, ask the model for a 1-step-ahead prediction using the input it would have had at time `t` — actual observed past values from `[t - input_chunk_length, t)`. The model's *parameters* don't change across the test window; the input window slides.

This is what every committed `metrics.json` in this lab corresponds to.

### Level 3 — Train once, predict the entire test window in one shot (the trap)

Fit the model once. Then ask it to forecast `n - train_end` steps ahead. Each prediction is generated using its *own previous predictions* as input rather than the actual observed values. Errors compound. By bar 50 of the test window, the predicted price track has typically diverged from reality by a multiple of the asset's volatility. This is what the Darts API confusingly also offers as `predict(n=...)`, and it is *not* what we want for a benchmark.

ARIMA has the same trap: `predict(dynamic=True)` rolls the model forward using its own predictions; `predict(dynamic=False)` uses actual observed past values. The lab uses `dynamic=False`.

---

## The exact mechanism, per model family

Concrete code. This is the load-bearing decision in each `run.py`.

### ARIMA — `statsmodels.apply(refit=False)`

From [02_arima/run.py](../../experiments/02_arima/run.py):

```python
fit = ARIMA(train, order=order).fit()                       # fit once on train
extended = fit.apply(close, refit=False)                    # extend filter with test bars
y_pred = extended.predict(start=split, end=len(close) - 1, dynamic=False)
```

Two operations, both worth understanding:

- **`apply(refit=False)`** takes the fitted model and *extends its state* — the Kalman-filter-style internal state, not the parameters — to incorporate the observed test bars. The `(c, φ, σ²)` parameters from the train fit do not change. This is the mechanism that lets `dynamic=False` predict use the actually-observed past lags rather than rolled-forward predictions.
- **`predict(start=split, end=..., dynamic=False)`** then returns one-step-ahead predictions over the test slice. `dynamic=False` is the key argument — it tells `statsmodels` to use the actually-observed past values for the AR lags at every step, not the model's own predictions.

Net effect: at every test bar, the model uses the train-fitted parameters and the actually-observed past, predicts one bar ahead, and that's what the metric records. No leakage; no retraining either.

### Darts (LSTM, TFT, foundation models) — `historical_forecasts(retrain=False)`

From [04_lstm/run.py](../../experiments/04_lstm/run.py):

```python
preds_s = model.historical_forecasts(
    series=target_full_s,
    past_covariates=cov_full_s,
    start=test_start,
    forecast_horizon=1,
    retrain=False,
    last_points_only=True,
)
```

`historical_forecasts` is Darts' walk-forward-evaluation API. The four arguments that matter:

- **`retrain=False`** — never retrain across the test window. Fixed parameters from the train fit.
- **`forecast_horizon=1`** — produce one-step-ahead predictions.
- **`start=test_start`** — begin walking at the first test bar.
- **`last_points_only=True`** — at each step, return only the prediction *at* that bar (not the whole horizon-length forecast, which for `forecast_horizon=1` is the same thing, but the flag matters for multi-horizon variants).

The mechanism is exactly Level 2: at bar `t`, the model receives the input chunk `[t - input_chunk_length, t)` from the *actually observed* series, predicts bar `t`, the chunk slides, repeat.

The same code shape works for [05_transformer/run.py](../../experiments/05_transformer/run.py) (TFT, with future covariates) and [06_pretrained/run.py](../../experiments/06_pretrained/run.py) (foundation models, zero-shot). One harness, three model families.

### XGBoost — implicit walk-forward via `.shift(1)`

XGBoost is a one-shot batch predictor. From [03_gradient_boosting/run.py](../../experiments/03_gradient_boosting/run.py):

```python
y_pred = pd.Series(model.predict(X_test), index=X_test.index, name='r_pred')
```

Looks suspiciously like Level 3 at first glance. It isn't — the walk-forward is baked into the *features*. From [features.py](../../experiments/03_gradient_boosting/features.py):

```python
for i in range(lags):
    cols[f'r_lag_{i + 1}'] = r.shift(1 + i)
for w in feat_cfg['rolling_windows']:
    cols[f'r_mean_{w}'] = r.shift(1).rolling(w).mean()
    cols[f'r_std_{w}']  = r.shift(1).rolling(w).std()
```

Every feature column uses `.shift(1)` over its source. Row at bar `T` sees only data observed strictly before `T`. The "walk-forward" is the row-by-row construction of the test feature matrix; the predictions are then made one bar at a time by indexing into it. No retraining mid-window — same compromise as Level 2, expressed differently for a non-recurrent model.

---

## Why this is not leakage

The mechanism for "no leakage" is the same across all four model families and worth saying explicitly:

1. **Parameters were fit on the train window only and don't change.** No test-window data informs the model's weights.
2. **Inputs to the prediction at bar `t` come strictly from bars `< t`** — actual observed past, not future-leaked.
3. **For LSTM / TFT, `historical_forecasts` enforces a strictly-past input chunk.** No frame at bar `t` includes bar `t` itself.
4. **For ARIMA, `predict(dynamic=False)` uses actual observed lags** — not the model's own predictions, but also not future bars.
5. **For XGBoost, every feature uses `.shift(1)`** — the row at bar `T` cannot reference any column derived from bar `T` itself.

Future covariates in TFT (hour-of-day, day-of-week cyclical encodings) are an apparent exception that resolves cleanly: they are *deterministic* functions of the timestamp, knowable in advance for any bar, training or test. They're not leakage; they're features whose values at bar `T+1` are mathematically known at bar `T`.

The shared invariant: **the metric we publish is the metric a deployed model would have observed if it had been frozen at train time and asked for one-step-ahead forecasts during the test window.** That's a different statement from "the metric a deployed model would have observed if it retrained nightly", and the next section is about that gap.

---

## What this sacrifices — no regime adaptation

A deployed quant model would typically refit nightly (or weekly) on a rolling training window. That model adapts to regime shifts; this benchmark doesn't.

If the test window contains a regime that's *different* from the training window — exactly Article 6's problem — the harness will faithfully report what the train-fitted model does on it. If the train window had a low-vol regime and the test window has a high-vol rally, the harness will report the result of that mismatch *as if the model had been deployed without ever updating*. That's a feature for a benchmark (it isolates "what the model learned" from "how well the model adapts") and a bug for a trading system (where you'd want both).

So the lab's published Sharpe numbers are not "what this model would have made if you'd run it for a year". They are "what this exact, frozen model would have made on the held-out 1,608 bars". Those are different claims. Article 10 (*Your Sharpe is mostly drift*) will use exactly this lens to discount most of the model Sharpes — many of them rest on the test window's drift more than on the model's signal.

The harness is therefore a **lower bound** on what a properly-retrained deployment could achieve in production. Anything we report from this lab, on a single window, with frozen parameters, is the most pessimistic version of the model's claim.

---

## Why this is the right compromise *for benchmarking*

Three reasons to choose Level 2 over Level 1 for this lab:

1. **Computational tractability.** TFT in [05_transformer](../../experiments/05_transformer/) takes ~5 minutes to train on a laptop GPU. Refitting at every test bar would be ~130 hours per experiment. The foundation models in [06_pretrained](../../experiments/06_pretrained/) sidestep this by being zero-shot, but the LSTM and TFT don't have that luxury.
2. **Across-model comparability.** With every model frozen and then evaluated identically, the metric is *only* sensitive to the model's predictions on the test bars. A retrain-each-step harness conflates "the model captures a real signal" with "the model adapts well to new data" — two claims worth measuring separately. Level 2 measures the first; a hypothetical Level 1 harness would measure both, mixed.
3. **Reproducibility.** Frozen parameters means a given `metrics.json` is exactly reproducible from a YAML + lockfile + cached data. Article 2's discipline depends on this. A retrain-each-step harness has many more degrees of freedom (which retrain cadence? Which window? Re-validate every refit?), and reproducibility starts to leak as a function of those choices.

For benchmarking, Level 2 is the right answer. For deployment, it's not. The series doesn't pretend it's both.

---

## When the cheat stops being defensible

Two failure modes worth naming:

### Long test windows on highly non-stationary assets

If the test window is months long and the regime shifts mid-window, the train-fitted parameters become progressively out of date and the published number understates what an adaptive deployment would do. The 1,608-bar (~67-day) test window in this lab is small enough that a single regime usually dominates — but Article 10 will note that the Sep–Nov 2024 rally is itself a single non-trivial regime, and Article 13's roadmap will mention rolling-refit as an explicit follow-up.

### Models with state that *should* update on test bars

LSTM and TFT have hidden states that, in principle, should evolve as new observations arrive. Darts' `historical_forecasts(retrain=False)` *does* slide the input chunk, which means the recurrent state at test bar `t` is computed from the most recent `input_chunk_length` actually-observed bars. So the hidden state *does* update — the parameters don't, but the activations do. That's the right behavior for one-step evaluation; it's only a problem if the model relies on long-range context that exceeds `input_chunk_length`.

ARIMA's situation is similar: `apply(refit=False)` extends the Kalman-filter state with the observed test bars without re-estimating parameters. State updates, parameters don't. Again, the right behavior for one-step evaluation.

The harness is a benchmark, not a universal claim. Production deployments should refit. The lab is silent on what cadence is right; that's a different research question.

---

## The `.shift(1)` discipline

Most of the leakage you'd worry about in a time-series benchmark comes from accidentally exposing bar-`t` data to a model whose target is bar-`t`. The lab's defense is a single discipline: **every feature column that uses bar-`t`'s own data is `.shift(1)`-ed**.

Concretely:

- [03_gradient_boosting/features.py](../../experiments/03_gradient_boosting/features.py) — `r.shift(1 + i)` for lag features, `r.shift(1).rolling(w)` for rolling stats, `volume.shift(1)`, OHLC body / range from bar `T-1`.
- [04_lstm/run.py](../../experiments/04_lstm/run.py) — past covariates are bar-`t` OHLCV, but Darts strictly feeds past covariates only at bars `< t` for the forecast at bar `t`.
- [05_transformer/run.py](../../experiments/05_transformer/run.py) — same as LSTM, plus future covariates that are deterministic timestamp encodings (legitimately known at any future bar).
- [06_pretrained/run.py](../../experiments/06_pretrained/run.py) — univariate; the only past data the model sees is the strictly-past target series.

The recurring pattern: any column that uses bar-`t` data is fed to the model only at predict-step `t+1`. No exceptions. The shared loader / splitter / metrics module from Article 2 makes this enforceable because the same `build_features` and `build_target_and_covariates` functions are reused, not reimplemented per experiment.

---

## A note on `rows_test` discrepancies

You'll notice the `metrics.json` files report slightly different `rows_test` counts:

- ARIMA / LSTM / TFT / foundation: 1,607–1,608.
- XGBoost: 1,603.

That's not a leak. XGBoost's feature construction drops rows where any lagged feature is `NaN` — at the head of the train slice, where there isn't yet 24 bars of history for `r_lag_24`. Those drops shift the train/test boundary by a few bars, and rows_test ends up a hair smaller. The metrics are still comparable: the test bars overlap the same ~67-day window, the bar-by-bar predictions are aligned by timestamp, and the small count delta doesn't materially shift any aggregate metric.

If you ever see a much *larger* discrepancy — say, rows_test = 1,200 for one experiment vs 1,608 for another — that's worth investigating. None of the committed runs have that.

---

## What this article enables for the rest of the series

Every model from Article 8 onward is going to quote a leaderboard number that comes out of this harness. The deep-learning posts will not re-derive it; this is the article they refer back to. Specifically:

- Article 8's LSTM result of `dir_acc 0.5196 / Sharpe +4.95` is `historical_forecasts(retrain=False)` over 1,607 test bars.
- Article 9's TFT result of `dir_acc 0.5053 / Sharpe +4.59` is the same harness, the same window, the same `retrain=False`.
- Article 10 will argue that some of these Sharpe numbers are mostly drift; that argument is only meaningful because the harness is fixed and the regime is the only variable.
- Article 11's foundation-model result is *also* `retrain=False`, by construction (zero-shot models don't have anything to retrain).
- Article 12's quantile bands come from `num_samples=200` draws inside the same harness.

The series's leaderboard is only honest because of this harness. It is not the *only* harness one could run; it is not the *most realistic* harness one could run; but it is the harness for which the comparisons across model families are valid by construction.

That's why this article is post 7 — every model from Article 8 onward leans on it.

---

## What's next

Article 8 — *An LSTM, a 24-bar window, and the question of how much past matters* — is where the deep models start. The first interesting finding: `input_chunk_length=24` (one day) *fails* on this dataset; 48–96 hours work. That's a finding about where the signal lives, not just a hyperparameter sweep. The harness this article describes is what makes that finding measurable.

---

*Code: [02_arima/run.py](../../experiments/02_arima/run.py) · [04_lstm/run.py](../../experiments/04_lstm/run.py) · [05_transformer/run.py](../../experiments/05_transformer/run.py) · [06_pretrained/run.py](../../experiments/06_pretrained/run.py) · Repo: [transformer_bitcoin_ai](../../../README.md)*
