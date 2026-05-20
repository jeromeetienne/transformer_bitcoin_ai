# Outline — Walk-forward without retraining: an honest cheat I keep using

## One-line pitch
Methods post on the eval harness. Every model in the lineup uses `statsmodels.apply(refit=False)` (ARIMA) or Darts' `historical_forecasts(retrain=False)` (LSTM, TFT, foundation models). What's preserved (no leakage), what's sacrificed (no regime adaptation), and when the cheat stops being defensible. Placed before the deep-learning posts because every model from 04 onward depends on this harness.

## Audience
ML practitioners who have wired up time-series eval before and felt the pull of "just retrain at every step". Series readers who want to know what the benchmark numbers in articles 4, 8, 9, 10, 11 actually correspond to.

## Thesis
"True walk-forward" — retraining the model at every test step — is the gold standard but is computationally untenable for hundreds of test bars on deep models. "Walk-forward without retraining" is a clean compromise: train once on the train window, *evaluate* one step at a time on the test window with frozen parameters. It avoids leakage by construction and produces metrics directly comparable across model families. It is *not* the same as deploying a model in production — it's a benchmark, not a strategy backtest, and the article will be transparent about the difference.

## Structure

### 1. The hook — three levels of walk-forward
Three distinct things people call "walk-forward":
1. **Re-fit at every test bar.** Honest, expensive, gold standard. Some test bars get a model trained on more data than others; the metric is a measure of "what would have happened if we had retrained nightly".
2. **Train once, evaluate one bar at a time without retraining.** What this lab does. The model's *parameters* are frozen; the model's *input window* slides over the test set as new bars arrive.
3. **Train once, predict the entire test window in one shot.** The naive approach. Uses the trained model state to roll forward without ever ingesting test bars. Compounds errors quickly.

This article is about #2, why it's not #1, and why it's not #3.

### 2. The exact mechanism per model
Concrete code, per model family.

**ARIMA** ([02_arima/run.py](../../experiments/02_arima/run.py)):
```python
fit = ARIMA(train, order=order).fit()
extended = fit.apply(close, refit=False)
y_pred = extended.predict(start=split, end=len(close) - 1, dynamic=False)
```
- `apply(refit=False)` extends the fitted model's filter with the *observed* test bars without re-estimating `(c, φ, σ²)`.
- `predict(dynamic=False)` produces one-step-ahead in-sample predictions on the test slice — i.e., uses *actual* past values at every step, not its own predicted ones.
- Net effect: at every test bar, the model uses the train-fitted parameters and the actually-observed past, predicts one bar ahead, and that's the published metric.

**Darts (LSTM / TFT / foundation models)** ([04_lstm/run.py](../../experiments/04_lstm/run.py)):
```python
preds = model.historical_forecasts(
    series=target_full_s,
    past_covariates=cov_full_s,
    start=test_start,
    forecast_horizon=1,
    retrain=False,
    last_points_only=True,
)
```
- `retrain=False` is the entire point.
- `forecast_horizon=1` and `last_points_only=True` give one-step-ahead predictions across the test window.
- The input window (`input_chunk_length`) slides; the parameters don't change.

**XGBoost** ([03_gradient_boosting/run.py](../../experiments/03_gradient_boosting/run.py)):
- Single `model.predict(X_test)` because XGBoost's features are already lagged. The implicit walk-forward is in the feature construction (`.shift(1)` on every column).
- No retraining mid-window — same compromise, expressed differently for a non-recurrent model.

### 3. Why this avoids leakage
A short, careful section. The mechanism for "no leakage" is:
- Parameters were fit on the train window only. They don't change during the test window.
- Inputs to the predict-step at bar `t` come strictly from bars `< t` — actual observed past, not future-leaked.
- For LSTM/TFT, `historical_forecasts` ensures the input chunk at time `t` is `[t - input_chunk_length, t)`, fully past.
- For ARIMA, `predict(dynamic=False)` uses actual observed lags rather than rolled-forward predicted lags — this *is* the standard non-leaking one-step-ahead evaluation.

The shared invariant: **the metric computed is the metric a deployed model would have observed if it had been frozen at train time and asked for one-step-ahead forecasts during the test window**.

### 4. What this *sacrifices* — no regime adaptation
The honest disclaimer:
- A deployed quant model would typically refit nightly (or weekly) on a rolling training window. That model adapts to regime shifts; this benchmark does not.
- If the test window contains a regime that's *different* from the training window (Article 6's problem), the harness will faithfully report what the train-fitted model does on it — including breaking. That's a feature for benchmarking; it's a bug for trading.
- "Walk-forward without retraining" is therefore a *lower bound* on what a properly-retrained model could achieve in production. The lab's published Sharpe numbers are not "what this model would have made"; they are "what this exact, frozen model would have made".

### 5. Why this is the right compromise *for benchmarking*
Three reasons:
1. **Computational tractability.** TFT in [05_transformer](../../experiments/05_transformer/) takes ~5 minutes to train. Refitting at every test bar would take ~5 minutes × 1,608 = 130+ hours. Not happening on a laptop.
2. **Across-model comparability.** With every model frozen-then-evaluated, the metric is *only* sensitive to the model's predictions on the test bars. A retrain-each-step setup conflates "the model works" with "the model adapts well".
3. **Reproducibility.** Frozen parameters means a given `metrics.json` is exactly reproducible from a YAML + lockfile + cached data. Article 2's discipline depends on it.

### 6. When the cheat stops being defensible
Two failure modes the article explicitly acknowledges:
- **Long test windows on highly non-stationary assets.** If the test window is *months* long and the regime shifts mid-window, the train-fitted model is increasingly out of date and the published number understates what an adaptive deployment would do. Article 10 will use this lens to discount some of the Sharpe numbers in this lab.
- **Models with state that *should* update.** Foundation models in [06_pretrained_direct](../../experiments/06_pretrained_direct/) don't have this problem (zero-shot); ARIMA's tiny state is the right size to refresh. LSTM's hidden state arguably *should* update on test bars even without re-estimating weights — and Darts' `retrain=False` does keep updating the recurrent state via the sliding input window. So this is only a partial concern.

The article should not pretend the harness is universal. It's the right benchmark for *this lab's* research questions; production deployments should refit.

### 7. The walk-forward visualization that makes it click
Quick description (no actual figure required for the article — point at the structure):
- Imagine the test window as 1,608 bars laid out left-to-right.
- The model's input window is a sliding box of length `input_chunk_length` (24 to 256 depending on model).
- At each step, the box slides one bar forward; the model produces one prediction; the metric tracker increments.
- Parameters never change. Box slides; prediction comes out; bar moves.

### 8. The `.shift(1)` discipline
A short subsection on why every feature in this lab uses `.shift(1)`:
- [features.py](../../experiments/03_gradient_boosting/features.py) for XGBoost.
- [04_lstm/run.py](../../experiments/04_lstm/run.py) `build_target_and_covariates` — past covariates are the bar's-own OHLCV, but Darts feeds them strictly *before* each forecast step.
- [05_transformer/run.py](../../experiments/05_transformer/run.py) — same plus *future* covariates (cyclical hour/day-of-week), which are deterministic functions of the timestamp and therefore legitimately known at any future bar.
- The recurring pattern: any column that uses bar-`t` data is fed to the model only at predict-step `t+1`. No exceptions.

### 9. The `*_train`, `*_val`, `*_test` count discipline
- Every `metrics.json` records `rows_train`, `rows_val` (where applicable), `rows_test`.
- The split is the same across models within a window (data is shared).
- `rows_test` differing slightly between experiments (e.g., 1608 for ARIMA vs 1603 for XGBoost) is from the dropna of lagged-feature rows at the head of the train slice. The article should mention this — it's not a leak, it's a small bookkeeping detail.

### 10. Closing — what this enables for the rest of the series
- Articles 8, 9, 10, 11, 12 will all quote `historical_forecasts(retrain=False)` numbers.
- Article 13's roadmap will list "rerun with rolling-refit" as an explicit follow-up — which would be a different question (adaptive deployment) but a fair one.
- The series's leaderboard is only honest because of this harness. It's worth reading once and trusting going forward.

## Key code/file references
- [experiments/02_arima/run.py](../../experiments/02_arima/run.py) `apply(refit=False)`
- [experiments/04_lstm/run.py](../../experiments/04_lstm/run.py) `historical_forecasts(retrain=False)`
- [experiments/05_transformer/run.py](../../experiments/05_transformer/run.py) — same Darts pattern
- [experiments/06_pretrained_direct/run.py](../../experiments/06_pretrained_direct/run.py) — zero-shot inherits the same pattern
- [experiments/03_gradient_boosting/features.py](../../experiments/03_gradient_boosting/features.py) — `.shift(1)` discipline

## Tone notes
- Methods post. Engineering-honest, not hype.
- Be precise about what the harness *is* and what it *isn't*. The "honest cheat" framing should land — this is a useful benchmark, not a deployable system.

## Length target
~1,400–1,800 words.
