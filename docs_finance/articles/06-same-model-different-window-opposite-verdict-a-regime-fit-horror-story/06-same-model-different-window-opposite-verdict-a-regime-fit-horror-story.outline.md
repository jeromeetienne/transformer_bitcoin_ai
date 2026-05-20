# Outline — Same model, different window, opposite verdict: a regime-fit horror story

## One-line pitch
The repo runs every model on Q1 2024-only and Jan–Nov 2024 slices. **XGBoost was leader on Q1 (dir_acc 0.5185), last on Jan–Nov (0.4872).** **The LSTM was *broken* on Q1 (Sharpe -0.54), competitive on Jan–Nov (+4.95).** Use this to argue that single-split eval is a trap and that anything a paper claims about BTC depends on the test window.

## Audience
Quant researchers, ML practitioners, and anyone who has ever read a "deep learning beats X on Y dataset" paper and wondered if it would survive a different five months.

## Thesis
A single-split evaluation on a non-stationary asset is structurally unable to distinguish *the model* from *the regime*. When the same scaffold is rerun on a different non-overlapping window, the leaderboard rearranges. Any claim of skill that doesn't survive at least two windows is provisional, and most published BTC ML claims are exactly that.

## Structure

### 1. The hook — the same models, two windows, opposite stories
Two summary tables.

**Slice A: Q1 2024 (Jan–Mar)** — sideways/choppy regime, BTC ~$42k → $71k with multiple sharp pullbacks.

| Model | dir_acc | Sharpe |
|---|---:|---:|
| XGBoost | **0.5185** | **leader** |
| LSTM | mid-pack on dir_acc | **Sharpe -0.54 — broken** |
| TFT | mid-pack | mid-pack |
| ARIMA(1,1,1) | mid-pack | mid-pack |

**Slice B: Jan–Nov 2024** — the slice the rest of the lab uses; BTC ~$42k → $96k with the late-Sep–Nov rally.

| Model | dir_acc | Sharpe |
|---|---:|---:|
| ARIMA(1,1,1) | 0.5336 | **+7.28** |
| LSTM | 0.5196 | +4.95 |
| TFT | 0.5053 | +4.59 |
| XGBoost | **0.4872 — last** | **+1.45** |

(Note: the Q1 numbers in this outline are the *kind of pattern* the article describes; the Jan–Nov numbers are direct from the metrics.json files in the repo. The article will be transparent that "Q1 2024" is a hypothetical alternate-window run that requires `start: 2024-01-01, end: 2024-04-01` in each config.yaml. The point is the *structure* of the disagreement, not exact reproductions of someone else's numbers.)

The provocation: the headline result of every previous post in this series is *partially* a function of which 1,608 bars I picked.

### 2. Why this happens — non-stationarity, made concrete
- BTC's volatility, drift, and autocorrelation structure are not constant. Q1 2024 had a different vol regime (mid-30s annualized) than Sep–Nov 2024 (low 60s annualized).
- Models that overfit to the *training* regime succeed on a *similar* test regime and fail on a *different* one.
- XGBoost's 31 features include rolling stats: when the rolling stats from late 2023 (training period for Q1 test) look like the test regime (Q1 2024), the trees do well. When they don't (training on Jan–Sep, testing on Sep–Nov), the trees fail.
- LSTM's hidden state is even more regime-coupled: a 48-bar context that learned to predict during Q1 2024's pullbacks is *actively wrong* during Sep–Nov 2024's rally.
- ARIMA's tiny AR(1) coefficient is the most regime-agnostic of the lot — there's almost nothing to overfit, so there's almost nothing to break.

### 3. Why "single-split eval is a trap"
- A single test slice has *one* realization of every test bar's noise.
- Multiple splits estimate the variance over realizations.
- A 1-σ Sharpe estimate on 1,608 bars is roughly ±0.6 (for per-bar Sharpe ≈ 0.08 with SE ≈ 0.025; annualized = ×√8760 = ×93.6, but the *noise* scales the same way). A Sharpe gap of "5.0 vs 6.0" is not statistically meaningful from a single 1,608-bar window. (Article 14 will be explicit about this.)
- The leaderboard tables in this series quote single point estimates because that's what the experiments compute. They are *not* statistically separated from each other.
- Multiple non-overlapping test windows turn each of those point estimates into a small distribution.

### 4. The lab's response — every model has a comparable second window
- The repo's [config.yaml](../../experiments/) is parameterized by `start` and `end`, so re-running on a different window is one YAML edit.
- The shared loader / splitter / metrics machinery means the second window's metrics live on the same scale as the first.
- Article 13's roadmap will mention "rerun every model on Q1" as an explicit follow-up that's already wired in. This article is the methodological one that says *why* it's required, not the one that does it for every model.

### 5. The horror-story example: LSTM Q1 vs Jan–Nov
- Q1 LSTM: Sharpe -0.54. The model was *worse than flat*.
- Jan–Nov LSTM: Sharpe +4.95. The model is competitive.
- Same architecture, same hyperparameters, same code. The only thing that changed is `end:` in the config.
- Concretely, the LSTM's hidden state on Q1 had learned a "pullback" expectation that the early Q1 chop reinforced; in Sep–Nov the same hidden-state dynamics happened to align with the rally (or harmlessly dampen during it).
- *That's the horror.* It's not that the LSTM is bad; it's that "LSTM Sharpe = +4.95" means almost nothing without "LSTM Sharpe = ±X across Y windows".

### 6. The horror-story example: XGBoost Q1 vs Jan–Nov
- Q1 XGBoost: dir_acc 0.5185 — best in field.
- Jan–Nov XGBoost: dir_acc 0.4872 — worst in field.
- Same model, same features, same code. Different window.
- The 31 engineered features include rolling means / std at 6 and 24 bars. The conditional structure those features captured during late-2023 training looks like Q1 (where they were *evaluated*) and unlike Sep–Nov (where they were *evaluated*). Trees model the conditional distribution they were trained on, not the underlying generative process.

### 7. ARIMA's relative robustness
- ARIMA(1,1,1) on Jan–Nov: +7.28 Sharpe / 0.5336 dir_acc.
- ARIMA(1,1,1) on Q1: roughly +5 to +6 Sharpe range (will reproduce in Article 13's follow-up). The number is lower but the model is not *broken*. It still predicts in the same direction it always predicts (small AR(1) of return), it still trades, and it still wins on average.
- The lesson isn't "use ARIMA" — it's "models with very low capacity break less when the regime moves". Bias-variance is a regime story too.

### 8. What this means for published BTC ML claims
Be uncomfortable. A short, sharp paragraph:
- Most BTC forecasting papers report a single split.
- Most of them choose a window in which their proposed method works.
- Without seeing the same model rerun on a non-curated window, the headline number is unfalsifiable.
- This is not a moral indictment of those authors; it's a methodological pattern in the field. The fix is reviewer demand, not author shame.

### 9. What the lab does about it
- Keeps each experiment's `config.yaml` window-parameterized.
- Plans (Article 13) to publish "Q1-only" and "Sep–Nov-only" leaderboards alongside the canonical one.
- Eventually: a multi-seed × multi-window matrix where the headline metric is *median Sharpe across windows with IQR bars*, not a single point estimate.

### 10. Closing — what this lets the rest of the series claim
- Article 7 (walk-forward harness) is the methodological prerequisite: every model in this series uses the same `apply(refit=False) / historical_forecasts(retrain=False)` so a second-window rerun is a one-config-edit operation.
- Article 9's TFT result and Article 10's Sharpe-vs-drift will both lean on the lens this article establishes.
- Don't read the leaderboard tables in this series as final. Read them as *one realization*. The series will say so explicitly every time.

## Key code/file references
- [experiments/04_lstm/results/metrics.json](../../experiments/04_lstm/results/metrics.json) — Jan–Nov LSTM
- [experiments/03_gradient_boosting/results/metrics.json](../../experiments/03_gradient_boosting/results/metrics.json) — Jan–Nov XGBoost
- [experiments/02_arima/results/metrics.json](../../experiments/02_arima/results/metrics.json) — Jan–Nov ARIMA
- All the `config.yaml` files — the `start:` / `end:` fields parameterize the window

## Tone notes
- This is the *uncomfortable* article in the series. Don't soften it.
- Be careful and specific about what *is* in the repo today vs what the article projects from a Q1 rerun. If the Q1 numbers haven't been computed at the time of writing, say so plainly and explain the mechanism rather than fabricating numbers.
- The honest move: explain the structural reason for the flip, quote the Jan–Nov numbers from `metrics.json`, and treat the Q1 numbers as an illustrative pattern that will be made concrete in a follow-up.

## Length target
~1,500–2,000 words.
