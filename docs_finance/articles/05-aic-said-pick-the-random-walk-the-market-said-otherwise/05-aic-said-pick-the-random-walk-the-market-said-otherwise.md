# AIC said pick the random walk. The market said otherwise.

*Article 5 of the series* **Can you actually predict Bitcoin? An honest lab notebook.**

---

This is a short, sharp post. The data is the argument. The argument is that information criteria and out-of-sample trading metrics, on the same near-random-walk, disagree by exactly one parameter — and that one parameter is the difference between "the model says nothing" and "the model produces a +7.52 Sharpe trading rule".

Run `make 02_arima_sweep`. The full table from [02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv), sorted by AIC:

| order | AIC | BIC | dir_acc | Sharpe | cum_ret |
|---|---:|---:|---:|---:|---:|
| **(0, 1, 0)** | **93377.5** | **93384.3** | **NaN** | **NaN** | **0.000** |
| (2, 1, 0) | 93379.5 | 93399.8 | 0.5019 | +1.73 | +10.1 % |
| (1, 1, 0) | 93379.5 | 93393.1 | **0.5373** | **+7.52** | **+53.5 %** |
| (0, 1, 1) | 93379.5 | 93393.1 | **0.5373** | **+7.52** | **+53.5 %** |
| (0, 1, 2) | 93379.5 | 93399.8 | 0.5025 | +1.83 | +10.8 % |
| (5, 1, 5) | 93379.1 | 93453.6 | 0.4826 | +1.46 | +8.7 % |
| (1, 1, 1) | 93381.5 | 93401.8 | 0.5336 | +7.28 | +53.0 % |
| (2, 1, 2) | 93383.3 | 93417.2 | 0.4876 | +2.38 | +15.2 % |
| (5, 1, 0) | 93385.0 | 93425.6 | 0.4956 | +2.88 | +17.9 % |
| (0, 1, 5) | 93385.1 | 93425.7 | 0.4975 | +2.96 | +18.4 % |
| (3, 1, 3) | 93386.8 | 93434.2 | 0.4882 | +2.89 | +18.5 % |
| (1, 0, 1) | 93404.0 | 93431.0 | 0.4820 | +2.88 | +1.0 % |

The AIC-best order is `(0, 1, 0)` — literally the naive last-value model from Article 1. Its `directional_accuracy` and `Sharpe` are `NaN`, because the model never expresses a direction. The Sharpe-best order is `(1, 1, 0)` (tied with `(0, 1, 1)`), one parameter heavier, with dir_acc 0.5373 / Sharpe +7.52.

Two perfectly defensible model-selection criteria, on identical data, with the same code, point at *opposite* models. This article is about why that disagreement isn't a bug, and what to actually do about it.

---

## What AIC is rewarding, and why it's not lying

AIC is `2k - 2 ln L`, with `k` the number of parameters and `ln L` the in-sample log-likelihood. Lower is better. Each extra parameter costs a flat `+2`, and has to buy back at least one unit of log-likelihood to be worth it.

On a near-random-walk like 1h BTC log-returns, the autocorrelation at lag 1 is small — a few percent. That means *adding* an AR(1) term improves `ln L` only marginally over the random-walk null, because the noise variance dominates the likelihood. The marginal `ln L` improvement is barely larger than the `+2` parameter penalty AIC charges to buy it. Result: `(0,1,0)` and `(1,1,0)` tie at AIC `93379.5`, and `(0,1,0)` wins on the BIC tiebreaker — its 1-parameter model has no parameter penalty to pay at all.

This is AIC working as designed. It is *correctly* telling you that the additional parameter in `(1,1,0)` is paying its way only marginally on the in-sample fit. There is no leakage, no methodological error, no degenerate boundary case. AIC's verdict is honest — it just doesn't know that the parameter we're considering, while small, is exactly the parameter that turns a flat strategy into a directional opinion that pays out on the test set.

BIC behaves similarly with a stronger penalty (`ln(n) * k` instead of `2k`), which is why it also picks `(0,1,0)`. The two information criteria agree, and they agree on the random walk. They are not in disagreement with each other. They are in disagreement with the *test-set Sharpe*.

---

## What Sharpe is rewarding, and why it's not lying either

Sharpe is computed on the test slice, with the standard long-flat strategy gate (`pred > ref`). It rewards getting the *sign* of the predicted move right at the right moments, weighted by the size of the realized return. It does not care about model parsimony, in-sample fit, or parameter count.

`(1,1,0)`'s tiny AR coefficient produces predictions that are small but signed mostly with `r_{T-1}`. On the test window — the late-2024 rally — that's a one-bar momentum rule with a calibrated magnitude. It wins because hourly BTC has a mild positive autocorrelation at lag 1 that survives, in this slice, just enough to clear the strategy gate above coin-flip.

`(0,1,0)` produces no signed prediction at all. The strategy is permanently flat. The Sharpe is `NaN` and the cumulative return is exactly zero. Sharpe correctly judges this as "not a strategy". It doesn't disagree with AIC about parsimony; it answers a different question.

---

## Why "AIC vs Sharpe" is not a bug, and not actually a contradiction

The two metrics are measuring orthogonal things:

- **AIC asks:** *given the in-sample data, what's the most parsimonious model that fits the noise?* Answer: the random walk.
- **Sharpe asks:** *given the held-out window, which model's directional opinions, when traded, produce the best risk-adjusted return?* Answer: the AR(1).

Both questions have correct, faithful answers. The answers are different because the questions are different. When people complain that "auto_arima picks a useless model", what they usually mean is "I asked auto_arima a parsimony question and I'm dissatisfied that it didn't answer my trading question". That's user error, not a flaw in AIC.

The mistake to avoid is *picking* the model on the metric you also use to evaluate it. Use AIC when you're choosing a model class on training data; use Sharpe when you're judging a chosen model on test data. The trap that swallows most people is using either metric as both selector and judge.

---

## Why you can't just Sharpe-shop the order

The next thought is "OK so let me just pick `(1,1,0)` because it's the Sharpe-best in the sweep". Don't do that.

The sweep was run on the same 1,608 test bars that everything else in this lab is reported on. If I pick the order whose Sharpe is best on that window, I have implicitly trained on the test set. The +7.52 Sharpe stops being a measurement of the model and becomes a measurement of *me, choosing the model after seeing the answer*. Article 6 will demonstrate exactly this: take the same orders to a different test slice and the ranking shifts.

This is why the lab ships `(1, 1, 1)` as the default in [02_arima/config.yaml](../../experiments/02_arima/config.yaml):

```yaml
# Common starting points for hourly BTC close:
#   [1, 1, 0]  pure AR(1) on first differences — simplest non-trivial model
#   [1, 1, 1]  classic Box-Jenkins default; one AR + one MA term
#   [2, 1, 2]  slightly richer; more parameters to estimate
# Pick explicitly. There is no auto-selection in this experiment.
order: [1, 1, 1]
```

`(1, 1, 1)` is the canonical Box-Jenkins default for differenced-series forecasting — it pre-dates this dataset by half a century and was not chosen because it won the sweep. It clocks dir_acc 0.5336 / Sharpe +7.28 on the test slice — second-best behind `(1,1,0)`, and *consistent* with `(1,1,0)` being a nearby small perturbation of the same family. That consistency is the audit. **The sweep's purpose is falsification, not selection.** If `(2,1,2)` had been a runaway winner on Sharpe and AIC simultaneously, that would be evidence we picked the wrong default. Instead, every order from `(2,1,0)` outward collapses to dir_acc ≈ 0.50 and Sharpe ≈ +2 (the long-during-rally floor — Article 10's territory), which says (a) the signal is genuinely concentrated in a single AR(1) lag, and (b) the default we chose by prior-knowledge is in the right neighborhood. Both are reassuring.

---

## A small structural surprise: `(1,1,0)` and `(0,1,1)` tie

Look at the table again. `(1,1,0)` and `(0,1,1)` have identical dir_acc, identical Sharpe, identical cum_ret, and near-identical AIC.

This is not coincidence. On first-differenced returns, an AR(1) model and an MA(1) model parameterize the *same* one-step autocorrelation through different mechanisms. The fitted coefficients are slightly different (AR(1) on differences vs MA(1) on residuals of differences), but for small magnitudes their one-step-ahead predictions converge. The two parameterizations give the same trading signal almost bar for bar.

The fact that they tie is the *opposite* of suspicious — it's a sanity check that the sweep is not artifacted by `statsmodels` initialization or some quirk of the AR/MA estimator. Two parameterizations recovering the same out-of-sample number is the kind of agreement we expect from reality, not from a leak.

---

## The general lesson

For any serial-data model selection problem with a low signal-to-noise ratio:

1. **Use information criteria for what they're good for.** AIC/BIC tell you when you're paying parameters that don't earn their keep on training data. They are *useful* for ruling out over-rich models — note how `(5,1,5)` has AIC 93379.1 (nearly equal to `(0,1,0)`) and dir_acc 0.4826 (below chance). AIC correctly flagged that the extra ten parameters did nothing for in-sample fit. The fact that it also flagged the random walk is a feature when you're picking *between* over-rich models.

2. **Use OOS metrics for what they're good for.** Sharpe / dir_acc tell you whether a chosen model does the thing you actually care about, on data it hasn't seen.

3. **Never let OOS metrics select the model.** That's leakage. You can use them to *audit* a model — "is the chosen order at least competitive on Sharpe?" — but not to choose it. The order with the highest test-Sharpe is, by construction, partly a function of the test slice.

4. **Never let IC select the model alone on a near-random-walk.** The minimum will sit at the no-skill point and you will ship a flat strategy.

5. **Pick the model from prior knowledge, then check both metrics.** That's what the lab does. `(1,1,1)` is theory-defensible; AIC is in the right ballpark; Sharpe is competitive. None of those are picking the model — they're auditing the pick.

In a sentence: **information criteria ask "is this the simplest model that fits?", and Sharpe asks "is this the model that pays?". For a near-random-walk, those two questions have different correct answers.**

---

## What this article is *not* saying

- It's not saying AIC is broken. It's saying AIC's question isn't your question if your goal is a P&L.
- It's not saying Sharpe is the right model-selection metric. It's saying Sharpe is the right model-evaluation metric, and using it for selection is leakage.
- It's not saying `(1,1,1)` is the optimal model. It's saying it's a *defensible* model, which is a different and weaker (and honest) claim.
- It's not saying you should never use `auto_arima`. It's saying that if you do, you should be clear-eyed about what you've asked it to optimize.

---

## What's next

Article 6 — *Same model, different window, opposite verdict* — will rerun the entire model lineup on Q1 2024 only and Jan–Nov 2024, and watch the leaderboard rearrange. The post-Article 4 ranking is XGBoost dead last on Jan–Nov; on Q1 it leads. The LSTM was *broken* on Q1 (Sharpe -0.54), competitive on Jan–Nov (+4.95). The `(1,1,0)` Sharpe edge in this article is on one window. Article 6's job is to show that one window of Sharpe is not a finding, even when the lab is set up correctly.

That's the real reason this article is titled "AIC said pick the random walk, the market said otherwise" rather than "Sharpe said pick `(1,1,0)`". The market was right *on this slice*. Whether it's still right on a different slice is exactly the question Article 6 is built to answer.

---

*Code: [experiments/02_arima/sweep.py](../../experiments/02_arima/sweep.py) · [experiments/02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv) · Repo: [transformer_bitcoin_ai](../../../README.md)*
