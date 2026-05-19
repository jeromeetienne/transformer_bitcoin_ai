# Outline — AIC said pick the random walk. The market said otherwise.

## One-line pitch
The ARIMA `(p, d, q)` sweep ([02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv)) shows the **AIC-best order is `(0, 1, 0)`** — literally the naive baseline. The **Sharpe-best is `(1, 1, 0)`**. Information criteria and out-of-sample trading metrics disagree by exactly one parameter and a sign of conviction. Which one to trust depends on what the model is *for*.

## Audience
Anyone who has ever looked at `auto_arima`'s output and shipped it. Quantitative researchers who use AIC/BIC routinely. Series readers who saw Article 4 wave at the sweep and want the actual table.

## Thesis
AIC and BIC reward parsimony given fit, not out-of-sample skill. On a near-random-walk, the parsimony minimum is *exactly the model that says nothing*. If your goal is a P&L, you cannot ship the AIC-best order — but you also can't blindly Sharpe-shop on the test set. The honest move is to pick a small, defensible `(p, d, q)` from prior knowledge, treat the sweep as a sanity check, and write down which metric you used so future-you knows.

## Structure

### 1. The hook — the table
Reproduce the sweep table here. Sort by AIC (best-first). Show:
- AIC-best: `(0,1,0)`. AIC=93377.54. dir_acc=NaN, Sharpe=NaN.
- Sharpe-best: `(1,1,0)` and `(0,1,1)` (tied). dir_acc=0.5373, Sharpe=+7.52.
- Default config: `(1,1,1)`. AIC=93381.54, dir_acc=0.5336, Sharpe=+7.28.
- The wider orders `(2,1,0), (0,1,2), (2,1,2), (3,1,3), (5,1,0), (0,1,5), (5,1,5)` — AIC creeps up, dir_acc collapses to ≈0.50, Sharpe falls.

The provocation: AIC and Sharpe disagree by one parameter, and that one parameter is the difference between "express no opinion" and "produce a +7.52 Sharpe trading rule".

### 2. What AIC actually rewards
- AIC = `2k - 2 ln L`. Lower is better.
- `k` is the number of parameters; `ln L` is the in-sample log-likelihood.
- Adding parameters: penalty `+2`. Has to buy back at least 1 unit of log-likelihood to be worth it.
- On a near-random-walk, `(0,1,0)` (the random walk itself) has the smallest `k` and the noise variance dominates `ln L`. Extra parameters barely move `ln L`, so they get penalized.
- BIC behaves similarly with a stronger penalty. The fact that AIC and BIC both prefer `(0,1,0)` is consistent.

### 3. Why AIC is *literally telling the truth* — and still the wrong metric
- AIC is asking: "given the in-sample fit, what is the best bias-variance trade-off?"
- For a near-random-walk, that trade-off is *minimized* by the model that does no work. AIC is correct.
- But AIC has nothing to say about *trading performance*. It's optimizing in-sample log-likelihood, not directional accuracy on a held-out window.
- Concrete: `(0,1,0)` has dir_acc=NaN and Sharpe=NaN — the strategy is permanently flat (it never predicts a positive move). AIC said yes. The trading desk says you're not doing anything.

### 4. Why Sharpe disagrees
- Sharpe / dir_acc are computed on the *test* slice with a non-trivial strategy gate (`pred > ref`).
- `(1,1,0)` produces tiny but signed predictions that cross the gate at the right moments, capturing mild momentum on the test window.
- It pays. AIC can't see that pay-off because it's measured on the wrong distribution.

### 5. Why you can't just optimize Sharpe over the order
- Sharpe-shopping over `(p, d, q)` on the test slice is exactly the leakage Article 2 set up the lab to prevent.
- The sweep is run on the same test slice that everything else in the lab is reported on. If you pick the order whose Sharpe is best, you have implicitly trained on the test set.
- That's why we ship `(1,1,1)` (a defensible Box-Jenkins default — one AR, one MA, on differences) rather than `(1,1,0)`. It's a-priori plausible, it's not the test-slice winner, and it's not chosen *because* it won the sweep.
- The sweep's purpose is **falsification**, not selection. If `(2,1,2)` had been a runaway winner on Sharpe and AIC simultaneously, that would be evidence we picked the wrong default. Instead, every order from `(2,1,0)` outward collapses to dir_acc ≈ 0.50, which says (a) the signal is genuinely concentrated in a single AR(1) lag, and (b) we picked a nearby plausible default. Both are reassuring.

### 6. Tied orders and what they tell us
- `(1,1,0)` and `(0,1,1)` produce identical dir_acc, Sharpe, cum_ret, and near-identical AIC. This is structural, not coincidence: on first-differenced returns, AR(1) and MA(1) of order one are observationally near-equivalent on a near-random-walk because both are picking up the same `r_{t-1}`-style autocorrelation through different mechanisms.
- The fact that they tie is the *opposite* of suspicious — it's a sanity check that the sweep is not artifacted by `statsmodels` initialization. Two different parameterizations recovering the same out-of-sample number is good news.

### 7. The general lesson
For any serial-data model selection problem:
- **Use IC for what it's good for**: bias-variance on in-sample fit. It tells you when you're paying parameters that don't earn their keep.
- **Use OOS metrics for what they're good for**: out-of-sample skill. They tell you whether your model does the thing you actually care about.
- **Never let OOS metrics select the model**. That's leakage. Use them to *audit* a defensible-by-prior model.
- **Never let IC select the model on a near-random-walk**. The minimum will be the no-skill point and you will ship a flat strategy.

The only sustainable workflow: pick `(p, d, q)` from theory or prior data; check that AIC/BIC don't disagree wildly; check that OOS Sharpe is in the right ballpark; if any of those conditions fails, go back and pick a different family — not a different order.

### 8. The repo's actual choice and why
- We ship `order: [1, 1, 1]` in [02_arima/config.yaml](../../experiments/02_arima/config.yaml). The YAML even spells out the candidates and their roles:
  ```yaml
  # [1, 1, 0]  pure AR(1) on first differences — simplest non-trivial model
  # [1, 1, 1]  classic Box-Jenkins default; one AR + one MA term
  # [2, 1, 2]  slightly richer; more parameters to estimate
  # Pick explicitly. There is no auto-selection in this experiment.
  ```
- The "no auto-selection" line is the point of the article in five words.

### 9. Closing
- AIC told the truth about parsimony. Sharpe told the truth about trading. Neither is the metric you should ship by alone.
- Tease Article 6: even *Sharpe* on a single window is fragile. Different test slice, different best order. The article on regime-fit is up next.

## Key code/file references
- [experiments/02_arima/sweep.py](../../experiments/02_arima/sweep.py) — the order grid
- [experiments/02_arima/results/sweep.csv](../../experiments/02_arima/results/sweep.csv) — the canonical table
- [experiments/02_arima/config.yaml](../../experiments/02_arima/config.yaml) — the explicit-pick comments

## Tone notes
- Short, sharp post. ~1,200–1,500 words. The data is the argument.
- Avoid the temptation to pick a side. AIC and Sharpe both say something true.

## Length target
~1,200–1,500 words.
