# Articles TODO

## Prompt

> i would like to write a series of articles to communicate the work on this repo
> - please suggest articles title + short each description of each of the posts

## Proposed series — *Can you actually predict Bitcoin? An honest lab notebook*

**1. "The Floor: why a one-line BTC predictor is so hard to beat"**
The naive last-value baseline ([01_baseline](../experiments/01_baseline/)) clocks MAE = $260.50 on 1h BTCUSDT. Frame why "predict the last price" is a mathematical floor on a near-random-walk, why MAE at this resolution measures volatility rather than skill, and why every later experiment is judged by **direction**, not error.

**2. "How I structured a forecasting lab so nothing rots"**
Engineering post about the repo, not the models: numbered self-contained experiments, YAML as single source of truth, shared `src/btc_ai/eval/` so every leaderboard row is comparable, Makefile-as-API, uv for envs. Includes the "every experiment ships a report, even the negative ones" discipline — the lab notebook is the product, not the leaderboard. Aimed at ML practitioners who keep meaning to clean up their notebook graveyard.

**4. "ARIMA(1,1,0) beat my XGBoost. One parameter beat thirty-one features."**
The headline result from [02_arima](../experiments/02_arima/) and [03_gradient_boosting](../experiments/03_gradient_boosting/): a single AR coefficient on differenced returns produces dir_acc 0.5373 / Sharpe +7.52, while XGBoost with 24 lagged returns + rolling stats + volume + OHLC lands at 0.4872 / +1.45. Discuss signal-to-noise, what trees can and can't do (no extrapolation → predict returns, reconstruct price), and what "feature engineering didn't help" actually means.

**5. "AIC said pick the random walk. The market said otherwise."**
Spin-off from the ARIMA order sweep ([02_arima/results/sweep.csv](../experiments/02_arima/results/sweep.csv)): the AIC-best order is `(0,1,0)` — literally naive — while the Sharpe-best is `(1,1,0)`. A short, sharp post on why information criteria and out-of-sample trading metrics disagree, and which one to trust when the goal is a P&L.

**6. "Same model, different window, opposite verdict: a regime-fit horror story"**
The repo runs every model on Q1-only and Jan–Nov 2024 slices. XGBoost was leader on Q1 (dir_acc 0.5185), last on Jan–Nov (0.4872). The LSTM was *broken* on Q1 (Sharpe -0.54), competitive on Jan–Nov (+4.95). Use this to argue that single-split eval is a trap and that anything a paper claims about BTC depends on the test window.

**7. "Walk-forward without retraining: an honest cheat I keep using"**
Methods post on the evaluation harness — `statsmodels.apply(refit=False)` for ARIMA, `historical_forecasts(retrain=False)` for Darts. What's preserved (no leakage), what's sacrificed (no regime adaptation), and when the cheat stops being defensible. Placed before the deep-learning posts because every model from 04 onward depends on this harness.

**8. "An LSTM, a 24-bar window, and the question of how much past matters"**
Walkthrough of [04_lstm](../experiments/04_lstm/) using Darts' `BlockRNNModel`: why log-returns as the target, why past-covariates only, and the sweep finding that `input_chunk_length=24` (one day) fails while 48–96h works. The interesting bit: longer context didn't beat ARIMA's one lag — discuss what that says about where the signal lives.

**9. "I built the transformer my repo is named after. ARIMA still won."**
The headline negative result from [05_transformer](../experiments/05_transformer/): a Temporal Fusion Transformer with 4 attention heads, 48h of context, multivariate past covariates, and cyclical future covariates lands at Sharpe +4.59 / dir_acc 0.5053 — behind LSTM (+4.95 / 0.5196) and well behind ARIMA(1,1,1) (+7.28 / 0.5336). The best swept config (`48/64/8/1/0.1`) closes some of the gap (+4.83) but doesn't catch LSTM. Inside the post: a **calendar-blindness** section using the [05_transformer sweep](../experiments/05_transformer/results/sweep.csv). Across 6 configs the Sharpe range was +3.45 to +4.83; the wider configs that *had* capacity to exploit hour_sin/cos and dow_sin/cos cyclical inputs didn't pull away from the smaller ones. If hourly BTC carried meaningful intraday or weekly seasonality, capacity-vs-Sharpe would look different. It doesn't. Discuss what "more capacity didn't help" means at this signal-to-noise level, why *width* beat *depth* in the sweep (deeper LSTM-stack configs all underperformed), and the moral discomfort of publishing a model whose final leaderboard position is worse than a 2-parameter linear model from 1970.

**10. "Your Sharpe is mostly drift: separating skill from beta in a rising market"**
The cleanest cross-cutting insight, made unmissable by [06_pretrained_direct](../experiments/06_pretrained_direct/) (Sharpe +4.29 with dir_acc 0.5019 — chance-level direction, yet a positive number that looks like skill). Any long/flat strategy that *occasionally* says "long" — even at random — captures part of the Sep–Nov 2024 rally as BTC moved from ~$58 k to ~$96 k. The Sharpe number is real but it's measuring the test window's drift, not the model's signal. Apply the same lens retroactively to LSTM (+4.95) and TFT (+4.59); only ARIMA's +7.28 has a dir_acc strong enough to be doing real directional work on top of that drift. Argue that any single-window Sharpe headline on BTC is partially beta, and that the only fix is rerunning on a falling or chopping regime.

**11. "What does a model that has never seen Bitcoin think Bitcoin will do?"**
Walkthrough of [06_pretrained_direct](../experiments/06_pretrained_direct/) using Amazon's Chronos-2 (120 M params), zero-shot, no fine-tuning, on the same 1h BTCUSDT log-returns slice as 04/05. Honest framing: if a pretrained prior over millions of unrelated series transfers, that's the result; if MAE sits near the naive floor (~$260) and dir_acc ≈ 0.50, the data — not the architecture, not the parameter count — is the bottleneck. The actual finding: Chronos-2 lands at MAE $262 (within $2 of naive) with dir_acc 0.5019 (chance), confidently centred on `close_{T-1}`. The first time the lineup answers "is BTC unusually hard, or are we just bad at it?" *(TimesFM 2.5 is wired in the harness but not yet run; deferred to a follow-up post once that result lands.)*

**12. "Probabilistic forecasts, finally: what a q10/q50/q90 band changes"**
Methods post, generalized beyond 06. The first experiment in the lineup with calibrated prediction intervals ([06_pretrained_direct](../experiments/06_pretrained_direct/), via `QuantileRegression([0.1, 0.5, 0.9])` and `num_samples=200`). What intervals give you that point predictions don't: risk-aware position sizing, calibration diagnostics ("did the realized return fall inside the 80% band the predicted fraction of the time?"), and the difference between "the model predicts +1 %" and "the model says 20 % chance of −2 % and 20 % chance of +3 %". The negative finding worth showing: on hourly BTC the q0.1 – q0.9 spread is *narrow and centred on the last value* — quantile loss is doing exactly what it should; there's just no asymmetry to exploit. Ties back to why every prior post talked about Sharpe but never about uncertainty.

**13. "Where this repo goes next"**
Roadmap post — what's left now that the headline architecture is in. Open threads: TimesFM 2.5 zero-shot to confirm or break the Chronos result, ARIMAX with funding rate / perp basis, fine-tuning the foundation models instead of zero-shot, multi-seed runs to turn point-estimate Sharpe gaps into distributions, and re-running every model on a regime that *isn't* the Sep–Nov 2024 rally. Honest about which of these are "fix the methodology" vs. "chase signal that may not exist."

**14. (optional) "Stop annualizing your Sharpe"**
Short methodological piece prompted by the +7.28 Sharpe headlines. Per-bar Sharpe ≈ 0.080 with SE ≈ 0.025 on 1,608 bars — about 3σ from zero. The √8760 multiplier makes the number look like a hedge fund. Useful to write because every BTC ML post on the internet does this and never says it.
