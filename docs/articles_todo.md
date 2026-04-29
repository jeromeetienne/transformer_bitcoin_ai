# Articles TODO

## Prompt

> i would like to write a series of articles to communicate the work on this repo
> - please suggest articles title + short each description of each of the posts

## Proposed series — *Can you actually predict Bitcoin? An honest lab notebook*

**1. "The Floor: why a one-line BTC predictor is so hard to beat"**
The naive last-value baseline ([01_baseline_naive](../experiments/01_baseline_naive/)) clocks MAE = $260.50 on 1h BTCUSDT. Frame why "predict the last price" is a mathematical floor on a near-random-walk, why MAE at this resolution measures volatility rather than skill, and why every later experiment is judged by **direction**, not error.

**2. "How I structured a forecasting lab so nothing rots"**
Engineering post about the repo, not the models: numbered self-contained experiments, YAML as single source of truth, shared `src/btc_ai/eval/` so every leaderboard row is comparable, Makefile-as-API, uv for envs. Aimed at ML practitioners who keep meaning to clean up their notebook graveyard.

**3. "The mean-reverter that fixed my metrics"**
[01b_moving_average](../experiments/01b_moving_average/) is 3× worse than naive on MAE but unlocks directional accuracy (naive returns NaN — it never expresses a direction). Use it to introduce dir_acc, cumulative return, Sharpe, and why having a *bad* baseline is structurally necessary.

**4. "ARIMA(1,1,0) beat my XGBoost. One parameter beat thirty-one features."**
The headline result from [02_arima](../experiments/02_arima/) and [03_gradient_boosting](../experiments/03_gradient_boosting/): a single AR coefficient on differenced returns produces dir_acc 0.5373 / Sharpe +7.52, while XGBoost with 24 lagged returns + rolling stats + volume + OHLC lands at 0.4872 / +1.45. Discuss signal-to-noise, what trees can and can't do (no extrapolation → predict returns, reconstruct price), and what "feature engineering didn't help" actually means.

**5. "AIC said pick the random walk. The market said otherwise."**
Spin-off from the ARIMA order sweep ([02_arima/results/sweep.csv](../experiments/02_arima/results/sweep.csv)): the AIC-best order is `(0,1,0)` — literally naive — while the Sharpe-best is `(1,1,0)`. A short, sharp post on why information criteria and out-of-sample trading metrics disagree, and which one to trust when the goal is a P&L.

**6. "Same model, different window, opposite verdict: a regime-fit horror story"**
The repo runs every model on Q1-only and Jan–Nov 2024 slices. XGBoost was leader on Q1 (dir_acc 0.5185), last on Jan–Nov (0.4872). The LSTM was *broken* on Q1 (Sharpe -0.54), competitive on Jan–Nov (+4.95). Use this to argue that single-split eval is a trap and that anything a paper claims about BTC depends on the test window.

**7. "An LSTM, a 24-bar window, and the question of how much past matters"**
Walkthrough of [04_lstm](../experiments/04_lstm/) using Darts' `BlockRNNModel`: why log-returns as the target, why past-covariates only, and the sweep finding that `input_chunk_length=24` (one day) fails while 48–96h works. The interesting bit: longer context didn't beat ARIMA's one lag — discuss what that says about where the signal lives.

**8. "Walk-forward without retraining: an honest cheat I keep using"**
Methods post on the evaluation harness — `statsmodels.apply(refit=False)` for ARIMA, `historical_forecasts(retrain=False)` for Darts. What's preserved (no leakage), what's sacrificed (no regime adaptation), and when the cheat stops being defensible.

**9. "Why this repo is named `transformer_bitcoin_ai` and still doesn't have a transformer"**
Roadmap post. Where the lineup goes next: ARIMAX with funding rate / perp basis, a transformer / TFT, then zero-shot foundation models (Chronos, TimesFM). Frame the LSTM result as the baseline the transformer has to clear and be honest about whether attention is even expected to help at this signal-to-noise level.

**10. (optional) "Stop annualizing your Sharpe"**
Short methodological piece prompted by the +7.28 Sharpe headlines. Per-bar Sharpe ≈ 0.080 with SE ≈ 0.025 on 1,608 bars — about 3σ from zero. The √8760 multiplier makes the number look like a hedge fund. Useful to write because every BTC ML post on the internet does this and never says it.
