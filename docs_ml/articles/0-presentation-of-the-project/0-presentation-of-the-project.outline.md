# Outline — Article 0: Presentation of the project

Working title: *"Predicting Bitcoin with machine learning: what this series is about."*

The curtain-raiser. Frames the question, the data, the evaluation philosophy, and the repo. No model, no results — but a deliberate spoiler of the punchline so the rest of the series can land honestly.

## Beats

1. **Hook — one paragraph.** "Bitcoin price prediction" is the lab problem; the actual subject is how seven model classes behave on the same 4h-bar slice. The order is on purpose: zero-parameter naive → ARIMA → XGBoost → LSTM → Temporal Fusion Transformer → zero-shot foundation models. The ladder is the point.

2. **The question, restated as an ML problem.**
   - Target: one-step-ahead log-return `r_T = log(close_T / close_{T-1})`, reconstructed to price as `close_pred = close_{T-1} * exp(r_pred)`.
   - Why log-returns and not price levels: stationarity, scale invariance, comparable across models.
   - Why one-step-ahead and not multi-horizon: keeps the playing field flat between models that natively predict one step (ARIMA, XGBoost regressor) and ones that can do multi-step.

3. **The data slice.**
   - `BTCUSDT` 4-hour bars from Binance Vision, calendar 2024 (through end of November) for the per-model train slices; a longer 2023-01-01 → 2024-08-01 train window for the foundation models.
   - Train / validation / test split: `2023-01-01` → `2024-08-01` (train), `2024-08-01` → `2024-10-01` (validation, used for early stopping by the deep-learning models), `2024-10-01` → `2024-12-01` (test). Test = **366 bars** at 4h, all in UTC, half-open intervals.
   - The regime caveat up front: the test window is the post-election BTC rally. Every directional-skill claim in the series is conditional on that.

4. **Evaluation philosophy.** What "honest" means in this series:
   - Walk-forward 1-step-ahead, weights frozen at train time (`retrain=False`).
   - Same metric set across every model: MAE, RMSE, MAPE, directional accuracy, cumulative return, annualized Sharpe. Single source of truth in [src/btc_ai/eval/metrics.py](src/btc_ai/eval/metrics.py).
   - Three NaN values are kept on purpose — they communicate "model expressed no opinion." Don't fill them, don't hide them.

5. **The spoiler.** A 3-parameter ARIMA(3, 1, 3) beats the Temporal Fusion Transformer, the LSTM, and zero-shot Chronos-2 on point error. **MAE band 539–550 USD on six of the seven models** — the linear baseline is at the bottom of that band and the deep learning models do not earn the top. Headline: complexity does not pay on this signal-to-noise ratio. The series is structured around explaining *why*.

6. **The repo, end to end.**
   - One folder per experiment under `experiments/NN_*/`, each with `run.py`, `config.yaml`, `results/`.
   - YAML is the single source of truth for the data slice and model parameters.
   - All commands go through the Makefile.
   - `uv` for env, deps, and runs; `uv.lock` committed.
   - Shared library at `src/btc_ai/` so the metric module, the data loader, and the config schema are not re-implemented per experiment.

7. **Preview of articles 1–5.** One sentence each:
   - Article 1: 01_baseline + 02_arima. The floor, plus the linear AR / MA model that ties it on MAE and beats every later model on point error.
   - Article 2: 03_xgboost. 31 engineered features. The model is generic, the features carry the signal.
   - Article 3: 04_lstm. The first deep learning model. Sequence input. What it learns and does not learn.
   - Article 4: 05_transformer. The Temporal Fusion Transformer from Darts. Attention plus future covariates. The named model of the repo.
   - Article 5: 06_pretrained. Zero-shot Chronos-2 and TimesFM 2.5. What it means that a model that has never seen Bitcoin has an opinion.

8. **Reproduce.** `git clone …` + `uv sync` + `make help`. Code is the canonical artefact; articles are commentary.

## Open knobs settled in this article

- Spoiler policy: **spoil the ARIMA-beats-deep-learning punchline in article 0.** The series sells itself on the surprise; hiding it for four articles flattens the hook.
- Personal-credibility paragraph: **skip.** The editorial voice is code-first; the repo is the credibility.
- Methodology split: framing here, specifics in article 1.

## What this article is NOT

- A finance article. No Sharpe-as-a-strategy framing, no trading advice.
- A "transformers are the future" article.
- A "this is hard so I give up" article. The series shows what works, what doesn't, and what the gap tells us.

## Reproduce
- Repo: `transformer_bitcoin_ai`
- `make help` lists every experiment target.
