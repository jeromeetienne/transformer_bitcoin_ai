# Predicting Bitcoin with machine learning: what this series is about

This is the first of six articles on machine-learning models applied to Bitcoin price forecasting. The subject is *not* Bitcoin and it is *not* trading — it is what happens to seven model classes when you put them on the same 4-hour-bar slice and grade them with the same metric module. Naive last-value. ARIMA. XGBoost. LSTM. Temporal Fusion Transformer. Two zero-shot foundation models. Same data, same split, same numbers in `results/metrics.json`.

The order of the ladder is the point. The articles that follow walk up it one rung at a time and look at where capacity helps, where it stalls, and where it actively goes backwards. There is a surprise on the way up and I would rather state it now than save it for article 4: **a 3-parameter ARIMA(3, 1, 3) beats the Temporal Fusion Transformer, the LSTM, and zero-shot Chronos-2 on point error on this slice.** Six of seven models cluster inside an MAE band of $539–$551. The transformer sits at $891. The series exists to explain why.

If you want to skip ahead, [the repo is here](https://github.com/jetienne/transformer_bitcoin_ai) and `make help` lists every target. The rest of this article frames the question, walks through the data and the evaluation, and previews articles 1–5.

## The question, restated as a machine-learning problem

The lab task is one-step-ahead forecasting of the BTCUSDT close price. Every model in the repo predicts the same target, in the same units, on the same test bars. The internal target is the log-return of the next bar:

```
r_T  = log(close_T / close_{T-1})
```

Forecasts are produced in log-return space and reconstructed back to price as `close_pred = close_{T-1} * exp(r_pred)` so that MAE, RMSE, MAPE, directional accuracy, cumulative return, and annualized Sharpe are all denominated in dollars or fractions — directly comparable across the linear, classical-machine-learning, deep-learning, and foundation-model rows of the leaderboard.

Two early framing choices worth flagging:

- **Log-returns, not prices.** Log-returns are roughly stationary, scale-invariant, and additive across time. Predicting the price level directly invites every model to "predict tomorrow ≈ today" and call it a day — which is exactly what the naive baseline does, and is exactly the thing we want every later model to beat without cheating.
- **One step ahead, not multi-horizon.** ARIMA, the XGBoost regressor, and the foundation models in zero-shot configuration all natively produce one-step forecasts; the Darts sequence models can multi-step but are pinned to `output_chunk_length = 1` to keep the playing field flat.

This is a forecasting problem, framed for clean comparison across model classes. It is not a trading strategy. The cumulative-return and Sharpe figures that show up in `metrics.json` exist because they are useful as *direction-aware* extensions of MAE — a long-flat rule on top of the model's forecast surfaces whether the model is right about sign as well as right about magnitude. That is all they are for in this series. Nothing in the articles will say "deploy this."

## The data

`BTCUSDT` 4-hour bars from [Binance Vision](https://data.binance.vision), pulled by [`scripts/fetch_data.py`](scripts/fetch_data.py) and cached under `data/raw/`. The slice is defined once, in YAML, and re-used by every experiment:

```yaml
# configs/datasets/btc_4h_2024.dataset.yaml
training:
  - market: spot
    symbol: BTCUSDT
    interval: 4h
    period: monthly
    start: 2023-01-01      # UTC, inclusive
    end:   2024-08-01      # UTC, exclusive
validation:
  - market: spot
    symbol: BTCUSDT
    interval: 4h
    period: monthly
    start: 2024-08-01
    end:   2024-10-01
test:
  - market: spot
    symbol: BTCUSDT
    interval: 4h
    period: monthly
    start: 2024-10-01
    end:   2024-12-01
```

Three slices, half-open in UTC, no overlap. **Test = 366 bars at 4h** — every model in this series reports on those same 366 bars. Train is 19 months for the per-experiment configs (a bit over 3 800 bars). Validation is two months and gets consumed only by the deep-learning models for early stopping; the linear baseline, ARIMA, and XGBoost don't use it.

There is a regime caveat that belongs up front. The test window — `2024-10-01` UTC → `2024-12-01` UTC — covers the post-election Bitcoin rally. Every positive directional-skill or Sharpe number in the series is conditional on a strongly trending up-regime, with no shorting and no second test window. Read these as "skill *conditional on* this regime," not "skill in general." The point of the series is the *ranking* of models on a common slice, not the absolute Sharpes.

## Walk-forward, one step ahead, weights frozen

Every model on the ladder is evaluated the same way:

1. Fit once on the train slice. The deep-learning models also see the validation slice for early stopping.
2. Walk forward one bar at a time across the 366 test bars, producing a one-step-ahead forecast at each bar from the *actual* history up to that point.
3. **Weights stay frozen** across the test window. No re-fitting, no parameter updates between test bars. ARIMA uses `fit.apply(full_series, refit=False)`; the Darts sequence models use `historical_forecasts(retrain=False, last_points_only=True)`; the foundation models do not train at all.

The walk-forward shape matters more than it looks. With frozen weights, the model has to make its forecast at bar `t` using only what was known before bar `t` — which is also the only honest way to grade a forecaster. With refitting between bars the picture would be optimistic by an amount that is impossible to bound from outside. The article 1 baselines establish this protocol; every later article inherits it.

Metrics are computed by a single shared module — [`src/btc_ai/eval/metrics.py`](src/btc_ai/eval/metrics.py) — so a number called "MAE" means the same thing in every `metrics.json` in the repo. The metric set is:

- **MAE / RMSE / MAPE.** Point error in USD and percent.
- **Directional accuracy.** Fraction of test bars where `sign(pred − ref)` equals `sign(true − ref)`. Excludes bars where either side is zero — explained below.
- **Cumulative return** and **annualized Sharpe** of a long-flat rule: long for the next bar when `pred > ref`, flat otherwise. No shorting, no transaction costs, no slippage. This is the simplest possible direction-aware extension of point error.

Three of those metrics report `NaN` on the naive baseline, and that is by design rather than a bug. When the naive predictor returns `close_pred[t] = close[t-1]`, the predicted direction is always zero, so the directional-accuracy mask is empty and the function short-circuits to `NaN` ([src/btc_ai/eval/metrics.py:23-34](src/btc_ai/eval/metrics.py)). The strategy never goes long, so cumulative return is empty and annualized Sharpe degenerates. `NaN` here means *"the model expressed no opinion."* Future articles will see one model in the ARIMA sweep produce the same `NaN` pair — the `(0, 1, 0)` order is the random walk, which *is* the naive baseline up to floating point — and one LSTM sweep row that hits `NaN` Sharpe because the strategy went flat for every test bar. These are useful failure modes to recognize on a leaderboard; mistaking them for missing runs is easy.

## The spoiler

Six models clustered at MAE 539–551 USD on 366 test bars. One outlier at 891. The headline number from the global report:

| Model | MAE (USD) | Directional accuracy | Annualized Sharpe |
|---|---|---|---|
| Naive last-value | 540.96 | NaN | — |
| ARIMA(3, 1, 3) | **539.15** | 0.5082 | 6.8559 |
| XGBoost (31 features) | 550.85 | 0.5082 | 6.1388 |
| LSTM (Darts BlockRNN) | 539.23 | 0.5301 | 4.8221 |
| Temporal Fusion Transformer | 891.09 | 0.5055 | 2.5390 |
| Chronos-2 small (zero-shot) | 546.66 | 0.4754 | 2.9722 |

Same 366 test bars on every row. Same metric module. The 3-parameter ARIMA leads on point error, beats every later trained model on Sharpe (6.86 vs. LSTM's 4.82 vs. the transformer's 2.54), and is within $11 of the naive baseline that has zero parameters. The Temporal Fusion Transformer is *65 % worse on MAE than the naive predictor.* The zero-shot foundation model with 28 million pretrained parameters is below coin-flip on direction.

This is not the result I was hoping to publish when I started the repo. It is more interesting than the result I was hoping to publish. The series walks through, in turn, the data slice and metric scaffolding (article 1), the only model class that actually clears the floor on Sharpe besides ARIMA (article 2, XGBoost), the first deep-learning model that adds capacity and goes backwards (article 3, LSTM), the named-after-it-in-the-repo Temporal Fusion Transformer (article 4) — and finally a model that has never seen Bitcoin and forms an opinion anyway (article 5, zero-shot Chronos-2 and TimesFM 2.5). The honest story is "where capacity helps and where it hurts," and the answer changes class by class.

## Walking the repo

The repository is organized so that every experiment is self-contained and every comparison is apples-to-apples:

```
transformer_bitcoin_ai/
├── Makefile                    # canonical command surface — every target wraps `uv run`
├── pyproject.toml + uv.lock    # dependencies, managed by uv
├── configs/
│   └── datasets/               # one YAML per data slice
├── src/btc_ai/                 # shared library
│   ├── config.py               # YAML schema loaders
│   ├── data/                   # Binance Vision loader + on-disk cache
│   └── eval/metrics.py         # the single source of truth for all metrics
├── experiments/
│   ├── 01_baseline/            # naive last-value
│   ├── 02_arima/               # ARIMA(p, d, q) + sweep
│   ├── 03_xgboost/             # XGBoost on engineered features
│   ├── 04_lstm/                # Darts BlockRNN LSTM
│   ├── 05_transformer/         # Darts Temporal Fusion Transformer
│   └── 06_pretrained/          # zero-shot Chronos-2 + TimesFM 2.5
└── scripts/
    └── fetch_data.py           # pre-warm the data cache
```

Three conventions that the articles will keep referring back to:

- **One folder per experiment, with its own `run.py`, `config.yaml`, and `results/`.** New experiments never edit a previous one. The numbered prefix preserves chronology and makes `results/` line up as a leaderboard.
- **YAML is the single source of truth.** A model's parameters and the data slice both live in YAML. `run.py` consumes the config; so does `scripts/fetch_data.py`. Re-running an experiment is a one-line Make target with the YAML path interpolated.
- **Make is the canonical command surface.** Every target wraps `uv run`. You will never see `python experiments/04_lstm/run.py` in this series; you will see `make 04_lstm`. The shared library, the YAML, and the Make target combine to mean that "reproducing article 3" is `make 04_lstm`, end of instructions.

## What each article will do

The five model articles share a structure: hook, setup recap, the model in plain English, what happened (with the numbers from the experiment's `metrics.json` quoted verbatim), what it tells us about that model class, and a one-line `make` command to reproduce. None of them will hide a loss.

- **Article 1 — Baselines you need to beat.** Naive last-value and ARIMA. Establishes the data slice, the walk-forward protocol, and the metric module in code-level detail. The ARIMA sweep over 12 orders is where the linear-versus-deep-learning theme of the series first surfaces.
- **Article 2 — XGBoost and feature engineering.** 31 engineered features — lagged log-returns, rolling moments, a per-bar OHLCV summary — fed into a non-linear gradient-boosted tree ensemble. The classical-machine-learning article; the contract reverses for everything that follows.
- **Article 3 — LSTM.** First deep-learning model. A stacked recurrent network on a window of past bars. The first time on the ladder where added capacity goes backwards on a trading metric.
- **Article 4 — Transformer.** Darts' Temporal Fusion Transformer. Attention, variable-selection networks, future covariates. The model the repo is named after. Honest comparison to ARIMA included.
- **Article 5 — Zero-shot foundation models.** Chronos-2 (28 M and 120 M) and TimesFM 2.5 (200 M) on Bitcoin without ever being shown Bitcoin during pretraining. Probabilistic forecasts via Q10 / Q50 / Q90 bands.

The articles can be read in order, or independently — each model article opens with a one-paragraph methodology recap. The leaderboard rows accumulate as the series progresses; by article 5 it is complete.

## Reproduce

The whole series is reproducible from a clean clone:

```
git clone https://github.com/jetienne/transformer_bitcoin_ai
cd transformer_bitcoin_ai
uv sync               # creates .venv from pyproject.toml + uv.lock
make fetch            # pre-warm the data cache
make help             # list every experiment target
```

Then run any model article's experiment with a one-liner — `make 01_baseline`, `make 02_arima`, …, `make 06_pretrained`. Each target writes `experiments/NN_*/results/btc_4h_2024/metrics.json`, the file from which every number in the next five articles is quoted.

The articles are commentary. The code is the artefact. Article 1 starts at the floor.
