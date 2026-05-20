# Report — `06_pretrained`

**Run date:** 2026-05-20
**Status:** completed (canonical trial: `btc_4h_2024.chronos-small`; two siblings on disk for the backend × size head-to-head)

## What this experiment is

First **zero-shot foundation model** in the lineup. Runs Amazon's Chronos-2 (encoder-only T5-style; `autogluon/chronos-2-small` is 28 M parameters, `amazon/chronos-2` is 120 M) or Google's TimesFM 2.5 (decoder-only patch-transformer, 200 M parameters), selected via `backend:` and `hub_model_name:` in the per-variant config — with **no training**: weights come straight from HuggingFace Hub, `fit()` is a no-op, and walk-forward 1-step probabilistic forecasts run on the held-out test slice. Differentiator vs. all earlier experiments: the model has never seen Bitcoin. The question is whether the prior learned on millions of unrelated time series transfers to BTC 4h log-returns.

```
HF Hub weights ──► darts FoundationModel ──► historical_forecasts(retrain=False, num_samples=200)
                                              │
                                              └──► quantile(0.1) / 0.5 / 0.9 ──► reconstruct close
```

Library: **darts** (`darts.models.Chronos2Model`, `darts.models.TimesFM2p5Model`). Walk-forward shape: `historical_forecasts(retrain=False, last_points_only=True)` matches 02 / 04 / 05, plus `num_samples = 200` per step — each forecast is a stochastic draw, and three quantiles (q10 / q50 / q90) are stored alongside the median in `predictions.parquet`. The median (q50) feeds the leaderboard metrics. See [experiments/06_pretrained/README.md](../../experiments/06_pretrained/README.md) for the full narrative.

## Configuration

From [experiments/06_pretrained/configs/btc_4h_2024.chronos-small.config.yaml](../../experiments/06_pretrained/configs/btc_4h_2024.chronos-small.config.yaml) (the canonical trial; matches the per-experiment Makefile's default `CONFIG ?=`):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `4h` |
| dataset | `btc_4h_2024` ([configs/datasets/btc_4h_2024.dataset.yaml](../../configs/datasets/btc_4h_2024.dataset.yaml)) |
| train window | `2023-01-01` → `2024-08-01` UTC |
| val window | `2024-08-01` → `2024-10-01` UTC (unused — zero-shot has no early stopping) |
| test window | `2024-10-01` → `2024-12-01` UTC |
| backend | `chronos` |
| hub_model_name | `autogluon/chronos-2-small` (28 M parameters) |
| model.input_chunk_length | `256` (≈ 42 days at 4h) |
| model.output_chunk_length | `1` |
| model.num_samples | `200` |
| model.quantiles | `[0.1, 0.5, 0.9]` |

Total bars: **4 199**. Scaler-fit slice ("train" — the model itself does not train): **3 833**. Test: **366**.

## Results — single fit (Chronos-2 small, zero-shot)

From [experiments/06_pretrained/results/btc_4h_2024.chronos-small/metrics.json](../../experiments/06_pretrained/results/btc_4h_2024.chronos-small/metrics.json):

| Metric | Value |
|---|---|
| MAE | 546.66 USD |
| RMSE | 817.81 USD |
| MAPE | 0.7003 % |
| Directional accuracy | 0.4754 (below coin-flip — see Interpretation) |
| Cumulative return | 0.2225 |
| Annualized Sharpe | 2.9722 |

## Cross-experiment comparison

4h-slice leaderboard. Same `btc_4h_2024` data slice, same `2024-10-01` → `2024-12-01` test window, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| [01_baseline](01_baseline.report.md) | 540.96 | 813.14 | 0.6922 % | NaN | — | — |
| [02_arima (3, 1, 3)](02_arima.report.md) | **539.15** | 808.79 | 0.6903 % | 0.5082 | **0.5269** | **6.8559** |
| [03_xgboost (n_feat=31)](03_xgboost.report.md) | 550.85 | 808.65 | 0.7096 % | 0.5082 | 0.4149 | 6.1388 |
| [04_lstm](04_lstm.report.md) | 539.23 | 808.29 | **0.6909 %** | **0.5301** | 0.4284 | 4.8221 |
| [05_transformer](05_transformer.report.md) | 891.09 | 1 271.26 | 1.1499 % | 0.5055 | 0.1219 | 2.5390 |
| **06_pretrained (chronos-2 small)** | **546.66** | **817.81** | **0.7003 %** | **0.4754** | **0.2225** | **2.9722** |

The zero-shot prior does **not** transfer at chronos-2-small. dir_acc 0.4754 is below coin-flip — the small Chronos checkpoint actively *misreads* direction on this slice — and Sharpe 2.97 trails every other ML model except the TFT. MAE 546.66 sits $7.51 above ARIMA(3, 1, 3)'s 539.15 (within sweep-noise) but RMSE 817.81 is the worst in the leaderboard ex-TFT. The leaderboards in older reports (e.g. [06_pretrained's prior run](#), [05_transformer.report.md](05_transformer.report.md)) cite a different slice (rows_test ≈ 401) and an earlier ARIMA order (1, 1, 1) — they are snapshots and have not been re-edited; only this report's table reflects the current `metrics.json` files. See the Variants section below for the timesfm head-to-head, which tells a different story.

## Variants

The experiment ships three variants in `experiments/06_pretrained/configs/`, all on the same 4h test window. The Makefile default pins the small variant for sanity-check speed; the headline result lives in `timesfm`:

| Variant | hub_model_name | params | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|---|
| `btc_4h_2024.chronos-small` (canonical) | `autogluon/chronos-2-small` | 28 M | **546.66** | **817.81** | **0.7003 %** | 0.4754 | 0.2225 | 2.9722 |
| `btc_4h_2024.chronos-large` | `amazon/chronos-2` | 120 M | 560.59 | 837.11 | 0.7159 % | 0.4344 | 0.1116 | 1.9004 |
| `btc_4h_2024.timesfm` | `google/timesfm-2.5-200m-pytorch` | 200 M | 570.79 | 843.75 | 0.7309 % | **0.4863** | **0.3164** | **4.1080** |

The variant axis is **backend × parameter count**. Three observations:

- **Bigger Chronos is worse, not better.** The 120 M chronos-large is uniformly behind the 28 M chronos-small across every metric (MAE +13.93, dir_acc −0.041, Sharpe −1.07). The extra 92 M parameters of pretrained TS prior actively hurt on this slice. The small variant is not just a sanity-check checkpoint — it's the better Chronos for this data.
- **TimesFM wins the trading metrics by a margin large enough to matter.** Sharpe 4.108 vs. chronos-small's 2.972 is a 38 % relative lift; cum_ret 0.316 vs. 0.223 is a 42 % lift. dir_acc 0.4863 is still below coin-flip but it's the highest among the three. The decoder-only TimesFM prior is meaningfully better-tuned for BTC than either Chronos-2 checkpoint at this size and context length.
- **All three lose to ARIMA(3, 1, 3) on every metric.** Even the best variant (timesfm, Sharpe 4.11) trails the linear baseline (Sharpe 6.86). The pretrained prior alone is not enough to beat a 7-parameter AR / MA model on this slice. That is the experiment's honest answer to its motivating question.

A backend rotation isn't justified — timesfm wins trading metrics, but the small chronos checkpoint wins point error, and the experiment's design intent is the head-to-head itself (not picking a winner). Leaving the canonical pinned at chronos-small keeps the per-experiment headline aligned with the smallest, fastest variant; the timesfm result is loud enough in this Variants table.

## Interpretation

1. **dir_acc 0.4754 on the canonical row is below coin-flip.** On `366` test bars, chronos-small's median forecast points the wrong way 191 times out of 366. Sharpe 2.97 is positive only because the strategy is long/flat (no shorts) and the test slice's drift was strongly up — the strategy collects the drift on the bars it correctly stays long, and only forgoes profit on the bars it goes flat. A symmetric long/short rule on chronos-small's median would lose money on this slice.
2. **RMSE 817.81 is now worst-ex-TFT.** In the earlier 1h slice and earlier 4h run, the foundation model's probabilistic head squeezed RMSE below ARIMA's. Not anymore. With 366 test bars (a much narrower window than the prior 401-row slice) the foundation model's hedging shows up less — fewer chances to average down the outliers — and ARIMA / XGBoost / LSTM all post RMSE below 810.
3. **Bigger Chronos is uniformly worse — and that's not within noise.** chronos-large trails chronos-small by $13.93 on MAE, 0.041 on dir_acc, and 1.07 on Sharpe. The same architecture family, same context length, same num_samples — only the parameter count changes. On near-random-walk financial data the extra capacity adds variance to the predictive distribution without adding signal. This is consistent with the "complexity-doesn't-pay" arc the global report tracks.
4. **TimesFM's better Sharpe is bigger samples + decoder-only prior.** Same `num_samples = 200` and same `input_chunk_length = 256`, but the decoder-only patch-transformer family has a different inductive bias from Chronos's encoder-only T5. Sharpe 4.108 / per-bar `0.0878` against `SE = 0.0523` gives a `1.68 σ` ratio — interesting but not significant. Treat the 38 % Sharpe lift as a meaningful direction signal, not a publication number.
5. **NaN dir_acc on naive is by design.** [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py) returns NaN when the predicted direction is constant (the naive baseline always predicts no-change). The foundation models all express direction — even when they get it wrong — so they get a finite dir_acc.

### Bottom line

**Zero-shot transfer is real but weak at chronos-small** — Sharpe 2.97 is positive only thanks to the strategy's long/flat asymmetry on an uptrending slice. Per-bar Sharpe `0.0635` against `SE = 0.0523` → `1.21 σ`, **not** significant. The TimesFM variant lifts Sharpe to 4.11 (`1.68 σ`) and cum_ret to 0.32 — still not significant, still trailing ARIMA. Test slice covers `2024-10-01` → `2024-12-01` (post-election BTC rally, 366 bars at 4h). **No variant of zero-shot beats the linear baseline on this slice;** the question of whether *fine-tuning* closes that gap is answered in [07_finetuned](07_finetuned.report.md).

## Caveats

- **Three variants, one slice.** The backend × parameter-count cell is filled in; an `input_chunk_length` sweep is not yet on disk. `make 06_pretrained_sweep` was the original vehicle for that and has not been re-run on the current slice.
- **Univariate only.** TimesFM 2.5 does not accept covariates; Chronos-2 supports them but is kept univariate so the head-to-head stays fair on identical inputs. Adding covariates to Chronos-2 only is a clean follow-up ablation.
- **No fine-tuning.** Zero-shot is the question this experiment is asking; fine-tuning is [07_finetuned](07_finetuned.report.md).
- **Walk-forward without re-estimation.** Weights frozen across the entire test window. Matches 02 / 04 / 05's walk-forward shape.
- `output_chunk_length: 1`. One-step-ahead only.
- **Single stochastic seed per variant.** `num_samples = 200` per step gives a reasonable predictive distribution within one seed, but two runs may produce slightly different quantile bands. Multi-seed runs would tighten the per-variant numbers (07's headline relies on 5-seed comparison; 06 does not).
- Long/flat strategy on the median (q50) forecast, no shorting, no transaction costs. q10 / q90 bands are stored but not used in the strategy.
- **Single test regime.** The test slice is the post-election BTC rally; no variant has been tested out-of-regime.

## Files produced

- [experiments/06_pretrained/results/btc_4h_2024.chronos-small/metrics.json](../../experiments/06_pretrained/results/btc_4h_2024.chronos-small/metrics.json) (canonical)
- [experiments/06_pretrained/results/btc_4h_2024.chronos-small/predictions.parquet](../../experiments/06_pretrained/results/btc_4h_2024.chronos-small/predictions.parquet)
- [experiments/06_pretrained/results/btc_4h_2024.chronos-small/plot.png](../../experiments/06_pretrained/results/btc_4h_2024.chronos-small/plot.png)
- Variant artefacts (same set per variant): [`btc_4h_2024.chronos-large/`](../../experiments/06_pretrained/results/btc_4h_2024.chronos-large/), [`btc_4h_2024.timesfm/`](../../experiments/06_pretrained/results/btc_4h_2024.timesfm/)

## How to reproduce

```
make 06_pretrained                                                                         # canonical: chronos-small
make 06_pretrained CONFIG=experiments/06_pretrained/configs/btc_4h_2024.chronos-large.config.yaml
make 06_pretrained CONFIG=experiments/06_pretrained/configs/btc_4h_2024.timesfm.config.yaml
make 06_pretrained_sweep                                                                   # backend × context-length sweep (not yet on disk for current slice)
```

(The `CONFIG=` override is required for any non-canonical variant.)
