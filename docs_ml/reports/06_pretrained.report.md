# Report — `06_pretrained`

**Run date:** 2026-04-29
**Status:** completed (single fit, no sweep yet) — **on stale 1h slice; pending re-run on 4h**

> The `config.yaml` for this experiment now declares `interval: 4h` and `backend: chronos` as the default selection (matching commit 81d4fcb's kline switch), but the `metrics.json` artefact in this report was last regenerated on **2026-04-29** against the previous 1h slice, with `backend = timesfm` (`google/timesfm-2.5-200m-pytorch`). Total bars 8 039 / test 1 608 corresponds to 1h, not 4h. All numbers below are honest reports of the artefact on disk; they are **not** comparable to the 4h leaderboard in 01–04's reports.

## What this experiment is

First **zero-shot foundation model** in the lineup. Runs Google's TimesFM 2.5 (decoder-only patch-transformer, 200 M parameters) — selected via `backend:` in the config — with **no training**: weights come straight from HuggingFace Hub, `fit()` is a no-op, and walk-forward 1-step probabilistic forecasts run on the held-out test slice. The differentiator vs. all earlier experiments: this model has never seen Bitcoin. It is asked whether the prior learned on millions of unrelated time series transfers to BTC log-returns.

```
HF Hub weights ──► darts FoundationModel ──► historical_forecasts(retrain=False, num_samples=200)
                                              │
                                              └──► quantile(0.1) / 0.5 / 0.9 ──► reconstruct close
```

Library: **darts** (`darts.models.TimesFM2p5Model`, with `darts.models.Chronos2Model` available as the alternative backend). Walk-forward shape: same `historical_forecasts(retrain=False, last_points_only=True)` as 04 / 05, plus `num_samples=200` per step — each forecast is a stochastic draw, and three quantiles (q10 / q50 / q90) are stored alongside the median in `predictions.parquet`. The median (q50) feeds the leaderboard metrics. See [experiments/06_pretrained/README.md](../../experiments/06_pretrained/README.md) for the full narrative.

## Configuration

From [experiments/06_pretrained/config.yaml](../../experiments/06_pretrained/config.yaml) (with the actual backend used in the stale run shown second):

| Field | Value (config) | Stale-artefact value |
|---|---|---|
| symbol | `BTCUSDT` | `BTCUSDT` |
| interval | `4h` | **1h** |
| start | `2024-01-01` UTC | `2024-01-01` UTC |
| end | `2024-12-01` UTC | `2024-12-01` UTC |
| period | `monthly` | `monthly` |
| test_fraction | `0.2` | `0.2` |
| backend | `chronos` | **`timesfm`** |
| hub_model_name | `amazon/chronos-2` | `google/timesfm-2.5-200m-pytorch` |
| model.input_chunk_length | `256` | `256` |
| model.output_chunk_length | `1` | `1` |
| model.num_samples | `200` | `200` |
| model.quantiles | `[0.1, 0.5, 0.9]` | `[0.1, 0.5, 0.9]` |

Total bars on the stale 1h artefact: **8 039**. Train: **6 431** (Scaler-fit slice; the model itself does not train). Test: **1 608**. No val slice — zero-shot has no early stopping.

## Results — single fit (TimesFM 2.5 on stale 1h slice)

From [experiments/06_pretrained/results/metrics.json](../../experiments/06_pretrained/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | 268.89 USD |
| RMSE | 405.14 USD |
| MAPE | 0.3521 % |
| Directional accuracy | **0.4677** (below coin-flip) |
| Cumulative return | 0.1690 |
| Annualized Sharpe | 2.4444 |

The MAE 268.89 is the lowest in any report in this batch — but that is on the 1h slice with 256 bars of input context, not because the prior outperforms a trained model. The honest comparison is the 1h-only leaderboard below.

## Cross-experiment comparison

1h-slice leaderboard. Same data slice, same split, same metrics. Only experiments still on the stale 1h slice appear here.

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| [05_transformer](05_transformer.report.md) | 373.70 | 546.68 | 0.4879 % | **0.5053** | **0.2957** | **4.5864** |
| **06_pretrained (timesfm)** | **268.89** | **405.14** | **0.3521 %** | 0.4677 | 0.1690 | 2.4444 |

The split is informative: a trained TFT with covariates leads the trading metrics (dir_acc, cum_ret, Sharpe); a zero-shot 200 M-parameter foundation model leads point error (MAE, RMSE, MAPE) by committing less to a direction. (Experiments 01–04 are on the fresh 4h slice; their reports contain a separate leaderboard.)

## Interpretation

1. **The zero-shot prior did not transfer to BTC direction.** dir_acc 0.4677 is **below coin-flip** — the foundation model has a small but real anti-direction bias on this slice. A model trained on millions of unrelated series has *some* opinion about what comes next, and on BTC 1h log-returns that opinion is mildly wrong about direction more often than right.
2. **Low MAE here is a confidence story, not a skill story.** TimesFM hedges. Its median forecast moves less from the reference than 05_transformer's does, so the long/flat rule (`pred > ref → long`) triggers less often and the bars where it *is* long contribute less to MAE. The model "doesn't say much" → "isn't wrong by much on average." That is the trade-off behind the 268.89 figure.
3. **Sharpe 2.4444 is at the edge of the noise.** Per-bar Sharpe ≈ 2.4444 / √8760 = 0.0261; standard error ≈ 1 / √1608 = 0.0249; ratio ≈ **1.05 σ**. Statistically indistinguishable from zero edge.
4. **The probabilistic head is the architectural novelty of this experiment.** `num_samples = 200` and `quantiles = [0.1, 0.5, 0.9]` are produced explicitly in the run — the median feeds the leaderboard metrics above, q10 and q90 are stored in `predictions.parquet` and shown as a shaded band on `plot.png`. None of 01–05 produce calibrated prediction intervals; 06 is the only row in the series that does. The interval calibration is not summarised in `metrics.json` and is not analysed in this report; pulling `(close ∈ [pred_lo, pred_hi])` from the parquet would close that gap.
5. **The config's default `backend: chronos` was not the backend that produced these numbers.** The stale `metrics.json` records `backend = timesfm` with `hub_model_name = google/timesfm-2.5-200m-pytorch`. On the next re-run (4h slice, Chronos default) the entire row of headline numbers will change, and the Chronos-vs-TimesFM head-to-head — explicitly part of this experiment's design — will need a re-sweep to surface.

### Bottom line

A 200 M-parameter generic foundation model **does not produce a directional edge on 1h BTC log-returns zero-shot**. dir_acc 0.4677 sits below coin-flip; Sharpe 2.4444 is 1.05 σ above zero against a 1 608-bar standard error — statistically inseparable from no skill. The model's MAE is the lowest in the report batch, but that reflects forecast caution (hedging toward the reference price), not better calibration. Test slice covers approximately Sep 25 → Dec 1 2024 (post-election BTC rally). **The most ML-credible finding in the series** is that scaling a generic time-series prior to 200 M parameters does not, in this case, beat a 3-parameter ARIMA — and re-running on 4h with the Chronos backend is the obvious next step, both to make the row comparable to 02 / 03 / 04 and to surface the Chronos vs. TimesFM head-to-head that the experiment was designed to expose.

## Caveats

- **Stale 1h slice.** Re-run `make 06_pretrained` after confirming the config's `interval: 4h`.
- **Single backend in this artefact.** `metrics.json` records `backend = timesfm`; the design intent is a head-to-head with `backend = chronos`. Needs a second run on the same slice to compare.
- **Univariate only.** TimesFM 2.5 does not accept covariates at all; Chronos-2 supports them but is kept univariate in the config so the head-to-head is on the same inputs. Adding covariates to Chronos-2 is a clean follow-up ablation, not v1.
- **No fine-tuning.** Both `darts` foundation-model classes support partial / full fine-tuning; this run does not. The question being asked is specifically about the zero-shot prior.
- **Walk-forward without re-estimation.** Weights are frozen across the entire test window. Matches 02 / 04 / 05's walk-forward shape.
- `output_chunk_length: 1`. One-step-ahead only.
- **Single stochastic sample seed.** `num_samples = 200` gives a reasonable distribution within one seed; two runs may produce slightly different quantile bands.
- Long/flat strategy on the median forecast, no shorting, no transaction costs. The q10 / q90 bands are not used in the strategy yet.

## Files produced

- [experiments/06_pretrained/results/metrics.json](../../experiments/06_pretrained/results/metrics.json) (**stale 1h, TimesFM backend**)
- [experiments/06_pretrained/results/predictions.parquet](../../experiments/06_pretrained/results/predictions.parquet) (**stale 1h**)
- [experiments/06_pretrained/results/plot.png](../../experiments/06_pretrained/results/plot.png) (**stale 1h**)

## How to reproduce

```
make 06_pretrained
make 06_pretrained_sweep         # head-to-head across backends + context lengths; not yet on disk
```
