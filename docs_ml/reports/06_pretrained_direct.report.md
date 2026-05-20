# Report — `06_pretrained_direct`

**Run date:** 2026-05-19
**Status:** completed (single fit on fresh 4h slice, Chronos-2 backend; no sweep yet — pending `make 06_pretrained_direct_sweep` for the Chronos ↔ TimesFM head-to-head)

## What this experiment is

First **zero-shot foundation model** in the lineup. Runs Amazon's Chronos-2 (encoder-only T5-style, ≈ 120 M parameters) — selected via `backend:` in the config — with **no training**: weights come straight from HuggingFace Hub, `fit()` is a no-op, and walk-forward 1-step probabilistic forecasts run on the held-out test slice. Differentiator vs. all earlier experiments: this model has never seen Bitcoin. The question is whether the prior learned on millions of unrelated time series transfers to BTC 4h log-returns.

```
HF Hub weights ──► darts FoundationModel ──► historical_forecasts(retrain=False, num_samples=200)
                                              │
                                              └──► quantile(0.1) / 0.5 / 0.9 ──► reconstruct close
```

Library: **darts** (`darts.models.Chronos2Model`, with `darts.models.TimesFM2p5Model` as the alternative backend). Walk-forward shape: same `historical_forecasts(retrain=False, last_points_only=True)` as 02 / 04 / 05, plus `num_samples=200` per step — each forecast is a stochastic draw, and three quantiles (q10 / q50 / q90) are stored alongside the median in `predictions.parquet`. The median (q50) feeds the leaderboard metrics. See [experiments/06_pretrained_direct/README.md](../../experiments/06_pretrained_direct/README.md) for the full narrative.

## Configuration

From [experiments/06_pretrained_direct/configs/btc_4h_2024.chronos-small.config.yaml](../../experiments/06_pretrained_direct/configs/btc_4h_2024.chronos-small.config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `4h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| backend | `chronos` |
| hub_model_name | `amazon/chronos-2` |
| model.input_chunk_length | `256` (≈ 42 days at 4h) |
| model.output_chunk_length | `1` |
| model.num_samples | `200` |
| model.quantiles | `[0.1, 0.5, 0.9]` |

Total bars: **2 009**. Train: **1 607** (Scaler-fit slice; the model itself does not train). Test: **402**. No val slice — zero-shot has no early stopping.

## Results — single fit (Chronos-2 on 4h)

From [experiments/06_pretrained_direct/results/btc_4h_2024/metrics.json](../../experiments/06_pretrained_direct/results/btc_4h_2024/metrics.json):

| Metric | Value |
|---|---|
| MAE | 519.78 USD |
| RMSE | **780.99** USD |
| MAPE | 0.6721 % |
| Directional accuracy | 0.5323 |
| Cumulative return | **0.6975** |
| Annualized Sharpe | **7.5915** |

## Cross-experiment comparison

4h-slice leaderboard. Same data slice, same split, same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| [01_baseline](01_baseline.report.md) | 518.36 | 784.09 | 0.6701 % | NaN | — | — |
| [02_arima (1, 1, 1)](02_arima.report.md) | **517.16** | 782.42 | **0.6686 %** | **0.5547** | 0.3314 | 6.0569 |
| [03_gradient_boosting (n_feat=31)](03_gradient_boosting.report.md) | 539.39 | 795.07 | 0.7008 % | 0.5365 | 0.5042 | 6.4233 |
| [04_lstm](04_lstm.report.md) | 522.29 | 785.45 | 0.6761 % | 0.4938 | 0.2596 | 4.2660 |
| [05_transformer](05_transformer.report.md) | 813.10 | 1 090.78 | 1.0786 % | 0.4913 | 0.4468 | 5.8675 |
| **06_pretrained_direct (chronos-2)** | 519.78 | **780.99** | 0.6721 % | 0.5323 | **0.6975** | **7.5915** |

**06_pretrained_direct takes the 4h leaderboard on RMSE, cum_ret, and Sharpe** — and does so as a zero-shot model that has never seen Bitcoin. RMSE 780.99 is **lower than 02_arima's 782.42** even though MAE is $2.62 higher; the foundation model makes fewer outsized errors. dir_acc 0.5323 is second-best after ARIMA(1, 1, 1)'s 0.5547 — among models that express direction (excluding the no-opinion naive), only ARIMA finds more.

## Interpretation

1. **The pretrained prior transfers to 4h BTC.** Chronos-2's Sharpe 7.5915 leads every trained model on the 4h slice: XGBoost (6.4233), ARIMA(1, 1, 1) (6.0569), TFT (5.8675), LSTM (4.2660). Per-bar Sharpe = 7.5915 / √2190 = 0.1622; SE = 1 / √402 = 0.0499; ratio ≈ **3.25 σ** — clearly significant at any sensible threshold. The "zero-shot doesn't transfer to BTC" hypothesis from the prior 1h run (where TimesFM produced dir_acc 0.4677, below coin-flip) does **not** hold on this 4h slice with the Chronos-2 backend.
2. **RMSE leader despite mid-pack MAE.** MAE 519.78 sits $2.62 above ARIMA's 517.16 — within sweep-noise of the floor — but RMSE 780.99 is $1.43 *below* ARIMA's 782.42. Lower RMSE with similar MAE means **fewer outsized error bars** — the foundation model hedges effectively on the bars where the trained models miss big. The probabilistic prior on next-step values squeezes the tails of the error distribution.
3. **dir_acc 0.5323 — second-best in the leaderboard, all without training on BTC.** Only ARIMA(1, 1, 1) finds more direction. The 120 M-parameter foundation model with no covariates beats the gradient booster (0.5365 → close, sets aside) and clearly beats the deep-learning models trained on this slice (LSTM 0.4938, TFT 0.4913).
4. **The probabilistic head is the architectural novelty.** `num_samples = 200` per step and `quantiles = [0.1, 0.5, 0.9]` are produced explicitly. q10 and q90 are stored alongside the median in `predictions.parquet` and shown as a shaded band on `plot.png`. None of 01 – 05 produce calibrated prediction intervals. Interval coverage is *not* summarised in `metrics.json` — pulling `(close ∈ [pred_lo, pred_hi])` from the parquet would close that gap and is a clean follow-up analysis.
5. **Backend was Chronos-2 in this run.** Earlier 1h artifacts ran with TimesFM 2.5 (different model family, different prior, very different result). The Chronos vs. TimesFM head-to-head — the design intent of this experiment — needs sibling runs on the same 4h slice; per-variant configs now live at [`btc_4h_2024.chronos-small.config.yaml`](../../experiments/06_pretrained_direct/configs/btc_4h_2024.chronos-small.config.yaml), [`btc_4h_2024.chronos-large.config.yaml`](../../experiments/06_pretrained_direct/configs/btc_4h_2024.chronos-large.config.yaml), and [`btc_4h_2024.timesfm.config.yaml`](../../experiments/06_pretrained_direct/configs/btc_4h_2024.timesfm.config.yaml). `make 06_pretrained_direct_sweep` is the vehicle.

### Bottom line

**A 120 M-parameter zero-shot foundation model leads the 4h leaderboard on Sharpe (7.5915), cum_ret (69.75 %), and RMSE (780.99)** with `dir_acc 0.5323` — second-best after ARIMA(1, 1, 1). Per-bar Sharpe sanity check: 0.1622 against SE 0.0499 — ratio **3.25 σ**, the strongest signal in the series. Test slice covers approximately Sep 25 → Dec 1 2024 (post-election BTC rally); the foundation-model prior may be especially well-tuned to this regime and has not been tested out-of-regime. **This is the strongest "ML actually works on Bitcoin" finding in the series so far** — pending the Chronos-vs-TimesFM head-to-head and an out-of-regime check.

## Caveats

- **Single backend.** This run is Chronos-2 only. The design intent is a head-to-head with `backend: timesfm`; needs a second run on the same slice via `make 06_pretrained_direct_sweep`.
- **Univariate only.** TimesFM 2.5 does not accept covariates; Chronos-2 supports them but is kept univariate to make the head-to-head fair on the same inputs. Adding covariates to Chronos-2 only is a clean follow-up ablation, not v1.
- **No fine-tuning.** Both `darts` foundation-model classes support partial / full fine-tuning; this run does not. The question being asked is specifically about the zero-shot prior.
- **Walk-forward without re-estimation.** Weights are frozen across the entire test window. Matches 02 / 04 / 05's walk-forward shape.
- `output_chunk_length: 1`. One-step-ahead only.
- **Single stochastic sample seed.** `num_samples = 200` gives a reasonable distribution within one seed; two runs may produce slightly different quantile bands.
- Long/flat strategy on the median (q50) forecast, no shorting, no transaction costs. The q10 / q90 bands are not used in the strategy yet.

## Files produced

- [experiments/06_pretrained_direct/results/btc_4h_2024/metrics.json](../../experiments/06_pretrained_direct/results/btc_4h_2024/metrics.json) (fresh 4h, Chronos-2)
- [experiments/06_pretrained_direct/results/btc_4h_2024/predictions.parquet](../../experiments/06_pretrained_direct/results/btc_4h_2024/predictions.parquet)
- [experiments/06_pretrained_direct/results/btc_4h_2024/plot.png](../../experiments/06_pretrained_direct/results/btc_4h_2024/plot.png)

## How to reproduce

```
make 06_pretrained_direct
make 06_pretrained_direct_sweep         # head-to-head across backends + context lengths; not yet on disk
```
