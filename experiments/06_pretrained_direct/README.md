# 06_pretrained_direct

First **zero-shot foundation model** in the lineup. Runs either Amazon's [Chronos-2](https://huggingface.co/amazon/chronos-2) (encoder-only T5-style, 120M params) or Google's [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (decoder-only patch-transformer, 200M params) — selected by `backend:` in [`config.yaml`](config.yaml) — with **no training**: weights come straight from HuggingFace Hub, `fit()` is a no-op, and we walk-forward 1-step probabilistic forecasts on the held-out test slice.

```
HF Hub weights ──► darts FoundationModel ──► historical_forecasts(retrain=False, num_samples=200)
                                              │
                                              └──► quantile(0.1) / 0.5 / 0.9 ──► reconstruct close
```

The target is the same bar-T log-return `r_T = log(close_T / close_{T-1})` used by [`04_lstm`](../04_lstm/) and [`05_transformer`](../05_transformer/), and predictions are reconstructed to price as `close_pred = close_{T-1} * exp(r_q0.5)` so MAE / RMSE / MAPE / directional_accuracy / Sharpe are directly comparable to all earlier experiments.

## Why this experiment

The first five experiments all *train* on this exact slice of BTC. The honest follow-up is: **does a model that has never seen this data already have a useful prior?** Time-series foundation models (Chronos, TimesFM) are trained on millions of unrelated series and ship with a learned distribution over plausible next-step values. If that generic prior transfers to 1h BTC log-returns, it shows up here. If it doesn't — and the realistic outcome is MAE near the naive floor with `directional_accuracy ≈ 0.50` — that's still a clean data point: it tells us the signal is in the data, not the architecture, and that paying for a 120–200M-parameter pretrained model gets you nothing a one-line baseline doesn't already deliver.

A second value proposition that earlier experiments didn't have: **probabilistic forecasts**. Both backends accept `QuantileRegression([0.1, 0.5, 0.9])` and return a calibrated prediction interval, not just a point estimate. The median (q=0.5) feeds the existing leaderboard metrics; q10 and q90 are stored alongside in `predictions.parquet` and shown as a shaded band on `plot.png`.

## Why univariate

This experiment **does not use covariates** for either backend. Two reasons:

1. **TimesFM 2.5 doesn't support covariates at all.** It's univariate-only; the `darts` wrapper doesn't expose a covariate channel.
2. **Chronos-2 supports past + future covariates** (it's the same MixedCovariates surface as TFT in [`05_transformer`](../05_transformer/)), but we deliberately don't use them here. With one backend forced to univariate, giving the other backend extra inputs would muddy the head-to-head: any difference in metrics would conflate "which prior is stronger?" with "did the covariates help?". Keeping both univariate isolates the question this experiment is asking — *does the pretrained prior alone transfer to BTC?*

Adding covariates to Chronos-2 only is a clean follow-up ablation, but not v1.

## How it's done

1. Load OHLCV via the shared loader ([src/btc_ai/data](../../src/btc_ai/data)).
2. Build target `r_T = log(close_T / close_{T-1})` and `ref_T = close_{T-1}` for downstream price reconstruction. **Univariate only** — no past or future covariates are constructed.
3. Two-way time split: `[0:train_end]` for "train" (where the Scaler is fit; the model itself does not train), `[train_end:n]` for test. No val slice — there's no early stopping in zero-shot.
4. Fit a `darts.dataprocessing.transformers.Scaler` on the **training slice only**. Cast to `float32` so the model can run on Apple MPS or CUDA.
5. Build the model based on `cfg['backend']`:
   - `chronos` → `Chronos2Model(input_chunk_length, output_chunk_length=1, hub_model_name, likelihood=QuantileRegression([0.1, 0.5, 0.9]))`
   - `timesfm` → `TimesFM2p5Model(input_chunk_length, output_chunk_length=1, hub_model_name, likelihood=QuantileRegression([0.1, 0.5, 0.9]))`

   Both classes inherit from `darts.models.forecasting.foundation_model.FoundationModel`, so the rest of the pipeline is identical.
6. Call `model.fit(target_train_scaled)`. The darts API requires this; for foundation models it performs **no weight updates** — it's there to lock the schema.
7. Walk-forward predict on the test slice via `model.historical_forecasts(start=test_start, forecast_horizon=1, retrain=False, last_points_only=True, num_samples=200)`. Each step samples 200 futures from the model's predictive distribution.
8. Inverse-scale the stochastic series, then extract `quantile(0.1) / quantile(0.5) / quantile(0.9)` to get `r_q_lo / r_q_med / r_q_hi`.
9. Reconstruct prices: `close_pred = ref * exp(r_q_med)` for the point prediction (the median), plus `pred_lo / pred_hi` for the interval.
10. Compute MAE, RMSE, MAPE, directional accuracy, cumulative return, and annualized Sharpe on the **median** prediction via the shared metrics in [src/btc_ai/eval/metrics.py](../../src/btc_ai/eval/metrics.py).
11. Write `results/metrics.json`, `results/predictions.parquet` (with `close, pred, pred_lo, pred_hi, ref, strategy_return`), `results/plot.png` (close + median + shaded q10–q90 band).

Library: **[darts](https://unit8co.github.io/darts/)** (`darts.models.Chronos2Model`, `darts.models.TimesFM2p5Model`). The HuggingFace download (~120–200 MB depending on backend) is automatic on first run and cached in `~/.cache/huggingface/hub/`.

## How to run

```
make 06_pretrained_direct
```

Pick a backend in [`config.yaml`](config.yaml) — there is **no default**, you must set one:

```yaml
backend: chronos          # or 'timesfm'
hub_model_name: amazon/chronos-2
                          # chronos: amazon/chronos-2 | autogluon/chronos-2-small | autogluon/chronos-2-synth
                          # timesfm: google/timesfm-2.5-200m-pytorch
```

Tune the prediction without re-training (because there *is* no training):

```yaml
model:
  input_chunk_length: 256       # bars of history fed to the model
                                # (chronos-2 max=8192, timesfm-2.5 max=16384)
  output_chunk_length: 1        # one-step-ahead forecast
  num_samples: 200              # samples drawn from the predictive distribution
  quantiles: [0.1, 0.5, 0.9]    # subset of the model's pre-trained quantiles
```

First run downloads the HF weights (one-time, then cached). Walk-forward over ~1100 test bars takes single-digit minutes on Apple MPS or a recent CUDA GPU; CPU is several times slower.

## How to interpret

Compare against earlier experiments **on the same data slice**:

| | Naive | ARIMA(1,1,1) | XGBoost | LSTM | TFT | **Chronos-2 / TimesFM 2.5** |
|---|---|---|---|---|---|---|
| Family | last-value | linear, stationary | tabular GBT | recurrent NN | attention NN | **pretrained foundation model** |
| Trained on this slice | — | yes | yes | yes | yes | **no (zero-shot)** |
| Probabilistic | — | — | — | — | — | **yes (q10 / q50 / q90)** |
| Past covariates | — | — | engineered lags | OHLCV | OHLCV | — |
| Future covariates | — | — | — | — | hour/dow cyclical | — |

What the numbers tell you:

- **`mae` near the naive floor (~$260)** — the typical outcome on hourly BTC. Generic TS priors don't transfer; the model has no calibrated guess for this asset/timescale beyond "near the last value." This is consistent with the LSTM / TFT result and is genuinely informative: it means the bottleneck is the data, not the architecture or the parameter count.
- **`mae` clearly below the naive floor** — surprising. The pretrained model is finding structure that a fresh-from-scratch LSTM didn't. Worth replicating with a different `input_chunk_length` and the other backend before believing it.
- **`directional_accuracy` ≈ 0.50** — the prior is symmetric on direction; no transfer. Expected.
- **`directional_accuracy > 0.51`** — the foundation model has a faint up/down bias that pays out. Cross-check with the strategy column and the Sharpe; usually requires `input_chunk_length >= 256`.
- **`annualized_sharpe` close to ARIMA's `+7.52`** — the deep prior caught up on the metric that matters for trading, with zero training. That would be a real result.
- **`annualized_sharpe ≪ ARIMA's`** — the pretrained prior, at least off-the-shelf, is worse for *this asset* than a single AR(1) coefficient on differenced returns. Honest.

### Caveats baked into this v1

- **Univariate.** No past covariates (volume, OHLC range/body) and no future covariates (hour-of-day, day-of-week). TimesFM doesn't accept any; Chronos-2 does but we deliberately match. See [Why univariate](#why-univariate) above.
- **No fine-tuning.** Both `darts` classes support partial / full fine-tuning via `enable_finetuning`, but this experiment is the zero-shot question specifically. Fine-tuning is a separate experiment.
- **Walk-forward without re-estimation.** `retrain=False` keeps weights frozen across the entire test window. Matches [`02_arima`](../02_arima/) / [`04_lstm`](../04_lstm/) / [`05_transformer`](../05_transformer/).
- **`output_chunk_length: 1`.** One-step-ahead only. Multi-horizon is a one-line config change but changes the loss surface and the comparability with 01–05.
- **HuggingFace download on first run.** Network access is required the first time you run with a given `hub_model_name`; subsequent runs are offline.
- **`amazon/chronos-2` and `autogluon/chronos-2-synth` are 120M params; `autogluon/chronos-2-small` is 28M.** The small variant is faster and useful for sanity checks; the headline result should use the full model.
- **Single sampling seed.** `num_samples=200` gives a reasonable distribution but two runs may produce slightly different quantile bands. Average over a few seeds for a publication-grade number.

## Sweeping backends and context lengths

For comparing backends and context lengths on the same data slice without re-running the full experiment each time:

```
make 06_pretrained_direct_sweep
```

Edit the `GRID` list at the top of [`sweep.py`](sweep.py) to change which combos are tried. Output: a printed table to stdout (sorted in source order, easy to scan) and a `results/sweep.csv` for downstream analysis. The sweep does not touch `metrics.json` / `predictions.parquet` / `plot.png` — those reflect the single config in [`config.yaml`](config.yaml) set via `make 06_pretrained_direct`.

What to look at in the sweep:

- **`backend` × `input_chunk_length`** — does more context help? Foundation models are advertised as "the more history the better," but BTC log-returns are near-i.i.d. and the asymptotic gain often plateaus by 256.
- **MAE on the same test slice** — only number that says "this config predicts better than that one." On hourly BTC the differences are usually within a few dollars on a ~$300 baseline; treat sub-1% MAE deltas as noise.
- **`directional_accuracy` and `annualized_sharpe`** — the metrics that *trade*. They often rank configs differently from MAE because point error and direction-getting-right are not the same thing.
- **Chronos-2 small vs. large** — the small variant is ~4× faster. If both produce indistinguishable BTC metrics, that's itself a finding (the 120M params buy nothing extra on this signal).

## Files

```
experiments/06_pretrained_direct/
├── README.md           # this file
├── config.yaml         # data slice + test_fraction + backend + hub_model_name + model knobs
├── run.py              # entry point — fits the single config in config.yaml
├── sweep.py            # sweeps multiple (backend, hub_model_name, icl) combos, writes results/sweep.csv
└── results/
    ├── metrics.json        # produced by run.py
    ├── predictions.parquet # close + pred + pred_lo + pred_hi + ref + strategy_return on the test slice
    ├── plot.png            # produced by run.py — close + median + shaded q10–q90 band
    └── sweep.csv           # produced by sweep.py
```
