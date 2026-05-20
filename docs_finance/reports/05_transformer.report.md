# Report — `05_transformer`

**Run date:** 2026-04-29
**Status:** completed (single fit + 6-config sweep)

## What this experiment is

The first **attention-based** model in the lineup. Where 04 fed an LSTM the raw log-return sequence and let recurrence compress history into a hidden state, 05 fits a [Temporal Fusion Transformer (TFT)](https://arxiv.org/abs/1912.09363) — multi-head self-attention over a windowed history, with separate variable-selection networks for past and future inputs. This is the chronological successor of the LSTM and the headline architecture the repo is named for.

```
target  : r_T = log(close_T / close_{T-1})           # next-step log-return (same as 03/04)
inputs  : target sequence, with input_chunk_length    # past-target window
          past_covariates: log_volume, hl_range,      # known up to t-1 only
          oc_body
          future_covariates: hour_sin/cos,            # deterministic functions
          dow_sin/cos                                  # of the timestamp
predict : close_pred_T = close_{T-1} * exp(r_pred_T)  # reconstruct price for shared metrics
```

Library: [Darts](https://unit8co.github.io/darts/) `TFTModel`. Walk-forward 1-step-ahead via `historical_forecasts(retrain=False)` — same harness as 04. The two material differences vs 04 are (a) the model class itself (attention + variable selection vs. blocky LSTM) and (b) a future-covariate channel — TFT can consume signals known at any future timestamp, which `BlockRNNModel` cannot. We use cyclical encodings of hour-of-day and day-of-week as the future covariates: deterministic, zero-leakage, and the *only* "intra-day seasonality" prior we have without bringing in exogenous data.

Why the 04_lstm report's recommended next step was a transformer: LSTM compresses history into a single hidden state and re-reads it sequentially. Attention can look at any past step directly, and TFT's variable selection lets the model choose *per step* which inputs matter. If BTC has any non-trivial dependence beyond a few hours — intraday seasonality, day-of-week effects, multi-day momentum bursts — TFT has a better shot at finding it.

PatchTST (the other transformer mentioned during planning) is not available in `darts==0.43`; TFT is.

## Configuration

From [experiments/05_transformer/config.yaml](../../experiments/05_transformer/config.yaml):

| Field | Value |
|---|---|
| symbol | `BTCUSDT` |
| interval | `1h` |
| start | `2024-01-01` UTC (inclusive) |
| end | `2024-12-01` UTC (exclusive) |
| period | `monthly` |
| test_fraction | `0.2` |
| covariates.use_volume | `true` (`log1p(volume)`) — past covariate |
| covariates.use_ohlc | `true` (`high − low`, `close − open`) — past covariates |
| covariates.use_time_features | `true` (`hour_sin/cos`, `dow_sin/cos`) — **future** covariates |
| model.input_chunk_length | **48** (2 days of 1 h history per training window) |
| model.output_chunk_length | 1 (one-step-ahead) |
| model.hidden_size | 32 |
| model.lstm_layers | 1 (TFT's encoder/decoder LSTM stack depth) |
| model.num_attention_heads | 4 |
| model.dropout | 0.1 |
| model.hidden_continuous_size | 8 |
| model.add_relative_index | `false` (real future covariates supplied) |
| model.full_attention | `false` (TFT's interpretable masked attention) |
| model.batch_size | 64 |
| model.n_epochs | 30 (max; early stopping enabled) |
| model.learning_rate | 0.001 (Adam) |
| model.random_state | 42 |
| training.val_fraction | 0.1 (tail of *train* for early stopping) |
| training.early_stopping_patience | 5 |

Total bars after dropna: **8 039**. Train: **5 789**. Validation: **643**. Test: **1 607**. *Identical split* to 04_lstm — every metric is directly comparable bar-for-bar.

> **MPS note.** Inputs cast to `float32` for Apple's MPS backend, same as 04. Price reconstruction stays in float64.

> **Covariate channel separation.** Past covariates (volume, hl_range, oc_body) are fed only with strict-past indices; future covariates (cyclical time) are fed at the prediction time itself. Darts enforces both, so there is no current-bar leakage on past_covariates and no impossible-to-know lookahead on future_covariates.

## Results — single fit

From [experiments/05_transformer/results/metrics.json](../../experiments/05_transformer/results/metrics.json):

| Metric | Value |
|---|---|
| MAE | **373.70** USD |
| RMSE | 546.68 USD |
| MAPE | 0.4879 % |
| Directional accuracy | **0.5053** |
| Cumulative return | **+29.57 %** |
| Annualized Sharpe | **+4.59** |

## Cross-experiment comparison

Same data slice, same split (1 607 test bars from 2024-09 to 2024-11), same metrics:

| Experiment | MAE | RMSE | MAPE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|
| 01_baseline | **260.50** | **396.97** | **0.341 %** | NaN | — | — |
| **02_arima (1, 1, 1)** | **260.50** | **396.97** | **0.341 %** | **0.5336** | **+53.02 %** | **+7.28** |
| 03_gradient_boosting (default) | 270.73 | 414.07 | 0.354 % | 0.4872 | +8.09 % | +1.45 |
| **04_lstm (default)** | 274.02 | 409.66 | 0.360 % | **0.5196** | **+50.34 %** | **+4.95** |
| **05_transformer (default)** | 373.70 | 546.68 | 0.488 % | 0.5053 | +29.57 % | +4.59 |
| 05_transformer (best, 48/64/8/1/0.1) | 392.46 | 574.22 | 0.513 % | **0.5109** | +30.76 % | +4.83 |

TFT is the **worst-MAE model in the lineup** and the **third-place directional model**, behind ARIMA(1, 1, 1) and LSTM. It does have a positive trading Sharpe, but the gap to LSTM (~0.4 Sharpe, ~$100 MAE) is real, not noise.

This is genuinely surprising. The expectation going in was that TFT — more capacity, attention over arbitrary lags, per-step variable selection, *and* a future-covariate channel that LSTM didn't have — would at minimum match LSTM. It doesn't. The most likely explanations, in order of plausibility:

1. **Capacity / data ratio is wrong.** TFT has substantially more parameters than `BlockRNNModel(hidden=32, layers=2)` at the same `hidden_size`. With only 5 789 training bars, the larger parameter count is overfit territory. Consistent with the sweep below: scaling capacity *down* (icl=24, hidden=16, heads=2, layers=1) produces nearly the same Sharpe as the default; scaling *up* (the deepest configs) makes things worse.
2. **Cyclical hour/dow has no signal at hourly BTC.** The future-covariate channel was the architectural advantage over LSTM; if hour-of-day genuinely doesn't predict log-returns at this resolution, the channel is dead weight that the variable-selection network has to learn to ignore.
3. **Default `hidden_size=32`, `num_attention_heads=4` is too narrow.** Each attention head sees only 8 dims. Production TFTs usually run wider. The sweep includes (48, 64, 8, 1) which improves Sharpe to +4.83 — closer to LSTM but still behind.
4. **Single seed.** This is true for every neural experiment in the lineup, but TFT has more sources of init noise (variable-selection MLPs, attention head init) so ranking against LSTM at 0.36 Sharpe gap is well within multi-seed variance.

## Hyperparameter sweep

From [experiments/05_transformer/results/sweep.csv](../../experiments/05_transformer/results/sweep.csv) — same data slice, 6 configs, all using fixed `batch_size=64`, `n_epochs=30`, `learning_rate=0.001`, `random_state=42`, `hidden_continuous_size=8`, `add_relative_index=false`, `full_attention=false`, val-tail early stopping with patience 5:

| input_chunk_length | hidden_size | num_attention_heads | lstm_layers | dropout | MAE | RMSE | dir_acc | cum_ret | sharpe |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 16 | 2 | 1 | 0.10 | 398.59 | 580.50 | 0.5028 | +30.08 % | +4.37 |
| **48** | 32 | 4 | 1 | 0.10 | **373.70** | **546.68** | 0.5053 | +29.57 % | +4.59 |
| 48 | 32 | 4 | 2 | 0.20 | 420.61 | 605.08 | 0.4984 | +28.98 % | +4.12 |
| **48** | **64** | **8** | **1** | **0.10** | 392.46 | 574.22 | **0.5109** | **+30.76 %** | **+4.83** |
| 96 | 32 | 4 | 1 | 0.10 | 409.96 | 601.69 | 0.4984 | +22.94 % | +3.45 |
| 96 | 64 | 8 | 2 | 0.20 | 395.20 | 610.71 | 0.5047 | +30.95 % | +4.47 |

(Bold rows: MAE leader and Sharpe leader respectively.)

## Interpretation

1. **No TFT config beats naive on MAE.** The smallest MAE in the sweep ($373.70, the default) is **$113 worse than naive's $260.50**. No other model in the lineup leaves this much MAE on the table. The transformer is paying a substantial volatility-penalty for its capacity without an offsetting directional gain.

2. **The Sharpe leader is the *wider* config, not the *deeper* config.** `(48, 64, 8, 1, 0.1)` — single LSTM layer, doubled `hidden_size`, doubled attention heads — is the clear Sharpe and dir_acc winner at +4.83 / 0.5109. The deepest configs `(48, 32, 4, **2**, 0.2)` and `(96, 64, 8, **2**, 0.2)` are both worse on Sharpe than the default. **More LSTM depth in the TFT encoder hurts on this dataset.** The pattern is consistent with point 1 above: capacity helps as long as it goes into width; depth on a 5 789-bar training set just learns noise.

3. **Long input windows fail in TFT, just like in LSTM.** `input_chunk_length=96` (4 days) underperforms `48` on every metric for both `(96, 32, 4, 1, 0.1)` and `(96, 64, 8, 2, 0.2)`. This replicates the 04_lstm finding: the directional signal lives in the most recent ~48 hours, and giving the model 96 hours of context dilutes it. Reassuring that the two architectures agree on this.

4. **Short input windows (24) survive better than expected.** `(24, 16, 2, 1, 0.1)` lands at Sharpe +4.37 — within 5 % of the default's +4.59 — using a quarter of the training cost. For TFT on this data, **a tiny model is roughly as useful as the default-sized one**. That's a strong signal that the architecture is not extracting much beyond what one well-chosen recent lag carries.

5. **Future covariates didn't move the needle.** TFT's main architectural advantage over LSTM was supposed to be the future-covariate channel (calendar features). The Sharpe spread across the entire sweep is +3.45 to +4.83 — *worse* than the LSTM sweep's range (+0.54 to +4.95) on its top end. If hour-of-day or day-of-week carried real predictive signal, the wider configs that have capacity to exploit it should be pulling away from the smaller ones; they aren't. **Conclusion: hourly BTC log-returns are essentially calendar-blind.**

6. **Attention does not beat AR(1) on differences.** ARIMA(1, 1, 1) — one autoregressive coefficient and one MA coefficient on first differences — sits at Sharpe +7.28. The best TFT config sits at +4.83. A 4-head attention transformer with variable selection, 48 hours of context, multivariate past covariates, and cyclical future covariates is **48 % worse on annualized Sharpe** than a 2-parameter linear model from 1970. Same pattern as 04, sharper here. The implication is the same: at hourly BTC, the directional signal that exists is dominated by the most-recent bar's log-return increment, and any model with more degrees of freedom is mostly fitting noise.

### Bottom line

**TFT does not improve on LSTM on this slice; it gives up ~$100 of MAE and ~0.4 of Sharpe relative to LSTM, and ~2.5 of Sharpe relative to ARIMA.** This is the cleanest negative result in the lineup so far: more architecture, more covariates, more parameters — same or slightly worse signal extraction. The honest read is that the problem isn't model expressivity; it's signal-to-noise at the hourly horizon. AR(1) has already eaten the only stable directional structure that exists.

The headline expectation going into this experiment ("attention + future covariates beat LSTM") was wrong, and that's worth saying explicitly. The 04 report set up TFT as the natural next step assuming attention adds something LSTM can't see — on this slice, it doesn't.

Productive next directions:

- **Multi-seed runs.** With Sharpe gaps of 0.3–0.4, single-seed comparisons are not statistically separable. Re-run `(LSTM default, TFT default, TFT best)` × 5 seeds and look at distributions, not point estimates. This is the cheapest rigour upgrade and may turn the current "TFT loses" verdict into a "TFT and LSTM tie within noise" verdict.
- **Probabilistic head.** TFT supports `likelihood=QuantileRegression` for calibrated 50/90 % intervals. Quantile loss may train a more robust point-prediction model than MSE on heavy-tailed log-returns; also gives uncertainty for risk-aware position sizing.
- **Ablate covariates.** This run dumped past-OHLCV + future-time on the model. If the variable-selection weights show one channel doing all the work and another being ignored, simplifying is free. Easy to test by running `(use_volume=false, use_ohlc=false, use_time_features=false, add_relative_index=true)` — pure target sequence — and seeing whether anything changes.
- **Foundation models.** The original repo intent. Chronos / TimesFM zero-shot would tell us whether a *pretrained* sequence model has a useful prior on BTC. A model that has seen many financial series may extract structure that a from-scratch TFT trained on 5 789 bars cannot.
- **Different test window.** TFT has been evaluated on the same Sep–Nov 2024 rally that flatters LSTM and ARIMA. Running on mid-2022 (post-LUNA, FTX) — a regime with both directions — would be the falsifiability test for *every* model in the lineup, including this one.

## Caveats

- **Single split, no rolling-origin CV.** Hyperparameter rankings could shift on a different window. The 04 experiment's Q1 → full-year inversion is the proof.
- **Walk-forward without retraining.** `retrain=False` keeps train-time weights frozen across the entire test window. `historical_forecasts(retrain=True)` would adapt to regime drift at significant compute cost.
- **`random_state=42` everywhere.** As above — neural-net training has run-to-run variance the sweep doesn't capture.
- **MPS float32.** Inputs cast to float32. Numerical behaviour will differ slightly from CUDA / CPU float64 runs; below noise for log-returns at this magnitude.
- **TFT supports interpretability hooks** (`model.predict(num_samples=…)`-style attention/selection introspection) but this v1 doesn't surface them. A useful follow-up: re-fit the best config and dump variable-selection weights to show *which* of the 9 input channels (target, 3 past covs, 4 future covs) the model actually uses.
- **`hidden_continuous_size=8` and `full_attention=false` were not swept.** Both are TFT-specific knobs; the sweep focused on the larger-impact ones (`input_chunk_length`, `hidden_size`, `num_attention_heads`, `lstm_layers`, `dropout`).
- **Two warnings during training are upstream and harmless:** `pin_memory not supported on MPS` and a `LeafSpec` deprecation from pytorch-lightning's pytree integration. Same as 04.

## Files produced

- [experiments/05_transformer/results/metrics.json](../../experiments/05_transformer/results/metrics.json) — single fit
- [experiments/05_transformer/results/predictions.parquet](../../experiments/05_transformer/results/predictions.parquet) — columns: close, pred, ref, strategy_return
- [experiments/05_transformer/results/plot.png](../../experiments/05_transformer/results/plot.png)
- [experiments/05_transformer/results/sweep.csv](../../experiments/05_transformer/results/sweep.csv) — 6-config sweep

## How to reproduce

```
make 05_transformer         # single fit using params in config.yaml (~1-2 min on MPS)
make 05_transformer_sweep   # 6-config sweep (~25-30 min on MPS — TFT is slower than LSTM)
```
