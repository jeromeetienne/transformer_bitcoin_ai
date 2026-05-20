# Attention on a price series — and what it changes (or doesn't)

The repository this series ships from is called `transformer_bitcoin_ai`. This article is about the experiment the repository is named after: the Darts Temporal Fusion Transformer on `BTCUSDT` 4-hour bars. It is also, by some margin, the experiment that loses the hardest in the entire series. **Mean absolute error 891.09 USD — sixty-five percent worse than naive last-value, fifty-one percent worse than every other trained model on the leaderboard.** Directional accuracy 0.5055 — essentially coin-flip. Annualized Sharpe 2.5390 — *still positive*, but only barely above the standard-error threshold, on a slice that flatters every long-flat strategy. Three answers, one model.

There is a thing the series is *not* arguing. It is not arguing that transformers do not work on time series. The Temporal Fusion Transformer was introduced by Bryan Lim and colleagues with strong results on multi-series benchmarks — M5 retail, electricity load, traffic. The literature is real. What this article is about is what happens when an architecture of that richness — about thirty thousand parameters across variable-selection networks, an LSTM encoder / decoder, multi-head self-attention, gated residual networks, and a future-covariate channel — meets one single price series with 3 467 training rows. That ratio is the entire story.

## Methodology, in one paragraph

Target: one-step-ahead log-return `r_T = log(close_T / close_{T-1})`, reconstructed back to price as `close_pred = close_{T-1} * exp(r_pred)`. Data: `BTCUSDT` 4-hour bars from Binance Vision. Split: train `2023-01-01` → `2024-08-01` UTC (3 467 rows after the 48-bar window and validation trim), validation `2024-08-01` → `2024-10-01` UTC (366 rows for early stopping), test `2024-10-01` → `2024-12-01` UTC (**366 bars**). Walk-forward, one step ahead, weights frozen across the test window. Metrics from the shared module — [`src/btc_ai/eval/metrics.py`](src/btc_ai/eval/metrics.py). Reproduce: `make 05_transformer`. **New this article:** future covariates — calendar-cyclical encodings of hour-of-day and day-of-week that the Temporal Fusion Transformer can legitimately consume at prediction time because they are deterministic functions of the timestamp.

## What the model is

The Temporal Fusion Transformer is a stack of components, each doing a specific job:

- **Variable selection networks** that learn, per time step, which of the input variables matter for the prediction. A gating layer suppresses irrelevant features dynamically rather than averaging over them.
- **LSTM encoder / decoder.** Yes, the Temporal Fusion Transformer has an LSTM inside it. The encoder summarizes the past window; the decoder primes the model for the prediction step.
- **Multi-head self-attention** over the encoded sequence. Long-range dependencies across the input window are addressed directly, rather than compressed into a single hidden state.
- **Gated residual networks** between the major components for feature mixing and skip connections. The gating controls how much each transformation contributes — a learnable "do I want this?" per step.
- **Past covariates + future covariates + static metadata.** The model keeps these as three separate channels rather than merging them, because a value observed at prediction time (the calendar timestamp of the next bar) carries different information than a value last observed before the prediction (the previous bar's volume).

Library: `darts.models.TFTModel`, running on PyTorch / PyTorch Lightning. Counted naively, about thirty thousand trainable parameters — three times the LSTM in article 3 with a much richer set of inductive biases. The configuration:

```yaml
# experiments/05_transformer/configs/btc_4h_2024.config.yaml — model section
model:
  input_chunk_length: 48      # 8 days at 4h
  output_chunk_length: 1
  hidden_size: 32
  lstm_layers: 1
  num_attention_heads: 4
  dropout: 0.1
  hidden_continuous_size: 8
  add_relative_index: false
  full_attention: false
  batch_size: 64
  n_epochs: 30                # early-stopped on val_loss
  learning_rate: 0.001
  random_state: 42

covariates:
  use_volume: true
  use_ohlc: true
  use_time_features: true     # ← new: hour_sin, hour_cos, dow_sin, dow_cos
```

Past covariates (`log_volume`, `hl_range`, `oc_body`) are the same three the LSTM in article 3 saw. The future covariates — sine and cosine encodings of hour-of-day and day-of-week — are new. They give the Temporal Fusion Transformer something to put through the future-covariate channel that the LSTM did not have access to.

## The walk-forward shape

```python
preds = model.historical_forecasts(
    series=close_series,
    past_covariates=past_cov,
    future_covariates=future_cov,
    start=test_start,
    forecast_horizon=1,
    retrain=False,
    last_points_only=True,
)
```

Same harness as article 3's LSTM — `historical_forecasts(retrain=False, last_points_only=True)`. The model is trained once on the training slice with early stopping on the validation slice, and then walks forward through the 366 test bars with its weights frozen. The single architectural difference vs. article 3 is the model class and the future-covariates argument. Everything else is bit-identical.

## The numbers

From [`experiments/05_transformer/results/btc_4h_2024/metrics.json`](experiments/05_transformer/results/btc_4h_2024/metrics.json):

| Metric | Temporal Fusion Transformer |
|---|---|
| MAE | **891.09 USD** |
| RMSE | 1271.26 USD |
| MAPE | 1.1499 % |
| Directional accuracy | 0.5055 |
| Cumulative return | 0.1219 |
| Annualized Sharpe | 2.5390 |

Side-by-side with the rest of the leaderboard so far:

| Model | MAE | dir_acc | Sharpe |
|---|---|---|---|
| Naive last-value | 540.96 | NaN | — |
| ARIMA(3, 1, 3) | **539.15** | 0.5082 | **6.8559** |
| XGBoost (31 features) | 550.85 | 0.5082 | 6.1388 |
| LSTM | 539.23 | **0.5301** | 4.8221 |
| **Temporal Fusion Transformer** | **891.09** | 0.5055 | 2.5390 |

The Temporal Fusion Transformer's MAE is the worst in the leaderboard. By a lot. It is sixty-five percent above the naive baseline. It is fifty-one percent above the LSTM. It is the same architectural family as the LSTM with additional machinery on top, and that machinery cost $352 per bar in mean absolute error. The directional accuracy is essentially coin-flip; the cumulative return is the lowest non-naive row in the leaderboard.

That is the result. The article is about why.

## Why a 65 % MAE penalty

The numerator is 30 000 parameters. The denominator is 3 467 training rows. The ratio is about *9 training rows per parameter* on a target — one-step-ahead log-returns on a heavy-tailed financial series — that is dominated by noise. Three components of the Temporal Fusion Transformer's architecture are absorbing capacity that the data does not have signal to constrain:

- **Variable selection networks.** With four to seven candidate variables per time step (past covariates plus future calendar features), the network is learning per-step gates over a roughly random feature ranking. The training slice is small enough that the gate weights fit slice-specific noise; the test slice rolls a different noise realization and the gates are wrong.
- **Multi-head self-attention.** Four heads × thirty-two hidden dimensions × forty-eight input steps gives the model a large bilinear lookup table from any input step to any other. On a real long-range time series (electricity load, traffic) those interactions encode periodicity. On a heavy-tailed nearly-random-walk series, they encode whatever lag pairs happened to be co-extreme in the training slice — noise that does not generalize.
- **Gated residual networks.** Every place the architecture has a gate, the gate's weights are being fit to the training slice. With this many gates and this little data, the gates encode training-slice specifics rather than general structure.

The future covariates didn't help. Calendar-cyclical features encode intraday and intraweek seasonality. On 4-hour Bitcoin during the post-election rally, there is essentially no exploitable seasonal structure — the price ran roughly continuously upward for two months. The future-covariates channel adds capacity without adding signal.

The Temporal Fusion Transformer is *the right model for a different data regime*. Bryan Lim's original results were on M5 retail (3 049 stores × multiple products × multiple horizons), on electricity load (370 customers, hourly), on traffic data (963 sensors, hourly). What those datasets share is *cross-series transfer*: the variable-selection networks, attention, and gated residuals get to learn structure from many related series, and that structure transfers. On *one* `BTCUSDT` price series, there is no cross-series transfer; the model has to learn everything from 3 467 rows; and the architectural richness becomes a liability rather than an advantage.

## Sharpe-positive, but only because the slice is

Sharpe 2.5390 looks like a positive number. A Sharpe-positive long-flat strategy on top of a Sharpe-positive trained model is the kind of thing people frame as "the model works." The per-bar Sharpe significance back-of-envelope is the right place to push back on that. Per-bar Sharpe = 2.5390 / √2190 = 0.0543. Standard error on the per-bar mean ≈ 1 / √366 = 0.0523. Ratio ≈ **1.04 σ** — not significant. Under the same i.i.d. assumption that gave ARIMA(3, 1, 3) 2.80 σ and the LSTM 1.97 σ, the Temporal Fusion Transformer's Sharpe is consistent with no skill.

The mechanism that makes it positive at all is the regime tailwind. The long-flat strategy goes long whenever `pred > ref`. The Temporal Fusion Transformer's predictions are wrong-about-magnitude — wildly so, hence the MAE — but the magnitude errors are not symmetric. The model produces a fair share of strongly positive forecasts on a slice where the price is trending strongly up. So the strategy is long for a meaningful fraction of bars during a strongly trending up window and collects the drift on the up bars. It also collects very large losses on the down bars where it was wildly wrong, but the drift is bigger than the down-bar damage on this slice. Net positive return; positive Sharpe; not statistically distinguishable from zero.

This is the cleanest example in the series of why a positive Sharpe in isolation is *not* enough to call a model "good." The MAE row of the same metrics.json says the model is wrong by 891 USD per bar on average. The Sharpe row says the trading metric on top of the model is positive. Both numbers are true. The dir_acc 0.5055 sitting between them tells the rest of the story — coin-flip about direction, 65 % worse on magnitude than naive, drift-favourable strategy on a drift-favourable slice. The article 3 dir_acc-vs-Sharpe split was sharp; the article 4 MAE-vs-Sharpe split is sharper.

## The sweep, briefly

The sweep at [`experiments/05_transformer/results/btc_4h_2024/sweep.csv`](experiments/05_transformer/results/btc_4h_2024/sweep.csv) was last regenerated against the previous 1-hour data slice. Its MAE figures (range 373–420 USD) are on the 1-hour scale and *not* comparable to the 891.09 above. Rerunning is on the operational follow-up list. The structural lesson from the stale sweep is unchanged: hyperparameter rankings on financial time series are sensitive to slice, and the configured default is not guaranteed to be the sweep's Sharpe or MAE winner.

## What this article tells us about the model class

**Capacity is not a free variable on small datasets.** Thirty thousand parameters on 3 467 training rows is on the wrong side of the bias-variance tradeoff for a target that is dominated by noise. The variable-selection networks, the attention layers, and the gated residual networks fit training-slice specifics that do not generalize. The price is a 65 % MAE blowout and a Sharpe figure that is not statistically distinguishable from zero. This is not a bug in the Temporal Fusion Transformer. It is the wrong tool for this data regime.

**Attention does not inherently outperform recurrence on time series.** The LSTM in article 3 was richer than ARIMA and went backwards on Sharpe; the Temporal Fusion Transformer is richer than the LSTM and goes backwards on MAE too. Both lessons are consistent with the broader observation that *what the model is asked to predict from* matters more than *how much machinery sits between input and output*. The next article — zero-shot foundation models — pushes the parameter count up by three orders of magnitude (28 million pretrained parameters for Chronos-2 small) and finds the same ceiling. The ceiling is the data, not the model.

**A positive Sharpe with a worst-in-leaderboard MAE is a methodology lesson, not a result.** Reading any one column of `metrics.json` in isolation can flatter or damn a model that the full row of columns places in context. The series-wide protocol of reporting MAE, RMSE, MAPE, dir_acc, cum_ret, and Sharpe together — every model, every slice — exists exactly so that a row like this one cannot be presented as "the transformer worked." It didn't. The Sharpe was positive on a slice that is positive for almost everything.

## Regime caveat

The test slice is the post-election Bitcoin rally; nothing here has been tested out-of-regime; the 2.5 Sharpe is most generously read as "barely above noise on a slice that flatters." On a sideways slice the Sharpe would almost certainly invert.

## Reproduce

```
make 05_transformer
make 05_transformer_sweep   # stale on 1h; pending re-run on 4h
```

The first writes [`experiments/05_transformer/results/btc_4h_2024/metrics.json`](experiments/05_transformer/results/btc_4h_2024/metrics.json) — the numbers in this article. The second writes [`experiments/05_transformer/results/btc_4h_2024/sweep.csv`](experiments/05_transformer/results/btc_4h_2024/sweep.csv) — pending a rerun on the current 4-hour slice.

Article 5, the last in the series, is about models that have never seen Bitcoin and form an opinion anyway — zero-shot foundation models. Twenty-eight million pretrained parameters of Chronos-2, 120 million of the large Chronos checkpoint, 200 million of TimesFM 2.5. The question of whether "borrowing parameters" closes the gap article 4 just opened is the question article 5 answers. Onwards.
