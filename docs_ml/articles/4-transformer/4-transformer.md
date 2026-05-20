# The Architecture That Took Over AI, Now Trying to Read Bitcoin's Mind

The LSTM solved one problem elegantly — sequential processing with gated memory — and created another: to make a prediction, it must compress the entire input window into a single fixed-size vector. For a 48-bar input and a 32-dimensional hidden state, the entire history of the last two days of Bitcoin trading has to fit in 32 numbers. If a bar 40 steps back contains a relevant signal — say, the start of a volatility cluster that is still playing out — the LSTM must have held that information alive through 39 forget-gate applications.

The attention mechanism, introduced in the 2017 "Attention Is All You Need" paper, replaces the sequential compression with a direct lookup. Instead of funneling history through a bottleneck, attention lets the model query every past timestep directly when forming the prediction. A bar 40 steps back is as accessible as a bar 1 step back. Whether that matters depends on whether the relevant signal is local or long-range.

This experiment applies attention to Bitcoin price forecasting via the Temporal Fusion Transformer — a purpose-built architecture for time series that adds several useful structures on top of basic attention.

## Why the Temporal Fusion Transformer

The Temporal Fusion Transformer, proposed by Google in 2019, is not a direct port of the NLP Transformer. It is designed specifically for multi-horizon time-series forecasting with mixed inputs, and it solves a set of problems that vanilla Transformers ignore:

**Variable selection networks** — at each timestep, a gating mechanism determines which input features are relevant. If volume is not useful at a particular step, it gets a low weight and contributes little to the representation. This is automatic feature selection within the forward pass.

**LSTM encoder and decoder** — TFT uses LSTM layers to process the input sequence before attention. This is not a contradiction; it means the model gets both the local sequential processing of recurrent networks and the global direct-access of attention. The LSTM encoder handles local patterns; the attention layer handles long-range ones.

**Multi-head self-attention** — the standard Transformer attention, applied to the encoder outputs. Multiple heads allow the model to attend to different aspects of the history simultaneously: one head might attend to high-volatility bars regardless of recency, while another attends to the immediately preceding bars.

**Gated residual networks** — skip connections with learned gating throughout the architecture. These allow the model to bypass components that are not useful for a given input, which is particularly helpful when some inputs (like time-of-day) are only relevant for certain types of predictions.

The result is a model with more moving parts than the LSTM, more potential for overfitting on small datasets, and more potential for capturing complex temporal structure if that structure exists.

## The new input channel: future covariates

The LSTM in article 4 consumed past covariates — volume and OHLC data that was available up to bar *T-1* when predicting bar *T*. TFT adds a second covariate channel: **future covariates** that are known deterministically at any future bar.

For Bitcoin price, the natural candidates are calendar features: the hour of the day and the day of the week. These are deterministic functions of the bar's timestamp — you know exactly what hour it will be 24 hours from now. If BTC has any intraday seasonality (more volatile during certain hours, trending at market opens in different time zones), TFT can exploit this signal while LSTM cannot.

The features are encoded as sine-cosine pairs to preserve their cyclical nature:

```python
hour_sin = sin(2π · hour / 24)
hour_cos = cos(2π · hour / 24)
dow_sin  = sin(2π · day_of_week / 7)
dow_cos  = cos(2π · day_of_week / 7)
```

The sine-cosine encoding ensures that hours 23 and 0 are close in the feature space (they are adjacent in the cycle), unlike a raw integer encoding where 23 and 0 are maximally far apart.

Whether Bitcoin actually has exploitable intraday seasonality at the hourly level is an empirical question this experiment answers. The Nasdaq opens at 9:30 Eastern, which is roughly 13:30 UTC. Asian markets are active from around 00:00 UTC. These overlaps produce real patterns in Bitcoin volume, but whether they produce patterns in price *returns* — which is what TFT is predicting — is less clear.

## The experiment

The experiment is in [`experiments/05_transformer/`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/05_transformer), using the Darts `TFTModel`. The pipeline is identical to the LSTM experiment: same data loader, same target (log-return), same three-way split, same walk-forward evaluation. The only differences are the model class and the addition of the future covariate channel.

```bash
make 05_transformer
```

The ablation — disabling time features to isolate the contribution of attention alone relative to LSTM — is a one-line config change:

```yaml
covariates:
  use_time_features: false
model:
  add_relative_index: true   # required when use_time_features is false
```

With `add_relative_index: true`, Darts auto-generates a relative position index as the future covariate, giving TFT the positional encoding it needs to run without calendar features. Comparing this configuration against the full TFT (with calendar features) measures what the calendar adds; comparing it against the LSTM measures what attention adds.

## The hyperparameter surface

TFT has more configuration knobs than the LSTM:

**`hidden_size`** (default: 32) — the width of all internal representations. Larger means more capacity and more risk of overfitting. Must be divisible by `num_attention_heads`.

**`num_attention_heads`** (default: 4) — how many attention heads run in parallel. Each head attends to different aspects of the history. With `hidden_size=32` and 4 heads, each head operates on an 8-dimensional subspace.

**`lstm_layers`** (default: 1) — depth of the LSTM encoder/decoder within TFT. Usually one layer is enough; the attention mechanism provides the long-range capacity, and stacking LSTM layers on top adds parameters without clear benefit.

**`dropout`** (default: 0.1) — applied throughout the gated residual networks. More important here than in the LSTM because TFT has more parameters to overfit with.

**`hidden_continuous_size`** (default: 8) — the width of the gated residual networks applied to continuous covariates before they are fed to the variable selection networks. A smaller value reduces the covariate processing capacity, which can actually help on small datasets by reducing the number of parameters.

A configuration note: the runner raises an error if both `use_time_features: false` and `add_relative_index: false` are set simultaneously. TFT requires some form of future covariate; without it, the architecture cannot compute its decoder attention properly.

## How to interpret the results

The full comparison table:

| | Naive | ARIMA | XGBoost | LSTM | **TFT** |
|---|---|---|---|---|---|
| Family | last-value | linear | tabular trees | recurrent NN | **attention NN** |
| Past covariates | — | — | engineered scalars | OHLCV sequences | OHLCV sequences |
| Future covariates | — | — | — | — | **hour-of-day, day-of-week** |
| Self-attention | — | — | — | — | **yes** |
| MAE (USD) | 337.89 | ~337–340 | ~337–340 | ~337–340 | *run the experiment* |

Three possible outcomes, and what they each mean:

**TFT matches or exceeds LSTM on MAE**: The attention mechanism and calendar features are not finding signal that the LSTM missed. This is the most likely outcome on hourly BTC log-returns. It is also the most informative one: it tells us that the bottleneck is the signal itself, not the architecture. LSTM's compression was not losing useful information because there was not much useful information to begin with.

**TFT beats LSTM but not by much**: The calendar features (or attention) are providing a marginal directional signal — perhaps exploiting a real but small intraday pattern. Verify with the ablation: if the TFT-without-time-features matches the LSTM, the gain is from the calendar encoding, not from attention itself.

**TFT beats LSTM substantially**: The architecture is finding long-range temporal structure that LSTM's hidden state could not carry. Worth examining which attention heads have high weight on which lags — though see the caution below.

### Attention is not interpretation

High attention weight on a lag does not mean that lag *caused* the prediction. Attention weights are a property of the learned computation, not a causal graph. A head that consistently attends to lag -24 might be tracking a 24-hour autocorrelation in Bitcoin price, or it might be an artifact of the training dynamics on a particular random seed. Causal claims require causal tools; attention maps are at best a starting point for investigation.

This caveat applies especially to TFT's variable-selection network outputs, which show per-feature "importance" across the input window. These weights are learned, not computed from held-out data, so they reflect what the model learned to use — which may or may not correspond to what is actually predictive.

## What TFT cannot do

TFT, like LSTM and XGBoost, has only ever seen Bitcoin. Its parameters encode patterns that exist in this specific asset's history. If those patterns do not generalize — if the 2024 hourly BTC dynamics are genuinely different from the 2023 dynamics the model was trained on — TFT will fail the same way the simpler models fail.

The next two articles ask a different question: what if the model has seen not just Bitcoin, but millions of time series from domains as varied as electricity consumption, retail sales, hospital admissions, and weather stations? Does a model with a broad prior over what time series look like perform better on BTC than a model that has seen only BTC?

That is the zero-shot foundation model question, and article 6 answers it.
