# Outline — Article 4: Transformer

Working title: *"Attention on a price series — and what it changes (or doesn't)."*

Source experiment: [`05_transformer`](experiments/05_transformer/).

## Editorial framing

The named model of the repo. The Darts Temporal Fusion Transformer on `BTCUSDT` 4h, compared honestly against ARIMA(3, 1, 3) — which it loses to on every single metric in the leaderboard. The point of the article is not that "transformers don't work on time series" (they do, in other contexts). The point is that *adding attention, future covariates, variable-selection networks, and gated residual connections on top of an LSTM-like encoder costs a 65 % MAE penalty and a 4-Sharpe-points loss against a 3-parameter linear model on this slice*, and *what that means about capacity-vs-data tradeoffs* on noisy financial time series.

## Beats

1. **Hook.** This is the experiment the repo is named after. It is also the experiment that loses the hardest in the entire series. Darts' Temporal Fusion Transformer on 4h Bitcoin reports MAE 891.09 USD — *65 % worse than the naive baseline*, *51 % worse than every other trained model* in the leaderboard. Directional accuracy 0.5055 — essentially coin-flip. The lesson is not "transformers are bad"; the lesson is what happens when ~30 000 parameters of attention + variable selection + gated residual networks meet 3 467 training rows.

2. **Methodology recap.** Same 366-bar test slice. Same walk-forward shape (`historical_forecasts(retrain=False)`). Same metrics module. The new thing this article: **future covariates** — calendar-cyclical encodings of hour-of-day and day-of-week that the TFT can legitimately consume at prediction time because they are deterministic functions of the timestamp.

3. **What the model is — Temporal Fusion Transformer in plain English.**
   - **Variable selection networks** that learn, per time step, which inputs matter.
   - **LSTM encoder / decoder** — yes, the TFT has an LSTM inside it, *plus* attention.
   - **Multi-head self-attention** for long-range dependencies across the input window.
   - **Gated residual networks** for feature mixing and skip connections — controls how much each transformation contributes.
   - **Past covariates + future covariates + static metadata**, all kept as separate channels.
   - Library: `darts.models.TFTModel`. Lifts from the canonical Bryan Lim et al. paper, with sensible defaults wrapped by Darts.

4. **The configuration.** From [`experiments/05_transformer/configs/btc_4h_2024.config.yaml`](experiments/05_transformer/configs/btc_4h_2024.config.yaml):
   - `input_chunk_length: 48` (= 8 days at 4h)
   - `output_chunk_length: 1`
   - `hidden_size: 32`
   - `lstm_layers: 1`
   - `num_attention_heads: 4`
   - `dropout: 0.1`
   - `hidden_continuous_size: 8`
   - `add_relative_index: false`, `full_attention: false`
   - Adam, lr 0.001, batch_size 64, n_epochs 30 (early-stopped), `random_state: 42`
   - val_fraction 0.1, `early_stopping_patience: 5`
   - Past covariates: `log_volume`, `hl_range`, `oc_body`
   - Future covariates: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` (calendar cyclical)

5. **The training shape.** Train 3 467 rows. Validation 366 rows. Test 366 rows. PyTorch Lightning trainer. Early stopping on `val_loss`. About 30 000 parameters total — three times the LSTM in article 3.

6. **The result — verbatim from `metrics.json`.**
   - MAE **891.09 USD** (worst in the leaderboard)
   - RMSE 1271.26 USD
   - MAPE 1.1499 %
   - dir_acc 0.5055 (essentially coin-flip)
   - cum_ret 0.1219
   - Sharpe 2.5390

7. **The leaderboard, as of article 4.**
   | Model | MAE | dir_acc | Sharpe |
   |---|---|---|---|
   | Naive | 540.96 | NaN | — |
   | ARIMA(3, 1, 3) | **539.15** | 0.5082 | **6.8559** |
   | XGBoost | 550.85 | 0.5082 | 6.1388 |
   | LSTM | 539.23 | **0.5301** | 4.8221 |
   | **Temporal Fusion Transformer** | **891.09** | 0.5055 | 2.5390 |

8. **The 65 % MAE penalty.** This is the article's load-bearing observation. The MAE jumped from 539 (LSTM) to 891 (TFT) — same training slice, same walk-forward harness, same target. Adding attention and variable selection and gated residuals and future covariates on top of the LSTM **cost $352 per bar on average**. The model is fitting the training slice in a way that does not generalize to the test slice — the extra capacity is being absorbed by random structure that wasn't there in the LSTM-only model.

9. **Why this happened — the model class teaching beat.**
   - **30 000 parameters on 3 467 training rows.** The ratio is about 9 training rows per parameter. For a sequence model on heavy-tailed financial data, that is not enough to constrain the architecture's degrees of freedom. The variable selection networks, gated residuals, and attention layers find structure in the training slice that does not replicate in the test slice.
   - **The deep-learning literature suggests the TFT shines with thousands of related time series, not with one BTCUSDT series.** Bryan Lim et al. evaluated the TFT on M5 retail data (3 049 stores × multiple products), on traffic data, on electricity-load forecasting — multi-series benchmarks where the model can transfer structure across related series. On a single financial series, the inductive bias mismatches the data shape.
   - **The future covariates didn't help at this slice.** Hour-of-day and day-of-week cyclical features should, in principle, capture intraday and intraweek seasonality if any exists. On 4h Bitcoin during the post-election rally, there is essentially no exploitable seasonal structure, so the future-covariates channel adds capacity without adding signal.

10. **The Sharpe-positive-despite-MAE-collapse pattern.** TFT Sharpe is 2.54 — *still positive*, *still arguably significant*. How does a model with worst-in-leaderboard MAE post a positive Sharpe? Same long-flat / favourable regime mechanism we've seen in article 3: the strategy goes long whenever `pred > ref`. The TFT, being wrong-about-magnitude-but-trending-with-the-rally, generates enough strongly positive predictions to be long during many of the rally's up bars. It collects the drift; it just also collects huge errors on the down bars where it was wildly wrong. MAE-collapse + drift-favourable regime = Sharpe-positive-anyway. This is the cleanest example in the series of why a positive Sharpe in isolation is *not enough* to call a model "good."

11. **Per-bar Sharpe significance.** 2.5390 / √2190 = 0.0543. SE 1 / √366 = 0.0523. Ratio ≈ **1.04 σ** — *not significant*. Even with the regime tailwind, the Sharpe doesn't survive the standard error test. This is the right way to read the TFT row: model says one thing (positive Sharpe), per-bar significance says another (consistent with no skill).

12. **The sweep is stale on 1h** — same note as article 2. The lesson stands generically but the specific MAE values are not comparable to the 4h headline. Mention the pending re-run.

13. **What this article tells us about the model class.**
   - **Capacity is not a free variable on small datasets.** The TFT's architectural richness needs either enormous training data or multi-series transfer to pay back. Neither applies on one BTCUSDT 4h slice with 3 467 training rows.
   - **Attention does not inherently outperform recurrence on time series.** The LSTM in article 3 was richer than ARIMA and went backwards on Sharpe; the TFT is richer than the LSTM and goes backwards on MAE too. Both lessons are consistent with the broader observation that *what the model is asked to predict from* matters more than *how much machinery sits between input and output*.
   - **The deep-learning hype is real in some domains and an active liability in others.** This is the article that lands that observation in code.

14. **Regime caveat** — same paragraph as articles 1 / 2 / 3.

15. **Reproduce.** `make 05_transformer` + `make 05_transformer_sweep`.

## What this article is NOT

- A claim that transformers do not work on time series. The literature on the Temporal Fusion Transformer in multi-series benchmarks is strong. The lesson is what happens on *one* BTCUSDT series with 3 467 training rows.
- A claim that ARIMA "wins" in any general sense. ARIMA wins on this slice.
- A transformer tutorial. Refer the curious reader to Bryan Lim et al.'s original TFT paper and the Darts documentation.
