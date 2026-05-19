# Outline — I built the transformer my repo is named after. ARIMA still won.

## One-line pitch
The Temporal Fusion Transformer ([05_transformer](../../experiments/05_transformer/)) — 4 attention heads, 48h context, multivariate past covariates, cyclical future covariates — lands at `Sharpe +4.59 / dir_acc 0.5053`, behind LSTM (+4.95 / 0.5196) and well behind ARIMA(1,1,1) (+7.28 / 0.5336). The best swept config (`48/64/8/1/0.1`) closes some of the gap (+4.83) but doesn't catch LSTM. Plus a calendar-blindness section: across 6 sweep configs the Sharpe range is +3.45 to +4.83; capacity-vs-Sharpe is essentially flat.

## Audience
Deep-learning practitioners who default to "use a transformer" on any sequence problem. Series readers who watched LSTM tie ARIMA on MAE and lose on direction (Article 8) and want to see whether attention changes the picture.

## Thesis
A serious modern transformer — TFT, with future covariates, multivariate past covariates, attention, gating, variable selection — does not extract more directional skill than a 22k-parameter LSTM, and neither extracts more directional skill than a 1970 linear model. Capacity-vs-Sharpe is flat in the sweep, which is the most damning data point in the article: even the configs that *had* capacity to exploit hour/day-of-week cyclicality didn't pull away from configs that didn't. **There is no detectable intraday or weekly seasonality in hourly BTC log-returns at this scale.**

## Structure

### 1. The hook — the transformer the repo was named for
- The repo is called `transformer_bitcoin_ai`. This is the post about the transformer.
- TFT: 4 attention heads, 48h of context, hidden size 32, 1 LSTM layer in encoder/decoder, dropout 0.1, future covariates (hour_sin/cos, dow_sin/cos), past covariates (log_volume, hl_range, oc_body). ~50k+ parameters.
- The committed result ([05_transformer/results/metrics.json](../../experiments/05_transformer/results/metrics.json)):
  ```json
  {
    "directional_accuracy": 0.5053,
    "annualized_sharpe":   4.586,
    "cumulative_return":   0.296,
    "mae":  373.70
  }
  ```
- Below LSTM. Below MA(24). Behind ARIMA by every column.

### 2. The leaderboard
| Model | dir_acc | Sharpe | cum_ret | MAE |
|---|---:|---:|---:|---:|
| ARIMA(1,1,1) | 0.5336 | +7.28 | +53.0 % | $260.50 |
| LSTM | 0.5196 | +4.95 | +50.3 % | $274.02 |
| MA(24) | 0.5143 | +4.33 | +25.7 % | $756.19 |
| **TFT** | **0.5053** | **+4.59** | **+29.6 %** | **$373.70** |
| XGBoost | 0.4872 | +1.45 | +8.1 % | $270.73 |

(Sep–Nov 2024 window.)

### 3. What TFT is doing that LSTM can't
- **Attention.** Each forecast step can look at any prior bar in the input chunk with learned weights, rather than relying on hidden-state recurrence alone.
- **Future covariates.** TFT consumes deterministic time features (hour_sin/cos, dow_sin/cos) at *future* bars. Translation: the model is told what hour of the day and what day of the week it's predicting *for*, not just what's in the input history.
- **Variable selection networks.** Per-input gating decides which features matter at each step.
- **Interpretable attention.** TFT's masked attention is designed for time series — temporal locality is preserved, future bars are masked.

In other words: TFT is the architecture you'd reach for if you genuinely thought BTC had intraday or weekly seasonality and wanted a model that could *find it*.

### 4. The result: the future covariates didn't pay off
- LSTM, no future covariates, `dir_acc 0.5196`.
- TFT, with cyclical future covariates plus everything LSTM has, `dir_acc 0.5053`.
- Net effect of attention + future covariates on this asset: nothing positive measurable. TFT is the architecturally richer model and the leaderboard column it controls is *worse*.

### 5. The sweep — capacity-vs-Sharpe is flat
From [05_transformer/results/sweep.csv](../../experiments/05_transformer/results/sweep.csv):

| icl | hidden | heads | layers | dropout | dir_acc | Sharpe | MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 16 | 2 | 1 | 0.1 | 0.5028 | +4.37 | $398.59 |
| 48 | 32 | 4 | 1 | 0.1 | 0.5053 | +4.59 | $373.70 |
| 48 | 32 | 4 | 2 | 0.2 | 0.4984 | +4.12 | $420.61 |
| **48** | **64** | **8** | **1** | **0.1** | **0.5109** | **+4.83** | **$392.46** |
| 96 | 32 | 4 | 1 | 0.1 | 0.4984 | **+3.45** | $409.96 |
| 96 | 64 | 8 | 2 | 0.2 | 0.5047 | +4.47 | $395.20 |

Sharpe range across 6 configs: **+3.45 to +4.83**. Capacity-vs-Sharpe is essentially flat. The best config has `8 heads, 1 layer, hidden 64` — wider, not deeper. The deepest config (`layers=2, dropout=0.2`) regresses.

### 6. The calendar-blindness story
This is the most interesting piece in the article.
- TFT's whole pitch over LSTM is *future covariates*: it can look at "predicting for hour 14, Tuesday" vs "predicting for hour 03, Sunday" and decide whether that matters.
- If hourly BTC carried meaningful intraday or weekly seasonality, the TFT configs with more capacity would pull away — they have more parameters to exploit those cyclical inputs.
- They don't pull away. Capacity-vs-Sharpe is flat.
- **Conclusion: at the hourly resolution, there's no detectable cyclical seasonality in BTC log-returns that a TFT can exploit.** That's a substantive claim, supported by the sweep.
- This is also why the wider/shallower configs do best: there *is* a small benefit from more attention heads / more channels for the recurrence-style contribution, but no benefit from depth-stacking — which is what you'd expect if the model is essentially learning a slightly fancier AR(1).

### 7. Width beat depth in the sweep
- `(48, 64, 8, 1, 0.1)` — wider, single-layer — best Sharpe.
- `(48, 32, 4, 2, 0.2)` and `(96, 32, 4, 1, 0.1)` — deeper or longer-context — worse.
- This is consistent with Article 8's LSTM finding: longer context didn't help, and stacking more recurrent layers regressed.
- TFT generalizes the lesson: extra capacity *across time* (depth, longer context) doesn't pay; extra capacity *across features* (heads, channel width) pays a tiny amount.

### 8. The moral discomfort of the title
- This repo is named `transformer_bitcoin_ai`. The transformer it was nominally built around is the model that finishes 4th out of 5 trained models on the leaderboard.
- That is the most honest thing this lab could possibly publish.
- The transformer is *not bad* — it works, it converges, it produces sensible predictions, it survives the harness. It's just *outclassed* by a 1970 linear model with fewer than ten parameters on a near-random-walk asset.
- The discomfort is good. It's the data point a reader can't get from a vendor blog post.

### 9. What the result is *not*
- Not "transformers don't work on time series". TFT is excellent on retail demand, energy, weather. The point is *hourly BTC log-returns specifically*.
- Not "future covariates are useless". They're not. They're useless *here* because the underlying data has no cyclical structure to exploit.
- Not "more capacity always hurts". It says more capacity didn't help — which is a weaker and supported claim.

### 10. Reproducing
```
make 05_transformer
make 05_transformer_sweep   # writes results/sweep.csv
```

### 11. Closing — articles 10 and 11
- Article 10 will deflate the +4.59 TFT Sharpe by separating signal from drift. Spoiler: most of it is drift.
- Article 11 brings in zero-shot foundation models (Chronos-2, TimesFM 2.5) — pretrained on millions of unrelated series. The same lesson: capacity is not the bottleneck.

## Key code/file references
- [experiments/05_transformer/run.py](../../experiments/05_transformer/run.py) — TFT harness
- [experiments/05_transformer/config.yaml](../../experiments/05_transformer/config.yaml) — architecture knobs
- [experiments/05_transformer/results/sweep.csv](../../experiments/05_transformer/results/sweep.csv) — the calendar-blindness story
- [experiments/05_transformer/results/metrics.json](../../experiments/05_transformer/results/metrics.json) — the headline +4.59 Sharpe

## Tone notes
- Honest discomfort, not gloating. ARIMA winning is the data, not a meme.
- The calendar-blindness section is the most interesting piece — give it room.

## Length target
~2,000–2,400 words.
