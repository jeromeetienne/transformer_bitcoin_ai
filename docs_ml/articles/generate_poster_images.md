# Cover-image prompts (one per article)

This file holds one short cover-image prompt per article in the series. Each prompt describes the title, the subtitle, the secondary visual motif, and the headline metric badges baked into the corresponding `<slug>.poster.svg` in the article's folder.

The SVGs are the canonical artefact (see [generation_image.md](../generation_image.md)). These prompts exist so a reader can:

- Regenerate the cover in claude.ai or another raster generator without re-deriving the design.
- Remix the cover (swap the motif, change the metric badge) without losing the series-level constraints.

Shared constraints across every article in the series:

- Canvas: 1200 by 627 (LinkedIn feed format).
- Dark navy gradient background (`#0b1220` to `#101b30`), Bitcoin-orange accent (`#f7931a`).
- Typography: Inter Tight, large left-aligned title, smaller subtitle, small uppercase series tag above the title.
- Series tag: `BITCOIN ML SERIES • ARTICLE N OF 7` (article 0 is the curtain-raiser; it carries the same tag with N = 0).
- Subtle horizontal grid lines in the background.
- A motif specific to the article's method, rendered as a watermark (opacity ~0.18 to 0.25), never competing with the headline.
- A metric strip at the bottom showing one to four headline numbers from the article.
- Footer: `jerome etienne • github.com/jeromeetienne/transformer_bitcoin_ai`.

---

## Article 0 - Presentation of the project

- Title: `Can Machine Learning Predict Bitcoin?`
- Subtitle: `Seven models, one data slice, one honest scoreboard. From naive last-value to fine-tuned foundation models.`
- Motif: a Bitcoin-orange price line drifting upward across the canvas, with a row of seven ladder dots along the bottom (one per article in the series).
- No metric badges (article 0 has no headline numbers).
- File: [0-presentation-of-the-project/0-presentation-of-the-project.poster.svg](0-presentation-of-the-project/0-presentation-of-the-project.poster.svg)

## Article 1 - Baseline (naive last-value)

- Title: `The Dumbest Bitcoin Predictor That Works`
- Subtitle: `Naive last-value: zero parameters, MAE $518. The floor every later model must clear.`
- Motif: a stepped flat-line pattern that visualizes the `close_pred[t] = close[t-1]` predictor: horizontal segments at every step.
- Metric badges: `MAE $518`, `PARAMETERS 0`, `DIR_ACC NaN`.
- File: [1-baseline/1-baseline.poster.svg](1-baseline/1-baseline.poster.svg)

## Article 2 - ARIMA

- Title: `Three Parameters, Half a Century of Lead.`
- Subtitle: `ARIMA(1,1,1) on Bitcoin: $1.20 better than naive, Sharpe 6.06, and a directional edge that pays.`
- Motif: a horizontal axis with small orange bars above and below it, representing the differenced log-returns ARIMA actually fits.
- Metric badges: `MAE $517`, `DIR_ACC 0.5547`, `SHARPE 6.06`.
- File: [2-arima/2-arima.poster.svg](2-arima/2-arima.poster.svg)

## Article 3 - XGBoost

- Title: `When 31 Features Lose to 3 Parameters.`
- Subtitle: `XGBoost trades point error for trading aggression. Worse MAE than naive. Best Sharpe in the series so far.`
- Motif: three small decision-tree silhouettes, each three levels deep, evoking a gradient-boosted ensemble.
- Metric badges: `MAE $539`, `CUM_RET 50.42%`, `SHARPE 6.42`.
- File: [3-xgboost/3-xgboost.poster.svg](3-xgboost/3-xgboost.poster.svg)

## Article 4 - LSTM

- Title: `The First Deep Learning Failure.`
- Subtitle: `LSTM on 1,448 Bitcoin bars: directional accuracy below a coin flip. Capacity without data is overfit.`
- Motif: a chain of seven LSTM cell boxes connected by arrows, each labeled `h_t` through `h_7`, representing recurrent hidden state.
- Metric badges: `DIR_ACC 0.4938`, `MAE $522`, `SHARPE 4.27`.
- File: [4-lstm/4-lstm.poster.svg](4-lstm/4-lstm.poster.svg)

## Article 5 - Transformer

- Title: `When Attention Pays Nothing.`
- Subtitle: `The Temporal Fusion Transformer on 1,448 bars: 57% worse MAE than naive. Below coin-flip direction.`
- Motif: a fully-connected attention pattern across seven dots, with crossing lines representing pairwise attention weights.
- Metric badges: `MAE $813`, `DIR_ACC 0.4913`, `SHARPE 5.87`.
- File: [5-transformer/5-transformer.poster.svg](5-transformer/5-transformer.poster.svg)

## Article 6 - Pretrained foundation models

- Title: `200 Million Parameters, Zero Bitcoin Training.`
- Subtitle: `Chronos-2 and TimesFM 2.5, zero-shot on Bitcoin. Every variant loses to a 7-parameter ARIMA.`
- Motif: a probabilistic forecast band: dashed q10/q90 lines bounding a solid q50 median, with the interval lightly shaded - the foundation models' probabilistic output.
- Metric badges (Sharpe across variants): `CHRONOS-SMALL 2.97`, `CHRONOS-LARGE 1.90`, `TIMESFM 4.11`, `ARIMA 6.86` (highlighted in orange).
- File: [6-pretrained/6-pretrained.poster.svg](6-pretrained/6-pretrained.poster.svg)

## Article 7 - Fine-tuned foundation models

- Title: `When Fine-Tuning Finally Beats ARIMA.`
- Subtitle: `Encoder-only Chronos-2 on 4.7 years of BTC. Single seed 8.20. 5-seed mean 5.52. Headline lesson.`
- Motif: two training curves (gray train loss, orange validation loss) converging then diverging, with a circular marker on the validation curve's minimum labeled `best val_loss` - the checkpoint that gets restored.
- Metric badges: `SHARPE (SEED 42) 8.20` (orange, the lucky seed), `5-SEED MEAN 5.52`, `DIR_ACC 0.5683`, `CUM_RET 73.29%`.
- File: [7-finetuned/7-finetuned.poster.svg](7-finetuned/7-finetuned.poster.svg)

---

## How to regenerate a poster

1. Edit the prompt above (title, subtitle, motif, badges) until it matches what you want.
2. Update `<slug>/<slug>.poster.svg` accordingly, keeping the series-level constraints intact.
3. Re-render the PNG: `rsvg-convert -w 1200 -h 627 <slug>/<slug>.poster.svg -o <slug>/<slug>.poster.png`.

If you need a raster generator instead of the inline SVG, paste the prompt - title, subtitle, motif, badges, plus the shared constraints at the top of this file - into claude.ai and ask for a 1200x627 LinkedIn-feed cover.
