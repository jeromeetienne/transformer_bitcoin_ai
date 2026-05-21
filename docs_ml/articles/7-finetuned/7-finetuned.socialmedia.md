# Social media posts - Fine-tuned foundation models

## Twitter

🎯 Fine-tuning a foundation model on Bitcoin finally beats ARIMA.

Encoder-only recipe + 4.7 years of training data + best-checkpoint restoration = Sharpe **8.20** on the lucky seed.

5-seed mean Sharpe: 5.52. The headline ML lesson hides in that gap. 🧵

#MachineLearning #Bitcoin

## Bluesky

Article 7: Fine-tuned foundation models on Bitcoin.

Same Chronos-2 small as the zero-shot. But now `fit()` updates the encoder against a validation slice with early stopping and best-checkpoint restoration. Plus 4.7 years of training data instead of 1.6.

Single-seed Sharpe: 8.20. Lucky.
5-seed mean Sharpe: 5.52 ± 0.42.

Either way: beats ARIMA. The first model class that does. 🚀

## LinkedIn

The final article of my 7-experiment Bitcoin ML series, and the first time deep learning actually wins.

I fine-tuned Amazon's Chronos-2-small (28M parameters) on 4.7 years of 4-hour Bitcoin data. Same backbone as the zero-shot experiment in the previous article. Three conditions had to align for it to outperform a 3-parameter ARIMA:

1. **Enough training data.** 4.7 years of history (~10,000 bars), not the 1.6 years used in the zero-shot baseline. Less data, the model overfits within 1-3 epochs.

2. **The right recipe.** Encoder-only fine-tuning. Freeze the prediction head, update the encoder. Not "head-only" (too little adaptation). Not "full" (too much overfitting). Encoder-only.

3. **Best-checkpoint restoration.** After training, load the lowest-validation-loss weights before test inference. Without this, the model silently uses end-of-training (overfit) weights. The repo's README documents a prior incident where the same configuration posted Sharpe 6.14 without restoration and Sharpe 3.46 with it.

The result, single seed on disk (`random_state: 42`):

- MAE: 549.85 USD (10 dollars above ARIMA)
- Directional accuracy: **0.5683** (highest in the series)
- Cumulative return: **73.29 %** (highest in the series)
- Annualized Sharpe: **8.20** (highest in the series)

But here is the honest caveat. That 8.20 is **one favourable seed**. The repo's README ships a 5-seed comparison for the same recipe with these numbers:

- 5-seed mean Sharpe: **5.523 ± 0.418** (95 % CI: 5.004 to 6.043)
- 5-seed mean cum_ret: **44.52 % ± 5.48 pp**

The CI lower bound (5.00) still clears the zero-shot Sharpe (2.97). The seed distribution beats zero-shot at 95 % confidence. Per-bar significance ratio: 2.04 σ at the CI lower bound, 3.35 σ on the single seed. Borderline to clean depending on which number you cite.

The headline ML lesson is in that gap between 8.20 and 5.52: **single-seed numbers overstate certainty on stochastic trainers.** Always read foundation-model fine-tuning results through multi-seed CIs.

The series ends here. Seven models. One slice. One scoreboard. The fine-tuned foundation model takes the trading-metric lead - on the right recipe, enough data, and proper checkpoint handling - while ARIMA keeps the point-error crown. The lesson is bigger than any single row: data-to-parameter ratio matters, and inductive bias matters, and how you report results matters.

What is the biggest "single-seed lied" moment you have run into?

#MachineLearning #FoundationModels #FineTuning #Chronos #Bitcoin #TimeSeries
