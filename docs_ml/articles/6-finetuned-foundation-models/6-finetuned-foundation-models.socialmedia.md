# Social Media Posts — Article 6: When You Teach the Giant

## Twitter

**Thread: I fine-tuned a foundation model on Bitcoin. Sharpe went from 4.74 to 5.52. Here's exactly what worked. 🎓🧵**

1/ 🪙 Zero-shot foundation models gave us MAE near the $338 floor on hourly BTC. The pretrained prior didn't transfer cleanly.

So I let the model update its weights on BTC data.

**Result: Sharpe 4.74 → 5.52 (5/5 seeds beat zero-shot).** 📈

2/ 📚 **Lesson 1: dataset size matters.**

First attempts used 1.6 years of 4h BTC data (~3,500 rows). Model overfit in 1-3 epochs. Worse than zero-shot.

Fix: extend to 4.7 years (~10,000 rows). Now the fine-tuning signal is large enough to update representations, not just memorize noise. 📊

3/ 🧊 **Lesson 2: freeze the head, fine-tune the encoder.**

Head-only fine-tuning *eroded* the pretrained directional prior. Accuracy dropped below 0.49 — worse than a coin flip.

Why? The head encodes directional structure. Updating it with BTC noise teaches it to suppress that structure. ⚠️

4/ 💾 **Lesson 3: restore the best-val_loss checkpoint.**

After early stopping, `model.fit()` keeps the *last epoch's* weights in memory — not the best ones.

One run looked like Sharpe=6.146. Loading the actual best checkpoint dropped it to 3.46. The 6.146 was an artifact. 😬

5/ 🏆 The headline result on `btc_4h_2020_2024`, 5-seed mean:

• Sharpe: **5.52 ± 0.42** (vs 4.74 zero-shot) — 5/5 seeds win
• Cumulative return: **44.52% ± 5.48pp** (vs 36.64%) — 5/5 seeds win
• Directional accuracy: 0.529 (barely above zero-shot)

6/ 🎯 The interesting part: Sharpe improved substantially while directional accuracy barely moved.

The fine-tuning didn't teach the model to predict direction better.

It taught the model to predict **magnitude** better. Smaller intervals when quiet, wider when volatile. ⚖️

7/ 💡 The path to better Bitcoin forecasting isn't a bigger architecture.

It's more data + better calibration.

The fine-tuned model knows when to be confident and when not to. That's what moved the needle — not better sign-prediction. 🎯

Full article → [link]

#MachineLearning #FineTuning #Bitcoin

---

## Bluesky

🎓 **I fine-tuned a foundation model on Bitcoin. Sharpe went from 4.74 to 5.52. 5/5 random seeds beat zero-shot.**

Three lessons learned the hard way:

1️⃣ Dataset size matters — 1.6 years overfit in 3 epochs. 4.7 years gave the model enough signal to actually update representations.

2️⃣ Freeze the head, fine-tune the encoder. Head-only training *eroded* the pretrained directional prior. The head encodes structure that BTC's noisy signal shouldn't be teaching it to discard.

3️⃣ Restore the best-val_loss checkpoint. After early stopping, in-memory weights are the *last* epoch's — not the best. One run looked like Sharpe 6.146 until I loaded the actual best weights (3.46).

🎯 The interesting finding: Sharpe improved while directional accuracy barely moved. The fine-tuning taught the model magnitude calibration — knowing when to be confident — not better sign prediction.

Full article → [link]

#MachineLearning #FineTuning

---

## LinkedIn

🎓 **I fine-tuned a foundation model on Bitcoin. The Sharpe ratio improved from 4.74 to 5.52 — and every single random seed beat zero-shot. Here's exactly what worked, and the three things I had to get right.**

Article 6 closes this series with the first model that meaningfully beats the rest. Encoder-only fine-tuning of Chronos-2-small on 4.7 years of 4-hour Bitcoin data, evaluated across 5 random seeds:

| Metric | Fine-tuned | Zero-shot |
|---|---|---|
| **Annualized Sharpe** | **5.52 ± 0.42** | 4.74 |
| **Cumulative return** | **44.52% ± 5.48pp** | 36.64% |
| Directional accuracy | 0.529 | 0.516 |
| MAE | 539.41 USD | 540.37 USD |

5/5 seeds independently beat zero-shot on Sharpe and cumulative return. Formally significant at the 5% level. ✅

**📚 Lesson 1: dataset size matters more than I expected.** The first attempts used 1.6 years of 4h BTC data (~3,500 training rows). The model overfit within 1-3 epochs. Validation loss reversed quickly. Results were *worse* than zero-shot. The fix was extending to 4.7 years (~10,000 rows) — enough fine-tuning signal to update representations rather than memorize noise. Fine-tuning a foundation model on a few months of data will likely make things worse, not better.

**🧊 Lesson 2: freeze the head, fine-tune the encoder.** The natural defaults — fine-tune everything, or fine-tune the head only — are both wrong. Head-only fine-tuning *eroded* the pretrained directional prior: accuracy dropped from 0.516 to below 0.49, worse than a coin flip. The head encodes directional structure learned during pretraining; updating it with BTC's near-noise signal teaches it to suppress that structure. Encoder-only fine-tuning (94.3% of parameters trainable, head frozen) preserves the directional prior while letting the encoder adapt to BTC's distributional properties.

**💾 Lesson 3: restore the best-val_loss checkpoint.** After `model.fit()` returns following early stopping, the in-memory weights are from the *last training epoch* — not the epoch with the best validation loss. The `ModelCheckpoint` callback saves the best weights to disk, but they're not in memory. Without explicit restoration, inference uses overfit weights. One configuration appeared to show Sharpe = 6.146; loading the actual best checkpoint dropped it to 3.46. The earlier result was an artifact. ⚠️

**🎯 The interesting finding: it's magnitude, not direction.** Sharpe improved substantially while directional accuracy barely moved. The fine-tuning didn't teach the model to predict direction better — it taught the model to predict *magnitude* better. Smaller predicted intervals when the market was quiet, larger intervals when it was volatile. A position-sized strategy earns more per correct call when the magnitude estimate is right, even if the direction call is no better than before. ⚖️

**💡 What this series reveals.** Seven models, one question, one honest answer: on short-timescale Bitcoin price, the primary bottleneck is not the model — it's the signal. ARIMA couldn't beat naive by much. XGBoost couldn't beat ARIMA. LSTM and TFT couldn't beat XGBoost. Zero-shot foundation models couldn't beat any of them.

What finally moved the needle: more data + better calibration. Not a bigger architecture. The fine-tuned model knows when to be confident and when not to — and that's enough.

Article 6 is live → [link]

#MachineLearning #FineTuning #QuantitativeFinance
