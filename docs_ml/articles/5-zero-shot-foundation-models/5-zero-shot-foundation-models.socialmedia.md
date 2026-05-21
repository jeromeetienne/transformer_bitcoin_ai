# Social Media Posts — Article 5: A Model That Has Never Seen Bitcoin Has Opinions About It

## Twitter

**Thread: I gave Bitcoin to two foundation models that have never seen a financial time series. They had opinions anyway. 🤖🧵**

1/ 🪙 Every model in articles 1-4 was trained on Bitcoin.

Naive needed no data. ARIMA fit 7 parameters to BTC. XGBoost fit thousands of tree splits to BTC features. LSTM and TFT fit tens of thousands of weights to BTC log-returns.

This experiment does something different. 🔄

2/ 🧠 Two foundation models, zero BTC training:
• **Chronos-2** (Amazon, T5-based, 120M params)
• **TimesFM 2.5** (Google, patch-transformer, 200M params)

Trained on electricity, retail, hospital admissions, weather. Hundreds of millions of time-series observations. Zero financial data. 🌐

3/ ❓ The question: do general "this is what a time series looks like" priors transfer to Bitcoin?

Both models output **probabilistic forecasts** — 200 samples per step. The median is the point prediction; q10/q90 form a calibration band. 📊

4/ ⚖️ Both kept univariate for a clean methodology.

Chronos-2 supports covariates. TimesFM doesn't. Giving one extra inputs while denying the other would conflate "stronger prior?" with "more inputs help?"

Isolating the prior is the whole point of this experiment. 🎯

5/ 📉 Expected result: MAE near $338. Directional accuracy near 0.50.

Foundation models were trained on series with strong autocorrelation, clear seasonality, mean-reversion. Hourly BTC has almost none of that.

A miscalibrated prior, applied honestly. 🪙

6/ 🔍 The more interesting question: does the pretrained prior have any *directional* bias?

A "things tend to continue" prior is implicitly momentum.
A "things tend to mean-revert" prior is implicitly contrarian.

If either is right about BTC's test period, directional accuracy > 0.50.

7/ 🎒 Zero-shot can't adapt to BTC's heavy tails (kurtosis 10-15), absence of clear seasonality in returns, or regime-switching.

The question for article 6: what happens if we let the model update its weights on BTC data? Fine-tuning is next. 🛠️

Full article → [link]

#MachineLearning #FoundationModels

---

## Bluesky

🤖 **I gave Bitcoin to two foundation models that have never seen a financial time series. They had opinions anyway.**

Chronos-2 and TimesFM 2.5 were trained on electricity consumption, retail sales, hospital admissions, weather — hundreds of millions of time-series observations from non-financial domains. Then I asked them to forecast hourly BTC. 🌐

📊 Both produce probabilistic forecasts — 200 samples per step, with q10/q50/q90 quantiles. The median is the point prediction; the q10–q90 band is the calibration signal.

📉 Expected outcome: MAE near the $338 naive floor. These models were trained on series with strong autocorrelation, clear seasonality, and mean-reversion. Hourly BTC has almost none of that. A miscalibrated prior, applied honestly.

The interesting follow-up: what happens when we let the model update its weights on Bitcoin? That's article 6. 🛠️

Full article → [link]

#MachineLearning #FoundationModels

---

## LinkedIn

🤖 **What happens when you give Bitcoin to a model that has never seen a financial time series? You get an honest answer about transfer learning.**

Article 5 of this forecasting series evaluates two time-series foundation models — Chronos-2 (Amazon) and TimesFM 2.5 (Google) — on hourly Bitcoin price. Neither model has ever been trained on financial data.

**🌐 What they have seen instead.** Both models were trained on hundreds of millions of time-series observations from the M competitions, the Monash Time-Series Repository, and other public corpora — electricity consumption curves, retail sales, hospital admissions, weather station readings, traffic counts. The pretrained prior encodes generic regularities: autocorrelation, seasonality, mean reversion, volatility clustering. The empirical question: does that prior transfer to Bitcoin?

**📊 The first probabilistic forecasts in this series.** Both models output a distribution over future values rather than a point prediction:

• **q10**: the 10th percentile — the model thinks there's a 90% chance the actual value exceeds this
• **q50**: the median — used for all standard metrics (MAE, RMSE, MAPE, Sharpe)
• **q90**: the 90th percentile

A well-calibrated model should see ~80% of actual values fall within the q10–q90 band. In practice, financial series are heavy-tailed and difficult to calibrate.

**⚖️ Univariate by design.** Chronos-2 supports past and future covariates. TimesFM doesn't. Giving one model extra inputs while denying the other would conflate "which pretrained prior is stronger?" with "did the extra inputs help?" Both kept univariate isolates the actual research question. The covariate ablation for Chronos-2 is a clean follow-up, not this experiment.

📉 **Expected result**: MAE near the $338 naive floor, directional accuracy near 0.50, Sharpe near zero. Foundation models trained on retail sales and electricity grids learned that "things tend to be autocorrelated, with strong seasonality." Hourly Bitcoin log-returns are close to white noise — almost no autocorrelation, no strong intraday seasonality in returns, no reliable mean-reversion. The prior is real, but it is miscalibrated for this asset class.

🎯 **The more interesting question** is whether the pretrained directional bias is helpful at all. A "things continue" prior is implicitly momentum; a "things mean-revert" prior is implicitly contrarian. If either happens to align with BTC's test period, directional accuracy will exceed 0.50 without any BTC-specific knowledge.

🛠️ The natural follow-up — what if we let the model update its weights on Bitcoin data? — is article 6. Fine-tuning is where this experiment finally moves the needle.

Article 5 is live → [link]

#MachineLearning #FoundationModels #TimeSeries
