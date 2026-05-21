# Social Media Posts — Article 4: The Architecture That Took Over AI

## Twitter

**Thread: I pointed the architecture that took over AI at Bitcoin. Here's what attention actually buys you. 🎯🧵**

1/ 🧠 The LSTM has one fundamental flaw: it must compress 48 bars of history into a 32-dim hidden state before predicting anything.

If the relevant bar is 40 steps back, the model had to keep that signal alive through 39 forget-gate applications. 🗜️

2/ ⚡ Attention sidesteps the bottleneck entirely.

Instead of compressing history into a state, attention lets the model look *directly* at any past bar when forming the prediction.

A bar 40 steps back is as accessible as a bar 1 step back. 🔍

3/ 🏗️ I used the Temporal Fusion Transformer (TFT) — purpose-built for time series:
• Variable selection networks (automatic feature gating)
• LSTM encoder + decoder (local sequential processing)
• Multi-head self-attention (long-range structure)
• Gated residual networks (learned skip connections)

4/ 🆕 The new input channel: future covariates.

The LSTM saw past data only. TFT also gets calendar features — hour of day, day of week — that are deterministically known for future bars.

Encoded as sin/cos pairs so hour 23 and hour 0 are adjacent in feature space. 🌀

5/ ❓ Does Bitcoin have exploitable intraday seasonality at hourly resolution?

Asian markets open. Nasdaq opens. These produce real patterns in *volume*. Whether they produce patterns in *returns* — what TFT actually predicts — is what this experiment answers. 🌏

6/ 📉 Most likely result: MAE close to LSTM, near the $337 floor.

That's informative, not disappointing. It tells us the bottleneck is the signal, not the architecture. LSTM's compression wasn't losing useful information — there wasn't much to begin with. 🪙

7/ ⚠️ Attention is not interpretation.

High attention weight on a lag doesn't mean that lag *caused* the prediction. Attention maps are properties of the learned computation, not causal graphs. Be careful with causal claims. 🚨

Full article → [link]

#MachineLearning #Transformers

---

## Bluesky

🎯 **I pointed the architecture that took over AI at Bitcoin. Here's what attention actually buys you.**

The LSTM compresses 48 bars of history into a 32-dim hidden state. The Temporal Fusion Transformer's attention mechanism replaces that bottleneck with direct lookup — a bar 40 steps back is as accessible as a bar 1 step back. ⚡

The new ingredient: future covariates. TFT consumes calendar features (hour, day-of-week) encoded as sin/cos pairs, alongside the past OHLCV trajectory. If BTC has any intraday seasonality at hourly resolution, this is where it shows up. 🌀

📉 Most likely outcome: MAE close to LSTM, near the $337 floor. That's informative — it tells us the bottleneck is the signal itself, not the architecture.

⚠️ Caution: attention weights are not causal explanations. They reflect what the model learned to use, not what's actually predictive.

Full article → [link]

#MachineLearning #Transformers

---

## LinkedIn

🎯 **The architecture that took over AI, now trying to read Bitcoin's mind.**

Article 4 of this forecasting series applies attention to Bitcoin price via the Temporal Fusion Transformer (TFT) — a purpose-built architecture for multi-horizon time-series forecasting.

**🚧 The problem attention solves.** The LSTM in article 3 had to compress 48 bars of Bitcoin history into a 32-dim hidden state before making any prediction. If the most relevant bar was 40 steps back, the model had to keep that signal alive through 39 forget-gate applications. Attention sidesteps this entirely: instead of funneling history through a bottleneck, it lets the model query every past timestep directly when forming the prediction. ⚡

**🏗️ Why TFT and not a vanilla Transformer.** TFT adds several structures purpose-built for time series:

• **Variable selection networks** — automatic feature gating at each timestep
• **LSTM encoder/decoder** — local sequential processing, paired with attention for long-range structure
• **Multi-head self-attention** — different heads attend to different aspects of history
• **Gated residual networks** — learned skip connections throughout the architecture

**🆕 The new input channel: future covariates.** The LSTM consumed past data only. TFT also receives calendar features — hour of day, day of week — that are deterministically known for any future bar. Encoded as sin/cos pairs to preserve cyclical structure (so hour 23 and hour 0 are adjacent in feature space, not maximally far apart).

📊 The empirical question this experiment answers: does Bitcoin have exploitable intraday seasonality at hourly resolution? Asian markets and Nasdaq opens produce real patterns in BTC *volume* — but whether they produce patterns in *returns* (what TFT actually predicts) is much less clear.

📉 The expected result: MAE close to the LSTM, near the $337 naive floor. That outcome is the most informative one — it tells us the bottleneck is the signal itself, not the architecture. LSTM's compression wasn't losing useful information; there wasn't much useful information to begin with.

⚠️ **One critical caveat on interpretability.** High attention weight on a particular lag does *not* mean that lag caused the prediction. Attention maps are properties of the learned computation, not causal graphs. Causal claims require causal tools.

The next question — what if the model has seen not just Bitcoin, but millions of time series from electricity grids, retail, weather, and hospital admissions? That's the zero-shot foundation model question. Article 5.

Article 4 is live → [link]

#MachineLearning #Transformers #DeepLearning
