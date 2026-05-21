# Social Media Posts — Article 3: I Gave a Neural Network a Memory

## Twitter

**Thread: I gave an LSTM a 48-hour memory of Bitcoin. It still couldn't predict it. 🧠🧵**

1/ 🌳 XGBoost saw 31 features per bar — unordered. A tree doesn't care which lag came first.

The LSTM cares about order. It processes the sequence one bar at a time, building a hidden state that summarizes everything it has seen.

A fundamentally different inductive bias. 🔄

2/ ⚙️ How it works in one tweet:
• Forget gate decides what to drop
• Input gate decides what to write
• Output gate decides what to expose

A 32-dimensional hidden state compresses the entire input window into 32 numbers. 🗜️

3/ 🎛️ The four knobs that actually matter:
• `input_chunk_length=48` (two days of history)
• `hidden_dim=32` (any bigger overfits)
• `n_rnn_layers=2`
• Early stopping on val_loss with patience=5

Bigger ≠ better on 5,000 rows.

4/ 📊 Expected result: MAE close to the $337 floor. Directional accuracy near 0.50. Sharpe near zero.

That's not the model failing. That's the honest result on hourly BTC: more capacity does not conjure signal from a near-i.i.d. process. 🪙

5/ 🔍 If the LSTM *does* beat XGBoost, it's the covariates.

XGBoost saw `log_volume_{T-1}` as a scalar. The LSTM saw the full 48-bar trajectory of volume.

Sometimes that richer temporal context extracts signal a snapshot misses. Sometimes it doesn't.

6/ 🚧 The LSTM's fundamental limit: every prediction must squeeze through that 32-dim hidden state.

If the relevant bar is 40 steps back, the model had to keep that signal alive through 39 forget-gate applications.

That's what the Transformer fixes. Article 4 →

Full article → [link]

#MachineLearning #DeepLearning

---

## Bluesky

🧠 **I gave an LSTM a 48-hour memory of Bitcoin. It still couldn't beat the $337 naive floor.**

The LSTM's inductive bias is fundamentally different from XGBoost: it processes the sequence one bar at a time, compressing the past into a 32-dim hidden state. Trees don't care about order; LSTMs do. ⚙️

The new ingredient here: past covariates as *sequences* rather than scalars. XGBoost saw `log_volume_{T-1}` as one number. The LSTM sees the full 48-bar trajectory of volume.

📉 Expected result: MAE near the floor. That's the honest signal on hourly BTC — more model capacity doesn't conjure signal from a near-i.i.d. process.

The architectural limit that sets up article 4 → the LSTM has to compress all of history into a fixed-size state. The Transformer doesn't.

Full article → [link]

#MachineLearning #DeepLearning

---

## LinkedIn

🧠 **I gave a recurrent neural network a memory of Bitcoin's last 48 hours. It still couldn't predict the next bar.**

Article 3 of this forecasting series introduces the first deep-learning model: an LSTM (long short-term memory network) with past covariates.

**🔄 A fundamentally different inductive bias.** XGBoost saw 31 features per bar and made an unordered prediction — a tree doesn't care which column comes first. The LSTM processes the sequence one bar at a time, maintaining a hidden state that summarizes everything it has seen. The prediction is no longer a function of a row of features; it's a function of the entire *trajectory* of bars leading up to the target.

**⚙️ The architecture in one paragraph.** Three gates (forget, input, output) decide what to discard, what to write to the cell state, and what to expose as the hidden output. With `hidden_dim=32`, the entire 48-bar history of Bitcoin trading gets compressed into 32 numbers before any prediction is made. That bottleneck matters.

**📊 Past covariates as sequences.** XGBoost saw `log_volume_{T-1}` as a single scalar feature — yesterday's value only. The LSTM consumes the full 48-bar trajectory of volume, high-low range, and candle body alongside the target log-return. Same information, much richer temporal context.

⚠️ The expected result: MAE close to the $337 naive floor, directional accuracy near 0.50, Sharpe near zero. This is the honest finding on hourly BTC. More model capacity does not conjure signal from a near-i.i.d. process — and that conclusion holds whether the model is a 7-parameter ARIMA or a 30,000-parameter recurrent neural network.

💡 Where LSTM can still surprise: directional accuracy and Sharpe sometimes diverge from MAE. Sequential patterns — trajectory shapes rather than point values — can carry weak directional signal that tabular trees cannot represent. The ablation (turning off covariates) tells you whether the signal was real.

🚧 The LSTM's fundamental limit: every prediction must squeeze through that 32-dim hidden state. If the most relevant bar is 40 steps back, the model had to keep that signal alive through 39 forget-gate applications. That's exactly what the Transformer's attention mechanism solves — and that's article 4.

Article 3 is live → [link]

#MachineLearning #DeepLearning #TimeSeries
