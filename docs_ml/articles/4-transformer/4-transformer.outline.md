# Outline: Transformer — Attention on Time Series

## 1. The problem with recurrent memory
- The LSTM's fixed-size hidden state as an information bottleneck
- Long-range dependencies and the vanishing gradient problem revisited
- Why attention is a structurally different answer

## 2. The Temporal Fusion Transformer
- Why TFT rather than a vanilla Transformer
- The four components: variable selection, LSTM encoder/decoder, multi-head attention, gated residual networks
- Past covariates vs. future covariates: the key architectural distinction

## 3. Future covariates: the new input channel
- Cyclical time encodings: hour-of-day, day-of-week (sin/cos)
- Why these are legitimate future-known signals
- The intraday seasonality hypothesis: does BTC have one?

## 4. The experiment setup
- Darts TFTModel: same harness as LSTM, only model class changes
- Three-way split and early stopping
- Walk-forward via historical_forecasts(retrain=False)

## 5. The hyperparameter surface
- hidden_size × num_attention_heads: the capacity-interpretability trade
- lstm_layers, dropout, hidden_continuous_size
- The ablation: turning off time features to isolate the attention contribution

## 6. The numbers and what they mean
- MAE / directional accuracy / Sharpe against the full baseline table
- When TFT beats LSTM and the likely explanation
- Attention is not interpretation: a caution on reading too much into the attention maps

## 7. Setting up the foundation model articles
- What TFT cannot do: it has only seen BTC
- What a pretrained model brings: a prior over all time series
