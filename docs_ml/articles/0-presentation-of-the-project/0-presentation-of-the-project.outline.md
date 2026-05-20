# Outline: Presentation of the Project

## 1. The question
- What are we actually trying to predict, and why Bitcoin?
- The appeal and the trap: high liquidity, 24/7 data, huge noise-to-signal ratio
- Framing it as a forecasting problem, not a trading system

## 2. The data
- OHLCV from Binance: symbol, intervals (1h, 4h), date range
- What the shared data loader does (src/btc_ai/data)
- Why closing price is the target (not returns, not log-returns)

## 3. The evaluation philosophy
- Time-ordered train/test split — no shuffling, ever
- Walk-forward evaluation: each prediction uses only past actuals
- The four metrics: MAE, RMSE, MAPE, directional accuracy
- Why directional accuracy matters for trading even when MAE doesn't move

## 4. The model lineup
- Brief map of articles 1–7: naive → ARIMA → XGBoost → LSTM → Transformer → Chronos/TimesFM → fine-tuned
- The progression: parameter count and inductive bias, not just complexity
- What "beating the baseline" means in this context

## 5. What this series is and isn't
- Not a trading system; no backtesting, no live deployment
- Not a leaderboard chase: honest results over cherry-picked configs
- A reproducible ML study with a clear floor and a clear ceiling
