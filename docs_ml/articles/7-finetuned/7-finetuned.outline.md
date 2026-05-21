# Outline — Fine-tuned Foundation Models

## Opening hook
- Zero-shot foundation models lost to ARIMA
- But what if we fine-tune? Start from the general prior and update weights on Bitcoin?
- The question: can transfer learning beat a hand-crafted statistical model?

## The three conditions for success
- Enough training data (4.7 years vs. 1.6 years)
- The right fine-tuning recipe (encoder-only vs. head-only vs. full)
- Checkpoint restoration (best-val_loss weights, not end-of-training)

## Fine-tuning mechanics
- Start with pretrained weights (28M or 200M parameters)
- Set aside 366 bars for validation (no leakage)
- Train for up to 50 epochs with early stopping (monitor val_loss)
- Restore best-val_loss checkpoint before test inference
- Why checkpoint restoration: prevents silent overfitting

## The canonical result: 4.7-year encoder-only Chronos-2 small
- MAE: 549.85 USD
- RMSE: 813.80 USD
- Directional accuracy: **0.5683** (better than ARIMA's 0.5082!)
- Cumulative return: **0.7329** (73.29% - better than ARIMA's 52.69%)
- Sharpe: **8.1987** (on single seed)
- **5-seed mean Sharpe: 5.523 ± 0.418** (95% CI 5.004-6.043) - the number to trust

## The caveat upfront: single seed sits above CI
- On-disk Sharpe 8.20 is a lucky seed
- Seed mean Sharpe is 5.52
- CI lower bound is 5.00 - still clears zero-shot at 95% confidence
- Per-bar Sharpe ratio at CI lower bound: 2.04 σ (borderline significant)

## Why it beats zero-shot
- More training data: 4.7 years vs. the zero-shot's implicit training on general data
- Right recipe: encoder-only (freeze the head, fine-tune the encoder)
- Checkpoint restoration: use the best validation loss weights, not the last epoch

## What encoder-only means
- The encoder is the feature extraction part (~94% of parameters)
- The output head is the regression head (~6% of parameters)
- Freeze: keep the head static
- Finetune: update the encoder on Bitcoin data
- Intuition: learn Bitcoin-specific patterns while keeping the general shape

## The variants
- Same backend (chronos-2-small), different train slices (1.6 yr vs. 4.7 yr)
- Same backend, different recipes (head-only vs. encoder-only vs. full)
- Different backend (timesfm), same recipe (encoder-only)
- Chronos-2-large (filename misnamed - actually small)

## Key variant insights
- Train slice length is load-bearing: 4.7 yr + encoder-only = 8.20 Sharpe
- Same backend, 1.6 yr + head-only = 5.60 Sharpe (46% worse)
- Backend matters: chronos encoder-only 8.20 vs. timesfm encoder-only 5.39
- Encoder-only pairs better with T5-style encoder than decoder-only

## The sweep: recipe × learning rate on 1.6-yr data
- 10 configurations on the 1.6-yr sibling (former canonical)
- Encoder-only + 1e-5 wins on Sharpe (6.87) and dir_acc (0.5464)
- Same recipe direction wins on 4.7-yr slice (8.20 Sharpe)
- Best val_loss doesn't predict best Sharpe (same lesson as ARIMA)

## The critical insight: in-sample loss doesn't rank out-of-sample metrics
- Sweep's lowest val_loss row (full × 1e-5) posts Sharpe 5.73
- Sharpe leader posts higher val_loss (encoder-only × 1e-5, val_loss 0.03505)
- Selecting fine-tuning recipe by val_loss alone misses the trading-metric winner
- Same pattern as ARIMA: different metrics rank differently

## What fine-tuning traded for Sharpe
- MAE stayed high: 549.85 vs. ARIMA's 539.15
- But dir_acc went up: 0.5683 vs. ARIMA's 0.5082
- Cumulative return jumped: 73.29% vs. ARIMA's 52.69%
- The gain is prediction magnitude calibration, not point error

## When fine-tuning beats zero-shot
- Zero-shot Sharpe ~3.0 → Fine-tuned Sharpe ~5.5 (5-seed mean)
- The conditions: 4.7-year train slice, encoder-only recipe, checkpoint restoration
- Swap any one out: the lift collapses
- Proof: 1.6-yr head-only is Sharpe 5.60 (vs. 8.20)

## The honest bottom line
- Single seed 8.20 is lucky
- Reproducible number is 5-seed mean 5.52 (CI 5.00-6.04)
- Per-bar Sharpe ratio at CI lower bound: ~2.04 σ (borderline)
- Still significant vs. zero-shot (mean 2.97)

## Caveats
- Single seed on disk; trust the 5-seed CI from README instead
- Frozen weights after fine-tuning (no retraining during test)
- Univariate only
- One regime (post-election rally)
- Encoder-only recipe only works well with encoder-heavy architectures (good for T5, bad for decoder-only)

## Closing
- Fine-tuning works under the right conditions
- Not the right conditions being "throw more parameters at it"
- The right conditions: enough data, the right architectural recipe, proper checkpoint handling
- The series ends here, but the question remains open: how much data does each architecture need?
