# Outline: Fine-tuned Foundation Models

## 1. The fine-tuning hypothesis
- From "broad prior" to "BTC-specific calibration"
- Why the naive answer (just call fit()) doesn't work
- The three things that had to be true for fine-tuning to win

## 2. The three hard-won lessons
- Lesson 1: dataset size matters — 1.6 years vs. 4.7 years of training data
- Lesson 2: freeze the head, fine-tune the encoder — why head-only fine-tuning destroys the directional prior
- Lesson 3: restore the best-val_loss checkpoint — the overfit endpoint trap

## 3. The architecture of fine-tuning
- What encoder-only fine-tuning means for Chronos-2
- The freeze/unfreeze pattern syntax and how to verify it worked
- Training curves: val_loss divergence within 1-4 epochs is normal

## 4. The headline result
- Five-seed comparison: mean ± std, 95% confidence intervals
- What the CI tells us: Sharpe and cumulative_return exclude zero-shot at 95%
- The surprising insight: magnitude calibration, not directional bias

## 5. Why magnitude calibration is the interesting story
- Directional accuracy barely above zero-shot (within noise)
- But Sharpe and cumulative_return improve by a meaningful margin
- What this tells us about what fine-tuning actually learned

## 6. Caveats and open questions
- Five seeds on one test regime is a thin evidence base
- num_samples discrepancy between articles 6 and 7
- TimesFM fine-tuning: the pattern doesn't transfer
- The full series leaderboard: where fine-tuned Chronos lands

## 7. Closing: what the series reveals
- The honest scorecard across all seven models
- Where the bottleneck actually is (data, not architecture)
- What the next step would be
