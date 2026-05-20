# articles_todo.md

Working tracker for the six-article ML series (one project-intro piece plus five model articles). See [README.md](README.md) for the editorial principles.

## Status at a glance

| #   | Article                                 | Source experiments                                                   | Status      |
| --- | --------------------------------------- | -------------------------------------------------------------------- | ----------- |
| 0   | Presentation of the project             | — (meta-article, no model)                                           | not started |
| 1   | Baselines you need to beat              | `01_baseline`, `02_arima`                                      | not started |
| 2   | XGBoost and feature engineering         | `03_gradient_boosting`                                               | not started |
| 3   | LSTM                                    | `04_lstm`                                                            | not started |
| 4   | Transformer                             | `05_transformer`                                                     | not started |
| 5   | Zero-shot foundation models             | `06_pretrained`                                                      | not started |

Status values: `not started` → `outlined` → `drafting` → `numbers verified` → `ready for review` → `published`.

## Series-level decisions

Open knobs to settle before drafting article 0:

- [ ] Blog platform confirmed (and URL pattern decided, so cross-links between articles are stable).
- [ ] Repo visibility — public from article 0 (current plan) vs. publish at end. Confirm.
- [x] Editorial voice — code-first, show the losses, no financial framing. See [README.md](README.md).
- [x] Article boundaries — 6 articles: article 0 = project intro (no model); article 1 = pure baselines incl. ARIMA; article 2 = XGBoost; articles 3–5 = LSTM, transformer, zero-shot.
- [ ] Spoil the "ARIMA beats deep learning" punchline in article 0, or save it for article 4? Decide before drafting article 0.
- [ ] Personal-credibility paragraph in article 0 — include (helps the "make me look good in ML" goal) or skip (risks bragging)? Decide before drafting.
- [ ] Methodology split — high-level framing in article 0 vs. specifics (split indices, log-return reconstruction, metric definitions) in article 1. Settle to avoid redundancy.
- [ ] Methodology recap snippet — a 1-paragraph reusable block on data + walk-forward eval, to drop into articles 2–5 instead of re-deriving.
- [ ] Visual conventions — plot style, code-block conventions, whether to include screenshots of plots or describe them.
- [ ] Cover-image strategy for LinkedIn (per-article image? series-wide template?).

## Per-article checklist

The same six-item checklist applies to every article. Tick as you go.

- [ ] **Outline** — bullet list of beats, including the "what it means" punchline.
- [ ] **Draft** — written following the README structure (hook → setup → model → results → meaning → reproduce).
- [ ] **Numbers verified** — every figure in the article comes from the corresponding `experiments/NN_*/results/metrics.json` (or equivalent). No remembered or rounded numbers.
- [ ] **Code links** — at least one deep-link to `experiments/NN_*/` confirmed to resolve on the public repo.
- [ ] **LinkedIn teaser** — 2–4 paragraphs, leading with the most ML-credible beat (often the surprise).
- [ ] **Published** — blog live, LinkedIn cross-posted, internal status above updated.

---

Per-article sections below are deliberately kept short — title plus a short description. Detailed outlines (key beats, open questions, source experiments, raw material) will be generated separately when each article moves from `not started` to `outlined`.

## Article 0 — Presentation of the project

Working title: *"Predicting Bitcoin with machine learning: what this series is about."*

The curtain-raiser. Frames the question (forecasting BTC log-returns as an ML problem), introduces the data and walk-forward evaluation at a high level, walks through the repo, and previews articles 1–5. No model, no results.

## Article 1 — Baselines you need to beat

Working title: *"Forecasting Bitcoin with ML, Part 1: the baselines you need to beat."*

The first model article. Defines target and evaluation specifics, and runs naive / ARIMA as the floor every later model must clear.

## Article 2 — XGBoost and feature engineering

Working title: TBD. Candidate angle: *"When the features do the work — XGBoost on engineered Bitcoin features."*

The classical-ML article. The model is generic, the features carry the signal — the opposite contract from the deep-learning articles that follow.

## Article 3 — LSTM

Working title: TBD. Candidate angle: *"What an LSTM learns from a window of Bitcoin history."*

The first deep-learning article. The model now consumes a sequence (plus a few past covariates) rather than the hand-crafted feature vector that XGBoost was fed.

## Article 4 — Transformer

Working title: TBD. Candidate angle: *"Attention on a price series — and what it changes (or doesn't)."*

The titular model of the repo. The Darts Temporal Fusion Transformer applied to BTC, compared honestly against LSTM and ARIMA.

## Article 5 — Zero-shot foundation models

Working title: TBD. Candidate angle: *"What does a model that has never seen Bitcoin think Bitcoin will do?"*

The capstone. Two off-the-shelf pretrained models (Chronos-2, TimesFM 2.5) run zero-shot on the BTC series — no training, no fine-tuning. Probabilistic forecasts via Q10/Q50/Q90 bands.
