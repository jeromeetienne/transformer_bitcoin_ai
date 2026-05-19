# article_todo.md

Working tracker for the five-article ML series. See [README.md](README.md) for the editorial principles.

## Status at a glance

| #   | Article                                 | Source experiments                                                   | Status      |
| --- | --------------------------------------- | -------------------------------------------------------------------- | ----------- |
| 1   | Baselines you need to beat              | `01_baseline_naive`, `01b_moving_average`, `02_arima`                | not started |
| 2   | XGBoost and feature engineering         | `03_gradient_boosting`                                               | not started |
| 3   | LSTM                                    | `04_lstm`                                                            | not started |
| 4   | Transformer                             | `05_transformer`                                                     | not started |
| 5   | Zero-shot foundation models             | `06_pretrained`                                                      | not started |

Status values: `not started` → `outlined` → `drafting` → `numbers verified` → `ready for review` → `published`.

## Series-level decisions

Open knobs to settle before drafting article 1:

- [ ] Blog platform confirmed (and URL pattern decided, so cross-links between articles are stable).
- [ ] Repo visibility — public from article 1 (current plan) vs. publish at end. Confirm.
- [x] Editorial voice — code-first, show the losses, no financial framing. See [README.md](README.md).
- [x] Article boundaries — 5 articles, with pure baselines (incl. ARIMA) in article 1 and XGBoost split out into article 2.
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

## Article 1 — Baselines you need to beat

Working title: *"Forecasting Bitcoin with ML, Part 1: the problem, the data, and the baselines you need to beat."*

Carries the project-introduction load for the series. Frames the problem, the data, and the walk-forward evaluation methodology, then runs through naive / moving-average / ARIMA as the floor every later model must clear.

- Source experiments: `experiments/01_baseline_naive/`, `experiments/01b_moving_average/`, `experiments/02_arima/`.
- Raw material from prior series: `docs_original_finance/reports/01_baseline_naive.report.md`, `01b_moving_average.report.md`, `02_arima.report.md`. Pull numbers and structure; rewrite voice.
- Key beats to surface:
  - Why a one-line "predict the last value" baseline is hard to beat on short horizons.
  - What walk-forward evaluation is and why it matters here (no look-ahead leakage).
  - ARIMA as the strongest baseline — set up the recurring theme that simple models are hard to beat.
- Open questions:
  - How much methodology to put in article 1 vs. how much to recap in later articles.

## Article 2 — XGBoost and feature engineering

Working title: TBD. Candidate angle: *"When the features do the work — XGBoost on engineered Bitcoin features."*

Classical-ML article. The point is that the model is generic and the features carry the signal — the opposite contract from the deep-learning articles that follow.

- Source experiment: `experiments/03_gradient_boosting/`.
- Raw material: `docs_original_finance/reports/03_gradient_boosting.report.md`.
- Key beats to surface:
  - What feature engineering looked like here (lagged returns, rolling stats, volatility, etc.).
  - The "generic model + engineered features" contract.
  - Honest result vs. ARIMA from article 1.
- Open questions:
  - Whether to spend any time on XGBoost mechanics (gradient boosting, second-order Taylor expansion, regularization) or assume reader familiarity.

## Article 3 — LSTM

Working title: TBD. Candidate angle: *"What an LSTM learns from a window of Bitcoin history."*

First deep-learning article. The model now consumes a sequence (plus a few past covariates — volume, OHLC range and body) rather than the flat, hand-crafted feature vector that XGBoost was fed.

- Source experiment: `experiments/04_lstm/`.
- Raw material: `docs_original_finance/reports/04_lstm.report.md`.
- Key beats to surface:
  - Sequence length / window choice and what it implies.
  - Training stability and what tuning actually mattered.
  - Result vs. the classical methods — keep it honest if classical wins.
- Open questions:
  - How much LSTM architecture to explain (gates, cell state) vs. linking out.

## Article 4 — Transformer

Working title: TBD. Candidate angle: *"Attention on a price series — and what it changes (or doesn't)."*

The titular model of the repo. Use the Temporal Fusion Transformer from Darts.

- Source experiment: `experiments/05_transformer/`.
- Raw material: `docs_original_finance/reports/05_transformer.report.md`.
- Key beats to surface:
  - What attention buys you on a time series (in plain English).
  - TFT-specific bits (variable selection, gating) only if they actually mattered for this problem.
  - Honest comparison vs. LSTM and ARIMA — don't pretend transformer wins if it doesn't.
- Open questions:
  - Whether to lean into the "the repo is named after this and ARIMA still won" angle if that's the result.

## Article 5 — Zero-shot foundation models

Working title: TBD. Candidate angle: *"What does a model that has never seen Bitcoin think Bitcoin will do?"*

The capstone. Two off-the-shelf pretrained models (Chronos-2, TimesFM 2.5) run zero-shot on the BTC series — no training, no fine-tuning.

- Source experiment: `experiments/06_pretrained/`.
- Raw material: `docs_original_finance/reports/06_pretrained.report.md`.
- Key beats to surface:
  - What zero-shot forecasting means and why time-series foundation models exist.
  - The probabilistic-forecast angle (Q10/Q50/Q90 bands) if the experiment surfaces it.
  - Honest comparison vs. every prior article — how does a model that's never seen BTC compare to one that's trained on it?
- Open questions:
  - Whether to wrap the series with a "where this goes next" closing section, or save that for a separate epilogue.
