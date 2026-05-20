# articles_todo.md

Working tracker for the seven-article ML series (one project-intro piece plus six model articles). See [README.md](README.md) for the editorial principles. **Note:** the series was originally scoped as six articles (0–5), with `07_finetuned` flagged as out-of-scope in article 5's wrap-up. Article 6 was added later as a follow-up to article 5's "does fine-tuning close the zero-shot gap?" question.

## Status at a glance

| #   | Article                                 | Source experiments                                                   | Status            |
| --- | --------------------------------------- | -------------------------------------------------------------------- | ----------------- |
| 0   | Presentation of the project             | — (meta-article, no model)                                           | numbers verified  |
| 1   | Baselines you need to beat              | `01_baseline`, `02_arima`                                            | numbers verified  |
| 2   | XGBoost and feature engineering         | `03_xgboost`                                                         | numbers verified  |
| 3   | LSTM                                    | `04_lstm`                                                            | numbers verified  |
| 4   | Transformer                             | `05_transformer`                                                     | numbers verified  |
| 5   | Zero-shot foundation models             | `06_pretrained`                                                      | numbers verified  |
| 6   | Fine-tuned foundation models            | `07_finetuned`                                                       | numbers verified  |

Status values: `not started` → `outlined` → `drafting` → `numbers verified` → `ready for review` → `published`.

## Series-level decisions

Open knobs to settle before drafting article 0:

- [ ] Blog platform confirmed (and URL pattern decided, so cross-links between articles are stable).
- [ ] Repo visibility — public from article 0 (current plan) vs. publish at end. Confirm.
- [x] Editorial voice — code-first, show the losses, no financial framing. See [README.md](README.md).
- [x] Article boundaries — **revised to 7 articles**: article 0 = project intro (no model); article 1 = pure baselines incl. ARIMA; article 2 = XGBoost; articles 3–5 = LSTM, transformer, zero-shot; article 6 = fine-tuned foundation models (`07_finetuned`). Original plan was 6 articles with 07 out-of-scope; article 6 was added later to answer article 5's natural follow-up.
- [x] Spoil the "ARIMA beats deep learning" punchline in article 0, or save it for article 4? **Spoiled in article 0** — the series sells on the surprise; hiding it for four articles would flatten the hook.
- [x] Personal-credibility paragraph in article 0 — include or skip? **Skipped** — editorial voice is code-first; the repo is the credibility.
- [x] Methodology split — high-level framing in article 0 vs. specifics in article 1. **Settled**: article 0 frames data + walk-forward + metric philosophy; article 1 walks the actual split indices, the `dynamic=False` flag, the metric module code.
- [x] Methodology recap snippet — a 1-paragraph reusable block on data + walk-forward eval, to drop into articles 2–5 instead of re-deriving. **Done** — same paragraph (with the model line swapped) opens articles 1 through 5.
- [ ] Visual conventions — plot style, code-block conventions, whether to include screenshots of plots or describe them. (Current drafts describe plots in prose and quote configs / metrics as code blocks; no `plot.png` screenshots embedded yet.)
- [ ] Cover-image strategy for LinkedIn (per-article image? series-wide template?).

## Per-article checklist

The same six-item checklist applies to every article. Tick as you go. (Counts reflect status across all seven articles after the article-6 addition.)

- [x] **Outline** (7 / 7) — bullet list of beats, including the "what it means" punchline. See `articles/${index}-${slug}/${index}-${slug}.outline.md` per article.
- [x] **Draft** (7 / 7) — written following the README structure (hook → setup → model → results → meaning → reproduce). See `articles/${index}-${slug}/${index}-${slug}.md` per article.
- [x] **Numbers verified** (7 / 7) — every figure quoted from the corresponding `experiments/NN_*/results/.../metrics.json`. Sweep tables in articles 1–4 and 6 are quoted from the corresponding `sweep.csv`; articles 2, 4, and 6 flag stale-slice issues explicitly. Article 6 additionally cites the 5-seed CI from `experiments/07_finetuned/README.md` alongside the single-seed `metrics.json` row.
- [x] **Code links** (7 / 7) — every article deep-links to `experiments/NN_*/run.py`, `configs/*.config.yaml`, and / or `results/.../metrics.json`. Articles 2 and 3 additionally link `experiments/03_xgboost/features.py` and `src/btc_ai/eval/metrics.py`. Article 6 links the `btc_4h_2020_2024.dataset.yaml` extended-slice config and the encoder-only fine-tuning recipe.
- [ ] **LinkedIn teaser** (0 / 7) — 2–4 paragraphs, leading with the most ML-credible beat (often the surprise).
- [ ] **Published** (0 / 7) — blog live, LinkedIn cross-posted, internal status above updated.

---

Per-article sections below are deliberately kept short — title plus a short description. Detailed outlines now live under `articles/${index}-${slug}/${index}-${slug}.outline.md`; full drafts live alongside in `articles/${index}-${slug}/${index}-${slug}.md`.

## Article 0 — Presentation of the project

Working title: *"Predicting Bitcoin with machine learning: what this series is about."*

The curtain-raiser. Frames the question (forecasting BTC log-returns as an ML problem), introduces the data and walk-forward evaluation at a high level, walks through the repo, and previews articles 1–5. No model, no results.

- Outline: [`articles/0-presentation-of-the-project/0-presentation-of-the-project.outline.md`](articles/0-presentation-of-the-project/0-presentation-of-the-project.outline.md)
- Draft: [`articles/0-presentation-of-the-project/0-presentation-of-the-project.md`](articles/0-presentation-of-the-project/0-presentation-of-the-project.md)

## Article 1 — Baselines you need to beat

Working title: *"Forecasting Bitcoin with machine learning, Part 1: the baselines you need to beat."*

The first model article. Defines target and evaluation specifics, and runs naive / ARIMA as the floor every later model must clear.

- Outline: [`articles/1-baselines-you-need-to-beat/1-baselines-you-need-to-beat.outline.md`](articles/1-baselines-you-need-to-beat/1-baselines-you-need-to-beat.outline.md)
- Draft: [`articles/1-baselines-you-need-to-beat/1-baselines-you-need-to-beat.md`](articles/1-baselines-you-need-to-beat/1-baselines-you-need-to-beat.md)
- Headline numbers (verbatim from `metrics.json`): naive MAE 540.96; ARIMA(3, 1, 3) MAE 539.15, Sharpe 6.8559, dir_acc 0.5082 on 366 test bars.

## Article 2 — XGBoost and feature engineering

Working title: *"When the features do the work — XGBoost on engineered Bitcoin features."*

The classical-machine-learning article. The model is generic, the features carry the signal — the opposite contract from the deep-learning articles that follow.

- Outline: [`articles/2-xgboost-and-feature-engineering/2-xgboost-and-feature-engineering.outline.md`](articles/2-xgboost-and-feature-engineering/2-xgboost-and-feature-engineering.outline.md)
- Draft: [`articles/2-xgboost-and-feature-engineering/2-xgboost-and-feature-engineering.md`](articles/2-xgboost-and-feature-engineering/2-xgboost-and-feature-engineering.md)
- Headline numbers: MAE 550.85, Sharpe 6.1388, dir_acc 0.5082 — ties ARIMA on direction, loses on MAE and Sharpe.
- Sweep table on stale 1h slice — flagged explicitly in the draft; pending re-run via `make 03_xgboost_sweep`.

## Article 3 — LSTM

Working title: *"What an LSTM learns from a window of Bitcoin history."*

The first deep-learning article. The model now consumes a sequence (plus a few past covariates) rather than the hand-crafted feature vector that XGBoost was fed.

- Outline: [`articles/3-lstm/3-lstm.outline.md`](articles/3-lstm/3-lstm.outline.md)
- Draft: [`articles/3-lstm/3-lstm.md`](articles/3-lstm/3-lstm.md)
- Headline numbers: MAE 539.23, dir_acc 0.5301 (first model to beat ARIMA on direction), Sharpe 4.8221 (loses to ARIMA's 6.86 — first articulation of the dir_acc-vs-Sharpe split).
- Sweep on current 4h slice; NaN-Sharpe row explained as "model that never goes long" failure mode.

## Article 4 — Transformer

Working title: *"Attention on a price series — and what it changes (or doesn't)."*

The titular model of the repo. The Darts Temporal Fusion Transformer applied to Bitcoin, compared honestly against LSTM and ARIMA.

- Outline: [`articles/4-transformer/4-transformer.outline.md`](articles/4-transformer/4-transformer.outline.md)
- Draft: [`articles/4-transformer/4-transformer.md`](articles/4-transformer/4-transformer.md)
- Headline numbers: MAE 891.09 (worst in the leaderboard, 65 % above naive), dir_acc 0.5055 (coin-flip), Sharpe 2.5390 (1.04 σ — not significant). The cleanest example of "positive Sharpe with worst-in-leaderboard MAE on a slice that flatters everything."
- Sweep on stale 1h slice — flagged in draft; pending re-run via `make 05_transformer_sweep`.

## Article 5 — Zero-shot foundation models

Working title: *"What does a model that has never seen Bitcoin think Bitcoin will do?"*

Originally the series capstone. Two off-the-shelf pretrained models (Chronos-2 in two sizes, TimesFM 2.5) run zero-shot on the Bitcoin series — no training, no fine-tuning. Probabilistic forecasts via Q10 / Q50 / Q90 bands.

- Outline: [`articles/5-zero-shot-foundation-models/5-zero-shot-foundation-models.outline.md`](articles/5-zero-shot-foundation-models/5-zero-shot-foundation-models.outline.md)
- Draft: [`articles/5-zero-shot-foundation-models/5-zero-shot-foundation-models.md`](articles/5-zero-shot-foundation-models/5-zero-shot-foundation-models.md)
- Headline numbers: Chronos-2 small (canonical) MAE 546.66, dir_acc 0.4754 (below coin-flip), Sharpe 2.9722; Chronos-2 large uniformly worse; TimesFM 2.5 best zero-shot Sharpe at 4.1080 — still below ARIMA's 6.86.
- Drafted with a series-wide bottom line that called the series complete; **article 6 was added later** to answer the natural follow-up question (does fine-tuning close the zero-shot gap?). Article 5's closing paragraph still flags `07_finetuned` as "out of scope" — that wording is stale and should be revised when this article moves to `ready for review`.

## Article 6 — Fine-tuned foundation models

Working title: *"When fine-tuning beats zero-shot — and the three conditions that make it happen."*

The follow-up article 5 set up. Same two foundation-model backends, but `fit()` now updates weights against a held-out validation slice with `EarlyStopping(monitor='val_loss')` and the best-validation-loss checkpoint is restored before walk-forward inference. The honest story is that fine-tuning takes the Sharpe / direction / cumulative-return crown from ARIMA *under three conditions* — 4.7-year train slice + encoder-only recipe + best-checkpoint restoration — and on a favourable single seed; the 5-seed CI lower bound still clears 06 zero-shot.

- Outline: [`articles/6-finetuned-foundation-models/6-finetuned-foundation-models.outline.md`](articles/6-finetuned-foundation-models/6-finetuned-foundation-models.outline.md)
- Draft: [`articles/6-finetuned-foundation-models/6-finetuned-foundation-models.md`](articles/6-finetuned-foundation-models/6-finetuned-foundation-models.md)
- Headline numbers (single-seed on disk): MAE 549.85, dir_acc **0.5683**, cum_ret **0.7329**, Sharpe **8.1987**. 5-seed mean per `experiments/07_finetuned/README.md`: Sharpe **5.523 ± 0.418**, CI (5.004, 6.043); cum_ret 44.52 % ± 5.48 pp, CI (37.70 %, 51.33 %). Per-bar Sharpe ratio range: 2.04 σ (CI lower bound) to 3.35 σ (single seed on disk).
- ARIMA(3, 1, 3) retains the MAE crown ($10.70 below 07's). The variants table walks the three conditions one swap at a time: backend swap (chronos → timesfm) drops Sharpe to 5.39; train-slice swap (4.7 → 1.6 yr) on head-only drops to 5.60; checkpoint-restoration regression flips honest 3.46 to overfit 6.146 per the README's documented incident.
- Sweep on the 1.6-yr sibling slice; a 4.7-yr sweep does not exist on disk — operational follow-up via `make 07_finetuned_sweep`.
