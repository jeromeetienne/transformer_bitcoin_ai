# docs_ml — ML article series principles

A short manifesto for the article series in this folder. Eight articles (one project-intro piece plus seven model articles, one per experiment), published on a personal blog and cross-posted to LinkedIn, on machine learning applied to Bitcoin price prediction.

Article numbering matches the experiment number: article `N` maps to [`experiments/0N_*/`](../experiments/), and the working title is driven by that experiment. Article 0 (the project presentation) is the only article without a matching experiment.

The goal is to showcase **ML craft, not trading intuition**. The financial-flavor articles in `../docs_finance/articles/` are explicitly out of scope and should not be used as templates or sources of prose.

## Audience

ML engineers and curious technical readers. They want to see:

- How the problem was framed.
- What the data looks like.
- What models were tried and *why*.
- What was honest about evaluation.
- Code they can actually run.

They do **not** want trading advice, get-rich framing, or finance jargon.

## The eight articles

0. **Presentation of the project** — the curtain-raiser. Frames the question, the data, and the evaluation philosophy, and previews the model articles that follow. No model, no results.
1. **Baseline** ([`01_baseline`](../experiments/01_baseline/)) — naive last-value as the zero-parameter floor every later model must clear. Also where the methodology (target, slice, split, walk-forward, metric module) is established.
2. **ARIMA** ([`02_arima`](../experiments/02_arima/)) — the 3-parameter linear statistical floor. The classical companion to article 1.
3. **XGBoost** ([`03_xgboost`](../experiments/03_xgboost/)) — what classical ML brings to the table: the model is generic, the features carry the signal.
4. **LSTM** ([`04_lstm`](../experiments/04_lstm/)) — recurrent nets on raw sequences. Sequence length, training stability, what they do and don't learn.
5. **Transformer** ([`05_transformer`](../experiments/05_transformer/)) — attention applied to time series, via the Temporal Fusion Transformer in Darts.
6. **Pretrained foundation models** ([`06_pretrained`](../experiments/06_pretrained/)) — zero-shot Chronos-2 and TimesFM 2.5. What it means that a model that has never seen Bitcoin has an opinion.
7. **Fine-tuned foundation models** ([`07_finetuned`](../experiments/07_finetuned/)) — the same backbones as article 6, but `fit()` updates weights against a held-out validation slice. When fine-tuning beats zero-shot, and the conditions that make it happen.

## Voice principles

- **Code-first.** Every claim is backed by a file path in this repo. Articles deep-link to `experiments/NN_*/`.
- **Show the losses.** When ARIMA beats the transformer, say so. Showing what didn't work is the most ML-credible thing in the series.
- **Intuition, then math.** Lead with what the model is doing in plain English. Equations are optional and should serve the intuition, not replace it.
- **No financial framing.** Don't talk about returns, Sharpe ratios, or strategies as if the goal were to trade. Talk about forecasting accuracy, model behavior, and ML methodology.
- **Methodology hygiene visible.** Walk-forward splits, look-ahead leakage, baselines — these get airtime because they are what an ML reader judges you on.

## Structure of each article

A loose template, not a rigid one:

1. **Hook** — one paragraph. What this article is about and why an ML reader should care.
2. **Setup** — data, target, evaluation. Articles 0 and 1 establish this between them (framing in 0, specifics in 1); later articles recap in one short paragraph.
3. **The model** — intuition first, then enough detail that the reader knows what is actually being trained.
4. **What happened** — results. Include the surprises.
5. **What it means** — one short section. Not "what you should do" but "what this tells us about the model class."
6. **Reproduce** — the `make` command and the path to the experiment folder.

## What to avoid

- Trading strategy backtests, P&L, or Sharpe ratios as the main metric.
- Financial advice in any form, even disclaimed.
- Hype: claims like "transformers crush time series" that are not backed by *your* results.
- Reusing prose from `../docs_finance/articles/`. Different series, different framing.

## Cross-posting

Three tiers:

- **Blog** — the canonical home for each article. Long form. The blog post URL is what every other channel ultimately points back to.
- **LinkedIn article** — the same content as the blog article, re-published as a LinkedIn article. Opens with a standard attribution paragraph noting the piece was originally published on the blog and linking to the blog URL (the usual cross-post wording).
- **LinkedIn posts** — several short posts per article, say 3-4 posts, each teasing one beat and linking to the LinkedIn article. Lead with the most ML-credible beat (often the surprise: *"ARIMA beat my XGBoost"*).

## Workflow

- Drafts live in `docs_ml/articles/NN-slug.md`.
- Code under `experiments/NN_*/` is the source of truth. If an article cites a number, that number comes from the actual `results/metrics.json` in the experiment.
- Per-experiment technical reports under `../docs_finance/reports/` are useful raw material — pull from them, but rewrite for this series' voice.
