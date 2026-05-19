# docs_ml — ML article series principles

A short manifesto for the article series in this folder. Five articles, published on a personal blog and cross-posted to LinkedIn, on machine learning applied to Bitcoin price prediction.

The goal is to showcase **ML craft, not trading intuition**. The financial-flavor articles in `../docs_finance/articles/` are explicitly out of scope and should not be used as templates or sources of prose.

## Audience

ML engineers and curious technical readers. They want to see:

- How the problem was framed.
- What the data looks like.
- What models were tried and *why*.
- What was honest about evaluation.
- Code they can actually run.

They do **not** want trading advice, get-rich framing, or finance jargon.

## The five articles

1. **Baselines you need to beat** — project setup, data, walk-forward evaluation methodology, naive / moving-average / ARIMA baselines.
2. **XGBoost and feature engineering** — what classical ML brings to the table: the model is generic, the features carry the signal.
3. **LSTM** — recurrent nets on raw sequences. Sequence length, training stability, what they do and don't learn.
4. **Transformer** — attention applied to time series, via the Temporal Fusion Transformer in Darts.
5. **Zero-shot foundation models** — Chronos-2 and TimesFM 2.5. What it means that a model that has never seen Bitcoin has an opinion.

## Voice principles

- **Code-first.** Every claim is backed by a file path in this repo. Articles deep-link to `experiments/NN_*/`.
- **Show the losses.** When ARIMA beats the transformer, say so. Showing what didn't work is the most ML-credible thing in the series.
- **Intuition, then math.** Lead with what the model is doing in plain English. Equations are optional and should serve the intuition, not replace it.
- **No financial framing.** Don't talk about returns, Sharpe ratios, or strategies as if the goal were to trade. Talk about forecasting accuracy, model behavior, and ML methodology.
- **Methodology hygiene visible.** Walk-forward splits, look-ahead leakage, baselines — these get airtime because they are what an ML reader judges you on.

## Structure of each article

A loose template, not a rigid one:

1. **Hook** — one paragraph. What this article is about and why an ML reader should care.
2. **Setup** — data, target, evaluation. Article 1 establishes this; later articles recap in one short paragraph.
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
