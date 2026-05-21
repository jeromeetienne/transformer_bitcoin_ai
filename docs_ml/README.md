# docs_ml — ML article series principles

A short manifesto for the article series in this folder. Eight articles (one project-intro piece plus seven model articles, one per experiment), on machine learning applied to Bitcoin price prediction.

Article numbering matches the experiment number: article `N` maps to [`experiments/0N_*/`](../experiments/), and the working title is driven by that experiment. Article 0 (the project presentation) is the only article without a matching experiment.

## Goals

This series exists for two reasons. The first is to demonstrate applied machine learning knowledge across the full model spectrum — from naive baselines and classical statistics through gradient boosting, recurrent networks, attention-based architectures, and modern foundation models. The second is to show that I can communicate technical work clearly: framing a real problem, explaining what each model does and why it was chosen, and drawing honest conclusions from results.

The intended reader is a technical hiring manager or machine learning practitioner who wants to see both depth of knowledge and quality of reasoning — not just that experiments ran, but that I understood what they revealed.

## Style

See [STYLE.md](STYLE.md) for writing conventions, tone, and title guidelines.

## Folder hierarchy

- [README.md](README.md) - this file: series principles, goals, requirements.
- [STYLE.md](STYLE.md) - writing conventions, tone, title guidelines, and hard linking/formatting requirements.
- [generation_article.md](generation_article.md) - process for generating an article from its experiment report.
- [generation_report.md](generation_report.md) - process for generating an experiment report from its results.
- [generation_social.md](generation_social.md) - process for generating social media posts from an article.
- [generation_image.md](generation_image.md) - process for generating cover-image prompts (one per article) to paste into claude.ai.
- [reports/](reports/) - one Markdown report per experiment (`NN_<name>.report.md`), plus `XX_global.report.md` for the cross-experiment synthesis. These are the source material the articles draw from.
- [articles/](articles/) - one folder per article, named `N-<slug>/`, plus [articles/generate_poster_images.md](articles/generate_poster_images.md) holding the per-article cover-image prompts. Each article folder contains:
  - `<slug>.outline.md` - the article outline.
  - `<slug>.article.md` - the article draft.
  - `<slug>.socialmedia.md` - the matching social media posts.

## The eight articles

0. **Presentation of the project** — the curtain-raiser. Frames the question, the data, and the evaluation philosophy, and previews the model articles that follow. No model, no results.
1. **Baseline** ([`01_baseline`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/01_baseline)) — naive last-value as the zero-parameter floor every later model must clear. Also where the methodology (target, slice, split, walk-forward, metric module) is established.
2. **ARIMA** ([`02_arima`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/02_arima)) — the 3-parameter linear statistical floor. The classical companion to article 1.
3. **XGBoost** ([`03_xgboost`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/03_xgboost)) — what classical ML brings to the table: the model is generic, the features carry the signal.
4. **LSTM** ([`04_lstm`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/04_lstm)) — recurrent nets on raw sequences. Sequence length, training stability, what they do and don't learn.
5. **Transformer** ([`05_transformer`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/05_transformer)) — attention applied to time series, via the Temporal Fusion Transformer in Darts.
6. **Pretrained foundation models** ([`06_pretrained`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/06_pretrained)) — zero-shot Chronos-2 and TimesFM 2.5. What it means that a model that has never seen Bitcoin has an opinion.
7. **Fine-tuned foundation models** ([`07_finetuned`](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/07_finetuned)) — the same backbones as article 6, but `fit()` updates weights against a held-out validation slice. When fine-tuning beats zero-shot, and the conditions that make it happen.

