# docs_ml — ML article series principles

A short manifesto for the article series in this folder. Eight articles (one project-intro piece plus seven model articles, one per experiment), on machine learning applied to Bitcoin price prediction.

Article numbering matches the experiment number: article `N` maps to [`experiments/0N_*/`](../experiments/), and the working title is driven by that experiment. Article 0 (the project presentation) is the only article without a matching experiment.

## Goals

This series exists for two reasons. The first is to demonstrate applied machine learning knowledge across the full model spectrum — from naive baselines and classical statistics through gradient boosting, recurrent networks, attention-based architectures, and modern foundation models. The second is to show that I can communicate technical work clearly: framing a real problem, explaining what each model does and why it was chosen, and drawing honest conclusions from results.

The intended reader is a technical hiring manager or machine learning practitioner who wants to see both depth of knowledge and quality of reasoning — not just that experiments ran, but that I understood what they revealed.

## Style

Each article is written as a self-contained piece, not a notebook dump. The structure is consistent: what problem this model class solves, how it was set up here, what the numbers say, and what that means for the broader picture. Results are never cherry-picked — the baseline exists precisely so every subsequent model can be judged against a floor.

The tone is direct and precise. Mathematical notation is used when it adds clarity, not to perform rigor. Jargon is earned: a term like "temporal fusion" is introduced when it describes something specific, then used without apology. The series treats the reader as an intelligent adult who does not need hand-holding but does deserve an honest explanation.

The writing is also entertaining and didactic. A reader who does not work in machine learning should still be able to follow along, enjoy the ride, and come away having learned something real. Analogies, concrete intuitions, and a bit of wit make the material accessible without making it shallow. The goal is that someone curious about how machine learning actually works — not just that it exists — finds each article worth reading for its own sake, not just as a portfolio artifact.

## The eight articles

0. **Presentation of the project** — the curtain-raiser. Frames the question, the data, and the evaluation philosophy, and previews the model articles that follow. No model, no results.
1. **Baseline** ([`01_baseline`](../experiments/01_baseline/)) — naive last-value as the zero-parameter floor every later model must clear. Also where the methodology (target, slice, split, walk-forward, metric module) is established.
2. **ARIMA** ([`02_arima`](../experiments/02_arima/)) — the 3-parameter linear statistical floor. The classical companion to article 1.
3. **XGBoost** ([`03_xgboost`](../experiments/03_xgboost/)) — what classical ML brings to the table: the model is generic, the features carry the signal.
4. **LSTM** ([`04_lstm`](../experiments/04_lstm/)) — recurrent nets on raw sequences. Sequence length, training stability, what they do and don't learn.
5. **Transformer** ([`05_transformer`](../experiments/05_transformer/)) — attention applied to time series, via the Temporal Fusion Transformer in Darts.
6. **Pretrained foundation models** ([`06_pretrained`](../experiments/06_pretrained/)) — zero-shot Chronos-2 and TimesFM 2.5. What it means that a model that has never seen Bitcoin has an opinion.
7. **Fine-tuned foundation models** ([`07_finetuned`](../experiments/07_finetuned/)) — the same backbones as article 6, but `fit()` updates weights against a held-out validation slice. When fine-tuning beats zero-shot, and the conditions that make it happen.
