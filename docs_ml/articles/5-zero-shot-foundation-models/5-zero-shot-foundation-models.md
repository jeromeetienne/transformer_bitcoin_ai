# A Model That Has Never Seen Bitcoin Has Opinions About It

Every model in the previous four articles learned from Bitcoin. The naive predictor needed no data at all, but it also expressed no opinion about what BTC specifically does. ARIMA fit its three parameters to BTC's training slice. XGBoost fit thousands of tree splits to BTC features. The LSTM and TFT fit tens of thousands of neural network weights to BTC log-returns.

This article does something different. It evaluates two models that have never seen a single bar of Bitcoin data - that have never even been shown a financial time series during training. They were trained on electricity consumption curves, retail sales, hospital admissions, weather station readings, traffic counts, and thousands of other time series from the M competitions, the Monash Time-Series Repository, and other public corpora. Then they were handed a Bitcoin price series and asked: what comes next?

The question is genuine. If a model has learned something true about how time series work in general - that values tend to be autocorrelated, that volatility clusters, that mean reversion is common - those truths might transfer to Bitcoin even without task-specific training. Or they might not. This experiment finds out.

## The two models

**Chronos-2** is Amazon's time-series foundation model, based on the T5 encoder-decoder architecture. T5 was originally a language model; Chronos-2 replaces the text token vocabulary with a quantized representation of time-series values. The model is trained to predict the distribution of future values given a context window, and it ships in several sizes. This experiment uses `autogluon/chronos-2-small` (28M parameters) for the sweep and `amazon/chronos-2` (120M parameters) for the headline run.

**TimesFM 2.5** is Google's time-series foundation model, using a decoder-only patch-transformer architecture. Rather than tokenizing individual values, it processes patches of the input series - short subsequences - which makes it efficient at long context lengths. The current version has approximately 200M parameters and accepts context windows up to 16 384 bars.

Both models are accessed through Darts' `Chronos2Model` and `TimesFM2p5Model` classes, which wrap the HuggingFace Hub downloads behind the same API as the LSTM and TFT experiments. The weights download automatically on the first run and are cached locally thereafter.

## Why univariate only

TimesFM 2.5 does not support covariates. It takes a single time series as input and produces a probabilistic forecast. Chronos-2 does support past and future covariates - its darts wrapper exposes the same mixed-covariate interface as the TFT - but this experiment deliberately keeps both models univariate.

The reason is methodological. If Chronos-2 received volume and OHLC covariates while TimesFM received none, any difference in their results would conflate "which pretrained prior is stronger?" with "did the extra inputs help?" Keeping both univariate isolates the question this article is asking: *does the pretrained prior alone transfer to Bitcoin?*

The covariate ablation for Chronos-2 - feeding it volume and intraday calendar features - is a clean follow-up experiment. It is not this one.

## Probabilistic forecasting

This is the first experiment in the series to produce something beyond a point forecast. Both models output a distribution over future values, sampled at inference time. The experiment draws 200 samples per step and reports three quantiles:

- **q=0.1**: the 10th percentile. The model thinks there is a 90% chance the actual value will be above this.
- **q=0.5**: the median. The model's best single-number prediction, used for all standard leaderboard metrics (MAE, RMSE, MAPE, directional accuracy, Sharpe).
- **q=0.9**: the 90th percentile. The model thinks there is a 90% chance the actual value will be below this.

The q10–q90 interval is shown as a shaded band on `results/plot.png`. A well-calibrated model should see approximately 80% of actual values fall within this band - neither too wide (overconfident) nor too narrow (underconfident). In practice, financial series are heavy-tailed and difficult to calibrate; the interval is often narrower than the realized empirical coverage.

The median is used for the standard metrics rather than the mean, because on heavy-tailed log-return distributions the mean of a stochastic sample is sensitive to occasional extreme draws. The median is more robust.

## The experiment

The experiment is in [experiments/06_pretrained/](https://github.com/jeromeetienne/transformer_bitcoin_ai/tree/HEAD/experiments/06_pretrained). Because there is no training, there is no three-way split: the scaler is fit on the training slice (to normalize the input series to a range the model expects), but the model weights are not updated.

```bash
make 06_pretrained
```

The backend is selected in `config.yaml` - there is no default, and the config requires an explicit choice:

```yaml
backend: chronos
hub_model_name: amazon/chronos-2
```

Or:

```yaml
backend: timesfm
hub_model_name: google/timesfm-2.5-200m-pytorch
```

The `model.fit()` call is a no-op for foundation models in Darts - it is required by the API but performs no weight updates. The actual work happens in:

```python
model.historical_forecasts(
    start=test_start,
    forecast_horizon=1,
    retrain=False,
    last_points_only=True,
    num_samples=200,
)
```

This slides the input window across the 437-bar test slice, predicting one step ahead at each position using only actual past values. At `num_samples=200`, each step draws 200 samples from the model's predictive distribution. Walk-forward over 437 bars takes single-digit minutes on Apple Metal Performance Shaders or a recent GPU; on CPU it can be several times slower.

The sweep covers different backends, model sizes, and context lengths:

```bash
make 06_pretrained_sweep
```

## How to interpret the results

The expected outcome: MAE near the naive floor (~$338), directional accuracy near 0.50, Sharpe near zero. This is not a failure of the models - it is the expected result of applying a model with a broad prior to an asset class whose near-term dynamics are largely random.

Here is why this expectation is well-founded. The models were trained on time series that typically exhibit strong autocorrelation, clear seasonality, and mean-reversion properties. Retail sales go up before Christmas. Electricity consumption peaks in the evening. Hospital admissions follow day-of-week patterns. These regularities are real and learnable. But Bitcoin's hourly log-returns are close to white noise: the expected autocorrelation at any lag is near zero, there is no strong intraday seasonality in returns (as opposed to volume), and there is no reliable mean-reversion on hourly timescales. A prior trained on "typical time series" will be miscalibrated for an asset that behaves like a near-random-walk.

The more interesting question is whether the models' pretrained directional bias is helpful at all. A model that has learned "things tend to continue" has an implicit momentum prior. A model that has learned "things tend to mean-revert" has an implicit contrarian prior. If either of those priors happens to be right about Bitcoin during the test period, directional accuracy will be above 0.50 even though the model has no BTC-specific knowledge.

### Reading the sweep

The sweep compares:
- **Chronos-2 small (28M) vs. Chronos-2 full (120M)**: if both perform identically on BTC, the extra capacity of the full model is being wasted on this signal. That is itself a finding.
- **Short vs. long context**: the models can consume hundreds of bars of history. Does more context help? On a near-i.i.d. series, the answer is typically no - there is nothing in bar -256 that predicts bar 0. The sweep makes this visible.
- **Chronos-2 vs. TimesFM**: different architectures, different pretraining corpora, different tokenization approaches. Which prior transfers better to hourly BTC?

Differences smaller than about 5 USD in MAE on a 437-bar test set are within the noise of the estimator. Focus on directional accuracy and Sharpe, which are more sensitive indicators of whether the pretrained prior is finding any signal at all.

## What zero-shot cannot do

Both models, regardless of size or context length, are forecasting BTC without having adapted to its specific distributional properties. The hourly BTC log-return has a kurtosis around 10–15 - far heavier tails than most training series. It has no strong seasonal component in returns. It has regime-switching behavior (trending for weeks, then chopping) that looks different from retail or electricity series.

The question of what happens when the model is *allowed* to update its weights on BTC data is the subject of article 7. The hypothesis is straightforward: fine-tuning should improve calibration even if the architecture is identical, because the weights will shift from "this is what a typical time series looks like" to "this is what BTC specifically looks like." Whether that hypothesis holds - and under what conditions - is more complicated than it sounds.
