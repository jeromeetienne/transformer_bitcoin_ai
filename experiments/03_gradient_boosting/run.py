import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from features import build_features

from btc_ai.config import load_yaml
from btc_ai.data import (
        DatasetSplits,
        concat_selector_frames,
        group_selectors_by_symbol,
        load_dataset_from_experiment_cfg,
)
from btc_ai.eval.aggregate import (
        PerSymbolResult,
        concat_predictions,
        write_aggregated_metrics_json,
)
from btc_ai.eval.metrics import (
        annualized_sharpe,
        cumulative_return,
        directional_accuracy,
        mae,
        mape,
        periods_per_year,
        rmse,
        strategy_returns,
)

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / 'data' / 'raw'

logger = logging.getLogger(__name__)


def main() -> None:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

        parser = argparse.ArgumentParser(description='XGBoost on engineered features.')
        parser.add_argument('--config', type=Path, required=True)
        args = parser.parse_args()

        results_dir = EXPERIMENT_DIR / 'results' / args.config.stem.removesuffix('.config')
        results_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_yaml(args.config)
        feat_cfg = cfg['features']
        model_cfg = cfg['model']

        splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)
        ppy = periods_per_year(splits.interval)

        results: list[PerSymbolResult] = []
        for (market, symbol), per_symbol in group_selectors_by_symbol(splits).items():
                result = _run_one_symbol(market, symbol, per_symbol, feat_cfg, model_cfg, ppy)
                logger.info(
                        '%s %s: mae=%.2f sharpe=%.3f',
                        market, symbol,
                        result.metrics['mae'], result.metrics['annualized_sharpe'],
                )
                results.append(result)

        payload = write_aggregated_metrics_json(
                results,
                results_dir / 'metrics.json',
                experiment='03_gradient_boosting',
                dataset=cfg['dataset'],
                interval=splits.interval,
        )

        concat_predictions(results).to_parquet(results_dir / 'predictions.parquet')

        _plot(results, splits.interval, results_dir / 'plot.png')

        print(json.dumps(payload, indent=2))


def _run_one_symbol(
        market: str, symbol: str, splits: DatasetSplits,
        feat_cfg: dict, model_cfg: dict, ppy: int,
) -> PerSymbolResult:
        # XGBoost doesn't use a validation slice: fold val into the training segment
        # so behavior matches the legacy single-split path. Split before feature
        # building would lose continuity for lag/rolling features that cross the
        # train/test boundary, so we concat then split by index.
        train_df = concat_selector_frames(splits.train)
        test_df = concat_selector_frames(splits.test)
        val_df = (concat_selector_frames(splits.validation)
                  if len(splits.validation) > 0 else train_df.iloc[0:0])
        df = pd.concat([train_df, val_df, test_df])
        pre_test_rows = len(train_df) + len(val_df)

        X, y, ref = build_features(df, feat_cfg)

        # build_features drops leading rows that don't have enough history for the
        # longest lag/rolling window. Translate the bar-level pre_test_rows index
        # into a feature-row index via the index intersection.
        split = X.index.get_indexer([df.index[pre_test_rows]])[0]
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test, y_test = X.iloc[split:], y.iloc[split:]
        ref_test = ref.iloc[split:]

        model = xgb.XGBRegressor(
                n_estimators=int(model_cfg['n_estimators']),
                max_depth=int(model_cfg['max_depth']),
                learning_rate=float(model_cfg['learning_rate']),
                subsample=float(model_cfg['subsample']),
                colsample_bytree=float(model_cfg['colsample_bytree']),
                reg_lambda=float(model_cfg['reg_lambda']),
                random_state=int(model_cfg['random_state']),
                tree_method='hist',
                n_jobs=-1,
                objective='reg:squarederror',
        )
        model.fit(X_train, y_train)

        y_pred = pd.Series(model.predict(X_test), index=X_test.index, name='r_pred')
        close_test = ref_test * np.exp(y_test)
        pred_close = ref_test * np.exp(y_pred)

        strat = strategy_returns(close_test, pred_close, ref_test)

        predictions = pd.DataFrame({
                'close': close_test,
                'pred': pred_close,
                'ref': ref_test,
                'strategy_return': strat,
        })
        return PerSymbolResult(
                symbol=symbol,
                market=market,
                n_test_rows=int(len(X_test)),
                metrics={
                        'mae': mae(close_test, pred_close),
                        'rmse': rmse(close_test, pred_close),
                        'mape': mape(close_test, pred_close),
                        'directional_accuracy': directional_accuracy(
                                close_test, pred_close, ref_test,
                        ),
                        'cumulative_return': cumulative_return(strat),
                        'annualized_sharpe': annualized_sharpe(strat, ppy),
                },
                predictions=predictions,
                extras={'rows_train': int(len(X_train)), 'n_features': int(X.shape[1])},
        )


def _plot(results: list[PerSymbolResult], interval: str, out_path: Path) -> None:
        n = len(results)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)
        for ax, r in zip(axes[:, 0], results, strict=True):
                close_test = r.predictions['close']
                pred_close = r.predictions['pred']
                ax.plot(close_test.index, close_test.values, label='close', linewidth=1)
                ax.plot(
                        pred_close.index, pred_close.values,
                        label='XGBoost pred', linewidth=1, alpha=0.7,
                )
                ax.set_title(f'{r.symbol} {interval} — XGBoost (test set)')
                ax.set_ylabel('price')
                ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


if __name__ == '__main__':
        main()
