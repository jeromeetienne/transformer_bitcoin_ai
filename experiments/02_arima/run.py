import argparse
import json
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

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

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)


def main() -> None:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

        parser = argparse.ArgumentParser(description='ARIMA(p, d, q) baseline.')
        parser.add_argument('--config', type=Path, required=True)
        args = parser.parse_args()

        results_dir = EXPERIMENT_DIR / 'results' / args.config.stem.removesuffix('.config')
        results_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_yaml(args.config)
        order = tuple(cfg['order'])
        if len(order) != 3:
                raise ValueError(f'order must be a 3-element list [p, d, q], got {order}')

        splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)
        ppy = periods_per_year(splits.interval)

        results: list[PerSymbolResult] = []
        for (market, symbol), per_symbol in group_selectors_by_symbol(splits).items():
                result = _run_one_symbol(market, symbol, per_symbol, order, ppy)
                logger.info(
                        '%s %s: aic=%.1f mae=%.2f',
                        market, symbol, result.metrics['aic'], result.metrics['mae'],
                )
                results.append(result)

        payload = write_aggregated_metrics_json(
                results,
                results_dir / 'metrics.json',
                experiment='02_arima',
                dataset=cfg['dataset'],
                interval=splits.interval,
                extra_run_fields={'order': list(order)},
        )

        concat_predictions(results).to_parquet(results_dir / 'predictions.parquet')

        _plot(results, splits.interval, order, results_dir / 'plot.png')

        print(json.dumps(payload, indent=2))


def _run_one_symbol(
        market: str, symbol: str,
        splits: DatasetSplits,
        order: tuple[int, int, int], ppy: int,
) -> PerSymbolResult:
        train_df = concat_selector_frames(splits.train)
        test_df = concat_selector_frames(splits.test)
        val_df = (concat_selector_frames(splits.validation)
                  if len(splits.validation) > 0 else train_df.iloc[0:0])
        df = pd.concat([train_df, val_df, test_df])
        split = len(train_df) + len(val_df)

        close = df['close'].astype('float64')
        # statsmodels does not preserve a tz-aware DatetimeIndex through ARIMA; drop tz here.
        close.index = close.index.tz_localize(None)

        train = close.iloc[:split]
        test = close.iloc[split:]
        ref = close.iloc[split - 1:-1]
        ref.index = test.index

        logger.info('fitting ARIMA%s on %d training rows for %s', order, len(train), symbol)
        fit = ARIMA(train, order=order).fit()

        # Walk-forward via apply(refit=False): reuse the fitted (p, d, q) and let
        # predict(dynamic=False) consume the actual past values.
        extended = fit.apply(close, refit=False)
        y_pred = extended.predict(start=split, end=len(close) - 1, dynamic=False)
        y_pred.index = test.index

        strat = strategy_returns(test, y_pred, ref)

        predictions = pd.DataFrame({
                'close': test,
                'pred': y_pred,
                'ref': ref,
                'strategy_return': strat,
        })
        return PerSymbolResult(
                symbol=symbol,
                market=market,
                n_test_rows=int(len(test)),
                metrics={
                        'aic': float(fit.aic),
                        'bic': float(fit.bic),
                        'mae': mae(test, y_pred),
                        'rmse': rmse(test, y_pred),
                        'mape': mape(test, y_pred),
                        'directional_accuracy': directional_accuracy(test, y_pred, ref),
                        'cumulative_return': cumulative_return(strat),
                        'annualized_sharpe': annualized_sharpe(strat, ppy),
                },
                predictions=predictions,
                extras={'rows_train': int(len(train))},
        )


def _plot(
        results: list[PerSymbolResult], interval: str,
        order: tuple[int, int, int], out_path: Path,
) -> None:
        n = len(results)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)
        for ax, r in zip(axes[:, 0], results):
                test = r.predictions['close']
                pred = r.predictions['pred']
                ax.plot(test.index, test.values, label='close', linewidth=1)
                ax.plot(
                        pred.index, pred.values,
                        label=f'ARIMA{order} pred', linewidth=1, alpha=0.7,
                )
                ax.set_title(f'{r.symbol} {interval} — ARIMA{order} (test set)')
                ax.set_ylabel('price')
                ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


if __name__ == '__main__':
        main()
