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
from btc_ai.data import load_dataset_from_experiment_cfg
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
# results_dir is derived inside main() from the config filename stem
# (e.g. configs/btc_4h_2024.yaml → results/btc_4h_2024/).
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

        results_dir = EXPERIMENT_DIR / 'results' / args.config.stem
        results_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_yaml(args.config)
        order = tuple(cfg['order'])
        if len(order) != 3:
                raise ValueError(f'order must be a 3-element list [p, d, q], got {order}')

        splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)
        # ARIMA doesn't use a validation slice: fold val into the training segment
        # so behavior matches the legacy single-split path.
        df = pd.concat([splits.train, splits.validation, splits.test])
        split = len(splits.train) + len(splits.validation)
        logger.info('loaded %d rows from %s to %s', len(df), df.index[0], df.index[-1])

        close = df['close'].astype('float64')
        # statsmodels does not preserve a tz-aware DatetimeIndex through ARIMA; drop tz here.
        close.index = close.index.tz_localize(None)

        train = close.iloc[:split]
        test = close.iloc[split:]
        ref = close.iloc[split - 1:-1]
        ref.index = test.index

        # Fit ARIMA params on the training window only — no leakage.
        logger.info('fitting ARIMA%s on %d training rows', order, len(train))
        fit = ARIMA(train, order=order).fit()

        # Walk-forward: extend the fitted model with test observations, then ask
        # for one-step-ahead in-sample predictions on the test slice. With
        # apply(refit=False) the (p, d, q) parameters from train are reused, and
        # predict(dynamic=False) uses actual past values — true 1-step walk-forward.
        extended = fit.apply(close, refit=False)
        y_pred = extended.predict(start=split, end=len(close) - 1, dynamic=False)
        y_pred.index = test.index

        strat = strategy_returns(test, y_pred, ref)
        ppy = periods_per_year(splits.interval)

        metrics = {
                'experiment': '02_arima',
                'dataset': cfg['dataset'],
                'symbol': splits.symbol_test,
                'interval': splits.interval,
                'order': list(order),
                'rows_total': int(len(close)),
                'rows_train': int(len(train)),
                'rows_test': int(len(test)),
                'aic': float(fit.aic),
                'bic': float(fit.bic),
                'mae': mae(test, y_pred),
                'rmse': rmse(test, y_pred),
                'mape': mape(test, y_pred),
                'directional_accuracy': directional_accuracy(test, y_pred, ref),
                'cumulative_return': cumulative_return(strat),
                'annualized_sharpe': annualized_sharpe(strat, ppy),
        }

        (results_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

        predictions = pd.DataFrame({
                'close': test,
                'pred': y_pred,
                'ref': ref,
                'strategy_return': strat,
        })
        predictions.to_parquet(results_dir / 'predictions.parquet')

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(test.index, test.values, label='close', linewidth=1)
        ax.plot(y_pred.index, y_pred.values, label=f'ARIMA{order} pred', linewidth=1, alpha=0.7)
        ax.set_title(
                f'{splits.symbol_test} {splits.interval} — ARIMA{order} (test set)',
        )
        ax.set_ylabel('price')
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / 'plot.png', dpi=120)
        plt.close(fig)

        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
        main()
