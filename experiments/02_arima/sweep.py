import argparse
import csv
import logging
import warnings
from pathlib import Path

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

# Edit this list to change which (p, d, q) orders the sweep evaluates.
# All orders run against the same data slice / split defined in config.yaml.
ORDERS: list[tuple[int, int, int]] = [
        (0, 1, 0),  # random walk — should match naive last-value
        (1, 1, 0),  # AR(1) on first differences
        (0, 1, 1),  # MA(1) on first differences (exponential smoothing)
        (1, 1, 1),  # Box-Jenkins default
        (2, 1, 0),
        (0, 1, 2),
        (2, 1, 1),
        (1, 1, 2),
        (2, 1, 2),
        (3, 1, 0),
        (0, 1, 3),
        (3, 1, 1),
        (1, 1, 3),
        (3, 1, 2),
        (2, 1, 3),
        (3, 1, 3),
        (4, 1, 0),
        (0, 1, 4),
        (4, 1, 4),
        (5, 1, 0),
        (0, 1, 5),
        (5, 1, 5),
        (7, 1, 0),
        (0, 1, 7),
        (7, 1, 7),
        (10, 1, 0),
        (0, 1, 10),
        (10, 1, 10),
        (1, 0, 1),  # ARMA on RAW price (no differencing) — usually worse
        (2, 0, 2),  # ARMA on RAW price, slightly richer
]

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

logger = logging.getLogger(__name__)


def main() -> None:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

        parser = argparse.ArgumentParser(
                description='Sweep ARIMA orders on the same data slice as run.py.',
        )
        parser.add_argument('--config', type=Path, required=True)
        args = parser.parse_args()

        results_dir = EXPERIMENT_DIR / 'results' / args.config.stem
        results_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_yaml(args.config)
        splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)
        # Mirrors run.py: ARIMA has no validation slice, fold val into training.
        df = pd.concat([splits.train, splits.validation, splits.test])
        split = len(splits.train) + len(splits.validation)
        close = df['close'].astype('float64')
        close.index = close.index.tz_localize(None)

        train = close.iloc[:split]
        test = close.iloc[split:]
        ref = close.iloc[split - 1:-1]
        ref.index = test.index

        ppy = periods_per_year(splits.interval)
        header = [
                'order', 'aic', 'bic',
                'mae', 'rmse', 'mape', 'dir_acc', 'cum_ret', 'sharpe',
        ]
        print(
                f'{"order":<11} {"aic":>10} {"bic":>10} '
                f'{"mae":>9} {"rmse":>9} {"mape":>9} {"dir_acc":>9} '
                f'{"cum_ret":>10} {"sharpe":>8}'
        )
        print('-' * 95)

        rows: list[dict[str, object]] = []
        for order in ORDERS:
                try:
                        fit = ARIMA(train, order=order).fit()
                        extended = fit.apply(close, refit=False)
                        y_pred = extended.predict(
                                start=split, end=len(close) - 1, dynamic=False,
                        )
                        y_pred.index = test.index
                        strat = strategy_returns(test, y_pred, ref)
                        row = {
                                'order': str(order),
                                'aic': float(fit.aic),
                                'bic': float(fit.bic),
                                'mae': mae(test, y_pred),
                                'rmse': rmse(test, y_pred),
                                'mape': mape(test, y_pred),
                                'dir_acc': directional_accuracy(test, y_pred, ref),
                                'cum_ret': cumulative_return(strat),
                                'sharpe': annualized_sharpe(strat, ppy),
                        }
                        print(
                                f'{row["order"]:<11} {row["aic"]:>10.1f} {row["bic"]:>10.1f} '
                                f'{row["mae"]:>9.2f} {row["rmse"]:>9.2f} '
                                f'{row["mape"] * 100:>8.3f}% {row["dir_acc"]:>9.4f} '
                                f'{row["cum_ret"] * 100:>9.3f}% {row["sharpe"]:>8.3f}'
                        )
                        rows.append(row)
                except Exception as exc:  # noqa: BLE001
                        print(f'{str(order):<11} FAILED: {exc}')

        out = results_dir / 'sweep.csv'
        with open(out, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
        print(f'\nwrote {out}')


if __name__ == '__main__':
        main()
