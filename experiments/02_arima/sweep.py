import argparse
import csv
import logging
import warnings
from pathlib import Path

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from btc_ai.config import kline_request_from_config, load_yaml
from btc_ai.data import BinanceVisionLoader
from btc_ai.eval.metrics import directional_accuracy, mae, mape, rmse

EXPERIMENT_DIR = Path(__file__).parent
DEFAULT_CONFIG = EXPERIMENT_DIR / 'config.yaml'
RESULTS_DIR = EXPERIMENT_DIR / 'results'
CACHE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw'

# Edit this list to change which (p, d, q) orders the sweep evaluates.
# All orders run against the same data slice / split defined in config.yaml.
ORDERS: list[tuple[int, int, int]] = [
        (0, 1, 0),  # random walk — should match naive last-value
        (1, 1, 0),  # AR(1) on first differences
        (0, 1, 1),  # MA(1) on first differences (exponential smoothing)
        (1, 1, 1),  # Box-Jenkins default
        (2, 1, 0),
        (0, 1, 2),
        (2, 1, 2),
        (3, 1, 3),
        (5, 1, 0),
        (0, 1, 5),
        (5, 1, 5),
        (1, 0, 1),  # ARMA on RAW price (no differencing) — usually worse
]

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

logger = logging.getLogger(__name__)


def main() -> None:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

        parser = argparse.ArgumentParser(
                description='Sweep ARIMA orders on the same data slice as run.py.',
        )
        parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        args = parser.parse_args()

        cfg = load_yaml(args.config)
        req = kline_request_from_config(cfg)
        test_fraction: float = cfg['test_fraction']

        loader = BinanceVisionLoader(cache_dir=CACHE_DIR)
        df = loader.load(req)
        close = df['close'].astype('float64')
        close.index = close.index.tz_localize(None)

        split = int(len(close) * (1.0 - test_fraction))
        train = close.iloc[:split]
        test = close.iloc[split:]
        ref = close.iloc[split - 1:-1]
        ref.index = test.index

        header = ['order', 'aic', 'bic', 'mae', 'rmse', 'mape', 'dir_acc']
        print(
                f'{"order":<11} {"aic":>10} {"bic":>10} '
                f'{"mae":>9} {"rmse":>9} {"mape":>9} {"dir_acc":>9}'
        )
        print('-' * 73)

        rows: list[dict[str, object]] = []
        for order in ORDERS:
                try:
                        fit = ARIMA(train, order=order).fit()
                        extended = fit.apply(close, refit=False)
                        y_pred = extended.predict(
                                start=split, end=len(close) - 1, dynamic=False,
                        )
                        y_pred.index = test.index
                        row = {
                                'order': str(order),
                                'aic': float(fit.aic),
                                'bic': float(fit.bic),
                                'mae': mae(test, y_pred),
                                'rmse': rmse(test, y_pred),
                                'mape': mape(test, y_pred),
                                'dir_acc': directional_accuracy(test, y_pred, ref),
                        }
                        print(
                                f'{row["order"]:<11} {row["aic"]:>10.1f} {row["bic"]:>10.1f} '
                                f'{row["mae"]:>9.2f} {row["rmse"]:>9.2f} '
                                f'{row["mape"] * 100:>8.3f}% {row["dir_acc"]:>9.4f}'
                        )
                        rows.append(row)
                except Exception as exc:  # noqa: BLE001
                        print(f'{str(order):<11} FAILED: {exc}')

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / 'sweep.csv'
        with open(out, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
        print(f'\nwrote {out}')


if __name__ == '__main__':
        main()
