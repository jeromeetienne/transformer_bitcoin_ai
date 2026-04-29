import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from btc_ai.config import kline_request_from_config, load_yaml
from btc_ai.data import BinanceVisionLoader
from btc_ai.eval.metrics import directional_accuracy, mae, mape, rmse

EXPERIMENT_DIR = Path(__file__).parent
DEFAULT_CONFIG = EXPERIMENT_DIR / 'config.yaml'
RESULTS_DIR = EXPERIMENT_DIR / 'results'
CACHE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw'

logger = logging.getLogger(__name__)


def main() -> None:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

        parser = argparse.ArgumentParser(description='Moving-average baseline.')
        parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        args = parser.parse_args()

        cfg = load_yaml(args.config)
        req = kline_request_from_config(cfg)
        test_fraction: float = cfg['test_fraction']
        window: int = cfg['window']

        loader = BinanceVisionLoader(cache_dir=CACHE_DIR)
        df = loader.load(req)
        logger.info('loaded %d rows from %s to %s', len(df), df.index[0], df.index[-1])

        close = df['close']
        if len(close) <= window:
                raise ValueError(
                        f'series too short ({len(close)}) for window={window}; '
                        f'increase the [start, end) range or shrink window.'
                )

        # MA prediction: at time t, average the previous `window` closes (strictly < t).
        ma = close.shift(1).rolling(window).mean()

        split = int(len(close) * (1.0 - test_fraction))
        test = close.iloc[split:]
        ref = close.iloc[split - 1:-1]
        ref.index = test.index
        y_pred = ma.iloc[split:]

        metrics = {
                'experiment': '01b_moving_average',
                'window': window,
                'rows_total': int(len(close)),
                'rows_test': int(len(test)),
                'mae': mae(test, y_pred),
                'rmse': rmse(test, y_pred),
                'mape': mape(test, y_pred),
                'directional_accuracy': directional_accuracy(test, y_pred, ref),
        }

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

        predictions = pd.DataFrame({'close': test, 'pred': y_pred, 'ref': ref})
        predictions.to_parquet(RESULTS_DIR / 'predictions.parquet')

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(test.index, test.values, label='close', linewidth=1)
        ax.plot(y_pred.index, y_pred.values, label=f'MA({window}) pred', linewidth=1, alpha=0.7)
        ax.set_title(
                f'{req.symbol} {req.interval} — MA({window}) baseline (test set)',
        )
        ax.set_ylabel('price')
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / 'plot.png', dpi=120)
        plt.close(fig)

        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
        main()
