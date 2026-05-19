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
CACHE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw'

logger = logging.getLogger(__name__)


def main() -> None:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

        parser = argparse.ArgumentParser(description='Moving-average baseline.')
        parser.add_argument('--config', type=Path, required=True)
        args = parser.parse_args()

        results_dir = EXPERIMENT_DIR / 'results' / args.config.stem
        results_dir.mkdir(parents=True, exist_ok=True)

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

        strat = strategy_returns(test, y_pred, ref)
        ppy = periods_per_year(req.interval)

        metrics = {
                'experiment': '01b_moving_average',
                'window': window,
                'rows_total': int(len(close)),
                'rows_test': int(len(test)),
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
        ax.plot(y_pred.index, y_pred.values, label=f'MA({window}) pred', linewidth=1, alpha=0.7)
        ax.set_title(
                f'{req.symbol} {req.interval} — MA({window}) baseline (test set)',
        )
        ax.set_ylabel('price')
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / 'plot.png', dpi=120)
        plt.close(fig)

        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
        main()
