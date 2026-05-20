import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

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
from btc_ai.eval.metrics import directional_accuracy, mae, mape, rmse

EXPERIMENT_DIR = Path(__file__).parent
# results_dir is derived inside main() from the config filename stem
# (e.g. configs/btc_4h_2024.config.yaml → results/btc_4h_2024/).
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / 'data' / 'raw'

logger = logging.getLogger(__name__)


def main() -> None:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

        parser = argparse.ArgumentParser(description='Naive last-value baseline.')
        parser.add_argument('--config', type=Path, required=True)
        args = parser.parse_args()

        results_dir = EXPERIMENT_DIR / 'results' / args.config.stem.removesuffix('.config')
        results_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_yaml(args.config)
        splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)

        results: list[PerSymbolResult] = []
        for (market, symbol), per_symbol in group_selectors_by_symbol(splits).items():
                result = _run_one_symbol(market, symbol, per_symbol)
                logger.info(
                        '%s %s: mae=%.2f rmse=%.2f mape=%.4f',
                        market, symbol,
                        result.metrics['mae'], result.metrics['rmse'], result.metrics['mape'],
                )
                results.append(result)

        payload = write_aggregated_metrics_json(
                results,
                results_dir / 'metrics.json',
                experiment='01_baseline_naive',
                dataset=cfg['dataset'],
                interval=splits.interval,
        )

        concat_predictions(results).to_parquet(results_dir / 'predictions.parquet')

        _plot(results, splits.interval, results_dir / 'plot.png')

        print(json.dumps(payload, indent=2))


def _run_one_symbol(market: str, symbol: str, splits: DatasetSplits) -> PerSymbolResult:
        # Naive baseline doesn't use a validation slice: fold val into the
        # "pre-test" segment so behavior matches the legacy single-split path.
        train_df = concat_selector_frames(splits.train)
        test_df = concat_selector_frames(splits.test)
        val_df = (concat_selector_frames(splits.validation)
                  if len(splits.validation) > 0 else train_df.iloc[0:0])
        df = pd.concat([train_df, val_df, test_df])
        split = len(train_df) + len(val_df)

        close = df['close']
        test = close.iloc[split:]
        ref = close.iloc[split - 1:-1]
        ref.index = test.index
        y_pred = ref

        predictions = pd.DataFrame({'close': test, 'pred': y_pred})
        return PerSymbolResult(
                symbol=symbol,
                market=market,
                n_test_rows=int(len(test)),
                metrics={
                        'mae': mae(test, y_pred),
                        'rmse': rmse(test, y_pred),
                        'mape': mape(test, y_pred),
                        'directional_accuracy': directional_accuracy(test, y_pred, ref),
                },
                predictions=predictions,
        )


def _plot(results: list[PerSymbolResult], interval: str, out_path: Path) -> None:
        n = len(results)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)
        for ax, r in zip(axes[:, 0], results):
                test = r.predictions['close']
                pred = r.predictions['pred']
                ax.plot(test.index, test.values, label='close', linewidth=1)
                ax.plot(pred.index, pred.values, label='naive pred', linewidth=1, alpha=0.7)
                ax.set_title(f'{r.symbol} {interval} — naive baseline (test set)')
                ax.set_ylabel('price')
                ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


if __name__ == '__main__':
        main()
