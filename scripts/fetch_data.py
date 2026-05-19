import argparse
import logging
from pathlib import Path

from btc_ai.config import load_yaml
from btc_ai.data import load_dataset_from_experiment_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / 'data' / 'raw'


def main() -> None:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

        parser = argparse.ArgumentParser(
                description='Pre-warm the data cache for an experiment config.'
        )
        parser.add_argument(
                '--config', type=Path, required=True,
                help='path to experiment config.yaml',
        )
        args = parser.parse_args()

        cfg = load_yaml(args.config)
        # Resolves the referenced dataset and warms the cache via BinanceVisionLoader
        # (one fetch for shape A, three for shape B). Returned splits are discarded.
        splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)
        n_train = len(splits.train)
        n_val = len(splits.validation)
        n_test = len(splits.test)
        print(f'dataset: {cfg["dataset"]}')
        print(
                f'rows: train={n_train} validation={n_val} test={n_test} '
                f'total={n_train + n_val + n_test}'
        )
        print(f'cache: {CACHE_DIR}')


if __name__ == '__main__':
        main()
