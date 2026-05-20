import argparse
import logging
from pathlib import Path

from btc_ai.config import load_yaml
from btc_ai.data import SelectorFrame, load_dataset_from_experiment_cfg

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
        # Resolves the referenced dataset and warms the cache via BinanceVisionLoader,
        # one fetch per selector. Returned splits are discarded after row counting.
        splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)
        print(f'dataset: {cfg["dataset"]}')
        print(f'interval: {splits.interval}')
        totals: dict[str, int] = {}
        for name, frames in (
                ('train', splits.train),
                ('validation', splits.validation),
                ('test', splits.test),
        ):
                totals[name] = sum(len(f.df) for f in frames)
                print(f'{name}: {len(frames)} selector(s), {totals[name]} rows total')
                for f in frames:
                        _print_selector(name, f)
        print(
                f'rows: train={totals["train"]} validation={totals["validation"]} '
                f'test={totals["test"]} total={sum(totals.values())}'
        )
        print(f'cache: {CACHE_DIR}')


def _print_selector(split_name: str, f: SelectorFrame) -> None:
        first = f.df.index[0]
        last = f.df.index[-1]
        print(
                f'  [{split_name}] {f.market} {f.symbol}: {len(f.df)} rows, '
                f'{first} -> {last}'
        )


if __name__ == '__main__':
        main()
