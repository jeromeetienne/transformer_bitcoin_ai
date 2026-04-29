import argparse
import logging
from pathlib import Path

from btc_ai.config import kline_request_from_config, load_yaml
from btc_ai.data import BinanceVisionLoader

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
        req = kline_request_from_config(cfg)

        loader = BinanceVisionLoader(cache_dir=CACHE_DIR)
        df = loader.load(req)

        print(f'rows: {len(df)}')
        print(f'range: {df.index[0]} -> {df.index[-1]}')
        print(f'cache: {CACHE_DIR}')


if __name__ == '__main__':
        main()
