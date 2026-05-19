import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from btc_ai.config import _parse_dt, load_yaml
from btc_ai.data.binance_vision import BinanceVisionLoader
from btc_ai.data.schema import KlineRequest

logger = logging.getLogger(__name__)

SOURCE_FIELDS = ('market', 'symbol', 'interval', 'period')
SPLIT_NAMES = ('train', 'validation', 'test')


@dataclass(frozen=True)
class DatasetSplits:
        train: pd.DataFrame
        validation: pd.DataFrame
        test: pd.DataFrame
        interval: str
        symbol_train: str
        symbol_test: str


def load_dataset_yaml(path: Path, cache_dir: Path) -> DatasetSplits:
        cfg = load_yaml(path)
        if 'splits' not in cfg:
                raise ValueError(f'{path}: missing top-level `splits:` block')
        splits_cfg = cfg['splits']
        for name in SPLIT_NAMES:
                if name not in splits_cfg:
                        raise ValueError(f'{path}: `splits.{name}` is required')

        has_top_source = 'source' in cfg
        per_split_has_source = {
                name: any(f in splits_cfg[name] for f in SOURCE_FIELDS)
                for name in SPLIT_NAMES
        }

        if has_top_source and any(per_split_has_source.values()):
                raise ValueError(
                        f'{path}: mixed shape — top-level `source:` is set, but some '
                        f'splits also carry source fields. Use one shape only.'
                )
        if not has_top_source and not all(per_split_has_source.values()):
                missing = [n for n, has in per_split_has_source.items() if not has]
                raise ValueError(
                        f'{path}: no top-level `source:`, so every split must carry '
                        f'{SOURCE_FIELDS}. Missing source fields in: {missing}'
                )

        loader = BinanceVisionLoader(cache_dir=cache_dir)

        if has_top_source:
                return _load_shape_a(cfg, splits_cfg, loader, path)
        return _load_shape_b(splits_cfg, loader, path)


def load_dataset_from_experiment_cfg(
        exp_cfg: dict[str, Any], repo_root: Path, cache_dir: Path,
) -> DatasetSplits:
        if 'dataset' not in exp_cfg:
                raise ValueError(
                        'experiment config must reference a dataset via `dataset: <name>` '
                        '(name resolves to configs/datasets/<name>.dataset.yaml)'
                )
        name = exp_cfg['dataset']
        path = repo_root / 'configs' / 'datasets' / f'{name}.dataset.yaml'
        if not path.exists():
                raise FileNotFoundError(
                        f'dataset ref `{name}` not found at {path}'
                )
        return load_dataset_yaml(path, cache_dir)


def _load_shape_a(
        cfg: dict[str, Any],
        splits_cfg: dict[str, Any],
        loader: BinanceVisionLoader,
        path: Path,
) -> DatasetSplits:
        src = _parse_source_block(cfg['source'], path, ctx='source')

        starts = {n: _parse_dt(splits_cfg[n]['start']) for n in SPLIT_NAMES}
        ends = {n: _parse_dt(splits_cfg[n]['end']) for n in SPLIT_NAMES}

        union_req = KlineRequest(
                symbol=src['symbol'],
                interval=src['interval'],
                start=min(starts.values()),
                end=max(ends.values()),
                period=src['period'],
                market=src['market'],
        )
        df = loader.load(union_req)
        logger.info(
                'shape-A dataset: %d rows from %s for %s splits',
                len(df), src['symbol'], '/'.join(SPLIT_NAMES),
        )

        slices = {n: _slice_df(df, starts[n], ends[n], path, n) for n in SPLIT_NAMES}
        _check_non_empty(slices, path)
        _check_ordered_disjoint(starts, ends, path)

        return DatasetSplits(
                train=slices['train'],
                validation=slices['validation'],
                test=slices['test'],
                interval=src['interval'],
                symbol_train=src['symbol'],
                symbol_test=src['symbol'],
        )


def _load_shape_b(
        splits_cfg: dict[str, Any],
        loader: BinanceVisionLoader,
        path: Path,
) -> DatasetSplits:
        reqs: dict[str, KlineRequest] = {}
        for name in SPLIT_NAMES:
                blk = splits_cfg[name]
                src = _parse_source_block(blk, path, ctx=f'splits.{name}')
                reqs[name] = KlineRequest(
                        symbol=src['symbol'],
                        interval=src['interval'],
                        start=_parse_dt(blk['start']),
                        end=_parse_dt(blk['end']),
                        period=src['period'],
                        market=src['market'],
                )

        intervals = {reqs[n].interval for n in SPLIT_NAMES}
        if len(intervals) != 1:
                raise ValueError(
                        f'{path}: per-split sources must share `interval` '
                        f'(metrics rely on a single bar size); got {intervals}'
                )

        slices = {n: loader.load(reqs[n]) for n in SPLIT_NAMES}
        _check_non_empty(slices, path)

        return DatasetSplits(
                train=slices['train'],
                validation=slices['validation'],
                test=slices['test'],
                interval=reqs['train'].interval,
                symbol_train=reqs['train'].symbol,
                symbol_test=reqs['test'].symbol,
        )


def _parse_source_block(
        block: dict[str, Any], path: Path, ctx: str,
) -> dict[str, str]:
        missing = [f for f in SOURCE_FIELDS if f not in block]
        if missing:
                raise ValueError(
                        f'{path}: `{ctx}` is missing required fields: {missing}'
                )
        return {f: block[f] for f in SOURCE_FIELDS}


def _slice_df(
        df: pd.DataFrame, start: datetime, end: datetime, path: Path, name: str,
) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize('UTC')
        if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize('UTC')
        sliced = df.loc[(df.index >= start_ts) & (df.index < end_ts)]
        if len(sliced) == 0:
                raise ValueError(
                        f'{path}: split `{name}` [{start_ts}, {end_ts}) is empty after '
                        f'loading; check that the source range covers it'
                )
        return sliced


def _check_non_empty(slices: dict[str, pd.DataFrame], path: Path) -> None:
        empties = [n for n, df in slices.items() if len(df) == 0]
        if empties:
                raise ValueError(f'{path}: empty splits after loading: {empties}')


def _check_ordered_disjoint(
        starts: dict[str, datetime], ends: dict[str, datetime], path: Path,
) -> None:
        # Splits must be train < validation < test, non-overlapping. We don't require
        # adjacency — gaps between slices are allowed.
        if not (ends['train'] <= starts['validation']):
                raise ValueError(
                        f'{path}: train.end ({ends["train"]}) must be <= '
                        f'validation.start ({starts["validation"]})'
                )
        if not (ends['validation'] <= starts['test']):
                raise ValueError(
                        f'{path}: validation.end ({ends["validation"]}) must be <= '
                        f'test.start ({starts["test"]})'
                )
