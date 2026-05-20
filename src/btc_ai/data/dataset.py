import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from btc_ai.config import _parse_dt, load_yaml
from btc_ai.data.binance_vision import BinanceVisionLoader
from btc_ai.data.schema import KlineRequest

logger = logging.getLogger(__name__)

SELECTOR_FIELDS = ('market', 'symbol', 'interval', 'period', 'start', 'end')
SPLIT_NAMES = ('train', 'validation', 'test')
# YAML files use the human-readable 'training' for the training array; the
# DatasetSplits field name stays 'train' to keep call sites short.
_YAML_KEYS: dict[str, str] = {
        'train': 'training',
        'validation': 'validation',
        'test': 'test',
}


@dataclass(frozen=True)
class SelectorFrame:
        df: pd.DataFrame
        symbol: str
        market: str


@dataclass(frozen=True)
class DatasetSplits:
        train: list[SelectorFrame]
        validation: list[SelectorFrame]
        test: list[SelectorFrame]
        interval: str


def load_dataset_yaml(path: Path, cache_dir: Path) -> DatasetSplits:
        cfg = load_yaml(path)
        for name in SPLIT_NAMES:
                yaml_key = _YAML_KEYS[name]
                if yaml_key not in cfg:
                        raise ValueError(f'{path}: top-level `{yaml_key}:` array is required')
                if not isinstance(cfg[yaml_key], list) or len(cfg[yaml_key]) == 0:
                        raise ValueError(
                                f'{path}: `{yaml_key}` must be a non-empty array of CSV selectors'
                        )

        reqs: dict[str, list[KlineRequest]] = {}
        for name in SPLIT_NAMES:
                yaml_key = _YAML_KEYS[name]
                reqs[name] = [
                        _parse_selector(entry, path, ctx=f'{yaml_key}[{i}]')
                        for i, entry in enumerate(cfg[yaml_key])
                ]

        interval = _validate_uniform_interval(reqs, path)

        loader = BinanceVisionLoader(cache_dir=cache_dir)
        frames: dict[str, list[SelectorFrame]] = {}
        for name in SPLIT_NAMES:
                frames[name] = [_load_selector(req, loader, path, name, i)
                                for i, req in enumerate(reqs[name])]

        _check_per_symbol_ordering(reqs, path)

        logger.info(
                'loaded dataset: train=%d selector(s) %d rows, validation=%d selector(s) %d rows, '
                'test=%d selector(s) %d rows',
                len(frames['train']), sum(len(f.df) for f in frames['train']),
                len(frames['validation']), sum(len(f.df) for f in frames['validation']),
                len(frames['test']), sum(len(f.df) for f in frames['test']),
        )

        return DatasetSplits(
                train=frames['train'],
                validation=frames['validation'],
                test=frames['test'],
                interval=interval,
        )


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


def group_selectors_by_symbol(
        splits: DatasetSplits,
) -> dict[tuple[str, str], DatasetSplits]:
        # Returns one DatasetSplits per unique (market, symbol) appearing in `test`.
        # For each test key, training and validation selectors with the same
        # (market, symbol) are matched in. A test (market, symbol) with no
        # training match raises ValueError. Training selectors whose (market, symbol)
        # does not appear in test are ignored (logged at WARNING).
        def _key(sf: SelectorFrame) -> tuple[str, str]:
                return (sf.market, sf.symbol)

        train_by_key: dict[tuple[str, str], list[SelectorFrame]] = defaultdict(list)
        val_by_key: dict[tuple[str, str], list[SelectorFrame]] = defaultdict(list)
        for sf in splits.train:
                train_by_key[_key(sf)].append(sf)
        for sf in splits.validation:
                val_by_key[_key(sf)].append(sf)

        test_keys: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for sf in splits.test:
                k = _key(sf)
                if k not in seen:
                        seen.add(k)
                        test_keys.append(k)

        unused_train = [k for k in train_by_key if k not in seen]
        for k in unused_train:
                logger.warning(
                        'training selector for %s has no matching test symbol; '
                        'ignored by per-series experiments', k,
                )

        out: dict[tuple[str, str], DatasetSplits] = {}
        for k in test_keys:
                if k not in train_by_key:
                        raise ValueError(
                                f'test symbol {k} has no matching training selector; '
                                f'per-series experiments cannot fit a model for it'
                        )
                out[k] = DatasetSplits(
                        train=sorted(train_by_key[k], key=lambda sf: sf.df.index[0]),
                        validation=sorted(val_by_key.get(k, []), key=lambda sf: sf.df.index[0]),
                        test=[sf for sf in splits.test if _key(sf) == k],
                        interval=splits.interval,
                )
        return out


def _parse_selector(
        entry: dict[str, Any], path: Path, ctx: str,
) -> KlineRequest:
        if not isinstance(entry, dict):
                raise ValueError(f'{path}: `{ctx}` must be a mapping, got {type(entry).__name__}')
        missing = [f for f in SELECTOR_FIELDS if f not in entry]
        if missing:
                raise ValueError(
                        f'{path}: `{ctx}` is missing required fields: {missing}'
                )
        return KlineRequest(
                symbol=entry['symbol'],
                interval=entry['interval'],
                start=_parse_dt(entry['start']),
                end=_parse_dt(entry['end']),
                period=entry['period'],
                market=entry['market'],
        )


def _validate_uniform_interval(
        reqs: dict[str, list[KlineRequest]], path: Path,
) -> str:
        intervals = {r.interval for split_reqs in reqs.values() for r in split_reqs}
        if len(intervals) != 1:
                raise ValueError(
                        f'{path}: every selector must share `interval` '
                        f'(metrics rely on a single bar size); got {sorted(intervals)}'
                )
        return next(iter(intervals))


def _load_selector(
        req: KlineRequest,
        loader: BinanceVisionLoader,
        path: Path,
        split_name: str,
        idx: int,
) -> SelectorFrame:
        df = loader.load(req)
        if len(df) == 0:
                raise ValueError(
                        f'{path}: selector `{split_name}[{idx}]` ({req.market} {req.symbol} '
                        f'[{req.start}, {req.end})) is empty after loading'
                )
        return SelectorFrame(df=df, symbol=req.symbol, market=req.market)


def _check_per_symbol_ordering(
        reqs: dict[str, list[KlineRequest]], path: Path,
) -> None:
        # For each (market, symbol) that appears in test, ensure the per-symbol
        # train/val/test ranges are ordered train.end <= validation.start and
        # validation.end <= test.start, matching the prior single-source invariant.
        # Selectors for that symbol may be split across multiple entries; we compare
        # the per-symbol envelope (min start / max end).
        keys: set[tuple[str, str]] = {(r.market, r.symbol) for r in reqs['test']}
        for k in keys:
                env = {n: _envelope([r for r in reqs[n] if (r.market, r.symbol) == k])
                       for n in SPLIT_NAMES}
                if env['train'] is None:
                        continue  # group_selectors_by_symbol raises; not our job here
                if env['validation'] is not None and not (env['train'][1] <= env['validation'][0]):
                        raise ValueError(
                                f'{path}: for {k}, train.end ({env["train"][1]}) must be <= '
                                f'validation.start ({env["validation"][0]})'
                        )
                last_pre_test = env['validation'] if env['validation'] is not None else env['train']
                if not (last_pre_test[1] <= env['test'][0]):
                        raise ValueError(
                                f'{path}: for {k}, the last pre-test end ({last_pre_test[1]}) '
                                f'must be <= test.start ({env["test"][0]})'
                        )


def _envelope(reqs: list[KlineRequest]) -> tuple[datetime, datetime] | None:
        if len(reqs) == 0:
                return None
        return (min(r.start for r in reqs), max(r.end for r in reqs))


def concat_selector_frames(frames: list[SelectorFrame]) -> pd.DataFrame:
        # Convenience for callers that fit per-symbol: concatenate same-symbol
        # selector frames (already sorted by group_selectors_by_symbol) into one
        # DataFrame indexed by open_time. Asserts no overlapping timestamps.
        if len(frames) == 0:
                raise ValueError('concat_selector_frames: no frames to concatenate')
        if len(frames) == 1:
                return frames[0].df
        df = pd.concat([f.df for f in frames]).sort_index()
        if df.index.has_duplicates:
                raise ValueError(
                        f'concat_selector_frames: overlapping timestamps across selectors '
                        f'for symbol={frames[0].symbol}'
                )
        return df


