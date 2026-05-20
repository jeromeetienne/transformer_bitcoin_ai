import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PerSymbolResult:
        symbol: str
        market: str
        n_test_rows: int
        metrics: dict[str, float]
        predictions: pd.DataFrame
        extras: dict[str, object] = field(default_factory=dict)


def aggregate_metrics(results: list[PerSymbolResult]) -> dict[str, float]:
        # Macro mean across symbols for every scalar metric key present anywhere.
        # NaNs propagate — a symbol that produced NaN for a metric makes the
        # aggregate NaN for that metric (we never silently drop a symbol).
        if len(results) == 0:
                raise ValueError('aggregate_metrics: no per-symbol results to aggregate')
        keys: list[str] = []
        seen: set[str] = set()
        for r in results:
                for k in r.metrics:
                        if k not in seen:
                                seen.add(k)
                                keys.append(k)
        out: dict[str, float] = {}
        for k in keys:
                values = [r.metrics.get(k, float('nan')) for r in results]
                if any(_is_nan(v) for v in values):
                        out[k] = float('nan')
                else:
                        out[k] = float(sum(values) / len(values))
        return out


def write_aggregated_metrics_json(
        results: list[PerSymbolResult],
        out_path: Path,
        *,
        experiment: str,
        dataset: str,
        interval: str,
        extra_run_fields: dict[str, object] | None = None,
) -> dict[str, object]:
        # Builds and writes the metrics.json. Returns the dict that was written so
        # callers can also print it / log it.
        agg = aggregate_metrics(results)
        per_symbol_blocks: list[dict[str, object]] = []
        for r in results:
                block: dict[str, object] = {
                        'symbol': r.symbol,
                        'market': r.market,
                        'n_test_rows': int(r.n_test_rows),
                        **r.metrics,
                }
                if len(r.extras) > 0:
                        block['extras'] = r.extras
                per_symbol_blocks.append(block)
        payload: dict[str, object] = {
                'experiment': experiment,
                'dataset': dataset,
                'interval': interval,
                **(extra_run_fields or {}),
                'rows_test': int(sum(r.n_test_rows for r in results)),
                'aggregate': agg,
                'per_symbol': per_symbol_blocks,
        }
        out_path.write_text(json.dumps(payload, indent=2, default=_json_default) + '\n')
        return payload


def concat_predictions(results: list[PerSymbolResult]) -> pd.DataFrame:
        # Stacks per-symbol prediction frames with a (symbol, market) MultiIndex
        # prefix so a single parquet file holds every symbol's bars.
        if len(results) == 0:
                return pd.DataFrame()
        chunks = []
        for r in results:
                tagged = r.predictions.copy()
                tagged.insert(0, 'symbol', r.symbol)
                tagged.insert(1, 'market', r.market)
                chunks.append(tagged)
        return pd.concat(chunks)


def _is_nan(v: object) -> bool:
        return isinstance(v, float) and math.isnan(v)


def _json_default(obj: object) -> object:
        if hasattr(obj, 'isoformat'):
                return obj.isoformat()  # datetime / pd.Timestamp
        raise TypeError(f'not JSON-serializable: {type(obj).__name__}')
