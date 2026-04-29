from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import yaml

from btc_ai.data.schema import KlineRequest


def load_yaml(path: Path) -> dict[str, Any]:
        with open(path) as f:
                return yaml.safe_load(f)


def kline_request_from_config(cfg: dict[str, Any]) -> KlineRequest:
        return KlineRequest(
                symbol=cfg['symbol'],
                interval=cfg['interval'],
                start=_parse_dt(cfg['start']),
                end=_parse_dt(cfg['end']),
                period=cfg['period'],
                market=cfg.get('market', 'spot'),
        )


def _parse_dt(value: Any) -> datetime:
        if isinstance(value, datetime):
                return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, date):
                return datetime.combine(value, time.min, tzinfo=UTC)
        if isinstance(value, str):
                return datetime.fromisoformat(value).replace(tzinfo=UTC)
        raise TypeError(f'unsupported datetime value: {value!r}')
