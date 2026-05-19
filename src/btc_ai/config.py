from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
        with open(path) as f:
                return yaml.safe_load(f)


def _parse_dt(value: Any) -> datetime:
        if isinstance(value, datetime):
                return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, date):
                return datetime.combine(value, time.min, tzinfo=UTC)
        if isinstance(value, str):
                return datetime.fromisoformat(value).replace(tzinfo=UTC)
        raise TypeError(f'unsupported datetime value: {value!r}')
