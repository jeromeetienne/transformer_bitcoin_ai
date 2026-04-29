from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Interval = Literal[
        '1s', '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '3d', '1w', '1mo',
]

Period = Literal['monthly', 'daily']

Market = Literal['spot', 'futures/um', 'futures/cm']


@dataclass(frozen=True)
class KlineRequest:
        symbol: str
        interval: Interval
        start: datetime
        end: datetime
        period: Period
        market: Market = 'spot'

        def __post_init__(self) -> None:
                if self.end <= self.start:
                        raise ValueError(f'end ({self.end}) must be > start ({self.start})')
