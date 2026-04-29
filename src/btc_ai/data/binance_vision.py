import logging
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from btc_ai.data.cache import download_to_cache
from btc_ai.data.schema import KlineRequest

logger = logging.getLogger(__name__)

BASE_URL = 'https://data.binance.vision/data'

KLINE_COLUMNS = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_base', 'taker_quote', 'ignore',
]

KEPT_COLUMNS = [
        'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_base', 'taker_quote',
]


class BinanceVisionLoader:
        def __init__(self, cache_dir: Path) -> None:
                self.cache_dir = Path(cache_dir)

        def load(self, req: KlineRequest) -> pd.DataFrame:
                urls = self._enumerate_urls(req)
                logger.info(
                        'loading %d %s zip(s) for %s %s',
                        len(urls), req.period, req.symbol, req.interval,
                )

                frames = [self._parse_zip(download_to_cache(url, self.cache_dir)) for url in urls]
                df = pd.concat(frames, ignore_index=True)

                df = df.drop_duplicates(subset='open_time').sort_values('open_time')
                df = df.set_index('open_time')

                start_utc = _to_utc(req.start)
                end_utc = _to_utc(req.end)
                df = df.loc[(df.index >= start_utc) & (df.index < end_utc)]

                _warn_on_gaps(df, req.interval)

                return df[KEPT_COLUMNS]

        def _enumerate_urls(self, req: KlineRequest) -> list[str]:
                base = f'{BASE_URL}/{req.market}/{req.period}/klines/{req.symbol}/{req.interval}'
                start = req.start.date() if isinstance(req.start, datetime) else req.start
                end = req.end.date() if isinstance(req.end, datetime) else req.end

                if req.period == 'monthly':
                        return [
                                f'{base}/{req.symbol}-{req.interval}-{y:04d}-{m:02d}.zip'
                                for y, m in _iter_months(start, end)
                        ]

                return [
                        f'{base}/{req.symbol}-{req.interval}-{d.isoformat()}.zip'
                        for d in _iter_days(start, end)
                ]

        def _parse_zip(self, path: Path) -> pd.DataFrame:
                with zipfile.ZipFile(path) as zf:
                        members = [n for n in zf.namelist() if n.endswith('.csv')]
                        if len(members) != 1:
                                raise ValueError(f'expected exactly 1 csv in {path}, got {members}')
                        with zf.open(members[0]) as f:
                                df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)

                # Newer zips ship with a header row; detect by checking if first row is non-numeric.
                if df.iloc[0]['open_time'] == 'open_time':
                        df = df.iloc[1:].reset_index(drop=True)

                df['open_time'] = pd.to_datetime(
                        df['open_time'].astype('int64'), unit='ms', utc=True,
                )
                float_cols = [
                        'open', 'high', 'low', 'close', 'volume',
                        'quote_volume', 'taker_base', 'taker_quote',
                ]
                for col in float_cols:
                        df[col] = df[col].astype('float64')
                df['trades'] = df['trades'].astype('int64')

                return df


def _to_utc(dt: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(dt)
        return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')


def _iter_months(start: date, end: date) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
                # Drop the trailing month if [start, end) ends exactly on its first day.
                if (y, m) == (end.year, end.month) and end.day == 1:
                        break
                out.append((y, m))
                m += 1
                if m == 13:
                        m = 1
                        y += 1
        return out


def _iter_days(start: date, end: date) -> list[date]:
        out: list[date] = []
        d = start
        while d < end:
                out.append(d)
                d += timedelta(days=1)
        return out


_INTERVAL_TO_TIMEDELTA: dict[str, timedelta] = {
        '1s': timedelta(seconds=1),
        '1m': timedelta(minutes=1), '3m': timedelta(minutes=3), '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15), '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1), '2h': timedelta(hours=2), '4h': timedelta(hours=4),
        '6h': timedelta(hours=6), '8h': timedelta(hours=8), '12h': timedelta(hours=12),
        '1d': timedelta(days=1), '3d': timedelta(days=3), '1w': timedelta(weeks=1),
}


def _warn_on_gaps(df: pd.DataFrame, interval: str) -> None:
        expected = _INTERVAL_TO_TIMEDELTA.get(interval)
        if expected is None or len(df) < 2:
                return
        deltas = df.index.to_series().diff().dropna()
        gaps = deltas[deltas != expected]
        if not gaps.empty:
                logger.warning(
                        '%d gap(s) detected in kline series (interval=%s); first at %s',
                        len(gaps), interval, gaps.index[0],
                )

