from dataclasses import dataclass

import pandas as pd

from btc_ai.data.dataset import (
        DatasetSplits,
        concat_selector_frames,
        group_selectors_by_symbol,
)


@dataclass(frozen=True)
class PerSymbolFrame:
        market: str
        symbol: str
        # concat of train+validation+test for this symbol
        df: pd.DataFrame
        # tz-naive; None if this symbol has no validation selector
        val_start_ts: pd.Timestamp | None
        # tz-naive; never None (every test symbol has >=1 test selector)
        test_start_ts: pd.Timestamp


def build_per_symbol_frames(splits: DatasetSplits) -> list[PerSymbolFrame]:
        # For each (market, symbol) in `test`, concatenate the matching training,
        # validation, and test selector frames (in temporal order) into a single
        # DataFrame ready to feed into a sequence model. Returns one PerSymbolFrame
        # per test symbol; the timestamps are tz-naive (darts indexes are unambiguous
        # without timezone). Raises ValueError if any test symbol has no matching
        # training selector — that propagates from group_selectors_by_symbol.
        grouped = group_selectors_by_symbol(splits)
        out: list[PerSymbolFrame] = []
        for (market, symbol), s in grouped.items():
                train_df = concat_selector_frames(s.train)
                test_df = concat_selector_frames(s.test)
                val_df = (concat_selector_frames(s.validation)
                          if len(s.validation) > 0 else train_df.iloc[0:0])
                df = pd.concat([train_df, val_df, test_df])
                df.index = df.index.tz_localize(None)
                val_start_ts = (val_df.index[0].tz_localize(None)
                                if len(val_df) > 0 else None)
                test_start_ts = test_df.index[0].tz_localize(None)
                out.append(PerSymbolFrame(
                        market=market,
                        symbol=symbol,
                        df=df,
                        val_start_ts=val_start_ts,
                        test_start_ts=test_start_ts,
                ))
        return out
