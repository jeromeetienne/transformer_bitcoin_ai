from typing import Any

import numpy as np
import pandas as pd


def build_features(
        df: pd.DataFrame,
        feat_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        # Row at index T targets the log-return r_T = log(close_T / close_{T-1}).
        # Every feature column is .shift(1) over its source so a row at T sees only
        # data observed strictly before T — no current-bar leakage. ref_close is
        # close_{T-1}, used downstream to reconstruct prices as close_{T-1} * exp(r).
        close = df['close'].astype('float64')
        r = np.log(close).diff()

        lags = int(feat_cfg['return_lags'])
        if lags < 1:
                raise ValueError(f'return_lags must be >= 1, got {lags}')

        cols: dict[str, pd.Series] = {}
        for i in range(lags):
                cols[f'r_lag_{i + 1}'] = r.shift(1 + i)

        for w_raw in feat_cfg['rolling_windows']:
                w = int(w_raw)
                if w < 2:
                        raise ValueError(f'rolling window must be >= 2, got {w}')
                cols[f'r_mean_{w}'] = r.shift(1).rolling(w).mean()
                cols[f'r_std_{w}'] = r.shift(1).rolling(w).std()

        if feat_cfg.get('use_volume') is True:
                cols['log_volume'] = np.log1p(df['volume'].astype('float64')).shift(1)

        if feat_cfg.get('use_ohlc') is True:
                cols['hl_range'] = (df['high'] - df['low']).astype('float64').shift(1)
                cols['oc_body'] = (df['close'] - df['open']).astype('float64').shift(1)

        X = pd.DataFrame(cols, index=df.index)
        y = r.rename('r_target')
        ref = close.shift(1).rename('ref_close')

        joined = pd.concat([X, y, ref], axis=1).dropna()
        X_clean = joined[list(cols.keys())].copy()
        y_clean = joined['r_target'].copy()
        ref_clean = joined['ref_close'].copy()
        return X_clean, y_clean, ref_clean
