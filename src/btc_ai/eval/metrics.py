import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
        return float(np.mean(np.abs(y_true.to_numpy() - y_pred.to_numpy())))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
        diff = y_true.to_numpy() - y_pred.to_numpy()
        return float(np.sqrt(np.mean(diff * diff)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
        yt = y_true.to_numpy()
        yp = y_pred.to_numpy()
        nonzero = yt != 0
        if not nonzero.any():
                return float('nan')
        return float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])))


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series, ref: pd.Series) -> float:
        # Fraction of steps where the predicted move direction matches the actual move,
        # measured against `ref` (typically the last observed price before the step).
        # Steps where either direction is exactly flat (sign == 0) are excluded —
        # a flat prediction means the model expressed no opinion. Returns NaN if
        # the model never expressed a direction (e.g. naive last-value baseline).
        true_dir = np.sign(y_true.to_numpy() - ref.to_numpy())
        pred_dir = np.sign(y_pred.to_numpy() - ref.to_numpy())
        mask = (true_dir != 0) & (pred_dir != 0)
        if not mask.any():
                return float('nan')
        return float(np.mean(true_dir[mask] == pred_dir[mask]))


def strategy_returns(y_true: pd.Series, y_pred: pd.Series, ref: pd.Series) -> pd.Series:
        # Per-bar realized return of a long/flat rule: go long for the next bar when
        # the model predicts a strictly positive move (y_pred > ref), else flat.
        # Realized return is (y_true / ref - 1) — the actual close-to-close move from
        # the reference price to the next bar. No transaction costs / slippage applied.
        position = (y_pred.to_numpy() > ref.to_numpy()).astype('float64')
        realized = y_true.to_numpy() / ref.to_numpy() - 1.0
        return pd.Series(position * realized, index=y_true.index, name='strategy_return')


def cumulative_return(strat_ret: pd.Series) -> float:
        return float((1.0 + strat_ret.to_numpy()).prod() - 1.0)


def annualized_sharpe(strat_ret: pd.Series, periods_per_year: int) -> float:
        # Caller passes periods_per_year explicitly (e.g. 24 * 365 = 8760 for 1h bars)
        # to avoid hard-coding interval assumptions in the metric. Returns NaN when the
        # strategy was flat across the whole window or had a degenerate stddev.
        r = strat_ret.to_numpy()
        sd = r.std(ddof=1)
        if not np.isfinite(sd) or sd == 0.0:
                return float('nan')
        return float(np.sqrt(periods_per_year) * r.mean() / sd)


PERIODS_PER_YEAR_BY_INTERVAL: dict[str, int] = {
        '1m':  525_600,
        '5m':  105_120,
        '15m': 35_040,
        '30m': 17_520,
        '1h':  8_760,
        '2h':  4_380,
        '4h':  2_190,
        '6h':  1_460,
        '8h':  1_095,
        '12h': 730,
        '1d':  365,
        '1w':  52,
}


def periods_per_year(interval: str) -> int:
        # Lookup matching the explicit-config / no-auto-pick convention: caller must
        # pass an interval string that exists in the table; raises if the interval is
        # one we haven't budgeted (e.g. '3d', '1M'). Add to the table when needed.
        if interval not in PERIODS_PER_YEAR_BY_INTERVAL:
                raise ValueError(
                        f'no periods_per_year mapping for interval={interval!r}; '
                        f'add it to PERIODS_PER_YEAR_BY_INTERVAL.'
                )
        return PERIODS_PER_YEAR_BY_INTERVAL[interval]
