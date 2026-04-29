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
