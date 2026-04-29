import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from features import build_features

from btc_ai.config import kline_request_from_config, load_yaml
from btc_ai.data import BinanceVisionLoader
from btc_ai.eval.metrics import directional_accuracy, mae, mape, rmse

EXPERIMENT_DIR = Path(__file__).parent
DEFAULT_CONFIG = EXPERIMENT_DIR / 'config.yaml'
RESULTS_DIR = EXPERIMENT_DIR / 'results'
CACHE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw'

# Edit this list to change which (n_estimators, max_depth, learning_rate) combos
# the sweep evaluates. All combos run against the same data slice / split / feature
# set defined in config.yaml. Fixed regularization + RNG come from config.yaml too.
GRID: list[dict[str, float | int]] = [
        {'n_estimators': 200,  'max_depth': 3, 'learning_rate': 0.10},
        {'n_estimators': 200,  'max_depth': 5, 'learning_rate': 0.10},
        {'n_estimators': 400,  'max_depth': 3, 'learning_rate': 0.05},
        {'n_estimators': 400,  'max_depth': 5, 'learning_rate': 0.05},
        {'n_estimators': 400,  'max_depth': 7, 'learning_rate': 0.05},
        {'n_estimators': 800,  'max_depth': 5, 'learning_rate': 0.03},
        {'n_estimators': 800,  'max_depth': 7, 'learning_rate': 0.03},
        {'n_estimators': 1200, 'max_depth': 5, 'learning_rate': 0.02},
]

logger = logging.getLogger(__name__)


def main() -> None:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')

        parser = argparse.ArgumentParser(
                description='Sweep XGBoost hyperparameters on the same data slice as run.py.',
        )
        parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        args = parser.parse_args()

        cfg = load_yaml(args.config)
        req = kline_request_from_config(cfg)
        test_fraction: float = cfg['test_fraction']
        feat_cfg = cfg['features']
        model_cfg = cfg['model']

        loader = BinanceVisionLoader(cache_dir=CACHE_DIR)
        df = loader.load(req)
        X, y, ref = build_features(df, feat_cfg)

        split = int(len(X) * (1.0 - test_fraction))
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test, y_test = X.iloc[split:], y.iloc[split:]
        ref_test = ref.iloc[split:]
        close_test = ref_test * np.exp(y_test)

        header = ['n_estimators', 'max_depth', 'learning_rate', 'mae', 'rmse', 'mape', 'dir_acc']
        print(
                f'{"n_est":>6} {"depth":>6} {"lr":>6} '
                f'{"mae":>10} {"rmse":>10} {"mape":>9} {"dir_acc":>9}'
        )
        print('-' * 65)

        rows: list[dict[str, object]] = []
        for params in GRID:
                try:
                        model = xgb.XGBRegressor(
                                n_estimators=int(params['n_estimators']),
                                max_depth=int(params['max_depth']),
                                learning_rate=float(params['learning_rate']),
                                subsample=float(model_cfg['subsample']),
                                colsample_bytree=float(model_cfg['colsample_bytree']),
                                reg_lambda=float(model_cfg['reg_lambda']),
                                random_state=int(model_cfg['random_state']),
                                tree_method='hist',
                                n_jobs=-1,
                                objective='reg:squarederror',
                        )
                        model.fit(X_train, y_train)
                        y_pred = pd.Series(model.predict(X_test), index=X_test.index)
                        pred_close = ref_test * np.exp(y_pred)
                        row = {
                                'n_estimators': int(params['n_estimators']),
                                'max_depth': int(params['max_depth']),
                                'learning_rate': float(params['learning_rate']),
                                'mae': mae(close_test, pred_close),
                                'rmse': rmse(close_test, pred_close),
                                'mape': mape(close_test, pred_close),
                                'dir_acc': directional_accuracy(close_test, pred_close, ref_test),
                        }
                        print(
                                f'{row["n_estimators"]:>6d} {row["max_depth"]:>6d} '
                                f'{row["learning_rate"]:>6.3f} '
                                f'{row["mae"]:>10.2f} {row["rmse"]:>10.2f} '
                                f'{row["mape"] * 100:>8.3f}% {row["dir_acc"]:>9.4f}'
                        )
                        rows.append(row)
                except Exception as exc:  # noqa: BLE001
                        print(
                                f'{int(params["n_estimators"]):>6d} '
                                f'{int(params["max_depth"]):>6d} '
                                f'{float(params["learning_rate"]):>6.3f} FAILED: {exc}'
                        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / 'sweep.csv'
        with open(out, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
        print(f'\nwrote {out}')


if __name__ == '__main__':
        main()
