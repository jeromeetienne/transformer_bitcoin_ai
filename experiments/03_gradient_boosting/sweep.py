import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from features import build_features

from btc_ai.config import load_yaml
from btc_ai.data import load_dataset_from_experiment_cfg
from btc_ai.eval.metrics import (
        annualized_sharpe,
        cumulative_return,
        directional_accuracy,
        mae,
        mape,
        periods_per_year,
        rmse,
        strategy_returns,
)

EXPERIMENT_DIR = Path(__file__).parent
# results_dir is derived inside main() from the config filename stem
# (e.g. configs/btc_4h_2024.yaml → results/btc_4h_2024/).
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / 'data' / 'raw'

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
        parser.add_argument('--config', type=Path, required=True)
        args = parser.parse_args()

        results_dir = EXPERIMENT_DIR / 'results' / args.config.stem
        results_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_yaml(args.config)
        feat_cfg = cfg['features']
        model_cfg = cfg['model']

        splits = load_dataset_from_experiment_cfg(cfg, REPO_ROOT, CACHE_DIR)
        df = pd.concat([splits.train, splits.validation, splits.test])
        pre_test_rows = len(splits.train) + len(splits.validation)
        X, y, ref = build_features(df, feat_cfg)
        split = X.index.get_indexer([df.index[pre_test_rows]])[0]
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test, y_test = X.iloc[split:], y.iloc[split:]
        ref_test = ref.iloc[split:]
        close_test = ref_test * np.exp(y_test)

        ppy = periods_per_year(splits.interval)
        header = [
                'n_estimators', 'max_depth', 'learning_rate',
                'mae', 'rmse', 'mape', 'dir_acc', 'cum_ret', 'sharpe',
        ]
        print(
                f'{"n_est":>6} {"depth":>6} {"lr":>6} '
                f'{"mae":>10} {"rmse":>10} {"mape":>9} {"dir_acc":>9} '
                f'{"cum_ret":>10} {"sharpe":>8}'
        )
        print('-' * 87)

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
                        strat = strategy_returns(close_test, pred_close, ref_test)
                        row = {
                                'n_estimators': int(params['n_estimators']),
                                'max_depth': int(params['max_depth']),
                                'learning_rate': float(params['learning_rate']),
                                'mae': mae(close_test, pred_close),
                                'rmse': rmse(close_test, pred_close),
                                'mape': mape(close_test, pred_close),
                                'dir_acc': directional_accuracy(close_test, pred_close, ref_test),
                                'cum_ret': cumulative_return(strat),
                                'sharpe': annualized_sharpe(strat, ppy),
                        }
                        print(
                                f'{row["n_estimators"]:>6d} {row["max_depth"]:>6d} '
                                f'{row["learning_rate"]:>6.3f} '
                                f'{row["mae"]:>10.2f} {row["rmse"]:>10.2f} '
                                f'{row["mape"] * 100:>8.3f}% {row["dir_acc"]:>9.4f} '
                                f'{row["cum_ret"] * 100:>9.3f}% {row["sharpe"]:>8.3f}'
                        )
                        rows.append(row)
                except Exception as exc:  # noqa: BLE001
                        print(
                                f'{int(params["n_estimators"]):>6d} '
                                f'{int(params["max_depth"]):>6d} '
                                f'{float(params["learning_rate"]):>6.3f} FAILED: {exc}'
                        )

        out = results_dir / 'sweep.csv'
        with open(out, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
        print(f'\nwrote {out}')


if __name__ == '__main__':
        main()
