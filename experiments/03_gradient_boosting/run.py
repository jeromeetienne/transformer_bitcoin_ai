import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from features import build_features

from btc_ai.config import kline_request_from_config, load_yaml
from btc_ai.data import BinanceVisionLoader
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
CACHE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw'

logger = logging.getLogger(__name__)


def main() -> None:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

        parser = argparse.ArgumentParser(description='XGBoost on engineered features.')
        parser.add_argument('--config', type=Path, required=True)
        args = parser.parse_args()

        results_dir = EXPERIMENT_DIR / 'results' / args.config.stem
        results_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_yaml(args.config)
        req = kline_request_from_config(cfg)
        test_fraction: float = cfg['test_fraction']
        feat_cfg = cfg['features']
        model_cfg = cfg['model']

        loader = BinanceVisionLoader(cache_dir=CACHE_DIR)
        df = loader.load(req)
        logger.info('loaded %d rows from %s to %s', len(df), df.index[0], df.index[-1])

        X, y, ref = build_features(df, feat_cfg)
        logger.info('built %d feature rows × %d cols (after dropna)', len(X), X.shape[1])

        split = int(len(X) * (1.0 - test_fraction))
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test, y_test = X.iloc[split:], y.iloc[split:]
        ref_test = ref.iloc[split:]

        model = xgb.XGBRegressor(
                n_estimators=int(model_cfg['n_estimators']),
                max_depth=int(model_cfg['max_depth']),
                learning_rate=float(model_cfg['learning_rate']),
                subsample=float(model_cfg['subsample']),
                colsample_bytree=float(model_cfg['colsample_bytree']),
                reg_lambda=float(model_cfg['reg_lambda']),
                random_state=int(model_cfg['random_state']),
                tree_method='hist',
                n_jobs=-1,
                objective='reg:squarederror',
        )

        logger.info('fitting XGBoost on %d training rows', len(X_train))
        model.fit(X_train, y_train)

        y_pred = pd.Series(model.predict(X_test), index=X_test.index, name='r_pred')
        close_test = ref_test * np.exp(y_test)
        pred_close = ref_test * np.exp(y_pred)

        strat = strategy_returns(close_test, pred_close, ref_test)
        ppy = periods_per_year(req.interval)

        metrics = {
                'experiment': '03_gradient_boosting',
                'rows_total': int(len(X)),
                'rows_train': int(len(X_train)),
                'rows_test': int(len(X_test)),
                'n_features': int(X.shape[1]),
                'mae': mae(close_test, pred_close),
                'rmse': rmse(close_test, pred_close),
                'mape': mape(close_test, pred_close),
                'directional_accuracy': directional_accuracy(close_test, pred_close, ref_test),
                'cumulative_return': cumulative_return(strat),
                'annualized_sharpe': annualized_sharpe(strat, ppy),
        }

        (results_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

        predictions = pd.DataFrame({
                'close': close_test,
                'pred': pred_close,
                'ref': ref_test,
                'strategy_return': strat,
        })
        predictions.to_parquet(results_dir / 'predictions.parquet')

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(close_test.index, close_test.values, label='close', linewidth=1)
        ax.plot(pred_close.index, pred_close.values, label='XGBoost pred', linewidth=1, alpha=0.7)
        ax.set_title(f'{req.symbol} {req.interval} — XGBoost (test set)')
        ax.set_ylabel('price')
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / 'plot.png', dpi=120)
        plt.close(fig)

        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
        main()
