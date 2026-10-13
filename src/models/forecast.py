import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data.ingestion import load_raw_data
from src.features.build_features import build_features
from src.models.core import FINAL, ALPHA, _conformal_q, train_final, _get_params
from src.utils.config import config
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

TARGETS = [config["data"]["target1"], config["data"]["target2"]]
HORIZON_DAYS = config["model"]["horizon"]


def _refit(series: pd.Series, target: str, objective: str, n_est: int):
    """
    Rentrenamiento del modelo 
    """
    X, y = build_features(series.to_frame(target), target, use_inputs=False)
    n_val = min(HORIZON_DAYS, len(X) // 4)
    params = _get_params(target)
    model = lgb.train(
        params, lgb.Dataset(X.iloc[:-n_val], y.iloc[:-n_val]),
        num_boost_round=n_est,
        valid_sets=[lgb.Dataset(X.iloc[-n_val:], y.iloc[-n_val:])],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    return model, model.best_iteration


def recursive_forecast(target: str, y_hist: pd.Series, model, q: float,
                       horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    """
    Forescast recursivo, y rentrenamiento de acuerdo a las config. 
    """
    cfg = FINAL[target]
    w, refit_every = cfg["w_lag1"], cfg.get("refit_every", 0)
    future_dates = pd.bdate_range(y_hist.index[-1] + pd.Timedelta(days=1), periods=horizon)
    rows = []
    series = y_hist.copy()
    n_est = model.best_iteration

    for i, dt in enumerate(future_dates):
        if refit_every and i > 0 and i % refit_every == 0:
            model, n_est = _refit(series, target, cfg["objective"], n_est)
        temp = pd.DataFrame({target: series})
        temp.loc[dt] = series.iloc[-1]
        X_next, _ = build_features(temp, target, use_inputs=False)
        if dt not in X_next.index:
            raise RuntimeError(f"Features no computables para {dt}")
        x = X_next.loc[[dt]]
        p_model = float(model.predict(x, num_iteration=n_est)[0])
        lag1 = float(x[f"{target}_lag1"].iloc[0])
        pred = w * lag1 + (1 - w) * p_model
        series.loc[dt] = pred
        rows.append({"Date": dt, "pred": pred, "lo": pred - q, "hi": pred + q})

    return pd.DataFrame(rows).set_index("Date")


def main() -> None:
    h = (load_raw_data()["historico_equipos"].sort_values("Date").set_index("Date"))
    out_dir = config["paths"]["data_predictions"]
    import pathlib
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        X, y = build_features(h, target, use_inputs=False)
        model, q, meta = train_final(target, X, y)
        fc = recursive_forecast(target, h[target], model, q)

        out = fc.rename(columns={"pred": target, "lo": f"{target}_lo",
                                 "hi": f"{target}_hi"})
        out.to_csv(f"{out_dir}proyeccion_6m_{target}.csv")

        monthly = out.groupby(out.index.to_period("M")).agg(
            pred_mean=(target, "mean"), pred_min=(target, "min"),
            pred_max=(target, "max"), lo_mean=(f"{target}_lo", "mean"),
            hi_mean=(f"{target}_hi", "mean"),
        ).round(2)
        logger.info(f"=== {target} | q={q} | early_stop{meta['best_iteration']} ===")
        logger.info(f"\n{monthly.to_string()}")
        logger.info(
            f"Rango final: {fc['pred'].iloc[0]:.1f} → {fc['pred'].iloc[-1]:.1f} "
            f"({fc.index[0].date()} → {fc.index[-1].date()})"
        )


if __name__ == "__main__":
    main()
