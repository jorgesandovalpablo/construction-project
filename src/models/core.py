import numpy as np
import pandas as pd
import lightgbm as lgb
from src.utils.config import config

TARGETS = [config["data"]["target1"], config["data"]["target2"]]
ALPHA = 0.10

FINAL = {
    "Price_Equipo1": {"objective": "regression_l2", "w_lag1": 0.7, "refit_every": 20},
    "Price_Equipo2": {"objective": "regression_l1", "w_lag1": 0.9, "refit_every": 0},
}

def _metrics(y_true, y_pred):
    err = np.asarray(y_true) - np.asarray(y_pred)
    return {"MAE": round(float(np.mean(np.abs(err))), 2),
            "RMSE": round(float(np.sqrt(np.mean(err**2))), 2),
            "MAPE%": round(float(np.mean(np.abs(err / np.asarray(y_true))) * 100), 2)}

def _conformal_q(resid_val: np.ndarray) -> float:
    n = len(resid_val)
    return float(np.quantile(resid_val, np.ceil((n + 1) * (1 - ALPHA)) / n))

def _get_params(target: str) -> dict:
    """Params base + override tuneado por target."""
    params = {**config["model"]["params_lgbm_diario"],
              "objective": FINAL[target]["objective"]}
    override = config["model"].get("params_override", {}).get(target, {})
    params.update(override)
    return params

def train_final(target: str, X: pd.DataFrame, y: pd.Series,
                n_estimators_override: int | None = None):
    """Entrena con config FINAL; retorna (model, q, meta)."""
    cfg = FINAL[target]
    params = _get_params(target)
    if n_estimators_override is not None:
        params["n_estimators"] = n_estimators_override

    n_val = config["training"]["test_horizon_days"]
    X_tr, y_tr = X.iloc[:-n_val], y.iloc[:-n_val]
    X_val, y_val = X.iloc[-n_val:], y.iloc[-n_val:]

    model = lgb.train(
        params, lgb.Dataset(X_tr, y_tr),
        num_boost_round=params["n_estimators"],
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    ens_val = cfg["w_lag1"] * X_val[f"{target}_lag1"].values + (1 - cfg["w_lag1"]) * pred_val
    q = _conformal_q(np.abs(y_val.values - ens_val))
    meta = {"best_iteration": model.best_iteration, "w_lag1": cfg["w_lag1"],
            "objective": cfg["objective"], "q": q}
    return model, q, meta