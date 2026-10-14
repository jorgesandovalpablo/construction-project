import numpy as np
import pandas as pd

from src.utils.config import config

INPUTS = ["Price_X", "Price_Y", "Price_Z"]
LAGS_TARGET = [1, 2, 3, 5, 10, 15, 20, 30, 60, 90, 120, 180, 250, 365]
ROLL_WINDOWS = [5, 20, 60, 90]
LAGS_INPUT = [0, 1, 5, 10, 20]


def build_features(df: pd.DataFrame, target: str, use_inputs: bool = False) -> tuple:
    """
    df: DataFrame indexado por Date con target e insumos.
    use_inputs: si True incluye features de insumos (X/Y/Z); requiere que
        sus valores futuros sean conocidos en predicción — solo experimento.
    Retorna X (features), y (target), con NaNs eliminados.
    """
    out = pd.DataFrame(index=df.index)
    y = df[target]

    for lag in LAGS_TARGET:
        out[f"{target}_lag{lag}"] = y.shift(lag)

    for w in ROLL_WINDOWS:
        out[f"{target}_roll{w}_mean"] = y.shift(1).rolling(w).mean()
        out[f"{target}_roll{w}_std"] = y.shift(1).rolling(w).std()

    if use_inputs:
        for col in INPUTS:
            s = df[col]
            out[f"{col}_diff1"] = s.diff()
            out[f"{col}_roll20_mean"] = s.shift(1).rolling(20).mean()
            for lag in LAGS_INPUT:
                out[f"{col}_lag{lag}"] = s.shift(lag)

    dow = df.index.dayofweek
    for d in range(5):
        out[f"dow_{d}"] = (dow == d).astype(int)
    out["month"] = df.index.month
    out["trend"] = np.arange(len(df))

    mask = out.notna().all(axis=1) & y.notna()
    return out[mask], y[mask]


def get_splits(index: pd.DatetimeIndex) -> tuple:
    """Split holdout: últimas test_horizon_days filas (horizonte de despliegue)."""
    n_test = config["training"]["test_horizon_days"]
    return index[:-n_test], index[-n_test:]
