import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data.ingestion import load_raw_data
from src.features.build_features import build_features
from src.models import registry
from src.models.core import FINAL, ALPHA, _conformal_q, train_final, _get_params
from src.utils.config import config
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

TARGETS = [config["data"]["target1"], config["data"]["target2"]]
HORIZON_DAYS = config["model"]["horizon"]


def _refit(series: pd.Series, target: str, objective: str, n_est: int):
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
    cfg = FINAL[target]
    w, refit_every = cfg["w_lag1"], cfg.get("refit_every", 0)
    future_dates = pd.bdate_range(y_hist.index[-1] + pd.Timedelta(days=1),
                                  periods=horizon)
    rows = []
    series = y_hist.copy()
    n_est = model.best_iteration

    for i, dt in enumerate(future_dates):
        if refit_every and i > 0 and i % refit_every == 0:
            model, n_est = _refit(series, target, cfg["objective"], n_est)
        temp = pd.DataFrame({target: series})
        # dummy en la fila futura: las features de dt solo usan el pasado,
        # pero build_features filtra filas con y=NaN
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


def _write_json(results: dict, data_cutoff: str, out_dir: str) -> None:
    """Consolida la proyección en JSON para el consumo de un agente"""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_cutoff": data_cutoff,
        "horizon_days": HORIZON_DAYS,
        "interval": {"level": 1 - ALPHA, "method": "conformal_split"},
        "targets": {},
    }
    for target, (fc, q) in results.items():
        monthly = (fc.groupby(fc.index.to_period("M"))
                   .agg(pred_mean=(target, "mean"), pred_min=(target, "min"),
                        pred_max=(target, "max"), lo_mean=(f"{target}_lo", "mean"),
                        hi_mean=(f"{target}_hi", "mean"))
                   .round(2).reset_index())
        payload["targets"][target] = {
            "q": round(float(q), 2),
            "first": {"date": str(fc.index[0].date()),
                      "pred": round(float(fc[target].iloc[0]), 2),
                      "lo": round(float(fc[f"{target}_lo"].iloc[0]), 2),
                      "hi": round(float(fc[f"{target}_hi"].iloc[0]), 2)},
            "last": {"date": str(fc.index[-1].date()),
                     "pred": round(float(fc[target].iloc[-1]), 2),
                     "lo": round(float(fc[f"{target}_lo"].iloc[-1]), 2),
                     "hi": round(float(fc[f"{target}_hi"].iloc[-1]), 2)},
            "monthly": [
                {"month": str(row["Date"]), "pred_mean": row["pred_mean"],
                 "pred_min": row["pred_min"], "pred_max": row["pred_max"],
                 "lo_mean": row["lo_mean"], "hi_mean": row["hi_mean"]}
                for _, row in monthly.iterrows()
            ],
            "series": [
                {"date": str(dt.date()),
                 "pred": round(float(row[target]), 2),
                 "lo": round(float(row[f"{target}_lo"]), 2),
                 "hi": round(float(row[f"{target}_hi"]), 2)}
                for dt, row in fc.iterrows()
            ],
        }
    out_path = Path(out_dir) / "proyeccion_6m.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(f"Proyección JSON escrita en {out_path}")


def main() -> None:
    force_retrain = "--retrain" in sys.argv
    h = (load_raw_data()["historico_equipos"]
         .sort_values("Date").set_index("Date"))
    data_cutoff = str(h.index.max().date())
    out_dir = config["paths"]["data_predictions"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for target in TARGETS:
        artifact = None if force_retrain else registry.load_artifact(target)
        if artifact is not None and not registry.needs_retrain(target, data_cutoff):
            model, q, meta = artifact
            logger.info(f"=== {target} | modelo cacheado (corte {data_cutoff}) ===")
        else:
            X, y = build_features(h, target, use_inputs=False)
            model, q, meta = train_final(target, X, y)
            cfg = FINAL[target]
            registry.save_artifact(
                target, model, q,
                {"w_lag1": cfg["w_lag1"], "objective": cfg["objective"],
                 "refit_every": cfg.get("refit_every", 0),
                 "best_iteration": meta["best_iteration"]},
                data_cutoff,
            )
        fc = recursive_forecast(target, h[target], model, q)

        out = fc.rename(columns={"pred": target, "lo": f"{target}_lo",
                                 "hi": f"{target}_hi"})
        out.to_csv(f"{out_dir}proyeccion_6m_{target}.csv")

        monthly = out.groupby(out.index.to_period("M")).agg(
            pred_mean=(target, "mean"), pred_min=(target, "min"),
            pred_max=(target, "max"), lo_mean=(f"{target}_lo", "mean"),
            hi_mean=(f"{target}_hi", "mean"),
        ).round(2)
        logger.info(f"=== {target} | q={q:.2f} | early_stop@{meta['best_iteration']} ===")
        logger.info(f"\n{monthly.to_string()}")
        logger.info(
            f"Rango final: {fc['pred'].iloc[0]:.1f} → {fc['pred'].iloc[-1]:.1f} "
            f"({fc.index[0].date()} → {fc.index[-1].date()})"
        )
        results[target] = (out, q)

    _write_json(results, data_cutoff, out_dir)


if __name__ == "__main__":
    main()