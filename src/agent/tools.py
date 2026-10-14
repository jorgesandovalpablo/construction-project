import json
from pathlib import Path

PROJ_PATH = Path("data/predictions/proyeccion_6m.json")

_METRICS = {
    "metodo": "Test recursivo 130 dias habiles",
    "Price_Equipo1": {
        "MAE_modelo_6M": 36.46, "MAE_naive_6M": 52.67, "MAE_drift_6M": 54.14,
        "MAPE_modelo": "7.00%", "MAPE_naive": "10.40%",
        "reduccion_error_vs_naive": "-31%",
        "cobertura_real_banda_90pct": "54%",
        "por_horizonte_model_vs_naive": {
            "3M_60d": {"model": 19.41, "naive": 35.98},
            "4.5M_90d": {"model": 24.21, "naive": 39.90},
            "6M_130d": {"model": 36.46, "naive": 52.67},
        },
        "nota": "Gana a naive en todo horizonte.",
    },
    "Price_Equipo2": {
        "MAE_modelo_6M": 72.38, "MAE_naive_6M": 110.41, "MAE_drift_6M": 113.80,
        "MAPE_modelo": "6.33%", "MAPE_naive": "10.46%",
        "reduccion_error_vs_naive": "-35%",
        "cobertura_real_banda_90pct": "53%",
        "por_horizonte_model_vs_naive": {
            "3M_60d": {"model": 104.15, "naive": 69.06, "nota": "naive gana a 3M"},
            "4.5M_90d": {"model": 85.64, "naive": 87.21},
            "6M_130d": {"model": 72.38, "naive": 110.41},
        },
        "nota": "Utilidad concentrada en el tramo 4.5-6M; a 3 meses es poco fiable.",
    },
    "limitaciones": [
        "Cobertura real de bandas conformal 90% ~53-54% (los intervalos son optimistas).",
        "Fold 2022 = boom de commodities, régimen atípico que infla el error medio.",
        "Modelo autónomo: no usa materias primas futuras (el techo con insumos reales no es alcanzable sin proyecciones externas).",
    ],
}

def get_projection(target: str, month: str = None) -> dict:
    if not PROJ_PATH.exists():
        return {"error": f"No existe {PROJ_PATH}; ejecuta: python -m src.models.forecast"}
    data = json.loads(PROJ_PATH.read_text())
    t = data["targets"].get(target)
    if t is None:
        return {"error": f"target desconocido: {target}. Usa Price_Equipo1 o Price_Equipo2."}
    out = {
        "target": target,
        "generado_en": data["generated_at"],
        "corte_datos": data["data_cutoff"],
        "horizon_dias": data["horizon_days"],
        "intervalo": data["interval"],
        "q": t["q"],
        "primer_dia": t["first"],
        "ultimo_dia": t["last"],
        "monthly": t["monthly"],
    }
    if month:
        serie = [row for row in t["series"] if row["date"].startswith(month)]
        if not serie:
            out["error"] = f"sin datos para el mes {month}; meses disponibles: {[m['month'] for m in t['monthly']]}"
        else:
            out["series_diaria"] = serie
    return out

def get_metrics() -> dict:
    return _METRICS

def execute_tool(name: str, args: dict) -> dict:
    if name == "get_projection":
        return get_projection(args.get("target", ""), args.get("month"))
    if name == "get_metrics":
        return get_metrics()
    return {"error": f"tool desconocida: {name}"}