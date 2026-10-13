import json
from datetime import datetime, timezone
from pathlib import Path
import lightgbm as lgb
from src.utils.logger import get_logger

logger = get_logger(__name__)
MODELS_DIR = Path("models")

def _paths(target: str) -> tuple[Path, Path]:
    return MODELS_DIR / f"{target}.lgb", MODELS_DIR / f"{target}_meta.json"

def save_artifact(target: str, model: lgb.Booster, q: float, meta: dict,
                  data_cutoff: str) -> None:
    """Persiste modelo LightGBM + metadatos (q, config, corte de datos)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = _paths(target)
    model.save_model(str(model_path))
    payload = {
        "q": float(q),
        "w_lag1": float(meta["w_lag1"]),
        "objective": meta["objective"],
        "refit_every": int(meta["refit_every"]),
        "best_iteration": int(meta["best_iteration"]),
        "data_cutoff": data_cutoff,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"[registry] {target}: artefacto guardado en {model_path}")

def load_artifact(target: str):
    """Carga (model, q, meta). Retorna None si falta cualquiera de los archivos."""
    model_path, meta_path = _paths(target)
    if not model_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    model = lgb.Booster(model_file=str(model_path))
    logger.info(
        f"[registry] {target}: modelo cacheado (corte={meta['data_cutoff']}, "
        f"trained_at={meta['trained_at']})"
    )
    return model, float(meta["q"]), meta

def needs_retrain(target: str, data_cutoff: str) -> bool:
    """True si no hay artefacto o su corte de datos no coincide."""
    _, meta_path = _paths(target)
    if not meta_path.exists():
        return True
    meta = json.loads(meta_path.read_text())
    return meta.get("data_cutoff") != data_cutoff
