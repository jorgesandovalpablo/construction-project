# construction-project# Proyección de costos de equipos — constructora

Modelo ML (LightGBM recursivo) que proyecta el costo de dos
equipos de construcción a **130 días hábiles (6 meses)**, con **agente IA conversacional**
(Gemini con tool-calling).

## Prerrequisitos

- Python 3.12
- Docker (opcional, para correr en contenedor)
- API key gratis de [Google AI Studio](https://aistudio.google.com/apikey) (solo para el agente)

## Paso 1 — Ambiente virtual e instalación

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .                  # instala las dependencias de pyproject.toml
```

## Paso 2 — Pipeline de forecast

```bash
python -m src.models.forecast
```

Entrena (o reutiliza la caché en `models/`) y genera:

- `data/predictions/proyeccion_6m_Price_Equipo{1,2}.csv` — serie diaria proyectada
- `data/predictions/proyeccion_6m.json` — contrato de servicio que consume el agente

Flags: `--retrain` fuerza reentreno completo; sin flags solo reentrena si cambiaron los datos.

## Paso 3 — Docker

```bash
# Construir imagen
docker build -t constructora-costos .

# API del agente (default del contenedor)
docker run --rm -e GOOGLE_API_KEY=... -p 8000:8000 constructora-costos

# Batch de proyección (sin key: el modelo no la necesita)
docker run --rm constructora-costos python -m src.models.forecast
```

## Paso 4 — API del agente

```bash
export GOOGLE_API_KEY=...         # https://aistudio.google.com/apikey
uvicorn src.api.main:app --port 8000
```

Salud del servicio:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

Chat con el agente:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "¿Cómo se proyecta el precio del equipo 2 en diciembre? Incluye banda de confianza."}'
```

Respuesta: `{"reply": "...", "tools_used": ["get_projection"], "model": "..."}`.
El agente consulta siempre `proyeccion_6m.json` y las métricas walk-forward (nunca
inventa números). 

> Nota: el free tier de Gemini puede devolver 503 esporádicos por demanda — reintentar.

## Estructura

- `src/models/forecast.py` — proyección recursiva 130d (único script productivo)
- `src/models/core.py` — config FINAL, métricas, conformal, `train_final`
- `src/models/registry.py` — caché de artefactos (reentreno por cambio de datos)
- `src/data/ingestion.py` — carga de los 4 CSVs (formatos heterogéneos resueltos)
- `src/features/build_features.py` — lags/rollings/calendario
- `src/agent/` — agente Gemini (tools sobre `proyeccion_6m.json` + métricas)
- `src/api/main.py` — FastAPI `/health` + `/chat`
- `notebooks/01_experimentos.ipynb` — experimentos oficiales (EDA, walk-forward, sweep)
- `data/predictions/proyeccion_6m.json` — contrato de servicio (agente)

## Documentación

- **[informe.md](informe.md)** — entregable completo

## Cifras clave (walk-forward 3 folds × 130d)

| | Eq1 | Eq2 |
|---|---|---|
| MAE modelo / naive | **36.46** / 52.67 (−31%) | **72.38** / 110.41 (−35%) |
| Cobertura banda 90% | 54% | 53% |
| Proyección (corte 2023-08-31) | 453 → 437 (−3.6%) | 953 → 894 (−6.3%) |
