# Imagen de la solución: API del agente (default) o batch de proyección.
# Entrypoints:
#   API:   docker run -e OPENAI_API_KEY=... -p 8000:8000 <img>
#   Batch: docker run -e OPENAI_API_KEY=... <img> python -m src.models.forecast
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY data/raw ./data/raw
COPY data/predictions ./data/predictions
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
