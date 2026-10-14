from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Agente IA — Costos de Equipos", version="0.1.0")

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from src.agent.agent import CostAgent
        _agent = CostAgent()
    return _agent


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message vacío")
    try:
        return _get_agent().chat(req.session_id, req.message)
    except Exception as e:  # GOOGLE_API_KEY ausente, error de red, etc.
        raise HTTPException(status_code=502, detail=f"agente no disponible: {e}")