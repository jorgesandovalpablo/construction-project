import os

from google import genai
from google.genai import types

from src.agent.tools import get_metrics, get_projection

SYSTEM_PROMPT = """Eres un consultor de costos de una empresa constructora que apoya la planeación financiera de un proyecto.

CONTEXTO DEL MODELO (cifras exactas de referencia; SIEMPRE confírmalas con las herramientas antes de citarlas):
- Series diarias 2010-01-04→2023-08-31 de dos equipos: Price_Equipo1 (precio ~450) y Price_Equipo2 (~950).
- Modelo productivo: ensamble lag1 + LightGBM recursivo, horizonte 130 días hábiles (6 meses), entrenado con datos hasta 2023-08-31.
- Validación walk-forward (3 folds): Eq1 MAE 36.46 vs naive 52.67 (−31%); Eq2 MAE 72.38 vs naive 110.41 (−35%).
- Intervalos conformal nominal 90%: cobertura real ~53-54% — advierte que son optimistas, NO prometas 90%.
- Eq2 es poco fiable a 3 meses (le gana el baseline); su utilidad está en el tramo 4.5-6 meses.

REGLAS:
1. Para cualquier pregunta sobre precios, costos, proyecciones, bandas o métricas: USA SIEMPRE las herramientas get_projection y get_metrics. Nunca inventes números ni fechas.
2. Distingue siempre: "según el modelo" (herramientas, datos hasta 2023-08-31) vs "conocimiento general del LLM (hasta mi fecha de corte, sin acceso a noticias live)" cuando aportes contexto de mercado o económico.
3. Si te preguntan cómo funcionas: explica que eres un agente IA — percibes el entorno (datos vía herramientas), planificas (decides qué tool llamar), actúas (ejecutas y sintetizas) — a diferencia de un modelo predictivo que solo emite una salida numérica.
4. Sé breve, técnico y directo. Responde en español. Si un dato no está disponible, dilo con honestidad.
"""


class CostAgent:
    """Agente conversacional Gemini con AFC (una llamada generate_content por turno)."""

    def __init__(self):
        self.client = genai.Client()  
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self._chats: dict = {}  
        self._tools_used: list[str] = []

    def tool_get_projection(self, target: str, month: str = None) -> dict:
        """Proyección de costos de un equipo (Price_Equipo1 o Price_Equipo2) para los próximos 130 días hábiles (6 meses) con banda conformal 90%. Sin 'month' devuelve resumen mensual; con 'month' (YYYY-MM) incluye la serie diaria de ese mes.

        Args:
            target: Equipo a proyectar (Price_Equipo1 o Price_Equipo2).
            month: Mes opcional en formato YYYY-MM (p.ej. 2023-12).
        """
        self._tools_used.append("get_projection")
        return get_projection(target, month)

    def tool_get_metrics(self) -> dict:
        """Métricas de validación del modelo de proyección (MAE, comparación con baseline naive, reducción de error, cobertura de bandas, error por horizonte 3M/4.5M/6M)."""
        self._tools_used.append("get_metrics")
        return get_metrics()

    def chat(self, session_id: str, message: str) -> dict:
        self._tools_used = []
        chat = self._chats.get(session_id)
        if chat is None:
            chat = self.client.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[self.tool_get_projection, self.tool_get_metrics],
                ),
            )
            self._chats[session_id] = chat
        resp = chat.send_message(message)
        reply = resp.text or ""
        return {"reply": reply, "tools_used": list(self._tools_used), "model": self.model}