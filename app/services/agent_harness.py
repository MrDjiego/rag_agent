import logging
import os
from pydantic_ai import Agent

from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.models.schemas import AnswerResponse

logger = logging.getLogger(__name__)

# Aseguramos que la llave esté en las variables de entorno del sistema operativo
# porque la abstracción de PydanticAI la buscará ahí automáticamente.
os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY

# Instanciamos nuestra base de datos local
vector_store = VectorStoreService()

def retrieve_context(query: str) -> str:
    """
    Busca en la base de datos vectorial local información relevante.
    El LLM usará esto para obtener el contexto antes de formular su respuesta.
    """
    logger.info(f"🤖 El Agente decidió buscar en ChromaDB la query: '{query}'")
    results = vector_store.search(query, n_results=3)
    
    if not results:
        return "No se encontró información en la base de datos."

    context_blocks = []
    for res in results:
        context_blocks.append(f"Fuente [{res['source']}]:\n{res['content']}")
    
    return "\n\n---\n\n".join(context_blocks)


SYSTEM_PROMPT = (
    "Eres un Staff AI Engineer virtual y un asistente riguroso. "
    "Tu objetivo es responder a las preguntas basándote ÚNICAMENTE "
    "en la información proporcionada por tus herramientas de búsqueda local. "
    "REGLAS ESTRICTAS: "
    "1. SIEMPRE debes usar la herramienta 'retrieve_context' antes de responder. "
    "2. Si la herramienta no devuelve resultados útiles, DEBES poner has_information=False "
    "y responder que no hay información en la base documental. ¡No inventes respuestas! "
    "3. Si encuentras la respuesta, cita las fuentes exactas en la lista 'sources'. "
    "4. Ignora tu conocimiento pre-entrenado; tu única verdad es el contexto recuperado."
)


def _build_agent(model_name: str) -> Agent:
    """Construye un agente Gemini con el mismo contrato y herramienta RAG."""
    return Agent(
        f"google:{model_name}",
        output_type=AnswerResponse,
        system_prompt=SYSTEM_PROMPT,
        tools=[retrieve_context],
    )


# Primer modelo expuesto para compatibilidad con cualquier import externo.
rag_agent = _build_agent(settings.GEMINI_AGENT_MODEL)

class AgentService:
    async def ask(self, user_query: str) -> AnswerResponse:
        """Punto de entrada asíncrono para consultar al agente."""
        models = [settings.GEMINI_AGENT_MODEL, *settings.GEMINI_AGENT_FALLBACK_MODELS]
        models = list(dict.fromkeys(models))
        last_error = None

        for model_name in models:
            try:
                logger.info(f"Ejecutando agente con {model_name}...")
                agent = rag_agent if model_name == settings.GEMINI_AGENT_MODEL else _build_agent(model_name)
                result = await agent.run(user_query)
                return result.output
            except Exception as e:
                last_error = e
                logger.warning(f"Fallo del agente con {model_name}: {e}")

        logger.error("Fallaron todos los modelos configurados para el agente.")
        raise RuntimeError("No fue posible obtener una respuesta con los modelos Gemini configurados.") from last_error