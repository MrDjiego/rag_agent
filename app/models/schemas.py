from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query: str = Field(
        ..., 
        description="Pregunta del usuario para el agente RAG.",
        json_schema_extra={"example": "¿Quién es el fundador de Heuristia?"}
    )


class AnswerResponse(BaseModel):
    has_information: bool = Field(
        description="Booleano. True solo si la herramienta de búsqueda encontró contexto útil. False si la base de datos no tiene la respuesta."
    )
    answer: str = Field(
        description="La respuesta detallada a la pregunta del usuario. Si has_information es False, explica que no tienes la información en la base de datos."
    )
    sources: List[str] = Field(
        description="Lista de fuentes exactas (nombres de archivo) usadas para la respuesta. Lista vacía si no hay fuentes."
    )