from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    
    # Modelos dinámicos desde .env (con valores por defecto seguros)
    GEMINI_VISION_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_VISION_FALLBACK_MODELS: List[str] = ["gemini-3.6-flash", "gemini-3.1-pro-preview"]
    
    GEMINI_AGENT_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_AGENT_FALLBACK_MODELS: List[str] = ["gemini-3.6-flash", "gemini-3.1-pro-preview"]
    
    # Configuración de base de datos
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2" # Solo documental, Chroma usa IA local
    CHROMA_PERSIST_DIRECTORY: str = "./data/chromadb_store"
    COLLECTION_NAME: str = "rag_documents"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()