import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME
        )

    def ingest_documents(self, chunks: List[str], source_name: str) -> None:
        """Vectoriza los chunks usando IA local y los almacena en ChromaDB."""
        if not chunks:
            logger.warning("No hay chunks para ingestar.")
            return

        logger.info(f"Vectorizando {len(chunks)} chunks de {source_name} de forma LOCAL...")
        
        ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Ingesta completada exitosamente en la colección '{settings.COLLECTION_NAME}'.")

    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Busca los fragmentos más relevantes para una consulta semántica.
        Retorna una lista de diccionarios con el contexto y la metadata.
        """
        if self.collection.count() == 0:
            logger.warning("La colección está vacía. No se puede buscar.")
            return []

        logger.info(f"Ejecutando búsqueda semántica para: '{query}'")
        
        # Chroma se encarga automáticamente de vectorizar la query usando el mismo modelo local
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "content": results['documents'][0][i],
                    "source": results['metadatas'][0][i]['source'],
                    "distance": results['distances'][0][i] if 'distances' in results and results['distances'] else None
                })
        
        return formatted_results