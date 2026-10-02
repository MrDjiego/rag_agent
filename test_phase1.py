import sys
import os
from pathlib import Path

# Inyecta el directorio actual al path de módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.document_parser import DocumentParser
from app.services.vector_store import VectorStoreService

def run_phase_1():
    raw_dir = Path("./data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Extensiones que nuestro DocumentParser soporta
    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.txt']
    file_path = None
    
    # Escaneamos el directorio buscando el primer archivo válido
    for file in raw_dir.iterdir():
        if file.is_file() and file.suffix.lower() in supported_extensions:
            file_path = file
            break # Detenemos la búsqueda al encontrar el primero
            
    if not file_path:
        print(f"Por favor, coloca un archivo válido ({', '.join(supported_extensions)}) en la carpeta {raw_dir}.")
        return

    print(f"Iniciando Fase 1: Ingesta del archivo encontrado -> {file_path.name}")
    
    parser = DocumentParser(chunk_size=1000, chunk_overlap=200)
    
    # Extraer texto (el parser enrutará por su cuenta según la extensión)
    raw_text = parser.extract_text(file_path) 
    
    print("\n--- Texto extraído (Muestra de 300 caracteres) ---")
    print(raw_text[:300] + "...\n")

    chunks = parser.chunk_text(raw_text)
    print(f"Total de chunks semánticos generados: {len(chunks)}")

    vector_store = VectorStoreService()
    vector_store.ingest_documents(chunks, source_name=file_path.name)
    
    collection_count = vector_store.collection.count()
    print(f"Chunks totales actuales en la DB vectorial: {collection_count}")
    print("Fase 1 completada con éxito.")

if __name__ == "__main__":
    run_phase_1()