import sys
import os

# Inyecta el directorio actual al path de módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.vector_store import VectorStoreService

def run_phase_2():
    print("Iniciando Fase 2: Prueba del Retriever Semántico...\n")
    
    vector_store = VectorStoreService()
    
    # Haremos una pregunta específica sobre el documento de Heuristia que ingestaste
    query = "¿Quién es el fundador de Heuristia y en dónde se fundó la empresa?"
    print(f"Pregunta del Usuario: {query}\n")
    
    # Buscamos el top 2 de resultados más similares
    resultados = vector_store.search(query=query, n_results=2)
    
    print("--- Resultados Recuperados de ChromaDB ---")
    if not resultados:
        print("No se encontraron coincidencias.")
    else:
        for idx, res in enumerate(resultados):
            print(f"\n[Resultado {idx + 1}]")
            print(f"Fuente: {res['source']}")
            # La distancia indica qué tan cerca están los vectores (menor es mejor)
            print(f"Distancia Matemática: {res['distance']:.4f}")
            print(f"Contenido Extraído: {res['content'][:150]}...")

    print("\nFase 2 completada con éxito.")

if __name__ == "__main__":
    run_phase_2()