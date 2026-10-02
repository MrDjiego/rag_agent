import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.agent_harness import AgentService

async def run_phase_3():
    print("Iniciando Fase 3: Ejecución del Agente Inteligente...\n")
    
    agent_service = AgentService()
    
    # Lista de pruebas: Una real y una trampa para probar la mitigación de alucinaciones
    queries = [
        "¿Quién es el fundador de Heuristia y dónde se fundó?",
        "¿Cuál es la capital de Francia y cuántos habitantes tiene?"
    ]

    for query in queries:
        print(f"👤 Pregunta: {query}")
        print("⏳ Agente pensando (Evaluando herramientas...)...\n")
        
        # Ejecutamos el agente asíncrono
        respuesta = await agent_service.ask(query)
        
        print("🤖 --- Respuesta Validada (Pydantic) ---")
        print(f"Info Encontrada : {respuesta.has_information}")
        print(f"Respuesta       : {respuesta.answer}")
        print(f"Fuentes Citadas : {respuesta.sources}")
        print("="*60 + "\n")

if __name__ == "__main__":
    # Como FastAPI es asíncrono, ya preparamos el código para correr en un event loop
    asyncio.run(run_phase_3())