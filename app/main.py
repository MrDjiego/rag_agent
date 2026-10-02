from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="Heuristia AI RAG Agent",
    description="Motor de Extracción y Análisis Documental mediante Agentes Autónomos",
    version="1.0.0"
)

# Configuración CORS para permitir peticiones desde un frontend si se requiere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registramos los endpoints de nuestra API
app.include_router(router, prefix="/api/v1", tags=["RAG Services"])

@app.get("/health", tags=["System"])
def health_check():
    """Endpoint de monitoreo para balanceadores de carga (Kubernetes/AWS)"""
    return {"status": "ok", "message": "El motor RAG está operando nominalmente."}

## http://localhost:8000/docs