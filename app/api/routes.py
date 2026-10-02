import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File

from app.models.schemas import QueryRequest, AnswerResponse
from app.services.agent_harness import AgentService
from app.services.document_parser import DocumentParser
from app.services.vector_store import VectorStoreService

router = APIRouter()

# Inyección de dependencias (Factory pattern)
def get_agent_service() -> AgentService:
    return AgentService()

def get_vector_store() -> VectorStoreService:
    return VectorStoreService()

def get_document_parser() -> DocumentParser:
    return DocumentParser(chunk_size=1000, chunk_overlap=200)

@router.post("/upload_document", summary="Sube e ingesta un documento a la base vectorial")
async def upload_document(
    file: UploadFile = File(...),
    parser: DocumentParser = Depends(get_document_parser),
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    try:
        # 1. Guardar el archivo temporalmente
        raw_dir = Path("./data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        file_path = raw_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Extracción y Chunking
        raw_text = parser.extract_text(file_path)
        chunks = parser.chunk_text(raw_text)
        
        # 3. Vectorización en ChromaDB
        vector_store.ingest_documents(chunks, source_name=file.filename)
        
        return {
            "status": "success",
            "message": f"Documento '{file.filename}' ingestado correctamente.",
            "chunks_generados": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")

@router.post("/ask_agent", response_model=AnswerResponse, summary="Consulta al agente RAG")
async def ask_agent(
    request: QueryRequest,
    agent: AgentService = Depends(get_agent_service)
):
    try:
        response = await agent.ask(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el agente: {str(e)}")