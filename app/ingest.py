"""
Document Ingestion Endpoint.
exposes an API endpoint to ingest insurance policy documents.
"""
from fastapi import APIRouter, HTTPException #use FastAPI tools to create API routes and handle errors.

from app.rag_pipeline import ingest_documents #Import the function that actually performs document ingestion.
from app.models import IngestResponse

router = APIRouter() #attach the api endpoint to this router


@router.post("/ingest", response_model=IngestResponse) #it says when someone whats to post, it will call ingest function
def ingest() -> IngestResponse:
    """
    Ingest all insurance policy documents from data/insurance_policies/.
    
    Processes PDF and TXT files, extracts metadata (policy type, number, etc.),
    chunks documents preserving section boundaries, and stores in ChromaDB.
    
    Returns:
        IngestResponse with ingestion statistics
    """
    try:
        result = ingest_documents() #trigger the ingestion process to actaul rag pipeline, all laod chunk embed in chroma happens here in ingest_documents
        
        return IngestResponse(
            message=result.get("message", "Ingestion completed"), # all these for better debugging, can just return sucess
            chunks_count=result.get("chunks_count", 0),
            documents_processed=result.get("documents_processed", 0),
            policy_types_found=result.get("policy_types_found", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

