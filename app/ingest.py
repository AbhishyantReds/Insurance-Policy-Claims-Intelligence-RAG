"""
Document Ingestion Endpoint.
Handles ingestion of insurance policy documents into the vector database.
"""
from fastapi import APIRouter, HTTPException

from app.rag_pipeline import ingest_documents
from app.models import IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    """
    Ingest all insurance policy documents from data/insurance_policies/.
    
    Processes PDF and TXT files, extracts metadata (policy type, number, etc.),
    chunks documents preserving section boundaries, and stores in ChromaDB.
    
    Returns:
        IngestResponse with ingestion statistics
    """
    try:
        result = ingest_documents()
        
        return IngestResponse(
            message=result.get("message", "Ingestion completed"),
            chunks_count=result.get("chunks_count", 0),
            documents_processed=result.get("documents_processed", 0),
            policy_types_found=result.get("policy_types_found", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

