"""
Insurance Policy RAG QA API.
Main FastAPI application entry point.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from app.ingest import router as ingest_router
from app.query import router as query_router
from app.monitoring import get_metrics_tracker
from app.config import VECTOR_DB_PATH

# Create FastAPI app
app = FastAPI(
    title="Insurance Policy RAG API",
    description="""
    An intelligent RAG system for insurance policy document analysis.
    
    ## Features
    - **Document Ingestion**: Process insurance policy PDFs with metadata extraction
    - **Policy Q&A**: Ask questions about coverage, limits, and deductibles
    - **Coverage Check**: Determine if specific scenarios are covered
    - **Policy Comparison**: Compare aspects across multiple policies
    - **Hybrid Search**: BM25 + Semantic vector search for better retrieval
    - **Quality Monitoring**: Track performance metrics and answer quality
    
    ## Supported Policy Types
    - Homeowners Insurance
    - Auto Insurance
    - Commercial Property Insurance
    - Umbrella/Excess Liability
    """,
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration #CORS is Cross-Origin Resource Sharing, it allows web apps from one domain to access resources from another domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest_router, tags=["Ingestion"])
app.include_router(query_router, tags=["Query"])


@app.on_event("startup")
async def startup_event():
    """
    Auto-ingest default insurance documents on startup if vector DB is empty.
    This ensures the system is always ready with general insurance knowledge.
    """
    print("\n" + "="*80)
    print("🚀 Starting Insurance Policy RAG API...")
    print("="*80)
    
    # Check if vector database exists and has content
    chroma_db_file = os.path.join(VECTOR_DB_PATH, "chroma.sqlite3")
    
    if not os.path.exists(chroma_db_file):
        print("\n📚 Vector database not found. Auto-ingesting default insurance documents...")
        try:
            from app.rag_pipeline import ingest_documents
            result = ingest_documents()
            print(f"\n✅ Auto-ingestion complete!")
            print(f"   - Documents processed: {result.get('documents_processed', 0)}")
            print(f"   - Default docs: {result.get('default_docs_count', 0)}")
            print(f"   - Personal docs: {result.get('personal_docs_count', 0)}")
            print(f"   - Chunks created: {result.get('chunks_count', 0)}")
            print(f"   - Policy types: {', '.join(result.get('policy_types_found', []))}")
        except Exception as e:
            print(f"\n⚠️  Auto-ingestion failed: {str(e)}")
            print("   You can manually ingest documents using the /ingest endpoint")
    else:
        print("\n✅ Vector database found. Default insurance knowledge is ready!")
        
    print("\n" + "="*80)
    print("🌐 API is ready at http://127.0.0.1:8000")
    print("📖 Documentation at http://127.0.0.1:8000/docs")
    print("="*80 + "\n")


@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Insurance Policy RAG API",
        "version": "2.1.0",
        "features": ["hybrid_search", "validation", "monitoring"]
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "endpoints": {
            "ingest": "/ingest",
            "query": "/query",
            "check_coverage": "/check-coverage",
            "compare_policies": "/compare-policies",
            "metrics": "/metrics"
        }
    }


@app.get("/metrics", tags=["Monitoring"])
def get_metrics(days: int = 7) -> Dict[str, Any]:
    """
    Get aggregated performance metrics.
    
    Args:
        days: Number of days to include in metrics (default: 7)
        
    Returns:
        Dictionary with performance statistics
    """
    tracker = get_metrics_tracker()
    
    summary = tracker.get_metrics_summary(days=days)
    endpoint_stats = tracker.get_endpoint_stats()
    low_confidence = tracker.get_low_confidence_queries(limit=10)
    
    return {
        "period_days": days,
        "summary": summary,
        "by_endpoint": endpoint_stats,
        "low_confidence_queries": low_confidence,
        "alerts": _generate_alerts(summary)
    }


def _generate_alerts(summary: Dict[str, Any]) -> list:
    """Generate alerts based on metrics."""
    alerts = []
    
    if summary.get("success_rate", 1.0) < 0.9:
        alerts.append({
            "level": "warning",
            "message": f"Success rate is low: {summary['success_rate']:.1%}"
        })
    
    if summary.get("avg_response_time", 0) > 5.0:
        alerts.append({
            "level": "warning",
            "message": f"Average response time is high: {summary['avg_response_time']:.2f}s"
        })
    
    if summary.get("avg_confidence", 1.0) < 0.7:
        alerts.append({
            "level": "info",
            "message": f"Average confidence is below threshold: {summary['avg_confidence']:.2f}"
        })
    
    return alerts




