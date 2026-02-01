"""
Configuration settings for Insurance Policy RAG System.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Document Paths
DOCUMENTS_PATH = "data/documents"  # Legacy path for generic documents
INSURANCE_POLICIES_PATH = "data/insurance_policies"  # Insurance policy documents

# Vector Database
VECTOR_DB_PATH = "vectordb"
INSURANCE_COLLECTION_NAME = "insurance_policies"

# Chunking Configuration
CHUNK_SIZE = 1500  # Larger chunks to preserve policy sections
CHUNK_OVERLAP = 200  # Overlap to maintain context between chunks

# Retrieval Configuration
DEFAULT_K_RESULTS = 6  # Number of documents to retrieve
MAX_K_RESULTS = 10  # Maximum documents for comparison queries

# Hybrid Search Configuration
HYBRID_SEARCH_ENABLED = True  # Enable BM25 + semantic search
BM25_WEIGHT = 0.5  # Weight for BM25 keyword search (0.5 = equal weighting)
SEMANTIC_WEIGHT = 0.5  # Weight for semantic vector search
BM25_INDEX_PATH = "vectordb/bm25_index.pkl"  # BM25 index storage
MIN_RELEVANCE_SCORE = 0.5  # Minimum relevance score for retrieved docs

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"  # Model for insurance Q&A
LLM_TEMPERATURE = 0  # Deterministic for factual answers

# Validation & Quality Configuration
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for high-quality answers
ENABLE_FAITHFULNESS_CHECK = True  # Validate answers against context
MAX_RETRY_ATTEMPTS = 2  # Retry generation if validation fails

# Monitoring Configuration
MONITORING_ENABLED = True
MONITORING_DB_PATH = "monitoring.db"
LOG_QUERIES = True  # Log all queries for analysis
TRACK_METRICS = True  # Track performance metrics

# Insurance-Specific Configuration
SUPPORTED_POLICY_TYPES = [
    "homeowners",
    "home",
    "motor",
    "auto",
    "commercial",
    "health",
    "life",
    "travel"
]

# Indian Insurance Companies for detection
INDIAN_INSURERS = [
    "icici lombard",
    "bajaj allianz",
    "hdfc ergo",
    "tata aig",
    "reliance general",
    "new india assurance",
    "oriental insurance",
    "united india",
    "sbi general",
    "lic"
]

