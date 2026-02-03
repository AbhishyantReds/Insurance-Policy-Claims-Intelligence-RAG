"""
Configuration settings for Insurance Policy RAG System.
"""
import os
from dotenv import load_dotenv

load_dotenv() # loads the .env file

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Document Paths,   -it controls what documnents are read during ingestion
DOCUMENTS_PATH = "data/documents"  # Legacy path for generic documents
INSURANCE_POLICIES_PATH = "data/insurance_policies"  # Insurance user specific policy documents

# Vector Database,  -this is where the embeddings and indexes are stored
VECTOR_DB_PATH = "vectordb" # embedding are written to disk, doesnt redo it on every run
INSURANCE_COLLECTION_NAME = "insurance_policies"

# Chunking Configuration
CHUNK_SIZE = 1500  # Larger chunks to preserve policy sections, smaller will lose context, more is higher cost
CHUNK_OVERLAP = 200  # Overlap to maintain context between chunks

# Retrieval Configuration
DEFAULT_K_RESULTS = 6  # these are top 6 most relevant documents to retrieve, based on chunk score
MAX_K_RESULTS = 10  # if it requires more context can go up to 10

# Hybrid Search Configuration
HYBRID_SEARCH_ENABLED = True  # Enable BM25 + semantic search
BM25_WEIGHT = 0.5  # Weight for BM25 keyword search (0.5 = equal weighting)
SEMANTIC_WEIGHT = 0.5  # Weight for semantic vector search
BM25_INDEX_PATH = "vectordb/bm25_index.pkl"  # BM25 builds keyword index over all files, so it doesnt need to be redone every time
MIN_RELEVANCE_SCORE = 0.5  # Minimum relevance score for retrieved docs

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"  # Model for insurance Q&A, can be changed to any other openai model
LLM_TEMPERATURE = 0  # more factual answers, less creative
LLM_MAX_TOKENS = 1000  # Max tokens in the response, do that it doesnt go off-topic

# Validation & Quality Configuration
CONFIDENCE_THRESHOLD = 0.6  # its about How much of the answer overlaps with retrieved context
ENABLE_FAITHFULNESS_CHECK = True  # Validate answers against context
MAX_RETRY_ATTEMPTS = 2  # Retry generation if validation fails

# Monitoring Configuration
MONITORING_ENABLED = True #baically it keep records and help improve system over time
MONITORING_DB_PATH = "monitoring.db" # SQLite database, all the montoring data is stored here
LOG_QUERIES = True  # Log all queries for analysis
TRACK_METRICS = True  # Track performance metrics

# Insurance-Specific Configuration # giving context about what all we can answer
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

