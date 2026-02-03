"""
Insurance Policy RAG Pipeline.
Handles document ingestion, metadata extraction, and core RAG operations.
"""
import os                           
import re 
import pickle
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import (
    DOCUMENTS_PATH,
    INSURANCE_POLICIES_PATH,
    VECTOR_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_POLICY_TYPES,
    HYBRID_SEARCH_ENABLED,
    BM25_WEIGHT,
    SEMANTIC_WEIGHT,
    BM25_INDEX_PATH
)

# Get the base directory (works both locally and on HuggingFace)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to default insurance knowledge documents
DEFAULT_DOCS_PATH = os.path.join(BASE_DIR, "data", "default_insurance_docs")

# Supported file extensions and their loaders
SUPPORTED_FILE_TYPES = {
    ".pdf": "PDF documents",
    ".txt": "Text files",
    ".docx": "Microsoft Word documents",
    ".md": "Markdown files"
}


def extract_policy_metadata(text: str, filename: str) -> Dict[str, Any]: #extracts structured metadata from unstructured text
    """
    Extract metadata from policy document text using regex patterns.
    
    Args:
        text: The document text content
        filename: Source filename
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {
        "source": filename,             # source pattern
        "policy_type": "unknown",
        "policy_number": None,
        "policyholder": None,
        "effective_date": None,
        "state": None,
    }
    
    # Detect policy type from filename and content
    filename_lower = filename.lower()
    text_lower = text.lower()
    
    if "homeowner" in filename_lower or "homeowner" in text_lower:
        metadata["policy_type"] = "homeowners"
    elif "auto" in filename_lower or "auto insurance" in text_lower or "personal auto" in text_lower:
        metadata["policy_type"] = "auto"
    elif "commercial" in filename_lower or "commercial property" in text_lower:
        metadata["policy_type"] = "commercial"
    elif "umbrella" in filename_lower or "umbrella" in text_lower:
        metadata["policy_type"] = "umbrella"
    elif "renter" in filename_lower or "renter" in text_lower:
        metadata["policy_type"] = "renters"
    
    # Extract policy number patterns
    policy_patterns = [
        r"Policy\s*Number[:\s]+([A-Z]{2,4}[-\s]?\d{4}[-\s]?\d{4,6})",
        r"Policy\s*#[:\s]+([A-Z]{2,4}[-\s]?\d{4}[-\s]?\d{4,6})",
        r"Policy\s*No\.?[:\s]+([A-Z]{2,4}[-\s]?\d{4}[-\s]?\d{4,6})",
    ]
    for pattern in policy_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata["policy_number"] = match.group(1).strip()
            break
    
    # Extract policyholder name
    holder_patterns = [
        r"(?:Named\s+)?Insured[:\s]+([A-Za-z\s\.]+?)(?:\n|Address|$)",
        r"Policyholder[:\s]+([A-Za-z\s\.]+?)(?:\n|Address|$)",
    ]
    for pattern in holder_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata["policyholder"] = match.group(1).strip()
            break
    
    # Extract effective date
    date_patterns = [
        r"Effective\s+Date[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        r"Effective[:\s]+(\d{1,2}/\d{1,2}/\d{4})",
        r"Policy\s+Period[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata["effective_date"] = match.group(1).strip()
            break
    
    # Extract state
    state_pattern = r"State[:\s]+([A-Za-z\s]+?)(?:\n|$)"
    match = re.search(state_pattern, text, re.IGNORECASE)
    if match:
        metadata["state"] = match.group(1).strip()
    
    return metadata


def extract_section_and_page(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract section number and page number from a chunk of text.
    
    Args:
        text: The chunk text
        
    Returns:
        Tuple of (section, page) or (None, None)
    """
    section = None
    page = None
    
    # Extract section patterns
    section_patterns = [
        r"SECTION\s+(\d+(?:\.\d+)?)[:\s]",
        r"Section\s+(\d+(?:\.\d+)?)[:\s]",
        r"(\d+\.\d+)\s+[A-Z][A-Z\s]+",  # e.g., "2.1 COVERED PERILS"
    ]
    for pattern in section_patterns:
        match = re.search(pattern, text)
        if match:
            section = f"Section {match.group(1)}"
            break
    
    # Extract page patterns
    page_patterns = [
        r"Page\s+(\d+)",
        r"page\s+(\d+)",
        r"P\.\s*(\d+)",
    ]
    for pattern in page_patterns:
        match = re.search(pattern, text)
        if match:
            page = f"Page {match.group(1)}"
            break
    
    return section, page


def ingest_documents() -> Dict[str, Any]:
    """
    Reads documents from:
    1. default_insurance_docs (general insurance knowledge - always included)
    2. insurance_policies (user's personal policies - optional)
    
    Extracts metadata, splits into chunks, creates embeddings, and stores in ChromaDB.
    
    Returns:
        Dictionary with ingestion statistics
    """
    # Lazy imports for FastAPI stability
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma

    documents = []
    policy_types_found = set()
    files_processed = 0
    default_docs_count = 0
    personal_docs_count = 0
    
    # Debug: Print paths
    print(f"DEBUG: BASE_DIR = {BASE_DIR}")
    print(f"DEBUG: DEFAULT_DOCS_PATH = {DEFAULT_DOCS_PATH}")
    print(f"DEBUG: DEFAULT_DOCS_PATH exists = {os.path.exists(DEFAULT_DOCS_PATH)}")
    
    # Ensure default docs directory exists
    if not os.path.exists(DEFAULT_DOCS_PATH):
        print(f"WARNING: Default docs path does not exist, creating: {DEFAULT_DOCS_PATH}")
        os.makedirs(DEFAULT_DOCS_PATH)
    else:
        files_in_dir = os.listdir(DEFAULT_DOCS_PATH)
        print(f"DEBUG: Files in DEFAULT_DOCS_PATH: {files_in_dir}")
    
    # Ensure insurance policies directory exists
    if not os.path.exists(INSURANCE_POLICIES_PATH):
        os.makedirs(INSURANCE_POLICIES_PATH)

    # STEP 1: Load default insurance knowledge documents (always included)
    print(f"Loading default insurance knowledge documents from {DEFAULT_DOCS_PATH}...")
    for file in os.listdir(DEFAULT_DOCS_PATH):
        file_path = os.path.join(DEFAULT_DOCS_PATH, file)
        
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            elif file.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()
            else:
                continue
            
            # Mark as default knowledge document
            full_text = "\n".join([doc.page_content for doc in docs])
            base_metadata = extract_policy_metadata(full_text, file)
            base_metadata["document_category"] = "general_knowledge"
            base_metadata["is_default_doc"] = True
            
            # Add metadata to each document page
            for doc in docs:
                doc.metadata.update(base_metadata)
            
            documents.extend(docs)
            policy_types_found.add(base_metadata["policy_type"])
            files_processed += 1
            default_docs_count += 1
            print(f"  ✓ Loaded default document: {file}")
            
        except Exception as e:
            print(f"  ✗ Error loading default document {file}: {str(e)}")
            continue

    # STEP 2: Load personal insurance policies (optional)
    print(f"\nLoading personal insurance policies from {INSURANCE_POLICIES_PATH}...")
    for file in os.listdir(INSURANCE_POLICIES_PATH):
        file_path = os.path.join(INSURANCE_POLICIES_PATH, file)
        
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            elif file.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()
            else:
                continue
            
            # Extract metadata from full document text
            full_text = "\n".join([doc.page_content for doc in docs])
            base_metadata = extract_policy_metadata(full_text, file)
            base_metadata["document_category"] = "personal_policy"
            base_metadata["is_default_doc"] = False
            
            # Add metadata to each document page
            for doc in docs:
                doc.metadata.update(base_metadata)
            
            documents.extend(docs)
            policy_types_found.add(base_metadata["policy_type"])
            files_processed += 1
            personal_docs_count += 1
            print(f"  ✓ Loaded personal policy: {file}")
            
        except Exception as e:
            print(f"  ✗ Error loading personal policy {file}: {str(e)}")
            continue

    if not documents:
        return {
            "chunks_count": 0,
            "documents_processed": 0,
            "default_docs_count": 0,
            "personal_docs_count": 0,
            "policy_types_found": [],
            "message": "No documents found. Please add default insurance knowledge documents."
        }
    
    print(f"\nTotal documents loaded: {files_processed} ({default_docs_count} default, {personal_docs_count} personal)")

    # Split text into chunks with larger size for insurance documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n================================================================================\n",
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    chunks = splitter.split_documents(documents)
    
    # Enhance each chunk with section/page metadata
    for chunk in chunks:
        section, page = extract_section_and_page(chunk.page_content)
        if section:
            chunk.metadata["section"] = section
        if page:
            chunk.metadata["page"] = page

    # Create embeddings & store in vector DB
    embeddings = OpenAIEmbeddings()
    
    # Clear existing collection and create new one
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    # Create and save BM25 index for hybrid search
    if HYBRID_SEARCH_ENABLED:
        tokenized_chunks = [chunk.page_content.lower().split() for chunk in chunks]
        bm25_index = BM25Okapi(tokenized_chunks)
        
        # Save BM25 index and chunk metadata
        bm25_data = {
            "index": bm25_index,
            "chunks": chunks  # Store chunks for retrieval
        }
        
        os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25_data, f)

    return {
        "chunks_count": len(chunks),
        "documents_processed": files_processed,
        "default_docs_count": default_docs_count,
        "personal_docs_count": personal_docs_count,
        "policy_types_found": list(policy_types_found),
        "message": f"Successfully ingested {len(chunks)} chunks from {files_processed} documents ({default_docs_count} default, {personal_docs_count} personal)."
    }


def get_vectordb():
    """
    Get the ChromaDB vector database instance.
    
    Returns:
        Chroma vector database instance
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )


def detect_personal_query_intent(query: str) -> bool:
    """
    Detect if the query is asking about the user's personal policy.
    
    Args:
        query: The user's question
        
    Returns:
        True if query appears to be asking about personal policy, False otherwise
    """
    query_lower = query.lower()
    
    # Personal pronouns and phrases that indicate personal policy questions
    personal_indicators = [
        "my ", "am i ", "do i ", "does my", "is my", "will my",
        "what is my", "what's my", "how much is my", 
        "under my", "in my policy", "my coverage",
        "i am ", "i have"
    ]
    
    return any(indicator in query_lower for indicator in personal_indicators)


def retrieve_with_metadata_filter(
    query: str,
    k: int = 6,
    policy_type: Optional[str] = None,
    policy_number: Optional[str] = None,
    prefer_personal: Optional[bool] = None
) -> List[Document]:
    """
    Retrieve documents with optional metadata filtering using hybrid search.
    Automatically detects if query is personal and prioritizes accordingly.
    
    Args:
        query: The search query
        k: Number of results to return
        policy_type: Filter by policy type
        policy_number: Filter by policy number
        prefer_personal: Whether to prioritize personal policy documents (auto-detected if None)
        
    Returns:
        List of relevant documents with relevance scores
    """
    # Auto-detect if user is asking about personal policy
    if prefer_personal is None:
        prefer_personal = detect_personal_query_intent(query)
    
    if HYBRID_SEARCH_ENABLED and os.path.exists(BM25_INDEX_PATH):
        return _hybrid_retrieve(query, k, policy_type, policy_number, prefer_personal)
    else:
        # Fallback to pure semantic search
        return _semantic_retrieve(query, k, policy_type, policy_number)


def _semantic_retrieve(
    query: str,
    k: int = 6,
    policy_type: Optional[str] = None,
    policy_number: Optional[str] = None
) -> List[Document]:
    """Pure semantic vector search."""
    vectordb = get_vectordb()
    
    # Build filter if specified
    where_filter = None
    if policy_type or policy_number:
        conditions = []
        if policy_type:
            conditions.append({"policy_type": policy_type.lower()})
        if policy_number:
            conditions.append({"policy_number": policy_number})
        
        if len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}
    
    # Retrieve with or without filter
    if where_filter:
        retriever = vectordb.as_retriever(
            search_kwargs={"k": k, "filter": where_filter}
        )
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    return retriever.invoke(query)


def _hybrid_retrieve(
    query: str,
    k: int = 6,
    policy_type: Optional[str] = None,
    policy_number: Optional[str] = None,
    prefer_personal: bool = True
) -> List[Document]:
    """
    Hybrid retrieval combining BM25 keyword search and semantic vector search.
    NOW WITH DOCUMENT PRIORITIZATION: Personal policies boosted over default docs.
    
    Args:
        query: The search query
        k: Number of results to return
        policy_type: Filter by policy type
        policy_number: Filter by policy number
        prefer_personal: Whether to boost personal policy scores (default: True)
        
    Returns:
        List of documents ranked by hybrid score with prioritization
    """
    # Load BM25 index
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    
    bm25_index = bm25_data["index"]
    all_chunks = bm25_data["chunks"]
    
    # BM25 keyword search
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Get top k*2 from BM25 (we'll combine with semantic later)
    bm25_top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k * 2]
    
    # Semantic vector search
    semantic_docs = _semantic_retrieve(query, k * 2, policy_type, policy_number)
    
    # Create hybrid scores
    doc_scores = {}   
    
    # Personal policy boost: 1.5x higher score for personal documents
    PERSONAL_POLICY_BOOST = 1.5    # v imp and unique
    
    # Add BM25 scores (normalized)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    for idx in bm25_top_indices:
        chunk = all_chunks[idx]
        # Apply metadata filter
        if policy_type and chunk.metadata.get("policy_type", "").lower() != policy_type.lower():
            continue
        if policy_number and chunk.metadata.get("policy_number") != policy_number:
            continue
        
        doc_id = f"{chunk.metadata.get('source', '')}_{chunk.page_content[:50]}"
        normalized_bm25 = bm25_scores[idx] / max_bm25
        base_score = BM25_WEIGHT * normalized_bm25
        
        # Apply personal policy boost
        is_personal = not chunk.metadata.get("is_default_doc", False)
        boost = PERSONAL_POLICY_BOOST if (prefer_personal and is_personal) else 1.0
        
        doc_scores[doc_id] = {
            "doc": chunk,
            "score": base_score * boost,
            "is_personal": is_personal
        }
    
    # Add semantic scores (already normalized by ChromaDB)
    for doc in semantic_docs:
        doc_id = f"{doc.metadata.get('source', '')}_{doc.page_content[:50]}"
        
        # Apply personal policy boost
        is_personal = not doc.metadata.get("is_default_doc", False)
        boost = PERSONAL_POLICY_BOOST if (prefer_personal and is_personal) else 1.0
        semantic_score = SEMANTIC_WEIGHT * 1.0 * boost
        
        if doc_id in doc_scores:
            # Combine scores
            doc_scores[doc_id]["score"] += semantic_score
        else:
            doc_scores[doc_id] = {
                "doc": doc,
                "score": semantic_score,
                "is_personal": is_personal
            }
    
    # Sort by hybrid score and return top k
    sorted_docs = sorted(
        doc_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:k]
    
    # Add relevance scores to metadata
    results = []
    for item in sorted_docs:
        doc = item["doc"]
        doc.metadata["relevance_score"] = item["score"]
        doc.metadata["is_personal_policy"] = item["is_personal"]
        results.append(doc)
    
    return results


def format_docs_with_citations(docs: List[Document]) -> str:
    """
    Format documents with section and source citations for the LLM context.
    NOW CLEARLY LABELS PERSONAL vs DEFAULT DOCUMENTS.
    
    Args:
        docs: List of documents
        
    Returns:
        Formatted string with citations and document type labels
    """
    # Separate personal policies from default docs
    personal_docs = [doc for doc in docs if not doc.metadata.get("is_default_doc", False)]
    default_docs = [doc for doc in docs if doc.metadata.get("is_default_doc", False)]
    
    formatted_parts = []
    
    # Format personal policies FIRST (higher priority)
    if personal_docs:
        formatted_parts.append("=" * 80)
        formatted_parts.append("📄 PERSONAL POLICY DOCUMENTS (User's Actual Coverage)")
        formatted_parts.append("=" * 80)
        
        for i, doc in enumerate(personal_docs, 1):
            source = str(doc.metadata.get("source", "Unknown"))
            section = str(doc.metadata.get("section", ""))
            page = doc.metadata.get("page", "")
            policy_type = str(doc.metadata.get("policy_type", ""))
            policy_number = doc.metadata.get("policy_number", "")
            
            if page != "" and page is not None:
                page = f"Page {page}" if isinstance(page, int) else str(page)
            
            citation_parts = [f"[PERSONAL POLICY | Source: {source}"]
            if policy_number:
                citation_parts.append(f"Policy #: {policy_number}")
            if policy_type:
                citation_parts.append(f"Type: {policy_type}")
            if section:
                citation_parts.append(str(section))
            if page:
                citation_parts.append(str(page))
            citation = ", ".join(citation_parts) + "]"
            
            formatted_parts.append(f"\n--- Personal Policy Document {i} {citation} ---\n{doc.page_content}")
    
    # Format default docs SECOND (educational context)
    if default_docs:
        formatted_parts.append("\n" + "=" * 80)
        formatted_parts.append("📚 GENERAL INSURANCE GUIDES (Educational Reference Only)")
        formatted_parts.append("=" * 80)
        
        for i, doc in enumerate(default_docs, 1):
            source = str(doc.metadata.get("source", "Unknown"))
            section = str(doc.metadata.get("section", ""))
            page = doc.metadata.get("page", "")
            policy_type = str(doc.metadata.get("policy_type", ""))
            
            if page != "" and page is not None:
                page = f"Page {page}" if isinstance(page, int) else str(page)
            
            citation_parts = [f"[GENERAL GUIDE | Source: {source}"]
            if policy_type:
                citation_parts.append(f"Type: {policy_type}")
            if section:
                citation_parts.append(str(section))
            if page:
                citation_parts.append(str(page))
            citation = ", ".join(citation_parts) + "]"
            
            formatted_parts.append(f"\n--- General Guide {i} {citation} ---\n{doc.page_content}")
    
    return "\n\n".join(formatted_parts)


def extract_sources_from_docs(docs: List[Document]) -> List[str]:
    """
    Extract unique source filenames from documents.
    
    Args:
        docs: List of documents
        
    Returns:
        List of unique source filenames
    """
    return list(set([doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")]))

