"""
Test suite for hybrid search retrieval.
Tests BM25, semantic search, and hybrid combination.
"""
import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from app.rag_pipeline import (
    _semantic_retrieve,
    _hybrid_retrieve,
    retrieve_with_metadata_filter
)
from app.config import HYBRID_SEARCH_ENABLED


class TestSemanticRetrieval:
    """Test semantic vector search."""
    
    def test_semantic_retrieve_returns_documents(self):
        """Test that semantic search returns documents."""
        # This test requires actual vector DB - mock for now
        with patch('app.rag_pipeline.get_vectordb') as mock_db:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = [
                Document(page_content="Test content", metadata={"source": "test.pdf"})
            ]
            mock_db.return_value.as_retriever.return_value = mock_retriever
            
            docs = _semantic_retrieve("test query", k=5)
            
            assert len(docs) > 0
            assert isinstance(docs[0], Document)
    
    def test_semantic_retrieve_with_filters(self):
        """Test semantic search with metadata filters."""
        with patch('app.rag_pipeline.get_vectordb') as mock_db:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = []
            mock_db.return_value.as_retriever.return_value = mock_retriever
            
            docs = _semantic_retrieve("test", k=5, policy_type="homeowners")
            
            # Check that filter was applied
            call_args = mock_db.return_value.as_retriever.call_args
            assert "filter" in call_args[1]["search_kwargs"]


class TestHybridRetrieval:
    """Test hybrid BM25 + semantic search."""
    
    @pytest.mark.skipif(not HYBRID_SEARCH_ENABLED, reason="Hybrid search disabled")
    def test_hybrid_combines_scores(self):
        """Test that hybrid search combines BM25 and semantic scores."""
        # Requires BM25 index to exist - integration test
        pass
    
    def test_retrieve_falls_back_to_semantic(self):
        """Test fallback to semantic when hybrid unavailable."""
        with patch('app.rag_pipeline.HYBRID_SEARCH_ENABLED', False):
            with patch('app.rag_pipeline._semantic_retrieve') as mock_semantic:
                mock_semantic.return_value = []
                
                docs = retrieve_with_metadata_filter("test query")
                
                mock_semantic.assert_called_once()


class TestRetrievalQuality:
    """Test retrieval quality metrics."""
    
    def test_relevance_scores_added(self):
        """Test that relevance scores are added to metadata."""
        # Would need actual retrieval to test
        pass
    
    def test_metadata_filtering_works(self):
        """Test that policy_type and policy_number filters work."""
        with patch('app.rag_pipeline.get_vectordb') as mock_db:
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = [
                Document(
                    page_content="Test",
                    metadata={"policy_type": "homeowners", "source": "test.pdf"}
                )
            ]
            mock_db.return_value.as_retriever.return_value = mock_retriever
            
            docs = _semantic_retrieve("test", k=5, policy_type="homeowners")
            
            # Verify filter was constructed
            assert mock_db.return_value.as_retriever.called
