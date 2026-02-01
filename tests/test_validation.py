"""
Test suite for validation and hallucination prevention.
"""
import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from app.validation import (
    check_retrieval_quality,
    check_for_hallucinated_numbers,
    calculate_confidence_score,
    should_answer_question,
    add_confidence_disclaimer
)


class TestRetrievalQuality:
    """Test retrieval quality checks."""
    
    def test_empty_docs_returns_insufficient(self):
        """Test that empty doc list is flagged as insufficient."""
        is_sufficient, score, message = check_retrieval_quality([])
        
        assert not is_sufficient
        assert score == 0.0
        assert "No relevant documents" in message
    
    def test_low_relevance_scores_flagged(self):
        """Test that low relevance scores are flagged."""
        docs = [
            Document(page_content="test", metadata={"relevance_score": 0.3}),
            Document(page_content="test2", metadata={"relevance_score": 0.4})
        ]
        
        is_sufficient, score, message = check_retrieval_quality(docs)
        
        assert not is_sufficient
        assert score < 0.5
    
    def test_high_relevance_passes(self):
        """Test that high relevance scores pass."""
        docs = [
            Document(page_content="test", metadata={"relevance_score": 0.8}),
            Document(page_content="test2", metadata={"relevance_score": 0.9})
        ]
        
        is_sufficient, score, message = check_retrieval_quality(docs)
        
        assert is_sufficient
        assert score > 0.5


class TestHallucinationDetection:
    """Test hallucination detection."""
    
    def test_detects_fabricated_amounts(self):
        """Test detection of dollar amounts not in context."""
        answer = "The deductible is $5,000"
        context = "The deductible is $2,500"
        
        hallucinations = check_for_hallucinated_numbers(answer, context)
        
        assert len(hallucinations) > 0
        assert "$5,000" in hallucinations[0]
    
    def test_accepts_correct_amounts(self):
        """Test that correct amounts are not flagged."""
        answer = "The deductible is $2,500"
        context = "The deductible is $2,500 for all covered losses"
        
        hallucinations = check_for_hallucinated_numbers(answer, context)
        
        assert len(hallucinations) == 0
    
    def test_detects_fabricated_policy_numbers(self):
        """Test detection of policy numbers not in context."""
        answer = "Policy ABC-1234-5678 covers..."
        context = "This policy covers dwelling and personal property"
        
        hallucinations = check_for_hallucinated_numbers(answer, context)
        
        assert len(hallucinations) > 0


class TestConfidenceScoring:
    """Test confidence score calculation."""
    
    def test_high_confidence_calculation(self):
        """Test high confidence score calculation."""
        score, level = calculate_confidence_score(
            retrieval_score=0.9,
            faithfulness_score=0.95,
            has_specific_citations=True
        )
        
        assert score >= 0.8
        assert level == "high"
    
    def test_low_confidence_calculation(self):
        """Test low confidence score calculation."""
        score, level = calculate_confidence_score(
            retrieval_score=0.3,
            faithfulness_score=0.4,
            has_specific_citations=False
        )
        
        assert score < 0.6
        assert level == "low"
    
    def test_medium_confidence_calculation(self):
        """Test medium confidence score calculation."""
        score, level = calculate_confidence_score(
            retrieval_score=0.6,
            faithfulness_score=0.7,
            has_specific_citations=True
        )
        
        assert 0.6 <= score < 0.8
        assert level == "medium"


class TestAnswerDecision:
    """Test decision to answer or not."""
    
    def test_should_not_answer_when_insufficient(self):
        """Test that we don't answer with insufficient context."""
        docs = []
        quality = (False, 0.0, "No docs found")
        
        should_answer, reason = should_answer_question(docs, quality)
        
        assert not should_answer
        assert "No relevant" in reason
    
    def test_should_answer_when_sufficient(self):
        """Test that we answer with good context."""
        docs = [Document(page_content="test", metadata={"relevance_score": 0.8})]
        quality = (True, 0.8, "Good quality")
        
        should_answer, reason = should_answer_question(docs, quality)
        
        assert should_answer
        assert reason == ""


class TestDisclaimers:
    """Test confidence disclaimers."""
    
    def test_low_confidence_adds_disclaimer(self):
        """Test that low confidence adds warning."""
        answer = "The deductible is $2,500"
        
        result = add_confidence_disclaimer(answer, "low", 0.4)
        
        assert "Low Confidence" in result
        assert "⚠️" in result
    
    def test_high_confidence_no_disclaimer(self):
        """Test that high confidence doesn't add disclaimer."""
        answer = "The deductible is $2,500"
        
        result = add_confidence_disclaimer(answer, "high", 0.9)
        
        assert result == answer
        assert "⚠️" not in result
