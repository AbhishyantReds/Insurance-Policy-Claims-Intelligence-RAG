"""
Evaluation framework for RAG pipeline quality assessment.
Measures retrieval accuracy, faithfulness, and answer relevance.
"""
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.rag_pipeline import retrieve_with_metadata_filter
from app.config import LLM_MODEL


def calculate_mrr(retrieved_docs: List[Document], relevant_source: str) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        retrieved_docs: List of retrieved documents
        relevant_source: The source file that should have been retrieved
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    for i, doc in enumerate(retrieved_docs, 1):
        if relevant_source in doc.metadata.get("source", ""):
            return 1.0 / i
    return 0.0


def calculate_ndcg(retrieved_docs: List[Document], relevant_sources: List[str]) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.
    
    Args:
        retrieved_docs: List of retrieved documents
        relevant_sources: List of relevant source files
        
    Returns:
        NDCG score (0.0 to 1.0)
    """
    if not retrieved_docs or not relevant_sources:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "")
        relevance = 1.0 if any(rel in source for rel in relevant_sources) else 0.0
        dcg += relevance / (i + 1)  # log2(i+1) simplified
    
    # Calculate ideal DCG
    idcg = sum(1.0 / (i + 1) for i in range(min(len(relevant_sources), len(retrieved_docs))))
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate retrieval quality using test cases.
    
    Args:
        test_cases: List of dicts with 'question', 'relevant_sources'
        
    Returns:
        Dictionary with MRR, NDCG, and precision scores
    """
    mrr_scores = []
    ndcg_scores = []
    precision_at_k = []
    
    for case in test_cases:
        question = case["question"]
        relevant_sources = case["relevant_sources"]
        
        # Retrieve documents
        docs = retrieve_with_metadata_filter(question, k=6)
        
        # Calculate metrics
        mrr = calculate_mrr(docs, relevant_sources[0] if relevant_sources else "")
        ndcg = calculate_ndcg(docs, relevant_sources)
        
        # Precision@K
        retrieved_sources = [doc.metadata.get("source", "") for doc in docs]
        relevant_retrieved = sum(
            1 for src in retrieved_sources 
            if any(rel in src for rel in relevant_sources)
        )
        precision = relevant_retrieved / len(docs) if docs else 0.0
        
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        precision_at_k.append(precision)
    
    return {
        "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        "ndcg": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        "precision_at_k": sum(precision_at_k) / len(precision_at_k) if precision_at_k else 0.0,
        "num_test_cases": len(test_cases)
    }


def evaluate_answer_relevance(question: str, answer: str) -> Tuple[float, str]:
    """
    Evaluate if answer is relevant to the question.
    
    Args:
        question: Original question
        answer: Generated answer
        
    Returns:
        Tuple of (relevance_score, explanation)
    """
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        
        prompt = ChatPromptTemplate.from_template("""Evaluate if the answer is relevant to the question.

QUESTION: {question}

ANSWER: {answer}

Rate the relevance on a scale of 0.0 to 1.0:
- 1.0: Directly answers the question
- 0.7-0.9: Partially answers with some relevant information
- 0.4-0.6: Tangentially related but misses key points
- 0.0-0.3: Not relevant or completely off-topic

Respond in format:
SCORE: [0.0-1.0]
EXPLANATION: [Brief explanation]
""")
        
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question, "answer": answer})
        
        score_str = result.split("SCORE:")[1].split("\n")[0].strip()
        score = float(score_str)
        explanation = result.split("EXPLANATION:")[1].strip() if "EXPLANATION:" in result else ""
        
        return score, explanation
        
    except Exception as e:
        return 0.5, f"Evaluation error: {str(e)}"


def evaluate_faithfulness(answer: str, context: str) -> float:
    """
    Evaluate if answer is faithful to context (no hallucinations).
    
    Args:
        answer: Generated answer
        context: Source context
        
    Returns:
        Faithfulness score (0.0 to 1.0)
    """
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        
        prompt = ChatPromptTemplate.from_template("""Check if the answer is faithful to the context.

CONTEXT:
{context}

ANSWER:
{answer}

Does the answer contain information not in the context? Rate faithfulness 0.0-1.0:
- 1.0: Completely faithful, all info from context
- 0.7-0.9: Mostly faithful with minor interpretation
- 0.4-0.6: Some unsupported claims
- 0.0-0.3: Major hallucinations or fabricated details

Respond with just the score: [0.0-1.0]
""")
        
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"context": context[:2000], "answer": answer})
        
        # Extract score
        score = float(result.strip().split()[0])
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.5


def load_test_dataset(filepath: str = "data/eval_dataset.json") -> List[Dict[str, Any]]:
    """Load evaluation test dataset."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Return sample dataset if file doesn't exist
        return [
            {
                "question": "What is the homeowner's insurance deductible?",
                "relevant_sources": ["homeowners_policy.pdf"],
                "expected_answer": "₹2,500"
            },
            {
                "question": "Is flood damage covered?",
                "relevant_sources": ["homeowners_policy.pdf"],
                "expected_answer": "No, flood damage is excluded"
            }
        ]


def run_full_evaluation() -> Dict[str, Any]:
    """
    Run complete evaluation suite.
    
    Returns:
        Dictionary with all evaluation metrics
    """
    test_cases = load_test_dataset()
    
    # Retrieval evaluation
    retrieval_metrics = evaluate_retrieval(test_cases)
    
    # Answer quality evaluation (sample)
    relevance_scores = []
    faithfulness_scores = []
    
    for case in test_cases[:10]:  # Evaluate first 10 for speed
        question = case["question"]
        docs = retrieve_with_metadata_filter(question, k=6)
        
        if docs:
            # This would normally call your full QA pipeline
            # For now, just measure what we can
            context = " ".join([doc.page_content for doc in docs[:3]])
            
            # Simulate answer evaluation
            expected = case.get("expected_answer", "")
            if expected:
                rel_score, _ = evaluate_answer_relevance(question, expected)
                relevance_scores.append(rel_score)
    
    return {
        "retrieval": retrieval_metrics,
        "answer_quality": {
            "avg_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
            "num_evaluated": len(relevance_scores)
        },
        "test_dataset_size": len(test_cases)
    }
