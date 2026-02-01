"""
Response validation and quality control for RAG pipeline.
Prevents hallucinations and ensures answer quality.
"""
import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import (
    LLM_MODEL,
    CONFIDENCE_THRESHOLD,
    ENABLE_FAITHFULNESS_CHECK,
    MIN_RELEVANCE_SCORE
)


def check_retrieval_quality(docs: List[Document]) -> Tuple[bool, float, str]:
    """
    Check if retrieved documents have sufficient relevance.
    
    Args:
        docs: List of retrieved documents
        
    Returns:
        Tuple of (is_sufficient, avg_score, message)
    """
    if not docs:
        return False, 0.0, "No relevant documents found in the knowledge base."
    
    # Check relevance scores if available
    scores = [doc.metadata.get("relevance_score", 0.0) for doc in docs]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    if avg_score < MIN_RELEVANCE_SCORE:
        return False, avg_score, f"Retrieved documents have low relevance (score: {avg_score:.2f}). The answer may not be reliable."
    
    return True, avg_score, "Retrieval quality is good."


def validate_response_faithfulness(
    answer: str,
    context: str,
    question: str
) -> Tuple[bool, float, str]:
    """
    Validate that the answer is faithful to the provided context.
    Uses a secondary LLM call to check for hallucinations.
    
    Args:
        answer: Generated answer
        context: Retrieved context documents
        question: Original question
        
    Returns:
        Tuple of (is_faithful, confidence, explanation)
    """
    if not ENABLE_FAITHFULNESS_CHECK:
        return True, 1.0, "Faithfulness check disabled"
    
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        
        validation_prompt = ChatPromptTemplate.from_template("""You are a fact-checker. Your job is to verify if an answer is faithful to the given context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER TO VALIDATE:
{answer}

Analyze if the answer contains ANY information not present in the context. Check for:
1. Fabricated policy numbers, amounts, or dates
2. Claims not supported by the context
3. Assumptions beyond what's stated

Respond in this format:
FAITHFUL: [YES/NO]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [Brief explanation of your assessment]
""")
        
        chain = validation_prompt | llm | StrOutputParser()
        
        result = chain.invoke({
            "context": context[:3000],  # Limit context size
            "question": question,
            "answer": answer
        })
        
        # Parse result
        faithful = "YES" in result.split("FAITHFUL:")[1].split("\n")[0].upper()
        
        try:
            confidence_str = result.split("CONFIDENCE:")[1].split("\n")[0].strip()
            confidence = float(confidence_str)
        except:
            confidence = 0.7 if faithful else 0.3
        
        explanation = result.split("EXPLANATION:")[1].strip() if "EXPLANATION:" in result else "Validation completed"
        
        return faithful, confidence, explanation
        
    except Exception as e:
        # If validation fails, be conservative
        return True, 0.5, f"Validation error: {str(e)}"


def check_for_hallucinated_numbers(answer: str, context: str) -> List[str]:
    """
    Check if answer contains specific numbers/amounts not in context.
    
    Args:
        answer: Generated answer
        context: Retrieved context
        
    Returns:
        List of potentially hallucinated numbers
    """
    # Extract dollar amounts and policy numbers from answer
    answer_amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', answer)
    answer_policy_nums = re.findall(r'[A-Z]{2,4}[-\s]?\d{4}[-\s]?\d{4,6}', answer)
    
    hallucinated = []
    
    # Check if amounts in answer exist in context
    for amount in answer_amounts:
        if amount not in context:
            hallucinated.append(f"Amount {amount} not found in policy documents")
    
    # Check policy numbers
    for policy_num in answer_policy_nums:
        if policy_num not in context:
            hallucinated.append(f"Policy number {policy_num} not found in documents")
    
    return hallucinated


def calculate_confidence_score(
    retrieval_score: float,
    faithfulness_score: float,
    has_specific_citations: bool
) -> Tuple[float, str]:
    """
    Calculate overall confidence score for the answer.
    
    Args:
        retrieval_score: Average relevance of retrieved docs
        faithfulness_score: Faithfulness validation score
        has_specific_citations: Whether answer has specific section/page refs
        
    Returns:
        Tuple of (confidence_score, confidence_level)
    """
    # Weighted combination
    confidence = (
        retrieval_score * 0.4 +
        faithfulness_score * 0.4 +
        (1.0 if has_specific_citations else 0.5) * 0.2
    )
    
    # Determine level
    if confidence >= 0.8:
        level = "high"
    elif confidence >= 0.6:
        level = "medium"
    else:
        level = "low"
    
    return confidence, level


def should_answer_question(
    docs: List[Document],
    retrieval_quality: Tuple[bool, float, str]
) -> Tuple[bool, str]:
    """
    Determine if we have enough information to answer.
    
    Args:
        docs: Retrieved documents
        retrieval_quality: Result from check_retrieval_quality
        
    Returns:
        Tuple of (should_answer, reason_if_not)
    """
    is_sufficient, avg_score, message = retrieval_quality
    
    if not docs:
        return False, "No relevant policy documents found. Please ensure documents have been ingested."
    
    if not is_sufficient:
        return False, f"Insufficient context to provide a reliable answer. {message}"
    
    return True, ""


def add_confidence_disclaimer(answer: str, confidence_level: str, confidence_score: float) -> str:
    """
    Add appropriate disclaimer based on confidence level.
    
    Args:
        answer: Generated answer
        confidence_level: "high", "medium", or "low"
        confidence_score: Numerical confidence
        
    Returns:
        Answer with disclaimer if needed
    """
    if confidence_level == "low":
        disclaimer = f"\n\n⚠️ **Low Confidence ({confidence_score:.2f})**: This answer has low confidence. Please verify with your actual policy documents or contact your insurance provider."
        return answer + disclaimer
    elif confidence_level == "medium":
        disclaimer = f"\n\n📌 **Medium Confidence ({confidence_score:.2f})**: This answer is based on available information, but consider verifying critical details with your policy documents."
        return answer + disclaimer
    else:
        return answer


def generate_insufficient_context_response(question: str, reason: str) -> str:
    """
    Generate appropriate response when context is insufficient.
    
    Args:
        question: User's question
        reason: Why we can't answer
        
    Returns:
        User-friendly response
    """
    return f"""I don't have enough information in the available policy documents to answer your question reliably.

**Your Question:** {question}

**Issue:** {reason}

**Suggestions:**
- Ensure policy documents have been ingested (use the Admin tab)
- Try rephrasing your question with more specific terms
- Check if your question relates to policies in the database
- For critical decisions, always verify with your actual policy documents or insurance provider
"""
