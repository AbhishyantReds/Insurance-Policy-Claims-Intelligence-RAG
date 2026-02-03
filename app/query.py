"""
Insurance Policy Query Endpoints.
Handles policy Q&A, coverage checks, and policy comparisons.
"""
import json
import time
from typing import Optional, List
from fastapi import APIRouter, HTTPException

from langchain_openai import ChatOpenAI #llm from openai
from langchain_core.prompts import ChatPromptTemplate  #template for chat prompts
from langchain_core.output_parsers import StrOutputParser #convert llm output to string

from app.config import LLM_MODEL, LLM_TEMPERATURE, DEFAULT_K_RESULTS, MAX_K_RESULTS
from app.models import (
    QueryRequest, QueryResponse, Citation,
    CoverageCheckRequest, CoverageCheckResponse, ExclusionInfo,
    CompareRequest, CompareResponse, PolicyComparisonItem
)
from app.rag_pipeline import (    #all the stuff is happening in rag pipeline, the query is just the orchestration.
    retrieve_with_metadata_filter,
    format_docs_with_citations,
    extract_sources_from_docs
)
from app.validation import (   #again here its happening in validation module.
    check_retrieval_quality,
    validate_response_faithfulness,
    check_for_hallucinated_numbers,
    calculate_confidence_score,
    should_answer_question,
    add_confidence_disclaimer,
    generate_insufficient_context_response
)
from app.monitoring import get_metrics_tracker

router = APIRouter()


# ==============================================================================
# Insurance Query Prompt Templates
# ==============================================================================

INSURANCE_QA_PROMPT = """You are an expert insurance policy analyst. Answer questions based ONLY on the provided policy documents.

CRITICAL RULES:
1. ONLY use information from the provided context. Do not use external knowledge about insurance.
2. Always cite the specific policy section and page when available (e.g., "Section 3.1, Page 12").
3. When discussing coverage limits, always include the exact rupee amounts found in the documents.
4. When discussing deductibles, always include the exact rupee amounts found in the documents.
5. If something is NOT covered, clearly state it's an EXCLUSION and cite the exclusion section.
6. If the information is not found in the provided documents, respond EXACTLY with: "This information is not found in the provided policy documents."
7. Never fabricate policy numbers, rupee amounts, dates, or any other specific details.
8. If you're uncertain, say so explicitly.

EXAMPLES OF GOOD ANSWERS:

Q: What is my homeowner's deductible?
A: According to Section 2, Deductibles (Page 5), your homeowner's insurance deductible is ₹2,500 for all covered losses.

Q: Is flood damage covered?
A: No, flood damage is NOT covered. According to Section 4, Exclusions (Page 8), "We do not cover loss caused by flood, surface water, or water that backs up through sewers or drains."

Q: What is the dwelling coverage limit?
A: This information is not found in the provided policy documents.

POLICY DOCUMENTS:
{context}

QUESTION: {question}

Provide your answer in the following format:
ANSWER: [Your detailed answer with citations]
COVERAGE LIMITS: [List any coverage limits mentioned, or "Not applicable"]
DEDUCTIBLES: [List any deductibles mentioned, or "Not applicable"]
CITATIONS: [List each section/page cited]"""

COVERAGE_CHECK_PROMPT = """You are an expert insurance claims adjuster. Analyze whether the described scenario is covered under the provided policy documents.

CRITICAL RULES:
1. Carefully check BOTH coverage sections AND exclusion sections.
2. Exclusions override coverage - if an exclusion applies, the claim is NOT covered.
3. Be precise about policy section references - cite exact section numbers.
4. Include exact coverage limits and deductibles in rupees (₹) ONLY if found in the documents.
5. If you cannot determine coverage from the documents, set confidence to "low" and explain why.
6. Never fabricate policy details, amounts, or section numbers.
7. Look for specific exclusion language that might apply to the scenario.

EXAMPLES:

Scenario: "My basement flooded from heavy rain"
Analysis: Check for water damage coverage AND flood exclusions. If policy excludes "surface water" or "flooding", claim is NOT covered even if water damage is generally covered.

Scenario: "Someone slipped on my icy driveway"
Analysis: Check liability coverage section. If covered, note the liability limit (e.g., ₹300,000) and any applicable deductible.

POLICY DOCUMENTS:
{context}

CLAIM SCENARIO: {scenario}

Analyze this scenario and respond in EXACTLY this JSON format:
{{
    "is_covered": true or false,
    "coverage_determination": "Detailed explanation of why this is or is not covered, citing specific policy sections",
    "policy_section": "Section X.X" or null,
    "coverage_limit": "₹X,XXX" or null,
    "deductible": "₹X,XXX" or null,
    "exclusions_checked": [
        {{"section": "Section X.X", "description": "Brief description", "applies": true/false}}
    ],
    "conditions": "Any conditions or requirements for coverage" or null,
    "confidence": "high", "medium", or "low"
}}

Use "low" confidence if:
- The documents don't contain enough information
- The scenario is ambiguous
- Multiple interpretations are possible"""

COMPARE_POLICIES_PROMPT = """You are an expert insurance policy analyst. Compare the requested aspect across the provided policy documents.

POLICY DOCUMENTS:
{context}

COMPARISON REQUEST: {comparison_query}

Provide a comparison in EXACTLY this JSON format:
{{
    "comparison_type": "What is being compared (e.g., 'Deductibles', 'Liability Limits')",
    "comparison_items": [
        {{
            "policy_type": "homeowners/auto/commercial/etc",
            "policy_number": "Policy number if available" or null,
            "value": "The value being compared (e.g., '₹1,000')",
            "section": "Section reference" or null,
            "notes": "Any relevant notes" or null
        }}
    ],
    "summary": "Natural language summary of the comparison"
}}"""


# ==============================================================================
# Query Endpoint
# ==============================================================================

@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query insurance policy documents with intelligent retrieval and citations.
    Includes validation and monitoring.
    
    Args:
        request: QueryRequest with question and optional filters
        
    Returns:
        QueryResponse with answer, citations, limits, and sources
    """
    start_time = time.time()     #for monitoring
    metrics_tracker = get_metrics_tracker()
    
    try:
        # Retrieve relevant documents
        docs = retrieve_with_metadata_filter(   #runs hybrid search with metadata filtering and returns top k docs
            query=request.question,
            k=DEFAULT_K_RESULTS,
            policy_type=request.policy_type,
            policy_number=request.policy_number
        )
        
        # Check retrieval quality
        retrieval_quality = check_retrieval_quality(docs)
        can_answer, reason = should_answer_question(docs, retrieval_quality)  #if confidence is low or no docs, it will not answer and give reason instead, this is to avoid hallucination when we dont have enough context
        
        if not can_answer:
            response_time = time.time() - start_time
            metrics_tracker.log_query(
                question=request.question,
                endpoint="/query",
                response_time=response_time,
                confidence_score=0.0,
                confidence_level="low",
                retrieval_score=retrieval_quality[1],
                num_docs=len(docs),
                success=True
            )
            
            return QueryResponse(
                answer=generate_insufficient_context_response(request.question, reason),
                citations=[],
                sources=[],
                confidence="low"
            )
        
        # Format context with citations
        context = format_docs_with_citations(docs)
        sources = extract_sources_from_docs(docs)
        
        # Create LLM chain
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        prompt = ChatPromptTemplate.from_template(INSURANCE_QA_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        # Get answer
        response = chain.invoke({
            "context": context,
            "question": request.question
        })
        
        # Parse the response to extract structured data
        answer, citations, limits, deductibles = _parse_qa_response(response, docs)
        
        # Validate response faithfulness
        is_faithful, faithfulness_score, faith_explanation = validate_response_faithfulness(
            answer, context, request.question
        )
        
        # Check for hallucinated numbers
        hallucinations = check_for_hallucinated_numbers(answer, context)
        
        # Calculate confidence
        has_citations = len(citations) > 0
        confidence_score, confidence_level = calculate_confidence_score(
            retrieval_quality[1],
            faithfulness_score,
            has_citations
        )
        
        # Reduce confidence if hallucinations detected
        if hallucinations:
            confidence_score *= 0.5
            confidence_level = "low"
            answer += f"\n\n⚠️ Warning: Possible inconsistencies detected: {', '.join(hallucinations[:2])}"
        
        # Add disclaimer if needed
        answer = add_confidence_disclaimer(answer, confidence_level, confidence_score)
        
        # Log metrics
        response_time = time.time() - start_time
        metrics_tracker.log_query(
            question=request.question,
            endpoint="/query",
            response_time=response_time,
            token_count=len(response.split()),  # Rough token estimate
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            retrieval_score=retrieval_quality[1],
            num_docs=len(docs),
            faithfulness_score=faithfulness_score,
            success=True
        )
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            coverage_limits=limits if limits else None,
            deductibles=deductibles if deductibles else None,
            sources=sources,
            confidence=confidence_level
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        metrics_tracker.log_query(
            question=request.question,
            endpoint="/query",
            response_time=response_time,
            success=False,
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


def _parse_qa_response(response: str, docs: list) -> tuple:
    """Parse the LLM response to extract structured components."""
    answer = response
    citations = []
    limits = {}
    deductibles = {}
    
    # Extract sections from response
    if "ANSWER:" in response:
        parts = response.split("COVERAGE LIMITS:")
        answer = parts[0].replace("ANSWER:", "").strip()
        
        if len(parts) > 1:
            remaining = parts[1]
            if "DEDUCTIBLES:" in remaining:
                limits_part, remaining = remaining.split("DEDUCTIBLES:", 1)
                limits_text = limits_part.strip()
                if limits_text and limits_text.lower() != "not applicable":
                    # Parse limits
                    for line in limits_text.split("\n"):
                        if "$" in line and ":" in line:
                            key, val = line.split(":", 1)
                            limits[key.strip().strip("-").strip()] = val.strip()
                
                if "CITATIONS:" in remaining:
                    ded_part, cit_part = remaining.split("CITATIONS:", 1)
                    ded_text = ded_part.strip()
                    if ded_text and ded_text.lower() != "not applicable":
                        for line in ded_text.split("\n"):
                            if "$" in line and ":" in line:
                                key, val = line.split(":", 1)
                                deductibles[key.strip().strip("-").strip()] = val.strip()
    
    # Build citations from source documents
    for doc in docs:
        section = doc.metadata.get("section", "")
        page = doc.metadata.get("page", "")
        
        # Convert page to string (PDF loaders return int)
        if page is not None and page != "":
            page = str(page)
        else:
            page = None
            
        if section or page:
            citations.append(Citation(
                section=str(section) if section else "Not specified",
                page=page,
                text_snippet=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            ))
    
    return answer, citations, limits, deductibles


# ==============================================================================
# Coverage Check Endpoint
# ==============================================================================

@router.post("/check-coverage", response_model=CoverageCheckResponse)
def check_coverage(request: CoverageCheckRequest) -> CoverageCheckResponse:
    """
    Check if a specific claim scenario is covered under the policies.
    
    Args:
        request: CoverageCheckRequest with scenario description
        
    Returns:
        Structured CoverageCheckResponse with coverage determination
    """
    try:
        # Retrieve relevant documents - need more for coverage checks
        docs = retrieve_with_metadata_filter(
            query=request.scenario + " coverage exclusion",
            k=MAX_K_RESULTS,
            policy_type=request.policy_type,
            policy_number=request.policy_number
        )
        
        if not docs:
            return CoverageCheckResponse(
                scenario=request.scenario,
                is_covered=False,
                coverage_determination="No relevant policy documents found. Cannot determine coverage.",
                sources=[],
                confidence="low"
            )
        
        # Format context
        context = format_docs_with_citations(docs)
        sources = extract_sources_from_docs(docs)
        
        # Create LLM chain
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        prompt = ChatPromptTemplate.from_template(COVERAGE_CHECK_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        # Get analysis
        response = chain.invoke({
            "context": context,
            "scenario": request.scenario
        })
        
        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]
            
            result = json.loads(json_match.strip())
            
            exclusions = [
                ExclusionInfo(
                    section=exc.get("section", "Unknown"),
                    description=exc.get("description", ""),
                    applies=exc.get("applies", False)
                )
                for exc in result.get("exclusions_checked", [])
            ]
            
            return CoverageCheckResponse(
                scenario=request.scenario,
                is_covered=result.get("is_covered", False),
                coverage_determination=result.get("coverage_determination", "Unable to determine"),
                policy_section=result.get("policy_section"),
                coverage_limit=result.get("coverage_limit"),
                deductible=result.get("deductible"),
                exclusions_checked=exclusions,
                conditions=result.get("conditions"),
                sources=sources,
                confidence=result.get("confidence", "medium")
            )
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return CoverageCheckResponse(
                scenario=request.scenario,
                is_covered=False,
                coverage_determination=response,
                sources=sources,
                confidence="low"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coverage check failed: {str(e)}")


# ==============================================================================
# Policy Comparison Endpoint
# ==============================================================================

@router.post("/compare-policies", response_model=CompareResponse)
def compare_policies(request: CompareRequest) -> CompareResponse:
    """
    Compare aspects across multiple policy documents.
    
    Args:
        request: CompareRequest with comparison query
        
    Returns:
        CompareResponse with side-by-side comparison
    """
    try:
        # Retrieve from all or specified policy types
        all_docs = []
        
        if request.policy_types:
            for policy_type in request.policy_types:
                docs = retrieve_with_metadata_filter(
                    query=request.comparison_query,
                    k=4,
                    policy_type=policy_type
                )
                all_docs.extend(docs)
        else:
            all_docs = retrieve_with_metadata_filter(
                query=request.comparison_query,
                k=MAX_K_RESULTS
            )
        
        if not all_docs:
            return CompareResponse(
                comparison_type=request.comparison_query,
                comparison_items=[],
                summary="No relevant policy documents found for comparison.",
                sources=[]
            )
        
        # Format context
        context = format_docs_with_citations(all_docs)
        sources = extract_sources_from_docs(all_docs)
        
        # Create LLM chain
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        prompt = ChatPromptTemplate.from_template(COMPARE_POLICIES_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        # Get comparison
        response = chain.invoke({
            "context": context,
            "comparison_query": request.comparison_query
        })
        
        # Parse JSON response
        try:
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]
            
            result = json.loads(json_match.strip())
            
            items = [
                PolicyComparisonItem(
                    policy_type=item.get("policy_type", "Unknown"),
                    policy_number=item.get("policy_number"),
                    value=item.get("value", "Not found"),
                    section=item.get("section"),
                    notes=item.get("notes")
                )
                for item in result.get("comparison_items", [])
            ]
            
            return CompareResponse(
                comparison_type=result.get("comparison_type", request.comparison_query),
                comparison_items=items,
                summary=result.get("summary", ""),
                sources=sources
            )
            
        except json.JSONDecodeError:
            return CompareResponse(
                comparison_type=request.comparison_query,
                comparison_items=[],
                summary=response,
                sources=sources
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

