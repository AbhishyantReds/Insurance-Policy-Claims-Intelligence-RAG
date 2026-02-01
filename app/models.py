"""
Pydantic models for Insurance Policy RAG API.
Defines request/response schemas for all endpoints.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ==============================================================================
# Ingestion Models
# ==============================================================================

class IngestResponse(BaseModel):
    """Response model for document ingestion endpoint."""
    message: str
    chunks_count: int
    documents_processed: int
    policy_types_found: List[str] = Field(default_factory=list)


# ==============================================================================
# Query Models
# ==============================================================================

class QueryRequest(BaseModel):
    """Request model for insurance policy query endpoint."""
    question: str = Field(..., description="The question about insurance coverage")
    policy_type: Optional[str] = Field(None, description="Filter by policy type (homeowners, auto, commercial)")
    policy_number: Optional[str] = Field(None, description="Filter by specific policy number")


class Citation(BaseModel):
    """A citation to a specific policy section."""
    section: str = Field(..., description="Policy section reference (e.g., 'Section 3.1')")
    page: Optional[str] = Field(None, description="Page number if available")
    text_snippet: str = Field(..., description="Relevant text from the policy")


class QueryResponse(BaseModel):
    """Response model for insurance policy query endpoint."""
    answer: str = Field(..., description="The answer to the question")
    citations: List[Citation] = Field(default_factory=list, description="Citations from policy documents")
    coverage_limits: Optional[Dict[str, str]] = Field(None, description="Relevant coverage limits mentioned")
    deductibles: Optional[Dict[str, str]] = Field(None, description="Relevant deductibles mentioned")
    sources: List[str] = Field(default_factory=list, description="Source document filenames")
    confidence: str = Field("high", description="Confidence level: high, medium, low")


# ==============================================================================
# Coverage Check Models
# ==============================================================================

class CoverageCheckRequest(BaseModel):
    """Request model for coverage check endpoint."""
    scenario: str = Field(..., description="Description of the claim scenario (e.g., 'basement flood from sewer backup')")
    policy_type: Optional[str] = Field(None, description="Filter by policy type")
    policy_number: Optional[str] = Field(None, description="Filter by specific policy number")


class ExclusionInfo(BaseModel):
    """Information about an exclusion that was checked."""
    section: str = Field(..., description="Exclusion section reference")
    description: str = Field(..., description="Brief description of the exclusion")
    applies: bool = Field(..., description="Whether this exclusion applies to the scenario")


class CoverageCheckResponse(BaseModel):
    """Response model for structured coverage check."""
    scenario: str = Field(..., description="The claim scenario that was analyzed")
    is_covered: bool = Field(..., description="Whether the scenario is covered")
    coverage_determination: str = Field(..., description="Detailed explanation of coverage decision")
    policy_section: Optional[str] = Field(None, description="Relevant policy section for coverage")
    coverage_limit: Optional[str] = Field(None, description="Applicable coverage limit")
    deductible: Optional[str] = Field(None, description="Applicable deductible")
    exclusions_checked: List[ExclusionInfo] = Field(default_factory=list, description="Exclusions that were evaluated")
    conditions: Optional[str] = Field(None, description="Any conditions or requirements for coverage")
    sources: List[str] = Field(default_factory=list, description="Source policy documents")
    confidence: str = Field("high", description="Confidence level: high, medium, low")


# ==============================================================================
# Policy Comparison Models
# ==============================================================================

class CompareRequest(BaseModel):
    """Request model for policy comparison endpoint."""
    comparison_query: str = Field(..., description="What to compare (e.g., 'deductibles', 'liability limits')")
    policy_types: Optional[List[str]] = Field(None, description="Policy types to compare (defaults to all)")


class PolicyComparisonItem(BaseModel):
    """A single item in a policy comparison."""
    policy_type: str = Field(..., description="Type of policy")
    policy_number: Optional[str] = Field(None, description="Policy number")
    value: str = Field(..., description="The value being compared")
    section: Optional[str] = Field(None, description="Policy section reference")
    notes: Optional[str] = Field(None, description="Additional context or notes")


class CompareResponse(BaseModel):
    """Response model for policy comparison endpoint."""
    comparison_type: str = Field(..., description="What was compared")
    comparison_items: List[PolicyComparisonItem] = Field(default_factory=list)
    summary: str = Field(..., description="Natural language summary of the comparison")
    sources: List[str] = Field(default_factory=list, description="Source policy documents")


# ==============================================================================
# Document Metadata Model
# ==============================================================================

class PolicyMetadata(BaseModel):
    """Metadata extracted from a policy document."""
    policy_type: str = Field(..., description="Type of policy")
    policy_number: Optional[str] = Field(None, description="Policy number")
    policyholder: Optional[str] = Field(None, description="Name of policyholder")
    effective_date: Optional[str] = Field(None, description="Policy effective date")
    state: Optional[str] = Field(None, description="State where policy applies")
    source_file: str = Field(..., description="Source filename")
    section: Optional[str] = Field(None, description="Section reference if applicable")
    page: Optional[str] = Field(None, description="Page number if applicable")


# ==============================================================================
# Error Models
# ==============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
