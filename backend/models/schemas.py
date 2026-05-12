"""
backend/models/schemas.py

Pydantic Request and Response Schemas
======================================
All API input/output types live here.
Pydantic provides:
  - Automatic validation with field-level error messages
  - JSON serialization/deserialization
  - OpenAPI schema generation (FastAPI uses this for /docs)
  - Type safety throughout the codebase

Author: Enterprise RAG Assistant
"""

from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator

from backend.services.llm_manager import ModelType


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    """
    Three-tier confidence signal for every RAG response.

    HIGH:   Context directly and completely answers the question.
    MEDIUM: Context partially answers, or inference between passages required.
    LOW:    Context is tangential, grounding check failed, or refusal path.

    str mixin ensures JSON serialization produces "HIGH" not "ConfidenceLevel.HIGH".
    """
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Source Document
# ─────────────────────────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    """
    A single source chunk that contributed to the generated answer.

    This is the evidence trail — the user can expand each source card
    in the Streamlit UI to see the exact passage the answer was based on.
    """
    pillar: str = Field(
        description="AWS Well-Architected pillar name",
        example="Reliability",
    )
    source_file: str = Field(
        description="Source PDF filename",
        example="reliability.pdf",
    )
    page_number: int = Field(
        description="1-indexed page number in the source PDF",
        example=14,
    )
    section: str = Field(
        default="",
        description="Section heading if detected",
        example="Design Principles",
    )
    excerpt: str = Field(
        description="First 200 characters of the retrieved chunk",
        example="Design for failure is a core principle...",
    )
    relevance_score: float = Field(
        description="Cosine similarity score from retrieval (0.0 to 1.0)",
        example=0.847,
    )
    chunk_index: int = Field(
        default=0,
        description="Sequential index of this chunk within its source page",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Request Schema
# ─────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """
    Request body for POST /generate.

    Validation rules:
      - query: required, 1-1000 characters
      - use_fine_tuned: optional, default True
      - top_k: optional, 1-10, default from config
      - filter_pillar: optional, must match known pillar names if provided
    """
    query: str = Field(
        description="Natural language question about AWS architecture",
        min_length=1,
        max_length=1000,
        example="How should I design for failure in a multi-AZ deployment?",
    )
    use_fine_tuned: bool = Field(
        default=True,
        description=(
            "If True, use the fine-tuned Phi-3-mini model. "
            "If False or model unavailable, use OpenAI GPT-4o-mini fallback."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve. Range: 1-10.",
    )
    filter_pillar: Optional[str] = Field(
        default=None,
        description=(
            "Restrict retrieval to a specific pillar. "
            "Options: 'Operational Excellence', 'Security', 'Reliability', "
            "'Performance Efficiency', 'Cost Optimization', 'Sustainability'. "
            "None searches all pillars."
        ),
        example="Reliability",
    )

    @field_validator("filter_pillar")
    @classmethod
    def validate_pillar(cls, v: Optional[str]) -> Optional[str]:
        """Validate that the pillar name is one of the six known pillars."""
        if v is None:
            return v
        valid_pillars = {
            "Operational Excellence",
            "Security",
            "Reliability",
            "Performance Efficiency",
            "Cost Optimization",
            "Sustainability",
        }
        if v not in valid_pillars:
            raise ValueError(
                f"Invalid pillar '{v}'. "
                f"Must be one of: {sorted(valid_pillars)}"
            )
        return v


class IngestRequest(BaseModel):
    """Request body for POST /ingest (trigger re-ingestion)."""
    force_reindex: bool = Field(
        default=False,
        description=(
            "If True, re-embed all chunks even if they already exist in the index. "
            "Use when embedding model has changed. Expensive — avoid in production."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class RAGResponse(BaseModel):
    """
    Full response object from POST /generate.

    This schema carries more than just the answer text — it is a
    production-grade response object that gives:
      - The evaluation harness: confidence, grounding_score, model_used
      - The ops team: latency breakdowns, token counts, cache hits
      - The end user: answer, sources, confidence level
      - Debugging: prompt_version, grounding_flag

    Every field is intentional. Removing any field would break either
    the evaluation pipeline, the Streamlit UI, or the observability layer.
    """

    # Core answer
    answer: str = Field(
        description="Generated answer text, grounded in retrieved context",
    )
    sources: list[SourceDocument] = Field(
        description="Retrieved document chunks that informed the answer",
    )

    # Quality signals
    confidence: ConfidenceLevel = Field(
        description=(
            "Composite confidence: model self-assessment + grounding check. "
            "LOW if grounding check fails, regardless of model self-assessment."
        ),
    )
    confidence_reason: str = Field(
        description="One-sentence explanation of the confidence rating",
    )
    grounding_flag: bool = Field(
        description=(
            "True if the semantic grounding check detected potential drift "
            "between the answer and retrieved context. Indicates the user "
            "should verify the answer against source documents."
        ),
    )
    grounding_score: float = Field(
        description=(
            "Max cosine similarity between answer embedding and chunk embeddings. "
            "Range: 0.0 to 1.0. Below 0.50 triggers grounding_flag=True."
        ),
    )

    # Model attribution
    model_used: ModelType = Field(
        description="Which model generated this answer (fine-tuned or fallback)",
    )

    # Latency breakdown
    retrieval_latency_ms: int = Field(
        description="Time to embed query and retrieve chunks, in milliseconds",
    )
    generation_latency_ms: int = Field(
        description="Time for LLM to generate the answer, in milliseconds",
    )
    total_latency_ms: int = Field(
        description="Total end-to-end request latency, in milliseconds",
    )

    # Cost signals
    tokens_used: Optional[int] = Field(
        default=None,
        description=(
            "Total tokens consumed by the generation model. "
            "None for fine-tuned model (approximate counting not enabled). "
            "Exact for OpenAI fallback (from API response metadata)."
        ),
    )

    # Cache and routing
    cache_hit: bool = Field(
        description="True if this response was served from the query cache",
    )
    is_refusal: bool = Field(
        description="True if the system refused to answer (no context found)",
    )

    # Debugging
    prompt_version: str = Field(
        description="Version identifier of the prompt template used",
        example="v1",
    )

    @classmethod
    def from_guardrail_block(cls, reason: str) -> "RAGResponse":
        """
        Factory method for responses blocked by input guardrails.
        Returns a structured response rather than raising an exception.
        """
        return cls(
            answer=f"Request blocked: {reason}",
            sources=[],
            confidence=ConfidenceLevel.LOW,
            confidence_reason="Request blocked by input safety guardrail.",
            grounding_flag=False,
            grounding_score=0.0,
            model_used=ModelType.UNAVAILABLE,
            retrieval_latency_ms=0,
            generation_latency_ms=0,
            total_latency_ms=0,
            tokens_used=None,
            cache_hit=False,
            is_refusal=True,
            prompt_version=PROMPT_VERSION,
        )


class IngestResponse(BaseModel):
    """Response from POST /ingest."""
    status: str
    chunks_embedded: int
    chunks_indexed: int
    total_vectors_in_index: int
    duration_seconds: float
    message: str


class HealthResponse(BaseModel):
    """Response from GET /health."""
    status: str
    vector_store_ready: bool
    primary_model_available: bool
    fallback_model_available: bool
    vector_count: int
    version: str


class RefusalResponse(BaseModel):
    """
    Structured refusal when context is insufficient.
    Also returned for out-of-scope questions.
    """
    answer: str
    reason: str = "no_relevant_context"
    sources: list = []
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    is_refusal: bool = True

class ErrorResponse(BaseModel):
    """
    Structured error response returned for all non-2xx responses.

    Every error carries a request_id so the client can correlate
    their error with a specific log entry on the server.

    error_code: Machine-readable string for programmatic error handling.
                Client code should switch on error_code, not message text.
    message:    Human-readable description. May change between versions.
    request_id: UUID assigned to this request. Present in server logs.
    details:    Optional additional context (e.g. field-level validation errors).
    """
    error_code: str
    message: str
    request_id: str = "unknown"
    details: Optional[Any] = None