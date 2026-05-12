"""
backend/routes/generate.py

POST /generate — RAG Query Endpoint
=====================================
The primary endpoint. Accepts a natural language query, runs the full
RAG pipeline, and returns a structured response with answer, sources,
confidence, and latency metrics.

Timeout handling:
  asyncio.timeout() wraps the pipeline.run() call.
  If generation exceeds config.generation.timeout_seconds, TimeoutError
  is raised and caught by the global exception handler in main.py,
  which returns a 504 response with Retry-After header.

  Why asyncio.timeout() and not a threading timeout?
    FastAPI runs route handlers in an async event loop.
    asyncio.timeout() is cooperative — it signals the coroutine to
    cancel at the next await point. This is the correct mechanism for
    async code. A threading timeout would not interrupt an async coroutine.

    Note: if the LLM call is blocking (synchronous torch inference),
    it runs in a thread pool via run_in_executor(), which asyncio.timeout()
    can interrupt at the executor boundary.

Author: Enterprise RAG Assistant
"""

import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from backend.core.config import get_config
from backend.core.dependencies import RAGPipelineDep
from backend.core.logging import get_logger
from backend.core.metrics import (
    GENERATION_LATENCY,
    REQUESTS_TOTAL,
    RETRIEVAL_LATENCY,
    TOKENS_TOTAL,
    CACHE_HITS,
    FALLBACK_ACTIVATIONS,
    HALLUCINATION_FLAGS,
)
from backend.models.schemas import (
    GenerateRequest,
    RAGResponse,
    ErrorResponse,
)
from backend.services.llm_manager import ModelType

router = APIRouter()
logger = get_logger(__name__)
config = get_config()


@router.post(
    "/generate",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a grounded answer from AWS documentation",
    description=(
        "Accepts a natural language query about the AWS Well-Architected Framework. "
        "Retrieves relevant context from the indexed PDF corpus, generates a grounded "
        "answer using the fine-tuned model (or OpenAI fallback), and returns the "
        "answer with source citations, confidence rating, and latency metrics."
    ),
    responses={
        200: {"description": "Successful response with answer and sources"},
        422: {"description": "Input validation failed"},
        503: {"description": "Vector store not ready — run POST /ingest first"},
        504: {"description": "Generation timeout — model took too long"},
    },
)
async def generate_answer(
    request_body: GenerateRequest,
    request: Request,
    pipeline: RAGPipelineDep,
) -> RAGResponse:
    """
    Full RAG pipeline: query → retrieve → generate → validate → respond.

    The pipeline.run() call is wrapped in asyncio.timeout() to enforce
    the generation timeout from config. If the model is slow (CPU inference
    of the fine-tuned model), the timeout fires and a 504 is returned.

    Cache check:
      The L2 response cache is checked BEFORE calling the pipeline.
      A cache hit returns the stored response immediately, bypassing
      retrieval and generation entirely (<50ms vs 2-7s).
    """
    request_id = getattr(request.state, "request_id", "unknown")
    request_start = time.time()

    logger.info(
        "Generate request received",
        extra={
            "request_id": request_id,
            "query_length": len(request_body.query),
            "use_fine_tuned": request_body.use_fine_tuned,
            "top_k": request_body.top_k,
            "filter_pillar": request_body.filter_pillar,
        },
    )

    # ── L2 Cache Check ─────────────────────────────────────────────────────
    # Check the query response cache before running the full pipeline.
    # Cache is keyed by hash(normalized_query + top_k + filter_pillar).
    # TTL: 1 hour (configured in config.yaml).
    cached_response = _check_cache(request, request_body)
    if cached_response is not None:
        CACHE_HITS.inc()
        REQUESTS_TOTAL.labels(
            status="success", model="cache", endpoint="generate"
        ).inc()
        logger.info(
            "Cache hit — returning cached response",
            extra={
                "request_id": request_id,
                "total_ms": round((time.time() - request_start) * 1000),
            },
        )
        # Update the cached response's cache_hit flag
        cached_response.cache_hit = True
        return cached_response

    # ── Pipeline Execution with Timeout ────────────────────────────────────
    try:
        async with asyncio.timeout(config.generation.timeout_seconds):
            # Run the synchronous RAG pipeline in a thread pool.
            # This prevents blocking the async event loop during:
            #   - CPU-bound embedding computation
            #   - Synchronous torch inference (fine-tuned model)
            #   - Synchronous OpenAI API call
            #
            # asyncio.to_thread() runs the function in the default thread pool
            # executor and awaits its completion. The timeout above will
            # cancel the thread at the next cooperative point if exceeded.
            response: RAGResponse = await asyncio.to_thread(
                pipeline.run,
                query=request_body.query,
                use_fine_tuned=request_body.use_fine_tuned,
                top_k=request_body.top_k,
                filter_pillar=request_body.filter_pillar,
            )

    except asyncio.TimeoutError:
        # Let the global exception handler in main.py handle this
        # It returns a 504 with Retry-After header
        logger.error(
            "Generation timeout",
            extra={
                "request_id": request_id,
                "timeout_seconds": config.generation.timeout_seconds,
                "query_preview": request_body.query[:50],
            },
        )
        raise TimeoutError(
            f"Generation exceeded {config.generation.timeout_seconds}s timeout"
        )

    # ── Update Metrics ─────────────────────────────────────────────────────
    model_label = (
        "fine_tuned" if response.model_used == ModelType.FINE_TUNED
        else "openai_fallback"
    )

    REQUESTS_TOTAL.labels(
        status="success", model=model_label, endpoint="generate"
    ).inc()
    RETRIEVAL_LATENCY.observe(response.retrieval_latency_ms / 1000)
    GENERATION_LATENCY.observe(response.generation_latency_ms / 1000)

    if response.tokens_used:
        TOKENS_TOTAL.labels(model=model_label, direction="output").inc(
            response.tokens_used
        )

    if response.model_used == ModelType.OPENAI_FALLBACK:
        FALLBACK_ACTIVATIONS.inc()

    if response.grounding_flag:
        HALLUCINATION_FLAGS.inc()

    # ── Cache Write ────────────────────────────────────────────────────────
    # Only cache successful, non-refusal responses.
    # Refusals may change as the document corpus is updated.
    if not response.is_refusal and not response.grounding_flag:
        _write_cache(request, request_body, response)

    total_ms = round((time.time() - request_start) * 1000)
    logger.info(
        "Generate request complete",
        extra={
            "request_id": request_id,
            "total_ms": total_ms,
            "model_used": response.model_used,
            "confidence": response.confidence,
            "grounding_flag": response.grounding_flag,
            "is_refusal": response.is_refusal,
            "sources_count": len(response.sources),
        },
    )

    return response


def _check_cache(request: Request, request_body: GenerateRequest) -> Optional[RAGResponse]:
    """
    Check the disk-based response cache for a matching query.

    Cache key: SHA-256 of normalized(query + top_k + filter_pillar).
    Normalization: lowercase, strip whitespace, consistent concatenation.
    TTL: config.cache.ttl_seconds (default 3600 = 1 hour).

    Returns cached RAGResponse if found and not expired, else None.
    """
    if not config.cache.enabled:
        return None

    try:
        import diskcache
        import hashlib

        cache_key = hashlib.sha256(
            f"{request_body.query.strip().lower()}|"
            f"{request_body.top_k}|"
            f"{request_body.filter_pillar or ''}".encode()
        ).hexdigest()

        cache_dir = config.cache.cache_dir
        cache = diskcache.Cache(cache_dir)

        cached = cache.get(cache_key)
        cache.close()

        if cached is not None:
            return RAGResponse.model_validate(cached)

    except Exception as e:
        # Cache errors should never fail the request — log and continue
        logger.warning("Cache read error", extra={"error": str(e)})

    return None


def _write_cache(
    request: Request,
    request_body: GenerateRequest,
    response: RAGResponse,
) -> None:
    """
    Write a successful response to the disk cache.

    Uses diskcache with TTL enforcement.
    Serializes response to dict (JSON-compatible) for storage.
    """
    if not config.cache.enabled:
        return

    try:
        import diskcache
        import hashlib

        cache_key = hashlib.sha256(
            f"{request_body.query.strip().lower()}|"
            f"{request_body.top_k}|"
            f"{request_body.filter_pillar or ''}".encode()
        ).hexdigest()

        cache = diskcache.Cache(config.cache.cache_dir)
        cache.set(
            cache_key,
            response.model_dump(),
            expire=config.cache.ttl_seconds,
        )
        cache.close()

    except Exception as e:
        logger.warning("Cache write error", extra={"error": str(e)})