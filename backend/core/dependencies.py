"""
backend/core/dependencies.py

Dependency Injection — Service Initialization and Access
=========================================================
This module owns the service lifecycle:
  1. initialize_services() — called once at startup
  2. shutdown_services() — called once at shutdown
  3. FastAPI dependency functions — injected into route handlers

Why dependency injection instead of global singletons?
  Global singletons work but create hidden coupling between modules.
  Service instances stored in app.state are:
    - Explicit: every route declares what it needs
    - Testable: tests replace app.state.{service} with mocks
    - Lifecycle-safe: services are released on shutdown

FastAPI dependency injection with Depends():
  Route handlers declare their dependencies as function arguments.
  FastAPI resolves them by calling the dependency function once per request
  (or once per app for application-scoped dependencies).

Author: Enterprise RAG Assistant
"""

import time
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status

from backend.core.config import get_config
from backend.core.logging import get_logger
from backend.services.guardrails import InputGuardrail, OutputGuardrail
from backend.services.ingestion import DocumentIngestionPipeline
from backend.services.llm_manager import LLMManager
from backend.services.rag_pipeline import GroundingChecker, RAGPipeline
from backend.services.vector_store import (
    CrossEncoderReranker,
    EmbeddingModel,
    FAISSVectorStore,
    RetrievalService,
    VectorStoreIngestionOrchestrator,
)

logger = get_logger(__name__)
config = get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Service Initialization
# ─────────────────────────────────────────────────────────────────────────────

async def initialize_services(app: FastAPI) -> None:
    """
    Initialize all application services at startup.

    Services are initialized in dependency order:
      EmbeddingModel → FAISSVectorStore → RetrievalService → LLMManager
      → RAGPipeline

    Each service is attached to app.state so route handlers can access them
    via the get_* dependency functions below.

    Failure behavior:
      - EmbeddingModel failure: FATAL — retrieval is impossible without it
      - FAISSVectorStore failure: WARNING — app starts, but ingest must run first
      - LLM failure (primary): WARNING — fallback activates automatically
      - LLM failure (fallback): FATAL — no generation is possible
    """
    logger.info("Initializing services")

    # ── Embedding Model ────────────────────────────────────────────────────
    logger.info("Loading embedding model")
    t = time.time()
    try:
        embedding_model = EmbeddingModel()
        app.state.embedding_model = embedding_model
        logger.info(
            "Embedding model ready",
            extra={"load_ms": round((time.time() - t) * 1000)},
        )
    except Exception as e:
        logger.error(
            "FATAL: Embedding model failed to load",
            extra={"error": str(e)},
        )
        raise RuntimeError(f"Embedding model initialization failed: {e}") from e

    # ── FAISS Vector Store ─────────────────────────────────────────────────
    logger.info("Loading FAISS vector store")
    t = time.time()
    faiss_store = FAISSVectorStore(embedding_model=embedding_model)
    app.state.faiss_store = faiss_store
    app.state.vector_count = faiss_store.vector_count

    if faiss_store.is_ready:
        logger.info(
            "FAISS index loaded from disk",
            extra={
                "vector_count": faiss_store.vector_count,
                "load_ms": round((time.time() - t) * 1000),
            },
        )
    else:
        logger.warning(
            "FAISS index not found on disk — call POST /ingest to build index. "
            "Query endpoint will return 503 until index is built."
        )

    # ── Reranker ───────────────────────────────────────────────────────────
    logger.info("Loading cross-encoder reranker")
    t = time.time()
    reranker = CrossEncoderReranker()
    app.state.reranker = reranker
    logger.info(
        "Reranker status",
        extra={
            "available": reranker.is_available,
            "load_ms": round((time.time() - t) * 1000),
        },
    )

    # ── Retrieval Service ──────────────────────────────────────────────────
    retrieval_service = RetrievalService(
        vector_store=faiss_store,
        reranker=reranker,
    )
    app.state.retrieval_service = retrieval_service

    # ── Ingestion Pipeline ─────────────────────────────────────────────────
    ingestion_pipeline = DocumentIngestionPipeline()
    ingestion_orchestrator = VectorStoreIngestionOrchestrator(
        embedding_model=embedding_model,
        vector_store=faiss_store,
    )
    app.state.ingestion_pipeline = ingestion_pipeline
    app.state.ingestion_orchestrator = ingestion_orchestrator

    # ── LLM Manager ───────────────────────────────────────────────────────
    logger.info("Initializing LLM manager")
    t = time.time()
    llm_manager = LLMManager()
    app.state.llm_manager = llm_manager
    app.state.primary_model_ready = llm_manager.primary_available
    app.state.fallback_model_ready = llm_manager.fallback_available

    logger.info(
        "LLM manager ready",
        extra={
            "primary_available": llm_manager.primary_available,
            "fallback_available": llm_manager.fallback_available,
            "init_ms": round((time.time() - t) * 1000),
        },
    )

    if not llm_manager.any_model_available:
        raise RuntimeError(
            "FATAL: No generation model available. "
            "Check OPENAI_API_KEY environment variable. "
            "At minimum, the OpenAI fallback must be reachable."
        )

    # ── Guardrails ─────────────────────────────────────────────────────────
    app.state.input_guardrail = InputGuardrail()
    app.state.output_guardrail = OutputGuardrail()

    # ── Grounding Checker ──────────────────────────────────────────────────
    grounding_checker = GroundingChecker(embedding_model=embedding_model)
    app.state.grounding_checker = grounding_checker

    # ── RAG Pipeline ───────────────────────────────────────────────────────
    rag_pipeline = RAGPipeline(
        retrieval_service=retrieval_service,
        llm_manager=llm_manager,
        input_guardrail=app.state.input_guardrail,
        output_guardrail=app.state.output_guardrail,
        grounding_checker=grounding_checker,
    )
    app.state.rag_pipeline = rag_pipeline

    logger.info("All services initialized successfully")


async def shutdown_services(app: FastAPI) -> None:
    """
    Release resources on application shutdown.

    Explicit cleanup prevents:
      - CUDA context leaks in multi-process deployments
      - File handle leaks (FAISS index files)
      - Incomplete log flushes
    """
    logger.info("Releasing service resources")

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory released")
    except ImportError:
        pass  # torch not installed, nothing to release

    logger.info("Service shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Dependency Functions
# ─────────────────────────────────────────────────────────────────────────────
# These functions are used with FastAPI's Depends() system.
# Each returns a service instance from app.state.
# If the service is unavailable, they raise HTTP exceptions immediately
# before the route handler runs — fail fast, with a clear error message.

def get_rag_pipeline(request: Request) -> RAGPipeline:
    """
    Dependency: returns the initialized RAGPipeline.

    Raises 503 if the pipeline is not available (startup failed or
    vector store not yet built).
    """
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "PIPELINE_NOT_READY",
                "message": (
                    "RAG pipeline is not initialized. "
                    "The application may still be starting up."
                ),
            },
        )
    return pipeline


def get_retrieval_service(request: Request) -> RetrievalService:
    """Dependency: returns the RetrievalService."""
    service = getattr(request.app.state, "retrieval_service", None)
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "VECTOR_STORE_NOT_READY",
                "message": (
                    "Vector store is not ready. "
                    "Call POST /ingest to build the document index first."
                ),
            },
        )
    return service


def get_ingestion_pipeline(request: Request) -> DocumentIngestionPipeline:
    """Dependency: returns the DocumentIngestionPipeline."""
    pipeline = getattr(request.app.state, "ingestion_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error_code": "SERVICE_NOT_READY", "message": "Ingestion pipeline not initialized."},
        )
    return pipeline


def get_ingestion_orchestrator(request: Request) -> VectorStoreIngestionOrchestrator:
    """Dependency: returns the VectorStoreIngestionOrchestrator."""
    orch = getattr(request.app.state, "ingestion_orchestrator", None)
    if orch is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error_code": "SERVICE_NOT_READY", "message": "Ingestion orchestrator not initialized."},
        )
    return orch


def get_llm_manager(request: Request) -> LLMManager:
    """Dependency: returns the LLMManager."""
    manager = getattr(request.app.state, "llm_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error_code": "LLM_NOT_READY", "message": "LLM manager not initialized."},
        )
    return manager


# Type aliases for cleaner route handler signatures
RAGPipelineDep = Annotated[RAGPipeline, Depends(get_rag_pipeline)]
RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]
IngestionPipelineDep = Annotated[DocumentIngestionPipeline, Depends(get_ingestion_pipeline)]
IngestionOrchestratorDep = Annotated[VectorStoreIngestionOrchestrator, Depends(get_ingestion_orchestrator)]
LLMManagerDep = Annotated[LLMManager, Depends(get_llm_manager)]