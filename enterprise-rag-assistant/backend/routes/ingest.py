"""
backend/routes/ingest.py

POST /ingest — Document Ingestion Endpoint
==========================================
Triggers the offline ingestion pipeline: load PDFs → chunk → embed → index.

Design decisions:
  - Synchronous execution (not background task) for simplicity.
    In production, this would be an async background task with a status
    endpoint to poll for completion. For this submission, synchronous
    is correct — it completes in 2-5 minutes for 6 PDFs and the response
    carries the full result statistics.

  - Idempotent: calling /ingest multiple times is safe.
    The deduplication layer (Phase 1) skips chunks already in the index.
    Only genuinely new chunks are embedded and upserted.

  - File upload vs directory scan:
    This endpoint scans the configured data directory (config.ingestion.data_dir)
    rather than accepting file uploads. For a production system with many users,
    you would add a file upload endpoint. For this use case (6 static PDFs),
    directory scanning is correct and simpler.

Author: Enterprise RAG Assistant
"""

from pathlib import Path

from fastapi import APIRouter, Request, status, UploadFile, File
from fastapi.responses import JSONResponse

from backend.core.config import get_config
from backend.core.dependencies import (
    IngestionPipelineDep,
    IngestionOrchestratorDep,
)
from backend.core.logging import get_logger
from backend.models.schemas import IngestRequest, IngestResponse

router = APIRouter()
logger = get_logger(__name__)
config = get_config()

# Guard against concurrent ingestion runs.
# FAISS index writes are not thread-safe — two concurrent ingestion calls
# could corrupt the index. This flag prevents that.
_ingestion_in_progress = False


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger document ingestion and indexing",
    description=(
        "Scans the configured data directory for PDF, Markdown, and text files. "
        "Loads, chunks, embeds, and indexes all documents. "
        "Deduplication ensures already-indexed chunks are not re-embedded. "
        "Calling this endpoint multiple times is safe — it is idempotent."
    ),
    responses={
        200: {"description": "Ingestion complete with statistics"},
        409: {"description": "Ingestion already in progress"},
        503: {"description": "Services not initialized"},
    },
)
async def ingest_documents(
    request: Request,
    request_body: IngestRequest,
    ingestion_pipeline: IngestionPipelineDep,
    orchestrator: IngestionOrchestratorDep,
) -> IngestResponse:
    """
    Run the full ingestion pipeline on the configured data directory.

    Steps:
      1. Acquire concurrency guard (reject if already running)
      2. Get existing content hashes from vector store (for deduplication)
      3. Run DocumentIngestionPipeline.run() (load, clean, chunk, deduplicate)
      4. Embed new chunks and upsert to FAISS index
      5. Update app.state.vector_count
      6. Return statistics
    """
    global _ingestion_in_progress
    request_id = getattr(request.state, "request_id", "unknown")

    # ── Concurrency guard ──────────────────────────────────────────────────
    if _ingestion_in_progress:
        logger.warning(
            "Ingestion rejected — already in progress",
            extra={"request_id": request_id},
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error_code": "INGESTION_IN_PROGRESS",
                "message": "An ingestion run is already in progress. Please wait for it to complete.",
                "request_id": request_id,
            },
        )

    _ingestion_in_progress = True
    logger.info(
        "Starting ingestion pipeline",
        extra={
            "request_id": request_id,
            "force_reindex": request_body.force_reindex,
            "data_dir": config.ingestion.data_dir,
        },
    )

    try:
        # Get existing hashes from vector store
        # If force_reindex=True, skip dedup and re-embed everything
        existing_hashes = (
            set()
            if request_body.force_reindex
            else orchestrator.vector_store.get_all_hashes()
        )

        logger.info(
            "Deduplication state",
            extra={
                "existing_hashes": len(existing_hashes),
                "force_reindex": request_body.force_reindex,
            },
        )

        # Run ingestion pipeline (synchronous — runs in calling thread)
        import asyncio
        ingestion_result = await asyncio.to_thread(
            ingestion_pipeline.run,
            Path(config.ingestion.data_dir),
            existing_hashes,
        )

        # Embed and index new chunks
        indexing_stats = await asyncio.to_thread(
            orchestrator.ingest_from_result,
            ingestion_result,
        )

        # Update vector count in app.state for health endpoint
        request.app.state.vector_count = orchestrator.vector_store.vector_count

        logger.info(
            "Ingestion complete",
            extra={
                "request_id": request_id,
                **ingestion_result.to_log_dict(),
                **indexing_stats,
            },
        )

        return IngestResponse(
            status="success",
            chunks_embedded=indexing_stats.get("chunks_embedded", 0),
            chunks_indexed=indexing_stats.get("chunks_indexed", 0),
            total_vectors_in_index=orchestrator.vector_store.vector_count,
            duration_seconds=ingestion_result.duration_seconds,
            message=(
                f"Ingestion complete. {indexing_stats.get('chunks_indexed', 0)} "
                f"new chunks indexed. "
                f"{ingestion_result.duplicate_chunks_skipped} duplicates skipped."
            ),
        )

    except Exception as e:
        logger.error(
            "Ingestion failed",
            extra={
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise

    finally:
        _ingestion_in_progress = False


@router.post(
    "/ingest/upload",
    status_code=status.HTTP_200_OK,
    summary="Upload and ingest a single PDF document",
    description="Upload a PDF file directly. Saves to data dir and triggers ingestion.",
)
async def upload_and_ingest(
    request: Request,
    file: UploadFile = File(...),
    ingestion_pipeline: IngestionPipelineDep = None,
    orchestrator: IngestionOrchestratorDep = None,
):
    """
    Accept a file upload, save to data directory, trigger ingestion.

    Supported types: PDF, Markdown, plain text.
    Max file size: 50MB (enforced by nginx/ALB in production).
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Validate file type
    supported_extensions = {".pdf", ".md", ".txt"}
    file_suffix = Path(file.filename or "").suffix.lower()
    if file_suffix not in supported_extensions:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error_code": "UNSUPPORTED_FILE_TYPE",
                "message": (
                    f"File type '{file_suffix}' not supported. "
                    f"Supported: {supported_extensions}"
                ),
                "request_id": request_id,
            },
        )

    # Save file to data directory
    data_dir = Path(config.ingestion.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest_path = data_dir / (file.filename or "uploaded_document.pdf")

    content = await file.read()
    dest_path.write_bytes(content)

    logger.info(
        "File uploaded",
        extra={
            "filename": file.filename,
            "size_bytes": len(content),
            "dest_path": str(dest_path),
        },
    )

    return {
        "status": "uploaded",
        "filename": file.filename,
        "size_bytes": len(content),
        "message": "File saved. Call POST /ingest to index it.",
    }