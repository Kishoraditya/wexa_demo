"""
backend/main.py

FastAPI Application Factory
============================
This module is the single entry point for the FastAPI application.
It is responsible for:
  1. Creating the FastAPI app instance with metadata
  2. Registering all routers (ingest, generate, health, metrics)
  3. Registering global exception handlers
  4. Running startup and shutdown lifecycle events
  5. Wiring together all service dependencies

Application lifecycle:
  startup  → load config → initialize logging → initialize services
           → load embedding model → load FAISS index → load LLM
           → register Prometheus metrics → app ready
  shutdown → flush logs → close connections → release GPU memory

Dependency injection strategy:
  All shared service instances (RAGPipeline, RetrievalService, LLMManager)
  are created once at startup and stored in app.state.
  Route handlers access them via FastAPI's Request object or
  dependency injection functions defined in core/dependencies.py.
  This avoids global variables and makes testing straightforward —
  replace app.state.{service} with a mock before running tests.

Author: Enterprise RAG Assistant
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import get_config
from backend.core.logging import get_logger, setup_logging
from backend.core.metrics import setup_metrics
from backend.core.dependencies import initialize_services, shutdown_services
from backend.routes import generate, ingest, health
from backend.models.schemas import ErrorResponse

# ── Bootstrap logging before anything else ────────────────────────────────
# Logging must be initialized before any module that uses get_logger()
# is imported or instantiated. setup_logging() configures loguru.
config = get_config()
setup_logging(
    log_level=config.app.log_level,
    log_file="logs/app.log",
)
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Application Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Using the lifespan context manager (FastAPI >= 0.93) rather than
    @app.on_event("startup") / @app.on_event("shutdown") because:
      - Single function manages both lifecycle phases
      - Resources initialized before yield are guaranteed to be cleaned
        up after yield even if the app crashes
      - Easier to test — lifespan can be injected in test client

    Startup sequence and rationale:
      1. Config validation — fail fast if config is malformed
      2. Service initialization — embedding model, FAISS index, LLM
         These are heavy operations (model loading). Doing them at startup
         means the first request is not penalized with a cold start.
      3. Metrics setup — Prometheus counters initialized to zero
         (not initialized = metrics won't appear until first event)

    Shutdown sequence:
      - GPU memory released explicitly to prevent CUDA context leaks
        in multi-worker deployments
      - Log final stats for post-mortem debugging
    """
    # ── STARTUP ───────────────────────────────────────────────────────────
    startup_start = time.time()
    logger.info(
        "Application starting",
        extra={
            "app_name": config.app.name,
            "version": config.app.version,
            "environment": config.app.environment,
        },
    )

    try:
        # Initialize all services and attach to app.state
        await initialize_services(app)

        startup_ms = round((time.time() - startup_start) * 1000)
        logger.info(
            "Application startup complete",
            extra={
                "startup_ms": startup_ms,
                "vector_count": getattr(
                    app.state, "vector_count", 0
                ),
                "primary_model_ready": getattr(
                    app.state, "primary_model_ready", False
                ),
                "fallback_model_ready": getattr(
                    app.state, "fallback_model_ready", False
                ),
            },
        )

    except Exception as e:
        logger.error(
            "FATAL: Application startup failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        # Re-raise to prevent the app from starting in a broken state.
        # A broken startup is preferable to serving requests with
        # uninitialized dependencies.
        raise

    yield  # ← Application runs here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────
    logger.info("Application shutting down")
    await shutdown_services(app)
    logger.info("Application shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# App Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Using a factory function rather than a module-level app instance
    enables:
      - Multiple app instances in tests (each test gets a fresh app)
      - Easy configuration injection for different environments
      - Explicit dependency ordering (middleware before routes)

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=config.app.name,
        description=(
            "Production-grade RAG system for querying the AWS Well-Architected "
            "Framework. Fine-tuned Phi-3-mini generation with OpenAI fallback."
        ),
        version=config.app.version,
        docs_url="/docs",        # Swagger UI at /docs
        redoc_url="/redoc",      # ReDoc at /redoc
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────
    _register_middleware(app)

    # ── Exception Handlers ─────────────────────────────────────────────────
    _register_exception_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────
    _register_routers(app)

    # ── Prometheus Metrics ────────────────────────────────────────────────
    setup_metrics(app)

    return app


def _register_middleware(app: FastAPI) -> None:
    """Register middleware in the correct order. Order matters — first added = outermost."""

    # CORS — must be outermost middleware
    # Allow Streamlit frontend (localhost:8501) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Request ID middleware — inject unique request ID for tracing
    # Every log entry for a request should carry the same request_id
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """
        Assign a unique request ID to every incoming request.

        The request_id is:
          - Added to request.state for use in route handlers
          - Returned as X-Request-ID response header
          - Included in every log entry for this request

        Using UUID4 (random) rather than sequential IDs to prevent
        information leakage about request volume to external callers.
        """
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()

        response = await call_next(request)

        # Add request ID to response headers for client-side correlation
        response.headers["X-Request-ID"] = request_id

        # Log request completion with timing
        duration_ms = round((time.time() - request.state.start_time) * 1000)
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        return response


def _register_routers(app: FastAPI) -> None:
    """Register all API routers with their prefixes and tags."""
    app.include_router(
        health.router,
        prefix="",
        tags=["Health"],
    )
    app.include_router(
        generate.router,
        prefix="",
        tags=["Generation"],
    )
    app.include_router(
        ingest.router,
        prefix="",
        tags=["Ingestion"],
    )


def _register_exception_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers.

    Why centralized exception handlers?
      Without these, FastAPI returns raw Python tracebacks to the client
      for unhandled exceptions. This:
        1. Exposes internal implementation details (security risk)
        2. Returns non-JSON responses that break API clients
        3. Provides no request_id for client-side correlation

      Centralized handlers ensure every error response:
        - Is JSON-formatted with a consistent schema
        - Includes request_id for correlation with server logs
        - Uses appropriate HTTP status codes
        - Logs the full error server-side while returning a clean message
    """

    from fastapi.exceptions import RequestValidationError
    from pydantic import ValidationError

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """
        Handle Pydantic validation errors (422 Unprocessable Entity).

        Triggered when request body fails Pydantic schema validation.
        Returns field-level error details so the client knows exactly
        which field failed and why.
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(
            "Request validation failed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "errors": exc.errors(),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error_code="VALIDATION_ERROR",
                message="Request validation failed. Check the 'details' field for field-level errors.",
                request_id=request_id,
                details=exc.errors(),
            ).model_dump(),
        )

    @app.exception_handler(TimeoutError)
    async def timeout_exception_handler(
        request: Request,
        exc: TimeoutError,
    ) -> JSONResponse:
        """
        Handle asyncio.TimeoutError from generation (504 Gateway Timeout).

        Triggered when LLM generation exceeds config.generation.timeout_seconds.
        Returns 504 so load balancers and clients know to retry.
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            "Generation timeout",
            extra={
                "request_id": request_id,
                "timeout_seconds": config.generation.timeout_seconds,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=ErrorResponse(
                error_code="GENERATION_TIMEOUT",
                message=(
                    f"LLM generation exceeded the {config.generation.timeout_seconds}s "
                    "timeout. The model may be under load. Please retry."
                ),
                request_id=request_id,
            ).model_dump(),
            headers={"Retry-After": "30"},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """
        Catch-all handler for unhandled exceptions (500 Internal Server Error).

        Logs the full traceback server-side.
        Returns a clean error message to the client — no traceback exposure.
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(
            "Unhandled exception",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message=(
                    "An internal error occurred. The error has been logged. "
                    f"Reference ID: {request_id}"
                ),
                request_id=request_id,
            ).model_dump(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Application Instance
# ─────────────────────────────────────────────────────────────────────────────

app = create_app()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.app.environment == "development",
        log_level=config.app.log_level.lower(),
        # access_log=False because our middleware already logs requests
        access_log=False,
    )