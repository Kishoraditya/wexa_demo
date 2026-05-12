"""
backend/routes/health.py

GET /health — Health Check and Liveness Probe
==============================================
Used by:
  - Load balancers (ALB health check): determines if instance is healthy
  - Kubernetes liveness probe: determines if pod should be restarted
  - Monitoring systems: tracks service availability over time

Two health check variants:
  GET /health       — lightweight liveness check (is the process alive?)
  GET /health/ready — readiness check (are all dependencies healthy?)

Why separate liveness and readiness?
  Liveness: "Is the process running and responsive?"
    Should always return 200 as long as the process is alive.
    Used by container orchestrators to restart crashed processes.
    Must be fast (<100ms) — if it times out, the orchestrator kills the pod.

  Readiness: "Is the application ready to serve traffic?"
    Returns 200 only when all dependencies are healthy.
    Used by load balancers to route traffic.
    A 503 response means "take this instance out of rotation temporarily."
    Appropriate when: index not built yet, model loading, transient error.

Author: Enterprise RAG Assistant
"""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from backend.core.config import get_config
from backend.core.logging import get_logger
from backend.models.schemas import HealthResponse

router = APIRouter()
logger = get_logger(__name__)
config = get_config()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Returns 200 if the application process is running.",
)
async def health_check(request: Request) -> HealthResponse:
    """
    Lightweight liveness probe.

    Always returns 200 as long as the FastAPI process is alive.
    Does NOT check dependency health — that is /health/ready.
    """
    vector_count = getattr(request.app.state, "vector_count", 0)
    primary_ready = getattr(request.app.state, "primary_model_ready", False)
    fallback_ready = getattr(request.app.state, "fallback_model_ready", False)

    return HealthResponse(
        status="healthy",
        vector_store_ready=vector_count > 0,
        primary_model_available=primary_ready,
        fallback_model_available=fallback_ready,
        vector_count=vector_count,
        version=config.app.version,
    )


@router.get(
    "/health/ready",
    summary="Readiness check",
    description=(
        "Returns 200 if the application is ready to serve requests. "
        "Returns 503 if the vector store is not built or no model is available."
    ),
)
async def readiness_check(request: Request) -> JSONResponse:
    """
    Readiness probe — checks that all dependencies are operational.

    Returns 503 (not 500) because 503 is the correct HTTP status for
    "temporarily unavailable" — it signals to the load balancer to stop
    routing traffic to this instance without marking it as permanently failed.
    """
    issues = []
    vector_count = getattr(request.app.state, "vector_count", 0)
    primary_ready = getattr(request.app.state, "primary_model_ready", False)
    fallback_ready = getattr(request.app.state, "fallback_model_ready", False)

    if vector_count == 0:
        issues.append("Vector store is empty — call POST /ingest to build the index")

    if not primary_ready and not fallback_ready:
        issues.append("No generation model available — check OPENAI_API_KEY")

    if issues:
        logger.warning(
            "Readiness check failed",
            extra={"issues": issues},
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "issues": issues,
                "vector_count": vector_count,
                "primary_model_available": primary_ready,
                "fallback_model_available": fallback_ready,
            },
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ready",
            "vector_count": vector_count,
            "primary_model_available": primary_ready,
            "fallback_model_available": fallback_ready,
            "version": config.app.version,
        },
    )