"""
backend/core/metrics.py

Prometheus Metrics — Complete Implementation
=============================================
All custom metrics are defined here as module-level constants.
Import individual metrics wherever they need to be incremented/observed.

Metric types and when to use each:
  Counter:    Monotonically increasing value. Never decreases.
              Use for: requests processed, errors, tokens consumed.
              Query pattern: rate(metric[5m]) for per-second rate.

  Histogram:  Samples observations into configurable buckets.
              Use for: latency, response size, score distributions.
              Query pattern: histogram_quantile(0.95, metric) for p95.

  Gauge:      Point-in-time value. Can go up or down.
              Use for: current queue depth, loaded model count, index size.
              Query pattern: metric (current value).

  Summary:    Similar to histogram but with pre-defined quantiles.
              Prefer histogram — quantiles can be aggregated across instances.
              Summary quantiles cannot be meaningfully aggregated.

Bucket selection rationale:
  Retrieval buckets [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    Budget is 200ms. Buckets below budget give high resolution where it matters.
    Anything above 1s is already a p95 miss — we just need to know how bad.

  Generation buckets [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0]:
    Fine-tuned model on CPU: 5-20s. OpenAI: 1-3s. Timeout at 30s.
    Coarse buckets because generation has high natural variance.

Alert thresholds (implement in Prometheus alerting rules or Grafana):
  CRITICAL:
    rag_requests_total{status="error"} rate > 0.05/s for 2min
    rag_generation_latency_seconds p95 > 8.0
  WARNING:
    rag_fallback_activations_total rate > 0.3/s for 5min
    rag_hallucination_flags_total rate > 0.1/s for 10min
    rag_cache_hits_total rate / rag_requests_total rate < 0.1 (cache miss rate > 90%)

Grafana dashboard panels (recommended):
  Row 1: Request rate | Error rate | p50/p95 total latency
  Row 2: Retrieval p95 | Generation p95 | Cache hit rate
  Row 3: Model usage (fine_tuned vs fallback) | Token consumption rate
  Row 4: Hallucination flag rate | Vector store size

Author: Enterprise RAG Assistant
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import default

from backend.core.config import get_config
from backend.core.logging import get_logger

logger = get_logger(__name__)
config = get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Request Metrics
# ─────────────────────────────────────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    name="rag_requests_total",
    documentation=(
        "Total number of requests processed by the /generate endpoint. "
        "Labels: status (success|error|timeout|refusal|cache_hit), "
        "model (fine_tuned|openai_fallback|cache), "
        "endpoint (generate|ingest|health)"
    ),
    labelnames=["status", "model", "endpoint"],
)

REQUESTS_IN_FLIGHT = Gauge(
    name="rag_requests_in_flight",
    documentation=(
        "Number of requests currently being processed. "
        "High values indicate the system is under load. "
        "Alert if sustained > 10 (single instance capacity)."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Latency Metrics
# ─────────────────────────────────────────────────────────────────────────────

RETRIEVAL_LATENCY = Histogram(
    name="rag_retrieval_latency_seconds",
    documentation=(
        "Time to embed query and retrieve relevant chunks from FAISS. "
        "Includes: query embedding + FAISS search + reranking (if enabled). "
        "Budget: p95 < 0.2s. Alert at p95 > 0.5s."
    ),
    buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
)

GENERATION_LATENCY = Histogram(
    name="rag_generation_latency_seconds",
    documentation=(
        "Time for the LLM to generate a response after receiving the prompt. "
        "Labeled by model to compare fine-tuned vs fallback performance. "
        "Budget: fine_tuned p95 < 5s, openai p95 < 3s."
    ),
    buckets=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0],
    labelnames=["model"],
)

TOTAL_LATENCY = Histogram(
    name="rag_total_latency_seconds",
    documentation=(
        "End-to-end wall-clock latency for POST /generate requests. "
        "Budget: p95 < 7s. Cache hits should be < 0.05s."
    ),
    buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0],
    labelnames=["cache_hit"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

HALLUCINATION_FLAGS = Counter(
    name="rag_hallucination_flags_total",
    documentation=(
        "Responses where the semantic grounding check detected potential drift "
        "between the answer and retrieved context. "
        "Sustained rate > 10% warrants review of grounding_threshold or prompt. "
        "Production: run RAGAS faithfulness weekly to calibrate this counter."
    ),
)

REFUSALS_TOTAL = Counter(
    name="rag_refusals_total",
    documentation=(
        "Requests where the system returned a refusal (no relevant context). "
        "Labels: reason (no_context|guardrail_block|generation_error). "
        "High rate may indicate corpus gaps or query distribution shift."
    ),
    labelnames=["reason"],
)

CONFIDENCE_DISTRIBUTION = Counter(
    name="rag_confidence_level_total",
    documentation=(
        "Distribution of confidence levels across all responses. "
        "Label: level (HIGH|MEDIUM|LOW). "
        "Shift toward LOW indicates degrading retrieval or model quality."
    ),
    labelnames=["level"],
)

GROUNDING_SCORE = Histogram(
    name="rag_grounding_score",
    documentation=(
        "Distribution of semantic grounding scores (cosine similarity between "
        "answer embedding and retrieved context). "
        "Range: 0.0 to 1.0. Values below 0.50 trigger hallucination flag."
    ),
    buckets=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# ─────────────────────────────────────────────────────────────────────────────
# Model and Cost Metrics
# ─────────────────────────────────────────────────────────────────────────────

TOKENS_TOTAL = Counter(
    name="rag_tokens_total",
    documentation=(
        "Total tokens consumed across all generation calls. "
        "Labels: model (fine_tuned|openai_fallback), direction (input|output). "
        "Cost estimate: tokens_total{model='openai_fallback', direction='output'} "
        "× $0.00060 / 1000 = USD cost (GPT-4o-mini pricing, 2024)."
    ),
    labelnames=["model", "direction"],
)

FALLBACK_ACTIVATIONS = Counter(
    name="rag_fallback_activations_total",
    documentation=(
        "Number of times the OpenAI fallback was used instead of the fine-tuned "
        "primary model. Labels: reason (primary_unavailable|primary_failed|explicit). "
        "High rate indicates primary model issues — investigate adapter loading."
    ),
    labelnames=["reason"],
)

MODEL_LOAD_FAILURES = Counter(
    name="rag_model_load_failures_total",
    documentation=(
        "Number of times the fine-tuned model failed to load at startup or reload. "
        "Labels: error_type (hub_unreachable|oom|adapter_not_found|other)."
    ),
    labelnames=["error_type"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Cache Metrics
# ─────────────────────────────────────────────────────────────────────────────

CACHE_HITS = Counter(
    name="rag_cache_hits_total",
    documentation=(
        "Requests served from the query response cache. "
        "Labels: level (l1_embedding|l2_response). "
        "Cache hit rate = cache_hits / requests_total. "
        "Target: > 20% hit rate indicates effective caching."
    ),
    labelnames=["level"],
)

CACHE_MISSES = Counter(
    name="rag_cache_misses_total",
    documentation=(
        "Requests that missed the cache and required full pipeline execution."
    ),
)

CACHE_SIZE = Gauge(
    name="rag_cache_entries",
    documentation=(
        "Current number of entries in the response cache. "
        "Labels: cache_type (embedding|response). "
        "Grows until TTL expiry or manual flush."
    ),
    labelnames=["cache_type"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure Metrics
# ─────────────────────────────────────────────────────────────────────────────

VECTOR_STORE_SIZE = Gauge(
    name="rag_vector_store_size",
    documentation=(
        "Number of vectors currently in the FAISS index. "
        "Should be stable between ingestion runs. "
        "Alert if drops to 0 — index may have been deleted."
    ),
)

INGESTION_DURATION = Histogram(
    name="rag_ingestion_duration_seconds",
    documentation=(
        "Time to complete a full document ingestion run. "
        "Includes: loading, chunking, embedding, indexing."
    ),
    buckets=[30, 60, 120, 300, 600, 1200],
)

CHUNKS_PROCESSED = Counter(
    name="rag_chunks_processed_total",
    documentation=(
        "Total document chunks processed across all ingestion runs. "
        "Labels: outcome (indexed|deduplicated|filtered). "
        "deduplicated / (indexed + deduplicated) = dedup efficiency rate."
    ),
    labelnames=["outcome"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Application Info (static metadata, not time-series)
# ─────────────────────────────────────────────────────────────────────────────

APP_INFO = Info(
    name="rag_app",
    documentation="Static application metadata",
)


# ─────────────────────────────────────────────────────────────────────────────
# Setup Function
# ─────────────────────────────────────────────────────────────────────────────

def setup_metrics(app) -> None:
    """
    Initialize Prometheus instrumentation and expose /metrics endpoint.

    Called once during application startup (in main.py create_app()).

    prometheus_fastapi_instrumentator provides automatic metrics for ALL
    HTTP endpoints:
      http_requests_total{method, handler, status}
      http_request_duration_seconds{method, handler}

    We add our domain-specific RAG metrics on top of these automatic ones.

    /metrics endpoint format:
      Prometheus text exposition format (text/plain; version=0.0.4)
      Scraped by Prometheus every 15s (configurable in prometheus.yml)

    Production scrape config (prometheus.yml):
      scrape_configs:
        - job_name: 'rag-assistant'
          static_configs:
            - targets: ['rag-api:8000']
          scrape_interval: 15s
          metrics_path: /metrics

    Args:
        app: FastAPI application instance.
    """
    # Set application info (appears as a gauge with value 1.0 and labels)
    cfg = get_config()
    APP_INFO.info({
        "version": cfg.app.version,
        "environment": cfg.app.environment,
        "embedding_model": cfg.embedding.model,
        "base_llm_model": cfg.models.primary.base_model,
        "fallback_model": cfg.models.fallback.model,
    })

    # Initialize all label combinations to 0 so metrics appear immediately
    # in Prometheus without waiting for the first event.
    # A metric that has never been incremented won't appear in /metrics output,
    # which can cause "no data" gaps in Grafana dashboards.
    _initialize_metric_label_combinations()

    # Instrument FastAPI with automatic HTTP metrics
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=True,
        inprogress_labels=True,
        excluded_handlers=[
            "/metrics",      # don't track metrics about the metrics endpoint
            "/health",       # don't track health checks in request metrics
            "/health/ready",
            "/docs",
            "/openapi.json",
            "/redoc",
        ],
    )

    instrumentator.instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,  # Don't show in Swagger UI
        tags=["Observability"],
    )

    logger.info(
        "Prometheus metrics initialized",
        extra={"endpoint": "/metrics"},
    )


def _initialize_metric_label_combinations() -> None:
    """
    Pre-initialize all known label combinations so metrics appear immediately.

    Without this, a metric like rag_requests_total{status="error"} won't
    appear in /metrics until the first error occurs. Dashboards that reference
    this metric would show 'no data' until then, which can mask real issues
    (is the metric zero, or has there not been an error yet?).
    """
    # Request counters
    for status in ["success", "error", "timeout", "refusal", "cache_hit"]:
        for model in ["fine_tuned", "openai_fallback", "cache"]:
            REQUESTS_TOTAL.labels(
                status=status, model=model, endpoint="generate"
            )

    # Generation latency by model
    for model in ["fine_tuned", "openai_fallback"]:
        GENERATION_LATENCY.labels(model=model)

    # Total latency by cache hit
    for cache_hit in ["true", "false"]:
        TOTAL_LATENCY.labels(cache_hit=cache_hit)

    # Token counters
    for model in ["fine_tuned", "openai_fallback"]:
        for direction in ["input", "output"]:
            TOKENS_TOTAL.labels(model=model, direction=direction)

    # Fallback reasons
    for reason in ["primary_unavailable", "primary_failed", "explicit"]:
        FALLBACK_ACTIVATIONS.labels(reason=reason)

    # Refusal reasons
    for reason in ["no_context", "guardrail_block", "generation_error"]:
        REFUSALS_TOTAL.labels(reason=reason)

    # Confidence levels
    for level in ["HIGH", "MEDIUM", "LOW"]:
        CONFIDENCE_DISTRIBUTION.labels(level=level)

    # Cache metrics
    for level in ["l1_embedding", "l2_response"]:
        CACHE_HITS.labels(level=level)
    for cache_type in ["embedding", "response"]:
        CACHE_SIZE.labels(cache_type=cache_type)

    # Chunk outcomes
    for outcome in ["indexed", "deduplicated", "filtered"]:
        CHUNKS_PROCESSED.labels(outcome=outcome)

    # Model load failures
    for error_type in ["hub_unreachable", "oom", "adapter_not_found", "other"]:
        MODEL_LOAD_FAILURES.labels(error_type=error_type)

    logger.debug("Prometheus metric label combinations pre-initialized")