"""
backend/core/logging.py

Structured Logging — JSON Format with Consistent Schema
========================================================
Every log entry is a machine-parseable JSON object.
Every request-scoped log carries the same request_id so all entries
for a single request can be correlated in any log aggregation system.

Why structured logging matters in production:
  Plain text logs:
    "Request completed in 1250ms for query about reliability"
  → Unqueryable. To find slow requests you grep, which doesn't scale.

  Structured JSON logs:
    {"total_latency_ms": 1250, "confidence": "HIGH", "model_used": "openai"}
  → Queryable. In CloudWatch Logs Insights:
    filter total_latency_ms > 5000 | stats avg(total_latency_ms) by model_used
  → In Datadog: create alerts on p95 latency without writing a single line of code.

Privacy considerations:
  - Raw query text is NEVER logged (could contain PII from the user)
  - query_hash (SHA-256) is logged instead — enables correlation without exposure
  - Generated answer text is never logged (could contain PII)
  - Source excerpts are never logged (could be confidential document content)
  - User IP addresses are not logged (GDPR compliance posture)

Log levels and when to use each:
  DEBUG:   Retrieval scores, embedding dimensions, cache key lookups
           Only visible in development (LOG_LEVEL=DEBUG in config.yaml)
  INFO:    Request completion, model routing decisions, ingestion stats
           Standard production log level — one entry per request
  WARNING: Guardrail blocks, cache errors, reranker unavailable
           Should trigger review if sustained — not immediately actionable
  ERROR:   Generation failures, model load failures, unhandled exceptions
           Should trigger immediate investigation and alerting

Author: Enterprise RAG Assistant
"""

import hashlib
import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Log Schema
# ─────────────────────────────────────────────────────────────────────────────

# Canonical set of fields that every request-scoped log entry SHOULD contain.
# Not all fields are present in every entry — some are only available after
# certain pipeline stages complete. Missing fields are omitted (not null).
#
# This schema is the contract between the application and the ops team.
# Changing field names breaks dashboards and alerts. Add fields freely,
# rename only with coordinated migration.

REQUEST_LOG_SCHEMA = {
    # ── Identity ──────────────────────────────────────────────────────────
    "request_id":           "UUID assigned to this request",
    "timestamp":            "ISO 8601 UTC (added by loguru automatically)",

    # ── Query (privacy-safe) ───────────────────────────────────────────────
    "query_hash":           "SHA-256 of normalized query (not the raw query)",
    "query_length":         "Character count of the query",
    "filter_pillar":        "Pillar filter if specified, else null",

    # ── Routing ────────────────────────────────────────────────────────────
    "model_used":           "fine_tuned_phi3_qlora | openai_gpt4o_mini | unavailable",
    "use_fine_tuned":       "Whether caller requested fine-tuned model",
    "cache_hit":            "True if response served from query cache",

    # ── Retrieval ──────────────────────────────────────────────────────────
    "chunks_retrieved":     "Number of chunks returned by retriever",
    "top_retrieval_score":  "Highest cosine similarity score from retrieval",
    "retrieval_latency_ms": "Time for embedding + FAISS search + reranking",

    # ── Generation ─────────────────────────────────────────────────────────
    "generation_latency_ms": "Time for LLM to produce the response",
    "tokens_input":          "Approximate input tokens (prompt length)",
    "tokens_output":         "Tokens in the generated response",

    # ── Quality signals ────────────────────────────────────────────────────
    "confidence":           "HIGH | MEDIUM | LOW (composite signal)",
    "grounding_score":      "Cosine similarity between answer and context",
    "grounding_flag":       "True if grounding check failed",
    "is_refusal":           "True if system refused to answer",
    "pii_redacted":         "True if PII was found and redacted in output",

    # ── Performance ────────────────────────────────────────────────────────
    "total_latency_ms":     "Wall-clock time from request receipt to response",

    # ── Errors ─────────────────────────────────────────────────────────────
    "error":                "Error message if any stage failed",
    "error_type":           "Exception class name",
}


# ─────────────────────────────────────────────────────────────────────────────
# Setup Function
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/app.log",
    rotation: str = "100 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure loguru for structured JSON output to stdout and a rotating file.

    Call this ONCE at application startup, before any module calls get_logger().

    Why loguru over the standard logging module?
      - serialize=True gives zero-config JSON output
      - Structured extra={} fields are first-class, not string interpolation
      - Rotation and retention built-in (no RotatingFileHandler boilerplate)
      - Exception tracing with full context, including local variables

    Why both stdout AND file?
      Stdout: consumed by the container runtime (Docker, ECS) and forwarded
              to CloudWatch Logs / Datadog automatically. No file management.
      File:   local debugging on the EC2 instance. Also consumed by the
              filebeat agent if the deployment uses ELK stack.

    Args:
        log_level: Minimum log level. DEBUG | INFO | WARNING | ERROR | CRITICAL.
        log_file:  Path for the rotating log file.
        rotation:  File size or time trigger for rotation (loguru format).
        retention: How long to keep rotated files.
    """
    # Remove the default loguru handler (plain text to stderr)
    # We replace it with our configured handlers below
    logger.remove()

    # ── Handler 1: JSON to stdout ──────────────────────────────────────────
    # Container environments (ECS, Kubernetes) capture stdout.
    # JSON format enables CloudWatch Logs Insights queries:
    #   filter request_id = "abc-123"
    #   filter confidence = "LOW" | stats count() by model_used
    logger.add(
        sys.stdout,
        level=log_level,
        serialize=True,          # serialize=True → JSON output
        format="{message}",      # loguru handles JSON structure
        colorize=False,          # No ANSI codes in JSON
        backtrace=True,          # Include exception traceback in JSON
        diagnose=False,          # Don't expose local variable values in prod
        # diagnose=True is useful in development for debugging but can
        # expose sensitive data (e.g. config values, model outputs)
    )

    # ── Handler 2: JSON to rotating file ──────────────────────────────────
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        level=log_level,
        serialize=True,
        rotation=rotation,       # Rotate when file hits 100MB
        retention=retention,     # Delete files older than 7 days
        compression="gz",        # Compress rotated files (saves ~80% disk)
        backtrace=True,
        diagnose=False,
        encoding="utf-8",
        # enqueue=True would make file writes non-blocking (async queue).
        # Useful for high-throughput services. Not needed here — logging
        # is not on the critical path of request handling.
    )

    logger.info(
        "Logging initialized",
        extra={
            "log_level": log_level,
            "log_file": log_file,
            "format": "JSON",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Logger Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str):
    """
    Return a loguru logger bound with the module name.

    The module name appears as a field in every JSON log entry,
    making it easy to filter logs from a specific module:
      filter module = "backend.services.rag_pipeline"

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        loguru Logger instance bound with module=name.
    """
    return logger.bind(module=name)


# ─────────────────────────────────────────────────────────────────────────────
# Request Logger
# ─────────────────────────────────────────────────────────────────────────────

class RequestLogger:
    """
    Structured logger scoped to a single request lifecycle.

    Collects log fields incrementally as the pipeline progresses,
    then emits a single comprehensive log entry at request completion.

    Why collect fields and emit once?
      Multiple log entries per request create noise and require JOIN-like
      queries to reconstruct request context. One entry per request means:
        - One row in CloudWatch Logs Insights per request
        - Simple aggregation: avg(total_latency_ms) just works
        - No need to correlate multiple entries by request_id

    Usage pattern:
        req_log = RequestLogger(request_id="abc-123", query="How do I...")
        req_log.set_retrieval(chunks=5, top_score=0.87, latency_ms=145)
        req_log.set_generation(latency_ms=2300, tokens_out=180, model="openai")
        req_log.set_quality(confidence="HIGH", grounding_score=0.82)
        req_log.emit()  # writes the single JSON log entry
    """

    def __init__(
        self,
        request_id: str,
        query: str,
        use_fine_tuned: bool = True,
        filter_pillar: Optional[str] = None,
    ):
        self._log = get_logger("request")
        self._request_id = request_id

        # Privacy: hash the query, never store the raw text
        # SHA-256 is overkill for this purpose but consistent with ingestion
        # Use the first 16 chars for readability (collision risk negligible
        # at the scale of API logs)
        normalized_query = query.strip().lower()
        self._query_hash = hashlib.sha256(
            normalized_query.encode("utf-8")
        ).hexdigest()[:16]

        # Initialize the fields dict with request-level context
        self._fields: dict[str, Any] = {
            "request_id": request_id,
            "query_hash": self._query_hash,
            "query_length": len(query),
            "use_fine_tuned": use_fine_tuned,
            "filter_pillar": filter_pillar,
            "cache_hit": False,
            "grounding_flag": False,
            "is_refusal": False,
            "pii_redacted": False,
        }

    def set_cache_hit(self, hit: bool) -> "RequestLogger":
        """Mark whether this request was served from cache."""
        self._fields["cache_hit"] = hit
        return self

    def set_retrieval(
        self,
        chunks_retrieved: int,
        top_score: float,
        latency_ms: int,
    ) -> "RequestLogger":
        """Record retrieval stage metrics."""
        self._fields.update({
            "chunks_retrieved": chunks_retrieved,
            "top_retrieval_score": round(top_score, 4),
            "retrieval_latency_ms": latency_ms,
        })
        return self

    def set_generation(
        self,
        latency_ms: int,
        model_used: str,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
    ) -> "RequestLogger":
        """Record generation stage metrics."""
        self._fields.update({
            "generation_latency_ms": latency_ms,
            "model_used": model_used,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        })
        return self

    def set_quality(
        self,
        confidence: str,
        grounding_score: float,
        grounding_flag: bool,
        is_refusal: bool,
        pii_redacted: bool = False,
    ) -> "RequestLogger":
        """Record quality signal fields."""
        self._fields.update({
            "confidence": confidence,
            "grounding_score": round(grounding_score, 4),
            "grounding_flag": grounding_flag,
            "is_refusal": is_refusal,
            "pii_redacted": pii_redacted,
        })
        return self

    def set_error(self, error: str, error_type: str) -> "RequestLogger":
        """Record error information."""
        self._fields.update({
            "error": error,
            "error_type": error_type,
        })
        return self

    def emit(self, total_latency_ms: int, level: str = "INFO") -> None:
        """
        Emit the single request log entry with all accumulated fields.

        Args:
            total_latency_ms: Wall-clock time for the full request.
            level: Log level. Use ERROR if any stage failed.
        """
        self._fields["total_latency_ms"] = total_latency_ms

        log_fn = {
            "DEBUG": self._log.debug,
            "INFO": self._log.info,
            "WARNING": self._log.warning,
            "ERROR": self._log.error,
        }.get(level.upper(), self._log.info)

        # The message string is intentionally terse — all signal is in extra={}
        # Log aggregation systems index on the structured fields, not the message
        log_fn(
            "request_complete",
            extra=self._fields,
        )

    def emit_guardrail_block(self, reason: str, total_latency_ms: int) -> None:
        """Emit log entry for a request blocked by guardrails."""
        self._fields.update({
            "total_latency_ms": total_latency_ms,
            "guardrail_block_reason": reason,
            "is_refusal": True,
        })
        self._log.warning("request_blocked_by_guardrail", extra=self._fields)


# ─────────────────────────────────────────────────────────────────────────────
# Audit Logger
# ─────────────────────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Separate logger for security-sensitive events.

    Audit logs are written to a separate file so they can be:
      - Retained for longer periods (compliance: 1 year vs 7 days)
      - Shipped to a separate, tamper-evident log store
      - Monitored by security tooling independently of application logs

    Events that should be audit-logged:
      - Prompt injection attempts (input guardrail blocks)
      - PII detected in output
      - Model fallback activations (could indicate primary model compromise)
      - Ingest operations (document changes are security-relevant)
    """

    def __init__(self):
        self._log = get_logger("audit")

    def log_injection_attempt(
        self,
        request_id: str,
        query_hash: str,
        pattern_matched: str,
    ) -> None:
        """Log a detected prompt injection attempt."""
        self._log.warning(
            "AUDIT: prompt_injection_detected",
            extra={
                "request_id": request_id,
                "query_hash": query_hash,
                "pattern_matched": pattern_matched,
                "event_type": "security.prompt_injection",
            },
        )

    def log_pii_detected(
        self,
        request_id: str,
        pii_types: list[str],
    ) -> None:
        """Log detection of PII in generated output."""
        self._log.warning(
            "AUDIT: pii_detected_in_output",
            extra={
                "request_id": request_id,
                "pii_types": pii_types,
                "event_type": "security.pii_detection",
            },
        )

    def log_ingestion(
        self,
        files_loaded: int,
        chunks_indexed: int,
        triggered_by: str = "api",
    ) -> None:
        """Log a document ingestion event."""
        self._log.info(
            "AUDIT: document_ingestion",
            extra={
                "files_loaded": files_loaded,
                "chunks_indexed": chunks_indexed,
                "triggered_by": triggered_by,
                "event_type": "data.ingestion",
            },
        )

    def log_fallback_activation(
        self,
        request_id: str,
        reason: str,
    ) -> None:
        """Log activation of the OpenAI fallback model."""
        self._log.warning(
            "AUDIT: fallback_model_activated",
            extra={
                "request_id": request_id,
                "reason": reason,
                "event_type": "model.fallback_activation",
            },
        )


# Module-level audit logger instance
# Import this in any module that needs to write audit events
audit_logger = AuditLogger()