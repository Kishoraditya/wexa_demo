"""
tests/test_api.py

API Integration Tests
======================
Tests the FastAPI endpoints using TestClient.
Services are mocked — no real model inference, no Pinecone calls.

Testing strategy:
  - Unit tests (test_ingestion.py, test_vector_store.py): test service logic
  - Integration tests (this file): test HTTP layer, request validation,
    error handling, response schema compliance
  - End-to-end tests: run the full stack (not automated — manual smoke test)

Author: Enterprise RAG Assistant
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.models.schemas import (
    RAGResponse,
    ConfidenceLevel,
    SourceDocument,
    HealthResponse,
)
from backend.services.llm_manager import ModelType


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_mock_rag_response(
    answer: str = "Test answer.",
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    is_refusal: bool = False,
    grounding_flag: bool = False,
) -> RAGResponse:
    """Build a realistic mock RAGResponse for testing."""
    return RAGResponse(
        answer=answer,
        sources=[
            SourceDocument(
                pillar="Reliability",
                source_file="reliability.pdf",
                page_number=14,
                section="Design Principles",
                excerpt="Design for failure...",
                relevance_score=0.847,
                chunk_index=0,
            )
        ],
        confidence=confidence,
        confidence_reason="Direct match in context.",
        model_used=ModelType.OPENAI_FALLBACK,
        retrieval_latency_ms=120,
        generation_latency_ms=1500,
        total_latency_ms=1640,
        tokens_used=350,
        cache_hit=False,
        grounding_flag=grounding_flag,
        grounding_score=0.82,
        is_refusal=is_refusal,
        prompt_version="v1",
    )


@pytest.fixture
def mock_app():
    """
    Create a test app with all services mocked.

    Patches initialize_services to inject mock services into app.state
    rather than loading real models.
    """
    app = create_app()

    async def mock_initialize(app_instance):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = make_mock_rag_response()

        mock_retrieval = MagicMock()
        mock_retrieval.is_ready = True

        mock_ingestion = MagicMock()
        mock_ingestion.run.return_value = MagicMock(
            has_new_content=True,
            total_files_loaded=6,
            total_pages_loaded=150,
            total_chunks_created=800,
            new_chunks=[],
            duplicate_chunks_skipped=0,
            duration_seconds=5.2,
            to_log_dict=lambda: {},
        )

        mock_orchestrator = MagicMock()
        mock_orchestrator.ingest_from_result.return_value = {
            "chunks_embedded": 800,
            "chunks_indexed": 800,
            "total_vectors_in_index": 800,
        }
        mock_orchestrator.vector_store.get_all_hashes.return_value = set()
        mock_orchestrator.vector_store.vector_count = 800

        mock_llm = MagicMock()
        mock_llm.primary_available = False
        mock_llm.fallback_available = True
        mock_llm.any_model_available = True

        app_instance.state.rag_pipeline = mock_pipeline
        app_instance.state.retrieval_service = mock_retrieval
        app_instance.state.ingestion_pipeline = mock_ingestion
        app_instance.state.ingestion_orchestrator = mock_orchestrator
        app_instance.state.llm_manager = mock_llm
        app_instance.state.vector_count = 800
        app_instance.state.primary_model_ready = False
        app_instance.state.fallback_model_ready = True
        app_instance.state.input_guardrail = MagicMock()
        app_instance.state.output_guardrail = MagicMock()
        app_instance.state.grounding_checker = MagicMock()

    with patch("backend.core.dependencies.initialize_services", side_effect=mock_initialize):
        with patch("backend.core.dependencies.shutdown_services", new_callable=AsyncMock):
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client, app


# ─────────────────────────────────────────────────────────────────────────────
# Health Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, mock_app):
        client, app = mock_app
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, mock_app):
        client, app = mock_app
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "vector_store_ready" in data
        assert "primary_model_available" in data
        assert "fallback_model_available" in data
        assert "vector_count" in data

    def test_health_reflects_app_state(self, mock_app):
        client, app = mock_app
        response = client.get("/health")
        data = response.json()
        assert data["vector_count"] == 800
        assert data["fallback_model_available"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Generate Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateEndpoint:
    def test_valid_request_returns_200(self, mock_app):
        client, app = mock_app
        response = client.post(
            "/generate",
            json={"query": "How does AWS recommend handling failure?"},
        )
        assert response.status_code == 200

    def test_response_has_required_fields(self, mock_app):
        client, app = mock_app
        response = client.post(
            "/generate",
            json={"query": "What is least privilege?"},
        )
        data = response.json()
        required_fields = [
            "answer", "sources", "confidence", "model_used",
            "retrieval_latency_ms", "generation_latency_ms",
            "total_latency_ms", "cache_hit", "grounding_flag", "is_refusal",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_empty_query_returns_422(self, mock_app):
        client, app = mock_app
        response = client.post("/generate", json={"query": ""})
        assert response.status_code == 422

    def test_query_too_long_returns_422(self, mock_app):
        client, app = mock_app
        response = client.post("/generate", json={"query": "x" * 1001})
        assert response.status_code == 422

    def test_missing_query_returns_422(self, mock_app):
        client, app = mock_app
        response = client.post("/generate", json={})
        assert response.status_code == 422

    def test_invalid_pillar_returns_422(self, mock_app):
        client, app = mock_app
        response = client.post(
            "/generate",
            json={"query": "test", "filter_pillar": "InvalidPillar"},
        )
        assert response.status_code == 422

    def test_valid_pillar_filter_accepted(self, mock_app):
        client, app = mock_app
        response = client.post(
            "/generate",
            json={"query": "reliability best practices", "filter_pillar": "Reliability"},
        )
        assert response.status_code == 200

    def test_top_k_out_of_range_returns_422(self, mock_app):
        client, app = mock_app
        response = client.post(
            "/generate",
            json={"query": "test", "top_k": 11},  # max is 10
        )
        assert response.status_code == 422

    def test_top_k_zero_returns_422(self, mock_app):
        client, app = mock_app
        response = client.post(
            "/generate",
            json={"query": "test", "top_k": 0},  # min is 1
        )
        assert response.status_code == 422

    def test_request_id_in_response_headers(self, mock_app):
        client, app = mock_app
        response = client.post(
            "/generate",
            json={"query": "What is auto-scaling?"},
        )
        assert "x-request-id" in response.headers
        # UUID format: 8-4-4-4-12 hex characters
        request_id = response.headers["x-request-id"]
        assert len(request_id) == 36

    def test_use_fine_tuned_false_accepted(self, mock_app):
        client, app = mock_app
        response = client.post(
            "/generate",
            json={"query": "test query", "use_fine_tuned": False},
        )
        assert response.status_code == 200

    def test_pipeline_error_returns_500(self, mock_app):
        client, app = mock_app
        # Make the pipeline raise an exception
        app.state.rag_pipeline.run.side_effect = RuntimeError("Pipeline error")
        response = client.post(
            "/generate",
            json={"query": "test query"},
        )
        assert response.status_code == 500
        data = response.json()
        assert "error_code" in data
        assert "request_id" in data
        # Ensure no traceback exposed to client
        assert "Traceback" not in str(data)
        assert "RuntimeError" not in data.get("message", "")


# ─────────────────────────────────────────────────────────────────────────────
# Error Response Schema Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorResponseSchema:
    def test_validation_error_has_details(self, mock_app):
        """422 responses should include field-level error details."""
        client, app = mock_app
        response = client.post("/generate", json={"query": ""})
        data = response.json()
        assert "error_code" in data
        assert data["error_code"] == "VALIDATION_ERROR"
        assert "details" in data
        assert isinstance(data["details"], list)

    def test_error_response_has_request_id(self, mock_app):
        """All error responses must include request_id for log correlation."""
        client, app = mock_app
        response = client.post("/generate", json={})
        data = response.json()
        assert "request_id" in data


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_endpoint_returns_200(self, mock_app):
        client, app = mock_app
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type_is_prometheus(self, mock_app):
        client, app = mock_app
        response = client.get("/metrics")
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type

    def test_custom_metrics_present(self, mock_app):
        client, app = mock_app
        # Make a generate request first to populate metrics
        client.post("/generate", json={"query": "test"})
        response = client.get("/metrics")
        metrics_text = response.text
        assert "rag_requests_total" in metrics_text
        assert "rag_retrieval_latency_seconds" in metrics_text
        assert "rag_generation_latency_seconds" in metrics_text
        