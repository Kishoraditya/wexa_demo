"""
tests/test_vector_store.py

Unit tests for embedding cache, FAISS store, and reranker.
All tests are offline — no model downloads, no API calls.
Model-dependent tests are marked with @pytest.mark.slow.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain.schema import Document

from backend.services.vector_store import (
    EmbeddingCache,
    CrossEncoderReranker,
    RetrievalService,
)


class TestEmbeddingCache:
    """Test the disk-backed embedding cache."""

    def test_empty_cache_returns_none(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        assert cache.get("nonexistent_hash") is None

    def test_set_and_get_roundtrip(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        embedding = [0.1, 0.2, 0.3, 0.4]
        cache.set("hash_abc", embedding)
        retrieved = cache.get("hash_abc")
        assert retrieved == embedding

    def test_persists_to_disk(self, tmp_path):
        cache1 = EmbeddingCache(str(tmp_path))
        cache1.set("hash_xyz", [1.0, 2.0, 3.0])

        # Create a new cache instance pointing to same dir
        # It should load the persisted data
        cache2 = EmbeddingCache(str(tmp_path))
        assert cache2.get("hash_xyz") == [1.0, 2.0, 3.0]

    def test_set_batch_stores_all(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        entries = {
            "hash_1": [0.1, 0.2],
            "hash_2": [0.3, 0.4],
            "hash_3": [0.5, 0.6],
        }
        cache.set_batch(entries)
        for key, value in entries.items():
            assert cache.get(key) == value

    def test_size_reflects_entry_count(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        assert cache.size == 0
        cache.set("h1", [0.1])
        cache.set("h2", [0.2])
        assert cache.size == 2

    def test_corrupt_cache_file_starts_fresh(self, tmp_path):
        cache_file = tmp_path / "embedding_cache.json"
        cache_file.write_text("this is not valid json{{{")
        # Should not raise, should start with empty cache
        cache = EmbeddingCache(str(tmp_path))
        assert cache.size == 0


class TestCrossEncoderReranker:
    """Test reranker behavior — mocked to avoid model download in CI."""

    def test_reranker_disabled_returns_candidates_unchanged(self):
        """When reranker is disabled, candidates pass through unchanged."""
        with patch("backend.services.vector_store.config") as mock_config:
            mock_config.reranker.enabled = False
            mock_config.reranker.model = "BAAI/bge-reranker-base"
            mock_config.reranker.top_n = 5

            reranker = CrossEncoderReranker()
            assert not reranker.is_available

            doc = Document(page_content="test", metadata={})
            candidates = [(doc, 0.85), (doc, 0.75)]
            result = reranker.rerank("test query", candidates)
            assert result == candidates

    def test_reranker_returns_top_n(self):
        """Reranker should truncate to top_n results."""
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = MagicMock()

        import numpy as np
        # Mock scores: third candidate is best
        reranker._model.predict.return_value = np.array([0.5, 0.3, 0.9])

        with patch("backend.services.vector_store.config") as mock_config:
            mock_config.reranker.enabled = True
            mock_config.reranker.top_n = 2

            docs = [
                (Document(page_content=f"doc{i}", metadata={}), 0.8)
                for i in range(3)
            ]
            result = reranker.rerank("query", docs, top_n=2)

        assert len(result) == 2
        # doc2 (score 0.9) should be first
        assert result[0][0].page_content == "doc2"

    def test_empty_candidates_returns_empty(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = MagicMock()
        result = reranker.rerank("query", [])
        assert result == []


class TestRetrievalService:
    """Test the RetrievalService integration."""

    def _make_service(self, results, reranker_available=False):
        """Helper: build a RetrievalService with mocked dependencies."""
        mock_store = MagicMock()
        mock_store.is_ready = True
        mock_store.similarity_search_with_scores.return_value = results

        mock_reranker = MagicMock()
        mock_reranker.is_available = reranker_available

        with patch("backend.services.vector_store.config") as mock_config:
            mock_config.vector_store.retrieval.top_k = 5
            mock_config.vector_store.retrieval.score_threshold = 0.70
            mock_config.reranker.enabled = False

            service = RetrievalService(
                vector_store=mock_store,
                reranker=mock_reranker,
            )
            service.top_k = 5
            service.score_threshold = 0.70

        return service, mock_store, mock_reranker

    def test_empty_results_triggers_no_rerank(self):
        service, mock_store, mock_reranker = self._make_service(results=[])
        result = service.retrieve("query about reliability")
        assert result == []
        mock_reranker.rerank.assert_not_called()

    def test_returns_correct_number_of_results(self):
        docs = [
            (Document(page_content=f"doc{i}", metadata={}), 0.8 - i * 0.05)
            for i in range(5)
        ]
        service, _, _ = self._make_service(results=docs)
        result = service.retrieve("query")
        assert len(result) <= 5

    def test_is_ready_delegates_to_store(self):
        service, mock_store, _ = self._make_service(results=[])
        mock_store.is_ready = True
        assert service.is_ready is True
        mock_store.is_ready = False
        assert service.is_ready is False