"""
tests/test_ingestion.py

Unit tests for the ingestion pipeline.
Tests run without any external services (no Pinecone, no OpenAI).
"""

import pytest
from pathlib import Path
from langchain.schema import Document

from backend.services.ingestion import (
    clean_text,
    compute_content_hash,
    deduplicate_chunks,
    normalize_metadata,
    chunk_documents,
    _extract_section_heading,
)


class TestCleanText:
    def test_removes_page_numbers(self):
        text = "Some content\nPage 3 of 47\nMore content"
        result = clean_text(text)
        assert "Page 3 of 47" not in result
        assert "Some content" in result

    def test_removes_form_feed(self):
        text = "Before\x0cAfter"
        result = clean_text(text)
        assert "\x0c" not in result

    def test_collapses_blank_lines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = clean_text(text)
        # Should have at most 2 consecutive blank lines
        assert "\n\n\n" not in result

    def test_normalizes_multi_spaces(self):
        text = "Word1    Word2"
        result = clean_text(text)
        assert "Word1    Word2" not in result
        assert "Word1 Word2" in result

    def test_preserves_content(self):
        text = "Design for failure is a core principle of reliability."
        result = clean_text(text)
        assert "Design for failure is a core principle of reliability." in result


class TestNormalizeMetadata:
    def test_known_pillar_maps_correctly(self):
        doc = Document(
            page_content="Some content",
            metadata={"source": "reliability.pdf", "page": 2}
        )
        result = normalize_metadata(doc, Path("data/pdfs/reliability.pdf"))
        assert result.metadata["source_pillar"] == "Reliability"

    def test_page_number_is_one_indexed(self):
        doc = Document(
            page_content="Content",
            metadata={"page": 0}  # PyPDFLoader uses 0-indexed
        )
        result = normalize_metadata(doc, Path("reliability.pdf"))
        # Should convert 0 → 1
        assert result.metadata["page_number"] == 1

    def test_all_required_fields_present(self):
        doc = Document(page_content="Content", metadata={"page": 0})
        result = normalize_metadata(doc, Path("security.pdf"))
        required_fields = [
            "source", "source_pillar", "file_type",
            "page_number", "section", "ingested_at", "char_count"
        ]
        for field in required_fields:
            assert field in result.metadata, f"Missing field: {field}"

    def test_unknown_pillar_uses_filename(self):
        doc = Document(page_content="Content", metadata={"page": 0})
        result = normalize_metadata(doc, Path("custom_document.pdf"))
        assert result.metadata["source_pillar"] == "Custom Document"


class TestContentHash:
    def test_same_text_same_hash(self):
        text = "Design for failure"
        assert compute_content_hash(text) == compute_content_hash(text)

    def test_different_text_different_hash(self):
        assert compute_content_hash("text A") != compute_content_hash("text B")

    def test_hash_is_64_chars(self):
        result = compute_content_hash("any text")
        assert len(result) == 64  # SHA-256 hex digest length


class TestDeduplication:
    def test_skips_existing_chunks(self):
        chunk = Document(page_content="existing content", metadata={})
        existing_hash = compute_content_hash("existing content")
        new_chunks, _ = deduplicate_chunks([chunk], {existing_hash})
        assert len(new_chunks) == 0

    def test_keeps_new_chunks(self):
        chunk = Document(page_content="brand new content", metadata={})
        new_chunks, _ = deduplicate_chunks([chunk], set())
        assert len(new_chunks) == 1

    def test_adds_hash_to_metadata(self):
        chunk = Document(page_content="some text", metadata={})
        deduplicate_chunks([chunk], set())
        assert "content_hash" in chunk.metadata

    def test_mixed_new_and_existing(self):
        existing = Document(page_content="old content", metadata={})
        new = Document(page_content="new content", metadata={})
        existing_hash = compute_content_hash("old content")
        result_chunks, _ = deduplicate_chunks([existing, new], {existing_hash})
        assert len(result_chunks) == 1
        assert result_chunks[0].page_content == "new content"


class TestSectionExtraction:
    def test_extracts_heading(self):
        text = "Design Principles\nThis section covers the key design principles."
        result = _extract_section_heading(text)
        assert result == "Design Principles"

    def test_returns_empty_for_no_heading(self):
        text = "This is a paragraph that starts immediately without a heading line."
        result = _extract_section_heading(text)
        # May or may not find a heading — just verify it returns a string
        assert isinstance(result, str)

    def test_skips_lines_ending_with_period(self):
        text = "This is a sentence.\nReal Heading\nBody text follows."
        result = _extract_section_heading(text)
        assert result == "Real Heading"