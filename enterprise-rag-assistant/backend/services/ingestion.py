"""
backend/services/ingestion.py

Document Ingestion Pipeline — Phase 1
======================================
Responsibilities:
  1. Load PDF, Markdown, and plain-text files from a source directory
  2. Normalize all documents to a consistent Document(page_content, metadata) format
  3. Strip noise: page numbers, headers, footers, excessive whitespace
  4. Chunk documents using RecursiveCharacterTextSplitter
  5. Deduplicate chunks by SHA-256 hash to prevent re-embedding on re-ingestion

Design decision — why this module exists as a standalone service:
  The ingestion pipeline is an OFFLINE, batch process. It runs when documents
  change, not on every user query. Keeping it separate from the RAG pipeline
  makes the boundary explicit: ingest once, query many times.

Author: Enterprise RAG Assistant
"""

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from backend.core.config import get_config
from backend.core.logging import get_logger

logger = get_logger(__name__)
config = get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Supported file extensions and their corresponding loader classes.
# Adding a new format requires only adding an entry here — no pipeline changes.
LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
}

# Regex patterns for noise that should be stripped from document text.
# These patterns are specific to AWS PDF exports — adjust per corpus.
NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),   # "Page 3 of 47"
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),               # lone page numbers
    re.compile(r"Amazon Web Services\s*[-–]\s*", re.IGNORECASE),  # repeated header
    re.compile(r"AWS Well-Architected Framework\s*\n", re.IGNORECASE),
    re.compile(r"\x0c"),                                     # form feed characters
]


# ─────────────────────────────────────────────────────────────────────────────
# Document Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_document(file_path: Path) -> list[Document]:
    """
    Load a single document file and return a list of LangChain Document objects.

    Why return a list?
      PyPDFLoader returns one Document per page. UnstructuredMarkdownLoader may
      return one Document per section. TextLoader returns one Document total.
      Returning a list from all loaders gives the caller a uniform interface —
      it never needs to know which loader was used.

    Args:
        file_path: Absolute or relative path to the document file.

    Returns:
        List of Document objects with raw content and initial metadata.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the loader fails to parse the file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in LOADER_MAP:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported types: {list(LOADER_MAP.keys())}"
        )

    loader_class = LOADER_MAP[suffix]
    logger.info(
        "Loading document",
        extra={"file": str(file_path), "loader": loader_class.__name__},
    )

    try:
        loader = loader_class(str(file_path))
        documents = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}") from e

    logger.info(
        "Document loaded",
        extra={"file": str(file_path), "pages": len(documents)},
    )
    return documents


def load_directory(directory: Path) -> list[Document]:
    """
    Recursively load all supported documents from a directory.

    This is the primary entry point for batch ingestion. It walks the directory,
    loads every supported file, and returns a flat list of all Document objects.

    Args:
        directory: Path to the directory containing source documents.

    Returns:
        Flat list of all Document objects across all files.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    all_documents: list[Document] = []
    failed_files: list[str] = []

    supported_files = [
        f for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in LOADER_MAP
    ]

    logger.info(
        "Starting directory load",
        extra={"directory": str(directory), "files_found": len(supported_files)},
    )

    for file_path in supported_files:
        try:
            documents = load_document(file_path)
            all_documents.extend(documents)
        except Exception as e:
            # Log the failure but continue — one bad file should not abort
            # the entire ingestion run. Failed files are reported at the end.
            logger.error(
                "Failed to load file, skipping",
                extra={"file": str(file_path), "error": str(e)},
            )
            failed_files.append(str(file_path))

    if failed_files:
        logger.warning(
            "Some files failed to load",
            extra={"failed_count": len(failed_files), "files": failed_files},
        )

    logger.info(
        "Directory load complete",
        extra={
            "total_documents": len(all_documents),
            "failed_files": len(failed_files),
        },
    )
    return all_documents


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Normalization
# ─────────────────────────────────────────────────────────────────────────────

# Map from filename stem to AWS pillar name.
# This is intentionally hardcoded for this corpus — the six pillar PDFs have
# predictable names. A production system would derive this from document
# metadata or a manifest file.
PILLAR_MAP: dict[str, str] = {
    # Keys match the actual PDF filename stems in data/pdfs/
    "wellarchitected-operational-excellence-pillar": "Operational Excellence",
    "wellarchitected-security-pillar": "Security",
    "wellarchitected-reliability-pillar": "Reliability",
    "wellarchitected-performance-efficiency-pillar": "Performance Efficiency",
    "wellarchitected-cost-optimization-pillar": "Cost Optimization",
    "wellarchitected-sustainability-pillar": "Sustainability",
    # Also support short names if PDFs are renamed
    "operational_excellence": "Operational Excellence",
    "security": "Security",
    "reliability": "Reliability",
    "performance_efficiency": "Performance Efficiency",
    "cost_optimization": "Cost Optimization",
    "sustainability": "Sustainability",
}


def normalize_metadata(document: Document, file_path: Path) -> Document:
    """
    Normalize a Document's metadata to the canonical schema.

    All downstream components (retriever, response builder, evaluation)
    depend on these fields being present and consistently named. If metadata
    is missing here, it will silently break source citation in the API response.

    Canonical metadata schema:
        source          str   — filename, e.g. "reliability.pdf"
        source_pillar   str   — AWS pillar name, e.g. "Reliability"
        file_type       str   — extension without dot, e.g. "pdf"
        page_number     int   — 1-indexed page number (0 if unknown)
        section         str   — section heading if detectable, else ""
        ingested_at     str   — ISO 8601 UTC timestamp
        char_count      int   — character count of page_content

    Args:
        document: Raw Document from a LangChain loader.
        file_path: Path to the source file (used to extract metadata).

    Returns:
        Document with normalized metadata. page_content is unchanged at
        this stage — cleaning happens in a separate step.
    """
    stem = file_path.stem.lower()

    # Derive the pillar name from the filename.
    # If the filename doesn't match a known pillar, fall back to the stem.
    # This makes the system resilient to adding new documents without code changes.
    source_pillar = PILLAR_MAP.get(stem, stem.replace("_", " ").title())

    # LangChain's PyPDFLoader stores page number in metadata["page"] as 0-indexed.
    # Normalize to 1-indexed for display in citations.
    raw_page = document.metadata.get("page", 0)
    page_number = raw_page + 1 if isinstance(raw_page, int) else 0

    normalized_metadata = {
        "source": file_path.name,
        "source_pillar": source_pillar,
        "file_type": file_path.suffix.lstrip(".").lower(),
        "page_number": page_number,
        "section": _extract_section_heading(document.page_content),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "char_count": len(document.page_content),
    }

    return Document(
        page_content=document.page_content,
        metadata=normalized_metadata,
    )


def _extract_section_heading(text: str) -> str:
    """
    Attempt to extract the first heading-like line from document text.

    AWS whitepapers use consistent heading patterns: short lines in title case
    or all-caps that appear before body text. This heuristic captures the most
    common pattern without requiring a full document parser.

    Why a heuristic and not a proper parser?
      The PDF-to-text conversion by PyPDFLoader does not preserve document
      structure (bold, font size, heading levels). A proper parser would require
      a more expensive PDF library (pdfminer, pymupdf) and adds complexity that
      is not justified for this use case. The heuristic is good enough for
      citation purposes and adds meaningful signal over "no section."

    Args:
        text: Raw page_content string.

    Returns:
        First detected section heading, or empty string if none found.
    """
    lines = text.strip().split("\n")
    for line in lines[:10]:  # only look in first 10 lines of the chunk
        line = line.strip()
        # A heading is likely: short (< 80 chars), not ending in punctuation,
        # and containing at least 2 words
        if (
            5 < len(line) < 80
            and not line.endswith((".", ",", ";", ":"))
            and len(line.split()) >= 2
            and not line[0].isdigit()  # skip numbered list items
        ):
            return line
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Strip noise from document text and normalize whitespace.

    What counts as noise in AWS PDF exports:
      - Page number stamps ("Page 3 of 47")
      - Repeated document title headers on each page
      - Form feed characters (\x0c) from PDF structure
      - Excessive blank lines (more than 2 consecutive)
      - Leading/trailing whitespace per line

    What we deliberately do NOT strip:
      - Bullet points and list markers — they are structural signals
      - Code snippets and inline commands — they carry technical meaning
      - Numbers within sentences — they are content, not noise

    Args:
        text: Raw page_content string.

    Returns:
        Cleaned string with noise removed and whitespace normalized.
    """
    # Apply each noise pattern in sequence
    for pattern in NOISE_PATTERNS:
        text = pattern.sub(" ", text)

    # Normalize line-level whitespace
    lines = [line.strip() for line in text.split("\n")]

    # Collapse runs of more than 2 blank lines into a single blank line.
    # Why 2 and not 1? AWS docs use double blank lines as paragraph separators.
    # Collapsing to 1 would merge paragraphs that should stay distinct.
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    # Normalize remaining multi-space sequences (e.g. from PDF column layout)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def build_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Build and return the configured text splitter.

    Why RecursiveCharacterTextSplitter?
      It tries to split on natural boundaries in priority order:
      paragraph breaks (\n\n) → line breaks (\n) → spaces → characters.
      This preserves semantic units (paragraphs, sentences) as long as they
      fit within the chunk size. Only when no natural boundary fits does it
      fall back to hard character splits.

    Why 512 tokens (≈ 2048 characters)?
      - Below 256 tokens: chunks become too small to contain a complete idea.
        Embedding quality degrades because a single sentence out of context
        carries a different meaning than within its paragraph.
      - Above 1024 tokens: the embedding must represent too many ideas at once,
        diluting the signal. Retrieval precision drops — the chunk matches many
        queries loosely rather than one query precisely.
      - 512 is the empirically validated sweet spot for technical documentation
        Q&A, confirmed by the BEIR benchmark and the BGE model paper.
      - Tune DOWN to 256-384 for factual lookup queries (exact term extraction).
      - Tune UP to 768-1024 for summarization or multi-step reasoning queries.
      - This value is configurable via config.yaml — no code change required.

    Why 50-token overlap?
      Without overlap, a sentence that spans a chunk boundary is split. The
      first half ends one chunk, the second half starts the next. Neither chunk
      contains the complete sentence. Overlap ensures continuity at boundaries.
      50 tokens (≈ 200 characters) is roughly 2-3 sentences — enough to
      preserve boundary context without inflating the index size significantly.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    chunk_config = config.ingestion.chunking

    # Character-based length function.
    # We use characters rather than tokens because the tokenizer is not available
    # at ingestion time without loading the model. The approximation is:
    # 1 token ≈ 4 characters for English technical text.
    # At chunk_size=512 tokens → chunk_size_chars ≈ 2048 characters.
    chunk_size_chars = chunk_config.chunk_size * 4
    chunk_overlap_chars = chunk_config.chunk_overlap * 4

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        separators=chunk_config.separators,
        length_function=len,
        is_separator_regex=False,
        # add_start_index=True adds a "start_index" field to chunk metadata,
        # enabling exact character-level provenance tracking.
        # Useful for highlighting source text in the UI.
        add_start_index=True,
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split a list of normalized Documents into chunks.

    Metadata inheritance:
      Each chunk inherits all metadata from its parent document and adds
      two new fields:
        chunk_index     — sequential index within the parent page/document
        start_index     — character offset within the original page_content

    This means every chunk can be traced back to:
      source file → pillar → page number → section → character position

    Args:
        documents: List of normalized, cleaned Documents (one per page).

    Returns:
        List of chunked Documents ready for embedding.
    """
    splitter = build_text_splitter()

    logger.info(
        "Starting chunking",
        extra={"input_documents": len(documents)},
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index within each source page.
    # Group by (source, page_number) and assign sequential indices.
    _assign_chunk_indices(chunks)

    # Filter out chunks that are too short to carry meaningful content.
    # A chunk under 50 characters is likely a stray header or page artifact
    # that survived the cleaning step. Embedding it wastes API calls and
    # pollutes the index with low-signal vectors.
    min_chunk_chars = 50
    before_filter = len(chunks)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= min_chunk_chars]
    filtered = before_filter - len(chunks)

    if filtered > 0:
        logger.warning(
            "Filtered short chunks",
            extra={"filtered_count": filtered, "min_chars": min_chunk_chars},
        )

    logger.info(
        "Chunking complete",
        extra={
            "input_documents": len(documents),
            "output_chunks": len(chunks),
            "avg_chunks_per_doc": round(len(chunks) / max(len(documents), 1), 1),
        },
    )

    return chunks


def _assign_chunk_indices(chunks: list[Document]) -> None:
    """
    Assign sequential chunk_index values within each (source, page_number) group.
    Mutates chunks in place — no return value.
    """
    from collections import defaultdict
    counter: dict[tuple, int] = defaultdict(int)

    for chunk in chunks:
        key = (
            chunk.metadata.get("source", ""),
            chunk.metadata.get("page_number", 0),
        )
        chunk.metadata["chunk_index"] = counter[key]
        counter[key] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def compute_content_hash(text: str) -> str:
    """
    Compute a SHA-256 hash of chunk content for deduplication.

    Why SHA-256?
      - Collision probability is astronomically low (~2^-128 for practical
        document sizes). Two chunks with the same hash are, for all practical
        purposes, identical.
      - It is deterministic: the same text always produces the same hash,
        enabling idempotent re-ingestion.
      - It is fast: hashing 10,000 chunks takes milliseconds.

    Why hash content and not file path?
      File paths change (renamed files, moved directories). Content hashes
      are stable as long as the text is unchanged. This means a renamed PDF
      that has identical content will not be re-embedded — the correct behavior.

    Args:
        text: Chunk page_content string.

    Returns:
        64-character hex string (SHA-256 digest).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_chunks(
    chunks: list[Document],
    existing_hashes: set[str],
) -> tuple[list[Document], set[str]]:
    """
    Filter out chunks whose content hash already exists in the vector store.

    This prevents:
      1. Redundant embedding API calls (cost reduction on re-ingestion)
      2. Duplicate vectors in the index (retrieval quality degradation)
      3. Index bloat from repeated ingestion of unchanged documents

    The existing_hashes set is provided by the caller (the ingestion orchestrator),
    which fetches it from the vector store metadata before calling this function.
    This keeps the deduplication logic pure — it does not call the vector store.

    Args:
        chunks: All chunks from the current ingestion run.
        existing_hashes: Set of content_hash values already in the vector store.

    Returns:
        Tuple of:
          - new_chunks: Chunks not yet in the index (need embedding + upsert)
          - new_hashes: Set of hashes for the new chunks (for logging/tracking)
    """
    new_chunks: list[Document] = []
    new_hashes: set[str] = set()
    duplicate_count = 0

    for chunk in chunks:
        content_hash = compute_content_hash(chunk.page_content)
        chunk.metadata["content_hash"] = content_hash  # persist hash in metadata

        if content_hash in existing_hashes:
            duplicate_count += 1
        else:
            new_chunks.append(chunk)
            new_hashes.add(content_hash)

    logger.info(
        "Deduplication complete",
        extra={
            "total_chunks": len(chunks),
            "new_chunks": len(new_chunks),
            "duplicates_skipped": duplicate_count,
            "dedup_rate_pct": round(duplicate_count / max(len(chunks), 1) * 100, 1),
        },
    )

    return new_chunks, new_hashes


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — Primary Entry Point
# ─────────────────────────────────────────────────────────────────────────────

class DocumentIngestionPipeline:
    """
    Orchestrates the full document ingestion workflow.

    Stages:
      1. Load        — read files from disk using format-appropriate loaders
      2. Normalize   — standardize metadata schema across all file types
      3. Clean       — strip noise, normalize whitespace
      4. Chunk       — split into embedding-ready units
      5. Deduplicate — skip chunks already in the vector store

    Usage:
        pipeline = DocumentIngestionPipeline()
        result = pipeline.run(
            data_dir=Path("data/pdfs"),
            existing_hashes=vector_store.get_all_hashes(),
        )
        # result.new_chunks are ready for embedding and upsert
    """

    def __init__(self):
        self.splitter = build_text_splitter()
        logger.info("DocumentIngestionPipeline initialized")

    def run(
        self,
        data_dir: Path,
        existing_hashes: Optional[set[str]] = None,
    ) -> "IngestionResult":
        """
        Execute the full ingestion pipeline.

        Args:
            data_dir: Directory containing source documents.
            existing_hashes: Content hashes already in the vector store.
                             Pass an empty set for first-time ingestion.
                             Pass None to skip deduplication entirely.

        Returns:
            IngestionResult with statistics and the list of new chunks.
        """
        if existing_hashes is None:
            existing_hashes = set()

        start_time = datetime.now(timezone.utc)

        # ── Stage 1: Load ──────────────────────────────────────────────────
        logger.info("Stage 1/5: Loading documents", extra={"dir": str(data_dir)})
        raw_documents = load_directory(data_dir)

        # ── Stage 2: Normalize metadata ────────────────────────────────────
        logger.info("Stage 2/5: Normalizing metadata")
        normalized_documents: list[Document] = []
        for doc in raw_documents:
            # Reconstruct file_path from the source metadata set by loaders
            source_path = Path(doc.metadata.get("source", "unknown"))
            normalized = normalize_metadata(doc, source_path)
            normalized_documents.append(normalized)

        # ── Stage 3: Clean text ────────────────────────────────────────────
        logger.info("Stage 3/5: Cleaning text")
        cleaned_documents: list[Document] = []
        for doc in normalized_documents:
            cleaned_content = clean_text(doc.page_content)
            # Skip pages that are empty after cleaning
            # (e.g. table of contents pages that are all noise)
            if len(cleaned_content.strip()) < 50:
                continue
            cleaned_documents.append(
                Document(
                    page_content=cleaned_content,
                    metadata=doc.metadata,
                )
            )

        # ── Stage 4: Chunk ─────────────────────────────────────────────────
        logger.info("Stage 4/5: Chunking documents")
        all_chunks = chunk_documents(cleaned_documents)

        # ── Stage 5: Deduplicate ───────────────────────────────────────────
        logger.info("Stage 5/5: Deduplicating chunks")
        new_chunks, new_hashes = deduplicate_chunks(all_chunks, existing_hashes)

        end_time = datetime.now(timezone.utc)
        duration_seconds = (end_time - start_time).total_seconds()

        result = IngestionResult(
            total_files_loaded=len(set(
                d.metadata.get("source") for d in raw_documents
            )),
            total_pages_loaded=len(raw_documents),
            total_chunks_created=len(all_chunks),
            new_chunks=new_chunks,
            duplicate_chunks_skipped=len(all_chunks) - len(new_chunks),
            duration_seconds=duration_seconds,
        )

        logger.info(
            "Ingestion pipeline complete",
            extra=result.to_log_dict(),
        )

        return result


class IngestionResult:
    """
    Value object carrying the output and statistics of an ingestion run.

    Separating statistics from the chunk list makes it easy to log results
    and check whether anything new was ingested before calling the embedding API.
    """

    def __init__(
        self,
        total_files_loaded: int,
        total_pages_loaded: int,
        total_chunks_created: int,
        new_chunks: list[Document],
        duplicate_chunks_skipped: int,
        duration_seconds: float,
    ):
        self.total_files_loaded = total_files_loaded
        self.total_pages_loaded = total_pages_loaded
        self.total_chunks_created = total_chunks_created
        self.new_chunks = new_chunks
        self.duplicate_chunks_skipped = duplicate_chunks_skipped
        self.duration_seconds = duration_seconds

    @property
    def has_new_content(self) -> bool:
        """True if there are new chunks that need to be embedded and upserted."""
        return len(self.new_chunks) > 0

    def to_log_dict(self) -> dict:
        return {
            "total_files": self.total_files_loaded,
            "total_pages": self.total_pages_loaded,
            "total_chunks": self.total_chunks_created,
            "new_chunks": len(self.new_chunks),
            "duplicates_skipped": self.duplicate_chunks_skipped,
            "duration_seconds": round(self.duration_seconds, 2),
        }

    def __repr__(self) -> str:
        return (
            f"IngestionResult("
            f"files={self.total_files_loaded}, "
            f"pages={self.total_pages_loaded}, "
            f"chunks={self.total_chunks_created}, "
            f"new={len(self.new_chunks)}, "
            f"skipped={self.duplicate_chunks_skipped}, "
            f"duration={self.duration_seconds:.2f}s)"
        )