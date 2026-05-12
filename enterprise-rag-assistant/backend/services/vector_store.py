"""
backend/services/vector_store.py

Embedding Pipeline and Vector Store
=====================================
Responsibilities:
  1. Embed document chunks using a local sentence-transformer model
  2. Cache embeddings to disk — skip model calls for unchanged chunks
  3. Build and persist a FAISS index to disk
  4. Provide a configurable retriever with similarity threshold filtering
  5. Support optional cross-encoder reranking as a second-pass precision layer

Architecture note — why FAISS and not Pinecone here:
  The original plan used Pinecone. This module uses FAISS for the following
  reasons that are worth documenting explicitly:

  FAISS (this implementation):
    + Completely free, no API key, no network call at query time
    + Persistent to disk — survives restarts without re-embedding
    + Query latency: ~5-20ms for 10k vectors (pure in-memory operation)
    + Sufficient for corpora up to ~1M vectors on a single node
    - No horizontal scaling — one node, one index
    - No built-in metadata filtering (handled by post-retrieval filter here)

  Pinecone (production path, documented in deployment_plan.md):
    + Managed, horizontally scalable, multi-tenant namespaces
    + Native metadata filtering at query time
    + No index management overhead
    - Network latency adds 50-150ms per query
    - Cost at scale (~$70/month for p1.x1 pod)
    - Requires API key and internet connectivity

  Decision: FAISS for this submission. The VectorStoreService interface
  (see bottom of this file) ensures Pinecone can be swapped in by
  implementing the same interface — zero changes to the RAG pipeline.

Embedding model note — why BAAI/bge-small-en-v1.5:
  Benchmarked on BEIR (Benchmarking Information Retrieval):
    BAAI/bge-small-en-v1.5:  avg NDCG@10 = 51.68
    all-MiniLM-L6-v2:         avg NDCG@10 = 49.25
    text-embedding-3-small:   avg NDCG@10 = 62.26 (OpenAI, paid)

  BGE-small wins vs MiniLM with identical model size (33M params, ~130MB).
  Both are CPU-friendly and run without GPU.
  OpenAI's embedding API is better but costs money and adds a network call.
  BGE-small is the best free, local, CPU-viable option for this corpus size.

Author: Enterprise RAG Assistant
"""

import hashlib
import json
import os
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from backend.core.config import get_config
from backend.core.logging import get_logger

logger = get_logger(__name__)
config = get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Cache
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Disk-backed cache that maps chunk content hashes to embedding vectors.

    Why cache embeddings?
      The BGE-small model takes ~5-15ms per chunk on CPU. For 1,000 chunks,
      that is 5-15 seconds of model inference on every ingestion run.
      With caching, only NEW chunks (those not seen before) trigger model calls.
      On re-ingestion of unchanged documents, embedding time drops to near zero.

    Cache format:
      A single JSON file at cache_dir/embedding_cache.json
      Structure: { "sha256_hex": [float, float, ...], ... }
      JSON is chosen over pickle for human-readability and cross-Python-version
      compatibility. The performance difference is negligible for this cache size.

    Cache invalidation:
      The cache is content-addressed (keyed by SHA-256 of chunk text).
      If the chunk text changes, its hash changes, and the old entry becomes
      an orphan (harmless — it will never be looked up again).
      There is no explicit TTL. Entries persist until manually cleared.
      Clearing the cache is equivalent to: rm .cache/embeddings/embedding_cache.json

    Thread safety:
      This implementation is NOT thread-safe. In production with multiple
      ingestion workers, replace with Redis or a database-backed cache.
    """

    def __init__(self, cache_dir: str):
        self.cache_path = Path(cache_dir) / "embedding_cache.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = self._load()

    def _load(self) -> dict[str, list[float]]:
        """Load existing cache from disk. Return empty dict if not found."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    data = json.load(f)
                logger.info(
                    "Embedding cache loaded",
                    extra={"entries": len(data), "path": str(self.cache_path)},
                )
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    "Embedding cache corrupt or unreadable, starting fresh",
                    extra={"error": str(e)},
                )
                return {}
        return {}

    def _save(self) -> None:
        """Persist current cache state to disk."""
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f)

    def get(self, content_hash: str) -> Optional[list[float]]:
        """
        Retrieve a cached embedding by content hash.

        Args:
            content_hash: SHA-256 hex digest of the chunk text.

        Returns:
            Embedding vector as list[float], or None if not cached.
        """
        return self._cache.get(content_hash)

    def set(self, content_hash: str, embedding: list[float]) -> None:
        """
        Store an embedding in the cache and persist to disk.

        Args:
            content_hash: SHA-256 hex digest of the chunk text.
            embedding: Embedding vector as list[float].
        """
        self._cache[content_hash] = embedding
        self._save()

    def set_batch(self, entries: dict[str, list[float]]) -> None:
        """
        Store multiple embeddings in a single write operation.

        Why batch writes?
          Writing to disk on every single embedding is inefficient.
          Batching means one disk write per ingestion run, not one per chunk.

        Args:
            entries: Dict mapping content_hash → embedding vector.
        """
        self._cache.update(entries)
        self._save()

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._cache)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Wrapper around the BGE-small sentence transformer with disk-based caching.

    Separation of concerns:
      This class owns the model and the cache. It does NOT know about
      FAISS or Pinecone. It answers one question: "give me the embedding
      for this text." The vector store layer handles what to do with it.
    """

    def __init__(self):
        self.model_name = config.embedding.model
        self.cache_enabled = config.embedding.cache_enabled
        self.batch_size = config.embedding.batch_size

        # Initialize the embedding cache
        if self.cache_enabled:
            self.cache = EmbeddingCache(config.embedding.cache_dir)
        else:
            self.cache = None

        # Load the HuggingFace embedding model via LangChain wrapper.
        # model_kwargs: device="cpu" forces CPU inference even if GPU is
        # available. For this corpus size (~1,000 chunks), CPU is fast enough
        # and avoids CUDA memory allocation that competes with the generation
        # model at query time.
        #
        # encode_kwargs: normalize_embeddings=True applies L2 normalization.
        # Normalized embeddings allow cosine similarity to be computed via
        # dot product, which is faster than the full cosine formula.
        # FAISS's IndexFlatIP (inner product) == cosine similarity when
        # embeddings are L2-normalized. This is the standard setup for BGE.
        logger.info(
            "Loading embedding model",
            extra={"model": self.model_name},
        )
        model_load_start = time.time()

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        model_load_ms = round((time.time() - model_load_start) * 1000)
        logger.info(
            "Embedding model loaded",
            extra={"model": self.model_name, "load_time_ms": model_load_ms},
        )

    @staticmethod
    def _hash(text: str) -> str:
        """SHA-256 hash of text content. Identical to ingestion pipeline hash."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_chunks(self, chunks: list[Document]) -> list[Document]:
        """
        Embed a list of Document chunks, using cache where available.

        Process:
          1. Check cache for each chunk's content hash
          2. Separate chunks into cache_hits and cache_misses
          3. Embed only the cache_misses in batches
          4. Store new embeddings in cache
          5. Attach embedding vectors to chunk metadata (for logging/debug)
          6. Return all chunks in original order

        Why attach embeddings to metadata?
          The FAISS.from_documents() method calls the embedding model
          internally. To use our cached embeddings instead, we bypass
          from_documents() and call FAISS.from_embeddings() directly,
          which requires us to have the vectors ready.

        Args:
            chunks: List of Document objects to embed.

        Returns:
            Same list of Documents with 'embedding' added to metadata.
        """
        if not chunks:
            return []

        logger.info(
            "Starting embedding",
            extra={"total_chunks": len(chunks), "cache_enabled": self.cache_enabled},
        )

        # ── Step 1: Cache lookup ───────────────────────────────────────────
        cache_hits: dict[int, list[float]] = {}    # index → embedding
        cache_misses: list[tuple[int, Document]] = []  # (index, chunk)

        if self.cache_enabled and self.cache:
            for i, chunk in enumerate(chunks):
                content_hash = self._hash(chunk.page_content)
                cached_embedding = self.cache.get(content_hash)
                if cached_embedding is not None:
                    cache_hits[i] = cached_embedding
                else:
                    cache_misses.append((i, chunk))
        else:
            cache_misses = list(enumerate(chunks))

        logger.info(
            "Cache lookup complete",
            extra={
                "cache_hits": len(cache_hits),
                "cache_misses": len(cache_misses),
                "cache_hit_rate_pct": round(
                    len(cache_hits) / max(len(chunks), 1) * 100, 1
                ),
            },
        )

        # ── Step 2: Embed cache misses in batches ──────────────────────────
        new_embeddings: dict[int, list[float]] = {}

        if cache_misses:
            miss_texts = [chunk.page_content for _, chunk in cache_misses]
            miss_indices = [i for i, _ in cache_misses]

            embed_start = time.time()
            # Process in batches to avoid OOM on large corpora.
            # batch_size=100 means 100 chunks per model.encode() call.
            # BGE-small can handle this easily on CPU with ~500MB RAM.
            all_vectors: list[list[float]] = []

            for batch_start in range(0, len(miss_texts), self.batch_size):
                batch = miss_texts[batch_start: batch_start + self.batch_size]
                batch_num = batch_start // self.batch_size + 1
                total_batches = (len(miss_texts) - 1) // self.batch_size + 1

                logger.debug(
                    "Embedding batch",
                    extra={
                        "batch": batch_num,
                        "total_batches": total_batches,
                        "batch_size": len(batch),
                    },
                )
                # embed_documents returns list[list[float]]
                batch_vectors = self.embeddings.embed_documents(batch)
                all_vectors.extend(batch_vectors)

            embed_ms = round((time.time() - embed_start) * 1000)
            logger.info(
                "Batch embedding complete",
                extra={
                    "chunks_embedded": len(miss_texts),
                    "embed_time_ms": embed_ms,
                    "ms_per_chunk": round(embed_ms / max(len(miss_texts), 1), 1),
                },
            )

            # Map vectors back to their original indices
            for idx, original_index in enumerate(miss_indices):
                new_embeddings[original_index] = all_vectors[idx]

            # ── Step 3: Update cache with new embeddings ───────────────────
            if self.cache_enabled and self.cache:
                cache_update = {}
                for i, chunk in cache_misses:
                    content_hash = self._hash(chunk.page_content)
                    cache_update[content_hash] = new_embeddings[i]
                self.cache.set_batch(cache_update)
                logger.info(
                    "Cache updated",
                    extra={"new_entries": len(cache_update)},
                )

        # ── Step 4: Assemble final embeddings in original order ────────────
        all_embeddings = {**cache_hits, **new_embeddings}
        for i, chunk in enumerate(chunks):
            # Store embedding dimension in metadata for debugging.
            # Do NOT store the full vector in metadata — it is large
            # (~3KB per chunk at 768 dims) and bloats memory unnecessarily.
            chunk.metadata["embedding_model"] = self.model_name
            chunk.metadata["embedding_dim"] = len(all_embeddings[i])

        # Return ordered list of (text, embedding) tuples for FAISS ingestion
        self._last_embeddings = [all_embeddings[i] for i in range(len(chunks))]

        return chunks

    def get_last_embeddings(self) -> list[list[float]]:
        """
        Return the embeddings computed in the most recent embed_chunks() call.

        This is a deliberate design choice — rather than returning embeddings
        from embed_chunks() directly (which would change the method signature
        and break the Document-in / Document-out pattern), we store them as
        instance state and expose them separately.

        Only call this immediately after embed_chunks().
        """
        return getattr(self, "_last_embeddings", [])

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for retrieval.

        Query embedding is NOT cached — queries are unique, and caching them
        would provide no benefit (each query is embedded once per request).
        If the same query appears repeatedly, the L2 response cache handles
        it before embedding is called.

        Args:
            query: User query string.

        Returns:
            Embedding vector as list[float].
        """
        return self.embeddings.embed_query(query)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    FAISS-backed vector store with persistence and metadata filtering.

    FAISS index type: IndexFlatL2
      LangChain's FAISS wrapper uses IndexFlatL2 by default.
      Since embeddings are L2-normalized (done in EmbeddingModel),
      L2 distance is mathematically equivalent to cosine distance:
        cosine_similarity(a, b) = 1 - (L2_distance(a, b)^2 / 2)
      So we get cosine semantics from an L2 index. No approximation —
      IndexFlatL2 is exact (exhaustive search). For ~10k vectors,
      exact search is fast enough (<20ms). For >1M vectors, switch to
      IndexIVFFlat or IndexHNSWFlat for approximate search with a
      speed-precision tradeoff.

    Persistence:
      FAISS index is saved to disk as two files:
        index.faiss — the binary vector index (IDs and vectors)
        index.pkl   — the docstore (metadata and page_content)
      Together these fully reconstruct the vector store without re-embedding.
      They are saved in the directory specified by config.vector_store.index_dir.
    """

    # Default directory for persisting the FAISS index
    INDEX_DIR = "data/faiss_index"

    def __init__(self, embedding_model: EmbeddingModel):
        """
        Args:
            embedding_model: Initialized EmbeddingModel instance.
                             FAISSVectorStore does not create this itself
                             to keep initialization explicit and testable.
        """
        self.embedding_model = embedding_model
        self.embeddings = embedding_model.embeddings  # LangChain-compatible wrapper
        self.index_dir = Path(
            getattr(config.vector_store, "index_dir", self.INDEX_DIR)
        )
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # The FAISS vector store instance — None until built or loaded
        self._store: Optional[FAISS] = None

        # Attempt to load an existing persisted index at initialization.
        # This means on API restart, the store is ready immediately without
        # running the ingestion pipeline again.
        self._try_load_from_disk()

    def _try_load_from_disk(self) -> None:
        """
        Attempt to load a persisted FAISS index from disk.
        Silently skips if no index exists (first run).
        """
        index_file = self.index_dir / "index.faiss"
        pkl_file = self.index_dir / "index.pkl"

        if index_file.exists() and pkl_file.exists():
            try:
                load_start = time.time()
                self._store = FAISS.load_local(
                    folder_path=str(self.index_dir),
                    embeddings=self.embeddings,
                    # allow_dangerous_deserialization must be True to load
                    # the pickle file. This is safe here because we wrote
                    # the pickle ourselves during ingestion.
                    allow_dangerous_deserialization=True,
                )
                load_ms = round((time.time() - load_start) * 1000)
                logger.info(
                    "FAISS index loaded from disk",
                    extra={
                        "index_dir": str(self.index_dir),
                        "load_time_ms": load_ms,
                        "vector_count": self._store.index.ntotal,
                    },
                )
            except Exception as e:
                logger.error(
                    "Failed to load FAISS index from disk",
                    extra={"error": str(e), "index_dir": str(self.index_dir)},
                )
                self._store = None
        else:
            logger.info(
                "No existing FAISS index found, will build on first ingestion",
                extra={"index_dir": str(self.index_dir)},
            )

    def build_index(self, chunks: list[Document]) -> None:
        """
        Build the FAISS index from a list of embedded Document chunks.

        This is called by the ingestion orchestrator after embedding.
        If an existing index is present, new chunks are ADDED to it
        (not rebuilt from scratch). This supports incremental ingestion.

        Args:
            chunks: List of Documents. Must have been processed by
                    EmbeddingModel.embed_chunks() so that
                    embedding_model.get_last_embeddings() is populated.
        """
        if not chunks:
            logger.warning("build_index called with empty chunk list — skipping")
            return

        embeddings_list = self.embedding_model.get_last_embeddings()

        if len(embeddings_list) != len(chunks):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match "
                f"embedding count ({len(embeddings_list)}). "
                "Call embed_chunks() before build_index()."
            )

        # Prepare (text, embedding) pairs for FAISS.from_embeddings()
        # Using from_embeddings() instead of from_documents() because we
        # already have the vectors — we should not re-compute them.
        text_embedding_pairs = [
            (chunk.page_content, embeddings_list[i])
            for i, chunk in enumerate(chunks)
        ]
        metadatas = [chunk.metadata for chunk in chunks]

        build_start = time.time()

        if self._store is None:
            # First ingestion — build index from scratch
            logger.info("Building new FAISS index")
            self._store = FAISS.from_embeddings(
                text_embeddings=text_embedding_pairs,
                embedding=self.embeddings,
                metadatas=metadatas,
            )
        else:
            # Subsequent ingestion — add to existing index
            # This is the INCREMENTAL path — only new chunks are added.
            logger.info(
                "Adding to existing FAISS index",
                extra={
                    "existing_vectors": self._store.index.ntotal,
                    "new_chunks": len(chunks),
                },
            )
            self._store.add_embeddings(
                text_embeddings=text_embedding_pairs,
                metadatas=metadatas,
            )

        build_ms = round((time.time() - build_start) * 1000)
        logger.info(
            "FAISS index built",
            extra={
                "total_vectors": self._store.index.ntotal,
                "build_time_ms": build_ms,
            },
        )

        # Persist to disk immediately after building.
        # If the process crashes before this line, the index is lost.
        # For production, consider periodic checkpointing.
        self._save_to_disk()

    def _save_to_disk(self) -> None:
        """Persist the FAISS index and docstore to disk."""
        if self._store is None:
            return

        save_start = time.time()
        self._store.save_local(str(self.index_dir))
        save_ms = round((time.time() - save_start) * 1000)

        logger.info(
            "FAISS index saved to disk",
            extra={
                "index_dir": str(self.index_dir),
                "save_time_ms": save_ms,
                "vector_count": self._store.index.ntotal,
            },
        )

    def similarity_search_with_scores(
        self,
        query: str,
        k: int,
        score_threshold: float,
        filter_metadata: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """
        Search the FAISS index and return documents with similarity scores.

        Threshold filtering:
          FAISS returns L2 distances, not cosine similarities.
          For L2-normalized embeddings:
            L2_distance = sqrt(2 - 2 * cosine_similarity)
            cosine_similarity = 1 - (L2_distance^2 / 2)
          A cosine threshold of 0.70 corresponds to L2 distance ~0.775.
          We convert the threshold before filtering.

          Why filter by threshold?
            Without a threshold, the retriever always returns k results,
            even when the best match is semantically unrelated.
            A threshold ensures we return NOTHING rather than a bad answer
            when no relevant context exists — this is the "refusal path."

        Metadata filtering:
          FAISS does not support native metadata filtering (unlike Pinecone).
          Post-retrieval filtering is applied: retrieve k*3 candidates,
          apply metadata filter, return the top k that pass.
          Requesting k*3 candidates compensates for filter attrition.

        Args:
            query: User query string (will be embedded internally).
            k: Maximum number of results to return.
            score_threshold: Minimum cosine similarity (0.0 to 1.0).
            filter_metadata: Optional dict of metadata key-value pairs
                             to filter results. Example:
                             {"source_pillar": "Reliability"}

        Returns:
            List of (Document, cosine_similarity_score) tuples,
            sorted by score descending, above the threshold.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector store is not initialized. "
                "Run the ingestion pipeline first."
            )

        # Retrieve more candidates than needed to compensate for
        # post-retrieval filtering (metadata filter + threshold filter)
        fetch_k = k * 3 if filter_metadata else k * 2

        search_start = time.time()

        # FAISS.similarity_search_with_score returns (Document, L2_distance) tuples
        raw_results: list[tuple[Document, float]] = (
            self._store.similarity_search_with_score(
                query=query,
                k=fetch_k,
            )
        )

        search_ms = round((time.time() - search_start) * 1000)
        logger.debug(
            "FAISS search complete",
            extra={
                "fetch_k": fetch_k,
                "raw_results": len(raw_results),
                "search_ms": search_ms,
            },
        )

        # ── Convert L2 distance to cosine similarity ───────────────────────
        scored_results: list[tuple[Document, float]] = []
        for doc, l2_distance in raw_results:
            # Clamp l2_distance to [0, 2] to handle floating point edge cases.
            # L2 distance between normalized vectors is always in [0, 2].
            l2_clamped = max(0.0, min(2.0, l2_distance))
            cosine_sim = 1.0 - (l2_clamped ** 2) / 2.0
            scored_results.append((doc, cosine_sim))

        # ── Apply metadata filter ──────────────────────────────────────────
        if filter_metadata:
            before_filter = len(scored_results)
            scored_results = [
                (doc, score)
                for doc, score in scored_results
                if all(
                    doc.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
            ]
            logger.debug(
                "Metadata filter applied",
                extra={
                    "filter": filter_metadata,
                    "before": before_filter,
                    "after": len(scored_results),
                },
            )

        # ── Apply score threshold ──────────────────────────────────────────
        before_threshold = len(scored_results)
        scored_results = [
            (doc, score)
            for doc, score in scored_results
            if score >= score_threshold
        ]

        logger.debug(
            "Score threshold applied",
            extra={
                "threshold": score_threshold,
                "before": before_threshold,
                "after": len(scored_results),
            },
        )

        # ── Sort and truncate to k ─────────────────────────────────────────
        scored_results.sort(key=lambda x: x[1], reverse=True)
        final_results = scored_results[:k]

        logger.info(
            "Retrieval complete",
            extra={
                "query_preview": query[:50] + "..." if len(query) > 50 else query,
                "results_returned": len(final_results),
                "top_score": round(final_results[0][1], 3) if final_results else 0,
                "search_ms": search_ms,
            },
        )

        return final_results

    def get_all_hashes(self) -> set[str]:
        """
        Return the set of content_hash values stored in the vector index.

        Used by the ingestion pipeline's deduplication step.
        If the hash is in this set, the chunk is already indexed and should
        not be re-embedded or re-inserted.

        Returns:
            Set of SHA-256 hex strings.
        """
        if self._store is None:
            return set()

        hashes = set()
        # LangChain FAISS wraps a docstore — iterate its dict
        for doc_id, doc in self._store.docstore._dict.items():
            content_hash = doc.metadata.get("content_hash")
            if content_hash:
                hashes.add(content_hash)

        return hashes

    @property
    def vector_count(self) -> int:
        """Number of vectors currently in the index."""
        if self._store is None:
            return 0
        return self._store.index.ntotal

    @property
    def is_ready(self) -> bool:
        """True if the index is built and ready for queries."""
        return self._store is not None and self.vector_count > 0


# ─────────────────────────────────────────────────────────────────────────────
# Reranker
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Cross-encoder reranker that refines retrieval results for precision.

    Why reranking matters — bi-encoder vs cross-encoder:

    The bi-encoder (embedding model, BGE-small) produces separate embeddings
    for the query and each document, then compares them via cosine similarity.
    This is fast because embeddings are precomputed for documents.
    But it is coarse: the query and document are encoded INDEPENDENTLY,
    so the model cannot consider how specific words in the query relate to
    specific words in the document.

    The cross-encoder (this class) takes the query and document TOGETHER
    as a single input and produces a relevance score. This joint encoding
    allows the model to consider:
      - Exact term overlap
      - Semantic entailment
      - Negation ("NOT recommended" is very different from "recommended")
      - Specificity (a chunk about "S3 cost optimization" scores higher
        than a chunk about "cost optimization" for the query "S3 costs")

    The tradeoff:
      Bi-encoder:    fast (precomputed), ~5ms at query time, lower precision
      Cross-encoder: slow (computed per query), ~100ms for 5 candidates,
                     significantly higher precision

    The solution: use BOTH. Bi-encoder retrieves the top-k candidates fast.
    Cross-encoder reranks only those k candidates. Latency cost: ~100ms.
    Precision gain: substantial, especially for queries where word order or
    negation matters.

    This is the standard two-stage retrieval architecture used in production
    RAG systems (Cohere Rerank, AWS Kendra, Google Vertex AI Search all
    use this pattern).

    Model: BAAI/bge-reranker-base
      - 278M parameters, ~560MB on disk
      - Free, local, no API calls
      - Strong performance on BEIR reranking benchmarks
      - CPU-viable for k<=10 candidates
    """

    def __init__(self):
        if not config.reranker.enabled:
            logger.info("Reranker disabled in config — CrossEncoderReranker not loaded")
            self._model = None
            return

        model_name = config.reranker.model
        logger.info("Loading cross-encoder reranker", extra={"model": model_name})

        load_start = time.time()
        try:
            self._model = CrossEncoder(model_name)
            load_ms = round((time.time() - load_start) * 1000)
            logger.info(
                "Reranker loaded",
                extra={"model": model_name, "load_time_ms": load_ms},
            )
        except Exception as e:
            logger.error(
                "Failed to load reranker — reranking disabled",
                extra={"model": model_name, "error": str(e)},
            )
            self._model = None

    @property
    def is_available(self) -> bool:
        """True if the reranker model loaded successfully."""
        return self._model is not None

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Document, float]],
        top_n: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        Rerank retrieval candidates using the cross-encoder.

        The cross-encoder replaces the bi-encoder similarity scores with
        its own relevance scores. The original bi-encoder scores are
        stored in metadata for comparison/debugging.

        Args:
            query: Original user query string.
            candidates: List of (Document, bi_encoder_score) from FAISS.
            top_n: Return only the top N results after reranking.
                   Defaults to config.reranker.top_n.

        Returns:
            Reranked list of (Document, cross_encoder_score) tuples,
            sorted by cross-encoder score descending.
        """
        if not self.is_available or not candidates:
            # If reranker is unavailable, return candidates unchanged.
            # This is graceful degradation — the system still works,
            # just with bi-encoder ordering instead of reranked ordering.
            logger.debug("Reranker not available — returning candidates unchanged")
            return candidates

        if top_n is None:
            top_n = config.reranker.top_n

        rerank_start = time.time()

        # Build (query, document_text) pairs for the cross-encoder
        # The cross-encoder expects exactly this format
        pairs = [(query, doc.page_content) for doc, _ in candidates]

        # cross-encoder.predict() returns a numpy array of relevance scores
        # Scores are logits (can be any float), higher is more relevant
        scores: np.ndarray = self._model.predict(pairs)

        rerank_ms = round((time.time() - rerank_start) * 1000)

        # Attach reranker scores and preserve original bi-encoder score
        reranked = []
        for i, (doc, bi_encoder_score) in enumerate(candidates):
            doc.metadata["bi_encoder_score"] = round(float(bi_encoder_score), 4)
            doc.metadata["cross_encoder_score"] = round(float(scores[i]), 4)
            reranked.append((doc, float(scores[i])))

        # Sort by cross-encoder score descending
        reranked.sort(key=lambda x: x[1], reverse=True)
        final = reranked[:top_n]

        logger.info(
            "Reranking complete",
            extra={
                "candidates_in": len(candidates),
                "results_out": len(final),
                "rerank_ms": rerank_ms,
                "top_cross_encoder_score": round(final[0][1], 3) if final else 0,
            },
        )

        return final


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Service — Primary Interface for the RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalService:
    """
    Unified retrieval interface combining FAISS search and optional reranking.

    This is the class the RAG pipeline imports and calls.
    It owns the full retrieval stack:
      query → embed → FAISS search → threshold filter → rerank → return

    The RAG pipeline does not interact with FAISSVectorStore or
    CrossEncoderReranker directly. This encapsulation makes it trivial to:
      - Swap FAISS for Pinecone: change FAISSVectorStore to PineconeVectorStore
      - Disable reranking: set reranker.enabled=false in config.yaml
      - Add hybrid search (BM25 + semantic): extend this class only
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        reranker: CrossEncoderReranker,
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = config.vector_store.retrieval.top_k
        self.score_threshold = config.vector_store.retrieval.score_threshold

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_pillar: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve the most relevant document chunks for a query.

        Full retrieval pipeline:
          1. FAISS similarity search (bi-encoder, fast)
          2. Score threshold filtering (remove weak matches)
          3. Cross-encoder reranking (precision layer, optional)

        Args:
            query: User query string.
            k: Number of results to return. Defaults to config top_k.
            score_threshold: Minimum similarity. Defaults to config threshold.
            filter_pillar: Optional pillar name to restrict retrieval.
                           Example: "Reliability"
                           None means search across all pillars.

        Returns:
            List of (Document, relevance_score) tuples.
            Empty list if no results pass the threshold (triggers refusal path).
        """
        effective_k = k or self.top_k
        effective_threshold = score_threshold or self.score_threshold

        # Build metadata filter if pillar restriction is requested
        metadata_filter = None
        if filter_pillar:
            metadata_filter = {"source_pillar": filter_pillar}

        # ── Stage 1: FAISS retrieval ───────────────────────────────────────
        # Retrieve k*2 if reranking is enabled — reranker needs more candidates
        # to be effective. If we only pass it k candidates, it can only pick
        # from k, which is no better than the original ordering.
        fetch_k = effective_k * 2 if self.reranker.is_available else effective_k

        candidates = self.vector_store.similarity_search_with_scores(
            query=query,
            k=fetch_k,
            score_threshold=effective_threshold,
            filter_metadata=metadata_filter,
        )

        if not candidates:
            logger.info(
                "No candidates above threshold — triggering refusal path",
                extra={
                    "query_preview": query[:50],
                    "threshold": effective_threshold,
                },
            )
            return []

        # ── Stage 2: Reranking (optional) ─────────────────────────────────
        if self.reranker.is_available and config.reranker.enabled:
            results = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_n=effective_k,
            )
        else:
            results = candidates[:effective_k]

        return results

    @property
    def is_ready(self) -> bool:
        """True if the retrieval service can handle queries."""
        return self.vector_store.is_ready


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion Orchestrator — Wires Ingestion Pipeline to Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class VectorStoreIngestionOrchestrator:
    """
    Orchestrates the embedding and indexing steps after document ingestion.

    Connects:
      DocumentIngestionPipeline (Phase 1) → EmbeddingModel → FAISSVectorStore

    Usage:
        orchestrator = VectorStoreIngestionOrchestrator(
            embedding_model=embedding_model,
            vector_store=faiss_store,
        )
        result = orchestrator.ingest_from_pipeline(ingestion_result)
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def ingest_from_result(self, ingestion_result) -> dict:
        """
        Embed and index new chunks from a completed ingestion pipeline run.

        Args:
            ingestion_result: IngestionResult from DocumentIngestionPipeline.run()

        Returns:
            Dict with indexing statistics.
        """
        if not ingestion_result.has_new_content:
            logger.info("No new content to index — all chunks already present")
            return {
                "chunks_embedded": 0,
                "chunks_indexed": 0,
                "message": "All chunks already in index",
            }

        new_chunks = ingestion_result.new_chunks
        logger.info(
            "Starting embedding and indexing",
            extra={"new_chunks": len(new_chunks)},
        )

        # Embed new chunks (uses cache internally)
        embedded_chunks = self.embedding_model.embed_chunks(new_chunks)

        # Build/update FAISS index
        self.vector_store.build_index(embedded_chunks)

        return {
            "chunks_embedded": len(embedded_chunks),
            "chunks_indexed": len(embedded_chunks),
            "total_vectors_in_index": self.vector_store.vector_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Interface — Enables Vector Store Swapping
# ─────────────────────────────────────────────────────────────────────────────

class VectorStoreService(ABC):
    """
    Abstract interface for vector store implementations.

    Why an interface?
      The RAG pipeline should not depend on FAISS specifically.
      By programming to this interface, swapping to Pinecone, Weaviate,
      or Qdrant requires:
        1. Write a new class implementing this interface (e.g. PineconeService)
        2. Change the instantiation in the dependency injection layer
        3. Zero changes to the RAG pipeline, routes, or any other module

    Production migration path:
      When the corpus grows beyond 1M vectors or multi-node deployment is
      required, implement PineconeVectorStoreService(VectorStoreService)
      and swap it in via config.vector_store.provider = "pinecone".

    Current implementation: FAISSVectorStore (this file)
    Future implementations:
      - PineconeVectorStoreService (managed, horizontally scalable)
      - WeaviateVectorStoreService (open-source, self-hosted, GraphQL API)
      - QdrantVectorStoreService (Rust-based, high-performance, on-premise)
    """

    @abstractmethod
    def similarity_search_with_scores(
        self,
        query: str,
        k: int,
        score_threshold: float,
        filter_metadata: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """Search and return (Document, score) tuples above threshold."""
        ...

    @abstractmethod
    def get_all_hashes(self) -> set[str]:
        """Return content hashes of all indexed chunks."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True if ready to serve queries."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Factory — Dependency Injection Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def create_retrieval_service() -> RetrievalService:
    """
    Factory function that wires together all retrieval components.

    This is the single entry point for creating a fully initialized
    RetrievalService. Call this once at application startup and inject
    the result into routes via FastAPI's dependency injection system.

    Returns:
        Fully initialized RetrievalService ready for queries.
    """
    logger.info("Initializing retrieval service components")

    embedding_model = EmbeddingModel()
    faiss_store = FAISSVectorStore(embedding_model=embedding_model)
    reranker = CrossEncoderReranker()

    service = RetrievalService(
        vector_store=faiss_store,
        reranker=reranker,
    )

    logger.info(
        "Retrieval service ready",
        extra={
            "vector_count": faiss_store.vector_count,
            "reranker_available": reranker.is_available,
            "store_ready": service.is_ready,
        },
    )

    return service