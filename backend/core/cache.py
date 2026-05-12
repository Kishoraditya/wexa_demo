"""
backend/core/cache.py

Two-Level Cache Implementation
================================
Level 1 — Embedding Cache (covered in vector_store.py):
  Hash chunk content → store embedding vector to disk.
  Prevents re-calling the embedding model for unchanged document chunks.
  Implemented in: backend/services/vector_store.py → EmbeddingCache

Level 2 — Query Response Cache (this file):
  Hash normalized query → store full RAGResponse to disk.
  Prevents running the full pipeline for repeated identical queries.
  TTL: 1 hour (documentation is relatively static).

Production migration path:
  Current: diskcache (local disk, single-instance)
  Production: Redis (shared across multiple API instances)

  Why Redis in production?
    The diskcache is stored on the local filesystem of the API instance.
    In a multi-instance deployment (Auto Scaling Group with 3 instances),
    each instance has its own cache. Instance A caches query Q, but if
    the next request for Q routes to instance B, it's a cache miss.
    Redis (Elasticache) is shared across all instances — any instance
    can write and any instance can read the same cache entry.

  Migration cost:
    Replace diskcache.Cache with redis.Redis in this file.
    The cache key generation and serialization logic is identical.
    Zero changes to the routes or pipeline.

  Redis configuration for production:
    aws_elasticache_cluster = "my-rag-cache.abc123.cfg.use1.cache.amazonaws.com"
    REDIS_TTL_SECONDS = 3600
    REDIS_MAX_MEMORY = "2gb"
    REDIS_EVICTION_POLICY = "allkeys-lru"  # evict least-recently-used on full

Cache key design:
  Key = SHA-256(normalized_query + "|" + str(top_k) + "|" + filter_pillar)
  Normalization: strip whitespace, lowercase
  Rationale: "How does AWS handle failure?" and "how does aws handle failure?"
  should hit the same cache entry. top_k and filter_pillar are included because
  they affect the retrieval results and therefore the answer.

Cache invalidation:
  Current implementation: TTL-based (1 hour).
  On document update: manually flush cache or restart with fresh cache dir.
  Production: tag-based invalidation (flush all entries with tag "reliability"
  when the reliability.pdf is re-ingested). Requires Redis with tag support.

Author: Enterprise RAG Assistant
"""

import hashlib
import json
from typing import Optional, Any
from pathlib import Path

import diskcache

from backend.core.config import get_config
from backend.core.logging import get_logger
from backend.core.metrics import CACHE_HITS, CACHE_MISSES, CACHE_SIZE

logger = get_logger(__name__)
config = get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Cache Key Generation
# ─────────────────────────────────────────────────────────────────────────────

def make_cache_key(
    query: str,
    top_k: int,
    filter_pillar: Optional[str],
) -> str:
    """
    Generate a deterministic cache key for a query configuration.

    Key is SHA-256 of the canonical string representation.
    Using SHA-256 for uniform key length and collision resistance.

    Normalization decisions:
      - Lowercase: "Reliability" and "reliability" are the same intent
      - Strip: leading/trailing whitespace is not meaningful
      - top_k included: same query with k=3 vs k=5 returns different answers
      - filter_pillar included: same query filtered to "Security" vs unfiltered
        returns different answers

    Args:
        query: Raw user query string.
        top_k: Number of chunks to retrieve.
        filter_pillar: Optional pillar filter name.

    Returns:
        64-character hex string (SHA-256 digest).
    """
    canonical = (
        f"{query.strip().lower()}"
        f"|k={top_k}"
        f"|pillar={filter_pillar or 'all'}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Query Response Cache
# ─────────────────────────────────────────────────────────────────────────────

class QueryResponseCache:
    """
    Disk-backed cache for full RAG pipeline responses.

    Backed by diskcache.Cache which provides:
      - Thread-safe reads and writes
      - Automatic TTL enforcement
      - Disk persistence (survives process restart within TTL)
      - Size-limited eviction (LRU by default)

    Cache hit path performance:
      Cache read: ~5ms (diskcache with SSD)
      Saved: retrieval (~150ms) + generation (~2000ms) = ~2150ms
      Net improvement: ~2145ms per cache hit

    Cache miss behavior:
      On miss, the full pipeline runs and the result is written to cache.
      Total latency = pipeline latency + ~5ms cache write (negligible).
    """

    def __init__(self):
        if not config.cache.enabled:
            self._cache = None
            logger.info("Query response cache disabled in config")
            return

        cache_dir = Path(config.cache.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # diskcache.Cache configuration:
        #   size_limit: max cache size in bytes (1GB)
        #   disk_min_file_size: store values < 1024 bytes in SQLite instead of files
        #   eviction_policy: 'least-recently-used' — evict old queries on full
        self._cache = diskcache.Cache(
            directory=str(cache_dir),
            size_limit=1024 ** 3,   # 1 GB max cache size
            disk_min_file_size=1024,
            eviction_policy="least-recently-used",
        )

        logger.info(
            "Query response cache initialized",
            extra={
                "cache_dir": str(cache_dir),
                "backend": "diskcache",
                "ttl_seconds": config.cache.ttl_seconds,
                "production_alternative": "Redis (Elasticache) for multi-instance",
            },
        )

    def get(
        self,
        query: str,
        top_k: int,
        filter_pillar: Optional[str],
    ) -> Optional[dict]:
        """
        Look up a cached response.

        Args:
            query: User query string.
            top_k: Retrieval parameter (affects cache key).
            filter_pillar: Pillar filter (affects cache key).

        Returns:
            Cached response dict if found and not expired, else None.
            The caller is responsible for converting dict to RAGResponse.
        """
        if self._cache is None:
            return None

        cache_key = make_cache_key(query, top_k, filter_pillar)

        try:
            cached_value = self._cache.get(cache_key, default=None)

            if cached_value is not None:
                CACHE_HITS.labels(level="l2_response").inc()
                logger.debug(
                    "Query cache hit",
                    extra={"cache_key_prefix": cache_key[:8]},
                )
                return cached_value

            CACHE_MISSES.inc()
            return None

        except Exception as e:
            # Cache failures should never break the request
            # Treat as a miss and log the error
            logger.warning(
                "Cache read error — treating as miss",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            CACHE_MISSES.inc()
            return None

    def set(
        self,
        query: str,
        top_k: int,
        filter_pillar: Optional[str],
        response_dict: dict,
    ) -> None:
        """
        Store a response in the cache.

        Only call this for successful, non-refusal responses.
        Refusals may change if new documents are ingested.
        Low-confidence responses should still be cached — confidence
        is a quality signal, not a correctness signal.

        Args:
            query: User query string.
            top_k: Retrieval parameter.
            filter_pillar: Pillar filter.
            response_dict: RAGResponse.model_dump() output.
        """
        if self._cache is None:
            return

        cache_key = make_cache_key(query, top_k, filter_pillar)

        try:
            self._cache.set(
                cache_key,
                response_dict,
                expire=config.cache.ttl_seconds,
                # tag="response" would enable tag-based invalidation in Redis
                # diskcache supports tags but they require additional setup
            )

            # Update cache size gauge
            try:
                CACHE_SIZE.labels(cache_type="response").set(len(self._cache))
            except Exception:
                pass  # Metric update failure should not break cache write

            logger.debug(
                "Response cached",
                extra={
                    "cache_key_prefix": cache_key[:8],
                    "ttl_seconds": config.cache.ttl_seconds,
                },
            )

        except Exception as e:
            # Cache write failure is non-fatal — request already succeeded
            logger.warning(
                "Cache write error — response not cached",
                extra={"error": str(e)},
            )

    def invalidate(
        self,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_pillar: Optional[str] = None,
    ) -> bool:
        """
        Invalidate a specific cache entry or the entire cache.

        Args:
            query: If provided with top_k and filter_pillar, invalidate that
                   specific entry. If None, flush the entire cache.

        Returns:
            True if invalidation succeeded, False on error.
        """
        if self._cache is None:
            return False

        try:
            if query is not None and top_k is not None:
                cache_key = make_cache_key(query, top_k, filter_pillar)
                self._cache.delete(cache_key)
                logger.info(
                    "Cache entry invalidated",
                    extra={"cache_key_prefix": cache_key[:8]},
                )
            else:
                count = len(self._cache)
                self._cache.clear()
                logger.info(
                    "Cache flushed",
                    extra={"entries_cleared": count},
                )
            return True

        except Exception as e:
            logger.error(
                "Cache invalidation failed",
                extra={"error": str(e)},
            )
            return False

    def stats(self) -> dict:
        """Return cache statistics for the /health/ready endpoint."""
        if self._cache is None:
            return {"enabled": False}

        try:
            return {
                "enabled": True,
                "backend": "diskcache",
                "entries": len(self._cache),
                "size_bytes": self._cache.volume(),
                "ttl_seconds": config.cache.ttl_seconds,
                "production_backend": "redis_elasticache",
            }
        except Exception:
            return {"enabled": True, "backend": "diskcache", "error": "stats_unavailable"}

    def close(self) -> None:
        """Close the cache connection. Call during application shutdown."""
        if self._cache is not None:
            try:
                self._cache.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Cache Documentation
# ─────────────────────────────────────────────────────────────────────────────

CACHE_ARCHITECTURE_NOTES = """
TWO-LEVEL CACHE ARCHITECTURE
==============================

Level 1 — Embedding Cache (backend/services/vector_store.py → EmbeddingCache)
  What is cached:  Document chunk text → embedding vector
  Key:             SHA-256 of chunk content
  TTL:             No TTL (content-addressed, stable until chunk changes)
  Size:            ~3KB per entry (768-dim float32 vector as JSON)
  Backend:         JSON file on local disk
  Benefit:         Eliminates embedding model calls for unchanged chunks on re-ingestion
  Use case:        Called during POST /ingest pipeline runs
  Miss rate:       0% after first ingestion (all chunks are stable)
  Hit rate target: >95% (only new chunks miss)

Level 2 — Query Response Cache (this file → QueryResponseCache)
  What is cached:  Full RAGResponse (answer, sources, confidence, latency)
  Key:             SHA-256 of (normalized_query + top_k + filter_pillar)
  TTL:             3600 seconds (1 hour)
  Size:            ~5-10KB per entry (JSON-serialized RAGResponse)
  Backend:         diskcache (local disk) → Redis (production, multi-instance)
  Benefit:         Eliminates retrieval + generation for repeated queries
  Use case:        Called on every POST /generate request
  Expected pattern: Same query repeated (FAQ-style, team knowledge base)
  Hit rate target:  10-30% (depends on query diversity)

Cache hit path (L2):
  User query → hash → cache lookup → HIT → return cached response
  Latency: ~5ms
  Cost: $0 (no model calls)

Cache miss path (L2):
  User query → hash → cache lookup → MISS → full pipeline →
  embed (50ms) + retrieve (150ms) + generate (2000ms) + cache write (5ms)
  Latency: ~2200ms
  Cost: ~$0.001 (OpenAI) or $0 (fine-tuned)

Production Redis configuration:
  import redis
  r = redis.Redis(host=ELASTICACHE_HOST, port=6379, db=0)
  r.setex(cache_key, config.cache.ttl_seconds, json.dumps(response_dict))
  cached = r.get(cache_key)
  if cached:
      return json.loads(cached)
"""