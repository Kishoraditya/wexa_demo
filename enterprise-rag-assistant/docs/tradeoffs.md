# Architectural Tradeoffs — Enterprise RAG Assistant

**Purpose:** This document records every significant architectural decision,
the alternative that was rejected, what was traded away, and the conditions
under which the decision would be made differently.

This is not a justification document. It is an honest record of tradeoffs.
Every choice optimizes for some dimension at the cost of another.
A reviewer reading this should come away understanding not just *what* was built
but *why* — and under what circumstances a different choice would be correct.

---

## Tradeoff 1: FAISS vs Pinecone

### Decision Made
**FAISS** — local, in-memory vector store persisted to disk.

### What Was Built
At ingestion time, document chunks are embedded and stored in a FAISS
`IndexFlatL2` index saved to `data/faiss_index/`. At query time, the index
is loaded into memory and searched in ~10-30ms. The index file is ~10MB for
the current corpus (~843 vectors at 384 dimensions).

### What Was Traded Away

| Dimension | FAISS (chosen) | Pinecone (rejected) |
|---|---|---|
| Cost | $0 | $70/month (p1.x1 pod) |
| Setup | No API key, no account | API key, account, index configuration |
| Query latency | 10-30ms (in-memory) | 50-150ms (network round-trip) |
| Horizontal scaling | ❌ Single node only | ✅ Managed, multi-node |
| Metadata filtering | Post-retrieval only | Native at query time |
| Multi-tenancy | Manual (namespace by file) | Built-in namespaces |
| Index persistence | Local disk | Managed cloud (always available) |
| Multi-instance sharing | ❌ Each instance has own copy | ✅ Shared single index |

**FAISS limitations that matter in production:**
In a multi-instance deployment (3 API servers), each server loads its own copy
of the index. A re-ingestion run updates one copy; the others remain stale until
they reload. This creates a consistency window where different servers serve
different index versions. The current architecture mitigates this with an atomic
S3 pointer swap (see `docs/deployment_plan.md`), but it adds operational complexity
that Pinecone eliminates entirely.

### When to Choose Differently

**Choose Pinecone instead when:**
- Corpus exceeds ~500k vectors (FAISS exact search degrades with index size)
- Multi-instance deployment requires a shared, consistent index
- Metadata filtering at query time is required (e.g., filtering by user, date, category)
- Team does not want to manage index versioning and rollout

**Stay with FAISS when:**
- Corpus is small and stable (<100k vectors)
- Cost is a hard constraint
- Network latency to a managed service is unacceptable
- Data residency requirements prohibit external services

**Approximate middle ground:**
For large corpora with a cost constraint, FAISS with `IndexIVFFlat` (approximate
search) reduces query time from O(n) to O(√n) at the cost of ~5% recall loss.
This extends FAISS's practical ceiling to ~10M vectors.

---

## Tradeoff 2: RAG vs Pure Fine-Tuning

### Decision Made
**RAG** as the primary architecture, with fine-tuning as a behavioral layer on top.

### What Was Built
The system retrieves relevant passages at query time and injects them into the
prompt. The model answers from the retrieved context, not from parametric memory.
Fine-tuning (Phase 4) adapts the model's *behavior* (follow instructions, cite
sources, refuse when unsure) — not its *knowledge*.

### Why Not Pure Fine-Tuning

A model fine-tuned directly on AWS documentation Q&A would memorize the
documentation content in its weights. This approach has fundamental problems
for this use case:

**1. Knowledge staleness.** AWS updates the Well-Architected Framework regularly.
With pure fine-tuning, every update requires a new training run (~90 minutes)
and a new deployment. With RAG, updating the knowledge base takes ~5 minutes
(re-ingest the updated PDF).

**2. Source citation is impossible.** A model that answers from parametric
memory cannot tell you *where* in the documentation its answer came from.
RAG returns the exact chunks, enabling source citation down to the page number.

**3. Hallucination is uncontrollable.** A fine-tuned model will confabulate
answers for questions not well-covered in training data. RAG can refuse when
retrieval finds no relevant context.

**4. Training cost and iteration speed.** Fine-tuning requires a GPU, a training
dataset, and hours of compute per iteration. Changing the answer to a specific
question requires a new training run. With RAG, changing the answer requires
editing the source PDF and re-ingesting.

### What RAG Trades Away

**RAG is slower.** Retrieval adds 70-200ms to every request. Pure fine-tuning
answers immediately without retrieval.

**RAG requires a retrieval quality floor.** If retrieval fails to surface the
relevant passage, the model cannot answer correctly even if the information is
in the corpus. Pure fine-tuning (with the right training data) doesn't have
this failure mode.

**RAG requires maintained infrastructure.** The vector store, embedding model,
and index must all be operational. Fine-tuning's knowledge is in the model
weights — nothing can go down except the model server.

### When to Choose Differently

**Choose pure fine-tuning when:**
- The knowledge domain is closed and updates infrequently (< once/year)
- Source citation is not required
- Response latency is critical (<500ms)
- The training dataset is high quality and domain-specific

**The hybrid approach is almost always correct for enterprise knowledge bases:**
Use RAG for knowledge grounding + fine-tuning for behavioral alignment.
This is exactly what the system implements: RAG for factual grounding,
fine-tuning for consistent instruction following.

---

## Tradeoff 3: LoRA vs Full Fine-Tuning

### Decision Made
**LoRA (Low-Rank Adaptation)** via QLoRA (4-bit quantization) on Colab T4.

### What Was Built
A LoRA adapter with rank 16 targeting all linear projection layers in Phi-3-mini.
The adapter contains ~4.2M trainable parameters (0.11% of the base model's 3.8B).
The base model weights are frozen. Training runs on a free Colab T4 in ~90 minutes.

### What Was Traded Away

| Dimension | LoRA/QLoRA (chosen) | Full Fine-Tuning (rejected) |
|---|---|---|
| VRAM required | ~4GB | ~30GB+ |
| Training time | 90 min (Colab T4) | 4-8 hours (A100 required) |
| Training cost | $0 (Colab free tier) | ~$20-50 (A100 cloud instance) |
| Adapter size | ~60MB | ~7.5GB (full model weights) |
| Quality vs full FT | ~92-95% equivalent | Baseline |
| Deployment flexibility | Base + adapter separately | Single merged weights |
| Catastrophic forgetting | Minimal (frozen base) | Risk if LR too high |

**The quality tradeoff is acceptable for this use case:**
Full fine-tuning updates all 3.8B parameters, allowing the model to make
deeper modifications to its internal representations. LoRA updates only 4.2M
parameters, which is sufficient for behavioral adaptation (instruction following,
output format, citation style) but insufficient for deep knowledge acquisition.
Since knowledge comes from RAG, deep knowledge acquisition is not needed.

### What the Numbers Mean

LoRA rank 16 means each adapter layer approximates the weight update ΔW as:
`ΔW = B × A` where `A ∈ R^(16 × d_in)` and `B ∈ R^(d_out × 16)`.

The rank is the bottleneck dimension. Rank 16 captures 16 independent "directions"
of behavioral change. For instruction-following tasks on small datasets (~3k examples),
rank 16 provides sufficient capacity. For knowledge-intensive domain adaptation
on large datasets, rank 64+ is warranted.

Alpha=32 (2×rank) is the standard scaling convention. The effective learning rate
for the adapter is `(alpha/rank) × lr = 2.0 × 2e-4 = 4e-4`. This normalization
means the effective LR is independent of rank choice — changing rank doesn't
require changing the base LR.

### When to Choose Differently

**Choose full fine-tuning when:**
- VRAM and compute cost are not constraints
- The task requires deep architectural modification (not just behavioral)
- The dataset is very large (>100k examples) — LoRA may underfit

**Choose a higher LoRA rank (r=32, r=64) when:**
- Training dataset is larger (>10k examples)
- Task requires more expressive behavioral modification
- VRAM is sufficient (higher rank = more memory)

**Choose adapter merging (merge LoRA into base weights) when:**
- Inference latency is critical — merged weights skip adapter matrix multiplication
- Multiple adapters are not needed
- The adapter will not be updated after deployment

The current system does NOT merge because: (a) adapter updates without base
re-download are a key deployment feature, (b) the latency difference is <5ms
for the matrix sizes in Phi-3-mini.

---

## Tradeoff 4: Synchronous vs Asynchronous Generation

### Decision Made
**Synchronous generation** for the development submission, with async documented
for production via SQS.

### What Was Built
The `/generate` endpoint uses `asyncio.to_thread()` to run the synchronous
model inference in a thread pool. The FastAPI event loop is not blocked, but
the HTTP connection stays open until generation completes (2-5 seconds).
The client receives a single response after the full generation is done.

### What Was Traded Away

**Synchronous (chosen for v1):**
- ✅ Simple to implement and debug
- ✅ Single HTTP request/response cycle
- ✅ Client code is simple (standard POST request)
- ❌ HTTP connection stays open for 2-5 seconds
- ❌ Client sees nothing until generation is complete
- ❌ Load balancer timeout must be > generation time (not always configurable)
- ❌ Does not scale to high concurrency without many workers

**Asynchronous with SQS + polling (documented production architecture):**
- ✅ HTTP connection closed immediately after queuing (fast ACK)
- ✅ Server-side resources released while GPU processes
- ✅ Natural back-pressure via queue depth
- ✅ Separates API tier from inference tier (different scaling)
- ❌ Client must poll for result (2 round trips minimum)
- ❌ Result expiration logic needed (what if client disconnects?)
- ❌ More infrastructure (SQS, Redis result store)
- ❌ Harder to debug (request spans multiple services)

**Streaming via SSE (ideal but not implemented):**
- ✅ Time-to-first-token ~500ms (strong UX improvement)
- ✅ HTTP connection provides back-pressure naturally
- ✅ No polling needed
- ❌ Client must handle streaming response
- ❌ Caching a streaming response requires buffering the full stream
- ❌ LangChain streaming callbacks add implementation complexity

### When to Choose Differently

**Use synchronous for:**
- Internal tools where UX latency is acceptable
- Simple architectures without separate inference services
- Prototype/MVP stages

**Use async (SQS) for:**
- Production systems with variable load
- When API and inference need to scale independently
- When inference time is unpredictable (can't size timeout reliably)

**Use SSE streaming for:**
- Customer-facing products where perceived responsiveness matters
- When time-to-first-token is more important than total latency
- When the generation model supports streaming natively

---

## Tradeoff 5: Exact-Match Cache vs Semantic Cache

### Decision Made
**Exact-match cache** — SHA-256 hash of normalized(query + top_k + filter_pillar).

### What Was Built
The query response cache is keyed by a deterministic hash of the request
parameters. "What is least privilege?" and "What is least privilege access?"
are two different cache keys — a cache miss on the second despite semantic
equivalence.

### Why Not Semantic Cache

Semantic caching would embed the query, search a small vector index of
cached query embeddings, and return a cache hit if cosine similarity > 0.95.

**Problem 1: Cache lookup requires an embedding call.**
Checking the semantic cache requires embedding the query (~20ms). For a cache
miss, this adds 20ms of latency with no benefit. For an exact-match cache miss,
no embedding is needed. Since most queries miss the cache (first visit), the
embedding call amortizes poorly.

**Problem 2: Similarity threshold is a new parameter to tune.**
What constitutes "semantically equivalent enough to return the same answer"?
0.95 cosine similarity? 0.90? Too low → serve cached answers for semantically
different questions (wrong answer). Too high → effectively exact matching.
The threshold requires empirical tuning per domain.

**Problem 3: Cache coherence complexity.**
When an exact-match cache entry is invalidated (TTL expiry or manual flush),
it's straightforward: delete the key. When a semantic cache entry is invalidated,
what other entries in the semantic neighborhood should also be invalidated?
This requires tracking relationships between cache entries.

### What Exact-Match Trades Away

**Missed cache hits for paraphrased queries.** Two engineers asking the same
question in different words both incur full pipeline cost. Estimated miss rate
from paraphrasing: ~20-30% of "semantically equivalent" queries are phrased
differently enough to miss.

**Sensitivity to minor variations.** "what is least privilege" and
"What is least privilege?" are the same question (normalization handles case
and whitespace) but "What is least privilege?" and "What does least privilege
mean?" miss each other despite being functionally equivalent.

### Current Mitigations

- **Normalization:** lowercase, strip whitespace, canonical parameter serialization
- **TTL of 1 hour:** reduces the window during which paraphrased queries both
  need to go through the full pipeline

### When to Choose Differently

**Choose semantic cache when:**
- Query volume is high enough to justify the embedding overhead
- Query set has high paraphrase diversity (FAQ chatbots, customer support)
- Cache hit rate improvement >20% is expected
- An embedding infrastructure is already present (marginal cost is low)

**Implementation path from exact to semantic:**
1. Keep exact-match as L1 cache (no embedding cost)
2. Add semantic cache as L2 (embed only on L1 miss)
3. On L2 hit: return cached response and also write to L1 (promotes to exact match)
4. This hybrid approach reduces the embedding overhead of semantic caching significantly

---

## Tradeoff 6: Self-Hosted Embeddings vs OpenAI Embeddings

### Decision Made
**Self-hosted BGE-small-en-v1.5** via HuggingFace sentence-transformers.

### What Was Built
The embedding model runs locally in the same process as the FAISS index.
Model size: ~130MB on disk. Inference: ~20-60ms per batch on CPU.
Dimensions: 384 (compared to OpenAI text-embedding-3-small at 1536).

### Benchmark Comparison

| Model | BEIR NDCG@10 | Dimensions | Cost | Latency | Privacy |
|---|---|---|---|---|---|
| BGE-small-en-v1.5 (chosen) | 51.68 | 384 | $0 | 20-60ms (CPU) | ✅ Local |
| all-MiniLM-L6-v2 | 49.25 | 384 | $0 | 20-50ms (CPU) | ✅ Local |
| text-embedding-3-small | 62.26 | 1536 | $0.020/1M tokens | 50-150ms (API) | ❌ External |
| text-embedding-3-large | 64.56 | 3072 | $0.130/1M tokens | 100-300ms (API) | ❌ External |

### What Self-Hosted Trades Away

**Retrieval quality.** OpenAI's `text-embedding-3-small` achieves 62.26 NDCG@10
vs BGE-small's 51.68 — a 20% improvement in retrieval quality. This gap translates
directly to better context precision and context recall in RAGAS scores.

**Operational simplicity.** The self-hosted model must be loaded at startup
(~2 seconds), kept in memory, and managed as part of the service. OpenAI's
embedding API requires no local infrastructure.

**Batch efficiency.** OpenAI's API accepts batches of up to 2048 inputs.
The local model processes batches of 100 (configurable) with similar throughput
but higher memory usage for large batches.

### What Self-Hosted Gains

**Zero cost at scale.** At 10,000 requests/day with ~500 tokens per query:
- BGE-small: $0/month
- OpenAI text-embedding-3-small: 10k × 500 tokens × $0.020/1M = $0.10/day = $3/month

For this corpus size, the cost difference is negligible. For a 100M token/month
embedding workload, the difference is $2,000/month.

**No external dependency.** BGE-small works offline, with no API key, no network
call, and no external service availability dependency. Embedding never fails due
to OpenAI outages.

**Data privacy.** Document content is never sent to an external service. For
enterprise use cases with confidential documentation, this is often a hard
requirement.

### When to Choose Differently

**Choose OpenAI embeddings when:**
- Retrieval quality is critical and the 20% improvement translates to
  meaningful business value (e.g., legal document review, medical Q&A)
- The corpus is small enough that API cost is negligible
- Data privacy requirements do not apply
- Operational simplicity is prioritized over independence

**Stay with self-hosted when:**
- Data cannot leave the network perimeter (enterprise compliance)
- Cost at scale is a constraint
- Low-latency retrieval is required (no network round-trip)
- The retrieval quality difference does not justify the cost for the use case

**Hybrid approach for high-quality production:**
Use OpenAI embeddings for ingestion (one-time cost, best quality vectors)
and BGE-small for query embedding (per-request, cost-sensitive path).
This is not currently implemented but would reduce per-request cost while
maintaining high-quality index vectors.

---

## Summary: Decision Matrix

| Decision | Chosen | Key Reason | Scale Trigger to Switch |
|---|---|---|---|
| Vector store | FAISS | Zero cost, sufficient for corpus | >500k vectors or multi-instance |
| RAG vs FT | RAG + FT | Knowledge freshness + grounding | N/A — always hybrid |
| FT method | QLoRA r=16 | T4-compatible, 92% quality | Need deeper adaptation |
| Generation sync | Synchronous | Simplicity | >100 concurrent users |
| Cache type | Exact-match | No embedding overhead | >30% cache miss from paraphrasing |
| Embeddings | Self-hosted BGE | Zero cost, private | Quality gap costs real revenue |

Every decision in this table was made for the current scale and constraints.
At 10× the scale, 4 of these 6 decisions would be revisited.
That is not a flaw in the design — it is correct engineering judgment.
The right architecture for today's scale is not the right architecture for 10×.
Building for 10× scale on day one is premature optimization.