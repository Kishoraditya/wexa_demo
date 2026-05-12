# Enterprise Cloud Architecture Assistant

> A production-grade Retrieval-Augmented Generation (RAG) system for querying
> the AWS Well-Architected Framework. Fine-tuned Phi-3-mini generation layer
> with automatic OpenAI fallback, semantic grounding verification, and full
> observability stack.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-orange.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Stack and Rationale](#3-stack-and-rationale)
4. [Setup Instructions](#4-setup-instructions)
5. [API Reference](#5-api-reference)
6. [Evaluation Results](#6-evaluation-results)
7. [Fine-Tuning](#7-fine-tuning)
8. [Deployment Plan Summary](#8-deployment-plan-summary)
9. [Known Limitations](#9-known-limitations)
10. [Future Improvements](#10-future-improvements)

---

## 1. Project Overview

### Problem

Engineers and cloud architects working with AWS regularly need to reference the
Well-Architected Framework — a body of best practices spanning six pillars that
collectively runs to several hundred pages. The current experience is manual:
open a PDF, search for a keyword, read surrounding context, cross-reference other
pillars. This is slow and does not scale across a team making simultaneous
architecture decisions.

### Solution

A conversational AI assistant that:
- Retrieves precise, grounded answers from the official AWS Well-Architected
  documentation
- Surfaces the exact source passages that informed each answer
- Refuses to answer when no relevant context exists (no hallucination)
- Provides a confidence signal (HIGH/MEDIUM/LOW) on every response
- Runs with a fine-tuned local model or falls back to OpenAI transparently

### Scope

**In scope:** Questions about the six AWS Well-Architected pillars
(Operational Excellence, Security, Reliability, Performance Efficiency,
Cost Optimization, Sustainability).

**Out of scope:** Code generation, live AWS account access, multi-turn
conversation memory, non-AWS cloud platforms.

### User Journey

```
Engineer asks →  System retrieves  →  System generates  →  Engineer sees answer
"How does AWS       Top-5 relevant      Grounded response    + source citations
 recommend          chunks from         with confidence      + confidence level
 handling           FAISS index         assessment           + latency metrics
 failure?"          (< 200ms)           (2-5s)               (< 50ms if cached)
```

---

## 2. Architecture

### System Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════╗
║              INGESTION FLOW  (Offline / On Doc Change)          ║
╚══════════════════════════════════════════════════════════════════╝

  [6 AWS Pillar PDFs]  →  [PyPDFLoader]  →  [RecursiveTextSplitter]
                          metadata norm.     512 tokens, 50 overlap
                          source/page/pillar dedup by SHA-256 hash
                                 │
                    [Embedding Cache] ←── skip if hash exists
                    diskcache on disk
                         │  cache miss
                         ▼
                  [BGE-small-en-v1.5]  →  [FAISS Index]
                  HuggingFace local         persisted to disk
                  384-dim vectors           + metadata store


╔══════════════════════════════════════════════════════════════════╗
║                 QUERY FLOW  (Online / Per Request)              ║
╚══════════════════════════════════════════════════════════════════╝

  [User / Streamlit]                              [Response]
        │                                              ▲
        ▼                                              │
  [FastAPI :8000]─→[Input Guardrails]                 │
  request_id        max_len, injection,               │
  assigned          PII check                         │
        │                                              │
        ▼                                              │
  [L2 Cache]──── HIT (<50ms) ──────────────────────→  │
  diskcache          ↑ cache_hit=true                  │
  query hash         │                                 │
        │ MISS       │                                 │
        ▼            │                                 │
  [BGE-small embed]  │                                 │
  ~20-60ms           │                                 │
        │            │                                 │
        ▼            │                                 │
  [FAISS Retriever]  │    score < 0.70                 │
  top-5, score>0.70 ─┤─── NO RESULTS →[REFUSAL]──→    │
  ~10-30ms           │                                 │
        │            │                                 │
        ▼            │                                 │
  [CrossEncoder]     │                                 │
  bge-reranker-base  │                                 │
  ~40-120ms          │                                 │
        │            │                                 │
        ▼            │                                 │
  [Prompt Builder]   │                                 │
  context injection  │                                 │
  cite/refuse/conf   │                                 │
        │            │                                 │
        ▼            │                                 │
  [Model Router]     │                                 │
  ┌─────┴─────┐      │                                 │
  ▼           ▼      │                                 │
[Phi-3-mini] [GPT-4o-mini] (<5s)                      │
+LoRA adapter fallback if                              │
HF Hub fails  primary fails                            │
  └─────┬─────┘      │                                 │
        ▼            │                                 │
  [Output Guardrails]│                                 │
  grounding check    │                                 │
  PII redaction      │                                 │
  confidence parse   │                                 │
        │            │                                 │
        ▼            │                                 │
  [Cache Write + Observability]────────────────────→   │
  loguru JSON   Prometheus /metrics                    │
```

*See [`docs/architecture.png`](docs/architecture.png) for the full Excalidraw diagram.*

### Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Vector store | FAISS (local) | Free, persistent, sufficient for 6 PDFs (~800 chunks) |
| Embedding model | BGE-small-en-v1.5 | Free, CPU-friendly, outperforms MiniLM on BEIR |
| Reranker | CrossEncoder bge-reranker-base | Precision layer — joint query-document scoring |
| Generation (primary) | Phi-3-mini + LoRA adapter | Fine-tuned, self-hosted, MIT license |
| Generation (fallback) | OpenAI GPT-4o-mini | High availability, transparent activation |
| Hallucination protection | Semantic grounding check | Cosine similarity: answer vs retrieved context |
| Caching | Two-level (embedding + response) | Eliminates redundant model calls |

*Full decision rationale: [`docs/tradeoffs.md`](docs/tradeoffs.md)*

---

## 3. Stack and Rationale

| Layer | Technology | Version | Rationale |
|---|---|---|---|
| **API Framework** | FastAPI | 0.111 | Async, typed, Pydantic validation, OpenAPI auto-docs |
| **RAG Framework** | LangChain | 0.2 | Composable retrieval primitives, wide ecosystem |
| **Vector Store** | FAISS | 1.8 | Free, in-memory, persistent to disk, no API key |
| **Embedding Model** | BGE-small-en-v1.5 | — | Free local inference, BEIR NDCG@10 = 51.68 |
| **Reranker** | BGE-reranker-base | — | CrossEncoder joint scoring, free local inference |
| **Primary LLM** | Phi-3-mini-4k-instruct + LoRA | 3.8B | MIT license, T4-compatible, strong instruction following |
| **Fallback LLM** | OpenAI GPT-4o-mini | — | High quality, low cost ($0.00060/1k tokens output) |
| **Fine-tuning** | QLoRA via unsloth | — | 40% VRAM reduction, 2× speedup vs vanilla PEFT |
| **UI** | Streamlit | 1.36 | Rapid prototyping, source card rendering |
| **Logging** | loguru | 0.7 | JSON structured logs, rotating file sink |
| **Metrics** | Prometheus + FastAPI Instrumentator | — | /metrics endpoint, custom RAG-specific counters |
| **Caching** | diskcache | 5.6 | Thread-safe, TTL enforcement, Redis-compatible interface |
| **Evaluation** | RAGAS | 0.1 | Industry-standard RAG evaluation framework |
| **Testing** | pytest + httpx | — | Unit + integration tests, mocked services |

---

## 4. Setup Instructions

### Prerequisites

- Python 3.11+
- 8GB RAM minimum (16GB recommended for local model inference)
- OpenAI API key (for fallback generation and RAGAS evaluation)
- HuggingFace account with write token (for pushing fine-tuned adapter)

### Option A — Local Development (OpenAI Fallback, Recommended First Run)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/enterprise-rag-assistant
cd enterprise-rag-assistant

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set:
#   OPENAI_API_KEY=sk-...
#   HUGGINGFACE_TOKEN=hf_...  (optional for local dev)

# 5. Place AWS Well-Architected PDFs in data/pdfs/
# Download from: https://aws.amazon.com/architecture/well-architected/
# Expected files:
#   data/pdfs/operational_excellence.pdf
#   data/pdfs/security.pdf
#   data/pdfs/reliability.pdf
#   data/pdfs/performance_efficiency.pdf
#   data/pdfs/cost_optimization.pdf
#   data/pdfs/sustainability.pdf

# 6. Ingest documents (builds FAISS index — run once)
uvicorn backend.main:app --reload &
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"force_reindex": false}'
# Expected: {"status": "success", "chunks_indexed": ~800, ...}

# 7. Start the Streamlit UI (separate terminal)
streamlit run frontend/app.py
# Opens at http://localhost:8501

# 8. Or query the API directly
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How does AWS recommend designing for failure?",
       "use_fine_tuned": false,
       "top_k": 5
     }'
```

### Option B — With Fine-Tuned Model (After Running Colab Notebook)

```bash
# After completing notebooks/fine_tuning_phi3_qlora.ipynb:

# 1. Update config.yaml with your adapter path
# Edit config.yaml:
#   models:
#     primary:
#       adapter_repo: "your-username/phi3-mini-enterprise-qlora"

# 2. Ensure HUGGINGFACE_TOKEN is set in .env
# The API will load the adapter at startup automatically

# 3. Start the API (will attempt to load fine-tuned model first)
uvicorn backend.main:app --reload
# Look for: "Fine-tuned model loaded successfully" in logs
# If not found: "OpenAI fallback is active" — fallback activates automatically

# 4. Verify which model is active
curl http://localhost:8000/health
# Response includes: "primary_model_available": true/false
```

### Option C — Docker

```bash
# Build and run with Docker
docker build -t rag-assistant .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e HUGGINGFACE_TOKEN=hf_... \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.cache:/app/.cache \
  rag-assistant

# The data/ volume mount persists the FAISS index and document cache
# between container restarts
```

### Verifying the Setup

```bash
# Health check (should return 200)
curl http://localhost:8000/health

# Readiness check (returns 503 until /ingest is called)
curl http://localhost:8000/health/ready

# API documentation
open http://localhost:8000/docs

# Prometheus metrics
curl http://localhost:8000/metrics | grep rag_
```

### Environment Variables Reference

```bash
# .env.example — copy to .env and fill in values

# Required
OPENAI_API_KEY=sk-...          # OpenAI API key (fallback generation)

# Optional — only needed with fine-tuned model
HUGGINGFACE_TOKEN=hf_...       # HF Hub token (private adapter repos)

# Optional overrides (defaults set in config.yaml)
# ENVIRONMENT=development       # development | staging | production
# LOG_LEVEL=INFO                # DEBUG | INFO | WARNING | ERROR
```

---

## 5. API Reference

### Base URL

```
Local:      http://localhost:8000
Production: https://rag-api.internal.company.com
```

### Authentication

No authentication in v1 (internal tool, VPN-protected).
Production deployment: add AWS Cognito or API Gateway key-based auth.

---

### `POST /generate`

Generate a grounded answer from the AWS Well-Architected Framework.

**Request Body**

```json
{
  "query": "How does AWS recommend designing for failure?",
  "use_fine_tuned": true,
  "top_k": 5,
  "filter_pillar": "Reliability"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | ✅ | — | Natural language question (1–1000 chars) |
| `use_fine_tuned` | boolean | ❌ | `true` | Prefer fine-tuned model; falls back to OpenAI if unavailable |
| `top_k` | integer | ❌ | `5` | Chunks to retrieve (1–10) |
| `filter_pillar` | string | ❌ | `null` | Restrict to one pillar (see valid values below) |

**Valid `filter_pillar` values:**
`"Operational Excellence"` · `"Security"` · `"Reliability"` · `"Performance Efficiency"` · `"Cost Optimization"` · `"Sustainability"`

**Response: 200 OK**

```json
{
  "answer": "According to the Reliability pillar, AWS recommends designing systems to automatically recover from failure. Key practices include: eliminating single points of failure through redundancy [Source: Reliability, Page 14], implementing automatic failover mechanisms, and testing failure scenarios regularly through chaos engineering [Source: Reliability, Page 22].\n\nSystems should be designed with the assumption that components WILL fail, not if they fail.",
  "sources": [
    {
      "pillar": "Reliability",
      "source_file": "reliability.pdf",
      "page_number": 14,
      "section": "Design Principles",
      "excerpt": "Design your workload to automatically recover from failure. Automatically detect failures and recover without human intervention...",
      "relevance_score": 0.8847,
      "chunk_index": 2
    },
    {
      "pillar": "Reliability",
      "source_file": "reliability.pdf",
      "page_number": 22,
      "section": "Reliability Practices",
      "excerpt": "Use chaos engineering to test the resilience of your systems. Game days help validate that recovery mechanisms work as expected...",
      "relevance_score": 0.8312,
      "chunk_index": 0
    }
  ],
  "confidence": "HIGH",
  "confidence_reason": "Context directly addresses the question with multiple supporting passages.",
  "model_used": "openai_gpt4o_mini",
  "retrieval_latency_ms": 145,
  "generation_latency_ms": 1823,
  "total_latency_ms": 1989,
  "tokens_used": 487,
  "cache_hit": false,
  "grounding_flag": false,
  "grounding_score": 0.8341,
  "is_refusal": false,
  "prompt_version": "v1"
}
```

**Response: Refusal (no relevant context)**

```json
{
  "answer": "I could not find relevant information in the AWS Well-Architected Framework documents to answer this question...",
  "sources": [],
  "confidence": "LOW",
  "is_refusal": true,
  "model_used": "unavailable",
  "retrieval_latency_ms": 132,
  "generation_latency_ms": 0,
  "total_latency_ms": 134,
  "cache_hit": false,
  "grounding_flag": false,
  "grounding_score": 0.0,
  "tokens_used": null,
  "prompt_version": "v1"
}
```

**Error Responses**

| Status | Error Code | Cause |
|---|---|---|
| 422 | `VALIDATION_ERROR` | Query empty, too long (>1000 chars), invalid pillar name, top_k out of range |
| 200 | `is_refusal: true` | Query blocked by guardrail (injection detected) or no context found |
| 503 | `VECTOR_STORE_NOT_READY` | Index not built — call `POST /ingest` first |
| 504 | `GENERATION_TIMEOUT` | Model took longer than 30s — retry |
| 500 | `INTERNAL_ERROR` | Unexpected error — check logs with `request_id` |

**Error Response Schema**

```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Request validation failed. Check the 'details' field for field-level errors.",
  "request_id": "a3f2b1c4-d5e6-7890-abcd-ef1234567890",
  "details": [
    {
      "type": "string_too_long",
      "loc": ["body", "query"],
      "msg": "String should have at most 1000 characters",
      "input": "..."
    }
  ]
}
```

---

### `POST /ingest`

Trigger document ingestion and FAISS index building.

**Request Body**

```json
{
  "force_reindex": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `force_reindex` | boolean | `false` | Re-embed all chunks even if already indexed. Use when embedding model changes. |

**Response: 200 OK**

```json
{
  "status": "success",
  "chunks_embedded": 127,
  "chunks_indexed": 127,
  "total_vectors_in_index": 843,
  "duration_seconds": 47.3,
  "message": "Ingestion complete. 127 new chunks indexed. 716 duplicates skipped."
}
```

**Notes:**
- Idempotent — safe to call multiple times
- Deduplication skips chunks already in the index (content-hash based)
- Concurrent ingestion requests return HTTP 409 (only one run at a time)
- Expected duration: 30-120 seconds for 6 PDFs on first run

---

### `POST /ingest/upload`

Upload a single PDF file and add it to the corpus.

**Request:** `multipart/form-data`

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@/path/to/document.pdf"
```

**Response: 200 OK**

```json
{
  "status": "uploaded",
  "filename": "document.pdf",
  "size_bytes": 1048576,
  "message": "File saved. Call POST /ingest to index it."
}
```

---

### `GET /health`

Liveness check — returns 200 if the process is running.

**Response: 200 OK**

```json
{
  "status": "healthy",
  "vector_store_ready": true,
  "primary_model_available": false,
  "fallback_model_available": true,
  "vector_count": 843,
  "version": "1.0.0"
}
```

---

### `GET /health/ready`

Readiness check — returns 200 only when all dependencies are healthy.

**Response: 200 OK** (ready to serve)

```json
{
  "status": "ready",
  "vector_count": 843,
  "primary_model_available": false,
  "fallback_model_available": true,
  "version": "1.0.0"
}
```

**Response: 503 Service Unavailable** (not ready)

```json
{
  "status": "not_ready",
  "issues": [
    "Vector store is empty — call POST /ingest to build the index"
  ],
  "vector_count": 0,
  "primary_model_available": false,
  "fallback_model_available": true
}
```

---

### `GET /metrics`

Prometheus metrics exposition endpoint.

```bash
curl http://localhost:8000/metrics
```

```
# HELP rag_requests_total Total number of requests to the /generate endpoint
# TYPE rag_requests_total counter
rag_requests_total{endpoint="generate",model="openai_fallback",status="success"} 47.0
rag_requests_total{endpoint="generate",model="cache",status="cache_hit"} 12.0

# HELP rag_retrieval_latency_seconds Time to retrieve relevant chunks
# TYPE rag_retrieval_latency_seconds histogram
rag_retrieval_latency_seconds_bucket{le="0.1"} 41.0
rag_retrieval_latency_seconds_bucket{le="0.2"} 47.0
rag_retrieval_latency_seconds_p95 0.187

# HELP rag_hallucination_flags_total Responses flagged by grounding check
# TYPE rag_hallucination_flags_total counter
rag_hallucination_flags_total 2.0

# HELP rag_fallback_activations_total OpenAI fallback activations
# TYPE rag_fallback_activations_total counter
rag_fallback_activations_total{reason="primary_unavailable"} 47.0
```

---

### cURL Examples

```bash
# Example 1: Basic query
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is least privilege access?"}'

# Example 2: Pillar-specific query
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How should I implement cost allocation tagging?",
    "filter_pillar": "Cost Optimization",
    "top_k": 3
  }'

# Example 3: Explicitly use OpenAI fallback
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the trade-offs between RTO and cost?",
    "use_fine_tuned": false
  }'

# Example 4: Force re-index all documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reindex": true}'

# Example 5: Check metrics after a few requests
curl http://localhost:8000/metrics | grep "^rag_"
```

---

## 6. Evaluation Results

*Full evaluation report: [`docs/evaluation_report.md`](docs/evaluation_report.md)*

### RAGAS Quality Scores (25-question gold dataset)

| Metric | Score | Target | Status |
|---|---|---|---|
| **Faithfulness** | 0.81 | ≥ 0.75 | ✅ |
| **Answer Relevancy** | 0.79 | ≥ 0.75 | ✅ |
| **Context Precision** | 0.74 | ≥ 0.70 | ✅ |
| **Context Recall** | 0.71 | ≥ 0.70 | ✅ |

*RAGAS uses GPT-4 as the evaluation judge. Scores vary ±0.03 between runs.*

### Latency Benchmarks (50 queries, sequential)

| Stage | p50 | p95 | Target (p95) | Status |
|---|---|---|---|---|
| Retrieval (embed + FAISS + rerank) | 95ms | 180ms | ≤ 200ms | ✅ |
| Generation (OpenAI fallback) | 1,850ms | 4,200ms | ≤ 6,000ms | ✅ |
| Total end-to-end | 1,960ms | 4,420ms | ≤ 7,000ms | ✅ |
| Cache hit (Redis response cache) | 8ms | 45ms | ≤ 50ms | ✅ |

### Adversarial Test Results (7 cases)

| Test Type | Cases | Passed | Result |
|---|---|---|---|
| Out-of-scope queries (no context) | 2 | 2 | ✅ Correct refusal |
| Prompt injection attempts | 3 | 3 | ✅ Guardrail blocked |
| Oversized input (>1000 chars) | 1 | 1 | ✅ HTTP 422 |
| Empty input | 1 | 1 | ✅ HTTP 422 |

**All 7 adversarial cases handled correctly.**

### Source Retrieval Accuracy

| Metric | Value |
|---|---|
| Source hit rate (correct pillar retrieved) | 88% |
| Confidence distribution: HIGH / MEDIUM / LOW | 56% / 36% / 8% |
| Hallucination flag rate | 8% |

---

## 7. Fine-Tuning

*Full notebook: [`notebooks/fine_tuning_phi3_qlora.ipynb`](notebooks/fine_tuning_phi3_qlora.ipynb)*

### What Was Fine-Tuned

**Base model:** `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)  
**Method:** QLoRA (Quantized Low-Rank Adaptation)  
**Dataset:** `databricks/databricks-dolly-15k` filtered to `closed_qa` + `information_extraction` (~2,800 examples)  
**Platform:** Google Colab T4 GPU (16GB VRAM, free tier)  
**Training time:** ~60-90 minutes  
**Adapter size:** ~60MB (pushed to HuggingFace Hub)

### Why QLoRA

| Approach | VRAM Required | Training Time | Quality vs Full FT |
|---|---|---|---|
| Full fine-tuning | ~30GB | Hours | Baseline |
| LoRA (fp16) | ~12GB | ~2 hours | ~95% |
| **QLoRA (4-bit NF4)** | **~4GB** | **~60-90 min** | **~92%** |

QLoRA enables fine-tuning on a free Colab T4 with minimal quality compromise.

### Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| LoRA rank (r) | 16 | Captures task-specific signal without overfitting on ~2.8k examples |
| LoRA alpha | 32 | Standard 2×r initialization scale — normalizes update magnitude |
| Dropout | 0.05 | Minimal regularization; LoRA rank already constrains capacity |
| Target modules | All linear layers | Adapts both attention and MLP for comprehensive behavioral change |
| Learning rate | 2e-4 | Standard for LoRA instruction-tuning; converges in 300-500 steps |
| LR schedule | Cosine annealing | Keeps LR high early, smooth decay prevents late-stage overfitting |
| Effective batch | 8 (1 × 8 grad accum) | Stable gradient estimates within T4 VRAM constraints |

### Before/After Evaluation

| Metric | Base Model | Fine-Tuned | Delta |
|---|---|---|---|
| ROUGE-L (avg, 5 eval questions) | 0.387 | 0.431 | +0.044 |
| Instruction following | Inconsistent | Consistent | Qualitative ↑ |
| Context adherence | Supplements from training data | Stays in context | Qualitative ↑ |
| Response verbosity | Often padded | More concise | Qualitative ↑ |

**What fine-tuning improved:** Behavioral adaptation — the model follows the
"answer only from context" instruction more reliably and formats responses
more consistently.

**What fine-tuning did NOT improve:** Domain knowledge (AWS-specific facts).
Domain knowledge comes from the retrieved context, not the adapter.
Fine-tuning on dolly-15k taught the model HOW to respond, not WHAT to know.

### Running the Fine-Tuning Notebook

```bash
# Open in Google Colab:
# 1. Go to colab.research.google.com
# 2. File → Open notebook → GitHub tab
# 3. Enter: https://github.com/your-username/enterprise-rag-assistant
# 4. Select: notebooks/fine_tuning_phi3_qlora.ipynb
# 5. Runtime → Change runtime type → GPU → T4
# 6. Add HF_TOKEN to Colab Secrets (key icon in left sidebar)
# 7. Run all cells (Runtime → Run all)
```

---

## 8. Deployment Plan Summary

*Full deployment plan: [`docs/deployment_plan.md`](docs/deployment_plan.md)*

### Production Architecture

```
Internet → AWS ALB → ECS Fargate (API, CPU-only, min 2 tasks)
                  → SQS Queue → EC2 ASG (GPU workers, g4dn.xlarge, min 1)
                              ↕
                         ElastiCache Redis (shared response cache)
                         S3 (FAISS index, model artifacts)
```

### Cost Model

| Scenario | Monthly Cost | Per-Request Cost |
|---|---|---|
| 10k req/day, 70% cache hit rate | ~$609 | ~$0.002 |
| With spot instances (75% spot) | ~$400 | ~$0.001 |
| Development (1 instance, low traffic) | ~$200 | — |

### Key Infrastructure Choices

| Component | Choice | Rationale |
|---|---|---|
| GPU instance | `g4dn.xlarge` (T4 16GB) | Phi-3-mini in 4-bit needs ~4GB VRAM; $0.526/hr on-demand |
| API tier | ECS Fargate (CPU) | No GPU needed for routing/caching; scales independently |
| Queue | SQS | Decouples API from inference; absorbs burst demand |
| Cache | ElastiCache Redis | Shared across API instances; ~$0 per cache hit |

### Latency SLAs

| SLA Tier | Target | Metric |
|---|---|---|
| Cache hit | p95 < 50ms | Redis lookup |
| Full pipeline | p95 < 7,000ms | End-to-end |
| Availability | 99.5% uptime | Non-5xx rate |

---

## 9. Known Limitations

These limitations are documented honestly because understanding them is a
prerequisite for extending or operating this system in production.

### Retrieval Limitations

**FAISS does not scale horizontally.**
FAISS stores the index in the memory of a single process. In a multi-instance
deployment, each instance has its own copy of the index. This is acceptable for
a corpus of 6 PDFs (~843 vectors, ~10MB in memory) but would require migration
to Pinecone, Weaviate, or Qdrant for corpora exceeding ~1M vectors or deployments
requiring a shared, consistent index across instances.

**No metadata filtering at index time.**
FAISS does not support native metadata filtering (unlike Pinecone). The current
implementation retrieves k×3 candidates and filters post-retrieval. This means
pillar-specific queries fetch more vectors than necessary before filtering.
For large corpora, this becomes inefficient. Pinecone supports metadata filtering
at query time with no performance penalty.

**Static document corpus.**
The index is built from a snapshot of the six pillar PDFs. Changes to the AWS
Well-Architected Framework after ingestion are not reflected until a manual
re-ingestion is triggered. There is no automated detection of document changes.

### Generation Limitations

**Fine-tuning on general dataset, not domain-specific data.**
The LoRA adapter was trained on `databricks/databricks-dolly-15k` — a general
instruction-following dataset. It improves behavioral alignment (follow instructions,
cite sources, refuse when unsure) but does not improve domain knowledge of AWS.
A higher-quality fine-tuning approach would extract Q&A pairs directly from the
AWS Well-Architected documents and fine-tune on those domain-specific pairs.

**No streaming.**
The API returns the complete generated answer after generation finishes (2-5 seconds).
There is no Server-Sent Events (SSE) streaming endpoint. Users see a loading state
for the full generation duration rather than tokens appearing progressively.
Time-to-first-token with streaming would be ~500ms — a significant UX improvement.

**CPU inference is slow.**
Without a GPU, Phi-3-mini in 4-bit generates ~3-5 tokens/second (vs 20-30 on T4).
Local development without GPU will see 30-60 second generation times for the
fine-tuned model. The OpenAI fallback (1-3s) is strongly recommended for local
development.

### Conversational Limitations

**No multi-turn conversation memory.**
Each request is stateless. A follow-up question ("What about the cost implications
of that?") has no access to the previous exchange. Maintaining conversation context
requires a session store, context window management, and conversation history
injection into the prompt — all out of scope for v1.

**No user-specific context.**
The system has no knowledge of a user's specific AWS environment, existing
architecture, or organizational constraints. Answers are general framework
guidance, not personalized recommendations.

### Evaluation Limitations

**Small gold dataset (n=25).**
25 questions provide directional confidence in RAGAS scores but are not
statistically significant. The margin of error at n=25 is approximately ±0.05
per metric. A production evaluation baseline would use 200+ questions.

**RAGAS judge model bias.**
RAGAS uses GPT-4 as the faithfulness and context precision judge. Scores are
not model-agnostic and may reflect GPT-4's stylistic preferences as much as
genuine quality signals. Human evaluation on a 10% sample would provide an
independent validation.

---

## 10. Future Improvements

These improvements are sequenced by impact-to-effort ratio, not by complexity.

### High Impact, Achievable in v2

**Streaming responses via SSE.**
Replace the single-response endpoint with a Server-Sent Events stream.
Time-to-first-token drops from 2-5s to ~500ms. Dramatically improves perceived
responsiveness without changing total generation time. Implementation:
FastAPI's `StreamingResponse` + LangChain's streaming callbacks.

**Semantic response cache.**
Replace exact-match query hashing with approximate semantic matching.
"What is least privilege?" and "Explain least privilege access" should hit the
same cache entry. Implementation: embed the query, search a small vector index
of cached query embeddings, return cache hit if cosine similarity > 0.95.
Expected cache hit rate improvement: 2-3× over exact matching.

**Domain-specific fine-tuning dataset.**
Extract Q&A pairs directly from the six AWS pillar PDFs using GPT-4.
For each document chunk, generate 3-5 questions that the chunk answers.
Fine-tune the adapter on these domain-specific pairs instead of dolly-15k.
Expected improvement: faithfulness score increases from 0.81 → 0.88+, with
real AWS terminology in training examples rather than approximations.

### Medium Impact, v3 Roadmap

**User feedback loop.**
Add thumbs up/down buttons to the Streamlit UI. Store feedback with the
`request_id`, query hash, and response. Use negative feedback as a signal to:
(a) add negatively-rated Q&A pairs to a fine-tuning correction dataset,
(b) lower the cache TTL for queries with negative feedback history,
(c) alert the team to investigate systematic failures.

**Multi-pillar context retrieval.**
For cross-pillar questions ("How do reliability and cost trade off?"), the
current retriever uses a single FAISS search. A more sophisticated approach
retrieves from each pillar's sub-index separately, then merges and reranks.
This reduces inter-pillar semantic dilution and improves context recall for
analytical questions.

**Conversation memory for follow-up questions.**
Add an optional `session_id` field to the `/generate` request. Maintain a
server-side conversation buffer (Redis, 30-minute TTL). Inject conversation
history into the system prompt for follow-up resolution. Memory window: last
3 turns (to keep prompt length manageable).

### Strategic, v4+ Roadmap

**Multi-modal document support.**
AWS architecture diagrams embedded in PDFs contain information not captured
by text extraction (service relationships, data flows, component boundaries).
Add a vision model (LLaVA or GPT-4V) to extract structured descriptions from
embedded diagrams during ingestion. Store as text chunks with image metadata.

**A/B testing infrastructure for model variants.**
When a new fine-tuned adapter is ready, route a configurable percentage of
traffic to the new adapter and compare quality metrics between variants.
Implementation: feature flag in request routing, per-variant metrics in
Prometheus (label `adapter_version`), automatic promotion if evaluation
metrics improve by > 5%.

**Automated evaluation pipeline.**
Run RAGAS on a 5% random sample of production requests weekly.
Compare scores to the baseline established in v1 evaluation.
Alert if faithfulness drops more than 0.05 points — this signals document
corpus drift (AWS updated the whitepapers) or model degradation.

**Per-pillar fine-tuned adapters.**
Train a separate adapter for each of the six pillars, each fine-tuned on
pillar-specific Q&A. At query time, route to the pillar-specific adapter
if `filter_pillar` is set, or use the general adapter for cross-pillar queries.
This exploits PEFT's adapter-swapping capability — the base model is loaded
once and adapters are swapped in memory (<1s swap time).

---

## Project Structure

```
enterprise-rag-assistant/
│
├── backend/                    # FastAPI application
│   ├── main.py                 # App factory, lifespan, middleware
│   ├── routes/
│   │   ├── generate.py         # POST /generate
│   │   ├── ingest.py           # POST /ingest, POST /ingest/upload
│   │   └── health.py           # GET /health, GET /health/ready
│   ├── services/
│   │   ├── ingestion.py        # Document loading, chunking, deduplication
│   │   ├── vector_store.py     # FAISS, embedding, reranking
│   │   ├── rag_pipeline.py     # Full RAG orchestration
│   │   ├── llm_manager.py      # Model routing and fallback
│   │   ├── prompts.py          # Versioned prompt templates
│   │   └── guardrails.py       # Input/output safety checks
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response types
│   └── core/
│       ├── config.py           # Typed config loader (config.yaml)
│       ├── dependencies.py     # FastAPI DI, service initialization
│       ├── logging.py          # Structured JSON logging (loguru)
│       ├── metrics.py          # Prometheus custom metrics
│       └── cache.py            # Two-level cache (embedding + response)
│
├── frontend/
│   └── app.py                  # Streamlit UI
│
├── notebooks/
│   └── fine_tuning_phi3_qlora.ipynb  # QLoRA fine-tuning (Colab T4)
│
├── eval/
│   ├── gold_dataset.json       # 25 Q&A pairs + 7 adversarial cases
│   ├── run_ragas.py            # RAGAS evaluation pipeline
│   ├── benchmark.py            # Latency benchmark (50 queries)
│   └── evaluation_report.md   # Full evaluation results
│
├── docs/
│   ├── architecture.png        # System architecture diagram
│   ├── deployment_plan.md      # Full production deployment plan
│   └── tradeoffs.md            # Architectural decision record
│
├── tests/
│   ├── test_ingestion.py       # Unit tests: ingestion pipeline
│   ├── test_vector_store.py    # Unit tests: embedding + FAISS
│   ├── test_rag_pipeline.py    # Unit tests: prompts + guardrails
│   ├── test_observability.py   # Unit tests: logging + cache + guardrails
│   └── test_api.py             # Integration tests: all endpoints
│
├── data/
│   └── pdfs/                   # AWS Well-Architected PDFs (6 files)
│
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD pipeline (test → build → deploy)
│
├── config.yaml                 # All tunable parameters (no hardcoded values)
├── requirements.txt            # Pinned dependencies
├── Dockerfile                  # Production container image
├── .env.example                # Environment variable template
├── .gitignore                  # Excludes .env, cache, model weights
└── README.md                   # This file
```

---

## Contributing

This project is a take-home assessment submission. The architecture is designed
for clarity and reviewability rather than open-source contribution workflows.

For questions about design decisions, see [`docs/tradeoffs.md`](docs/tradeoffs.md).
For deployment questions, see [`docs/deployment_plan.md`](docs/deployment_plan.md).

---

## License

MIT License — see [LICENSE](LICENSE) file.

Base model (`microsoft/Phi-3-mini-4k-instruct`): MIT License.  
Fine-tuning dataset (`databricks/databricks-dolly-15k`): CC BY-SA 3.0.  
AWS Well-Architected Framework documentation: © Amazon Web Services.