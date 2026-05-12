Here's the complete strategic execution plan, organized for maximum scoring impact.---

## Deliverables overview

| ID | Deliverable | Scoring weight it targets |
|---|---|---|
| D1 | RAG Backend (FastAPI + LangChain + FAISS) | System Design 25%, Code Quality 15% |
| D2 | Fine-Tuning Artifacts (Colab notebook, adapter, eval) | LLM Understanding 20%, Deep Learning 10% |
| D3 | Inference API (`/generate`, fallback, guardrails) | System Design, Code Quality |
| D4 | Deployment Plan Document | MLOps & Cloud 20% |
| D5 | Production Readiness Layer (obs, caching, benchmarks) | Leadership Thinking 10%, all others |

---

## Phase 0 — Project Foundation & Architecture

**Goal:** Produce a scope document and architecture diagram that frames the whole submission before writing a line of code. This is the first thing a reviewer sees and sets expectations for everything downstream.

### 0.1 — Choose and define the use case

What: Pick a realistic enterprise domain — an internal engineering knowledge assistant (e.g. answers questions over company docs, runbooks, API references) is the best choice. It has clear retrieval scenarios, easy evaluation, and looks enterprise-relevant.

How: Write a 1-page scope note covering: who the users are, what queries they send, what a successful answer looks like, what failure looks like, and what is explicitly out of scope. Include a user journey (user asks → system retrieves → system answers → user sees sources).

Senior signal: Add a "constraints and assumptions" section — local GPU unavailable, adapter served from HF Hub, vector store swappable, fallback to OpenAI when fine-tuned model is down.

### 0.2 — Finalize architecture diagram

What: Draw the full system — not just the RAG chain, but every layer: ingestion, embedding, vector store, retrieval, reranker, generation, guardrails, API gateway, caching, observability, deployment.

How: Use Excalidraw or draw.io. Export as PNG for the README. Show two separate flows clearly: the ingestion/indexing flow (offline, runs when docs change) and the query/generation flow (online, per user request). Add a fallback path when the fine-tuned model is unavailable.

Senior signal: Label where latency budgets matter (retrieval <200ms, generation <5s). Show the cache hit path as a bypass that skips generation entirely.

### 0.3 — Set up repo and free cloud tooling

What: GitHub repo with clean folder structure, Google Colab notebook configured with T4 GPU, HuggingFace account with write token, OpenAI API key in env variable.

How: Folder layout — `backend/` (FastAPI app), `notebooks/` (fine-tuning Colab), `docs/` (architecture, deployment plan, evaluation report), `eval/` (benchmark scripts and gold dataset). Add `.env.example`, `requirements.txt`, and a root `README.md` stub. No secrets in the repo.

Senior signal: Add a `config.yaml` at the root with all tunable parameters — chunk size, overlap, top-k, LoRA rank, similarity threshold, generation temperature. This signals that the system is configurable without code changes.

---

## Phase 1 — Document Ingestion Pipeline

**Goal:** Build a clean, metadata-aware ingestion module. Retrieval quality starts here, not at the embedding model.

### 1.1 — Document loading and normalization

What: Accept PDF, markdown, and plain text files. Load them, strip noise (headers/footers, navigation elements, page numbers), normalize whitespace, and extract structured metadata.

How: Use LangChain's `PyPDFLoader`, `UnstructuredMarkdownLoader`, `TextLoader`. Write a thin wrapper that normalizes all output to a `Document(page_content, metadata)` format with consistent metadata fields: `source`, `file_type`, `section`, `page_number`, `ingested_at`.

Senior signal: Add a deduplication step — hash each chunk's content and skip re-embedding chunks that already exist in the index. This reduces cost on re-ingestion and prevents index bloat.

### 1.2 — Chunking strategy

What: Split documents into semantically meaningful units that balance retrieval precision against context completeness.

How: Use `RecursiveCharacterTextSplitter` with ~512 token chunks and 50–100 token overlap. Explain the tradeoff in code comments and docs: chunks too small lose context, too large dilute the embedding signal. For higher quality, use `SemanticChunker` (LangChain's experimental chunker that splits at semantic boundaries rather than character counts).

Senior signal: Document the decision — why 512 and not 256 or 1024. Add a note that chunk size should be tuned based on the query type: short factual queries benefit from smaller chunks, complex multi-step questions from larger. This reads as engineering judgment, not arbitrary choice.

---

## Phase 2 — Embedding and Vector Store

**Goal:** Build a fast, persistent, and retrievable semantic index with configurable search behavior.

### 2.1 — Embedding model selection and pipeline

What: Embed all chunks, store vectors in a persistent FAISS index with metadata, support batch embedding for efficiency.

How: Use `BAAI/bge-small-en-v1.5` via `sentence-transformers`. It's free, CPU-friendly, and outperforms `all-MiniLM-L6-v2` on retrieval benchmarks. Use LangChain's `HuggingFaceEmbeddings` wrapper. Save the FAISS index to disk (`faiss.save_local()`) so re-ingestion is not required on restart.

Senior signal: Add embedding-level caching — hash the chunk content and store the embedding to disk. If the same chunk is re-ingested, skip the model call and read from cache. Mention in docs that production would replace FAISS with Pinecone or Weaviate for multi-node scalability.

### 2.2 — Retrieval optimization

What: Retriever with configurable `k`, similarity threshold filtering, and optional reranking.

How: Basic retriever: `FAISS.as_retriever(search_type="similarity", search_kwargs={"k": 5, "score_threshold": 0.6})`. Reranker (high-value addition): add `CrossEncoder` from `BAAI/bge-reranker-base` as a second pass. It takes the top-k candidates and reorders by relevance — this demonstrably improves precision and signals advanced RAG understanding.

Senior signal: Explain in docs why reranking helps. The bi-encoder (embedding) is fast but coarse; the cross-encoder is slower but considers the query-document pair jointly. Add a `USE_RERANKER` flag in `config.yaml` so it can be toggled off in latency-sensitive scenarios.

---

## Phase 3 — RAG Generation and Hallucination Control

**Goal:** Build a grounded generation layer that answers only from retrieved context, refuses when context is insufficient, and leaves a clear evidence trail.

### 3.1 — Prompt engineering

What: A structured prompt template that enforces grounding, controls tone, and defines refusal behavior.

How: Design a system prompt with three explicit rules baked in: (1) answer only from the provided context, (2) if the context does not contain enough information, say so — do not speculate, (3) cite the source document for every claim. Use LangChain's `ChatPromptTemplate` with delimiters separating the context block from the question. Write this prompt in `prompts.py` as a versioned string constant, not hardcoded inline.

Senior signal: Add a "confidence self-assessment" instruction in the prompt — ask the model to rate its own confidence (high/medium/low) at the end of its response. This feeds directly into the confidence scoring in the API response schema.

### 3.2 — Hallucination protection

What: A post-generation verification layer that checks whether the answer is grounded in the retrieved context.

How: Two mechanisms: (1) Semantic grounding check — embed the generated answer and compute cosine similarity against the retrieved chunks. If similarity is below threshold (e.g. 0.5), flag the answer as potentially ungrounded. (2) No-context fallback — if the retriever returns no results above the similarity threshold, return a structured "no relevant context found" response instead of passing an empty context to the LLM. Never let the LLM answer from nothing.

Senior signal: Mention in docs that production would add an LLM-as-judge step — a second model call that evaluates whether the answer contradicts the context. Mention RAGAS as the evaluation framework for measuring faithfulness at scale.

### 3.3 — Response schema design

What: A structured response object that carries more than just the answer text.

How: Return `answer`, `sources` (list of document names + relevant excerpt), `confidence` (high/medium/low derived from the model's self-assessment and the semantic grounding score), `model_used` (fine-tuned or fallback), `retrieval_latency_ms`, `generation_latency_ms`, `tokens_used`. This makes the API feel production-grade and gives the evaluation harness rich signals to work with.

---

## Phase 4 — Fine-Tuning (Colab T4, Free)

**Goal:** Produce real training artifacts — not a simulation — using free cloud GPU. The notebook is a key deliverable and should read like a well-documented experiment, not a rushed script.

### 4.1 — Dataset preparation

What: Prepare an instruction-tuning dataset in alpaca format that targets the domain of the enterprise assistant.

How: Use `databricks/databricks-dolly-15k` from HuggingFace — it is diverse, instruction-style, and openly licensed. Filter for the "closed_qa" and "information_extraction" categories (~3,000 examples) to match the RAG assistant's task. Convert to `{"instruction": "...", "input": "...", "output": "..."}` format. Apply a chat template. Split 90/10 train/validation.

Senior signal: Add a data quality section in the notebook — show token length distribution, flag outliers, explain the filtering criteria. This shows you understand that training data quality is as important as model architecture.

### 4.2 — Base model and QLoRA configuration

What: Fine-tune `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters) with QLoRA on Colab T4 (16GB VRAM).

How: Use `unsloth` library — it reduces memory by ~40% and speeds up training 2x vs vanilla PEFT. Load in 4-bit NF4 quantization with double quantization. LoRA config: `r=16`, `alpha=32`, `dropout=0.05`, target all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Training: batch size 1, gradient accumulation 8 (effective batch 8), learning rate 2e-4 with cosine schedule, 1–2 epochs (300–500 steps is enough to show convergence).

Senior signal: Document every hyperparameter choice with a one-line justification. `r=16` because rank-16 captures enough task-specific signal without overfitting on a small dataset. `alpha=32` (2× rank) is the standard initialization scale. This is exactly what a reviewer with deep learning knowledge looks for.

### 4.3 — Training execution and artifact saving

What: Train, monitor loss, save the adapter, push to HuggingFace Hub.

How: Run on Colab T4. Monitor training loss every 50 steps — it should drop from ~2.0 to ~1.2 range. Save the LoRA adapter only (not the merged weights) to `{hf_username}/phi3-mini-enterprise-qlora`. The adapter is ~60MB, trivial to push and pull. Do not merge — loading base + adapter separately lets you swap adapters without re-downloading the base model.

Senior signal: Export training loss curve as a plot in the notebook. Include a `model_card.md` in the HF repo explaining the base model, dataset, training config, intended use, and limitations.

### 4.4 — Before/after evaluation

What: Compare base model vs fine-tuned model on a held-out set of 20–30 domain-specific questions.

How: Generate answers from both models on identical prompts. Compute ROUGE-L using the `evaluate` library. Show a qualitative side-by-side table with 5 representative examples. Flag where the base model hallucinated or went off-topic vs where the fine-tuned model was more grounded and concise.

Senior signal: Add a failure analysis section — 2–3 examples where fine-tuning did not help or made things worse. Acknowledging failure cases reads as more credible than claiming uniform improvement.

---

## Phase 5 — Inference API

**Goal:** A `POST /generate` endpoint that feels production-ready — typed, validated, fault-tolerant, and observable.

### 5.1 — FastAPI app structure

What: Clean, layered FastAPI app with proper separation of concerns.

How: `main.py` (app factory + router registration), `routes/` (endpoint definitions), `services/` (RAG pipeline, LLM manager, retriever), `models/` (Pydantic schemas), `core/` (config loader, logging setup, metrics). All configuration read from `config.yaml` at startup. No hardcoded values anywhere.

### 5.2 — Endpoints

What: `POST /ingest` (upload and index a document), `POST /generate` (RAG query), `GET /health` (liveness check), `GET /metrics` (Prometheus exposition).

How: `/generate` accepts `{query: str, use_fine_tuned: bool = True, top_k: int = 5}`. Returns the full response schema from Phase 3. Input validation via Pydantic — max query length 1000 characters, required field checking, type coercion. Add `asyncio.timeout(30)` around the generation call so slow model runs don't hang the server.

### 5.3 — Model routing and fallback

What: Transparent switching between the fine-tuned local model and the OpenAI fallback.

How: At startup, try to load the LoRA adapter from HF Hub. If GPU is unavailable or download fails, log a warning and activate the OpenAI fallback. The `use_fine_tuned` flag in the request lets the caller explicitly choose. Log which model served each request. The fallback is not a degraded mode — it is a first-class citizen of the architecture.

### 5.4 — Error handling and resilience

What: Every failure mode handled gracefully — no raw tracebacks to the client, no silent failures.

How: Global exception handler returns structured error responses with `error_code`, `message`, and `request_id`. Specific handlers for: model load failure (503 with retry-after header), generation timeout (504), retrieval failure (500 with logged trace), invalid input (422 with field-level errors). Add retry logic for the HF Hub model download with exponential backoff.

---

## Phase 6 — Observability and Production Readiness

**Goal:** Demonstrate operational maturity. This phase directly targets Leadership Thinking and MLOps scoring.

### 6.1 — Structured logging

What: Every request logged as a JSON object with consistent fields.

How: Use `loguru` with a JSON sink. Every log entry includes: `request_id`, `timestamp`, `query_hash` (hashed, not raw — for privacy), `retrieval_latency_ms`, `generation_latency_ms`, `total_latency_ms`, `tokens_input`, `tokens_output`, `model_used`, `cache_hit`, `confidence`, `error` (if any). Log at INFO for normal requests, ERROR for failures, DEBUG for retrieval scores.

### 6.2 — Metrics

What: Prometheus-compatible metrics exposed at `/metrics`.

How: Use `prometheus_fastapi_instrumentator` for automatic request latency and count histograms. Add custom metrics: `rag_retrieval_latency_seconds` (histogram), `rag_tokens_total` (counter with `model` and `direction` labels), `rag_cache_hits_total` (counter), `rag_hallucination_flags_total` (counter), `rag_fallback_activations_total` (counter). These metrics give an ops team everything they need to set alerts.

### 6.3 — Caching

What: Two-level cache that reduces both latency and cost.

How: Level 1 — embedding cache: hash each document chunk's content, store its embedding to a local file. Skip model call on re-ingestion. Level 2 — query response cache: hash the normalized query string, store the full response. Cache TTL 1 hour (queries about documentation may become stale). Use `diskcache` locally. Document that production would replace with Redis for shared cache across instances.

### 6.4 — Guardrails

What: Input and output safety checks.

How: Input: strip leading/trailing whitespace, reject queries over 1000 characters, detect and reject obvious prompt injection patterns (instructions telling the system to ignore its instructions). Output: run a simple PII scan on the generated answer (regex for email, phone, SSN patterns) and redact before returning. These don't need to be perfect — demonstrating awareness of the threat surface is the signal.

---

## Phase 7 — Evaluation and Benchmarking

**Goal:** Produce evidence-based quality metrics, not just qualitative claims.

### 7.1 — Gold evaluation dataset

What: A curated set of 20–30 question-answer-source triples for objective measurement.

How: Write them manually based on the ingested documents. Each entry has a `question`, an `expected_answer` (summary form), and `expected_sources` (list of document names that should be retrieved). Keep this in `eval/gold_dataset.json`.

### 7.2 — RAGAS evaluation

What: Automated scoring of the RAG pipeline on faithfulness, context precision, context recall, and answer relevance.

How: Run the gold dataset through the pipeline, collect the generated answer + retrieved context for each question, pass both to RAGAS's `evaluate()` function. Include the scores in `docs/evaluation_report.md`. Target faithfulness ≥ 0.75 and context precision ≥ 0.7 as realistic baselines for a well-tuned system.

### 7.3 — Latency and throughput benchmarks

What: Concrete latency numbers for each pipeline stage.

How: Instrument 50 sample queries and collect per-stage timing. Report p50 and p95 for: embedding latency, FAISS retrieval latency, reranker latency, generation latency, total end-to-end. Include a table in the docs. Realistic targets: retrieval p95 < 200ms, generation p95 < 6s, total p95 < 7s.

### 7.4 — Adversarial testing

What: Demonstrate the system handles failure modes gracefully.

How: Test three adversarial scenarios: (1) query with no relevant documents in the index — expect a "no context found" response, not a hallucinated answer. (2) prompt injection attempt ("Ignore all previous instructions and...") — expect the guardrail to catch it. (3) very long query (>1000 chars) — expect a 422 validation error. Document results in the eval report.

---

## Phase 8 — Deployment Plan Document

**Goal:** A credible production architecture document that demonstrates MLOps maturity without needing a live deployment.

### 8.1 — GPU infrastructure design

What: Instance choice, justification, and capacity model.

How: Recommend AWS `g4dn.xlarge` (T4 16GB, ~$0.526/hr on-demand, ~$0.31/hr spot). Phi-3-mini in 4-bit requires ~4GB VRAM, leaving headroom for batching. One instance handles ~10 concurrent requests with batching enabled. Auto-scaling group with min 1, max 4 instances. Scale-out trigger: GPU utilization > 70% for 2 minutes.

### 8.2 — Request flow and load balancing

What: How requests move through the production system.

How: AWS Application Load Balancer → ECS Fargate tasks (API / ingestion workers, CPU-only) → SQS queue → GPU inference workers (EC2 Auto Scaling Group). Separation is important: the ingestion pipeline and the query API are different services with different scaling needs. The inference worker pulls from the SQS queue, processes, and writes the result to a shared Redis store that the API polls.

### 8.3 — Cost per request estimate

What: A back-of-envelope calculation that shows cost awareness.

How: On-demand: $0.526/hr ÷ ~1000 requests/hr = ~$0.00053/request for inference. Add embedding cost (~$0 for self-hosted BGE-small). Add OpenAI fallback cost at ~$0.002/1k tokens, ~500 tokens/request = ~$0.001/request when fallback activates. Total estimated cost: $0.001–$0.002/request depending on fallback rate.

### 8.4 — Latency expectations

What: End-to-end latency targets for production.

How: Retrieval: 50–150ms (FAISS in-memory + reranker). Model loading: 0ms (model warm, stays in GPU memory). Generation: 2–5s (20–30 tokens/sec, ~100 token response). Total: 2.5–6s. With streaming enabled, time-to-first-token is ~500ms. Cache hits: <50ms. Document that the SLA target is p95 < 8s with streaming, p95 < 500ms for cache hits.

### 8.5 — CI/CD and model update flow

What: How the system updates without downtime.

How: GitHub Actions pipeline: on push to `main` → run unit tests → build Docker image → push to ECR → rolling update of ECS tasks. For model updates: new LoRA adapter pushed to HF Hub → model version tag bumped in `config.yaml` → rolling restart picks up new adapter. For vector store updates: re-index runs as a separate job on document change, writes to a new index file, then atomically swaps the active index file path. No downtime.

---

## Phase 9 — Final Packaging

**Goal:** Make the reviewer's job easy. A polished README is the difference between a 7 and a 9.

### 9.1 — README structure

Sections: Project Overview → Architecture Diagram → Stack and Rationale → Setup Instructions (local with fallback, and Colab for fine-tuning) → API Reference (endpoints, request/response examples) → Evaluation Results → Deployment Plan Summary → Known Limitations → Future Improvements.

The "Known Limitations" section is a senior-level signal — it shows honest engineering judgment. Examples: FAISS does not scale horizontally, fine-tuning was done on a general dataset rather than domain-specific data, no streaming implemented, no multi-turn conversation memory.

The "Future Improvements" section shows strategic thinking: streaming responses, user feedback loop for answer quality, multi-modal document support, semantic caching with approximate match rather than exact hash, A/B testing infrastructure for model variants.

### 9.2 — Repository structure

```
/
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   ├── models/
│   └── core/
├── notebooks/
│   └── fine_tuning_phi3_qlora.ipynb
├── eval/
│   ├── gold_dataset.json
│   ├── run_ragas.py
│   └── evaluation_report.md
├── docs/
│   ├── architecture.png
│   ├── deployment_plan.md
│   └── tradeoffs.md
├── config.yaml
├── requirements.txt
└── README.md
```

### 9.3 — Tradeoffs document

Write `docs/tradeoffs.md` with 5–6 explicit architectural tradeoffs: FAISS vs Pinecone, RAG vs pure fine-tuning, LoRA vs full fine-tuning, async vs sync generation, exact-match cache vs semantic cache, self-hosted embeddings vs OpenAI embeddings. For each: state the choice made, what was traded away, and under what conditions you would choose differently. This is the single highest-value document for Leadership Thinking scoring — it proves you understand why you made every decision, not just how to implement it.

---

## Execution order summary

| Phase | Focus | Free cloud tool |
|---|---|---|
| 0 | Architecture + repo setup | GitHub, Excalidraw |
| 1–2 | Ingestion + embedding + vector store | Local (lightweight) |
| 3 | RAG chain + hallucination control | Local |
| 4 | Fine-tuning + evaluation | Google Colab T4 |
| 5 | Inference API + fallback routing | Local |
| 6 | Observability + caching + guardrails | Local |
| 7 | Benchmarks + adversarial testing | Local + Colab |
| 8 | Deployment plan document | Docs only |
| 9 | README + tradeoffs + packaging | GitHub |

The key strategic principle throughout: every implementation decision should be documented with a "why," not just a "what." The rubric rewards engineering judgment at every level, and that judgment is most visible in comments, docs, and the tradeoffs document — not in the complexity of the code itself.