# Production Deployment Plan — Enterprise RAG Assistant

**Version:** 1.0.0  
**Date:** 2025-01-01  
**Status:** Design Complete — Ready for Implementation  
**Audience:** Engineering leads, DevOps, MLOps, Cloud architects

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [GPU Infrastructure Design](#3-gpu-infrastructure-design)
4. [Request Flow and Load Balancing](#4-request-flow-and-load-balancing)
5. [Cost Model](#5-cost-model)
6. [Latency Expectations and SLAs](#6-latency-expectations-and-slas)
7. [CI/CD and Model Update Flow](#7-cicd-and-model-update-flow)
8. [Observability and Alerting](#8-observability-and-alerting)
9. [Security Architecture](#9-security-architecture)
10. [Disaster Recovery](#10-disaster-recovery)
11. [Capacity Planning](#11-capacity-planning)
12. [Known Risks and Mitigations](#12-known-risks-and-mitigations)

---

## 1. Executive Summary

This document describes the production deployment architecture for the Enterprise
RAG Assistant — a system that serves natural language queries over the AWS
Well-Architected Framework documentation corpus using a fine-tuned Phi-3-mini
model with OpenAI fallback.

**Design constraints driving all architectural decisions:**

| Constraint | Impact |
|---|---|
| Fine-tuned model requires GPU inference | Dedicated GPU instances, not Fargate |
| Ingestion and query serving have different scaling needs | Separate services, separate scaling groups |
| Response latency SLA: p95 < 8s | Queue-based async architecture with cache bypass |
| High availability requirement: 99.5% uptime | Multi-AZ deployment, ALB health checks |
| Cost target: < $0.002/request at steady state | Spot instances for GPU, cache aggressively |
| Model updates must not cause downtime | Blue/green adapter swap, rolling restarts |

**Key architectural choices:**

- **GPU inference workers on EC2 Auto Scaling Group** (not Fargate — Fargate
  does not support GPU instances as of 2024)
- **ECS Fargate for the API and ingestion tiers** (CPU-only, scales independently)
- **SQS as the request queue** between API tier and GPU inference tier
- **ElastiCache Redis** for shared response cache and result passing
- **Application Load Balancer** for HTTPS termination and health-based routing

---

## 2. Architecture Overview

```
                                    ┌─────────────────────────────────────────┐
                                    │           AWS Account (Production)      │
                                    │                                         │
   ┌──────────┐    HTTPS    ┌───────┴───────┐                                 │
   │  Users   ├────────────►│     AWS ALB   │                                 │
   │(internal)│             │  (HTTPS :443) │                                 │
   └──────────┘             └───────┬───────┘                                 │
                                    │                                         │
                    ┌───────────────┼───────────────┐                         │
                    │               │               │                         │
             ┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐                  │
             │  API Task   │ │  API Task  │ │  API Task  │  ECS Fargate       │
             │  (CPU-only) │ │ (CPU-only) │ │ (CPU-only) │  Auto Scaling      │
             │  FastAPI    │ │  FastAPI   │ │  FastAPI   │  Min: 2, Max: 8    │
             └──────┬──────┘ └─────┬──────┘ └─────┬──────┘                  │
                    │              │               │                          │
                    └──────────────┼───────────────┘                         │
                                   │                                          │
                    ┌──────────────┼──────────────────┐                       │
                    │              │                  │                       │
             ┌──────▼──────┐ ┌────▼──────────┐ ┌────▼──────────┐            │
             │   Redis     │ │  SQS Queue    │ │   S3 Bucket   │            │
             │(ElastiCache)│ │  (inference   │ │  (FAISS index │            │
             │  Response   │ │   requests)   │ │   + adapters) │            │
             │  Cache      │ └────┬──────────┘ └───────────────┘            │
             └──────▲──────┘      │                                          │
                    │             │                                           │
             ┌──────┴──────┐ ┌────▼──────────────────────────────────┐      │
             │  Result     │ │         EC2 Auto Scaling Group          │      │
             │  written    │ │   GPU Inference Workers                 │      │
             │  to Redis   │ │   Instance: g4dn.xlarge (T4 16GB)      │      │
             └─────────────┘ │   Min: 1,  Max: 4                      │      │
                             │   Scaling: GPU util > 70% for 2min     │      │
                             │                                         │      │
                             │   ┌─────────────────────────────────┐  │      │
                             │   │ Phi-3-mini + LoRA adapter       │  │      │
                             │   │ Loaded at startup, warm in GPU  │  │      │
                             │   │ ~4GB VRAM (4-bit quantized)     │  │      │
                             │   └─────────────────────────────────┘  │      │
                             └───────────────────────────────────────┘      │
                                                                             │
             ┌───────────────────────────────────────────────────────┐      │
             │  Ingestion Service (ECS Fargate, separate task def)   │      │
             │  Triggered by: S3 event (new PDF) or POST /ingest     │      │
             │  Outputs: updated FAISS index written to S3           │      │
             └───────────────────────────────────────────────────────┘      │
                                                                             │
             ┌───────────────────────────────────────────────────────┐      │
             │  Observability Stack                                   │      │
             │  CloudWatch Logs (structured JSON) → Log Insights      │      │
             │  Prometheus /metrics → Grafana dashboards             │      │
             │  CloudWatch Alarms → SNS → PagerDuty                  │      │
             └───────────────────────────────────────────────────────┘      │
                                                                             │
                                    └─────────────────────────────────────────┘
```

---

## 3. GPU Infrastructure Design

### 3.1 Instance Selection

**Recommended instance: `g4dn.xlarge`**

| Attribute | Value | Rationale |
|---|---|---|
| GPU | NVIDIA T4 (16GB VRAM) | Phi-3-mini in 4-bit needs ~4GB → 4x headroom |
| vCPU | 4 | Sufficient for request parsing, embedding, FAISS search |
| RAM | 16GB | FAISS index (~500MB) + OS overhead + response buffers |
| Network | Up to 25 Gbps | Low latency for S3 index loading at startup |
| On-demand price | ~$0.526/hr | $378/month per instance |
| Spot price | ~$0.157/hr | 70% discount; use for non-critical inference |
| Storage | 125GB NVMe SSD | Fast local storage for FAISS index cache |

**Why not g4dn.2xlarge or p3.2xlarge?**

`g4dn.2xlarge` has the same T4 GPU but 2x CPU and RAM. Overkill — the bottleneck
is GPU throughput, not CPU. Extra cost ($0.752/hr) is not justified.

`p3.2xlarge` has a V100 GPU (16GB) which is faster than T4 but costs $3.06/hr —
5.8x more for 2-3x faster inference. Not cost-effective for Phi-3-mini at this scale.

`g5.xlarge` (A10G GPU, 24GB) is the next logical upgrade if VRAM becomes the
constraint (e.g., moving to a 7B+ model without quantization). Cost: $1.006/hr.

**VRAM Budget on g4dn.xlarge:**

```
Phi-3-mini (4-bit NF4):          ~2.0 GB
LoRA adapter (bfloat16):          ~0.5 GB
Inference activations (batch=1):  ~1.5 GB
KV cache (4k context window):     ~1.0 GB
CUDA runtime overhead:            ~0.5 GB
─────────────────────────────────────────
Total estimated:                  ~5.5 GB
Available headroom:               ~10.5 GB (for batching, future models)
```

### 3.2 Auto Scaling Group Configuration

```yaml
# CloudFormation / Terraform equivalent
AutoScalingGroup:
  name: rag-gpu-inference-asg
  launch_template: rag-gpu-worker-lt
  
  capacity:
    min: 1          # Always keep 1 warm instance — cold start is 3-5 min
    max: 4          # 4 instances = ~40 concurrent requests with batching
    desired: 1      # Start at 1, scale out on demand
  
  availability_zones:
    - us-east-1a
    - us-east-1b    # Multi-AZ for fault tolerance
  
  scaling_policies:
    scale_out:
      metric: GPUUtilization        # Custom CloudWatch metric from DCGM
      threshold: 70                 # Scale out above 70% GPU utilization
      evaluation_periods: 4         # Must exceed threshold for 4 consecutive 30s periods
      period_seconds: 30            # = 2 minutes sustained before scaling
      cooldown_seconds: 300         # 5 min cooldown — model loading takes 3-5 min
      adjustment: +1               # Add 1 instance at a time
    
    scale_in:
      metric: GPUUtilization
      threshold: 20                 # Scale in below 20% GPU utilization
      evaluation_periods: 10        # Must be below threshold for 5 minutes
      period_seconds: 30
      cooldown_seconds: 600         # 10 min cooldown to avoid thrashing
      adjustment: -1
  
  instance_config:
    spot_mixed_instances:
      on_demand_base_capacity: 1    # Always keep 1 on-demand (reliability)
      on_demand_percentage: 25      # 25% on-demand, 75% spot
      spot_allocation_strategy: capacity-optimized  # Minimize interruptions
      spot_instance_pools: 3        # Use 3 instance type pools for availability
```

**Why GPU utilization (not CPU or request count) as the scaling trigger?**

CPU utilization reflects API server load, not inference load. GPU utilization
directly measures whether the model is under pressure. A spike in requests that
hits the cache won't increase GPU utilization — we don't want to scale out for
cache hits. Only genuine inference demand should trigger scale-out.

**GPU utilization collection:**
NVIDIA DCGM (Data Center GPU Manager) exports GPU metrics to CloudWatch via a
CloudWatch agent plugin. The metric name is `DCGM_FI_DEV_GPU_UTIL`.

### 3.3 Model Loading Strategy

The fine-tuned model is loaded ONCE at worker startup and kept warm in GPU memory
for the lifetime of the instance. Loading time is 3-5 minutes at first start.

```
Worker startup sequence:
  1. Pull Docker image from ECR (~2 min, cached after first pull)
  2. Download FAISS index from S3 to local NVMe (~30s for ~500MB index)
  3. Load Phi-3-mini base weights from HuggingFace Hub (~2 min for ~2.2GB)
  4. Load LoRA adapter from HuggingFace Hub (~10s for ~60MB adapter)
  5. Apply adapter, set model to eval() mode
  6. Send readiness signal to ASG health check
  7. Begin pulling from SQS queue
```

**Cold start mitigation:**
The minimum capacity of 1 ensures there is always at least one warm instance.
New instances joining the ASG on scale-out take 3-5 minutes to become ready.
Requests arriving during scale-out continue being served by existing instances
(with increased latency from queue depth). The SQS queue absorbs burst demand.

---

## 4. Request Flow and Load Balancing

### 4.1 Service Separation Rationale

The system is split into three independently-scalable services:

```
┌─────────────────┐    ┌─────────────────────┐    ┌────────────────────┐
│   API Service   │    │  Inference Service  │    │ Ingestion Service  │
│  (ECS Fargate)  │    │  (EC2 ASG + GPU)    │    │  (ECS Fargate)     │
├─────────────────┤    ├─────────────────────┤    ├────────────────────┤
│ Handles:        │    │ Handles:            │    │ Handles:           │
│ - Input valid.  │    │ - Model inference   │    │ - PDF loading      │
│ - Auth (future) │    │ - SQS polling       │    │ - Chunking         │
│ - Cache lookups │    │ - Redis writes      │    │ - Embedding        │
│ - SQS writes    │    │ - GPU batching      │    │ - FAISS building   │
│ - Response poll │    │                     │    │ - S3 uploads       │
├─────────────────┤    ├─────────────────────┤    ├────────────────────┤
│ Scaling:        │    │ Scaling:            │    │ Scaling:           │
│ CPU/request     │    │ GPU utilization     │    │ Triggered on       │
│ Min: 2, Max: 8  │    │ Min: 1, Max: 4      │    │ document change    │
│ Cost: ~$50/mo   │    │ Cost: ~$400/mo      │    │ Run-to-completion  │
└─────────────────┘    └─────────────────────┘    └────────────────────┘
```

**Why separate API and inference?**

If inference runs inside the API container (Fargate does not support GPU), we
cannot use GPU at all on Fargate. Separating them allows:
- API tier on Fargate (no GPU needed for routing, validation, caching)
- Inference tier on EC2 with GPU (only the component that needs GPU has GPU)
- Independent scaling: API scales on request count, GPU scales on inference load
- Independent deployments: API updates do not restart the GPU workers

### 4.2 Request Flow — Cache Miss (Full Pipeline)

```
Step 1: Client → ALB (HTTPS :443)
  ALB terminates TLS using ACM certificate.
  Routes to healthy ECS Fargate task based on least-outstanding-requests algorithm.

Step 2: ALB → ECS Fargate (API Task)
  POST /generate received.
  Input validation (Pydantic schema).
  Input guardrail check (injection detection, length).
  L2 cache lookup (Redis):
    → HIT: return cached response immediately (< 50ms total)
    → MISS: continue to Step 3

Step 3: API Task → SQS (Inference Request Queue)
  API Task writes inference request to SQS:
    {
      "request_id": "uuid",
      "query": "...",
      "top_k": 5,
      "use_fine_tuned": true,
      "timestamp": "...",
      "result_key": "result:{request_id}"  ← Redis key for result
    }
  Visibility timeout: 90s (longer than max inference time of 30s)
  Message retention: 60s (expired messages not worth processing)

Step 4: SQS → GPU Inference Worker
  Worker polls SQS (long polling, 20s wait).
  On message receipt:
    a. Retrieval: embed query → FAISS search → rerank (in worker, CPU/GPU)
    b. Context check: if no results above threshold → write refusal to Redis
    c. Generation: format prompt → run model.generate() on GPU → parse output
    d. Post-processing: grounding check, PII redaction, confidence parsing
    e. Write full response to Redis at key "result:{request_id}"
       SET result:{request_id} {response_json} EX 120  ← 2 min TTL
    f. Delete message from SQS

Step 5: API Task ← Redis (Result Polling)
  API Task polls Redis for "result:{request_id}" every 250ms.
  Timeout: 35s (5s buffer above generation timeout).
  On result found: return response to client.
  On timeout: return 504 Gateway Timeout.

Step 6: API Task → L2 Cache Write
  If response is successful and not is_refusal:
    Write to Redis query response cache with 1hr TTL.

Step 7: API Task → Client
  Return full RAGResponse JSON.
  Total wall clock: 2.5-6s (cache miss, fine-tuned model)
                    0.05-0.5s (cache hit)
```

### 4.3 Request Flow — Cache Hit (Bypass Path)

```
Client → ALB → ECS Fargate API Task
  → Redis cache lookup → HIT
  → Return cached response (< 50ms)
  (SQS, GPU workers, and model never involved)
```

This path handles repeated identical queries without any GPU cost.
Target: achieve 20%+ cache hit rate through natural query repetition
(FAQ-style knowledge base queries repeat frequently).

### 4.4 ALB Configuration

```yaml
ALB:
  scheme: internal              # Internal ALB — not internet-facing
                                # (engineering team use case, VPN access)
  listeners:
    - port: 443
      protocol: HTTPS
      certificate: arn:aws:acm:us-east-1:...  # ACM certificate
      default_action: forward to target_group
    - port: 80
      protocol: HTTP
      default_action: redirect to HTTPS 443   # Force HTTPS
  
  target_group:
    protocol: HTTP
    port: 8000                  # FastAPI port inside container
    health_check:
      path: /health/ready       # Readiness check (not just /health)
      interval: 30s
      timeout: 5s
      healthy_threshold: 2
      unhealthy_threshold: 3
      # /health/ready returns 503 if vector store empty — ALB won't route
      # traffic to tasks that haven't loaded the index yet
    
    deregistration_delay: 30s   # Drain in-flight requests before termination
    
    stickiness: disabled        # Sessions are stateless — no stickiness needed
```

### 4.5 SQS Queue Configuration

```yaml
SQS:
  queue_name: rag-inference-requests.fifo
  type: Standard                # Standard (not FIFO) — order doesn't matter
                                # Standard has higher throughput than FIFO
  
  visibility_timeout: 90s       # Time a worker has to process a message
                                # Must be > max inference time (30s) + buffer
  
  message_retention: 120s       # Delete messages after 2 minutes
                                # A 2-minute-old inference request is stale —
                                # the client has already timed out
  
  dead_letter_queue:
    name: rag-inference-dlq
    max_receive_count: 2        # After 2 failed processing attempts, send to DLQ
    retention: 24h              # Analyze DLQ messages for debugging
  
  long_polling: 20s             # Workers wait 20s for a message before re-polling
                                # Reduces empty-receive API calls (cost + latency)
```

---

## 5. Cost Model

### 5.1 Infrastructure Cost Breakdown

*Assumptions: 10,000 requests/day, 70% cache hit rate, 1 GPU instance on-demand*

#### Fixed Monthly Costs (always-on infrastructure)

| Component | Instance/Config | Cost/hr | Monthly Cost |
|---|---|---|---|
| GPU inference worker (1 instance, min capacity) | g4dn.xlarge on-demand | $0.526 | $379 |
| ECS Fargate API tasks (2 tasks, always on) | 1 vCPU, 2GB RAM each | $0.072 | $104 |
| ElastiCache Redis | cache.t3.micro | $0.017 | $12 |
| Application Load Balancer | — | $0.008 | $22 |
| SQS (inference queue) | ~$0.40/million requests | — | $3 |
| S3 (FAISS index, models) | ~50GB | — | $2 |
| CloudWatch Logs | ~10GB/month | — | $5 |
| **Fixed total** | | | **~$527/month** |

#### Variable Costs (per-request, 30% cache miss rate)

*10,000 req/day × 0.30 miss rate = 3,000 GPU inferences/day = 90,000/month*

| Component | Per-Request Cost | Monthly (90k inferences) |
|---|---|---|
| GPU compute (g4dn.xlarge) | $0.526/hr ÷ 600 req/hr = $0.00088 | $79 |
| OpenAI fallback (when primary fails, est. 5% of inferences) | $0.00060 per 500 tokens | $3 |
| Embedding (BGE-small, self-hosted, included in GPU cost) | $0 | $0 |
| **Variable total** | | **~$82/month** |

#### Total Monthly Cost

| Scenario | Cost |
|---|---|
| Fixed infrastructure | $527/month |
| Variable (90k GPU inferences) | $82/month |
| **Total** | **~$609/month** |
| Per-request cost (all requests including cache hits) | $609 / 300,000 requests = **$0.0020/request** |
| Per-request cost (GPU inferences only, cache misses) | ($527 + $82) / 90,000 = **$0.0068/inference** |

### 5.2 Cost Optimization Levers

**Lever 1: Spot instances for GPU workers (highest impact)**

```
Current (on-demand):  g4dn.xlarge = $0.526/hr
Spot price (typical): g4dn.xlarge = $0.157/hr  (70% discount)
Monthly saving on 1 instance: ($0.526 - $0.157) × 720 = $266/month

Risk: Spot interruption (2-minute warning).
Mitigation: Keep 1 on-demand instance (min_capacity=1),
            use spot for additional instances only.
Recommended mix: 25% on-demand, 75% spot for scale-out instances.
```

**Lever 2: Increase cache hit rate**

```
Current: 70% cache hit rate (assumed)
Cache hits cost: $0 GPU, $0.00001 Redis lookup
Each 1% increase in hit rate saves ~900 GPU calls/month

Improvements:
  - Semantic cache (approximate match): catch "what is HA?" and
    "what does HA mean?" as the same cache entry
  - Increase cache TTL from 1hr to 24hr (documentation is stable)
  - Pre-populate cache with top-50 most common queries at deployment
```

**Lever 3: Request batching on GPU worker**

```
Current: batch_size = 1 (one request per model.generate() call)
Opportunity: batch_size = 4 processes 4 requests in ~1.5x the time
             of 1 request (GPU parallelism)

Throughput improvement: 4x requests per GPU-hour → 4x cost reduction
Latency cost: batch waits for 4 requests OR 500ms timeout (whichever first)
Acceptable for non-streaming use case.
Implementation: worker collects messages from SQS up to batch_size,
                pads to equal length, calls model.generate() once.
```

**Lever 4: Scheduled scaling**

```
Engineering team usage pattern: 9am-6pm weekdays (90hrs/week)
Off-hours: min_capacity=0 (scale to zero)
On-hours: min_capacity=1

Monthly saving: 720 total hours - 390 "on" hours = 330 off-hours
At g4dn.xlarge: 330 × $0.526 = $174/month saved
Tradeoff: First request after off-hours hits cold start (3-5 min wait)
Mitigation: Scheduled scale-up at 8:55am via EventBridge
```

### 5.3 Back-of-Envelope Cost Per Request

```
Scenario A: Cache hit (70% of requests)
  GPU cost:        $0.000
  Redis lookup:    $0.00001
  ────────────────────────
  Total:           $0.00001/request

Scenario B: Cache miss, fine-tuned model (28.5% of requests)
  GPU compute:     $0.526/hr ÷ 600 req/hr = $0.00088
  FAISS retrieval: included in GPU instance cost
  BGE-small embed: included in GPU instance cost
  ────────────────────────
  Total:           $0.00088/request

Scenario C: Cache miss, OpenAI fallback (1.5% of requests)
  OpenAI input:   ~400 tokens × $0.00015/1k = $0.00006
  OpenAI output:  ~500 tokens × $0.00060/1k = $0.00030
  API overhead:   included in Fargate cost
  ────────────────────────
  Total:           $0.00036/request

Weighted average:
  (0.70 × $0.00001) + (0.285 × $0.00088) + (0.015 × $0.00036)
  = $0.000007 + $0.000251 + $0.0000054
  = $0.000263/request (variable cost only)

Adding fixed cost allocation:
  $527/month ÷ 300,000 requests/month = $0.00176 fixed/request
  Total: $0.000263 + $0.00176 = $0.00202/request
```

---

## 6. Latency Expectations and SLAs

### 6.1 Per-Stage Latency Targets (Production, No Cache)

| Stage | Description | p50 | p95 | Budget |
|---|---|---|---|---|
| Network (client → ALB) | TLS handshake + routing | 5ms | 15ms | — |
| Input validation | Pydantic + guardrails | 1ms | 3ms | — |
| L2 cache lookup | Redis GET | 2ms | 8ms | — |
| Query embedding | BGE-small on CPU/GPU | 20ms | 60ms | — |
| FAISS retrieval | In-memory similarity search | 10ms | 30ms | — |
| Reranker (optional) | CrossEncoder on 5 candidates | 40ms | 120ms | — |
| **Retrieval subtotal** | Embed + FAISS + rerank | **70ms** | **200ms** | ≤ 200ms |
| Prompt construction | Context formatting | 1ms | 3ms | — |
| **Model generation** | Phi-3-mini on T4 GPU | **2,000ms** | **5,000ms** | ≤ 5,000ms |
| Post-processing | Grounding check + PII scan | 30ms | 80ms | — |
| Redis result write | SET with TTL | 2ms | 10ms | — |
| Result polling | Redis GET (API task polls) | 250ms | 500ms | — |
| **Total (no cache)** | End-to-end wall clock | **2,400ms** | **6,000ms** | ≤ 7,000ms |
| **Total (cache hit)** | Redis lookup + return | **8ms** | **45ms** | ≤ 50ms |

**Notes on generation latency:**
- T4 GPU generates approximately 20-30 tokens/second for Phi-3-mini in 4-bit
- A typical RAG response is 150-300 tokens
- p50 generation: 150 tokens ÷ 25 tok/s = 6s (without streaming)
- With streaming (time-to-first-token): ~500ms to first token, then 40ms/token

### 6.2 SLA Definitions

```
SLA Tier 1 — Cache Hit Response
  Definition:  p95 < 50ms wall clock
  Measurement: /metrics → rag_total_latency_seconds{cache_hit="true"} p95
  Alert:       p95 > 100ms for 5 consecutive minutes

SLA Tier 2 — Non-Cache Response (with fine-tuned model)
  Definition:  p95 < 7,000ms end-to-end
  Measurement: /metrics → rag_total_latency_seconds{cache_hit="false"} p95
  Alert:       p95 > 8,000ms for 5 consecutive minutes

SLA Tier 3 — Availability
  Definition:  99.5% of requests return non-5xx response
  Measurement: 1 - (rag_requests_total{status="error"} / rag_requests_total)
  Alert:       Error rate > 1% for 2 consecutive minutes

SLA Tier 4 — Streaming Time-to-First-Token (future)
  Definition:  p95 < 800ms time to first token
  Not implemented in v1 — streaming requires SSE endpoint changes
```

### 6.3 Latency Budget Analysis

```
Why 7s total budget with a 30s timeout?

  The 30s asyncio.timeout() in the API is a safety valve — it prevents
  a single stuck model call from holding a connection open indefinitely.

  The 7s SLA is the operating target for normal conditions.
  The gap (7s SLA → 30s timeout) covers:
    - Slow requests on overloaded GPU workers
    - Long prompts (complex cross-pillar questions)
    - OpenAI API latency spikes (fallback path)

  Expected p99 latency: ~10s (within timeout, above SLA)
  Requests between 7s and 10s: still served, SLA miss logged as metric
  Requests above 30s: 504 returned, client retries

Streaming as a latency perception improvement:
  Without streaming: user waits 4s, sees full answer appear at once.
  With streaming (SSE): user sees first tokens at 500ms, feels responsive.
  Actual total latency is identical, but perceived latency is much lower.
  Streaming is recommended as a v2 feature.
```

---

## 7. CI/CD and Model Update Flow

### 7.1 Application Deployment Pipeline

```
                    ┌─────────────────────────────────────────────────┐
                    │           GitHub Actions Pipeline                │
                    │                                                  │
  Developer         │  Trigger: push to main branch                   │
  pushes code  ────►│                                                  │
                    │  Step 1: Run unit tests                          │
                    │    pytest tests/ -x --tb=short                  │
                    │    Fail fast on first test failure               │
                    │                                                  │
                    │  Step 2: Run integration tests (mocked)         │
                    │    pytest tests/test_api.py                      │
                    │                                                  │
                    │  Step 3: Security scan                           │
                    │    bandit -r backend/ (Python security linter)   │
                    │    safety check (dependency vulnerability scan)  │
                    │                                                  │
                    │  Step 4: Build Docker image                      │
                    │    docker build -t rag-api:${GITHUB_SHA} .       │
                    │                                                  │
                    │  Step 5: Push to ECR                             │
                    │    docker tag rag-api:${GITHUB_SHA}              │
                    │          ${ECR_URI}:${GITHUB_SHA}                │
                    │    docker push ${ECR_URI}:${GITHUB_SHA}          │
                    │    docker tag ... ${ECR_URI}:latest              │
                    │                                                  │
                    │  Step 6: Update ECS service (API tier)           │
                    │    aws ecs update-service                        │
                    │      --cluster rag-production                    │
                    │      --service rag-api                           │
                    │      --force-new-deployment                      │
                    │    ECS performs rolling update:                  │
                    │      - Launch new task with new image            │
                    │      - Wait for /health/ready → 200              │
                    │      - Drain old task (30s deregistration delay) │
                    │      - Terminate old task                        │
                    │      - Repeat for each task (one at a time)     │
                    │                                                  │
                    │  Step 7: Smoke test (post-deploy)               │
                    │    curl /health/ready → 200                      │
                    │    curl POST /generate {"query": "test"} → 200  │
                    │    Fail pipeline if smoke test fails             │
                    │                                                  │
                    └─────────────────────────────────────────────────┘
```

### 7.2 GitHub Actions Workflow File

```yaml
# .github/workflows/deploy.yml
name: Build, Test, Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: rag-assistant
  ECS_SERVICE: rag-api
  ECS_CLUSTER: rag-production

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run unit tests
        run: |
          pytest tests/ -x --tb=short -q \
            --ignore=tests/test_api.py  # integration tests need mocked services
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_TEST }}
      
      - name: Run integration tests
        run: pytest tests/test_api.py -x --tb=short -q
      
      - name: Security scan
        run: |
          pip install bandit safety
          bandit -r backend/ -ll -q
          safety check --full-report

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Login to ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      
      - name: Build and push Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
                     $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT
      
      - name: Deploy to ECS (rolling update)
        run: |
          aws ecs update-service \
            --cluster ${{ env.ECS_CLUSTER }} \
            --service ${{ env.ECS_SERVICE }} \
            --force-new-deployment \
            --deployment-configuration \
              "maximumPercent=200,minimumHealthyPercent=100"
          
          # Wait for deployment to complete (max 10 minutes)
          aws ecs wait services-stable \
            --cluster ${{ env.ECS_CLUSTER }} \
            --services ${{ env.ECS_SERVICE }}
      
      - name: Smoke test
        run: |
          API_URL="${{ secrets.PRODUCTION_API_URL }}"
          
          # Health check
          HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
            ${API_URL}/health/ready)
          if [ "$HTTP_STATUS" != "200" ]; then
            echo "Health check failed: HTTP ${HTTP_STATUS}"
            exit 1
          fi
          
          # Generate endpoint check
          RESPONSE=$(curl -s -X POST ${API_URL}/generate \
            -H "Content-Type: application/json" \
            -d '{"query": "What is the AWS Well-Architected Framework?"}')
          
          if echo "$RESPONSE" | grep -q '"is_refusal": false'; then
            echo "Smoke test passed"
          else
            echo "Smoke test failed: unexpected response"
            echo "$RESPONSE"
            exit 1
          fi
```

### 7.3 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY backend/ ./backend/
COPY config.yaml .

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--access-log"]
```

### 7.4 Model Update Flow (Zero-Downtime Adapter Swap)

When a new fine-tuned adapter is ready (after re-training in Colab):

```
Step 1: Train new adapter
  - Run fine-tuning notebook in Colab
  - New adapter pushed to HF Hub as:
    {username}/phi3-mini-enterprise-qlora-v2

Step 2: Update version reference in config.yaml
  git diff config.yaml:
  - adapter_repo: "username/phi3-mini-enterprise-qlora"
  + adapter_repo: "username/phi3-mini-enterprise-qlora-v2"

Step 3: Run evaluation against new adapter (before promoting)
  python eval/run_ragas.py \
    --adapter-override username/phi3-mini-enterprise-qlora-v2
  # Confirm faithfulness >= 0.75, no regression vs previous version

Step 4: Commit config change → triggers CI/CD pipeline
  git add config.yaml
  git commit -m "model: promote adapter v2 to production"
  git push origin main

Step 5: Rolling restart picks up new adapter
  GitHub Actions builds new Docker image (same code, updated config.yaml)
  ECS rolling update:
    - New task starts, loads adapter v2 from HF Hub during startup
    - /health/ready returns 200 when model loaded
    - ALB routes traffic to new task
    - Old task drained and terminated
  Total downtime: 0 (rolling update, min_healthy_percent=100)

Rollback (if new adapter regresses):
  git revert HEAD
  git push origin main
  CI/CD automatically re-deploys previous config
  Previous adapter loaded within 5 minutes
```

### 7.5 Vector Store Update Flow (Zero-Downtime Index Swap)

When new documents are added to the corpus:

```
Step 1: New PDF added to S3 source bucket
  s3://rag-documents/pdfs/new_pillar_doc.pdf

Step 2: S3 event triggers ingestion Lambda (or ECS task)
  EventBridge rule: s3:ObjectCreated → trigger ingestion task

Step 3: Ingestion task runs (separate from API tasks)
  - Downloads all PDFs from S3
  - Runs DocumentIngestionPipeline (Phase 1)
  - Embeds new chunks only (deduplication skips existing)
  - Builds new FAISS index at path: s3://rag-index/index-{timestamp}/

Step 4: Atomic index swap (no downtime)
  - Write new index to S3: s3://rag-index/index-2025-01-15/
  - Update SSM Parameter Store pointer:
      /rag/active-index-path = "s3://rag-index/index-2025-01-15/"
  - API tasks poll SSM for pointer changes every 60s
  - On pointer change: download new index, hot-swap in memory
  - No request interruption: old index serves until new one is ready

Why SSM Parameter Store for the pointer?
  - Atomic write operation (no partial updates)
  - All API tasks see the same pointer simultaneously
  - Change history for audit/rollback
  - No S3 polling per-request (SSM cached in memory)
```

---

## 8. Observability and Alerting

### 8.1 Metrics Stack

```
Application (/metrics endpoint) → Prometheus scrape (15s interval)
                                → Grafana dashboards
                                → Grafana alerting rules
                                → PagerDuty

ECS/EC2 metrics → CloudWatch
               → CloudWatch Alarms
               → SNS Topic
               → PagerDuty

Application logs (JSON to stdout) → CloudWatch Logs
                                  → CloudWatch Logs Insights (ad-hoc queries)
                                  → CloudWatch Metric Filters (extract metrics)
```

### 8.2 Alert Runbook

| Alert | Condition | Severity | First Response |
|---|---|---|---|
| High error rate | error_rate > 1% for 2min | P1 | Check CloudWatch Logs for error_type. Rollback if recent deploy. |
| Generation timeout spike | timeout_rate > 5% for 5min | P1 | Check GPU utilization. Increase ASG min capacity. |
| Primary model unavailable | fallback_activations rate > 50% for 5min | P2 | Check HF Hub reachability. Check adapter repo access. |
| Cache miss rate high | cache_hit_rate < 10% for 15min | P3 | Check Redis connectivity. Check cache TTL config. |
| Retrieval SLA miss | retrieval_p95 > 500ms for 10min | P2 | Check FAISS index size. Check if re-index is running. |
| Hallucination flag spike | hallucination_flag_rate > 15% for 10min | P2 | Check grounding_threshold config. Review prompt version. |
| GPU utilization ceiling | gpu_util > 95% for 5min | P1 | Force ASG scale-out. Check SQS queue depth. |

### 8.3 Grafana Dashboard Layout

```
Row 1 — Golden Signals
  Panel 1: Request rate (req/s) — line chart, 1h window
  Panel 2: Error rate (%) — line chart, threshold line at 1%
  Panel 3: p50/p95 total latency — dual-line chart
  Panel 4: Cache hit rate (%) — stat panel, target 20%

Row 2 — Pipeline Performance
  Panel 5: Retrieval p95 latency — gauge, red > 200ms
  Panel 6: Generation p95 latency — gauge, red > 5000ms
  Panel 7: Model routing (fine-tuned vs fallback %) — pie chart
  Panel 8: Hallucination flag rate — stat panel, red > 10%

Row 3 — Infrastructure
  Panel 9: GPU utilization (%) — line chart per instance
  Panel 10: ASG instance count — step chart
  Panel 11: SQS queue depth — line chart
  Panel 12: Vector store size — stat panel

Row 4 — Cost Signals
  Panel 13: Tokens consumed / hour — bar chart by model
  Panel 14: Estimated cost / hour — calculated metric
  Panel 15: Fallback activations / hour — bar chart
  Panel 16: Cache savings (avoided GPU calls) — calculated
```

---

## 9. Security Architecture

### 9.1 Network Security

```
VPC Design:
  Public subnets:  ALB only (internet-facing for internal users via VPN)
  Private subnets: ECS tasks, EC2 GPU workers, ElastiCache, SQS endpoint
  
  No internet access from private subnets (no NAT Gateway):
    - ECS tasks reach ECR via VPC endpoint
    - GPU workers reach S3 via VPC endpoint
    - GPU workers reach HF Hub for adapter download via NAT Gateway
      (one-time at startup, then cached locally)

Security Groups:
  sg-alb:      Ingress: 443 from 0.0.0.0/0 (or VPN CIDR)
               Egress: 8000 to sg-api-tasks
  
  sg-api-tasks: Ingress: 8000 from sg-alb
                Egress: 6379 to sg-redis (Redis)
                        443 to SQS VPC endpoint
                        443 to OpenAI API (NAT Gateway)
  
  sg-gpu-workers: Ingress: none (pulls from SQS, not incoming connections)
                  Egress: 443 to SQS VPC endpoint
                          6379 to sg-redis
                          443 to HF Hub (NAT Gateway, startup only)
  
  sg-redis: Ingress: 6379 from sg-api-tasks, sg-gpu-workers
```

### 9.2 Secrets Management

```
All secrets stored in AWS Secrets Manager (not environment variables in ECS):

  /rag/openai-api-key     → OPENAI_API_KEY
  /rag/hf-token           → HUGGINGFACE_TOKEN
  /rag/redis-auth-token   → Redis AUTH token

ECS Task Definition uses Secrets Manager references:
  {
    "name": "OPENAI_API_KEY",
    "valueFrom": "arn:aws:secretsmanager:us-east-1:...:secret:/rag/openai-api-key"
  }

Rotation: OpenAI API key rotated every 90 days via Secrets Manager rotation Lambda.
IAM: ECS task role has secretsmanager:GetSecretValue only for specific ARNs.
```

### 9.3 IAM Roles

```
ECS Task Role (api-task-role):
  - secretsmanager:GetSecretValue (specific secret ARNs)
  - sqs:SendMessage (inference queue only)
  - elasticache:Connect (Redis cluster)
  - s3:GetObject (FAISS index bucket, read-only)
  - ssm:GetParameter (/rag/active-index-path)

GPU Worker Role (gpu-worker-role):
  - sqs:ReceiveMessage, sqs:DeleteMessage (inference queue)
  - elasticache:Connect (Redis cluster)
  - s3:GetObject (FAISS index, model artifacts)
  - secretsmanager:GetSecretValue (HF token only)

Ingestion Role (ingestion-task-role):
  - s3:GetObject, s3:PutObject (document bucket + index bucket)
  - s3:ListBucket
  - ssm:PutParameter (/rag/active-index-path)
```

---

## 10. Disaster Recovery

### 10.1 Recovery Objectives

| Metric | Target | Mechanism |
|---|---|---|
| RTO (recovery time) | < 15 minutes | ASG auto-replacement, ECS task restart |
| RPO (data loss) | 0 minutes | FAISS index on S3 (durable), Redis is cache-only (no RPO requirement) |
| Availability | 99.5% | Multi-AZ deployment, ALB health checks |

### 10.2 Failure Scenarios

**Scenario 1: GPU worker instance failure**

```
Detection: ASG health check fails → marks instance unhealthy
           ALB stops routing to failed tasks
Response:  ASG terminates instance, launches replacement in different AZ
           New instance loads model (3-5 min cold start)
           SQS messages become visible again after visibility timeout (90s)
           Requests in flight: served by remaining healthy instance(s)
Impact:    Increased latency during replacement (~5 min)
           Zero data loss (S3 index intact, Redis cache intact)
```

**Scenario 2: Redis failure**

```
Detection: ElastiCache cluster health alarm
Response:  Application detects Redis connection failure
           L2 cache degrades gracefully (treat all as cache misses)
           Response cache writes fail silently (logged, not fatal)
           All requests served without cache (higher latency, higher cost)
           ElastiCache Multi-AZ failover promotes replica (~60s)
Impact:    ~60s of cache unavailability, then automatic recovery
           Cache warmed again within 1hr as requests re-populate it
```

**Scenario 3: SQS queue message poison pill**

```
Detection: Message processed multiple times without success → DLQ
Response:  DLQ alarm triggers notification
           Engineer inspects DLQ message (request that caused worker crash)
           Fix deployed if code issue; message deleted from DLQ
Impact:    Single request fails, all others continue normally
```

**Scenario 4: Fine-tuned model adapter unavailable (HF Hub outage)**

```
Detection: Worker startup fails to load adapter
           Fallback activation rate spikes above threshold
Response:  LLM Manager activates OpenAI fallback transparently
           All requests continue being served via OpenAI
           Higher cost but maintained availability
           Alert sent to team to investigate HF Hub status
Impact:    Higher cost (~$0.00036 vs $0.00088 per inference)
           No user-visible degradation (fallback is first-class)
```

---

## 11. Capacity Planning

### 11.1 Throughput Estimates

| Scenario | Requests/hr | GPU Instances Required | Cost/hr |
|---|---|---|---|
| Development / staging | 10 | 1 (shared) | $0.53 |
| Small team (10 engineers) | 100 | 1 | $0.53 |
| Medium team (50 engineers) | 500 | 1 (cache handles most) | $0.53 |
| Large team (200 engineers) | 2,000 | 2 | $1.05 |
| Enterprise (1,000 engineers) | 10,000 | 4 (max ASG) or scale vertically | $2.10 |

**Throughput calculation:**
- 1 GPU instance (g4dn.xlarge) generates ~600 tokens/minute for Phi-3-mini
- Average response: ~200 tokens → 3 responses/minute = 180 responses/hour
- With 70% cache hit rate: 180 / 0.30 = 600 effective requests/hour per instance
- With 4 instances: ~2,400 requests/hour

### 11.2 Scaling Beyond 4 GPU Instances

At > 2,400 requests/hour (or ~40 concurrent active users), the ASG reaches its
max of 4 instances. Options at this scale:

**Option A: Vertical scaling**
  Switch to `g4dn.12xlarge` (4× T4 GPUs, $3.912/hr). Use tensor parallelism
  to serve one model across 4 GPUs simultaneously. More complex operationally
  but allows serving larger models (13B+).

**Option B: Horizontal scaling beyond ASG max**
  Increase `max_size` in ASG. Linear cost scaling. Simple to implement.
  Limited by HF Hub download concurrency if many instances start simultaneously
  (mitigate with pre-cached model artifacts in ECR or S3).

**Option C: Replace FAISS with Pinecone**
  At very high request volumes, FAISS is replicated per worker instance (each
  has its own copy of the index in memory). Pinecone provides a shared index
  with managed horizontal scaling. Migration requires only implementing
  `PineconeVectorStoreService` (the abstraction layer is already in place).

---

## 12. Known Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| HuggingFace Hub outage | Low | High | OpenAI fallback activates automatically; pre-cache adapter in S3 |
| OpenAI API outage | Low | High | Without fallback, system returns 503; pre-negotiate enterprise SLA |
| Spot instance interruption | Medium | Low | On-demand base capacity=1; SQS absorbs burst; 2-min warning allows graceful drain |
| FAISS index corruption | Very Low | High | Index rebuilt from source PDFs in S3; takes ~10 min |
| Prompt injection bypass | Low | Medium | Defense-in-depth (regex + LLM grounding); monitor injection_attempts metric |
| Model quality regression after re-training | Medium | Medium | Evaluation gate in CI/CD; adapter versioning allows rollback |
| Cost overrun from fallback spike | Low | Medium | Fallback rate alert at 50%; CloudWatch cost alarm at $100/day |
| Cold start SLA miss | Medium | Low | Min capacity=1 ensures one warm instance; EventBridge pre-warm on schedule |

---

*This deployment plan assumes familiarity with AWS services. Infrastructure-as-code
templates (Terraform/CloudFormation) for all components described here are outside
the scope of this document but represent the recommended next step before production
deployment.*

*All cost figures are based on us-east-1 pricing as of 2024 and are subject to change.
Verify current pricing at https://aws.amazon.com/ec2/pricing/ before finalizing budgets.*