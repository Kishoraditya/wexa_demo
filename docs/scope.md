# Project Scope: Enterprise Cloud Architecture Assistant
**Domain:** AWS Architecture & Engineering Advisory 
**Date:** 2026-05-11 | **Version:** 1.0.0 | **Status:** Draft

## 1. Executive Summary
This project delivers a Retrieval-Augmented Generation (RAG) assistant designed to query enterprise architectural guidelines. The knowledge base is restricted to the official AWS Well-Architected Framework (Operational Excellence, Security, Reliability, Performance Efficiency, Cost Optimization, Sustainability). The system leverages a hybrid generation layer, utilizing a custom fine-tuned local model with automated fallback to commercial LLMs to ensure high availability.

## 2. Target Users & Expected Queries

**Primary users:** Software engineers, DevOps Engineers, Site Reliability Engineers (SREs) and cloud architects who are actively designing, maintaining or reviewing AWS-based systems. They are technically literate, familiar with AWS terminology, and do not need the assistant to explain basic concepts — they need fast, precise answers to specific architectural questions.

**Secondary users:** Engineering managers Solutions Architects, Technical Product Managers and technical leads conducting architecture reviews or preparing design documents. Their queries tend to be broader ("what does the framework say about fault isolation?") rather than lookup-style.

**What users are not:** This system is not designed for business stakeholders with no AWS background. It assumes the user can interpret a technical answer and evaluate whether a cited source is relevant.

**Expected Queries:** 
Users will ask complex, structural, and best-practice questions, such as:
* *"What are the trade-offs between Multi-AZ and Multi-Region deployments for reliability?"*
* *"How should I handle data classification boundaries under the Security pillar?"*
* *"What are the recommended design principles for maximizing instance performance efficiency?"*
* *"How do we manage break-glass access during an operational incident?"*
* *"What does the Reliability pillar say about designing for failure?"*
* *"How should I implement least-privilege access according to the Security pillar?"*
* *"What are the recommended practices for cost allocation tagging?"*
* *"How does the Performance Efficiency pillar define selection of compute types?"*
* *"What operational metrics does AWS recommend tracking for workload health?"*
* *"How do the Reliability and Operational Excellence pillars differ in their approach to incident response?"* 

## 3. Definition of Success and Failure
**A Successful Answer:**
1. **Grounded:** Derives 100% of its factual claims from the retrieved PDFs.
2. **Cited:** Explicitly links claims to the source document (e.g., *"According to the Reliability Pillar, page 14..."*).
3. **Refusal-Capable:** If asked about a topic not in the framework (e.g., Azure architecture, general Python programming), the system explicitly refuses to answer rather than speculating.
4. **Appropriately scoped:** The answer addresses the question without over-generalizing. If the question is about one pillar, the answer does not pad with tangentially related content from other pillars unless the connection is direct and stated.
5. **Honest about confidence:** The response carries a confidence signal (high / medium / low) that reflects both the retrieval quality and the model's self-assessed certainty. A medium-confidence answer is still a successful answer — it is honest.

**A Failure State:**
1. **Hallucination:** Recommending an AWS service or design pattern that is technically valid but *not* present in the retrieved chunks.
2. **Context Dilution:** Answering based on the LLM's pre-trained weights rather than the provided AWS context.
3. **Silent Failures:** Returning a raw stack trace or generic 500 error to the user when a model timeout occurs.
4. **Over-retrieval:** Returning more than 5 relevant document chunks to the user.

## 4. Explicitly Out of Scope
To maintain a tight engineering boundary, the following features will *not* be implemented in Phase 1:
* **Code/Infrastructure Generation:** The assistant will not generate Terraform, CloudFormation, or CDK scripts. It provides architectural guidance only.
* **Live Environment Access:** The system will not fetch real-time AWS billing data or inspect live AWS accounts. 
* **Multi-turn Conversational Memory:** The API will treat each request as stateless. Managing chat history is delegated to the client layer.
* **Non-AWS Clouds:** Support for GCP, Azure, or on-premise frameworks.
* **Queries outside the six pillar corpus:** Questions about topics not covered in the six pillars will be rejected.

## 5. User Journey & System Flow
1. **User Asks:** The architect submits a query via the client UI: *"How do I implement cost-aware auto-scaling?"*
2. **System Normalizes & Embeds:** The backend validates the input, checks the Level-2 semantic cache, and if a miss occurs, embeds the query using `text-embedding-3-small`.
3. **System Retrieves:** The vector store (Pinecone) performs a semantic similarity search, applying a >0.7 confidence threshold to fetch the top 5 relevant document chunks from the Cost Optimization PDF.
4. **System Generates (with routing):** The system constructs a grounded prompt. It attempts to route the request to the fine-tuned domain model.
5. **System Validates:** The generated answer passes through a lightweight hallucination check (comparing response entities to context entities).
6. **User Receives:** The UI renders the answer, a system confidence score (High/Medium/Low), total latency metrics, and expandable citations showing the exact excerpts from the AWS documentation.

## User Journey

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER JOURNEY                            │
└─────────────────────────────────────────────────────────────────┘

  [1] USER ASKS
      Engineer types a natural language question into the
      Streamlit interface or sends a POST /generate request.
      Example: "What does AWS recommend for cross-AZ resiliency?"
                              │
                              ▼
  [2] SYSTEM RETRIEVES
      Query is embedded using the same embedding model used
      at ingestion time (text-embedding-3-small).
      Embedded query is compared against the Pinecone vector
      index. Top-5 most semantically similar chunks are
      retrieved, along with their metadata (source pillar,
      page number, section heading).
      Chunks below a similarity threshold (0.60) are dropped.
      If no chunks pass the threshold → go to [REFUSAL PATH].
                              │
                              ▼
  [3] SYSTEM ANSWERS
      Retrieved chunks are injected into a structured prompt
      that instructs the LLM to:
        (a) answer only from the provided context
        (b) cite the source for every claim
        (c) say "I don't have enough context" if unsure
        (d) self-assess confidence as high / medium / low
      The LLM (fine-tuned Phi-3-mini via HF Hub adapter,
      or OpenAI GPT-4o-mini as fallback) generates the answer.
      A post-generation grounding check compares the answer
      embedding against the retrieved chunks. If similarity
      is below 0.50, the answer is flagged as low-confidence.
                              │
                              ▼
  [4] USER SEES SOURCES
      The response object is returned containing:
        • answer          — the grounded response text
        • sources         — list of pillar doc + section + excerpt
        • confidence      — high / medium / low
        • model_used      — which model generated the answer
        • latency_ms      — retrieval and generation times
      In the Streamlit UI, sources are displayed as expandable
      cards below the answer so the user can verify every claim.

  ── REFUSAL PATH ──────────────────────────────────────────────
  If no chunks pass the similarity threshold:
      System returns: "I could not find relevant information in
      the AWS Well-Architected Framework documents to answer
      this question. Please rephrase or consult the source
      documentation directly."
      No LLM call is made. No hallucination is possible.
```



## 6. Constraints and Assumptions

* **Compute Limitations:** Local GPU resources are unavailable for live API serving. The fine-tuned inference stack must therefore support CPU execution for development, while production assumes GPU-backed deployment. The backend loads the base model and PEFT adapter dynamically from Hugging Face Hub at startup.

* **Adapter Delivery and Updates:** The LoRA/QLoRA adapter is not stored in the source repository. It is hosted on Hugging Face Hub to enable versioned, decoupled deployments. The backend loads `base_model + adapter` at runtime rather than a merged model. Base models and adapters can be updated independently, requiring only a configuration change and service restart.

* **Graceful Degradation (OpenAI Fallback):** The system assumes that local inference may fail due to startup errors, high latency, unavailable GPU resources, or Hugging Face Hub connectivity issues. In such cases, generation transparently falls back to `GPT-4o-mini`. The fallback path is treated as a first-class inference mode rather than a degraded state, and every response logs `model_used` for observability.

* **Swappable Vector Store Interface:** Although Pinecone is used for development and production simulation, the vector layer is abstracted behind a `VectorStoreService` interface to support future migration to FAISS, Weaviate, or Qdrant without modifying the RAG pipeline. This assumption supports potential enterprise requirements around cost control, on-premise deployment, and data residency compliance.

* **Single-User, Single-Index Deployment:** Due to Pinecone free-tier limitations, all six pillar documents share a single index and are differentiated using metadata (`source_pillar`). Cross-pillar retrieval is supported, while pillar-specific filtering is implemented through metadata filters. Multi-tenant isolation would require namespaces or a paid deployment tier.

* **Manually Curated Evaluation Corpus:** The evaluation dataset (20–30 Q&A pairs) is manually authored against known document content. Automated dataset generation is out of scope. As a result, evaluation metrics and RAGAS scores should be interpreted as directional indicators of retrieval quality rather than statistically significant benchmarks.

---
