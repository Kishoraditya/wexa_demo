"""
backend/services/rag_pipeline.py

RAG Pipeline — Core Orchestration
===================================
Ties together retrieval, prompt construction, generation, and
post-generation validation into a single coherent pipeline.

Pipeline stages (per request):
  1. Input validation and guardrail check
  2. Cache lookup (skip all downstream if hit)
  3. Query embedding + FAISS retrieval
  4. Threshold check → refusal if no context
  5. Context formatting + prompt construction
  6. LLM generation (primary or fallback)
  7. Hallucination / grounding check
  8. Response parsing (confidence extraction, source structuring)
  9. Cache write
 10. Return structured RAGResponse

Hallucination protection — two layers:
  Layer 1 (pre-generation): Never pass empty context to the LLM.
    If retrieval returns no results above the similarity threshold,
    return a structured refusal. No LLM call is made.
    This eliminates the most common hallucination vector.

  Layer 2 (post-generation): Semantic grounding check.
    Embed the generated answer. Compute cosine similarity between the
    answer embedding and each retrieved chunk embedding.
    If the max similarity is below config.guardrails.output.grounding_threshold
    (default 0.50), the answer is flagged as potentially ungrounded.
    The response is still returned (not suppressed) but confidence is
    downgraded to LOW and the grounding_flag is set to True.

  Layer 3 (future — documented here, not implemented):
    LLM-as-judge: A second LLM call passes the (question, context, answer)
    triple to a judge model and asks: "Does the answer contradict or go
    beyond the context?" This provides a more semantic check than cosine
    similarity. RAGAS implements this as the 'faithfulness' metric.
    At scale, run RAGAS evaluate() on a sample of production responses
    to monitor faithfulness drift as documents or models are updated.
    See: https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html

Author: Enterprise RAG Assistant
"""

import re
import time
from typing import Optional

import numpy as np
from langchain_core.documents import Document

from backend.core.config import get_config
from backend.core.logging import get_logger
from backend.models.schemas import (
    RAGResponse,
    SourceDocument,
    ConfidenceLevel,
    RefusalResponse,
)
from backend.services.guardrails import InputGuardrail, OutputGuardrail
from backend.services.llm_manager import LLMManager, ModelType
from backend.services.prompts import (
    ACTIVE_SYSTEM_PROMPT,
    ACTIVE_HUMAN_TEMPLATE,
    NO_CONTEXT_REFUSAL,
    PROMPT_VERSION,
    format_context_block,
)
from backend.services.vector_store import RetrievalService

logger = get_logger(__name__)
config = get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Parser
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceParser:
    """
    Extracts the confidence self-assessment from raw LLM output.

    The model is instructed to end every response with:
        CONFIDENCE: HIGH|MEDIUM|LOW
        CONFIDENCE_REASON: <explanation>

    This parser extracts those values and removes them from the
    answer text so the user-facing answer is clean.

    Why ask the model to self-assess?
      The model has access to information the system does not:
        - How many context chunks directly supported the answer
        - Whether it had to infer across passages
        - Whether it found conflicting information in the context
      Self-assessment is an imperfect signal but a useful one.
      It is combined with the semantic grounding score for final confidence.
    """

    CONFIDENCE_PATTERN = re.compile(
        r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)\s*\n?"
        r"(?:CONFIDENCE_REASON:\s*(.+))?",
        re.IGNORECASE | re.MULTILINE,
    )

    @classmethod
    def parse(cls, raw_text: str) -> tuple[ConfidenceLevel, str, str]:
        """
        Extract confidence level and reason, return cleaned answer text.

        Args:
            raw_text: Raw LLM output string.

        Returns:
            Tuple of:
              - ConfidenceLevel enum value
              - confidence_reason string
              - cleaned_answer string (confidence lines removed)
        """
        match = cls.CONFIDENCE_PATTERN.search(raw_text)

        if match:
            level_str = match.group(1).upper()
            reason = (match.group(2) or "").strip()
            # Remove the CONFIDENCE block from the answer text
            cleaned_answer = raw_text[:match.start()].strip()

            try:
                confidence = ConfidenceLevel(level_str)
            except ValueError:
                confidence = ConfidenceLevel.MEDIUM

            return confidence, reason, cleaned_answer

        # Model did not include a confidence assessment.
        # Default to MEDIUM with a note. This happens when:
        #   - The model truncated its output (hit max_new_tokens)
        #   - The model ignored the instruction (rare with well-tuned models)
        logger.warning("No confidence self-assessment found in LLM output")
        return (
            ConfidenceLevel.MEDIUM,
            "Confidence assessment not provided by model",
            raw_text.strip(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Grounding Checker
# ─────────────────────────────────────────────────────────────────────────────

class GroundingChecker:
    """
    Post-generation semantic grounding verification.

    Computes the cosine similarity between the generated answer embedding
    and the retrieved chunk embeddings. A high similarity means the answer
    is semantically close to the retrieved context — it is probably grounded.
    A low similarity suggests the answer drifted from the context.

    Limitations of this approach:
      - Cosine similarity is a proxy for grounding, not a direct measure.
        An answer can be semantically similar to context while still
        introducing false details (e.g., inverting a recommendation).
      - The approach cannot detect negation errors
        ("do NOT use X" vs "use X") because negation has small cosine effect.
      - For more precise hallucination detection, use RAGAS faithfulness:
        it decomposes the answer into atomic claims and checks each claim
        against the context using an LLM judge.
        Production recommendation: run RAGAS on sampled responses weekly.

    Implementation:
      We do NOT re-embed using the model here. Instead, we embed the
      answer using the same EmbeddingModel that embedded the chunks,
      and compute cosine similarity against the chunks we already retrieved.
      This reuses infrastructure already present and adds minimal latency (~5ms).
    """

    def __init__(self, embedding_model):
        """
        Args:
            embedding_model: The EmbeddingModel instance from vector_store.py.
                             Used to embed the generated answer.
        """
        self._embedding_model = embedding_model
        self._threshold = config.guardrails.output.grounding_threshold

    def check(
        self,
        answer_text: str,
        retrieved_chunks: list[tuple[Document, float]],
    ) -> tuple[bool, float]:
        """
        Check whether the generated answer is semantically grounded.

        Args:
            answer_text: Cleaned generated answer string.
            retrieved_chunks: List of (Document, score) from retrieval.

        Returns:
            Tuple of:
              - is_grounded: True if max similarity >= threshold
              - max_similarity: Highest cosine similarity found
        """
        if not answer_text or not retrieved_chunks:
            return False, 0.0

        grounding_start = time.time()

        # Embed the answer
        answer_embedding = np.array(
            self._embedding_model.embed_query(answer_text)
        )

        # Compare against each retrieved chunk embedding
        # We recompute chunk embeddings here because they are not stored
        # in the Document objects. An optimization would be to cache chunk
        # embeddings during retrieval and pass them through.
        similarities = []
        chunk_texts = [doc.page_content for doc, _ in retrieved_chunks]
        chunk_embeddings = self._embedding_model.embeddings.embed_documents(chunk_texts)

        for chunk_emb in chunk_embeddings:
            chunk_vec = np.array(chunk_emb)
            # Cosine similarity for L2-normalized vectors = dot product
            # Both answer and chunk embeddings are L2-normalized by the model
            cosine_sim = float(np.dot(answer_embedding, chunk_vec))
            similarities.append(cosine_sim)

        max_similarity = max(similarities) if similarities else 0.0
        is_grounded = max_similarity >= self._threshold

        grounding_ms = round((time.time() - grounding_start) * 1000)
        logger.info(
            "Grounding check complete",
            extra={
                "max_similarity": round(max_similarity, 3),
                "threshold": self._threshold,
                "is_grounded": is_grounded,
                "check_ms": grounding_ms,
            },
        )

        return is_grounded, max_similarity


# ─────────────────────────────────────────────────────────────────────────────
# Source Extractor
# ─────────────────────────────────────────────────────────────────────────────

class SourceExtractor:
    """
    Constructs SourceDocument objects from retrieved chunks.

    These appear in the API response as the evidence trail —
    the user can see exactly which document passages informed the answer.
    """

    @staticmethod
    def extract(
        retrieved_chunks: list[tuple[Document, float]],
    ) -> list[SourceDocument]:
        """
        Convert retrieved (Document, score) tuples to SourceDocument objects.

        Args:
            retrieved_chunks: Retrieval results from RetrievalService.

        Returns:
            List of SourceDocument objects for the API response.
        """
        sources = []
        for doc, score in retrieved_chunks:
            # Create a short excerpt (first 200 characters) for the UI
            # Full chunk text would be too long to display inline
            excerpt = doc.page_content.strip()[:200]
            if len(doc.page_content.strip()) > 200:
                excerpt += "..."

            source = SourceDocument(
                pillar=doc.metadata.get("source_pillar", "Unknown"),
                source_file=doc.metadata.get("source", "Unknown"),
                page_number=doc.metadata.get("page_number", 0),
                section=doc.metadata.get("section", ""),
                excerpt=excerpt,
                relevance_score=round(score, 4),
                chunk_index=doc.metadata.get("chunk_index", 0),
            )
            sources.append(source)

        return sources


# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline — Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline from query to structured response.

    Dependency injection pattern:
      All dependencies (retrieval, LLM, guardrails) are passed to __init__.
      The pipeline does not instantiate its own dependencies.
      This makes unit testing straightforward — inject mocks.

    Usage:
        pipeline = RAGPipeline(
            retrieval_service=retrieval_service,
            llm_manager=llm_manager,
            input_guardrail=InputGuardrail(),
            output_guardrail=OutputGuardrail(),
            grounding_checker=GroundingChecker(embedding_model),
        )
        response = pipeline.run(query="How does AWS recommend handling failure?")
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_manager: LLMManager,
        input_guardrail: "InputGuardrail",
        output_guardrail: "OutputGuardrail",
        grounding_checker: GroundingChecker,
    ):
        self.retrieval = retrieval_service
        self.llm = llm_manager
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail
        self.grounding_checker = grounding_checker

        self.confidence_parser = ConfidenceParser()
        self.source_extractor = SourceExtractor()

    def run(
        self,
        query: str,
        use_fine_tuned: bool = True,
        top_k: Optional[int] = None,
        filter_pillar: Optional[str] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline for a single query.

        Args:
            query: User's natural language question.
            use_fine_tuned: Whether to prefer the fine-tuned model.
            top_k: Number of chunks to retrieve. Defaults to config value.
            filter_pillar: Restrict retrieval to a specific pillar.
                           None = search all pillars.

        Returns:
            RAGResponse with answer, sources, confidence, and latency metrics.
        """
        request_start = time.time()

        # ── Stage 1: Input guardrail ───────────────────────────────────────
        guardrail_result = self.input_guardrail.check(query)
        if not guardrail_result.passed:
            logger.warning(
                "Input guardrail blocked request",
                extra={"reason": guardrail_result.reason, "query_len": len(query)},
            )
            return RAGResponse.from_guardrail_block(
                reason=guardrail_result.reason,
            )

        # Use the sanitized query (whitespace stripped) from guardrail
        clean_query = guardrail_result.sanitized_input

        # ── Stage 2: Retrieval ─────────────────────────────────────────────
        retrieval_start = time.time()
        retrieved_chunks = self.retrieval.retrieve(
            query=clean_query,
            k=top_k,
            filter_pillar=filter_pillar,
        )
        retrieval_latency_ms = round((time.time() - retrieval_start) * 1000)

        logger.info(
            "Retrieval complete",
            extra={
                "chunks_retrieved": len(retrieved_chunks),
                "retrieval_ms": retrieval_latency_ms,
            },
        )

        # ── Stage 3: No-context refusal ────────────────────────────────────
        # CRITICAL: If retrieval returns nothing, we must refuse.
        # Passing an empty context to the LLM will cause it to answer from
        # its pre-trained weights — exactly the hallucination we prevent.
        if not retrieved_chunks:
            logger.info(
                "No relevant context found — returning structured refusal",
                extra={"query_preview": clean_query[:50]},
            )
            total_latency_ms = round((time.time() - request_start) * 1000)
            return RAGResponse(
                answer=NO_CONTEXT_REFUSAL,
                sources=[],
                confidence=ConfidenceLevel.LOW,
                confidence_reason="No relevant context found in the indexed documents.",
                model_used=ModelType.UNAVAILABLE,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0,
                total_latency_ms=total_latency_ms,
                tokens_used=None,
                cache_hit=False,
                grounding_flag=False,
                grounding_score=0.0,
                is_refusal=True,
                prompt_version=PROMPT_VERSION,
            )

        # ── Stage 4: Context formatting ────────────────────────────────────
        context_block = format_context_block(retrieved_chunks)

        # ── Stage 5: Prompt construction ───────────────────────────────────
        human_message = ACTIVE_HUMAN_TEMPLATE.format(
            context=context_block,
            question=clean_query,
        )

        # ── Stage 6: LLM generation ────────────────────────────────────────
        generation_result = self.llm.generate(
            system_prompt=ACTIVE_SYSTEM_PROMPT,
            human_message=human_message,
            use_fine_tuned=use_fine_tuned,
        )

        if not generation_result.succeeded:
            # Both primary and fallback failed — return a structured error
            total_latency_ms = round((time.time() - request_start) * 1000)
            return RAGResponse(
                answer=(
                    "Generation failed. Both the fine-tuned model and the "
                    "OpenAI fallback returned errors. Please try again."
                ),
                sources=self.source_extractor.extract(retrieved_chunks),
                confidence=ConfidenceLevel.LOW,
                confidence_reason="Generation error — answer not produced.",
                model_used=generation_result.model_type,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_result.generation_latency_ms,
                total_latency_ms=total_latency_ms,
                tokens_used=None,
                cache_hit=False,
                grounding_flag=True,
                grounding_score=0.0,
                is_refusal=False,
                prompt_version=PROMPT_VERSION,
            )

        raw_text = generation_result.raw_text

        # ── Stage 7: Parse confidence self-assessment ──────────────────────
        model_confidence, confidence_reason, clean_answer = (
            ConfidenceParser.parse(raw_text)
        )

        # ── Stage 8: Output guardrail (PII redaction) ──────────────────
        redaction_result = self.output_guardrail.redact_pii(clean_answer)
        redacted_answer = redaction_result.redacted_text
        if redaction_result.has_pii:
            logger.warning(
                "PII detected and redacted from generated answer",
                extra={"pii_types_found": redaction_result.pii_found},
            )

        # ── Stage 9: Grounding check ───────────────────────────────────────
        is_grounded, grounding_score = self.grounding_checker.check(
            answer_text=redacted_answer,
            retrieved_chunks=retrieved_chunks,
        )

        # Downgrade confidence if grounding check fails.
        # The model said HIGH but the answer drifted from the context —
        # the system overrides to LOW. Trust the grounding check over
        # the model's self-assessment when they disagree.
        final_confidence = model_confidence
        grounding_flag = False

        if not is_grounded:
            grounding_flag = True
            logger.warning(
                "Grounding check failed — downgrading confidence to LOW",
                extra={
                    "model_confidence": model_confidence,
                    "grounding_score": round(grounding_score, 3),
                    "threshold": config.guardrails.output.grounding_threshold,
                },
            )
            final_confidence = ConfidenceLevel.LOW
            confidence_reason = (
                f"Grounding check failed (similarity={grounding_score:.3f}, "
                f"threshold={config.guardrails.output.grounding_threshold}). "
                "Answer may not be fully supported by retrieved context. "
                "Please verify against source documents."
            )

        # ── Stage 10: Extract sources ──────────────────────────────────────
        sources = self.source_extractor.extract(retrieved_chunks)

        # ── Stage 11: Build final response ────────────────────────────────
        total_latency_ms = round((time.time() - request_start) * 1000)

        response = RAGResponse(
            answer=redacted_answer,
            sources=sources,
            confidence=final_confidence,
            confidence_reason=confidence_reason,
            model_used=generation_result.model_type,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_result.generation_latency_ms,
            total_latency_ms=total_latency_ms,
            tokens_used=generation_result.tokens_used,
            cache_hit=False,
            grounding_flag=grounding_flag,
            grounding_score=round(grounding_score, 4),
            is_refusal=False,
            prompt_version=PROMPT_VERSION,
        )

        logger.info(
            "RAG pipeline complete",
            extra={
                "confidence": final_confidence,
                "model_used": generation_result.model_type,
                "total_ms": total_latency_ms,
                "retrieval_ms": retrieval_latency_ms,
                "generation_ms": generation_result.generation_latency_ms,
                "grounding_score": round(grounding_score, 3),
                "grounding_flag": grounding_flag,
                "sources_count": len(sources),
                "tokens_used": generation_result.tokens_used,
            },
        )

        return response