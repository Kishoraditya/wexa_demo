"""
backend/services/prompts.py

Prompt Templates — Versioned and Centralized
=============================================
All prompt strings live here. No prompt text is hardcoded inline in pipeline
code. This makes prompt iteration possible without touching business logic.

Versioning strategy:
  Each prompt constant is named with a version suffix (V1, V2, ...).
  The active prompt is aliased to ACTIVE_SYSTEM_PROMPT.
  When iterating on prompts, add a new versioned constant and update the
  alias — old versions are preserved for regression comparison.

Why prompts deserve their own module:
  Prompt engineering is a first-class engineering activity, not a string
  formatting detail. The system prompt is the primary control surface for:
    - Grounding enforcement (no hallucination)
    - Citation behavior (source traceability)
    - Refusal behavior (no speculation on missing context)
    - Confidence calibration (self-assessment)
    - Tone and format (technical, concise, structured)
  Keeping it here makes it reviewable, testable, and version-controlled
  as a first-class artifact — not buried in a service file.

Author: Enterprise RAG Assistant
"""

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt — Version 1
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_V1 = """You are an expert AWS cloud architecture assistant with \
deep knowledge of the AWS Well-Architected Framework.

Your knowledge base consists exclusively of the six AWS Well-Architected \
Framework pillars:
- Operational Excellence
- Security
- Reliability
- Performance Efficiency
- Cost Optimization
- Sustainability

════════════════════════════════════════════
STRICT OPERATING RULES — YOU MUST FOLLOW ALL
════════════════════════════════════════════

RULE 1 — ANSWER ONLY FROM PROVIDED CONTEXT:
You must derive every factual claim exclusively from the CONTEXT BLOCK below.
Do not use your pre-trained knowledge to supplement, extend, or fill gaps in
the provided context. If something is not in the context, it does not exist
for the purposes of this answer.

RULE 2 — CITE EVERY CLAIM:
For every factual statement, cite the source document and section in this
exact format: [Source: <pillar_name>, Page <number>]
Example: "AWS recommends designing for failure [Source: Reliability, Page 14]"
If you cannot identify a source for a claim, do not make the claim.

RULE 3 — REFUSE WHEN CONTEXT IS INSUFFICIENT:
If the provided context does not contain enough information to answer the
question, you MUST respond with exactly:
"INSUFFICIENT_CONTEXT: The AWS Well-Architected Framework documents provided
do not contain enough information to answer this question. Please consult
the relevant pillar documentation directly or rephrase your question."
Do not speculate. Do not use phrases like "generally speaking" or "typically".
Do not answer from memory. Refusal is the correct behavior when context
is absent.

RULE 4 — CONFIDENCE SELF-ASSESSMENT:
At the end of every response (except refusals), you MUST append a confidence
assessment in this exact format on its own line:

CONFIDENCE: <HIGH|MEDIUM|LOW>
CONFIDENCE_REASON: <one sentence explaining the rating>

Confidence rating criteria:
  HIGH   — Context directly and completely answers the question.
            Multiple relevant passages from the context support the answer.
  MEDIUM — Context partially answers the question or requires inference
            between passages to form a complete answer.
  LOW    — Context is tangentially related. The answer required significant
            inference. The user should verify against primary sources.

════════════════════════════════════════════
RESPONSE FORMAT
════════════════════════════════════════════

Structure your response as follows:
1. Direct answer to the question (1-3 paragraphs)
2. Key points as bullet points where appropriate
3. Citations inline with claims (not at the end)
4. CONFIDENCE line (mandatory, last line)

Tone: Technical, precise, concise. Assume the reader is a senior engineer.
Do not pad responses with generic introductions ("Great question!") or
conclusions ("I hope this helps!"). Answer directly.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Human Message Template
# ─────────────────────────────────────────────────────────────────────────────

# The delimiter pattern (════) serves two purposes:
#   1. Visual separation for human reviewers reading prompts in logs
#   2. A signal to the model that the context block has clear boundaries
#      Models trained on structured text respond better to explicit delimiters
#      than to context injected without markers.
#
# {context} and {question} are LangChain template variables.
# They are filled at runtime by the RAG pipeline.

HUMAN_PROMPT_TEMPLATE_V1 = """
════════════════════════════════════════════
CONTEXT BLOCK — RETRIEVED FROM AWS DOCUMENTATION
════════════════════════════════════════════

{context}

════════════════════════════════════════════
QUESTION
════════════════════════════════════════════

{question}

════════════════════════════════════════════
INSTRUCTIONS REMINDER
════════════════════════════════════════════

Answer using ONLY the content in the CONTEXT BLOCK above.
Cite every claim with [Source: <pillar>, Page <number>].
End your response with CONFIDENCE: HIGH|MEDIUM|LOW on its own line.
If the context is insufficient, respond with INSUFFICIENT_CONTEXT: ...
"""

# ─────────────────────────────────────────────────────────────────────────────
# Active Prompt Alias
# ─────────────────────────────────────────────────────────────────────────────

# Change this alias to switch prompt versions system-wide.
# All pipeline code imports ACTIVE_* constants, never the versioned ones.
ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT_V1
ACTIVE_HUMAN_TEMPLATE = HUMAN_PROMPT_TEMPLATE_V1
PROMPT_VERSION = "v1"


# ─────────────────────────────────────────────────────────────────────────────
# Refusal Response Template
# ─────────────────────────────────────────────────────────────────────────────

# This is the structured response returned when the retriever finds NO chunks
# above the similarity threshold. In this case, we never call the LLM at all —
# we return this template directly from the pipeline.
#
# Why a hardcoded refusal rather than letting the LLM refuse?
#   If we pass empty context to the LLM, it may still generate an answer
#   using its pre-trained weights — exactly the hallucination we are
#   trying to prevent. The only safe approach is to intercept before
#   the LLM call and return a deterministic refusal.

NO_CONTEXT_REFUSAL = (
    "I could not find relevant information in the AWS Well-Architected "
    "Framework documents to answer this question. The question may be "
    "outside the scope of the six pillars, or the relevant section may "
    "not have been indexed. Please rephrase your question or consult "
    "the AWS Well-Architected Framework documentation directly at "
    "https://aws.amazon.com/architecture/well-architected/"
)

# ─────────────────────────────────────────────────────────────────────────────
# Context Formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_context_block(
    retrieved_chunks: list[tuple],  # list of (Document, score)
) -> str:
    """
    Format retrieved chunks into a structured context block for injection.

    Each chunk is formatted with its source metadata clearly labeled.
    This serves two purposes:
      1. The model uses the metadata to construct citations
      2. The context block is human-readable in logs for debugging

    Format per chunk:
        [CHUNK 1 | Source: Reliability | Page: 14 | Score: 0.847]
        <chunk text>
        ---

    Args:
        retrieved_chunks: List of (Document, cosine_similarity_score) tuples
                          from the RetrievalService.

    Returns:
        Formatted multi-chunk context string ready for prompt injection.
    """
    if not retrieved_chunks:
        return "No relevant context found."

    formatted_parts = []
    for i, (doc, score) in enumerate(retrieved_chunks, start=1):
        pillar = doc.metadata.get("source_pillar", "Unknown Pillar")
        page = doc.metadata.get("page_number", "N/A")
        section = doc.metadata.get("section", "")
        section_str = f" | Section: {section}" if section else ""

        header = (
            f"[CHUNK {i} | Source: {pillar} | "
            f"Page: {page}{section_str} | "
            f"Relevance Score: {score:.3f}]"
        )
        formatted_parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(formatted_parts)


# ─────────────────────────────────────────────────────────────────────────────
# LangChain ChatPromptTemplate Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_prompt() -> ChatPromptTemplate:
    """
    Build and return the LangChain ChatPromptTemplate for RAG generation.

    The template has two message types:
      SystemMessage: Contains the operating rules and persona.
                     This is sent first and sets the model's behavior.
      HumanMessage:  Contains the context block and the user question.
                     This is the per-request content.

    Why separate system and human messages?
      Modern chat models (GPT-4, Phi-3, Llama-3) are instruction-tuned to
      follow system messages as persistent behavioral rules and human messages
      as per-turn inputs. Putting the rules in the system message increases
      the probability that the model treats them as inviolable constraints
      rather than suggestions embedded in the user turn.

    Returns:
        ChatPromptTemplate with {context} and {question} as input variables.
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(ACTIVE_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(ACTIVE_HUMAN_TEMPLATE),
    ])
    