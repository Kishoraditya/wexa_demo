"""  
frontend/app.py  
  
Enterprise Cloud Architecture Assistant — Streamlit UI  
=======================================================  
Connects to the FastAPI backend at BACKEND_URL (default: http://localhost:8000).  
  
Tabs:  
  1. Ask — main RAG query interface  
  2. Ingest — trigger document ingestion / upload files  
  3. System Status — health, model availability, vector count  
  4. About — architecture overview and project info  
  
Run:  
    streamlit run frontend/app.py  
"""  
  
import time  
from typing import Optional  
  
import requests  
import streamlit as st  
  
# ─────────────────────────────────────────────────────────────────────────────  
# Configuration  
# ─────────────────────────────────────────────────────────────────────────────  
  
BACKEND_URL = "http://localhost:8000"  
APP_TITLE = "Enterprise Cloud Architecture Assistant"  
APP_SUBTITLE = "Grounded answers from the AWS Well-Architected Framework"  
  
PILLARS = [  
    "All Pillars",  
    "Operational Excellence",  
    "Security",  
    "Reliability",  
    "Performance Efficiency",  
    "Cost Optimization",  
    "Sustainability",  
]  
  
PILLAR_ICONS = {  
    "Operational Excellence": "⚙️",  
    "Security": "🔒",  
    "Reliability": "🔄",  
    "Performance Efficiency": "⚡",  
    "Cost Optimization": "💰",  
    "Sustainability": "🌱",  
    "All Pillars": "🌐",  
}  
  
SAMPLE_QUESTIONS = [  
    ("Reliability", "How should I design systems to automatically recover from component failures?"),  
    ("Security", "What is the principle of least privilege and how does AWS recommend applying it?"),  
    ("Cost Optimization", "What are the key strategies for right-sizing EC2 instances?"),  
    ("Performance Efficiency", "How does AWS recommend selecting the right compute type for a workload?"),  
    ("Operational Excellence", "What does AWS recommend for implementing runbooks and playbooks?"),  
    ("Sustainability", "How can I reduce the carbon footprint of my AWS workloads?"),  
    ("Reliability", "What is the difference between RTO and RPO in disaster recovery planning?"),  
    ("Security", "How should I implement data classification and protection in AWS?"),  
]  
  
CONFIDENCE_COLORS = {  
    "HIGH": "green",  
    "MEDIUM": "orange",  
    "LOW": "red",  
}  
  
CONFIDENCE_LABELS = {  
    "HIGH": "HIGH — Context directly answers the question",  
    "MEDIUM": "MEDIUM — Context partially answers; some inference required",  
    "LOW": "LOW — Context is insufficient or answer may be unreliable",  
}  
  
# ─────────────────────────────────────────────────────────────────────────────  
# Page Config  
# ─────────────────────────────────────────────────────────────────────────────  
  
st.set_page_config(  
    page_title=APP_TITLE,  
    page_icon="☁️",  
    layout="wide",  
    initial_sidebar_state="expanded",  
)  
  
# ─────────────────────────────────────────────────────────────────────────────  
# Session State Initialization  
# ─────────────────────────────────────────────────────────────────────────────  
  
if "query_history" not in st.session_state:  
    st.session_state.query_history = []  # list of (query, response) dicts  
if "last_response" not in st.session_state:  
    st.session_state.last_response = None  
if "health_data" not in st.session_state:  
    st.session_state.health_data = None  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# API Helpers  
# ─────────────────────────────────────────────────────────────────────────────  
  
def get_health() -> Optional[dict]:  
    """Fetch /health from the backend. Returns None on connection error."""  
    try:  
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)  
        r.raise_for_status()  
        return r.json()  
    except requests.exceptions.ConnectionError:  
        return None  
    except Exception as e:  
        return {"error": str(e)}  
  
  
def get_readiness() -> Optional[dict]:  
    """Fetch /health/ready. Returns dict with status field."""  
    try:  
        r = requests.get(f"{BACKEND_URL}/health/ready", timeout=5)  
        return r.json()  
    except requests.exceptions.ConnectionError:  
        return None  
    except Exception as e:  
        return {"error": str(e)}  
  
  
def call_generate(  
    query: str,  
    use_fine_tuned: bool,  
    top_k: int,  
    filter_pillar: Optional[str],  
) -> dict:  
    """  
    POST /generate. Returns the full RAGResponse dict or an error dict.  
    Error dict always has an 'error' key with a human-readable message.  
    """  
    payload = {  
        "query": query,  
        "use_fine_tuned": use_fine_tuned,  
        "top_k": top_k,  
        "filter_pillar": filter_pillar,  
    }  
    try:  
        r = requests.post(  
            f"{BACKEND_URL}/generate",  
            json=payload,  
            timeout=150,  # fine-tuned model can be slow on CPU  
        )  
        if r.status_code == 200:  
            return r.json()  
        elif r.status_code == 422:  
            detail = r.json().get("detail", r.text)  
            return {"error": f"Validation error: {detail}"}  
        elif r.status_code == 503:  
            return {"error": "Backend not ready. Run ingestion first (Ingest tab)."}  
        elif r.status_code == 504:  
            return {"error": "Generation timed out. Try using OpenAI fallback mode."}  
        else:  
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}  
    except requests.exceptions.ConnectionError:  
        return {"error": f"Cannot connect to backend at {BACKEND_URL}. Is the server running?"}  
    except requests.exceptions.Timeout:  
        return {"error": "Request timed out after 150s. The model may be loading — try again."}  
    except Exception as e:  
        return {"error": str(e)}  
  
  
def call_ingest(force_reindex: bool = False) -> dict:  
    """POST /ingest. Returns IngestResponse dict or error dict."""  
    try:  
        r = requests.post(  
            f"{BACKEND_URL}/ingest",  
            json={"force_reindex": force_reindex},  
            timeout=300,  # ingestion can take 2-5 minutes  
        )  
        if r.status_code == 200:  
            return r.json()  
        elif r.status_code == 409:  
            return {"error": "Ingestion already in progress. Please wait."}  
        else:  
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}  
    except requests.exceptions.ConnectionError:  
        return {"error": f"Cannot connect to backend at {BACKEND_URL}."}  
    except requests.exceptions.Timeout:  
        return {"error": "Ingestion timed out after 5 minutes."}  
    except Exception as e:  
        return {"error": str(e)}  
  
  
def call_upload(file_bytes: bytes, filename: str) -> dict:  
    """POST /ingest/upload. Returns upload response dict or error dict."""  
    try:  
        r = requests.post(  
            f"{BACKEND_URL}/ingest/upload",  
            files={"file": (filename, file_bytes, "application/pdf")},  
            timeout=150,  
        )  
        if r.status_code == 200:  
            return r.json()  
        elif r.status_code == 422:  
            return {"error": r.json().get("message", "Unsupported file type.")}  
        else:  
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}  
    except requests.exceptions.ConnectionError:  
        return {"error": f"Cannot connect to backend at {BACKEND_URL}."}  
    except Exception as e:  
        return {"error": str(e)}  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# UI Components  
# ─────────────────────────────────────────────────────────────────────────────  
  
def render_confidence_badge(confidence: str) -> None:  
    """Render a colored confidence badge."""  
    color = CONFIDENCE_COLORS.get(confidence, "gray")  
    label = CONFIDENCE_LABELS.get(confidence, confidence)  
    st.markdown(  
        f'<span style="background-color:{color};color:white;'  
        f'padding:3px 10px;border-radius:12px;font-weight:bold;'  
        f'font-size:0.85em;">{confidence}</span> '  
        f'<span style="color:{color};font-size:0.85em;">{label.split("—",1)[1].strip()}</span>',  
        unsafe_allow_html=True,  
    )  
  
  
def render_model_badge(model_used: str) -> None:  
    """Render a badge showing which model generated the answer."""  
    if model_used == "fine_tuned_phi3_qlora":  
        st.markdown(  
            '<span style="background-color:#1f77b4;color:white;'  
            'padding:2px 8px;border-radius:8px;font-size:0.8em;">'  
            '🤖 Phi-3-mini (fine-tuned QLoRA)</span>',  
            unsafe_allow_html=True,  
        )  
    elif model_used == "openai_gpt4o_mini":  
        st.markdown(  
            '<span style="background-color:#2ca02c;color:white;'  
            'padding:2px 8px;border-radius:8px;font-size:0.8em;">'  
            '✨ GPT-4o-mini (OpenAI fallback)</span>',  
            unsafe_allow_html=True,  
        )  
    else:  
        st.markdown(  
            '<span style="background-color:#7f7f7f;color:white;'  
            'padding:2px 8px;border-radius:8px;font-size:0.8em;">'  
            '⚠️ Model unavailable</span>',  
            unsafe_allow_html=True,  
        )  
  
  
def render_source_card(source: dict, index: int) -> None:  
    """Render a single source document as an expandable card."""  
    pillar = source.get("pillar", "Unknown")  
    icon = PILLAR_ICONS.get(pillar, "📄")  
    score = source.get("relevance_score", 0.0)  
    page = source.get("page_number", "?")  
    section = source.get("section", "")  
    excerpt = source.get("excerpt", "")  
    source_file = source.get("source_file", "")  
  
    label = f"{icon} {pillar} — {source_file} (p.{page})"  
    if section:  
        label += f" · {section}"  
    label += f"  |  score: {score:.3f}"  
  
    with st.expander(label, expanded=(index == 0)):  
        col1, col2, col3 = st.columns(3)  
        col1.metric("Pillar", pillar)  
        col2.metric("Page", page)  
        col3.metric("Relevance", f"{score:.3f}")  
        if section:  
            st.caption(f"Section: {section}")  
        st.markdown("**Excerpt:**")  
        st.info(excerpt if excerpt else "_No excerpt available_")  
  
  
def render_metrics_row(response: dict) -> None:  
    """Render latency and token metrics in a compact row."""  
    col1, col2, col3, col4, col5 = st.columns(5)  
    col1.metric("Retrieval", f"{response.get('retrieval_latency_ms', 0)} ms")  
    col2.metric("Generation", f"{response.get('generation_latency_ms', 0)} ms")  
    col3.metric("Total", f"{response.get('total_latency_ms', 0)} ms")  
    tokens = response.get("tokens_used")  
    col4.metric("Tokens", str(tokens) if tokens else "N/A")  
    col5.metric(  
        "Cache",  
        "HIT" if response.get("cache_hit") else "MISS",  
        delta=None,  
    )  
  
  
def render_response(response: dict) -> None:  
    """Render a full RAGResponse dict."""  
    # ── Refusal ──────────────────────────────────────────────────────────────  
    if response.get("is_refusal"):  
        st.warning(  
            "**No relevant context found.**\n\n"  
            + response.get("answer", "The system could not answer this question."),  
            icon="⚠️",  
        )  
        return  
  
    # ── Grounding warning ────────────────────────────────────────────────────  
    if response.get("grounding_flag"):  
        st.warning(  
            f"**Grounding check flagged this response** "  
            f"(score: {response.get('grounding_score', 0):.3f} < 0.50). "  
            "The answer may not be fully grounded in the retrieved context. "  
            "Verify against the source documents below.",  
            icon="🔍",  
        )  
  
    # ── Answer ───────────────────────────────────────────────────────────────  
    st.markdown("### Answer")  
    st.markdown(response.get("answer", ""))  
  
    # ── Confidence + model badges ─────────────────────────────────────────────  
    st.markdown("---")  
    badge_col, model_col = st.columns([1, 1])  
    with badge_col:  
        render_confidence_badge(response.get("confidence", "LOW"))  
        st.caption(response.get("confidence_reason", ""))  
    with model_col:  
        render_model_badge(response.get("model_used", "unavailable"))  
        st.caption(  
            f"Prompt version: `{response.get('prompt_version', 'unknown')}`  |  "  
            f"Grounding score: `{response.get('grounding_score', 0):.3f}`"  
        )  
  
    # ── Latency metrics ───────────────────────────────────────────────────────  
    st.markdown("---")  
    st.markdown("**Performance**")  
    render_metrics_row(response)  
  
    # ── Sources ───────────────────────────────────────────────────────────────  
    sources = response.get("sources", [])  
    if sources:  
        st.markdown("---")  
        st.markdown(f"**Source Documents** ({len(sources)} retrieved)")  
        for i, src in enumerate(sources):  
            render_source_card(src, i)  
    else:  
        st.caption("No source documents returned.")  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# Sidebar  
# ─────────────────────────────────────────────────────────────────────────────  
  
def render_sidebar() -> None:  
    with st.sidebar:  
        st.title("☁️ " + APP_TITLE)  
        st.caption(APP_SUBTITLE)  
        st.divider()  
  
        # ── Live health status ────────────────────────────────────────────────  
        st.subheader("System Status")  
        health = get_health()  
        st.session_state.health_data = health  
  
        if health is None:  
            st.error("Backend offline")  
            st.caption(f"Expected at: `{BACKEND_URL}`")  
        elif "error" in health:  
            st.error(f"Error: {health['error']}")  
        else:  
            status_color = "green" if health.get("status") == "healthy" else "red"  
            st.markdown(  
                f'<span style="color:{status_color};font-weight:bold;">'  
                f'● {health.get("status", "unknown").upper()}</span>',  
                unsafe_allow_html=True,  
            )  
            vs_ready = health.get("vector_store_ready", False)  
            primary = health.get("primary_model_available", False)  
            fallback = health.get("fallback_model_available", False)  
            vector_count = health.get("vector_count", 0)  
  
            st.markdown(  
                f"{'✅' if vs_ready else '❌'} Vector store "  
                f"({'**' + str(vector_count) + ' vectors**' if vs_ready else 'empty — run Ingest'})"  
            )  
            st.markdown(f"{'✅' if primary else '⬜'} Phi-3-mini (fine-tuned)")  
            st.markdown(f"{'✅' if fallback else '❌'} GPT-4o-mini (fallback)")  
            st.caption(f"Version: `{health.get('version', 'unknown')}`")  
  
            if not vs_ready:  
                st.warning("Index is empty. Go to the **Ingest** tab to build it.")  
  
        st.divider()  
  
        # ── Query settings ────────────────────────────────────────────────────  
        st.subheader("Query Settings")  
        use_fine_tuned = st.toggle(  
            "Use fine-tuned Phi-3-mini",  
            value=False,  
            help=(  
                "When ON: uses the QLoRA-adapted Phi-3-mini (requires adapter_repo in config.yaml). "  
                "When OFF: uses OpenAI GPT-4o-mini fallback (requires OPENAI_API_KEY)."  
            ),  
        )  
        top_k = st.slider(  
            "Retrieved chunks (top_k)",  
            min_value=1,  
            max_value=10,  
            value=5,  
            help="Number of document chunks to retrieve. Higher = more context, slower generation.",  
        )  
        pillar_choice = st.selectbox(  
            "Filter by pillar",  
            options=PILLARS,  
            index=0,  
            help="Restrict retrieval to a specific AWS Well-Architected pillar.",  
        )  
        filter_pillar = None if pillar_choice == "All Pillars" else pillar_choice  
  
        st.divider()  
        st.caption("Backend: `" + BACKEND_URL + "`")  
        st.caption("Docs: [FastAPI /docs](" + BACKEND_URL + "/docs)")  
  
    return use_fine_tuned, top_k, filter_pillar  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# Tab 1: Ask  
# ─────────────────────────────────────────────────────────────────────────────  
  
def render_ask_tab(
    use_fine_tuned: bool,
    top_k: int,
    filter_pillar: Optional[str],
) -> None:
    """Render the chat-based AWS Well-Architected assistant UI."""

    st.header("Ask the AWS Well-Architected Assistant")

    st.caption(
        "Ask any question about the six AWS Well-Architected pillars. "
        "Answers are grounded in the official AWS documentation."
    )

    # ── Session state initialization ────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Sample questions ────────────────────────────────────────────────────
    render_sample_questions()

    # ── Active settings summary ─────────────────────────────────────────────
    render_settings_summary(
        use_fine_tuned=use_fine_tuned,
        top_k=top_k,
        filter_pillar=filter_pillar,
    )

    # ── Existing chat history ───────────────────────────────────────────────
    render_chat_history()

    # ── Pending sample question ─────────────────────────────────────────────
    pending_message = st.session_state.pop("pending_message", None)

    if pending_message:
        process_message(
            message=pending_message,
            use_fine_tuned=use_fine_tuned,
            top_k=top_k,
            filter_pillar=filter_pillar,
        )

    # ── Chat input ──────────────────────────────────────────────────────────
    if message := st.chat_input(
        "Ask about the AWS Well-Architected Framework...",
        max_chars=1000,
    ):
        process_message(
            message=message,
            use_fine_tuned=use_fine_tuned,
            top_k=top_k,
            filter_pillar=filter_pillar,
        )


def render_sample_questions() -> None:
    """Render expandable sample question buttons."""

    with st.expander("Sample questions — click to use", expanded=False):

        cols = st.columns(2)

        for i, (pillar, question) in enumerate(SAMPLE_QUESTIONS):

            icon = PILLAR_ICONS.get(pillar, "📄")

            col = cols[i % 2]

            if col.button(
                f"{icon} {question[:80]}{'...' if len(question) > 80 else ''}",
                key=f"sample_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_message = question
                st.rerun()


def render_settings_summary(
    use_fine_tuned: bool,
    top_k: int,
    filter_pillar: Optional[str],
) -> None:
    """Render current model and retrieval settings."""

    model_label = (
        "Phi-3-mini (fine-tuned)"
        if use_fine_tuned
        else "Fallback (OpenRouter/OpenAI)"
    )

    pillar_label = filter_pillar or "All Pillars"

    st.caption(
        f"Model: **{model_label}** · "
        f"Top-k: **{top_k}** · "
        f"Pillar: **{pillar_label}** "
        f"_(change in sidebar)_"
    )


def render_chat_history() -> None:
    """Render all previous chat messages."""

    for item in st.session_state.chat_history:

        with st.chat_message("user"):
            st.markdown(item["message"])

        with st.chat_message("assistant"):

            response = item["response"]

            if "error" in response:
                st.error(response["error"])
            else:
                render_response(response)


def validate_backend_health() -> bool:
    """Validate backend availability before processing requests."""

    health = st.session_state.get("health_data")

    if health is None:
        st.error(
            f"Backend is not reachable at `{BACKEND_URL}`. "
            "Start the server first."
        )
        return False

    return True


def generate_response(
    message: str,
    use_fine_tuned: bool,
    top_k: int,
    filter_pillar: Optional[str],
) -> dict:
    """Execute a single RAG generation request."""

    return call_generate(
        query=message,
        use_fine_tuned=use_fine_tuned,
        top_k=top_k,
        filter_pillar=filter_pillar,
    )


def process_message(
    message: str,
    use_fine_tuned: bool,
    top_k: int,
    filter_pillar: Optional[str],
) -> None:
    """Process one complete chat turn."""

    message = message.strip()

    if not message:
        return

    # ── Backend validation ──────────────────────────────────────────────────
    if not validate_backend_health():
        return

    # ── Render user message ─────────────────────────────────────────────────
    with st.chat_message("user"):
        st.markdown(message)

    # ── Generate and render assistant response ──────────────────────────────
    with st.chat_message("assistant"):

        with st.spinner("Retrieving context and generating answer..."):

            response = generate_response(
                message=message,
                use_fine_tuned=use_fine_tuned,
                top_k=top_k,
                filter_pillar=filter_pillar,
            )

        if "error" in response:
            st.error(response["error"])
        else:
            render_response(response)

    # ── Persist chat history ────────────────────────────────────────────────
    st.session_state.chat_history.append(
        {
            "message": message,
            "response": response,
        }
    )
  
# ─────────────────────────────────────────────────────────────────────────────  
# Tab 2: Ingest  
# ─────────────────────────────────────────────────────────────────────────────  
  
def render_ingest_tab() -> None:  
    st.header("Document Ingestion")  
    st.markdown(  
        "Build or refresh the FAISS vector index from the AWS Well-Architected PDFs. "  
        "You must run ingestion at least once before the assistant can answer questions."  
    )  
  
    # ── Current index status ──────────────────────────────────────────────────  
    health = st.session_state.health_data  
    if health and "vector_count" in health:  
        vc = health["vector_count"]  
        if vc > 0:  
            st.success(f"Index contains **{vc} vectors**. The assistant is ready.")  
        else:  
            st.warning("Index is empty. Run ingestion below.")  
  
    st.divider()  
  
    # ── Trigger ingestion ─────────────────────────────────────────────────────  
    st.subheader("Ingest from data directory")  
    st.markdown(  
        "Scans `data/pdfs/` for the 6 AWS Well-Architected PDFs, chunks them, "  
        "embeds each chunk, and builds the FAISS index. "  
        "Deduplication skips chunks already in the index — safe to run multiple times."  
    )  
  
    col1, col2 = st.columns([2, 1])  
    force_reindex = col2.checkbox(  
        "Force full re-index",  
        value=False,  
        help="Re-embed all chunks even if already indexed. Use when the embedding model changes.",  
    )  
  
    if col1.button("Run Ingestion", type="primary"):  
        with st.spinner(  
            "Ingesting documents... This may take 2-5 minutes for 6 PDFs. Please wait."  
        ):  
            result = call_ingest(force_reindex=force_reindex)  
  
        if "error" in result:  
            st.error(result["error"])  
        else:  
            st.success(result.get("message", "Ingestion complete."))  
            m1, m2, m3, m4 = st.columns(4)  
            m1.metric("Chunks embedded", result.get("chunks_embedded", 0))  
            m2.metric("Chunks indexed", result.get("chunks_indexed", 0))  
            m3.metric("Total vectors", result.get("total_vectors_in_index", 0))  
            m4.metric("Duration", f"{result.get('duration_seconds', 0):.1f}s")  
            # Refresh health state  
            st.session_state.health_data = get_health()  
  
    st.divider()  
  
    # ── File upload ───────────────────────────────────────────────────────────  
    st.subheader("Upload a document")  
    st.markdown(  
        "Upload a PDF, Markdown, or plain text file. "  
        "The file is saved to `data/pdfs/`. "  
        "After uploading, run **Ingest** above to index it."  
    )  
  
    uploaded = st.file_uploader(  
        "Choose a file",  
        type=["pdf", "md", "txt"],  
        help="Max 50MB. Supported: PDF, Markdown, plain text.",  
    )  
  
    if uploaded is not None:  
        st.caption(  
            f"Selected: `{uploaded.name}` ({uploaded.size / 1024:.1f} KB)"  
        )  
        if st.button("Upload file"):  
            with st.spinner(f"Uploading {uploaded.name}..."):  
                result = call_upload(uploaded.read(), uploaded.name)  
            if "error" in result:  
                st.error(result["error"])  
            else:  
                st.success(  
                    f"Uploaded `{result.get('filename')}` "  
                    f"({result.get('size_bytes', 0) / 1024:.1f} KB). "  
                    "Now run **Ingest** above to index it."  
                )  
  
    st.divider()  
    st.subheader("Expected PDF files")  
    st.markdown(  
        "Download from [aws.amazon.com/architecture/well-architected](https://aws.amazon.com/architecture/well-architected/) "  
        "and place in `data/pdfs/`:"  
    )  
    for name in [  
        "operational_excellence.pdf",  
        "security.pdf",  
        "reliability.pdf",  
        "performance_efficiency.pdf",  
        "cost_optimization.pdf",  
        "sustainability.pdf",  
    ]:  
        st.markdown(f"- `data/pdfs/{name}`")  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# Tab 3: System Status  
# ─────────────────────────────────────────────────────────────────────────────  
  
def render_status_tab() -> None:  
    st.header("System Status")  
  
    col_refresh, _ = st.columns([1, 4])  
    if col_refresh.button("Refresh"):  
        st.session_state.health_data = get_health()  
  
    health = get_health()  
    readiness = get_readiness()  
    st.session_state.health_data = health  
  
    st.subheader("Liveness — GET /health")  
    if health is None:  
        st.error(f"Cannot reach backend at `{BACKEND_URL}`")  
    elif "error" in health:  
        st.error(f"Error: {health['error']}")  
    else:  
        c1, c2, c3, c4 = st.columns(4)  
        c1.metric("Status", health.get("status", "unknown").upper())  
        c2.metric("Vector count", health.get("vector_count", 0))  
        c3.metric("Primary model", "Available" if health.get("primary_model_available") else "Unavailable")  
        c4.metric("Fallback model", "Available" if health.get("fallback_model_available") else "Unavailable")  
        st.caption(f"App version: `{health.get('version', 'unknown')}`")  
  
    st.divider()  
    st.subheader("Readiness — GET /health/ready")  
    if readiness is None:  
        st.error("Cannot reach backend.")  
    elif "error" in readiness:  
        st.error(f"Error: {readiness['error']}")  
    elif readiness.get("status") == "ready":  
        st.success("Ready — all dependencies healthy.")  
        st.json(readiness)  
    else:  
        st.warning("Not ready.")  
        issues = readiness.get("issues", [])  
        for issue in issues:  
            st.markdown(f"- {issue}")  
        st.json(readiness)  
  
    st.divider()  
    st.subheader("Prometheus Metrics")  
    st.markdown(  
        f"Raw metrics available at: "  
        f"[{BACKEND_URL}/metrics]({BACKEND_URL}/metrics)"  
    )  
    st.code(  
        "# Key metrics\n"  
        "rag_requests_total{status, model, endpoint}\n"  
        "rag_retrieval_latency_seconds\n"  
        "rag_generation_latency_seconds\n"  
        "rag_cache_hits_total\n"  
        "rag_fallback_activations_total\n"  
        "rag_hallucination_flags_total\n"  
        "rag_tokens_total{model, direction}",  
        language="text",  
    )  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# Tab 4: About  
# ─────────────────────────────────────────────────────────────────────────────  
  
def render_about_tab() -> None:  
    st.header("About this System")  
  
    st.markdown(  
        """  
A production-grade **Retrieval-Augmented Generation (RAG)** system for querying  
the AWS Well-Architected Framework. Built with FastAPI, FAISS, LangChain, and  
a QLoRA fine-tuned Phi-3-mini generation layer.  
"""  
    )  
  
    st.subheader("Architecture")  
    st.code(  
        """  
QUERY FLOW  
──────────  
User Query  
    │  
    ▼  
Input Guardrails ──── injection detection, length check  
    │  
    ▼  
L2 Cache Check ──── diskcache, 1h TTL, keyed by hash(query+top_k+pillar)  
    │ (miss)  
    ▼  
BGE-small Embedder ──── BAAI/bge-small-en-v1.5, 384-dim  
    │  
    ▼  
FAISS Index ──── IndexFlatL2, ~843 vectors, <30ms  
    │  
    ▼  
BGE Reranker ──── BAAI/bge-reranker-base, cross-encoder  
    │  
    ▼  
LLM Manager ──── Phi-3-mini QLoRA (primary) → GPT-4o-mini (fallback)  
    │  
    ▼  
Output Guardrails ──── grounding check, PII redaction  
    │  
    ▼  
RAGResponse ──── answer + sources + confidence + latency  
""",  
        language="text",  
    )  
  
    st.subheader("Six AWS Well-Architected Pillars")  
    pillar_data = {  
        "Operational Excellence": "Running and monitoring systems to deliver business value and continually improve processes.",  
        "Security": "Protecting information, systems, and assets while delivering business value through risk assessments.",  
        "Reliability": "Ensuring a workload performs its intended function correctly and consistently.",  
        "Performance Efficiency": "Using computing resources efficiently to meet system requirements.",  
        "Cost Optimization": "Avoiding unnecessary costs and understanding spending over time.",  
        "Sustainability": "Minimizing the environmental impacts of running cloud workloads.",  
    }  
    for pillar, desc in pillar_data.items():  
        icon = PILLAR_ICONS[pillar]  
        st.markdown(f"**{icon} {pillar}** — {desc}")  
  
    st.divider()  
    st.subheader("Fine-Tuning Summary")  
    st.markdown(  
        """  
| Attribute | Value |  
|---|---|  
| Base model | `microsoft/Phi-3-mini-4k-instruct` (3.8B params) |  
| Method | QLoRA — 4-bit NF4 quantization + LoRA adapters |  
| Dataset | `databricks/databricks-dolly-15k` (closed_qa + information_extraction, ~2,800 examples) |  
| Platform | Google Colab T4 GPU (free tier) |  
| Adapter size | ~60MB |  
| ROUGE-L improvement | 0.387 → 0.431 (+0.044) |  
| What improved | Instruction following, context adherence, refusal behavior, citation format |  
| What did NOT improve | AWS domain knowledge (that comes from RAG retrieval) |  
"""  
    )  
  
    st.divider()  
    st.subheader("Evaluation Targets (RAGAS)")  
    col1, col2, col3, col4 = st.columns(4)  
    col1.metric("Faithfulness", "≥ 0.75")  
    col2.metric("Answer Relevancy", "≥ 0.75")  
    col3.metric("Context Precision", "≥ 0.70")  
    col4.metric("Context Recall", "≥ 0.70")  
  
    st.divider()  
    st.subheader("Known Limitations")  
    st.markdown(  
        """  
- **No multi-turn memory** — each request is stateless  
- **No streaming** — full response returned after 2-5s generation  
- **FAISS does not scale horizontally** — single-process in-memory index  
- **CPU inference is slow** — 30-60s without GPU; use OpenAI fallback for local dev  
- **Small eval dataset** (n=25) — ±0.05 margin of error on RAGAS scores  
- **Fine-tuned on general dataset** (dolly-15k), not AWS-specific Q&A pairs  
"""  
    )  
  
    st.divider()  
    st.subheader("API Reference")  
    st.markdown(f"Interactive docs: [{BACKEND_URL}/docs]({BACKEND_URL}/docs)")  
    st.markdown(  
        """  
| Method | Endpoint | Description |  
|---|---|---|  
| `POST` | `/generate` | RAG query — returns answer + sources + confidence |  
| `POST` | `/ingest` | Trigger document ingestion from `data/pdfs/` |  
| `POST` | `/ingest/upload` | Upload a PDF/MD/TXT file |  
| `GET` | `/health` | Liveness probe |  
| `GET` | `/health/ready` | Readiness probe |  
| `GET` | `/metrics` | Prometheus metrics |  
"""  
    )  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# Main  
# ─────────────────────────────────────────────────────────────────────────────  
  
def main() -> None:  
    use_fine_tuned, top_k, filter_pillar = render_sidebar()  
  
    tab_ask, tab_ingest, tab_status, tab_about = st.tabs(  
        ["Ask", "Ingest", "System Status", "About"]  
    )  
  
    with tab_ask:  
        render_ask_tab(use_fine_tuned, top_k, filter_pillar)  
  
    with tab_ingest:  
        render_ingest_tab()  
  
    with tab_status:  
        render_status_tab()  
  
    with tab_about:  
        render_about_tab()  
  
  
if __name__ == "__main__":  
    main()