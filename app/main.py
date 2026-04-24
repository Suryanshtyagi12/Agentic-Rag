"""
main.py
-------
Streamlit UI for the Agentic RAG system.

Run with:
    streamlit run app/main.py
"""

# ── Force UTF-8 on Windows (fixes charmap errors with ✓ etc.) ───────────────
import sys
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader        import load_pdf
from src.ingestion.parser        import parse_pdf
from src.ingestion.chunking      import chunk_elements
from src.ingestion.run_ingestion import _cache_path_for, _load_from_cache  # caching helpers
from src.retriever.retriever     import Retriever
from src.agent.agent             import run_agent, AgentResult
from src.llm.groq_client         import MODEL_CONFIG  # centralised model config
import os as _os
ACTIVE_MODEL = _os.getenv("GROQ_MODEL", MODEL_CONFIG["primary"])  # respects env override

# ── Constants ────────────────────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
INDEX_NAME    = "rag_index"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "AgenticRAG",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85);
        border-right: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
    }

    /* ── Cards ── */
    .rag-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
        transition: border-color 0.2s;
    }
    .rag-card:hover { border-color: rgba(139,92,246,0.5); }

    /* ── Chunk card ── */
    .chunk-card {
        background: rgba(139,92,246,0.08);
        border: 1px solid rgba(139,92,246,0.25);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 0.88rem;
    }
    .chunk-badge {
        display: inline-block;
        background: rgba(139,92,246,0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        color: #c4b5fd;
    }
    .score-badge {
        display: inline-block;
        background: rgba(16,185,129,0.2);
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #6ee7b7;
    }

    /* ── Answer box ── */
    .answer-box {
        background: linear-gradient(135deg, rgba(139,92,246,0.12), rgba(59,130,246,0.08));
        border: 1px solid rgba(139,92,246,0.35);
        border-radius: 16px;
        padding: 24px 28px;
        line-height: 1.75;
    }

    /* ── Metric pills ── */
    .metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
    .metric-pill {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 30px;
        padding: 6px 16px;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .metric-pill span { color: #e2e8f0; font-weight: 600; }

    /* ── Step trace ── */
    .step-block {
        background: rgba(0,0,0,0.25);
        border-left: 3px solid #8b5cf6;
        border-radius: 0 8px 8px 0;
        padding: 10px 16px;
        margin-bottom: 10px;
        font-family: monospace;
        font-size: 0.82rem;
        color: #a5b4fc;
        white-space: pre-wrap;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.4rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(139,92,246,0.4);
        border-radius: 12px;
        padding: 12px;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(139,92,246,0.6) !important;
        box-shadow: 0 0 0 2px rgba(139,92,246,0.15) !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* ── Success / warning / error ── */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 8px !important;
        color: #c4b5fd !important;
    }

    /* ── Header gradient text ── */
    .gradient-title {
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 4px;
    }
    .subtitle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 24px; }

    /* ── Progress bar ── */
    .stProgress > div > div { background: linear-gradient(90deg,#7c3aed,#4f46e5) !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "retriever"       : None,
        "index_ready"     : False,
        "pdf_name"        : None,
        "total_chunks"    : 0,
        "chat_history"    : [],   # list of {query, result}
        "ingestion_log"   : [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Cached resource helpers ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_cached_retriever(index_name: str) -> Retriever:
    """Load an existing FAISS index once and cache it across reruns."""
    r = Retriever(index_name=index_name)
    r.load()
    return r


@st.cache_resource(show_spinner=False)
def get_cached_embedder():
    """
    Load the sentence-transformer model ONCE and keep it in memory.
    Subsequent ingestions reuse the cached model (saves ~10-20s per run).
    """
    from sentence_transformers import SentenceTransformer
    print("[app] Loading embedding model (cached) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Monkey-patch the module-level singleton so embedder.py reuses this instance
    import src.embeddings.embedder as emb_module
    emb_module._model = model
    print("[app] Embedding model ready.")
    return model


# ── Ingestion pipeline ────────────────────────────────────────────────────────
def run_ingestion_pipeline(uploaded_file) -> bool:
    """
    Save the uploaded PDF, run ingestion, build FAISS index.
    Returns True on success.
    """
    log = []
    progress = st.progress(0, text="Starting ingestion ...")

    try:
        # 1. Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        log.append(f"✓ PDF saved temporarily → {Path(tmp_path).name}")
        progress.progress(10, "PDF saved ...")

        # 2. Load + validate
        pdf_path = load_pdf(tmp_path)
        log.append(f"✓ Loaded: {pdf_path.name}")
        progress.progress(20, "PDF validated ...")

        # 3. Cache check — skip re-ingestion if same PDF was processed before
        cache_file = _cache_path_for(pdf_path)
        if cache_file.exists():
            log.append(f"⚡ Cache hit — loading pre-processed chunks from {cache_file.name}")
            progress.progress(60, "Loading from cache ...")
            chunks = _load_from_cache(cache_file)
            log.append(f"✓ Loaded {len(chunks)} cached chunks")
        else:
            # 4. Parse (parallel PyMuPDF)
            elements = parse_pdf(pdf_path)
            log.append(f"✓ Parsed: {len(elements)} elements (text + tables + images)")
            progress.progress(50, f"Parsed {len(elements)} elements ...")

            # 5. Chunk (fast char-based)
            chunks = chunk_elements(elements)
            log.append(f"✓ Chunked: {len(chunks)} chunks created")
            progress.progress(65, f"Created {len(chunks)} chunks ...")

            # 6. Save processed JSON cache
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            log.append(f"✓ Chunks cached → {cache_file.name}")

        progress.progress(75, "Building FAISS index ...")

        # 7. Embed + index
        retriever = Retriever(index_name=INDEX_NAME)
        retriever.build_from_chunks(chunks)
        log.append(f"✓ FAISS index built — {len(chunks)} vectors")
        progress.progress(95, "Index built ...")

        # 8. Store in session
        st.session_state.retriever    = retriever
        st.session_state.index_ready  = True
        st.session_state.pdf_name     = uploaded_file.name
        st.session_state.total_chunks = len(chunks)
        st.session_state.ingestion_log = log

        # Clear get_cached_retriever cache so next load picks up new index
        get_cached_retriever.clear()

        progress.progress(100, "✓ Done!")
        time.sleep(0.4)
        progress.empty()
        os.unlink(tmp_path)
        return True

    except Exception as e:
        progress.empty()
        log.append(f"✗ Error: {type(e).__name__}: {e}")
        st.session_state.ingestion_log = log
        return False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="gradient-title">🧠 AgenticRAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Powered by Groq × LLaMA 3 × FAISS</div>', unsafe_allow_html=True)
    st.divider()

    # Status indicator
    if st.session_state.index_ready:
        st.success(f"✅ Index ready: **{st.session_state.pdf_name}**")
        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-pill">Chunks <span>{st.session_state.total_chunks}</span></div>'
            f'<div class="metric-pill">Model <span>{ACTIVE_MODEL}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ No index loaded. Upload a PDF below.")

    st.divider()

    # ── PDF Upload ─────────────────────────────────────────────────────
    st.markdown("#### 📄 Upload PDF")
    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_upload",
    )

    if uploaded:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"📎 {uploaded.name}")
        with col2:
            process_btn = st.button("⚡ Process", use_container_width=True)

        if process_btn:
            with st.spinner("Running ingestion pipeline ..."):
                success = run_ingestion_pipeline(uploaded)
            if success:
                st.success("✅ PDF processed successfully!")
                st.rerun()
            else:
                st.error("❌ Ingestion failed. Check logs below.")

    # ── Ingestion log ──────────────────────────────────────────────────
    if st.session_state.ingestion_log:
        with st.expander("📋 Ingestion Log", expanded=False):
            for line in st.session_state.ingestion_log:
                st.markdown(f"`{line}`")

    st.divider()

    # ── Load existing index ────────────────────────────────────────────
    index_file = VECTOR_DB_DIR / f"{INDEX_NAME}.index"
    if index_file.exists() and not st.session_state.index_ready:
        if st.button("🔄 Load Existing Index", use_container_width=True):
            with st.spinner("Loading saved index ..."):
                try:
                    r = get_cached_retriever(INDEX_NAME)
                    st.session_state.retriever   = r
                    st.session_state.index_ready = True
                    st.session_state.pdf_name    = "Existing index"
                    st.session_state.total_chunks = r._db.total_vectors
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load index: {e}")

    # ── Clear chat ─────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.caption("AgenticRAG © 2025 | [GitHub](https://github.com/Suryanshtyagi12/Agentic-RAG)")


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="gradient-title">🧠 AgenticRAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions about your PDF — powered by an agentic Think → Retrieve → Evaluate → Answer loop</div>', unsafe_allow_html=True)

if not st.session_state.index_ready:
    # ── Welcome screen ─────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    for col, icon, title, desc in [
        (col1, "📄", "1. Upload PDF", "Drop your PDF in the sidebar and click **Process**."),
        (col2, "⚡", "2. Auto-Ingestion", "Text, tables, and images are extracted and embedded into FAISS."),
        (col3, "💬", "3. Ask Questions", "The agent retrieves relevant context and reasons before answering."),
    ]:
        with col:
            st.markdown(
                f'<div class="rag-card" style="text-align:center">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<div style="font-weight:600;margin:8px 0 4px">{title}</div>'
                f'<div style="color:#94a3b8;font-size:0.88rem">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.stop()

# ── Chat interface ────────────────────────────────────────────────────────────

# Render chat history
for item in st.session_state.chat_history:
    q = item["query"]
    r: AgentResult = item["result"]

    # User bubble
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**{q}**")

    # Assistant bubble
    with st.chat_message("assistant", avatar="🧠"):
        # Answer
        st.markdown(
            f'<div class="answer-box">{r.answer}</div>',
            unsafe_allow_html=True,
        )

        # Metrics row
        fallback_badge = '<div class="metric-pill" style="color:#f87171">Fallback used</div>' if r.fallback_used else ""
        st.markdown(
            f'<div class="metric-row" style="margin-top:12px">'
            f'<div class="metric-pill">Iterations <span>{r.iterations}</span></div>'
            f'<div class="metric-pill">Chunks retrieved <span>{len(r.retrieved_chunks)}</span></div>'
            f'{fallback_badge}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Retrieved chunks
        if r.retrieved_chunks:
            with st.expander(f"📚 Retrieved Chunks ({len(r.retrieved_chunks)})", expanded=False):
                for chunk in r.retrieved_chunks:
                    rank      = chunk["_rank"]
                    page      = chunk.get("page", "?")
                    ctype     = chunk.get("type", "text")
                    score     = chunk["_score"]
                    content   = chunk["content"]
                    preview   = content[:500]
                    ellipsis  = "..." if len(content) > 500 else ""
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<span class="chunk-badge">#{rank}</span>'
                        f'<span class="chunk-badge">Page {page}</span>'
                        f'<span class="chunk-badge">{ctype}</span>'
                        f'<span class="score-badge">score {score:.4f}</span>'
                        f'<div style="margin-top:10px;color:#cbd5e1;line-height:1.6">{preview}{ellipsis}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Agent reasoning trace
        with st.expander("🔍 Agent Reasoning Trace", expanded=False):
            if r.think_output:
                st.markdown("**🤔 Think Step:**")
                st.markdown(f'<div class="step-block">{r.think_output}</div>', unsafe_allow_html=True)
            if r.evaluate_output:
                st.markdown("**⚖️ Evaluate Step:**")
                st.markdown(f'<div class="step-block">{r.evaluate_output}</div>', unsafe_allow_html=True)

    st.divider()


# ── Query input ───────────────────────────────────────────────────────────────
st.markdown("#### 💬 Ask a Question")

with st.form(key="query_form", clear_on_submit=True):
    query_input = st.text_area(
        "Your question",
        placeholder="e.g. What are the main findings of this document?",
        height=90,
        label_visibility="collapsed",
        key="query_text",
    )
    col_a, col_b = st.columns([5, 1])
    with col_b:
        submitted = st.form_submit_button("🚀 Ask", use_container_width=True)

if submitted and query_input.strip():
    with st.spinner("🤖 Agent is thinking ..."):
        try:
            retriever = st.session_state.retriever
            agent_result = run_agent(query_input.strip(), retriever)
            st.session_state.chat_history.append({
                "query" : query_input.strip(),
                "result": agent_result,
            })
            st.rerun()
        except Exception as e:
            st.error(f"❌ Agent error: {type(e).__name__}: {e}")

elif submitted:
    st.warning("Please enter a question.")
