import streamlit as st
import tempfile, os

from sec_agent import ask as sec_ask
from rag import create_vectorstore, ask_llm

# ── Page config ───────────────────────────────────────────

st.set_page_config(page_title="SEC Edgar Doc Researcher", layout="centered")

st.markdown("""
<style>
    .block-container { max-width: 780px; padding-top: 2rem; }
    .stChatMessage { border-radius: 10px; }
    .badge {
        display: inline-block;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 99px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .badge-sec  { background: #1e3a5f; color: #93c5fd; }
    .badge-rag  { background: #1e3d2f; color: #6ee7b7; }
    .badge-info { background: #2d2518; color: #fcd34d; }
</style>
""", unsafe_allow_html=True)

st.title("SEC Edgar Doc Researcher")
st.caption("Ask about SEC filings — or upload a PDF and ask about its contents.")

# ── Session state ─────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

# ── Sidebar: PDF upload ───────────────────────────────────

with st.sidebar:
    st.header("Upload Document")
    uploaded = st.file_uploader("Upload a PDF for RAG", type=["pdf"])

    if uploaded and not st.session_state.pdf_ready:
        with st.spinner("Chunking & embedding…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            create_vectorstore(tmp_path)
            os.unlink(tmp_path)
            st.session_state.pdf_ready = True
        st.success("PDF indexed. Ask away!")

    if st.session_state.pdf_ready:
        st.info("PDF loaded")
        if st.button("Clear PDF"):
            st.session_state.pdf_ready = False
            st.rerun()

    st.divider()
    st.markdown("""
**Routing logic**
- **SEC mode** — questions about company filings, revenue, earnings, 8-K events
- **RAG mode** — questions answered from your uploaded PDF
    
If a PDF is loaded, queries that mention the doc (e.g. *"in this report"*, *"according to the document"*) route to RAG. Everything else goes to the SEC agent.
    """)

# ── Intent router (simple keyword heuristic) ─────────────

SEC_KEYWORDS = [
    "10-k", "10-q", "8-k", "sec", "edgar", "filing",
    "revenue", "net income", "earnings", "annual report",
    "quarterly", "ticker", "stock", "ipo", "s-1", "proxy",
]

def route(query: str) -> str:
    """Return 'rag' or 'sec'."""
    q = query.lower()
    doc_hints = ["this document", "this report", "this pdf", "uploaded", "according to", "in the file"]
    if st.session_state.pdf_ready and any(h in q for h in doc_hints):
        return "rag"
    if not st.session_state.pdf_ready:
        return "sec"
    if any(kw in q for kw in SEC_KEYWORDS):
        return "sec"
    # default: if PDF is loaded, use RAG; else SEC
    return "rag" if st.session_state.pdf_ready else "sec"

# ── Chat history ──────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "mode" in msg:
            label, cls = ("SEC Agent", "badge-sec") if msg["mode"] == "sec" else ("PDF RAG", "badge-rag")
            st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────

if prompt := st.chat_input("Ask a financial question or query your PDF…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    mode = route(prompt)

    with st.chat_message("assistant"):
        label, cls = ("SEC Agent", "badge-sec") if mode == "sec" else ("PDF RAG", "badge-rag")
        st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)

        with st.spinner("Fetching & reasoning…" if mode == "sec" else "Searching document…"):
            if mode == "sec":
                answer = sec_ask(prompt)
            else:
                answer = ask_llm(prompt)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "mode": mode})