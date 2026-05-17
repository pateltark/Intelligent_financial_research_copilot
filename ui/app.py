"""
FILE: ui/app.py  (updated — with auth + per-user chat history)
================================================================
NEW vs old app.py:
  ✅ Login / Register page
  ✅ JWT session management
  ✅ Per-user conversation history in sidebar
  ✅ Click any past chat to reload it
  ✅ New chat button + delete conversation
  ✅ Per-user watchlist saved to DB
  ✅ Every message auto-saved to SQLite

RUN: streamlit run ui/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Financial Research Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .user-msg {
        background:#1e2130; border:1px solid #2d3149; border-radius:12px;
        padding:12px 16px; margin:8px 0; color:#e8e8f0;
    }
    .ai-msg {
        background:#141824; border:1px solid #1a2f4a; border-left:3px solid #4a5fd4;
        border-radius:0 12px 12px 0; padding:12px 16px; margin:8px 0; color:#e8e8f0;
    }
    .source-card {
        background:#1a1f35; border:1px solid #2a3050; border-radius:8px;
        padding:10px 14px; margin:4px 0; font-size:13px; color:#c8c8e0;
    }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
def init_session():
    for k, v in {
        "token": None, "user_id": None, "user_email": None, "username": None,
        "current_conv_id": None, "messages": [], "prefill_q": "",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

def is_logged_in():
    if not st.session_state.get("token"):
        return False
    from auth.auth import is_token_valid
    return is_token_valid(st.session_state["token"])

def logout():
    for k in ["token","user_id","user_email","username","current_conv_id","messages"]:
        st.session_state[k] = None
    st.session_state["messages"] = []
    st.rerun()


# ══════════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════════
def show_auth_page():
    st.markdown("## 📊 Financial Research Copilot")
    st.markdown("*AI-powered SEC filing analysis — sign in to save your research*")
    st.divider()

    _, col, _ = st.columns([1, 2, 1])
    with col:
        tab_login, tab_reg = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            st.markdown("#### Welcome back")
            email = st.text_input("Email", key="li_email", placeholder="you@email.com")
            pw    = st.text_input("Password", key="li_pw", type="password", placeholder="••••••••")

            if st.button("Sign In →", type="primary", use_container_width=True, key="btn_li"):
                if not email or not pw:
                    st.error("Please fill in all fields")
                else:
                    from database.db_operations import login_user
                    from auth.auth import create_token
                    res = login_user(email, pw)
                    if res["success"]:
                        u = res["user"]   # now a plain dict
                        st.session_state.update({
                            "token": create_token(u["id"], u["email"]),
                            "user_id": u["id"], "user_email": u["email"], "username": u["username"],
                        })
                        st.rerun()
                    else:
                        st.error(res["error"])

        with tab_reg:
            st.markdown("#### Create your account")
            ru = st.text_input("Username",         key="ru", placeholder="yourname")
            re = st.text_input("Email",             key="re", placeholder="you@email.com")
            rp = st.text_input("Password",          key="rp", type="password", placeholder="min 6 chars")
            rc = st.text_input("Confirm password",  key="rc", type="password", placeholder="repeat password")

            if st.button("Create Account →", type="primary", use_container_width=True, key="btn_reg"):
                if not all([ru, re, rp, rc]):
                    st.error("Please fill in all fields")
                elif len(rp) < 6:
                    st.error("Password must be at least 6 characters")
                elif rp != rc:
                    st.error("Passwords don't match")
                else:
                    from database.db_operations import create_user
                    from auth.auth import create_token
                    res = create_user(re, ru, rp)
                    if res["success"]:
                        u = res["user"]   # now a plain dict
                        st.session_state.update({
                            "token": create_token(u["id"], u["email"]),
                            "user_id": u["id"], "user_email": u["email"], "username": u["username"],
                        })
                        st.rerun()
                    else:
                        st.error(res["error"])


# ══════════════════════════════════════════════════════════════════
# RAG (cached so model loads once)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_rag():
    try:
        from rag.chain import answer, compare, get_sentiment_timeline
        from processing.vector_store import get_stats
        return answer, compare, get_sentiment_timeline, get_stats, None
    except Exception as e:
        return None, None, None, None, str(e)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
def show_sidebar():
    from database.db_operations import (
        get_user_conversations, get_conversation_messages,
        delete_conversation, get_watchlist, add_to_watchlist, remove_from_watchlist
    )
    uid = st.session_state["user_id"]

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['username']}")
        st.caption(st.session_state["user_email"])
        if st.button("Sign out", use_container_width=True):
            logout()
        st.divider()

        if st.button("✏️ New Chat", type="primary", use_container_width=True, key="new_chat"):
            st.session_state["current_conv_id"] = None
            st.session_state["messages"]        = []
            st.rerun()

        st.divider()
        st.markdown("** Chat History**")
        convos = get_user_conversations(uid)

        if not convos:
            st.caption("No chats yet — ask your first question!")
        for conv in convos:
            is_active = conv["id"] == st.session_state.get("current_conv_id")
            c1, c2 = st.columns([5, 1])
            with c1:
                label = ("▶ " if is_active else "") + conv["title"][:38]
                if st.button(label, key=f"c_{conv['id']}", use_container_width=True):
                    st.session_state["current_conv_id"] = conv["id"]
                    st.session_state["messages"] = get_conversation_messages(conv["id"])
                    st.rerun()
            with c2:
                if st.button("🗑", key=f"d_{conv['id']}"):
                    delete_conversation(conv["id"], uid)
                    if st.session_state["current_conv_id"] == conv["id"]:
                        st.session_state["current_conv_id"] = None
                        st.session_state["messages"]        = []
                    st.rerun()
            ticker_info = f" · {conv['ticker']}" if conv["ticker"] else ""
            st.caption(f"{conv['message_count']} msgs{ticker_info} · {conv['updated_at'][:10]}")

        st.divider()
        st.markdown("** My Watchlist**")
        watchlist = get_watchlist(uid)
        for ticker in watchlist:
            w1, w2 = st.columns([4, 1])
            with w1: st.markdown(f"`{ticker}`")
            with w2:
                if st.button("✕", key=f"wr_{ticker}"):
                    remove_from_watchlist(uid, ticker)
                    st.rerun()

        new_t = st.text_input("Add ticker:", placeholder="NVDA", label_visibility="collapsed", key="add_t")
        if st.button("+ Add to watchlist") and new_t:
            if not add_to_watchlist(uid, new_t.upper()):
                st.warning("Already in watchlist")
            st.rerun()

        st.divider()
        st.markdown("** Search Filters**")
        all_opts = ["All"] + watchlist + ["AAPL","MSFT","GOOGL","NVDA","TSLA"]
        tf = st.selectbox("Company:", list(dict.fromkeys(all_opts)), key="tf")
        ty = st.selectbox("Filing type:", ["All","10-K","10-Q","8-K","NEWS"], key="ty")
        nc = st.slider("Context chunks:", 3, 10, 5, key="nc")

        st.session_state["_tf"] = None if tf == "All" else tf
        st.session_state["_ty"] = None if ty == "All" else ty
        st.session_state["_nc"] = nc


# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════
def show_main_app():
    show_sidebar()
    answer_fn, compare_fn, sentiment_fn, stats_fn, err = load_rag()
    uid = st.session_state["user_id"]

    st.markdown("#  Financial Research Copilot")
    tab1, tab2, tab3 = st.tabs([" Chat", "Sentiment", " Compare"])

    # ──────────────────────────────────────
    # TAB 1 — CHAT
    # ──────────────────────────────────────
    with tab1:
        from database.db_operations import create_conversation, save_message, get_conversation_messages

        # Status bar
        if stats_fn:
            try:
                s = stats_fn()
                if s["total_chunks"] == 0:
                    st.warning(" No data indexed. Run: `python run.py pipeline`")
                else:
                    st.caption(f" {s['total_chunks']:,} chunks · Tickers: {', '.join(s['tickers'])}")
            except: pass

        # Empty state
        msgs = st.session_state.get("messages", [])
        if not msgs and not st.session_state.get("current_conv_id"):
            st.markdown("""
            <div style='text-align:center;padding:3rem;color:#666880'>
                <div style='font-size:48px'>📊</div>
                <div style='font-size:18px;color:#9090b0;margin:0.5rem 0'>Ask anything about your companies</div>
                <div style='font-size:14px'>Answers are grounded in real SEC filings with source citations</div>
            </div>""", unsafe_allow_html=True)

            cols = st.columns(2)
            suggestions = [
                "What are Apple's biggest risk factors?",
                "How did Microsoft's cloud revenue grow?",
                "What did Nvidia say about AI chip demand?",
                "What is Tesla's outlook for next quarter?",
            ]
            for i, s in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(s, key=f"sg_{i}", use_container_width=True):
                        st.session_state["prefill_q"] = s
                        st.rerun()

        # Render messages
        for msg in msgs:
            ts = msg.get("created_at", "")
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg">🧑 <b>You</b> <span style="font-size:11px;color:#666880">· {ts}</span><br><br>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-msg">🤖 <b>Copilot</b> <span style="font-size:11px;color:#666880">· {ts}</span><br><br>{msg["content"]}</div>', unsafe_allow_html=True)
                if msg.get("citations"):
                    with st.expander(f"📎 {len(msg['citations'])} sources cited"):
                        for cite in msg["citations"]:
                            st.markdown(f"""<div class="source-card">
                                <b>[Source {cite['source_num']}]</b> {cite['ticker']} | {cite['filing_type']} | {cite['date']}<br>
                                <i>{cite['section']}</i><br>
                                <small>{cite.get('preview','')[:180]}...</small>
                            </div>""", unsafe_allow_html=True)

        st.divider()

        # Input
        prefill = st.session_state.pop("prefill_q", "") if st.session_state.get("prefill_q") else ""
        c1, c2 = st.columns([6, 1])
        with c1:
            question = st.text_input("Ask:", value=prefill, placeholder="What are the biggest risks facing Apple?", label_visibility="collapsed", key="qi")
        with c2:
            ask = st.button("Ask →", type="primary", use_container_width=True)

        if ask and question.strip():
            if not answer_fn:
                st.error(f"RAG error: {err}")
            else:
                tf = st.session_state.get("_tf")
                ty = st.session_state.get("_ty")
                nc = st.session_state.get("_nc", 5)

                # Create conversation if needed
                if not st.session_state["current_conv_id"]:
                    conv = create_conversation(uid, ticker=tf, filing_type=ty)
                    st.session_state["current_conv_id"] = conv.id

                cid = st.session_state["current_conv_id"]
                save_message(cid, "user", question)

                with st.spinner(" Searching filings..."):
                    try:
                        res = answer_fn(question=question, ticker=tf, filing_type=ty, n_chunks=nc)
                        save_message(cid, "assistant", res["answer"],
                                     citations=res["citations"],
                                     chunks_used=res.get("chunks_used"),
                                     model_used=res.get("model"))
                    except Exception as e:
                        save_message(cid, "assistant", f"Error: {e}")

                st.session_state["messages"] = get_conversation_messages(cid)
                st.rerun()

    # ──────────────────────────────────────
    # TAB 2 — SENTIMENT
    # ──────────────────────────────────────
    with tab2:
        st.markdown("### 📈 News Sentiment Timeline")
        chart_t = st.selectbox("Company:", ["AAPL","MSFT","GOOGL","NVDA","TSLA"], key="st_t")
        if st.button("Load Chart", type="primary", key="btn_sc") and sentiment_fn:
            with st.spinner("Loading..."):
                tl = sentiment_fn(chart_t)
            if tl:
                df = pd.DataFrame(tl)
                df["date"] = pd.to_datetime(df["date"])
                colors = ["#22c55e" if s > 0.15 else "#ef4444" if s < -0.15 else "#f59e0b" for s in df["score"]]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["date"], y=df["score"], mode="lines+markers",
                    line=dict(color="#4a5fd4", width=2), marker=dict(color=colors, size=8)))
                fig.add_hline(y=0, line_dash="dash", line_color="#666", opacity=0.5)
                fig.update_layout(title=f"{chart_t} Sentiment", plot_bgcolor="#0f1117",
                    paper_bgcolor="#0f1117", font=dict(color="#e8e8f0"),
                    xaxis=dict(gridcolor="#2d3149"), yaxis=dict(gridcolor="#2d3149", range=[-1,1]), height=400)
                st.plotly_chart(fig, use_container_width=True)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Points", len(df))
                c2.metric("Avg", f"{df['score'].mean():.3f}")
                c3.metric("Best", f"{df['score'].max():.3f}")
                c4.metric("Worst", f"{df['score'].min():.3f}")
            else:
                st.info("No data. Run `python ingestion/news_scraper.py` first.")

    # ──────────────────────────────────────
    # TAB 3 — COMPARE
    # ──────────────────────────────────────
    with tab3:
        st.markdown("### ⚖️ Compare Companies")
        cq = st.text_input("Question:", value="What are the biggest risk factors?", key="cq")
        ct = st.multiselect("Companies:", ["AAPL","MSFT","GOOGL","NVDA","TSLA","AMZN","META"], default=["AAPL","MSFT"], key="ct")
        if st.button("Compare →", type="primary", key="btn_cmp"):
            if not compare_fn: st.error("RAG not loaded")
            elif len(ct) < 2:  st.warning("Select at least 2 companies")
            else:
                with st.spinner(f"Analyzing {', '.join(ct)}..."):
                    try:
                        res  = compare_fn(cq, ct)
                        cols = st.columns(len(ct))
                        for col, ticker in zip(cols, ct):
                            with col:
                                comp = res["comparisons"].get(ticker, {})
                                st.markdown(f"#### {ticker}")
                                st.markdown(comp.get("answer", "No data"))
                                if comp.get("citations"):
                                    st.caption(f"📎 {len(comp['citations'])} sources")
                    except Exception as e:
                        st.error(f"Compare failed: {e}")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════
def main():
    from database.models import init_db
    init_db()
    init_session()
    if not is_logged_in():
        show_auth_page()
    else:
        show_main_app()

main()