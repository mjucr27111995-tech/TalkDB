"""TalkDB — Enterprise-grade natural language database interface."""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from talkdb.core.engine import TalkDB
from talkdb.databases.sqlite_db import SQLiteDatabase
from talkdb.llms.gemini_llm import GeminiLLM
from talkdb.llms.openai_llm import OpenAILLM
from talkdb.vectorstores.chroma_store import ChromaVectorStore

load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DB_PATH = os.getenv("DB_PATH", "demo.db")

st.set_page_config(page_title="TalkDB", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# Load external CSS
css_path = Path(__file__).parent / "talkdb" / "static" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────

def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_engine() -> TalkDB:
    api_key = GEMINI_API_KEY if LLM_PROVIDER == "gemini" else OPENAI_API_KEY
    if not api_key:
        st.error("API key not found. Add to `.env`.")
        st.stop()
    if not Path(DB_PATH).exists():
        st.error(f"Database `{DB_PATH}` not found.")
        st.stop()
    llm = GeminiLLM(api_key=api_key) if LLM_PROVIDER == "gemini" else OpenAILLM(api_key=api_key)
    vs = ChromaVectorStore()
    db = SQLiteDatabase()
    db.connect(db_path=DB_PATH)
    engine = TalkDB(llm=llm, vectorstore=vs, database=db)
    engine.index_all_ddls()
    return engine


def _get_tables() -> list[str]:
    db = SQLiteDatabase()
    db.connect(db_path=DB_PATH)
    t = list(db.get_all_ddls().keys())
    db.close()
    return t


# ── State ─────────────────────────────────────────────────

if "engine" not in st.session_state:
    st.session_state.engine = _build_engine()
    st.session_state.tables = _get_tables()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "explain_mode" not in st.session_state:
    st.session_state.explain_mode = True
if "chart_mode" not in st.session_state:
    st.session_state.chart_mode = True


# ── Sidebar ───────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()
    st.session_state.explain_mode = st.toggle("📝 SQL Explanations", value=st.session_state.explain_mode)
    st.session_state.chart_mode = st.toggle("📊 Auto Charts", value=st.session_state.chart_mode)
    st.divider()
    st.markdown("### 📋 Schema")
    for t in st.session_state.get("tables", []):
        st.code(t, language=None)
    st.divider()
    st.caption(f"**{LLM_PROVIDER.title()}** · `{DB_PATH}`")
    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.engine.clear_history()
        st.session_state.engine.clear_cache()
        st.rerun()


# ── Header ────────────────────────────────────────────────

tbl = len(st.session_state.get("tables", []))
qc = st.session_state.query_count
st.markdown(f"""
<div class="header-bar">
    <div class="header-left">
        <div class="header-icon">⚡</div>
        <div class="header-text">
            <h1>TalkDB</h1>
            <p>Natural Language Query Engine</p>
        </div>
    </div>
    <div class="header-right">
        <span class="h-chip h-chip-meta">{LLM_PROVIDER.title()}</span>
        <span class="h-chip h-chip-meta">{tbl} Tables</span>
        <span class="h-chip h-chip-meta">{qc} Queries</span>
        <span class="h-chip h-chip-live">● Connected</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── CHAT INPUT FIRST (so it always renders) ───────────────

pending = st.session_state.pop("pending_question", None)
question = pending or st.chat_input("Ask anything about your data...")


# ── Welcome (only when no messages) ──────────────────────

if not st.session_state.messages and not question:
    st.markdown("""
    <div class="hero">
        <h1>What do you want to know?</h1>
        <p>Ask questions about your data in natural language. TalkDB generates SQL, executes it, and explains the results.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    examples = [
        "🏙️  Which city has the most customers?",
        "💰  Total revenue by product category",
        "📦  Top 5 most ordered products",
        "📈  Monthly order trends",
        "🔍  Customers who haven't ordered",
        "🏷️  Average order value by city",
    ]
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_question = ex.split("  ", 1)[1]
                st.rerun()


# ── Render chat history ───────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="u-msg"><div class="u-bubble">{_esc(msg["content"])}</div></div>', unsafe_allow_html=True)
    else:
        # Card open
        st.markdown("""<div class="resp-card"><div class="resp-header">
            <div class="resp-avatar">⚡</div>
            <span class="resp-name">TalkDB Response</span>
        </div><div class="resp-body">""", unsafe_allow_html=True)

        st.markdown(f"""<div class="answer-block">
            <div class="answer-label">Answer</div>
            <div class="answer-text">{_esc(msg.get("answer", ""))}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="section"><div class="section-head">
            <div class="section-num">1</div>
            <span class="section-title">Generated SQL</span>
        </div></div>""", unsafe_allow_html=True)
        st.code(msg.get("sql", ""), language="sql")

        df = msg.get("dataframe")
        if df is not None and not df.empty:
            rc = len(df)
            st.markdown(f"""<div class="section"><div class="section-head">
                <div class="section-num">2</div>
                <span class="section-title">Results</span>
            </div></div>
            <div class="row-count">{rc} row{"s" if rc != 1 else ""} returned</div>""", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

        explanation = msg.get("explanation")
        if explanation:
            st.markdown("""<div class="section"><div class="section-head">
                <div class="section-num">3</div>
                <span class="section-title">Explanation</span>
            </div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="explain">
                <div class="explain-text">{explanation}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        if msg.get("chart"):
            st.plotly_chart(msg["chart"], use_container_width=True)


# ── Handle new question ───────────────────────────────────

if question:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": question})

    # Render user bubble
    st.markdown(f'<div class="u-msg"><div class="u-bubble">{_esc(question)}</div></div>', unsafe_allow_html=True)

    # Query
    error_msg = None
    result = None
    with st.spinner("Analyzing..."):
        try:
            result = st.session_state.engine.ask(
                question,
                explain=st.session_state.explain_mode,
                visualize=st.session_state.chart_mode,
            )
        except Exception as exc:
            error_msg = str(exc)

    if error_msg:
        st.error(f"❌ {error_msg}")
        st.session_state.messages.append({
            "role": "assistant", "answer": f"Error: {error_msg}",
            "sql": "", "dataframe": None, "explanation": None, "chart": None,
        })
    elif result:
        st.session_state.query_count += 1

        # Render response card
        st.markdown("""<div class="resp-card"><div class="resp-header">
            <div class="resp-avatar">⚡</div>
            <span class="resp-name">TalkDB Response</span>
        </div><div class="resp-body">""", unsafe_allow_html=True)

        st.markdown(f"""<div class="answer-block">
            <div class="answer-label">Answer</div>
            <div class="answer-text">{_esc(result.answer)}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="section"><div class="section-head">
            <div class="section-num">1</div>
            <span class="section-title">Generated SQL</span>
        </div></div>""", unsafe_allow_html=True)
        st.code(result.sql, language="sql")

        if not result.dataframe.empty:
            rc = len(result.dataframe)
            st.markdown(f"""<div class="section"><div class="section-head">
                <div class="section-num">2</div>
                <span class="section-title">Results</span>
            </div></div>
            <div class="row-count">{rc} row{"s" if rc != 1 else ""} returned</div>""", unsafe_allow_html=True)
            st.dataframe(result.dataframe, use_container_width=True, hide_index=True)

        if result.explanation:
            st.markdown("""<div class="section"><div class="section-head">
                <div class="section-num">3</div>
                <span class="section-title">Explanation</span>
            </div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="explain">
                <div class="explain-text">{result.explanation}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        if result.chart:
            st.plotly_chart(result.chart, use_container_width=True)

        st.session_state.messages.append({
            "role": "assistant",
            "answer": result.answer,
            "sql": result.sql,
            "dataframe": result.dataframe,
            "explanation": result.explanation,
            "chart": result.chart,
        })

    # Force rerun so chat input reappears cleanly
    st.rerun()
