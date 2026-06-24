"""
WebIntel AI — Premium RAG Dashboard & Chatbot

A true Retrieval-Augmented Generation dashboard designed to prioritize grounded Q&A, 
transparency of retrieval, crawl statistics, and strict factual validation.
"""

import streamlit as st
import requests
import time
import re

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="WebIntel AI — Grounded RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Backend Config
# ──────────────────────────────────────────────

API_BASE = "http://localhost:8001"

# ──────────────────────────────────────────────
# Custom CSS — Premium Glassmorphism Theme
# ──────────────────────────────────────────────

st.markdown("""
<style>
/* ── Import Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');

/* ── Global Styles ── */
.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}

/* ── Hide Streamlit Elements ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Glassmorphic Dashboard Cards ── */
.glass-card {
    background: rgba(22, 27, 34, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
}

.glass-card-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 1px solid rgba(48, 54, 61, 0.5);
    padding-bottom: 0.5rem;
}

/* ── RAG Debug Metrics ── */
.debug-metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.debug-metric {
    background: rgba(33, 38, 45, 0.8);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 8px;
    padding: 0.75rem;
    text-align: center;
}

.debug-metric-val {
    font-size: 1.3rem;
    font-weight: 700;
    color: #58a6ff;
}

.debug-metric-val.purple {
    color: #bc8cff;
}

.debug-metric-val.orange {
    color: #f0883e;
}

.debug-metric-lbl {
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 0.15rem;
}

/* ── Source Citation Cards ── */
.citation-card {
    background: rgba(22, 27, 34, 0.9);
    border: 1px solid rgba(56, 139, 253, 0.4);
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 0.85rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
}

.citation-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
    font-weight: 600;
}

.citation-title {
    color: #58a6ff;
}

.citation-score {
    background: rgba(56, 139, 253, 0.15);
    color: #58a6ff;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-size: 0.75rem;
}

.citation-url {
    font-size: 0.75rem;
    color: #8b949e;
    word-break: break-all;
    margin-bottom: 0.4rem;
}

.citation-text {
    color: #c9d1d9;
    font-style: italic;
    line-height: 1.4;
}

/* ── Score Rings & Metric Cards ── */
.score-ring {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0 auto 0.5rem auto;
    position: relative;
}

.score-ring::before {
    content: '';
    position: absolute;
    inset: 4px;
    border-radius: 50%;
    background: #0d1117;
}

.score-ring span {
    position: relative;
    z-index: 1;
}

.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-card .metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0.15rem 0;
}

.metric-card .metric-label {
    font-size: 0.8rem;
    color: #8b949e;
    font-weight: 600;
    text-transform: uppercase;
}

.metric-card .metric-reason {
    font-size: 0.75rem;
    color: #6e7681;
    margin-top: 0.35rem;
    line-height: 1.3;
}

/* ── Opportunity Cards ── */
.opp-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.6rem;
}

.opp-title {
    font-weight: 600;
    color: #e6edf3;
    font-size: 0.95rem;
}

.opp-badge {
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
}

.takeaway-card {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    border-radius: 0 8px 8px 0;
}

/* ── Tabs Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem;
}

.stTabs [data-baseweb="tab"] {
    font-size: 1rem;
    font-weight: 600;
    height: 3rem;
}

/* ── JetBrains Mono Font ── */
.mono-font {
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Persona Configurations
# ──────────────────────────────────────────────

PERSONAS = {
    "student": {"label": "🎓 Student", "color": "#58a6ff", "desc": "Learning, projects, research"},
    "job_seeker": {"label": "💼 Job Seeker", "color": "#f0883e", "desc": "Careers, skills, interviews"},
    "developer": {"label": "👨‍💻 Developer", "color": "#3fb950", "desc": "APIs, docs, tech stack"},
    "researcher": {"label": "🔬 Researcher", "color": "#bc8cff", "desc": "Papers, innovation, trends"},
    "investor": {"label": "📈 Investor", "color": "#f78166", "desc": "Business, market, growth"},
}

CATEGORY_ICONS = {
    "job": "💼", "internship": "🎓", "course": "📚", "certification": "🏆",
    "event": "📅", "program": "🚀", "scholarship": "🎯", "hackathon": "⚡",
    "competition": "🏅", "developer_program": "👨‍💻", "partner_program": "🤝",
}

CATEGORY_COLORS = {
    "job": "#f0883e", "internship": "#58a6ff", "course": "#3fb950",
    "certification": "#d2a8ff", "event": "#f78166", "program": "#79c0ff",
    "scholarship": "#ffa657", "hackathon": "#ff7b72", "competition": "#7ee787",
    "developer_program": "#3fb950", "partner_program": "#bc8cff",
}

DIFFICULTY_COLORS = {
    "beginner": "#3fb950", "intermediate": "#d29922", "advanced": "#f85149",
}

PRIORITY_COLORS = {
    "high": "#f85149", "medium": "#d29922", "low": "#3fb950",
}

# ──────────────────────────────────────────────
# API Call Wrapper
# ──────────────────────────────────────────────

def call_api(endpoint: str, method: str = "GET", payload: dict = None) -> dict | None:
    """Call backend FastAPI server."""
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "POST":
            resp = requests.post(url, json=payload, timeout=120)
        else:
            resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("❌ Cannot connect to backend. Verify FastAPI runs on port 8000.")
        return None
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"❌ API Error: {detail}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None

# ──────────────────────────────────────────────
# Helper Score Styles
# ──────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 0.8: return "#3fb950"
    elif score >= 0.6: return "#58a6ff"
    elif score >= 0.4: return "#d29922"
    else: return "#f85149"

def score_gradient(score: float) -> str:
    color = score_color(score)
    pct = score * 100
    return f"background: conic-gradient({color} {pct}%, #30363d {pct}%);"

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 1.8rem; font-weight: 900;
                    background: linear-gradient(135deg, #58a6ff, #bc8cff);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;">
            🧠 WebIntel RAG
        </div>
        <div style="font-size: 0.8rem; color: #8b949e; margin-top: 0.25rem; font-weight: 500;">
            Grounded Website Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # URL Input
    url_input = st.text_input(
        "🌐 Target Website URL",
        placeholder="https://github.com",
        help="Enter any website or documentation link to crawl and index",
    )

    # Persona Selector
    st.markdown("**👤 Select Target Persona**")
    persona_options = list(PERSONAS.keys())
    persona_labels = [f"{v['label']}" for v in PERSONAS.values()]
    selected_persona_label = st.radio(
        "Persona Selector",
        options=persona_labels,
        index=0,
        label_visibility="collapsed",
    )
    selected_persona = persona_options[persona_labels.index(selected_persona_label)]
    persona_info = PERSONAS[selected_persona]
    st.caption(f"_{persona_info['desc']}_")

    st.divider()

    # Crawl / Analyze Button
    analyze_clicked = st.button(
        "🚀 Crawl & Build RAG Index",
        use_container_width=True,
        type="primary",
        disabled=not url_input,
    )

    if st.session_state.get("analysis_data"):
        st.divider()
        st.caption("⚡ Active Session")
        curr = st.session_state["analysis_data"]
        st.markdown(f"**{curr.get('title', 'N/A')}**")
        st.caption(f"`{curr.get('url', '')}`")
        
        if st.button("🗑️ Reset RAG Session", use_container_width=True):
            st.session_state.pop("analysis_data", None)
            st.session_state.pop("chat_messages", None)
            st.session_state.pop("last_response", None)
            st.rerun()

    st.divider()
    st.caption("WebIntel AI — Hackathon Grounding Version")
    st.caption("Powered by Gemini 2.5 Flash + Local FAISS Vector Index")

# ──────────────────────────────────────────────
# Landing / Hero
# ──────────────────────────────────────────────

def render_hero():
    st.markdown("""
    <div style="text-align: center; padding: 4rem 1rem 2rem 1rem;">
        <div style="font-size: 3.5rem; font-weight: 900;
                    background: linear-gradient(135deg, #58a6ff, #bc8cff, #f78166);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; line-height: 1.1; margin-bottom: 0.5rem;">
            True Retrieval-Augmented Generation
        </div>
        <div style="font-size: 1.2rem; color: #8b949e; max-width: 700px; margin: 0 auto 3rem auto; line-height: 1.6;">
            Enter a website URL to recursively crawl internal pages, parse their contents into local FAISS vector stores,
            and engage in strict, hallucination-free grounded chatbot conversations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    features = [
        ("🔒 Zero-Hallucination Policy", "Gemini is constrained to retrieved vector chunks. Answers without evidence are strictly rejected."),
        ("🕸️ Recursive Scraper", "Crawl main URL and up to 14 relevant subpages, indexing internal resources and documentation."),
        ("🔍 Retrieval Transparency", "Inspect exact similarity scores, final prompts, token counts, and matching source cards in real-time.")
    ]
    for col, (title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; min-height: 180px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #e6edf3; margin-bottom: 0.5rem;">{title}</div>
                <div style="font-size: 0.85rem; color: #8b949e; line-height: 1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Main Application Dashboard
# ──────────────────────────────────────────────

data = st.session_state.get("analysis_data")

# Handle crawl execution
if analyze_clicked and url_input:
    st.session_state.pop("chat_messages", None)
    st.session_state.pop("last_response", None)

    with st.status("🔍 Initializing RAG Pipeline...", expanded=True) as status:
        status.update(label="🕸️ Crawling website subpages (this may take up to 20-30s)...")
        time.sleep(0.5)

        result = call_api("/api/analyze", method="POST", payload={
            "url": url_input,
            "persona": selected_persona,
        })

        if result and result.get("status") == "completed":
            st.session_state["analysis_data"] = result
            status.update(label="✅ Website Indexed & RAG Vector Database Built!", state="complete", expanded=False)
            st.rerun()
        else:
            status.update(label="❌ Ingestion failed", state="error")
            st.error("Failed to build RAG vector index. Please check that backend server is online.")

if data:
    # Title Header
    title = data.get("title", data.get("url", ""))
    domain = data.get("domain", "")
    persona_key = data.get("persona", "student")
    p_info = PERSONAS.get(persona_key, PERSONAS["student"])

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"### 🧠 RAG Session: {title}")
        st.caption(f"Domain: `{domain}` | Source URL: [{data.get('url')}]({data.get('url')})")
    with col_h2:
        st.markdown(
            f"<div style='text-align:right; padding-top: 0.5rem;'>"
            f"<span style='background: {p_info['color']}22; color: {p_info['color']}; "
            f"padding: 0.4rem 1rem; border-radius: 20px; font-weight: 600; font-size: 0.9rem; "
            f"border: 1px solid {p_info['color']}44;'>"
            f"{p_info['label']}</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()

    # Create Tabs
    tab_chat, tab_kb, tab_intel = st.tabs([
        "💬 Grounded RAG Chatbot", 
        "🌐 Crawler & Knowledge Base Stats", 
        "📊 Website Intelligence Reports"
    ])

    # ──────────────────────────────────────────────
    # TAB 1: Chatbot (Primary Feature)
    # ──────────────────────────────────────────────
    with tab_chat:
        chat_col, side_col = st.columns([5, 3])

        with chat_col:
            st.markdown("##### Ask grounded questions about the site content:")
            
            # Session History
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            # Display messages
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🧠"):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander("📎 Cited Sources Used", expanded=False):
                            for src in msg["sources"]:
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-header">
                                        <span class="citation-title">Chunk {src.get('chunk_id')}</span>
                                        <span class="citation-score">Similarity: {src.get('score', 0.0):.2f}</span>
                                    </div>
                                    <div class="citation-url">Source: <a href="{src.get('url')}" target="_blank">{src.get('url')}</a></div>
                                    <div class="citation-text">"{src.get('chunk_text', '')}"</div>
                                </div>
                                """, unsafe_allow_html=True)

            # Chat Input
            if prompt := st.chat_input("Ask a question about the crawled website..."):
                # Save & Display User msg
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="🧠"):
                    with st.spinner("Retrieving from index & generating answer..."):
                        result = call_api("/api/chat", method="POST", payload={
                            "analysis_id": data["id"],
                            "message": prompt,
                        })

                    if result:
                        answer = result.get("answer", "Error retrieving answer.")
                        sources = result.get("sources", [])
                        debug_info = result.get("debug", {})
                        
                        st.markdown(answer)
                        if sources:
                            with st.expander("📎 Cited Sources Used", expanded=False):
                                for src in sources:
                                    st.markdown(f"""
                                    <div class="citation-card">
                                        <div class="citation-header">
                                            <span class="citation-title">Chunk {src.get('chunk_id')}</span>
                                            <span class="citation-score">Similarity: {src.get('score', 0.0):.2f}</span>
                                        </div>
                                        <div class="citation-url">Source: <a href="{src.get('url')}" target="_blank">{src.get('url')}</a></div>
                                        <div class="citation-text">"{src.get('chunk_text', '')}"</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # Save Assistant Msg & Session Debug Info
                        msg_data = {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "debug": debug_info,
                            "chunk_count": result.get("chunk_count", 0),
                            "avg_similarity": result.get("avg_similarity", 0.0),
                        }
                        st.session_state.chat_messages.append(msg_data)
                        st.session_state.last_response = msg_data
                        st.rerun()
                    else:
                        st.error("Failed to query the RAG chatbot endpoint.")

        with side_col:
            # Checkbox to enable/disable RAG Debug
            enable_debug = st.checkbox("🛠️ Enable RAG Debug Panel", value=True)
            last_resp = st.session_state.get("last_response")

            if enable_debug:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="glass-card-header">🛠️ RAG Pipeline Debugger</div>', unsafe_allow_html=True)
                
                if last_resp and last_resp.get("debug"):
                    debug_data = last_resp["debug"]
                    
                    # Debug metrics grid
                    st.markdown(f"""
                    <div class="debug-metric-grid">
                        <div class="debug-metric">
                            <div class="debug-metric-val">{last_resp.get('chunk_count', 0)}</div>
                            <div class="debug-metric-lbl">Retrieved Chunks</div>
                        </div>
                        <div class="debug-metric">
                            <div class="debug-metric-val purple">{last_resp.get('avg_similarity', 0.0):.3f}</div>
                            <div class="debug-metric-lbl">Avg Similarity</div>
                        </div>
                        <div class="debug-metric">
                            <div class="debug-metric-val orange">{debug_data.get('context_length', 0)}</div>
                            <div class="debug-metric-lbl">Context Chars</div>
                        </div>
                        <div class="debug-metric">
                            <div class="debug-metric-val">{debug_data.get('tokens_sent', 0)}</div>
                            <div class="debug-metric-lbl">Tokens Sent</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Prompt toggle
                    with st.expander("📝 View Final Prompt Sent to Gemini"):
                        st.code(debug_data.get("final_prompt", ""), language="markdown")
                    
                    # Retrieved chunks list
                    st.markdown("📂 **All Retrieved Chunks & Similarity Scores:**")
                    for c in debug_data.get("retrieved_chunks", []):
                        st.markdown(f"""
                        <div class="citation-card" style="border-color: rgba(48, 54, 61, 0.8); border-left-color: #bc8cff;">
                            <div class="citation-header">
                                <span class="citation-title" style="color: #bc8cff;">Chunk {c.get('chunk_id')}</span>
                                <span class="citation-score" style="background: rgba(188, 140, 255, 0.15); color: #bc8cff;">Score: {c.get('score', 0.0):.3f}</span>
                            </div>
                            <div class="citation-url">Url: {c.get('url')}</div>
                            <div class="citation-text">"{c.get('chunk_text', '')[:250]}..."</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Ask a question to populate RAG pipeline debug statistics.")
                st.markdown('</div>', unsafe_allow_html=True)

            # Sources Card
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="glass-card-header">📎 Sources Actually Cited</div>', unsafe_allow_html=True)
            if last_resp and last_resp.get("sources"):
                for src in last_resp["sources"]:
                    st.markdown(f"""
                    <div style="font-size: 0.85rem; padding: 0.5rem 0; border-bottom: 1px solid rgba(48, 54, 61, 0.5);">
                        🌐 <b>Chunk {src.get('chunk_id')}</b> (Score: {src.get('score', 0.0):.2f})<br/>
                        <a href="{src.get('url')}" target="_blank" style="font-size: 0.75rem; color: #58a6ff;">{src.get('url')}</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No sources cited in the last answer.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    # TAB 2: Crawler & Knowledge Base Stats
    # ──────────────────────────────────────────────
    with tab_kb:
        st.markdown("#### 🌐 Crawled Pages Ingestion & Vector Statistics")
        st.caption("We recursively follow internal sublinks, prioritizing documentation, about, and product details.")

        kb_stats = data.get("kb_stats", {})
        indexed_pages = data.get("indexed_pages", [])

        # Fallback if old analysis structure
        if not indexed_pages:
            indexed_pages = [
                {
                    "url": data.get("url"),
                    "title": data.get("title", "Home Page"),
                    "chunk_count": kb_stats.get("total_chunks", 0) if kb_stats else 0,
                    "char_count": 0
                }
            ]
            kb_stats = {
                "total_pages": 1,
                "total_chunks": kb_stats.get("total_chunks", 0) if kb_stats else 0,
                "total_chars": 0
            }

        # Render Stats cards
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Pages Crawled</div>
                <div class="metric-value" style="color: #58a6ff;">{kb_stats.get('total_pages', 0)}</div>
                <div class="metric-reason">Target crawl count completed</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Vector Chunks</div>
                <div class="metric-value" style="color: #bc8cff;">{kb_stats.get('total_chunks', 0)}</div>
                <div class="metric-reason">Splits generated from site texts</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c3:
            # Format char count as KB/MB
            chars = kb_stats.get('total_chars', 0)
            chars_str = f"{chars / 1024:.1f} KB" if chars < 1024 * 1024 else f"{chars / (1024 * 1024):.2f} MB"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Knowledge Base Size</div>
                <div class="metric-value" style="color: #3fb950;">{chars_str}</div>
                <div class="metric-reason">Total characters cleaned & indexed</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("##### 📄 Pages Indexed List")
        
        # Format indexed pages table
        pages_data = []
        for idx, page in enumerate(indexed_pages, 1):
            url_str = page.get("url") if isinstance(page, dict) else page.url
            title_str = page.get("title") if isinstance(page, dict) else page.title
            chunks_cnt = page.get("chunk_count") if isinstance(page, dict) else page.chunk_count
            chars_cnt = page.get("char_count") if isinstance(page, dict) else page.char_count
            pages_data.append({
                "Index": idx,
                "Title": title_str,
                "Page URL": url_str,
                "Chunks": chunks_cnt,
                "Chars": chars_cnt
            })
            
        st.table(pages_data)

    # ──────────────────────────────────────────────
    # TAB 3: Business Intelligence Reports (Secondary)
    # ──────────────────────────────────────────────
    with tab_intel:
        st.markdown("#### 📊 Website Intelligence Reports (Secondary Focus)")
        st.caption("Auto-generated reports from homepage overview analysis. Subject to grounding validation constraints.")

        # Overall Score
        ws = data.get("website_score", {})
        if ws:
            overall = ws.get("overall", 0)
            dimensions = ws.get("dimensions", [])
            st.markdown(f"##### ⭐ Website Persona Score: {p_info['label']}")
            
            score_col, dims_col = st.columns([1, 3])
            with score_col:
                color = score_color(overall)
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem 0;">
                    <div class="score-ring" style="{score_gradient(overall)}">
                        <span style="color: {color};">{overall:.0%}</span>
                    </div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #8b949e;">Overall Score</div>
                </div>
                """, unsafe_allow_html=True)
            with dims_col:
                if dimensions:
                    cols = st.columns(min(len(dimensions), 4))
                    for i, dim in enumerate(dimensions):
                        with cols[i % len(cols)]:
                            s = dim.get("score", 0)
                            c = score_color(s)
                            reason = dim.get("reason", "")
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{dim.get("name", "")}</div>
                                <div class="metric-value" style="color: {c};">{s:.0%}</div>
                                <div class="metric-reason">{reason[:100]}</div>
                            </div>
                            """, unsafe_allow_html=True)
            st.divider()

        # Why It Matters & Recommended Actions
        wim = data.get("why_it_matters", {})
        if wim:
            st.markdown("##### 💡 Relevance & Recommended Actions")
            explanation = wim.get("explanation", "")
            if explanation:
                st.markdown(f"> {explanation}")
            
            w_cols = st.columns(2)
            with w_cols[0]:
                takeaways = wim.get("key_takeaways", [])
                if takeaways:
                    st.markdown("**🔑 Key Takeaways**")
                    for t in takeaways:
                        st.markdown(f'<div class="takeaway-card">✦ {t}</div>', unsafe_allow_html=True)
            with w_cols[1]:
                actions = wim.get("recommended_actions", [])
                if actions:
                    st.markdown("**🎯 Action Plan Items**")
                    for idx, a in enumerate(actions, 1):
                        st.markdown(f"**{idx}.** {a}")
            st.divider()

        # Opportunity Radar
        opps = data.get("opportunities", [])
        if opps:
            st.markdown("##### 🔥 Opportunity Radar")
            for opp in opps:
                cat = opp.get("category", "other")
                icon = CATEGORY_ICONS.get(cat, "📌")
                cat_color = CATEGORY_COLORS.get(cat, "#8b949e")
                match = opp.get("match_score", 0)
                diff = opp.get("difficulty", "intermediate")
                
                st.markdown(f"""
                <div class="opp-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem;">
                        <span class="opp-title">{icon} {opp.get('title')}</span>
                        <span class="opp-badge" style="background: {cat_color}22; color: {cat_color}; border: 1px solid {cat_color}44;">
                            {cat.replace('_', ' ').title()}
                        </span>
                    </div>
                    <div style="font-size:0.85rem; color: #8b949e; margin-bottom: 0.5rem;">{opp.get('description')}</div>
                    <div style="font-size:0.8rem; color:#8b949e;">
                        Match Score: <b>{match:.0%}</b> | Difficulty: <b>{diff.title()}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.divider()

        # Skill Gap Analyzer
        sg = data.get("skill_gap", {})
        if sg:
            st.markdown("##### 🧩 Skill Gap Analyzer")
            sg_cols = st.columns(2)
            with sg_cols[0]:
                st.markdown("**✅ Required Skills**")
                for s in sg.get("required_skills", []):
                    st.markdown(f"- **{s.get('skill')}** ({s.get('level').title()}): {s.get('context')}")
            with sg_cols[1]:
                st.markdown("**⚠️ Potential Missing Skills**")
                for s in sg.get("missing_skills", []):
                    st.markdown(f"- <span style='color: #f85149;'><b>{s.get('skill')}</b></span>: {s.get('reason')}", unsafe_allow_html=True)
            
            roadmap = sg.get("learning_roadmap", [])
            if roadmap:
                st.markdown("---")
                st.markdown("**🗺️ Learning Roadmap**")
                for step in roadmap:
                    st.markdown(f"**Step {step.get('step')}: {step.get('skill')}** ({step.get('resource_type').title()} · Time: {step.get('time_estimate')})")
            st.divider()

        # Similar Websites
        sim = data.get("similar_websites", [])
        if sim:
            st.markdown("##### 🌐 Similar Website Discovery")
            sim_cols = st.columns(min(len(sim), 3))
            for i, site in enumerate(sim):
                with sim_cols[i % len(sim_cols)]:
                    st.markdown(f"""
                    <div class="glass-card" style="min-height: 180px;">
                        <div style="font-weight: 600; color: #bc8cff;">{site.get('name')}</div>
                        <div style="font-size:0.75rem; color:#58a6ff; margin-bottom:0.4rem;">{site.get('url')}</div>
                        <div style="font-size:0.8rem; color:#8b949e; line-height:1.4;">{site.get('description')}</div>
                        <div style="font-size:0.75rem; color:#8b949e; font-style:italic; margin-top:0.4rem;">↳ {site.get('why_relevant')}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Overview Summary
        summ = data.get("summary", "")
        if summ:
            st.markdown("##### 📋 Overview Summary")
            st.markdown(summ)

else:
    render_hero()
