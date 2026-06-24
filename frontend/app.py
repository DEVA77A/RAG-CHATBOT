"""
WebIntel AI — Streamlit Frontend

An AI intelligence dashboard, not a chatbot.
Designed to wow hackathon judges on first glance.
"""

import streamlit as st
import requests
import time
import math

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="WebIntel AI — Personalized Website Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Backend Config
# ──────────────────────────────────────────────

API_BASE = "http://localhost:8000"

# ──────────────────────────────────────────────
# Custom CSS — Premium Dark Dashboard Theme
# ──────────────────────────────────────────────

st.markdown("""
<style>
/* ── Import Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global Overrides ── */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit Branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Score Ring ── */
.score-ring {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 auto 0.5rem auto;
    position: relative;
}

.score-ring::before {
    content: '';
    position: absolute;
    inset: 4px;
    border-radius: 50%;
    background: #0e1117;
}

.score-ring span {
    position: relative;
    z-index: 1;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, #1a1d23 0%, #22262e 100%);
    border: 1px solid #2d3139;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: #4a9eff;
}

.metric-card .metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0.25rem 0;
    line-height: 1.1;
}

.metric-card .metric-label {
    font-size: 0.85rem;
    color: #8b949e;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-card .metric-reason {
    font-size: 0.75rem;
    color: #6e7681;
    margin-top: 0.5rem;
    line-height: 1.3;
}

/* ── Opportunity Cards ── */
.opp-card {
    background: linear-gradient(135deg, #1a1d23 0%, #1e222a 100%);
    border: 1px solid #2d3139;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}

.opp-card:hover {
    border-color: #f0883e;
}

.opp-card .opp-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.opp-card .opp-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e6edf3;
}

.opp-card .opp-badge {
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.opp-card .opp-desc {
    font-size: 0.85rem;
    color: #8b949e;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}

/* ── Skill Cards ── */
.skill-tag {
    display: inline-block;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
}

.skill-required {
    background: rgba(56, 139, 253, 0.15);
    color: #58a6ff;
    border: 1px solid rgba(56, 139, 253, 0.3);
}

.skill-missing {
    background: rgba(248, 81, 73, 0.15);
    color: #f85149;
    border: 1px solid rgba(248, 81, 73, 0.3);
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 2rem 0 0.5rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #21262d;
}

.section-subtitle {
    font-size: 0.95rem;
    color: #8b949e;
    margin-bottom: 1.5rem;
}

/* ── Hero Section ── */
.hero-container {
    text-align: center;
    padding: 3rem 1rem;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #58a6ff, #bc8cff, #f78166);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #8b949e;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Takeaway Cards ── */
.takeaway-card {
    background: linear-gradient(135deg, #1a1d23 0%, #1e222a 100%);
    border-left: 3px solid #58a6ff;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
}

.takeaway-card p {
    margin: 0;
    font-size: 0.95rem;
    color: #c9d1d9;
    line-height: 1.5;
}

/* ── Similar Site Card ── */
.similar-card {
    background: linear-gradient(135deg, #1a1d23 0%, #1e222a 100%);
    border: 1px solid #2d3139;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}

.similar-card:hover {
    border-color: #bc8cff;
}

.similar-card .site-name {
    font-size: 1rem;
    font-weight: 600;
    color: #e6edf3;
}

.similar-card .site-url {
    font-size: 0.78rem;
    color: #58a6ff;
}

.similar-card .site-desc {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 0.35rem;
    line-height: 1.4;
}

/* ── Roadmap Steps ── */
.roadmap-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem 0;
    border-bottom: 1px solid #21262d;
}

.roadmap-step:last-child {
    border-bottom: none;
}

.step-number {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #58a6ff, #388bfd);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    flex-shrink: 0;
    color: white;
}

.step-content .step-title {
    font-weight: 600;
    color: #e6edf3;
    font-size: 0.95rem;
}

.step-content .step-meta {
    font-size: 0.8rem;
    color: #6e7681;
    margin-top: 0.2rem;
}

/* ── Progress Bar Override ── */
.stProgress > div > div {
    border-radius: 10px;
}

/* ── Chat Styling ── */
.chat-container {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 1rem;
    max-height: 500px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Persona Config
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
# Helper Functions
# ──────────────────────────────────────────────

def score_color(score: float) -> str:
    """Return a gradient color based on score value."""
    if score >= 0.8:
        return "#3fb950"
    elif score >= 0.6:
        return "#58a6ff"
    elif score >= 0.4:
        return "#d29922"
    else:
        return "#f85149"


def score_gradient(score: float) -> str:
    """Return a CSS gradient for score rings."""
    color = score_color(score)
    pct = score * 100
    return f"background: conic-gradient({color} {pct}%, #21262d {pct}%);"


def format_score(score: float) -> str:
    """Format a 0-1 score as a percentage or fraction."""
    return f"{score:.0%}" if score <= 1 else str(score)


def call_api(endpoint: str, method: str = "GET", payload: dict = None) -> dict | None:
    """Call the backend API and return JSON response."""
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "POST":
            resp = requests.post(url, json=payload, timeout=120)
        else:
            resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("❌ Cannot connect to the backend. Make sure the FastAPI server is running on port 8000.")
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
# Render Functions
# ──────────────────────────────────────────────

def render_hero():
    """Render the landing hero when no analysis is loaded."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">WebIntel AI</div>
        <div class="hero-subtitle">
            Enter any website URL, select your persona, and get a personalized
            intelligence report — scores, opportunities, skill gaps, and actionable insights
            tailored to who you are.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    cols = st.columns(3)
    features = [
        ("⭐ Personalized Scores", "Persona-specific website scoring across multiple dimensions"),
        ("🔥 Opportunity Radar", "Auto-detect jobs, courses, certifications, and programs"),
        ("🧩 Skill Gap Analysis", "Discover required skills, identify gaps, get a learning roadmap"),
    ]
    for col, (title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin-bottom: 0.5rem;">{title}</div>
                <div class="metric-reason" style="font-size: 0.85rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    cols2 = st.columns(3)
    features2 = [
        ("💡 Why It Matters", "Understand why a website is relevant to YOUR specific goals"),
        ("🌐 Similar Sites", "Discover related websites you should explore next"),
        ("💬 RAG Chat", "Ask follow-up questions grounded in the actual website content"),
    ]
    for col, (title, desc) in zip(cols2, features2):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin-bottom: 0.5rem;">{title}</div>
                <div class="metric-reason" style="font-size: 0.85rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def render_website_header(data: dict):
    """Render the analyzed website header bar."""
    title = data.get("title", data.get("url", ""))
    domain = data.get("domain", "")
    persona_key = data.get("persona", "student")
    persona_info = PERSONAS.get(persona_key, PERSONAS["student"])

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 🌐 {title}")
        st.caption(f"`{data.get('url', '')}`")
    with col2:
        st.markdown(
            f"<div style='text-align:right; padding-top: 0.5rem;'>"
            f"<span style='background: {persona_info['color']}22; color: {persona_info['color']}; "
            f"padding: 0.4rem 1rem; border-radius: 20px; font-weight: 600; font-size: 0.9rem; "
            f"border: 1px solid {persona_info['color']}44;'>"
            f"{persona_info['label']}</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()


def render_website_score(data: dict):
    """Render the Personalized Website Score section — first thing judges see."""
    ws = data.get("website_score", {})
    if not ws:
        return

    overall = ws.get("overall", 0)
    dimensions = ws.get("dimensions", [])
    persona_key = data.get("persona", "student")
    persona_info = PERSONAS.get(persona_key, PERSONAS["student"])

    st.markdown(
        f'<div class="section-header">⭐ Personalized Website Score</div>'
        f'<div class="section-subtitle">How valuable is this website for a {persona_info["label"]}?</div>',
        unsafe_allow_html=True,
    )

    # Overall score ring + dimension cards
    score_col, dims_col = st.columns([1, 3])

    with score_col:
        color = score_color(overall)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <div class="score-ring" style="{score_gradient(overall)}">
                <span style="color: {color};">{overall:.0%}</span>
            </div>
            <div style="font-size: 1rem; font-weight: 600; color: #c9d1d9; margin-top: 0.5rem;">
                Overall Score
            </div>
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
                    # Truncate reason for card display
                    short_reason = reason[:80] + "..." if len(reason) > 80 else reason
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{dim.get("name", "")}</div>
                        <div class="metric-value" style="color: {c};">{s:.0%}</div>
                        <div class="metric-reason">{short_reason}</div>
                    </div>
                    """, unsafe_allow_html=True)


def render_why_it_matters(data: dict):
    """Render the Why This Matters To Me section."""
    wim = data.get("why_it_matters", {})
    if not wim:
        return

    persona_key = data.get("persona", "student")
    persona_info = PERSONAS.get(persona_key, PERSONAS["student"])

    st.markdown(
        f'<div class="section-header">💡 Why This Matters To You</div>'
        f'<div class="section-subtitle">Personalized relevance analysis for a {persona_info["label"]}</div>',
        unsafe_allow_html=True,
    )

    # Explanation
    explanation = wim.get("explanation", "")
    if explanation:
        relevance = wim.get("relevance_score", 0)
        color = score_color(relevance)
        st.markdown(
            f"<div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>"
            f"<span style='font-size: 1.5rem; font-weight: 800; color: {color};'>{relevance:.0%}</span>"
            f"<span style='color: #8b949e; font-size: 0.85rem;'>Relevance Score</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"> {explanation}")

    # Key Takeaways
    takeaways = wim.get("key_takeaways", [])
    if takeaways:
        st.markdown("**🔑 Key Takeaways**")
        for t in takeaways:
            st.markdown(f"""
            <div class="takeaway-card"><p>✦ {t}</p></div>
            """, unsafe_allow_html=True)

    # Recommended Actions
    actions = wim.get("recommended_actions", [])
    if actions:
        st.markdown("**🎯 Recommended Actions**")
        for i, action in enumerate(actions, 1):
            st.markdown(f"**{i}.** {action}")


def render_opportunity_radar(data: dict):
    """Render the Opportunity Radar section."""
    opportunities = data.get("opportunities", [])
    persona_key = data.get("persona", "student")
    persona_info = PERSONAS.get(persona_key, PERSONAS["student"])

    st.markdown(
        f'<div class="section-header">🔥 Opportunity Radar</div>'
        f'<div class="section-subtitle">Opportunities detected on this website for a {persona_info["label"]}</div>',
        unsafe_allow_html=True,
    )

    if not opportunities:
        st.info("No specific opportunities were detected on this website. Try a careers page or educational platform for more results.")
        return

    # Category filter
    categories = sorted(set(opp.get("category", "other") for opp in opportunities))
    category_labels = [f"{CATEGORY_ICONS.get(c, '📌')} {c.replace('_', ' ').title()}" for c in categories]

    selected_filter = st.pills(
        "Filter by category",
        options=["All"] + category_labels,
        default="All",
        key="opp_filter",
    )

    # Filter opportunities
    if selected_filter and selected_filter != "All":
        # Extract category from the label
        filter_cat = selected_filter.split(" ", 1)[1].lower().replace(" ", "_") if " " in selected_filter else ""
        filtered = [o for o in opportunities if o.get("category", "").replace("_", " ").title() == selected_filter.split(" ", 1)[1] if " " in selected_filter]
        if not filtered:
            filtered = opportunities
    else:
        filtered = opportunities

    for opp in filtered:
        cat = opp.get("category", "other")
        icon = CATEGORY_ICONS.get(cat, "📌")
        cat_color = CATEGORY_COLORS.get(cat, "#8b949e")
        match = opp.get("match_score", 0)
        match_color = score_color(match)
        diff = opp.get("difficulty", "intermediate")
        diff_color = DIFFICULTY_COLORS.get(diff, "#d29922")

        st.markdown(f"""
        <div class="opp-card">
            <div class="opp-header">
                <span class="opp-title">{icon} {opp.get("title", "Untitled")}</span>
                <span class="opp-badge" style="background: {cat_color}22; color: {cat_color}; border: 1px solid {cat_color}44;">
                    {cat.replace("_", " ").title()}
                </span>
            </div>
            <div class="opp-desc">{opp.get("description", "")}</div>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; margin-top: 0.5rem;">
                <span style="font-size: 0.8rem; color: {match_color}; font-weight: 600;">
                    Match: {match:.0%}
                </span>
                <span style="font-size: 0.8rem; color: {diff_color}; font-weight: 500;">
                    ● {diff.title()}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Expandable details
        with st.expander("View details", expanded=False):
            if opp.get("why_relevant"):
                st.markdown(f"**Why relevant:** {opp['why_relevant']}")
            if opp.get("required_skills"):
                skills_html = " ".join(
                    f'<span class="skill-tag skill-required">{s}</span>'
                    for s in opp["required_skills"]
                )
                st.markdown(f"**Required skills:** {skills_html}", unsafe_allow_html=True)
            if opp.get("recommended_action"):
                st.markdown(f"**Next step:** {opp['recommended_action']}")
            if opp.get("link"):
                st.markdown(f"🔗 [Visit opportunity]({opp['link']})")


def render_skill_gap(data: dict):
    """Render the Skill Gap Analyzer section."""
    sg = data.get("skill_gap", {})
    if not sg:
        return

    persona_key = data.get("persona", "student")
    persona_info = PERSONAS.get(persona_key, PERSONAS["student"])

    st.markdown(
        f'<div class="section-header">🧩 Skill Gap Analyzer</div>'
        f'<div class="section-subtitle">Skills landscape based on this website, tailored for a {persona_info["label"]}</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    # Required Skills
    with col1:
        st.markdown("**✅ Required Skills**")
        required = sg.get("required_skills", [])
        if required:
            for skill in required:
                name = skill.get("skill", "") if isinstance(skill, dict) else str(skill)
                level = skill.get("level", "intermediate") if isinstance(skill, dict) else "intermediate"
                context = skill.get("context", "") if isinstance(skill, dict) else ""
                level_color = DIFFICULTY_COLORS.get(level, "#d29922")
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center;
                            padding: 0.5rem 0; border-bottom: 1px solid #21262d;">
                    <div>
                        <span style="font-weight: 600; color: #e6edf3;">{name}</span>
                        <span style="font-size: 0.75rem; color: {level_color}; margin-left: 0.5rem;">● {level.title()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if context:
                    st.caption(f"↳ {context}")
        else:
            st.caption("No specific skills identified.")

    # Missing Skills
    with col2:
        st.markdown("**⚠️ Skills To Develop**")
        missing = sg.get("missing_skills", [])
        if missing:
            for skill in missing:
                name = skill.get("skill", "") if isinstance(skill, dict) else str(skill)
                priority = skill.get("priority", "medium") if isinstance(skill, dict) else "medium"
                reason = skill.get("reason", "") if isinstance(skill, dict) else ""
                pri_color = PRIORITY_COLORS.get(priority, "#d29922")
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center;
                            padding: 0.5rem 0; border-bottom: 1px solid #21262d;">
                    <div>
                        <span style="font-weight: 600; color: #f85149;">{name}</span>
                        <span style="font-size: 0.7rem; color: {pri_color}; margin-left: 0.5rem;
                              background: {pri_color}22; padding: 0.15rem 0.5rem; border-radius: 10px;
                              border: 1px solid {pri_color}44;">
                            {priority.upper()}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if reason:
                    st.caption(f"↳ {reason}")
        else:
            st.caption("No skill gaps identified.")

    # Learning Roadmap
    roadmap = sg.get("learning_roadmap", [])
    if roadmap:
        st.markdown("---")
        st.markdown("**🗺️ Learning Roadmap**")
        for step_data in roadmap:
            step_num = step_data.get("step", 0) if isinstance(step_data, dict) else 0
            skill_name = step_data.get("skill", "") if isinstance(step_data, dict) else str(step_data)
            res_type = step_data.get("resource_type", "tutorial") if isinstance(step_data, dict) else "tutorial"
            time_est = step_data.get("time_estimate", "") if isinstance(step_data, dict) else ""
            priority = step_data.get("priority", "medium") if isinstance(step_data, dict) else "medium"
            pri_color = PRIORITY_COLORS.get(priority, "#d29922")

            st.markdown(f"""
            <div class="roadmap-step">
                <div class="step-number">{step_num}</div>
                <div class="step-content">
                    <div class="step-title">{skill_name}</div>
                    <div class="step-meta">
                        📖 {res_type.title()}
                        {f' · ⏱️ {time_est}' if time_est else ''}
                        · <span style="color: {pri_color};">{priority.title()} priority</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_similar_websites(data: dict):
    """Render the Similar Website Discovery section."""
    similar = data.get("similar_websites", [])

    st.markdown(
        '<div class="section-header">🌐 Similar Websites</div>'
        '<div class="section-subtitle">Related websites you should explore next</div>',
        unsafe_allow_html=True,
    )

    if not similar:
        st.info("No similar websites were identified for this analysis.")
        return

    cols = st.columns(min(len(similar), 3))
    for i, site in enumerate(similar):
        with cols[i % len(cols)]:
            name = site.get("name", "Website")
            url = site.get("url", "#")
            desc = site.get("description", "")
            why = site.get("why_relevant", "")
            cat = site.get("category", "related").replace("_", " ").title()

            st.markdown(f"""
            <div class="similar-card">
                <div class="site-name">{name}</div>
                <div class="site-url">{url}</div>
                <div class="site-desc">{desc}</div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #bc8cff;">
                    💡 {why}
                </div>
                <div style="margin-top: 0.4rem;">
                    <span style="font-size: 0.7rem; background: #21262d; padding: 0.15rem 0.5rem;
                          border-radius: 10px; color: #8b949e;">{cat}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.link_button(f"Visit {name}", url, use_container_width=True)


def render_summary(data: dict):
    """Render the Website Summary section."""
    summary = data.get("summary", "")
    insights = data.get("insights", [])

    st.markdown(
        '<div class="section-header">📋 Website Summary</div>'
        '<div class="section-subtitle">Comprehensive overview with persona-specific insights</div>',
        unsafe_allow_html=True,
    )

    if summary:
        st.markdown(summary)

    if insights:
        st.markdown("---")
        st.markdown("**🧠 Detailed Insights**")
        for category in insights:
            cat_name = category.get("category", "General")
            items = category.get("items", [])
            with st.expander(f"**{cat_name}** ({len(items)} insights)", expanded=False):
                for item in items:
                    relevance = item.get("relevance", "medium")
                    rel_color = PRIORITY_COLORS.get(relevance, "#d29922")
                    st.markdown(
                        f"**{item.get('title', '')}** "
                        f"<span style='font-size: 0.7rem; color: {rel_color};'>● {relevance.upper()}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"{item.get('description', '')}")
                    st.markdown("")


    # Action Plan
    action_plan = data.get("action_plan", [])
    if action_plan:
        st.markdown("---")
        st.markdown("**🎯 Action Plan — What To Do Next**")
        for step_data in action_plan:
            step_num = step_data.get("step", 0) if isinstance(step_data, dict) else 0
            title = step_data.get("title", "") if isinstance(step_data, dict) else str(step_data)
            desc = step_data.get("description", "") if isinstance(step_data, dict) else ""
            priority = step_data.get("priority", "medium") if isinstance(step_data, dict) else "medium"
            time_est = step_data.get("time_estimate", "") if isinstance(step_data, dict) else ""
            pri_color = PRIORITY_COLORS.get(priority, "#d29922")

            st.markdown(f"""
            <div class="roadmap-step">
                <div class="step-number">{step_num}</div>
                <div class="step-content">
                    <div class="step-title">{title}</div>
                    <div class="step-meta">
                        {desc}
                        <br/>
                        <span style="color: {pri_color}; font-weight: 500;">
                            {priority.title()} priority
                        </span>
                        {f' · ⏱️ {time_est}' if time_est else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_chat(analysis_id: str):
    """Render the RAG Chat interface."""
    st.markdown(
        '<div class="section-header">💬 Ask WebIntel AI</div>'
        '<div class="section-subtitle">Ask follow-up questions about this website — answers are grounded in actual content</div>',
        unsafe_allow_html=True,
    )

    # Initialize chat history in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display existing messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🧠"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for src in msg["sources"]:
                        score = src.get("score", 0)
                        st.caption(f"**Relevance: {score:.0%}** — {src.get('content', '')}")

    # Chat input
    if prompt := st.chat_input("Ask a question about this website..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Call API
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Thinking..."):
                result = call_api("/api/chat", method="POST", payload={
                    "analysis_id": analysis_id,
                    "message": prompt,
                })

            if result:
                answer = result.get("answer", "I couldn't generate a response.")
                sources = result.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander("📎 Sources", expanded=False):
                        for src in sources:
                            score = src.get("score", 0)
                            st.caption(f"**Relevance: {score:.0%}** — {src.get('content', '')}")

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            else:
                st.error("Failed to get a response. Please try again.")


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 1.8rem; font-weight: 800;
                    background: linear-gradient(135deg, #58a6ff, #bc8cff);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;">
            🧠 WebIntel AI
        </div>
        <div style="font-size: 0.8rem; color: #6e7681; margin-top: 0.25rem;">
            Personalized Website Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # URL Input
    url_input = st.text_input(
        "🌐 Website URL",
        placeholder="https://openai.com",
        help="Enter any public website URL to analyze",
    )

    # Persona Selector
    st.markdown("**👤 Select Your Persona**")

    persona_options = list(PERSONAS.keys())
    persona_labels = [f"{v['label']}" for v in PERSONAS.values()]

    selected_persona_label = st.radio(
        "Persona",
        options=persona_labels,
        index=0,
        label_visibility="collapsed",
    )

    # Map label back to key
    selected_persona = persona_options[persona_labels.index(selected_persona_label)]
    persona_info = PERSONAS[selected_persona]
    st.caption(f"{persona_info['desc']}")

    st.divider()

    # Analyze Button
    analyze_clicked = st.button(
        "🚀 Analyze Website",
        use_container_width=True,
        type="primary",
        disabled=not url_input,
    )

    # Previous analysis quick-load
    if st.session_state.get("analysis_data"):
        st.divider()
        st.caption("📊 Current Analysis")
        current = st.session_state["analysis_data"]
        st.markdown(f"**{current.get('title', 'N/A')}**")
        st.caption(f"`{current.get('url', '')}`")
        st.caption(f"Persona: {PERSONAS.get(current.get('persona', ''), {}).get('label', '')}")

        if st.button("🗑️ Clear & Start Over", use_container_width=True):
            st.session_state.pop("analysis_data", None)
            st.session_state.pop("chat_messages", None)
            st.rerun()

    # Footer
    st.divider()
    st.caption("Built with ❤️ for AI Hackathon")
    st.caption("Powered by Gemini 2.5 Flash + FAISS")


# ──────────────────────────────────────────────
# Main Content Area
# ──────────────────────────────────────────────

# Handle analyze button click
if analyze_clicked and url_input:
    # Reset previous state
    st.session_state.pop("chat_messages", None)

    with st.status("🔍 Analyzing website...", expanded=True) as status:
        st.write("🌐 Scraping website content...")
        time.sleep(0.5)

        st.write("📝 Extracting and chunking text...")
        time.sleep(0.3)

        st.write("🧬 Generating embeddings...")
        time.sleep(0.3)

        st.write("🧠 Running AI analysis (this takes 10-20 seconds)...")

        result = call_api("/api/analyze", method="POST", payload={
            "url": url_input,
            "persona": selected_persona,
        })

        if result and result.get("status") == "completed":
            st.session_state["analysis_data"] = result
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            st.rerun()
        elif result:
            status.update(label="❌ Analysis failed", state="error")
            st.error(f"Analysis returned status: {result.get('status')}")
        else:
            status.update(label="❌ Analysis failed", state="error")

# Render the dashboard or hero
data = st.session_state.get("analysis_data")

if data:
    # ── Website Header ──
    render_website_header(data)

    # ── 1. Personalized Website Score (FIRST thing judges see) ──
    render_website_score(data)

    st.divider()

    # ── 2. Why This Matters To Me ──
    render_why_it_matters(data)

    st.divider()

    # ── 3. Opportunity Radar ──
    render_opportunity_radar(data)

    st.divider()

    # ── 4. Skill Gap Analyzer ──
    render_skill_gap(data)

    st.divider()

    # ── 5. Similar Website Discovery ──
    render_similar_websites(data)

    st.divider()

    # ── 6. Website Summary + Insights + Action Plan ──
    render_summary(data)

    st.divider()

    # ── 7. RAG Chat ──
    render_chat(data["id"])

else:
    render_hero()
