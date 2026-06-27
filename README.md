---
title: RAG-X
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# WebIntel AI — Personalized Website Intelligence Assistant

<div align="center">

🧠 **WebIntel AI** transforms any website into personalized, actionable intelligence.

*Enter a URL. Pick your persona. Get a tailored intelligence report in seconds.*

</div>

---

## What It Does

WebIntel AI analyzes any website through the lens of **who you are**:

| Persona | What You Get |
|---------|-------------|
| 🎓 **Student** | Learning value, project ideas, research directions |
| 💼 **Job Seeker** | Hiring potential, skill requirements, career growth |
| 👨‍💻 **Developer** | API value, documentation quality, technical depth |
| 🔬 **Researcher** | Innovation score, research areas, emerging tech |
| 📈 **Investor** | Business potential, market strength, growth signals |

### 9 AI-Powered Features (from a single analysis)

1. **⭐ Personalized Website Score** — Multi-dimensional scoring calibrated to your persona
2. **💡 Why This Matters To Me** — Personalized relevance with key takeaways
3. **🔥 Opportunity Radar** — Auto-detected jobs, internships, courses, certifications
4. **🧩 Skill Gap Analyzer** — Required skills, missing skills, learning roadmap
5. **🌐 Similar Website Discovery** — Related sites you should explore next
6. **📋 Website Summary** — Concise overview tailored to your perspective
7. **🧠 Persona Insights** — Categorized, actionable intelligence
8. **🎯 Action Plan** — Step-by-step next actions with priorities
9. **💬 RAG Chat** — Ask follow-up questions grounded in actual website content

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | **Streamlit** |
| Backend | **FastAPI** |
| AI/LLM | **Gemini 2.5 Flash** |
| Embeddings | **sentence-transformers** (all-MiniLM-L6-v2) |
| Vector DB | **FAISS** |
| Database | **SQLite** |

**All 8 analysis outputs generated from a single Gemini API call.**

---

## Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd webintel-ai
```

### 2. Setup Environment

```bash
# Create .env file
cp .env.example .env
# Edit .env and add your Gemini API key
# Get one at: https://aistudio.google.com/apikey
```

### 3. Install Backend

```bash
cd backend
pip install -r requirements.txt
```

### 4. Install Frontend

```bash
cd frontend
pip install -r requirements.txt
```

### 5. Run

**Terminal 1 — Backend:**
```bash
cd backend
python main.py
```

**Terminal 2 — Frontend:**
```bash
cd frontend
streamlit run app.py
```

### 6. Open

Navigate to `http://localhost:8501` in your browser.

---

## Architecture

```
URL → Scrape (Trafilatura + BS4) → Chunk (500 chars) → Embed (MiniLM)
    → Store (FAISS) → Mega-Prompt (1 Gemini call → 8 outputs) → SQLite → Dashboard
```

```
d:\RAG\
├── backend/
│   ├── main.py           # FastAPI (4 endpoints)
│   ├── scraper.py         # Website scraping
│   ├── chunker.py         # Text chunking + embeddings
│   ├── vector_store.py    # FAISS wrapper
│   ├── ai_engine.py       # Gemini integration
│   ├── prompts.py         # Prompt templates
│   ├── database.py        # SQLite layer
│   ├── models.py          # Pydantic schemas
│   └── requirements.txt
├── frontend/
│   ├── app.py             # Streamlit dashboard
│   └── requirements.txt
└── data/                  # Auto-created at runtime
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Analyze a website for a persona |
| `GET` | `/api/analyze/{id}` | Fetch stored analysis |
| `POST` | `/api/chat` | RAG chat about the website |
| `GET` | `/api/health` | Health check |

---

## License

MIT

---

<div align="center">
Built with ❤️ for AI Hackathon | Powered by Gemini 2.5 Flash
</div>
