"""
WebIntel AI — FastAPI Application

4 endpoints:
  POST /api/analyze     — Scrape URL + run mega-prompt → full analysis
  GET  /api/analyze/{id} — Fetch stored analysis by ID
  POST /api/chat        — RAG chat with FAISS context
  GET  /api/health      — Health check
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    ChatSource,
    HealthResponse,
    MegaPromptOutput,
    Persona,
    IndexedPage,
    KBStats,
)
from database import (
    init_db,
    create_analysis,
    update_analysis,
    get_analysis,
    find_cached_analysis,
    save_chat_message,
    get_chat_history,
)
from scraper import scrape_url, crawl_website
from chunker import chunk_text, embed_texts, embed_query
from vector_store import FAISSStore
from ai_engine import analyze_website, rag_chat

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webintel")


# ──────────────────────────────────────────────
# App Lifecycle
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("Starting WebIntel AI...")
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down WebIntel AI")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="WebIntel AI",
    description="Personalized Website Intelligence Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# POST /api/analyze
# ──────────────────────────────────────────────

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Analyze a website for a specific persona by recursively crawling relevant internal pages.

    Pipeline: Crawl → Chunk → Embed → Store in FAISS → Mega-prompt → Return.
    """
    import hashlib
    url = str(request.url).strip()
    persona = request.persona.value

    # Validate URL format
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
        parsed = urlparse(url)
    if not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL format")

    domain = parsed.netloc
    analysis_id = str(uuid.uuid4())

    logger.info(f"[{analysis_id}] Starting crawl-based analysis: {url} (persona: {persona})")

    # ── Step 1: Recursively crawl the website ──
    crawled_pages = await crawl_website(url, max_pages=15)

    if not crawled_pages:
        raise HTTPException(
            status_code=422,
            detail="Failed to crawl and extract content from the website.",
        )

    homepage = crawled_pages[0]
    title = homepage.get("title", domain)
    links = [p["url"] for p in crawled_pages]

    total_pages = len(crawled_pages)
    total_chars = sum(len(p["content"]) for p in crawled_pages)

    # ── Step 2: Chunk and embed all pages ──
    all_chunks = []
    all_metadata = []
    indexed_pages_list = []
    global_chunk_idx = 0

    for page in crawled_pages:
        page_chunks = chunk_text(page["content"])
        if not page_chunks:
            continue
        
        indexed_pages_list.append(
            IndexedPage(
                url=page["url"],
                title=page["title"],
                chunk_count=len(page_chunks),
                char_count=len(page["content"])
            )
        )
        
        for chunk in page_chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "chunk_id": global_chunk_idx,
                "url": page["url"],
                "title": page["title"]
            })
            global_chunk_idx += 1

    if not all_chunks:
        raise HTTPException(status_code=422, detail="No meaningful content could be extracted from any pages")

    # Generate embeddings
    embeddings = embed_texts(all_chunks)

    # ── Step 3: Store in FAISS ──
    store = FAISSStore(analysis_id=analysis_id)
    store.add(chunks=all_chunks, embeddings=embeddings, metadata=all_metadata)
    store.save()

    content_hash = hashlib.md5(homepage["content"].encode()).hexdigest() if "content" in homepage else str(uuid.uuid4())

    # ── Step 4: Check cache ──
    cached = await find_cached_analysis(content_hash, persona)
    if cached:
        logger.info(f"[{analysis_id}] Cache hit for {url} (persona: {persona})")
        return _build_response_from_db(cached)

    # ── Step 5: Create DB record ──
    await create_analysis(analysis_id, url, persona, domain)

    # ── Step 6: Run mega-prompt ──
    try:
        ai_result: MegaPromptOutput = await analyze_website(
            content=homepage["content"],
            persona=persona,
            title=title,
            url=url,
            links=links,
        )
    except Exception as e:
        logger.error(f"[{analysis_id}] AI analysis failed: {e}")
        await update_analysis(analysis_id, status="failed")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

    # ── Step 7: Persist results ──
    kb_stats = KBStats(
        total_pages=total_pages,
        total_chunks=len(all_chunks),
        total_chars=total_chars
    )
    
    scraped_data_store = {
        "pages": [p.model_dump() for p in indexed_pages_list],
        "stats": kb_stats.model_dump()
    }

    await update_analysis(
        analysis_id,
        status="completed",
        title=title,
        summary=ai_result.summary,
        insights=json.dumps([cat.model_dump() for cat in ai_result.insights]),
        action_plan=json.dumps([step.model_dump() for step in ai_result.action_plan]),
        opportunities=json.dumps([opp.model_dump() for opp in ai_result.opportunities]),
        skill_gap=json.dumps(ai_result.skill_gap.model_dump()),
        website_score=json.dumps(ai_result.website_score.model_dump()),
        why_it_matters=json.dumps(ai_result.why_it_matters.model_dump()),
        similar_websites=json.dumps([sw.model_dump() for sw in ai_result.similar_websites]),
        scraped_content=json.dumps(scraped_data_store),
        content_hash=content_hash,
    )

    logger.info(f"[{analysis_id}] Analysis completed for {url}")

    return AnalyzeResponse(
        id=analysis_id,
        url=url,
        persona=persona,
        status="completed",
        title=title,
        domain=domain,
        summary=ai_result.summary,
        insights=ai_result.insights,
        action_plan=ai_result.action_plan,
        opportunities=ai_result.opportunities,
        skill_gap=ai_result.skill_gap,
        website_score=ai_result.website_score,
        why_it_matters=ai_result.why_it_matters,
        similar_websites=ai_result.similar_websites,
        indexed_pages=indexed_pages_list,
        kb_stats=kb_stats
    )


# ──────────────────────────────────────────────
# GET /api/analyze/{id}
# ──────────────────────────────────────────────

@app.get("/api/analyze/{analysis_id}", response_model=AnalyzeResponse)
async def get_analysis_endpoint(analysis_id: str):
    """Fetch a previously stored analysis by its ID."""
    record = await get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return _build_response_from_db(record)


# ──────────────────────────────────────────────
# POST /api/chat
# ──────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    RAG-powered chat about an analyzed website.

    Retrieves relevant chunks from FAISS, passes them as context to Gemini.
    """
    analysis_id = request.analysis_id
    question = request.message.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Fetch the analysis record
    record = await get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if record["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not yet completed")

    # Load FAISS index
    if not FAISSStore.exists(analysis_id):
        raise HTTPException(status_code=404, detail="Vector index not found for this analysis")

    store = FAISSStore.load(analysis_id)

    # Embed the query and search
    query_embedding = embed_query(question)
    context_chunks = store.search(query_embedding, k=5)

    # Get chat history
    history = await get_chat_history(analysis_id, limit=10)

    # Save user message
    user_msg_id = str(uuid.uuid4())
    await save_chat_message(user_msg_id, analysis_id, "user", question)

    # Generate answer with Gemini
    result = await rag_chat(
        question=question,
        context_chunks=context_chunks,
        persona=record["persona"],
        title=record.get("title", ""),
        url=record["url"],
        chat_history=history,
    )

    # Save assistant message
    assistant_msg_id = str(uuid.uuid4())
    await save_chat_message(
        assistant_msg_id,
        analysis_id,
        "assistant",
        result["answer"],
        sources=json.dumps(result.get("sources", [])),
    )

    return ChatResponse(
        answer=result["answer"],
        sources=[
            ChatSource(
                chunk_id=s["chunk_id"],
                url=s["url"],
                content=s["content"],
                chunk_text=s["chunk_text"],
                score=s["score"]
            )
            for s in result.get("sources", [])
        ],
        chunk_count=result.get("chunk_count", 0),
        avg_similarity=result.get("avg_similarity", 0.0),
        debug=result.get("debug")
    )


# ──────────────────────────────────────────────
# GET /api/health
# ──────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _build_response_from_db(record: dict) -> AnalyzeResponse:
    """Convert a database row dict into an AnalyzeResponse."""
    scraped_content_val = record.get("scraped_content", "")
    indexed_pages = []
    kb_stats = KBStats(total_pages=0, total_chunks=0, total_chars=0)
    
    if scraped_content_val:
        try:
            parsed_content = json.loads(scraped_content_val)
            if isinstance(parsed_content, dict) and "pages" in parsed_content:
                indexed_pages = [
                    IndexedPage(
                        url=p["url"],
                        title=p["title"],
                        chunk_count=p["chunk_count"],
                        char_count=p["char_count"]
                    )
                    for p in parsed_content.get("pages", [])
                ]
                stats_dict = parsed_content.get("stats", {})
                kb_stats = KBStats(
                    total_pages=stats_dict.get("total_pages", 0),
                    total_chunks=stats_dict.get("total_chunks", 0),
                    total_chars=stats_dict.get("total_chars", 0)
                )
        except Exception:
            pass

    return AnalyzeResponse(
        id=record["id"],
        url=record["url"],
        persona=record["persona"],
        status=record["status"],
        title=record.get("title", ""),
        domain=record.get("domain", ""),
        summary=record.get("summary", ""),
        insights=_safe_json_parse(record.get("insights"), []),
        action_plan=_safe_json_parse(record.get("action_plan"), []),
        opportunities=_safe_json_parse(record.get("opportunities"), []),
        skill_gap=_safe_json_parse(record.get("skill_gap"), {}),
        website_score=_safe_json_parse(record.get("website_score"), {}),
        why_it_matters=_safe_json_parse(record.get("why_it_matters"), {}),
        similar_websites=_safe_json_parse(record.get("similar_websites"), []),
        indexed_pages=indexed_pages,
        kb_stats=kb_stats
    )


def _safe_json_parse(value: str | None, default: any) -> any:
    """Safely parse a JSON string, returning a default on failure."""
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )
