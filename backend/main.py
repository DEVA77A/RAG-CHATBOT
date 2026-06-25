"""
WebIntel AI — FastAPI Application (Pure RAG)

3 endpoints:
  POST /api/analyze    — Crawl URL → Chunk → Embed → Store in FAISS
  POST /api/chat       — Hybrid RAG chat (FAISS + BM25 + RRF)
  GET  /api/health     — Health check
"""

import json
import logging
import uuid
import os
import time
import hashlib
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    ChatSource,
    HealthResponse,
    IndexedPage,
    KBStats,
    AnalysisListResponse,
    AnalysisHistoryItem,
    ChatHistoryMessage,
    ChatHistoryResponse,
)
from database import (
    init_db,
    create_analysis,
    update_analysis,
    get_analysis,
    save_chat_message,
    get_chat_history,
    get_all_analyses,
)
from scraper import crawl_website
from chunker import chunk_text_with_metadata, embed_texts, embed_query
from vector_store import FAISSStore
from retriever import simple_search
from ai_engine import rag_chat

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

llm_health_status = {"claude": "Unavailable", "gemini": "Unavailable"}

def health_check_llms():
    global llm_health_status
    logger.info("Running silent health check on LLM providers...")
    
    # Check Gemini Waterfall
    from ai_engine import get_chat_model
    gemini_models = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-1.0-pro"]
    for m_name in gemini_models:
        try:
            model = get_chat_model(m_name)
            response = model.generate_content("say hi")
            if response and response.text:
                llm_health_status["gemini"] = f"Connected ({m_name})"
                logger.info(f"Gemini health check: SUCCESS on {m_name}")
                break
        except Exception as e:
            logger.warning(f"Gemini health check failed on {m_name}: {e}")
    if "Connected" not in llm_health_status["gemini"]:
        llm_health_status["gemini"] = "Unavailable"

    # Check Claude
    try:
        import os
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "say hi"}]
            )
            if message and message.content:
                llm_health_status["claude"] = "Connected"
                logger.info("Claude health check: SUCCESS")
        else:
            logger.error("Claude health check failed: Missing ANTHROPIC_API_KEY")
    except Exception as e:
        llm_health_status["claude"] = "Unavailable"
        logger.error(f"Claude health check failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("Starting WebIntel AI v2.0 (Pure RAG)...")
    await init_db()
    logger.info("Database initialized")
    health_check_llms()
    yield
    logger.info("Shutting down WebIntel AI")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="WebIntel AI",
    description="Production-grade RAG Chatbot — Crawl, Index, Chat",
    version="2.0.0",
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
# POST /api/analyze — Crawl + Index (No AI Analysis)
# ──────────────────────────────────────────────

@app.get("/api/health_llm")
async def health_llm_endpoint():
    return llm_health_status


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest):
    """
    Crawl a website and build a FAISS + BM25 searchable knowledge base.
    
    Pipeline: Crawl → Section-Aware Chunk → Embed → Store → Return stats.
    No mega-prompt. No AI analysis. Pure indexing.
    """
    t_start = time.perf_counter()
    url = str(request.url).strip()
    max_pages = min(request.max_pages, 20)  # Cap at 20

    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
        parsed = urlparse(url)
    if not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL format")

    domain = parsed.netloc
    analysis_id = str(uuid.uuid4())

    logger.info(f"[{analysis_id}] Starting crawl: {url}, max_pages: {max_pages}")

    # ── Step 1: Crawl ──
    t_crawl_start = time.perf_counter()
    try:
        crawled_pages = await crawl_website(url, max_pages=max_pages)
    except Exception as e:
        logger.error(f"[{analysis_id}] Crawl failed: {e}")
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")

    if not crawled_pages:
        raise HTTPException(status_code=422, detail="No pages could be crawled from this URL")

    t_crawl_end = time.perf_counter()
    crawl_duration = t_crawl_end - t_crawl_start

    homepage = crawled_pages[0]
    title = homepage.get("title", domain)
    total_pages = len(crawled_pages)
    total_chars = sum(len(p["content"]) for p in crawled_pages)

    # ── Step 2: Section-Aware Chunk + Embed ──
    t_index_start = time.perf_counter()
    all_chunk_texts = []
    all_metadata = []
    indexed_pages_list = []
    global_chunk_idx = 0

    for page in crawled_pages:
        page_chunks = chunk_text_with_metadata(
            text=page["content"],
            page_title=page.get("title", ""),
            page_url=page.get("url", ""),
        )
        if not page_chunks:
            continue

        indexed_pages_list.append(
            IndexedPage(
                url=page["url"],
                title=page.get("title", ""),
                chunk_count=len(page_chunks),
                char_count=len(page["content"]),
            )
        )

        for chunk in page_chunks:
            all_chunk_texts.append(chunk["text"])
            all_metadata.append({
                "chunk_id": global_chunk_idx,
                "source_url": page["url"],
                "source_title": page.get("title", ""),
                "heading": chunk.get("heading", ""),
                "section_type": chunk.get("section_type", "body"),
            })
            global_chunk_idx += 1

    if not all_chunk_texts:
        raise HTTPException(status_code=422, detail="No content could be extracted from crawled pages")

    # Generate embeddings in one batch
    embeddings = embed_texts(all_chunk_texts)

    # Store in FAISS
    store = FAISSStore(analysis_id=analysis_id)
    store.add(chunks=all_chunk_texts, embeddings=embeddings, metadata=all_metadata)
    store.save()

    t_index_end = time.perf_counter()
    index_duration = t_index_end - t_index_start
    total_duration = time.perf_counter() - t_start

    # Persist to DB
    content_hash = hashlib.md5(homepage["content"].encode()).hexdigest()
    kb_stats = KBStats(
        total_pages=total_pages,
        total_chunks=len(all_chunk_texts),
        total_chars=total_chars,
    )
    scraped_data = {
        "pages": [p.model_dump() for p in indexed_pages_list],
        "stats": kb_stats.model_dump(),
    }

    await create_analysis(analysis_id, url, "rag", domain)
    await update_analysis(
        analysis_id,
        status="completed",
        title=title,
        summary="",
        scraped_content=json.dumps(scraped_data),
        content_hash=content_hash,
        crawl_time=crawl_duration,
        index_time=index_duration,
        total_time=total_duration,
    )

    logger.info(
        f"[{analysis_id}] Indexed {total_pages} pages, "
        f"{len(all_chunk_texts)} chunks in {total_duration:.2f}s "
        f"(crawl: {crawl_duration:.2f}s, index: {index_duration:.2f}s)"
    )

    return AnalyzeResponse(
        id=analysis_id,
        url=url,
        status="completed",
        title=title,
        domain=domain,
        indexed_pages=indexed_pages_list,
        kb_stats=kb_stats,
        crawl_time=crawl_duration,
        index_time=index_duration,
        total_time=total_duration,
    )


# ──────────────────────────────────────────────
# POST /api/chat — Hybrid RAG Chat
# ──────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    RAG-powered chat using hybrid retrieval (FAISS + BM25 + RRF).
    """
    t_start = time.perf_counter()
    analysis_id = request.analysis_id
    question = request.message.strip()
    top_k = request.top_k
    if "detail" in question.lower():
        top_k = max(top_k, 8)

    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Fetch analysis record
    record = await get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if record["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not yet completed")

    # Load FAISS index
    if not FAISSStore.exists(analysis_id):
        raise HTTPException(status_code=404, detail="Vector index not found")

    store = FAISSStore.load(analysis_id)

    # Simple FAISS search
    t_search_start = time.perf_counter()
    search_result = simple_search(
        query=question,
        faiss_store=store,
        embed_fn=embed_query,
        top_k=top_k,
    )
    retrieval_duration = time.perf_counter() - t_search_start

    context_chunks = search_result["chunks"]

    # Get chat history
    history = await get_chat_history(analysis_id, limit=10)

    # Save user message
    user_msg_id = str(uuid.uuid4())
    await save_chat_message(user_msg_id, analysis_id, "user", question)

    # Generate answer
    t_gen_start = time.perf_counter()
    result = await rag_chat(
        question=question,
        context_chunks=context_chunks,
        title=record.get("title", ""),
        url=record["url"],
        chat_history=history,
        question_embedding=search_result["debug"].get("query_emb", [])
    )
    t_gen_end = time.perf_counter()
    gen_duration = t_gen_end - t_gen_start
    total_duration = time.perf_counter() - t_start

    # Save assistant message
    assistant_msg_id = str(uuid.uuid4())
    await save_chat_message(
        assistant_msg_id, analysis_id, "assistant",
        result["answer"],
        sources=json.dumps(result.get("sources", [])),
    )

    # Inject timing into debug
    debug_info = result.get("debug", {})
    debug_info["retrieval_time"] = retrieval_duration
    debug_info["generation_time"] = gen_duration
    debug_info["total_time"] = total_duration
    debug_info["expanded_queries"] = search_result.get("expanded_queries", [])
    debug_info["dense_hits"] = search_result.get("dense_count", 0)
    debug_info["sparse_hits"] = search_result.get("sparse_count", 0)

    return ChatResponse(
        answer=result["answer"],
        sources=[
            ChatSource(
                chunk_id=s["chunk_id"],
                source_url=s["source_url"],
                source_title=s["source_title"],
                content=s["content"],
                chunk_text=s["chunk_text"],
                score=s["score"],
            )
            for s in result.get("sources", [])
        ],
        chunk_count=result.get("chunk_count", 0),
        avg_similarity=result.get("avg_similarity", 0.0),
        debug=debug_info,
        retrieval_time=retrieval_duration,
        generation_time=gen_duration,
        total_time=total_duration,
    )


# ──────────────────────────────────────────────
# GET /api/analyses — List all analyses
# ──────────────────────────────────────────────

@app.get("/api/analyses", response_model=AnalysisListResponse)
async def list_analyses():
    """List all previously crawled and analyzed websites."""
    analyses = await get_all_analyses()
    clean_analyses = []
    for a in analyses:
        clean_a = {k: (v if v is not None else "") for k, v in a.items()}
        clean_analyses.append(AnalysisHistoryItem(**clean_a))
    return AnalysisListResponse(analyses=clean_analyses)


# ──────────────────────────────────────────────
# GET /api/analyze/{analysis_id} — Fetch analysis details
# ──────────────────────────────────────────────

@app.get("/api/analyze/{analysis_id}", response_model=AnalyzeResponse)
async def get_analyze_endpoint(analysis_id: str):
    """Fetch details of an existing analysis."""
    data = await get_analysis(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    scraped = json.loads(data.get("scraped_content", "{}"))
    return AnalyzeResponse(
        id=data["id"],
        url=data["url"],
        status=data["status"],
        title=data.get("title", ""),
        domain=data.get("domain", ""),
        indexed_pages=scraped.get("pages", []),
        kb_stats=scraped.get("stats", {"total_pages": 0, "total_chunks": 0, "total_chars": 0}),
        crawl_time=data.get("crawl_time", 0.0),
        index_time=data.get("index_time", 0.0),
        total_time=data.get("total_time", 0.0),
    )


# ──────────────────────────────────────────────
# GET /api/chat/{analysis_id} — Fetch chat history
# ──────────────────────────────────────────────

@app.get("/api/chat/{analysis_id}", response_model=ChatHistoryResponse)
async def get_chat_history_endpoint(analysis_id: str):
    """Fetch previous chat messages for an analysis."""
    history = await get_chat_history(analysis_id, limit=50)
    messages = []
    for h in history:
        raw_sources = h.get("sources")
        parsed_sources = []
        if raw_sources:
            try:
                parsed_sources = json.loads(raw_sources)
            except Exception:
                pass
                
        messages.append(ChatHistoryMessage(
            role=h["role"],
            content=h["content"],
            sources=parsed_sources,
            created_at=h["created_at"] if h["created_at"] else "",
        ))
    return ChatHistoryResponse(messages=messages)


# ──────────────────────────────────────────────
# GET /api/health
# ──────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="2.0.0")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")


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
