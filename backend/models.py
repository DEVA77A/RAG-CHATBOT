"""
WebIntel AI — Pydantic Request/Response Models (Pure RAG)

Simplified: No persona analysis models. No dashboard intelligence.
Only crawl/index and chat models.
"""

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="Website URL to crawl and index")
    max_pages: int = Field(10, description="Max pages to crawl (default 10, max 20)")


class ChatRequest(BaseModel):
    analysis_id: str = Field(..., description="Analysis ID to chat about")
    message: str = Field(..., description="User's question")
    top_k: int = Field(5, description="Number of context chunks to retrieve (default 5)")


# ──────────────────────────────────────────────
# Response Sub-Models
# ──────────────────────────────────────────────

class IndexedPage(BaseModel):
    url: str
    title: str
    chunk_count: int
    char_count: int


class KBStats(BaseModel):
    total_pages: int
    total_chunks: int
    total_chars: int


class ChatSource(BaseModel):
    chunk_id: int
    source_url: str
    source_title: str
    content: str
    chunk_text: str
    score: float


# ──────────────────────────────────────────────
# API Response Models
# ──────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    id: str
    url: str
    status: str
    title: str = ""
    domain: str = ""
    indexed_pages: list[IndexedPage] = []
    kb_stats: KBStats = KBStats(total_pages=0, total_chunks=0, total_chars=0)
    crawl_time: float = 0.0
    index_time: float = 0.0
    total_time: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource] = []
    chunk_count: int = 0
    avg_similarity: float = 0.0
    debug: dict | None = None
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"


class AnalysisHistoryItem(BaseModel):
    id: str
    url: str
    status: str
    title: str = ""
    domain: str = ""
    created_at: str = ""


class AnalysisListResponse(BaseModel):
    analyses: list[AnalysisHistoryItem] = []


class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    sources: list[dict] | None = None
    created_at: str = ""


class ChatHistoryResponse(BaseModel):
    messages: list[ChatHistoryMessage] = []
