"""
WebIntel AI — Pydantic Request/Response Models

Defines all API schemas. Kept flat and simple — no nested inheritance trees.
"""

from pydantic import BaseModel, Field, HttpUrl
from enum import Enum


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class Persona(str, Enum):
    STUDENT = "student"
    JOB_SEEKER = "job_seeker"
    DEVELOPER = "developer"
    RESEARCHER = "researcher"
    INVESTOR = "investor"


class AnalysisStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ──────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="Website URL to analyze")
    persona: Persona = Field(..., description="User persona for personalized analysis")


class ChatRequest(BaseModel):
    analysis_id: str = Field(..., description="Analysis ID to chat about")
    message: str = Field(..., description="User's question")
    debug: bool = Field(False, description="Enable debug mode to return RAG pipeline metrics")


# ──────────────────────────────────────────────
# Response Sub-Models (for structured Gemini output)
# ──────────────────────────────────────────────

class InsightItem(BaseModel):
    title: str
    description: str
    relevance: str = "medium"


class InsightCategory(BaseModel):
    category: str
    items: list[InsightItem]


class ActionStep(BaseModel):
    step: int
    title: str
    description: str
    priority: str = "medium"
    time_estimate: str = ""


class Opportunity(BaseModel):
    category: str
    title: str
    description: str
    link: str = ""
    match_score: float = 0.0
    why_relevant: str = ""
    required_skills: list[str] = []
    difficulty: str = "intermediate"
    recommended_action: str = ""


class RequiredSkill(BaseModel):
    skill: str
    level: str = "intermediate"
    context: str = ""


class MissingSkill(BaseModel):
    skill: str
    priority: str = "medium"
    reason: str = ""


class LearningStep(BaseModel):
    step: int
    skill: str
    resource_type: str = "tutorial"
    time_estimate: str = ""
    priority: str = "medium"


class SkillGap(BaseModel):
    required_skills: list[RequiredSkill] = []
    missing_skills: list[MissingSkill] = []
    learning_roadmap: list[LearningStep] = []


class ScoreDimension(BaseModel):
    name: str
    score: float = 0.0
    reason: str = ""


class WebsiteScore(BaseModel):
    overall: float = 0.0
    dimensions: list[ScoreDimension] = []


class WhyItMatters(BaseModel):
    relevance_score: float = 0.0
    explanation: str = ""
    key_takeaways: list[str] = []
    recommended_actions: list[str] = []


class SimilarWebsite(BaseModel):
    name: str
    url: str
    description: str = ""
    why_relevant: str = ""
    category: str = ""


# ──────────────────────────────────────────────
# Full Mega-Prompt Output Model
# ──────────────────────────────────────────────

class MegaPromptOutput(BaseModel):
    """Schema for the single Gemini mega-prompt JSON response."""
    summary: str = ""
    insights: list[InsightCategory] = []
    action_plan: list[ActionStep] = []
    opportunities: list[Opportunity] = []
    skill_gap: SkillGap = SkillGap()
    website_score: WebsiteScore = WebsiteScore()
    why_it_matters: WhyItMatters = WhyItMatters()
    similar_websites: list[SimilarWebsite] = []


# ──────────────────────────────────────────────
# API Response Models
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


class AnalyzeResponse(BaseModel):
    id: str
    url: str
    persona: str
    status: str
    title: str = ""
    domain: str = ""
    summary: str = ""
    insights: list[InsightCategory] = []
    action_plan: list[ActionStep] = []
    opportunities: list[Opportunity] = []
    skill_gap: SkillGap = SkillGap()
    website_score: WebsiteScore = WebsiteScore()
    why_it_matters: WhyItMatters = WhyItMatters()
    similar_websites: list[SimilarWebsite] = []
    indexed_pages: list[IndexedPage] = []
    kb_stats: KBStats = KBStats(total_pages=0, total_chunks=0, total_chars=0)


class ChatSource(BaseModel):
    chunk_id: int
    url: str
    content: str
    chunk_text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource] = []
    chunk_count: int = 0
    avg_similarity: float = 0.0
    debug: dict | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
