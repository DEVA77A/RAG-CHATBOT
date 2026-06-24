"""
WebIntel AI — SQLite Database Layer

Two tables:
  - analyses: Stores URL analysis results (all 8 AI outputs as JSON columns)
  - chat_messages: Stores RAG chat conversation history

Uses aiosqlite for async operations with FastAPI.
"""

import aiosqlite
import os
from datetime import datetime, timezone

DATABASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATABASE_PATH = os.path.join(DATABASE_DIR, "webintel.db")

# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    persona TEXT NOT NULL,
    title TEXT,
    domain TEXT,
    status TEXT DEFAULT 'processing',
    summary TEXT,
    insights TEXT,
    action_plan TEXT,
    opportunities TEXT,
    skill_gap TEXT,
    website_score TEXT,
    why_it_matters TEXT,
    similar_websites TEXT,
    scraped_content TEXT,
    content_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);
"""

# ──────────────────────────────────────────────
# Init
# ──────────────────────────────────────────────

async def init_db():
    """Create the data directory and initialize tables."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.executescript(SCHEMA_SQL)
        await db.commit()


# ──────────────────────────────────────────────
# Analysis CRUD
# ──────────────────────────────────────────────

async def create_analysis(analysis_id: str, url: str, persona: str, domain: str):
    """Insert a new analysis row in 'processing' status."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """INSERT INTO analyses (id, url, persona, domain, status, created_at)
               VALUES (?, ?, ?, ?, 'processing', ?)""",
            (analysis_id, url, persona, domain, datetime.now(timezone.utc).isoformat())
        )
        await db.commit()


async def update_analysis(analysis_id: str, **kwargs):
    """
    Update any columns on an analysis row.

    Usage:
        await update_analysis("abc123", status="completed", summary="...", insights='[...]')
    """
    if not kwargs:
        return

    set_clause = ", ".join(f"{key} = ?" for key in kwargs.keys())
    values = list(kwargs.values()) + [analysis_id]

    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            f"UPDATE analyses SET {set_clause} WHERE id = ?",
            values
        )
        await db.commit()


async def get_analysis(analysis_id: str) -> dict | None:
    """Fetch a single analysis by ID. Returns dict or None."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def find_cached_analysis(content_hash: str, persona: str) -> dict | None:
    """Check if we already analyzed this content for this persona."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT * FROM analyses
               WHERE content_hash = ? AND persona = ? AND status = 'completed'
               ORDER BY created_at DESC LIMIT 1""",
            (content_hash, persona)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


# ──────────────────────────────────────────────
# Chat CRUD
# ──────────────────────────────────────────────

async def save_chat_message(
    message_id: str,
    analysis_id: str,
    role: str,
    content: str,
    sources: str | None = None
):
    """Save a chat message (user or assistant)."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """INSERT INTO chat_messages (id, analysis_id, role, content, sources, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (message_id, analysis_id, role, content, sources,
             datetime.now(timezone.utc).isoformat())
        )
        await db.commit()


async def get_chat_history(analysis_id: str, limit: int = 20) -> list[dict]:
    """Fetch recent chat messages for an analysis, ordered chronologically."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT * FROM chat_messages
               WHERE analysis_id = ?
               ORDER BY created_at ASC
               LIMIT ?""",
            (analysis_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
