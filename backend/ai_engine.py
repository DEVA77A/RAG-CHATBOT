"""
WebIntel AI — AI Engine

Two core functions:
  1. analyze_website()  — Runs the mega-prompt against Gemini 2.5 Flash
  2. rag_chat()          — RAG-powered Q&A using FAISS context + Gemini

All AI logic is contained here. No external orchestration needed.
"""

import json
import logging
import os
import re
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

from models import MegaPromptOutput, WhyItMatters, SkillGap, WebsiteScore
from prompts import SYSTEM_PROMPT, build_mega_prompt, build_chat_prompt

load_dotenv()
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Gemini Configuration
# ──────────────────────────────────────────────

_model = None


def get_model() -> genai.GenerativeModel:
    """Lazy-initialize the Gemini model."""
    global _model
    if _model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )
        logger.info("Gemini 2.5 Flash model initialized")
    return _model


# ──────────────────────────────────────────────
# Mega Analysis
# ──────────────────────────────────────────────

async def analyze_website(
    content: str,
    persona: str,
    title: str,
    url: str,
    links: list[str] | None = None,
) -> MegaPromptOutput:
    """
    Run the mega-prompt to generate all 8 analysis outputs in one call.

    Args:
        content: Cleaned website text content.
        persona: User persona (e.g. "student", "developer").
        title: Website title.
        url: Website URL.
        links: Optional list of links found on the website.

    Returns:
        MegaPromptOutput with all 8 analysis sections populated.
    """
    model = get_model()

    # Truncate content to fit within context window
    # Gemini 2.5 Flash has 1M context, but we want fast responses
    max_content = 30000
    truncated_content = content[:max_content]

    prompt = build_mega_prompt(
        content=truncated_content,
        persona=persona,
        title=title,
        url=url,
        links=links,
    )

    logger.info(f"Running mega-prompt for {url} (persona: {persona})")
    logger.info(f"Prompt length: {len(prompt)} chars, content length: {len(truncated_content)} chars")

    try:
        response = model.generate_content(prompt)
        raw_text = response.text

        # Parse the JSON response
        parsed = _parse_json_response(raw_text)

        # Validate through Pydantic (with defaults for any missing fields)
        result = MegaPromptOutput.model_validate(parsed)

        logger.info(
            f"Mega-prompt success: "
            f"{len(result.insights)} insight categories, "
            f"{len(result.action_plan)} action steps, "
            f"{len(result.opportunities)} opportunities, "
            f"{len(result.skill_gap.required_skills)} skills, "
            f"{len(result.similar_websites)} similar sites"
        )
        return result

    except Exception as e:
        logger.error(f"Mega-prompt failed: {e}")
        # Return a graceful fallback instead of crashing
        return _build_fallback_output(title, url, persona, str(e))


# ──────────────────────────────────────────────
# RAG Chat
# ──────────────────────────────────────────────

async def rag_chat(
    question: str,
    context_chunks: list[dict],
    persona: str,
    title: str,
    url: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Answer a user question using RAG context from FAISS.

    Args:
        question: The user's question.
        context_chunks: Retrieved chunks from FAISS [{content, score}, ...].
        persona: User persona.
        title: Website title.
        url: Website URL.
        chat_history: Previous messages [{role, content}, ...].

    Returns:
        {"answer": str, "sources": [{"content": str, "score": float}, ...]}
    """
    model = get_model()

    prompt = build_chat_prompt(
        question=question,
        context_chunks=context_chunks,
        persona=persona,
        title=title,
        url=url,
        chat_history=chat_history,
    )

    logger.info(f"RAG chat: '{question[:80]}...' with {len(context_chunks)} context chunks")

    try:
        # Use a separate generation config for chat (plain text, not JSON)
        chat_config = genai.GenerationConfig(
            temperature=0.7,
            max_output_tokens=2048,
        )
        response = model.generate_content(prompt, generation_config=chat_config)
        answer = response.text.strip()

        # Return the answer with source references
        sources = [
            {"content": chunk["content"][:200], "score": chunk.get("score", 0.0)}
            for chunk in context_chunks[:3]  # Top 3 most relevant
        ]

        return {"answer": answer, "sources": sources}

    except Exception as e:
        logger.error(f"RAG chat failed: {e}")
        return {
            "answer": f"I encountered an error processing your question. Please try again. ({str(e)})",
            "sources": [],
        }


# ──────────────────────────────────────────────
# JSON Parsing Helpers
# ──────────────────────────────────────────────

def _parse_json_response(raw_text: str) -> dict:
    """
    Parse JSON from Gemini's response, handling common formatting issues.

    Gemini sometimes wraps JSON in markdown fences or adds trailing text.
    """
    text = raw_text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in the text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_str = text[brace_start:brace_end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Last resort: try to fix common issues
    try:
        # Handle trailing commas
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    logger.error(f"Failed to parse JSON response. First 500 chars: {text[:500]}")
    raise ValueError(f"Could not parse Gemini response as JSON. Response starts with: {text[:200]}")


def _build_fallback_output(title: str, url: str, persona: str, error: str) -> MegaPromptOutput:
    """Build a minimal fallback output when the mega-prompt fails."""
    return MegaPromptOutput(
        summary=f"Analysis of {title} ({url}) could not be fully completed. Error: {error}",
        insights=[],
        action_plan=[],
        opportunities=[],
        skill_gap=SkillGap(),
        website_score=WebsiteScore(overall=0.0, dimensions=[]),
        why_it_matters=WhyItMatters(
            relevance_score=0.0,
            explanation=f"Analysis incomplete due to an error. Please try again.",
            key_takeaways=[],
            recommended_actions=["Try analyzing this website again"],
        ),
        similar_websites=[],
    )
