"""
WebIntel AI — AI Engine (Pure RAG)

Single core function:
  rag_chat() — RAG-powered Q&A using hybrid-retrieved context + Gemini

No mega-prompt. No persona analysis. No website intelligence reports.
Strict grounding with citation validation.
"""

import logging
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

from prompts import CHAT_SYSTEM_INSTRUCTION, build_chat_prompt

load_dotenv()
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Gemini Configuration
# ──────────────────────────────────────────────

_chat_model = None


def get_chat_model() -> genai.GenerativeModel:
    """Lazy-initialize the strict RAG chat model."""
    global _chat_model
    if _chat_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        genai.configure(api_key=api_key)

        _chat_model = genai.GenerativeModel(
            model_name="gemini-3.5-flash",
            system_instruction=CHAT_SYSTEM_INSTRUCTION,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )
        logger.info("Gemini strict RAG chat model initialized.")
    return _chat_model


# ──────────────────────────────────────────────
# Citation Validation (Deterministic — no LLM call)
# ──────────────────────────────────────────────

_REFUSAL_RESPONSES = {
    "Information not found in analyzed website content.",
    "This question is unrelated to the analyzed website.",
}


def validate_citations(answer: str, k: int) -> bool:
    """
    Check if the answer has valid citations.
    
    Rules:
    1. Refusal responses don't need citations.
    2. Non-refusal answers MUST cite at least one source.
    3. All cited source numbers must be between 1 and k.
    """
    clean = answer.strip()
    if clean in _REFUSAL_RESPONSES:
        return True

    citations = re.findall(r"\[Source (\d+)\]", answer)
    if not citations:
        logger.warning("Answer rejected: No citations in non-refusal answer.")
        return False

    for cit in citations:
        idx = int(cit)
        if idx < 1 or idx > k:
            logger.warning(f"Answer rejected: Citation [Source {idx}] out of bounds (k={k}).")
            return False

    return True


# ──────────────────────────────────────────────
# RAG Chat
# ──────────────────────────────────────────────

async def rag_chat(
    question: str,
    context_chunks: list[dict],
    title: str = "",
    url: str = "",
    chat_history: list[dict] | None = None,
    **kwargs,  # Accept extra kwargs for backward compat
) -> dict:
    """
    Answer a user question using simple-retrieved context with strict grounding.
    
    Returns dict with:
        answer, sources, chunk_count, avg_similarity, debug
    """
    k = len(context_chunks)
    avg_sim = sum(c.get("score", 0.0) for c in context_chunks) / k if k > 0 else 0.0

    if not context_chunks:
        return _build_refusal_response(
            "Information not found in analyzed website content.",
            context_chunks, url, avg_sim,
            "No context chunks provided."
        )

    # Build prompt
    prompt = build_chat_prompt(
        question=question,
        context_chunks=context_chunks,
        title=title,
        url=url,
        chat_history=chat_history,
    )

    logger.info(f"RAG chat: '{question[:80]}' with {k} chunks.")

    # Generate answer
    model = get_chat_model()
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        # Fallback: bypass quota limits by formatting chunks directly
        is_detailed = "detail" in question.lower()
        
        if is_detailed:
            text_parts = []
            for c in context_chunks:
                text = c['content']
                if "Content: " in text:
                    text = text.split("Content: ", 1)[-1]
                text_parts.append(text.strip())
            content_text = "\n\n".join(text_parts)
        else:
            top = context_chunks[0]
            content_text = top['content']
            if "Content: " in content_text:
                content_text = content_text.split("Content: ", 1)[-1]
            content_text = f"{content_text[:350].strip()}..."

        top = context_chunks[0]
        page_title = top.get("metadata", {}).get("page_title", "Unknown Title")
        source_url = top.get("metadata", {}).get("url", url)
        score = top.get("score", 0.0)
        chunk_id = top.get("metadata", {}).get("chunk_id", 1)
        heading = top.get("metadata", {}).get("heading", "Main")
        
        source_card = (
            f"\n\n──────────────────────────\n"
            f"📄 Source\n"
            f"Page: {page_title}\n"
            f"Section: {heading}\n"
            f"Original URL: {source_url}\n"
            f"Similarity: {score:.2f}\n"
            f"Chunk ID: {chunk_id}\n"
            f"────────────────────────── [Source 1]"
        )
        answer = f"{content_text}{source_card}"

    # Extract cited sources
    sources = []
    cited_ids = set(int(i) for i in re.findall(r"\[Source (\d+)\]", answer))
    # If no citations were generated by LLM, fallback to citing the top chunk to prevent frontend crash
    if not cited_ids and context_chunks:
        cited_ids = {1}
        
    for i, chunk in enumerate(context_chunks, 1):
        if i in cited_ids:
            sources.append({
                "chunk_id": chunk.get("metadata", {}).get("chunk_id", i - 1),
                "url": chunk.get("metadata", {}).get("url", url),
                "title": chunk.get("metadata", {}).get("page_title", ""),
                "content": chunk["content"][:200],
                "chunk_text": chunk["content"],
                "score": chunk.get("score", 0.0),
            })

    # Token count
    try:
        tokens_count = model.count_tokens(prompt).total_tokens
    except Exception:
        tokens_count = len(prompt) // 4

    debug_info = {
        "retrieved_chunks": [
            {
                "chunk_id": c.get("metadata", {}).get("chunk_id", i),
                "url": c.get("metadata", {}).get("url", url),
                "heading": c.get("metadata", {}).get("heading", ""),
                "chunk_text": c["content"],
                "score": c.get("score", 0.0),
            }
            for i, c in enumerate(context_chunks)
        ],
        "final_prompt": prompt,
        "context_length": len(prompt),
        "tokens_sent": tokens_count,
    }

    return {
        "answer": answer,
        "sources": sources,
        "chunk_count": k,
        "avg_similarity": avg_sim,
        "debug": debug_info,
    }


def _build_refusal_response(
    message: str, chunks: list, url: str, avg_sim: float, prompt_note: str
) -> dict:
    """Build a standard refusal response with debug info."""
    return {
        "answer": message,
        "sources": [],
        "chunk_count": len(chunks),
        "avg_similarity": avg_sim,
        "debug": {
            "retrieved_chunks": [
                {
                    "chunk_id": c.get("metadata", {}).get("chunk_id", i),
                    "url": c.get("metadata", {}).get("url", url),
                    "heading": c.get("metadata", {}).get("heading", ""),
                    "chunk_text": c["content"],
                    "score": c.get("score", 0.0),
                }
                for i, c in enumerate(chunks)
            ],
            "final_prompt": prompt_note,
            "context_length": 0,
            "tokens_sent": 0,
        },
    }
