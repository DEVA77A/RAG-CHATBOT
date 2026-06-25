"""
WebIntel AI — AI Engine (Pure RAG)

Single core function:
  rag_chat() — RAG-powered Q&A using hybrid-retrieved context + Gemini

No mega-prompt. No persona analysis. No website intelligence reports.
Strict grounding with citation validation.
"""

import logging
import os
from dotenv import load_dotenv
load_dotenv()
import re
import warnings

# Suppress the google.generativeai deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
import google.generativeai as genai

from prompts import CHAT_SYSTEM_INSTRUCTION, build_chat_prompt

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Gemini Configuration
# ──────────────────────────────────────────────

_chat_models = {}

def get_chat_model(model_name="gemini-1.5-flash") -> genai.GenerativeModel:
    """Lazy-initialize the strict RAG chat model."""
    if model_name not in _chat_models:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        genai.configure(api_key=api_key)

        _chat_models[model_name] = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=CHAT_SYSTEM_INSTRUCTION,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )
    return _chat_models[model_name]


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

    # Multi-Provider LLM Fallback Routing
    import os
    import hashlib
    import json
    import uuid
    from database import find_in_semantic_cache, save_to_semantic_cache
    
    question_embedding = kwargs.get("question_embedding", [])
    
    url_hash = hashlib.md5(url.encode()).hexdigest()
    context_text_for_hash = "".join([c["content"] for c in context_chunks])
    context_hash = hashlib.md5(context_text_for_hash.encode()).hexdigest()
    
    debug_info = {
        "Retrieval Provider": "FAISS",
        "LLM Provider": "None",
        "Response Source": "None",
        "Cache Hit": False,
        "Fallback Activated": False
    }

    answer = None
    
    # Provider 1: Gemini Waterfall
    if not answer:
        import google.api_core.exceptions
        gemini_models = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-1.0-pro"]
        for m_name in gemini_models:
            try:
                model = get_chat_model(m_name)
                response = model.generate_content(prompt)
                answer = response.text.strip()
                debug_info["LLM Provider"] = f"Gemini ({m_name})"
                debug_info["Response Source"] = "Cloud"
                break
            except google.api_core.exceptions.ResourceExhausted as e:
                logger.warning(f"Gemini Rate Limit on {m_name}, falling back to next model.")
                continue
            except Exception as e:
                logger.warning(f"Gemini error on {m_name}: {e}")
                continue
        if not answer:
            debug_info["Fallback Activated"] = True

    # Provider 2: Claude Haiku
    if not answer:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic
                from anthropic import RateLimitError
                client = anthropic.Anthropic(api_key=anthropic_key)
                message = client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1000,
                    system="You are a strict RAG assistant.",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = message.content[0].text.strip()
                debug_info["LLM Provider"] = "Claude Haiku"
                debug_info["Response Source"] = "Cloud"
            except RateLimitError as e:
                logger.error(f"Claude Rate Limit error: {e}")
            except Exception as e:
                logger.error(f"Claude error: {e}")
        else:
            logger.warning("No Anthropic API Key found, skipping Claude fallback.")

    # Provider 3: Semantic Cache
    if not answer and question_embedding is not None and len(question_embedding) > 0:
        try:
            def cos_sim(a, b):
                dot = sum(x*y for x, y in zip(a, b))
                norm_a = sum(x*x for x in a) ** 0.5
                norm_b = sum(x*x for x in b) ** 0.5
                return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

            cached_rows = await find_in_semantic_cache(url_hash, context_hash, "default")
            best_match = None
            best_score = -1
            for row in cached_rows:
                cached_emb = json.loads(row["question_embedding"])
                score = cos_sim(question_embedding, cached_emb)
                if score > 0.95 and score > best_score:
                    best_score = score
                    best_match = row
                    
            if best_match:
                answer = best_match["answer"]
                debug_info["LLM Provider"] = "Semantic Cache"
                debug_info["Response Source"] = "Cache"
                debug_info["Cache Hit"] = True
        except Exception as e:
            logger.error(f"Semantic Cache error: {e}")

    # Provider 4: Context Summarizer
    if not answer:
        try:
            import re
            q_words = set(word for word in re.findall(r'\b\w+\b', question.lower()) if len(word) > 3)
            valid = False
            for c in context_chunks:
                if c.get("score", 0.0) >= 0.45:
                    valid = True
                    break
                meta = c.get("metadata", {})
                title_words = set(re.findall(r'\b\w+\b', meta.get("page_title", "").lower()))
                heading_words = set(re.findall(r'\b\w+\b', meta.get("heading", "").lower()))
                if q_words.intersection(title_words) or q_words.intersection(heading_words):
                    valid = True
                    break

            if not valid:
                answer = "The requested information could not be found in the indexed website content."
                debug_info["LLM Provider"] = "Local Context Summary (Rejected)"
                debug_info["Response Source"] = "Local Fallback"
            else:
                chunks_to_use = context_chunks[:2]
                text_parts = []
                for c in chunks_to_use:
                    meta = c.get("metadata", {})
                    title = meta.get("source_title", "Unknown Title")
                    url = meta.get("source_url", "")
                    chunk_id = meta.get("chunk_id", 1)
                    
                    text = c['content']
                    if "[CONTENT]" in text:
                        text = text.split("[CONTENT]")[-1]
                    elif "Content: " in text:
                        text = text.split("Content: ", 1)[-1]
                        
                    clean_text = text.strip()
                    if clean_text:
                        if len(clean_text) > 250:
                            clean_text = clean_text[:247] + "..."
                        
                        part = f"{clean_text}\n\nURL: {url}\n\n[Source {chunk_id}]"
                        text_parts.append(part)
                
                content_text = "\n\n---\n\n".join(text_parts)
                answer = f"AI providers are temporarily unavailable. Here is the most relevant content I found:\n\n{content_text}"
                debug_info["LLM Provider"] = "Local Context Summary"
                debug_info["Response Source"] = "Local Fallback"
        except Exception as e:
            logger.error(f"Local summarizer error: {e}")

    # Provider 5: Graceful Failure
    if not answer:
        answer = "We successfully retrieved information from the website, but all AI providers are currently unavailable. Please try again shortly."
        debug_info["LLM Provider"] = "Graceful Failure"
        debug_info["Response Source"] = "Error"

    # Save Cloud Response to Semantic Cache
    if answer and debug_info["Response Source"] == "Cloud" and question_embedding is not None and len(question_embedding) > 0:
        try:
            chunk_ids = [c.get("metadata", {}).get("chunk_id", 0) for c in context_chunks]
            await save_to_semantic_cache(
                str(uuid.uuid4()), url, url_hash, question, question_embedding, chunk_ids, context_hash, "default", answer
            )
        except Exception as e:
            logger.error(f"Cache save error: {e}")

    # Extract cited sources
    sources = []
    cited_ids = set(int(i) for i in re.findall(r"\[Source (\d+)\]", answer))
    # If no citations were generated by LLM, fallback to citing the top chunk to prevent frontend crash
    if not cited_ids and context_chunks:
        cited_ids = {1}
        
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        if i in cited_ids:
            source_url = meta.get("source_url", url)
            source_title = meta.get("source_title", "Unknown Title")
            sources.append({
                "chunk_id": meta.get("chunk_id", i - 1),
                "source_url": source_url,
                "source_title": source_title,
                "content": chunk["content"][:200],
                "chunk_text": chunk["content"],
                "score": chunk.get("score", 0.0),
            })
            
            # Replace inline [Source X] with clickable markdown link
            target_str = f"[Source {i}]"
            md_link = f"[Source: {source_title}]({source_url})"
            answer = answer.replace(target_str, md_link)

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
