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

_analysis_model = None
_chat_model = None
_verification_model = None


def get_analysis_model() -> genai.GenerativeModel:
    """Lazy-initialize the Gemini model for structured analysis reports."""
    global _analysis_model
    if _analysis_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        genai.configure(api_key=api_key)
        _analysis_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )
        logger.info("Gemini Analysis model initialized.")
    return _analysis_model


def get_chat_model() -> genai.GenerativeModel:
    """Lazy-initialize the strict RAG chat model."""
    global _chat_model
    if _chat_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        genai.configure(api_key=api_key)
        
        chat_system_prompt = """You are a strict Retrieval-Augmented Generation (RAG) assistant.
Your ONLY source of knowledge is the provided context chunks.
You must NEVER use your own pre-trained knowledge, prior memory, or make assumptions.

Strict Grounding Rules:
1. If the question is completely unrelated to the website domain, topics, products, or services (as described in the retrieved chunks), respond with exactly: "This question is unrelated to the analyzed website." and do not add any other text.
2. If the retrieved context chunks do not contain direct, explicit information to answer the question, respond with exactly: "Information not found in analyzed website content." and do not add any other text.
3. Otherwise, answer the question truthfully and concisely, using ONLY information from the context.
4. You must cite the sources you use by appending [Source N] (e.g. [Source 1], [Source 2]) to the sentences that use that information. Do not invent sources.
5. If some parts of the question can be answered but others cannot, only answer the parts supported by the context and note that other parts were not found.
"""
        _chat_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=chat_system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )
        logger.info("Gemini strict Chat model initialized.")
    return _chat_model


def get_verification_model() -> genai.GenerativeModel:
    """Lazy-initialize the verification model."""
    global _verification_model
    if _verification_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        genai.configure(api_key=api_key)
        _verification_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=10,
            ),
        )
        logger.info("Gemini Verification model initialized.")
    return _verification_model


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
    model = get_analysis_model()

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
# RAG Chat & Grounding Safeguards
# ──────────────────────────────────────────────

async def verify_answer_claims(question: str, answer: str, context_chunks: list[dict]) -> bool:
    """
    Verify that the answer is completely supported by the retrieved chunks.
    Returns True if approved, False if rejected.
    """
    clean_ans = answer.strip()
    if clean_ans in [
        "Information not found in analyzed website content.",
        "This question is unrelated to the analyzed website."
    ]:
        return True

    model = get_verification_model()
    
    context_text = ""
    for i, c in enumerate(context_chunks, 1):
        context_text += f"[Source {i}] {c['content']}\n\n"
        
    prompt = f"""You are a factual verification agent. Given a user question, a set of retrieved text chunks, and a proposed answer, verify if every claim in the proposed answer is directly and fully supported by the text chunks.

User Question:
{question}

Retrieved Context Chunks:
{context_text}

Proposed Answer:
{answer}

Instructions:
1. If the answer is "Information not found in analyzed website content." or "This question is unrelated to the analyzed website.", respond with "APPROVE".
2. Check every claim in the Proposed Answer. If any statement or claim is not fully supported by the text chunks, respond with "REJECT".
3. If every claim is fully supported, respond with "APPROVE".
4. Output exactly "APPROVE" or "REJECT" with no explanation.
"""
    try:
        response = model.generate_content(prompt)
        verdict = response.text.strip().upper()
        logger.info(f"Factual verification verdict: {verdict}")
        return "APPROVE" in verdict
    except Exception as e:
        logger.error(f"Factual verification error: {e}")
        return False


def validate_citations_deterministically(answer: str, k: int) -> bool:
    """
    Checks if the answer's citations are valid.
    Rules:
    1. If the answer is a default refusal, it doesn't need citations.
    2. Otherwise, it must cite at least one source.
    3. All cited sources must be between 1 and k.
    """
    clean_ans = answer.strip()
    if clean_ans in [
        "Information not found in analyzed website content.",
        "This question is unrelated to the analyzed website."
    ]:
        return True
        
    citations = re.findall(r"\[Source (\d+)\]", answer)
    if not citations:
        logger.warning("Answer rejected: No citations found in non-refusal answer.")
        return False
        
    for cit in citations:
        idx = int(cit)
        if idx < 1 or idx > k:
            logger.warning(f"Answer rejected: Citation [Source {idx}] is out of bounds (k={k}).")
            return False
            
    return True


async def generate_answer_raw(prompt: str, model: genai.GenerativeModel) -> str:
    chat_config = genai.GenerationConfig(
        temperature=0.0,
        max_output_tokens=2048,
    )
    response = model.generate_content(prompt, generation_config=chat_config)
    return response.text.strip()


async def rag_chat(
    question: str,
    context_chunks: list[dict],
    persona: str,
    title: str,
    url: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Answer a user question using RAG context from FAISS with strict grounding and verification.
    """
    RETRIEVAL_THRESHOLD = 0.22
    k = len(context_chunks)
    
    # Step 5: Check retrieval confidence threshold
    max_score = max([c.get("score", 0.0) for c in context_chunks]) if context_chunks else 0.0
    if not context_chunks or max_score < RETRIEVAL_THRESHOLD:
        logger.info(f"Retrieval confidence below threshold ({max_score:.3f} < {RETRIEVAL_THRESHOLD}). Bypassing LLM.")
        return {
            "answer": "Information not found in analyzed website content.",
            "sources": [],
            "chunk_count": len(context_chunks),
            "avg_similarity": sum([c.get("score", 0.0) for c in context_chunks]) / len(context_chunks) if context_chunks else 0.0,
            "debug": {
                "retrieved_chunks": [
                    {
                        "chunk_id": c.get("metadata", {}).get("chunk_id", i),
                        "url": c.get("metadata", {}).get("url", url),
                        "chunk_text": c["content"],
                        "score": c.get("score", 0.0)
                    }
                    for i, c in enumerate(context_chunks)
                ],
                "final_prompt": "Bypassed LLM due to low retrieval score.",
                "context_length": 0,
                "tokens_sent": 0
            }
        }
        
    model = get_chat_model()
    prompt = build_chat_prompt(
        question=question,
        context_chunks=context_chunks,
        persona=persona,
        title=title,
        url=url,
        chat_history=chat_history,
    )
    
    logger.info(f"RAG chat: '{question[:80]}...' with {k} chunks. Max score: {max_score:.3f}")
    try:
        answer = await generate_answer_raw(prompt, model)
        
        # ── Deterministic Grounding Checks ──
        is_valid = validate_citations_deterministically(answer, k)
        
        # ── LLM Verification Safeguard ──
        if is_valid:
            is_valid = await verify_answer_claims(question, answer, context_chunks)
            
        # ── Regeneration if invalid ──
        if not is_valid:
            logger.warning("Initial answer failed grounding checks. Regenerating with strict parameters...")
            strict_prompt = prompt + "\n\nCRITICAL WARNING: Your previous answer was REJECTED for lack of grounding. You MUST only use the provided context. Cite correctly using [Source N]. If you cannot answer using ONLY the context, respond with exactly: 'Information not found in analyzed website content.'"
            answer = await generate_answer_raw(strict_prompt, model)
            
            # Verify again
            is_valid = validate_citations_deterministically(answer, k)
            if is_valid:
                is_valid = await verify_answer_claims(question, answer, context_chunks)
                
            if not is_valid:
                logger.warning("Regenerated answer also failed. Returning strict default refusal.")
                answer = "Information not found in analyzed website content."
                
    except Exception as e:
        logger.error(f"RAG Chat execution error: {e}")
        answer = "Information not found in analyzed website content."

    # Parse and compile sources actually used
    sources = []
    cited_ids = [int(i) for i in re.findall(r"\[Source (\d+)\]", answer)]
    for i, chunk in enumerate(context_chunks, 1):
        if i in cited_ids:
            sources.append({
                "chunk_id": chunk.get("metadata", {}).get("chunk_id", i - 1),
                "url": chunk.get("metadata", {}).get("url", url),
                "content": chunk["content"][:200],
                "chunk_text": chunk["content"],
                "score": chunk.get("score", 0.0)
            })

    avg_sim = sum([c.get("score", 0.0) for c in context_chunks]) / k if k > 0 else 0.0
    
    # Calculate tokens sent
    try:
        tokens_count = model.count_tokens(prompt).total_tokens
    except Exception:
        tokens_count = len(prompt) // 4
        
    debug_info = {
        "retrieved_chunks": [
            {
                "chunk_id": c.get("metadata", {}).get("chunk_id", i),
                "url": c.get("metadata", {}).get("url", url),
                "chunk_text": c["content"],
                "score": c.get("score", 0.0)
            }
            for i, c in enumerate(context_chunks)
        ],
        "final_prompt": prompt,
        "context_length": len(prompt),
        "tokens_sent": tokens_count
    }
    
    return {
        "answer": answer,
        "sources": sources,
        "chunk_count": len(context_chunks),
        "avg_similarity": avg_sim,
        "debug": debug_info
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
