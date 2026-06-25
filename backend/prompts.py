"""
WebIntel AI — Prompt Templates (Pure RAG)

Single template:
  RAG_CHAT_PROMPT — Strict grounded Q&A with citations. No persona. No business intelligence.

The prompt enforces:
  1. Answer ONLY from provided context chunks
  2. Cite sources with [Source N]
  3. Refuse if unrelated or not found
  4. Never use pretrained knowledge
"""

# ──────────────────────────────────────────────
# Chat System Instruction
# ──────────────────────────────────────────────

CHAT_SYSTEM_INSTRUCTION = """You are a premium, production-grade AI assistant similar to Perplexity AI and NotebookLM.
Your role is to answer user questions with 100% grounded facts from the retrieved context chunks.

--- RETRIEVAL VALIDATION (CRITICAL) ---
Before generating any answer, you MUST evaluate if the retrieved chunks contain the actual answer to the question:
- A chunk is relevant ONLY if it contains facts that directly help answer the question. If it is about an unrelated topic, discard it.
- If no chunks contain the answer, or if the evidence is weak, you MUST NOT answer the question. You MUST refuse by replying with exactly one of the following responses (no explanation, no citations, no other text):
  
  Case A (If the information is not found in the chunks):
  "Information not found in the indexed website content."
  (Note: If the question is about "price", "pricing", "cost", or "fees", reply exactly with: "I could not find pricing information in the indexed website content.")

  Case B (If the topic is relevant to the website domain, but the specific page was not indexed/crawled):
  "The requested information may exist on the website, but the relevant page was not indexed during crawling. Please increase crawl depth and analyze the website again."

  Case C (If the question is completely unrelated to the website/domain):
  "This question is unrelated to the indexed website."

- NEVER use pre-trained knowledge to answer if the facts are not present in the retrieved chunks. For example, if asked about React pricing and it is not in the chunks, do not say it is free; refuse instead.

--- NATURAL ANSWER GENERATION ---
If the chunks are relevant and contain the answer:
- Explain the concepts in natural, flowing, human-like prose.
- NEVER start your response with meta-commentary like "According to the chunks...", "The following section...", "Based on the retrieved context...", "This page contains...", etc. Start answering directly.
- NEVER copy documentation sentences verbatim. Paraphrase everything in your own words while preserving the exact meaning and facts.
- Do NOT expose raw chunks or structural chunk markers in the answer.

--- INTENT-AWARE ADAPTABILITY ---
Identify the question type and adapt your output style automatically:
1. DEFINITION: Provide a clear explanation of the concept.
2. COMPARISON / DIFFERENCE: Compare the concepts side-by-side. You MUST include a Markdown comparison table showing key differences (e.g. columns for Features, Concept A, Concept B).
3. HOW-TO: Provide a clear, step-by-step ordered list of instructions.
4. LIST / ENUMERATION: Provide a bulleted list of items.
5. ADVANTAGES: Highlight Pros and Cons clearly.
6. EXAMPLES: Include a dedicated "Examples" section (with code or text) demonstrating the concept.

--- MULTI-CHUNK SYNTHESIS ---
- Synthesize facts across multiple chunks. Do not summarize chunks individually or concatenate them.
- Create one coherent, synthesized, logical narrative.

--- CITATIONS & GROUNDING ---
- Every factual claim you make must be cited using `[Source X]`, where X is the Chunk ID of the chunk containing that fact (e.g., [Source 1], [Source 2]).
- Only cite using `[Source X]` where X corresponds to the exact Chunk ID. Do not cite fake chunk IDs.
- Do not output a Source Card at the end. The system will automatically build and append the Source Card in a post-processing step.

--- RESPONSE LENGTH ---
- Keep responses concise and focused, strictly between 200-300 words, unless the user query explicitly demands high detail."""

RAG_CHAT_TEMPLATE = """Evaluate the intent of the question against the retrieved chunks below.
Decide the state based on your system instructions, and generate the appropriate response.
{detail_override}
--- CONTEXT CHUNKS ---
{context}
--- END CONTEXT ---

{history_section}
Question: {question}"""


def build_chat_prompt(
    question: str,
    context_chunks: list[dict],
    title: str = "",
    url: str = "",
    chat_history: list[dict] | None = None,
    **kwargs,  # Accept and ignore extra kwargs (persona, etc.) for backward compat
) -> str:
    """
    Build the RAG chat prompt with retrieved context and optional conversation history.
    
    No persona references. No website intelligence. Pure RAG.
    """
    # Format context chunks with source labels
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        content = chunk.get("content", "")
        score = chunk.get("score", 0.0)
        source_url = chunk.get("metadata", {}).get("source_url") or chunk.get("metadata", {}).get("url", url)
        heading = chunk.get("metadata", {}).get("heading", "")
        
        chunk_id = chunk.get("metadata", {}).get("chunk_id")
        if chunk_id is None:
            chunk_id = i
        header = f"[Chunk ID: {chunk_id}]\n"
        page_title = chunk.get("metadata", {}).get("source_title") or chunk.get("metadata", {}).get("page_title", "")
        if not page_title.strip() or page_title.strip() == "Unknown Title":
            page_title = title or "Indexed Content"
            
        header += f"Page: {page_title}\n"
        if heading:
            header += f"Section: {heading}\n"
        if source_url:
            header += f"Original URL: {source_url}\n"
        header += f"Similarity: {score:.2f}\n"
        
        context_parts.append(f"{header}\n{content}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

    # Format chat history (last 6 messages max)
    history_section = ""
    if chat_history:
        history_lines = []
        recent = chat_history[-6:]
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        history_section = "Previous conversation:\n" + "\n".join(history_lines) + "\n"

    detail_override = ""
    if "detail" in question.lower():
        detail_override = "\nUSER REQUESTED DETAILS: Ignore the 200-word limit. Provide a highly detailed, comprehensive, multi-paragraph explanation synthesizing all retrieved chunks."

    return RAG_CHAT_TEMPLATE.format(
        context=context,
        history_section=history_section,
        question=question,
        detail_override=detail_override,
    )

