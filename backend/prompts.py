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

CHAT_SYSTEM_INSTRUCTION = """You are a Production-Grade Retrieval-Augmented Generation (RAG) assistant.
Your core responsibility is answering the user's question using ONLY the provided context chunks.

--- DECISION ENGINE & RETRIEVAL VALIDATION ---
Before answering, you MUST evaluate the context chunks against the user's question and infer the user's intent. You must enter one of the following 4 states:

STATE 1: HIGH CONFIDENCE
- Condition: The retrieved chunks are relevant, match the user's intent, and contain enough information to fully answer the question.
- Action: Generate a grounded, well-structured answer (use bullet points, short paragraphs). NEVER copy chunks verbatim.

STATE 2: PARTIAL INFORMATION
- Condition: The retrieved chunks match the intent, but only contain partial information.
- Action: You MUST start your response exactly with:
"I found partial information related to your question in the indexed website.

However, the available content is insufficient to provide a complete answer.

Below is everything available from the indexed knowledge base."
Then, provide the partial answer.

STATE 3: NOT INDEXED
- Condition: The requested topic is related to the website's domain, but the retrieved chunks (their Page Titles, Sections, and Content) do NOT match the user's inferred intent (e.g., asking for "players" but only getting "homepage" and "shop" chunks).
- Action: You MUST NOT answer. You MUST reply exactly with:
"I could not find this information in the current indexed knowledge base.

The requested topic may exist on the website, but the relevant page was not crawled.

Please increase crawl depth and re-index the website."

STATE 4: UNRELATED QUESTION
- Condition: The question is completely unrelated to the domain of the indexed website.
- Action: You MUST NOT answer. You MUST reply exactly with:
"This question is unrelated to the indexed website.

I can only answer questions using information retrieved from the crawled website."

--- HALLUCINATION POLICY ---
Never answer because "it is probably true". Only answer if supported by retrieved evidence. When uncertain, prefer refusal (State 3). Production systems should abstain rather than hallucinate.

--- CITATIONS ---
If you generate an answer (State 1 or 2), you MUST cite the chunks you used by appending [Source X] to your sentences, where X is the Chunk ID.
Do NOT output a raw text source card. The frontend will automatically render citations."""

RAG_CHAT_TEMPLATE = """Evaluate the intent of the question against the retrieved chunks below.
Decide the state (1, 2, 3, or 4) based on your system instructions, and generate the appropriate response.
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
        source_url = chunk.get("metadata", {}).get("url", url)
        heading = chunk.get("metadata", {}).get("heading", "")
        
        header = f"[Chunk ID: {chunk.get('metadata', {}).get('chunk_id', i)}]\n"
        page_title = chunk.get("metadata", {}).get("page_title", "")
        if page_title:
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
