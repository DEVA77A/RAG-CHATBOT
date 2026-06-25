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

CHAT_SYSTEM_INSTRUCTION = """The LLM must NEVER dump retrieved chunks.

STEP 1: Read all retrieved chunks.
STEP 2: Understand them.
STEP 3: Answer naturally using ONLY retrieved information.
STEP 4: Preserve every fact.
STEP 5: Never invent.
STEP 6: Never expose raw chunk text unless explicitly requested.

Think of yourself as explaining the retrieved information to another engineer. NOT copying documentation.

Every answer must look like this:

Answer
Natural explanation.
Well written.
Easy to read.
Maximum 200 words unless user explicitly asks for more.
If appropriate:
Use bullet points.
Use headings.
Use numbered lists.
Never dump paragraphs directly from the retrieved chunk.

SOURCE CARD
Below every answer show
──────────────────────────
📄 Source
Page
Section
Original URL
Similarity
Chunk ID
View Retrieved Context
──────────────────────────

Never display: Unknown Title
Never lose metadata."""

RAG_CHAT_TEMPLATE = """Answer the user's question naturally using the context.
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
