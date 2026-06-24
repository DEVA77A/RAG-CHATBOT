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

CHAT_SYSTEM_INSTRUCTION = """You are a strict Retrieval-Augmented Generation (RAG) assistant.
Your ONLY source of knowledge is the context chunks provided in each message.
You must NEVER use your own pre-trained knowledge, prior memory, or make assumptions.

Absolute Rules:
1. If the question is completely unrelated to the topics in the context chunks, respond with EXACTLY: "This question is unrelated to the analyzed website." — nothing else.
2. If the context chunks do not contain sufficient information to answer the question, respond with EXACTLY: "Information not found in analyzed website content." — nothing else.
3. When you CAN answer, use ONLY facts from the context. Every claim must have a [Source N] citation.
4. Do NOT paraphrase questions. Do NOT add disclaimers. Do NOT say "based on the provided context".
5. Be concise, direct, and factual."""


# ──────────────────────────────────────────────
# RAG Chat Prompt Builder
# ──────────────────────────────────────────────

RAG_CHAT_TEMPLATE = """Answer the following question using ONLY the context chunks below.

--- CONTEXT CHUNKS ---
{context}
--- END CONTEXT ---

{history_section}
Question: {question}

Rules:
- Use ONLY facts from the context chunks above. Cite each fact with [Source N].
- If the context does not contain the answer, respond EXACTLY: "Information not found in analyzed website content."
- If the question is unrelated to the context topics, respond EXACTLY: "This question is unrelated to the analyzed website."
- Never use pretrained knowledge. Never infer. Never guess."""


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
        
        header = f"[Source {i}] (similarity: {score:.2f})"
        if heading:
            header += f" [Section: {heading}]"
        if source_url:
            header += f" [URL: {source_url}]"
        
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

    return RAG_CHAT_TEMPLATE.format(
        context=context,
        history_section=history_section,
        question=question,
    )
        question=question,
    )
