"""
WebIntel AI — Simple Retriever

Pure FAISS retrieval. No hybrid search, no BM25, no query expansion, no thresholds.
"""

import logging
from vector_store import FAISSStore
from typing import Callable

logger = logging.getLogger(__name__)


def is_conversational_followup(query: str, chat_history: list[dict] | None) -> bool:
    """Detect if the query is a follow-up that refers to previous context."""
    if not chat_history:
        return False
    q = query.lower().strip()
    words = q.split()
    
    # 1. Very short queries
    if len(words) < 6:
        return True
        
    # 2. Starts with common follow-up question words
    followup_starts = ["how", "why", "what", "explain", "summarize", "summarise", "tell me", "can you", "show", "give me"]
    if any(q.startswith(start) for start in followup_starts):
        return True
        
    # 3. Contains reference pronouns
    reference_pronouns = ["it", "they", "this", "that", "these", "those", "them", "its"]
    if any(word in words for word in reference_pronouns):
        return True
        
    return False


def simple_search(
    query: str,
    faiss_store: FAISSStore,
    embed_fn: Callable[[str], list[float]],
    top_k: int = 5,
    chat_history: list[dict] | None = None
) -> dict:
    """
    Perform a diversified FAISS similarity search, with conversational query expansion if needed.
    """
    # ── Conversational Query Expansion ──
    search_query = query
    expanded = False
    expanded_queries = []
    
    if chat_history and is_conversational_followup(query, chat_history):
        last_user_query = ""
        for msg in reversed(chat_history):
            if msg.get("role") == "user":
                last_user_query = msg.get("content", "").strip()
                break
        if last_user_query:
            # Combine previous context and current question
            search_query = f"{last_user_query} {query}"
            expanded = True
            expanded_queries = [search_query]
            logger.info(f"Query expanded for follow-up search: '{query}' -> '{search_query}'")

    query_emb = embed_fn(search_query)
    
    # ── Diversified Retrieval ──
    # Retrieve a larger candidate set (min 15 or 3x top_k)
    candidate_k = max(15, top_k * 3)
    results = faiss_store.search(query_emb, k=candidate_k)
    
    # Group results by page URL
    by_page = {}
    for r in results:
        url = r["metadata"].get("source_url") or r["metadata"].get("url") or "default_url"
        if url not in by_page:
            by_page[url] = []
        by_page[url].append(r)
        
    # Sort chunks in each page by score descending
    for url in by_page:
        by_page[url].sort(key=lambda x: x["score"], reverse=True)
        
    # Sort pages by their highest scoring chunk's score descending
    sorted_pages = sorted(by_page.items(), key=lambda x: x[1][0]["score"], reverse=True)
    
    final_chunks = []
    
    # Round robin: first pick the top chunk from each page
    for url, chunks in sorted_pages:
        if len(final_chunks) >= top_k:
            break
        final_chunks.append(chunks[0])
        
    # If we have slots left, fill with remaining chunks sorted by score
    if len(final_chunks) < top_k:
        remaining_chunks = []
        for url, chunks in sorted_pages:
            remaining_chunks.extend(chunks[1:])
        remaining_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        for rc in remaining_chunks:
            if len(final_chunks) >= top_k:
                break
            final_chunks.append(rc)
            
    # Format final results
    formatted_chunks = []
    for r in final_chunks:
        formatted_chunks.append({
            "content": r["content"],
            "score": r["score"],
            "metadata": r["metadata"]
        })
        
    logger.info(f"Diversified search retrieved {len(formatted_chunks)} chunks for query: '{query}' (expanded: {expanded})")
    
    return {
        "chunks": formatted_chunks,
        "expanded_queries": expanded_queries,
        "debug": {
            "search_type": "diversified_faiss",
            "query": search_query,
            "raw_results": len(results),
            "query_emb": query_emb
        }
    }

