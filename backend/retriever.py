"""
WebIntel AI — Simple Retriever

Pure FAISS retrieval. No hybrid search, no BM25, no query expansion, no thresholds.
"""

import logging
from vector_store import FAISSStore
from typing import Callable

logger = logging.getLogger(__name__)

def simple_search(
    query: str,
    faiss_store: FAISSStore,
    embed_fn: Callable[[str], list[float]],
    top_k: int = 5
) -> dict:
    """
    Perform a simple FAISS similarity search.
    """
    query_emb = embed_fn(query)
    
    # 1. Search FAISS
    results = faiss_store.search(query_emb, k=top_k)
    
    # Format results
    final_chunks = []
    for r in results:
        final_chunks.append({
            "content": r["content"],
            "score": r["score"],
            "metadata": r["metadata"]
        })
        
    logger.info(f"Simple search retrieved {len(final_chunks)} chunks for query: '{query}'")
    
    return {
        "chunks": final_chunks,
        "debug": {
            "search_type": "simple_faiss",
            "query": query,
            "raw_results": len(results),
            "query_emb": query_emb
        }
    }
