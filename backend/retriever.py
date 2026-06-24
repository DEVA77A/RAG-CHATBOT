"""
WebIntel AI — Hybrid Retriever

Combines Dense (FAISS) + Sparse (BM25) retrieval with Reciprocal Rank Fusion.
Includes query expansion and metadata-aware boosting.

Pipeline:
  1. Query expansion (keyword extraction + section-aware synonyms)
  2. Dense search: FAISS cosine similarity → top-20
  3. Sparse search: BM25 term matching → top-20
  4. RRF merge: Reciprocal Rank Fusion → combined ranking
  5. Metadata boost: heading/title matches get score bonus
  6. Threshold filter: drop chunks below similarity threshold
  7. Return top-K results
"""

import logging
import re
import time
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

RRF_K = 60  # RRF smoothing constant
SIMILARITY_THRESHOLD = 0.35  # Minimum FAISS score to consider
METADATA_BOOST = 0.08  # Score boost when heading matches query keywords
DENSE_TOP_K = 20  # How many to retrieve from FAISS
SPARSE_TOP_K = 20  # How many to retrieve from BM25
FINAL_TOP_K = 5   # Final number of chunks to return

# Stopwords for tokenization
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "this", "that", "these", "those", "what", "which",
    "who", "whom", "how", "when", "where", "why", "if", "then", "else",
    "and", "or", "but", "not", "no", "nor", "so", "yet", "both",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "between", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "just", "also", "very", "really", "quite",
    "tell", "me", "please", "give", "show", "explain", "describe",
}

# Section-reference synonyms for query expansion
_SECTION_SYNONYMS = {
    "introduction": ["overview", "getting started", "about", "intro", "summary", "welcome"],
    "first paragraph": ["introduction", "opening", "overview", "beginning", "start"],
    "overview": ["introduction", "summary", "about", "getting started"],
    "getting started": ["introduction", "quickstart", "setup", "installation", "first steps"],
    "tutorial": ["guide", "walkthrough", "how to", "lesson", "getting started"],
    "api": ["endpoint", "rest", "interface", "sdk", "reference"],
    "faq": ["frequently asked", "questions", "help", "troubleshooting"],
    "install": ["setup", "installation", "getting started", "download"],
    "conclusion": ["summary", "wrap up", "final", "closing"],
}


# ──────────────────────────────────────────────
# Query Expansion
# ──────────────────────────────────────────────

def expand_query(question: str) -> list[str]:
    """
    Expand a user question into multiple search queries.
    
    Strategies:
      1. Original query (always included)
      2. Keywords-only version (stopwords removed)
      3. Section-synonym expansions (if question references a section)
    
    Returns list of query strings to search against.
    """
    queries = [question]
    
    lower_q = question.lower().strip()
    
    # Extract keywords (non-stopwords)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', lower_q)
    keywords = [w for w in words if w not in _STOPWORDS]
    if keywords and len(keywords) < len(words):
        queries.append(" ".join(keywords))
    
    # Section-synonym expansion
    for trigger, synonyms in _SECTION_SYNONYMS.items():
        if trigger in lower_q:
            for syn in synonyms[:3]:  # Limit to 3 expansions per trigger
                expanded = lower_q.replace(trigger, syn)
                if expanded != lower_q and expanded not in queries:
                    queries.append(expanded)
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        q_norm = q.strip().lower()
        if q_norm not in seen:
            seen.add(q_norm)
            unique.append(q)
    
    logger.info(f"Query expansion: '{question}' → {len(unique)} queries")
    return unique


# ──────────────────────────────────────────────
# BM25 Sparse Index
# ──────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r'\b[a-zA-Z0-9]{2,}\b', text.lower())


class BM25Index:
    """Lightweight BM25 index built from chunk texts."""
    
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        tokenized = [_tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        logger.info(f"BM25 index built with {len(chunks)} documents")
    
    def search(self, query: str, k: int = SPARSE_TOP_K) -> list[tuple[int, float]]:
        """
        Search BM25 index.
        
        Returns: list of (chunk_index, bm25_score) sorted by score descending.
        """
        if not self.bm25 or not self.chunks:
            return []
        
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []
        
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        return [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ──────────────────────────────────────────────
# Hybrid Retriever
# ──────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = RRF_K
) -> list[tuple[int, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    
    For each document d appearing in any list:
      RRF_score(d) = Σ 1 / (k + rank_i(d))
    
    Args:
        ranked_lists: List of ranked lists, each containing (doc_id, score) tuples.
        k: Smoothing constant (default 60).
    
    Returns:
        Merged list of (doc_id, rrf_score) sorted by score descending.
    """
    fusion_scores = {}
    
    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list):
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = 0.0
            fusion_scores[doc_id] += 1.0 / (k + rank + 1)
    
    # Sort by fusion score
    merged = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


def hybrid_search(
    query: str,
    faiss_store,
    bm25_index: BM25Index,
    embed_query_fn,
    top_k: int = FINAL_TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Full hybrid retrieval pipeline.
    
    1. Expand query
    2. Dense search (FAISS) for each expanded query
    3. Sparse search (BM25) for each expanded query
    4. RRF fusion
    5. Metadata boosting
    6. Threshold filtering
    7. Return top-K
    
    Returns dict with:
        - "chunks": list of result dicts
        - "retrieval_time": float (seconds)
        - "expanded_queries": list of str
        - "dense_count": int (pre-fusion dense hits)
        - "sparse_count": int (pre-fusion sparse hits)
    """
    t_start = time.perf_counter()
    
    # Step 1: Query expansion
    expanded_queries = expand_query(query)
    
    # Step 2 & 3: Dense + Sparse search for each expanded query
    all_dense_results = []
    all_sparse_results = []
    
    for eq in expanded_queries:
        # Dense: FAISS
        query_emb = embed_query_fn(eq)
        dense_results = faiss_store.search(query_emb, k=DENSE_TOP_K)
        
        # Convert to (index, score) pairs
        dense_ranked = []
        for r in dense_results:
            chunk_idx = r.get("metadata", {}).get("chunk_id", -1)
            if chunk_idx >= 0:
                dense_ranked.append((chunk_idx, r["score"]))
        all_dense_results.append(dense_ranked)
        
        # Sparse: BM25
        sparse_ranked = bm25_index.search(eq, k=SPARSE_TOP_K)
        all_sparse_results.append(sparse_ranked)
    
    # Step 4: RRF fusion across all result lists
    all_ranked_lists = all_dense_results + all_sparse_results
    fused = reciprocal_rank_fusion(all_ranked_lists)
    
    # Step 5: Build final results with metadata boosting
    query_keywords = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    query_keywords -= _STOPWORDS
    
    results = []
    for chunk_id, rrf_score in fused:
        if chunk_id < 0 or chunk_id >= len(faiss_store.chunks):
            continue
        
        chunk_text = faiss_store.chunks[chunk_id]
        chunk_meta = faiss_store.metadata[chunk_id] if chunk_id < len(faiss_store.metadata) else {}
        
        # Get the original FAISS similarity score for this chunk
        faiss_score = 0.0
        for ranked_list in all_dense_results:
            for cid, score in ranked_list:
                if cid == chunk_id:
                    faiss_score = max(faiss_score, score)
                    break
        
        # Metadata boost: if heading or title matches query keywords
        heading = chunk_meta.get("heading", "").lower()
        title = chunk_meta.get("title", "").lower()
        section_type = chunk_meta.get("section_type", "body")
        
        boost = 0.0
        for kw in query_keywords:
            if kw in heading or kw in title:
                boost += METADATA_BOOST
        
        # Intro sections get slight boost for introductory queries
        intro_query_words = {"introduction", "overview", "first", "start", "getting", "about", "what"}
        if section_type == "intro" and query_keywords & intro_query_words:
            boost += METADATA_BOOST
        
        final_score = rrf_score + boost
        
        results.append({
            "content": chunk_text,
            "score": faiss_score,  # Report FAISS cosine similarity as the visible score
            "rrf_score": final_score,
            "metadata": chunk_meta,
        })
    
    # Step 6: Sort by RRF score (with metadata boost)
    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    # Step 7: Apply threshold on FAISS score AND limit to top-K
    filtered = []
    for r in results:
        if r["score"] >= similarity_threshold or len(filtered) == 0:
            # Always include at least the top result even if below threshold
            # (the answer will be "not found" if score is too low)
            filtered.append(r)
        if len(filtered) >= top_k:
            break
    
    t_end = time.perf_counter()
    retrieval_time = t_end - t_start
    
    total_dense = sum(len(rl) for rl in all_dense_results)
    total_sparse = sum(len(rl) for rl in all_sparse_results)
    
    logger.info(
        f"Hybrid search: {total_dense} dense + {total_sparse} sparse → "
        f"{len(fused)} fused → {len(filtered)} final (threshold={similarity_threshold}) "
        f"in {retrieval_time*1000:.1f}ms"
    )
    
    return {
        "chunks": filtered,
        "retrieval_time": retrieval_time,
        "expanded_queries": expanded_queries,
        "dense_count": total_dense,
        "sparse_count": total_sparse,
    }
