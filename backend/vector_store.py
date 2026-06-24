"""
WebIntel AI — FAISS Vector Store

Simple wrapper around FAISS for storing and searching embeddings.
Each analysis gets its own FAISS index (stored as a file on disk).

Uses IndexFlatIP (inner product) on L2-normalized vectors = cosine similarity.
"""

import os
import json
import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)

FAISS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_indexes")


class FAISSStore:
    """
    Per-analysis FAISS vector store.

    Usage:
        store = FAISSStore(analysis_id="abc123", dim=384)
        store.add(chunks=["text1", "text2"], embeddings=np_array)
        results = store.search(query_embedding, k=5)
        store.save()

        # Later:
        store = FAISSStore.load(analysis_id="abc123")
        results = store.search(...)
    """

    def __init__(self, analysis_id: str, dim: int = 384):
        self.analysis_id = analysis_id
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized vecs)
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

    def add(self, chunks: list[str], embeddings: np.ndarray, metadata: list[dict] | None = None):
        """
        Add chunks and their embeddings to the index.

        Args:
            chunks: List of text chunks.
            embeddings: numpy array of shape (len(chunks), dim).
            metadata: Optional list of metadata dicts per chunk.
        """
        if len(chunks) == 0:
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings"
            )

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"chunk_index": i} for i in range(len(chunks))])

        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        """
        Search for the most similar chunks.

        Args:
            query_embedding: numpy array of shape (1, dim).
            k: Number of results to return.

        Returns:
            List of dicts: [{"content": str, "score": float, "metadata": dict}, ...]
        """
        if self.index.ntotal == 0:
            return []

        # Don't request more results than we have vectors
        k = min(k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({
                "content": self.chunks[idx],
                "score": float(scores[0][i]),
                "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
            })

        return results

    def save(self):
        """Persist the FAISS index and chunk data to disk."""
        os.makedirs(FAISS_DIR, exist_ok=True)
        base_path = os.path.join(FAISS_DIR, self.analysis_id)

        # Save FAISS index
        faiss.write_index(self.index, f"{base_path}.index")

        # Save chunks and metadata as JSON
        with open(f"{base_path}.json", "w", encoding="utf-8") as f:
            json.dump({
                "chunks": self.chunks,
                "metadata": self.metadata,
                "dim": self.dim
            }, f, ensure_ascii=False)

        logger.info(f"Saved FAISS index for analysis {self.analysis_id}")

    @classmethod
    def load(cls, analysis_id: str) -> "FAISSStore":
        """Load a persisted FAISS index from disk."""
        base_path = os.path.join(FAISS_DIR, analysis_id)
        index_path = f"{base_path}.index"
        data_path = f"{base_path}.json"

        if not os.path.exists(index_path) or not os.path.exists(data_path):
            raise FileNotFoundError(
                f"No FAISS index found for analysis {analysis_id}"
            )

        # Load FAISS index
        index = faiss.read_index(index_path)

        # Load chunks and metadata
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        store = cls(analysis_id=analysis_id, dim=data.get("dim", 384))
        store.index = index
        store.chunks = data["chunks"]
        store.metadata = data.get("metadata", [])

        logger.info(
            f"Loaded FAISS index for analysis {analysis_id} "
            f"({store.index.ntotal} vectors)"
        )
        return store

    @staticmethod
    def exists(analysis_id: str) -> bool:
        """Check if a FAISS index exists for the given analysis."""
        base_path = os.path.join(FAISS_DIR, analysis_id)
        return (
            os.path.exists(f"{base_path}.index")
            and os.path.exists(f"{base_path}.json")
        )
