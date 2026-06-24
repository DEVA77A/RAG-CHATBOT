"""
WebIntel AI — Text Chunker + Embedding Generator

Uses sentence-transformers (all-MiniLM-L6-v2) for local, free embeddings.
Chunks text with overlap for better retrieval context.

Model specs:
  - Dimensions: 384
  - Max sequence length: 256 tokens
  - Size: ~80MB
  - Speed: Very fast on CPU
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Global model (loaded once at import time)
# ──────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _model


# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[str]:
    """
    Split text into overlapping chunks by character count.

    Tries to split on paragraph boundaries first, then sentence boundaries,
    then falls back to character-level splitting.

    Args:
        text: The full text to chunk.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # If text is short enough, return as a single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to find a good break point
        if end < len(text):
            # Try to break at paragraph boundary
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break + 2  # Include the newlines

            else:
                # Try to break at sentence boundary
                for sep in [". ", ".\n", "! ", "? ", "\n"]:
                    sent_break = text.rfind(sep, start, end)
                    if sent_break > start + chunk_size // 3:
                        end = sent_break + len(sep)
                        break

                else:
                    # Try to break at word boundary
                    space_break = text.rfind(" ", start, end)
                    if space_break > start + chunk_size // 3:
                        end = space_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward, accounting for overlap
        start = end - chunk_overlap
        if start <= (end - chunk_size):
            # Prevent infinite loop if overlap >= chunk_size
            start = end

    return chunks


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate normalized embeddings for a list of texts.

    Args:
        texts: List of strings to embed.

    Returns:
        numpy array of shape (len(texts), 384) with L2-normalized vectors.
    """
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)

    model = get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32
    )
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        numpy array of shape (1, 384).
    """
    return embed_texts([query])
