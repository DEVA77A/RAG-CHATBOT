"""
WebIntel AI — Section-Aware Text Chunker + Embedding Generator

Chunks text into overlapping segments while preserving heading/section metadata.
Each chunk carries: heading, section_type (intro/body/appendix), page_title, page_url.

Uses sentence-transformers (all-MiniLM-L6-v2) for local, free embeddings.

Model specs:
  - Dimensions: 384
  - Max sequence length: 256 tokens
  - Size: ~80MB
  - Speed: Very fast on CPU
"""

import os
# Force 1 thread for libraries to prevent memory spikes in containers on Render/HuggingFace
if os.environ.get("RENDER") or os.environ.get("SPACE_ID"):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
import re
import numpy as np
import torch
if os.environ.get("RENDER") or os.environ.get("SPACE_ID"):
    torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Global model (loaded once)
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
# Section-Aware Chunking
# ──────────────────────────────────────────────

# Patterns that indicate a heading line
_HEADING_PATTERNS = [
    re.compile(r"^#{1,4}\s+(.+)$"),                # Markdown: # Heading
    re.compile(r"^(.+)\n[=]{3,}$", re.MULTILINE),  # Underline ===
    re.compile(r"^(.+)\n[-]{3,}$", re.MULTILINE),  # Underline ---
]

# Keywords that signal "introductory" sections
_INTRO_KEYWORDS = {
    "introduction", "overview", "getting started", "welcome", "about",
    "quick start", "quickstart", "summary", "home", "index", "what is",
    "tutorial", "guide", "basics", "first steps",
}

# Keywords that signal appendix/reference sections
_APPENDIX_KEYWORDS = {
    "appendix", "reference", "changelog", "license", "credits",
    "glossary", "footnotes", "bibliography",
}


def _detect_section_type(heading: str, page_url: str = "") -> str:
    """Classify section as intro/body/appendix based on heading text and URL."""
    lower_heading = heading.lower().strip()
    lower_url = page_url.lower()

    # Check URL path for clues
    for kw in _INTRO_KEYWORDS:
        if kw in lower_heading or kw in lower_url:
            return "intro"

    for kw in _APPENDIX_KEYWORDS:
        if kw in lower_heading or kw in lower_url:
            return "appendix"

    return "body"


def _extract_heading(line: str) -> tuple[int, str] | None:
    """Try to extract a heading from a line of text, returning (level, text)."""
    stripped = line.strip()
    # Markdown headings
    match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
    if match:
        return len(match.group(1)), match.group(2).strip()
    # Bold-style pseudo-headings (common in scraped text)
    match = re.match(r"^\*\*(.+)\*\*$", stripped)
    if match and len(match.group(1)) < 100:
        return 4, match.group(1).strip()
    return None


def chunk_text_with_metadata(
    text: str,
    page_title: str = "",
    page_url: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[dict]:
    """
    Split text into overlapping chunks while tracking headings and sections.

    Returns list of dicts:
        [{
            "text": str,           # The chunk text
            "heading": str,        # Current heading context
            "section_type": str,   # "intro" | "body" | "appendix"
            "page_title": str,
            "page_url": str,
        }, ...]
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # First pass: split into sections by headings
    lines = text.split("\n")
    sections = []
    
    current_h1 = page_title or "Main Content"
    current_h2 = ""
    current_h3 = ""
    current_heading = current_h1
    current_lines = []

    for line in lines:
        heading_info = _extract_heading(line)
        if heading_info:
            level, heading_text = heading_info
            
            # Flush current section
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append({
                        "heading": current_h1,
                        "sub_heading": current_h2,
                        "sub_sub_heading": current_h3,
                        "text": section_text,
                    })
            
            if level == 1:
                current_h1 = heading_text
                current_h2 = ""
                current_h3 = ""
            elif level == 2:
                current_h2 = heading_text
                current_h3 = ""
            else:
                current_h3 = heading_text
                
            current_heading = heading_text
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append({
                "heading": current_h1,
                "sub_heading": current_h2,
                "sub_sub_heading": current_h3,
                "text": section_text,
            })

    # If no sections found, treat entire text as one section
    if not sections:
        sections = [{"heading": page_title or "Main Content", "text": text}]

    # Second pass: chunk each section with overlap
    all_chunks = []
    for section in sections:
        section_type = _detect_section_type(section["heading"], page_url)
        section_text = section["text"]

        def format_chunk(chunk_content):
            intro = f"[PAGE]\n{page_title or 'Main Content'}\n\n[SECTION]\n{section['heading']}"
            if section.get('sub_heading'):
                intro += f"\n\n[SUB HEADING]\n{section['sub_heading']}"
            if section.get('sub_sub_heading'):
                intro += f"\n\n[SUB SUB HEADING]\n{section['sub_sub_heading']}"
            intro += f"\n\n[CONTENT]\n{chunk_content}"
            return intro

        if len(section_text) <= chunk_size:
            all_chunks.append({
                "text": format_chunk(section_text),
                "heading": section["heading"],
                "section_type": section_type,
                "page_title": page_title,
                "page_url": page_url,
            })
        else:
            # Split into overlapping chunks
            start = 0
            while start < len(section_text):
                end = start + chunk_size

                if end < len(section_text):
                    # Find good break point
                    for sep in ["\n\n", ". ", ".\n", "! ", "? ", "\n"]:
                        brk = section_text.rfind(sep, start, end)
                        if brk > start + chunk_size // 3:
                            end = brk + len(sep)
                            break
                    else:
                        space = section_text.rfind(" ", start, end)
                        if space > start + chunk_size // 3:
                            end = space + 1

                chunk = section_text[start:end].strip()
                if chunk:
                    all_chunks.append({
                        "text": format_chunk(chunk),
                        "heading": section["heading"],
                        "section_type": section_type,
                        "page_title": page_title,
                        "page_url": page_url,
                    })

                start = end - chunk_overlap
                if start <= (end - chunk_size):
                    start = end

    return all_chunks


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[str]:
    """
    Legacy interface — split text into overlapping chunks (plain strings).
    Used by any code that doesn't need metadata.
    """
    chunks = chunk_text_with_metadata(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [c["text"] for c in chunks]


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate normalized embeddings for a list of texts.

    Returns:
        numpy array of shape (len(texts), 384) with L2-normalized vectors.
    """
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)

    model = get_model()
    batch_size = 16 if os.environ.get("RENDER") else 64
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size
    )
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        numpy array of shape (1, 384).
    """
    return embed_texts([query])
