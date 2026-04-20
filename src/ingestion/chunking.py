"""
chunking.py
-----------
Splits parsed PDF elements into overlapping text chunks suitable
for embedding and retrieval.

Strategy:
  - Target chunk size : 500–800 tokens  (approximated as characters ÷ 4)
  - Overlap           : 100 tokens      (~400 characters)
  - Boundaries        : Sentence-aware splits on ". ", "\n\n", "\n"

Output format per chunk:
    {
        "chunk_id"   : int,
        "content"    : str,
        "type"       : "text" | "table" | "image",
        "page"       : int,
        "char_count" : int,
    }
"""

from typing import List, Dict, Any
import re

# ---------------------------------------------------------------------------
# Configuration  (characters ≈ tokens × 4)
# ---------------------------------------------------------------------------
CHUNK_SIZE_CHARS   = 2400   # ~600 tokens target
CHUNK_MAX_CHARS    = 3200   # ~800 tokens hard cap
CHUNK_OVERLAP_CHARS = 400   # ~100 tokens overlap

# Sentence/paragraph boundary patterns (ordered by preference)
SPLIT_PATTERNS = ["\n\n", "\n", ". ", "? ", "! ", " "]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_split_point(text: str, target: int) -> int:
    """
    Return the best split index near `target` by searching for a natural
    boundary (paragraph, sentence, word) working backwards from `target`.
    """
    if target >= len(text):
        return len(text)

    for pattern in SPLIT_PATTERNS:
        idx = text.rfind(pattern, 0, target)
        if idx != -1:
            return idx + len(pattern)

    # Fallback: hard cut at target
    return target


def _split_text(text: str) -> List[str]:
    """
    Split a single long string into overlapping chunks respecting
    CHUNK_SIZE_CHARS and CHUNK_OVERLAP_CHARS boundaries.
    """
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = _find_split_point(text, start + CHUNK_SIZE_CHARS)

        # If the chunk is still too large, force-cut at CHUNK_MAX_CHARS
        if end - start > CHUNK_MAX_CHARS:
            end = start + CHUNK_MAX_CHARS

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance with overlap
        start = max(start + 1, end - CHUNK_OVERLAP_CHARS)

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_elements(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Take the flat list of parsed PDF elements and produce overlapping chunks.

    Rules:
      - "text"  elements → split into multiple chunks if large
      - "table" elements → kept as a single chunk (splitting tables breaks them)
      - "image" elements → kept as a single chunk (caption only)

    Args:
        elements: Output of parser.parse_pdf()

    Returns:
        List of chunk dicts with chunk_id assigned.
    """
    print(f"[chunking] Processing {len(elements)} elements ...")
    all_chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for element in elements:
        content = element["content"]
        etype   = element["type"]
        page    = element["page"]

        if etype == "text" and len(content) > CHUNK_SIZE_CHARS:
            # Split long text into overlapping sub-chunks
            sub_chunks = _split_text(content)
            print(
                f"   [chunking] Page {page} text → "
                f"{len(sub_chunks)} chunks ({len(content)} chars)"
            )
            for sub in sub_chunks:
                all_chunks.append({
                    "chunk_id"  : chunk_id,
                    "content"   : sub,
                    "type"      : etype,
                    "page"      : page,
                    "char_count": len(sub),
                })
                chunk_id += 1
        else:
            # Tables and images: keep whole; short texts: keep whole
            all_chunks.append({
                "chunk_id"  : chunk_id,
                "content"   : content.strip(),
                "type"      : etype,
                "page"      : page,
                "char_count": len(content),
            })
            chunk_id += 1

    print(
        f"[chunking] ✓ Created {len(all_chunks)} chunks "
        f"(avg {sum(c['char_count'] for c in all_chunks) // max(len(all_chunks), 1)} chars each)"
    )
    return all_chunks
