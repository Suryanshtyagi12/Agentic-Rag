"""
chunking.py
-----------
Splits parsed PDF elements into overlapping text chunks suitable
for embedding and retrieval.

CHANGES (v2 - Optimized):
  - Replaced token-approximation chunking (slow) with pure character-based splitting
  - chunk_size  = 800 chars  (was 2400 — smaller → faster embed, better recall)
  - overlap     = 100 chars  (was 400)
  - Removed heavy regex SPLIT_PATTERNS in favour of a fast str.rfind loop
  - Added per-stage timing logs (chunking time)
  - No external tokenizer dependency

Output format per chunk:
    {
        "chunk_id"   : int,
        "content"    : str,
        "type"       : "text" | "table" | "image",
        "page"       : int,
        "char_count" : int,
    }
"""

import time
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Configuration — pure character counts (no tokenizer needed)
# ---------------------------------------------------------------------------
CHUNK_SIZE_CHARS    = 800    # target chunk size in characters
CHUNK_OVERLAP_CHARS = 100    # overlap between consecutive chunks

# Natural boundary characters (searched in descending preference)
_BOUNDARIES = ("\n\n", "\n", ". ", "? ", "! ", " ")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_split_point(text: str, target: int) -> int:
    """
    Return the best split index at or before `target` by searching for a
    natural boundary working backwards — paragraph, sentence, word.
    Falls back to a hard cut if no boundary is found.
    """
    if target >= len(text):
        return len(text)

    for boundary in _BOUNDARIES:
        idx = text.rfind(boundary, 0, target)
        if idx != -1:
            return idx + len(boundary)

    # Hard cut fallback
    return target


def _split_text(text: str) -> List[str]:
    """
    Split a single string into overlapping fixed-size character chunks.
    Pure Python — no tokenizer, no regex, no heavy dependencies.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start  = 0

    while start < len(text):
        end   = _find_split_point(text, start + CHUNK_SIZE_CHARS)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Advance with overlap
        next_start = end - CHUNK_OVERLAP_CHARS
        start = next_start if next_start > start else start + 1

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_elements(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Take the flat list of parsed PDF elements and produce overlapping chunks.

    Rules:
      - "text"  elements → split into multiple chunks if > CHUNK_SIZE_CHARS
      - "table" elements → kept as a single chunk (splitting tables breaks them)
      - "image" elements → kept as a single chunk (caption only)

    Args:
        elements: Output of parser.parse_pdf()

    Returns:
        List of chunk dicts with chunk_id assigned.
    """
    t_start = time.time()
    print(f"[chunking] Processing {len(elements)} elements "
          f"(chunk_size={CHUNK_SIZE_CHARS}, overlap={CHUNK_OVERLAP_CHARS}) ...")

    all_chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for element in elements:
        content = element["content"]
        etype   = element["type"]
        page    = element["page"]

        if etype == "text" and len(content) > CHUNK_SIZE_CHARS:
            # Split long text into overlapping sub-chunks
            sub_chunks = _split_text(content)
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

    t_elapsed = time.time() - t_start
    avg_chars = (
        sum(c["char_count"] for c in all_chunks) // max(len(all_chunks), 1)
    )

    print(
        f"[chunking] [OK] Created {len(all_chunks)} chunks "
        f"(avg {avg_chars} chars each) in {t_elapsed:.2f}s"
    )
    return all_chunks
