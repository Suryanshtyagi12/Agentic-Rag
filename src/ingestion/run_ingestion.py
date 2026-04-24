"""
run_ingestion.py
----------------
CLI entry-point for the full PDF ingestion pipeline.

Usage (from project root with venv active):
    python src/ingestion/run_ingestion.py <path_to_pdf>

Example:
    python src/ingestion/run_ingestion.py data/raw_pdfs/my_document.pdf

Output:
    data/processed/<pdf_stem>_chunks.json

CHANGES (v2 - Optimized):
  - Added file-hash-based caching: if the same PDF was previously processed,
    the cached JSON is loaded from data/processed/ and ingestion is skipped
  - Added per-stage timing logs (load, parse, chunk, save)
  - Total pipeline time reported at the end
"""

import sys
import os
import json
import time
import hashlib
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ingestion.loader   import load_pdf
from src.ingestion.parser   import parse_pdf
from src.ingestion.chunking import chunk_elements

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _file_hash(path: Path, algo: str = "md5") -> str:
    """
    Compute a hex digest of the file contents.
    Used to detect whether a PDF has already been processed.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path_for(pdf_path: Path) -> Path:
    """
    Return the expected processed-JSON path for a given PDF.
    Embeds the file hash in the filename to invalidate cache on content change.
    """
    pdf_hash = _file_hash(pdf_path)[:8]          # first 8 hex chars is enough
    stem = f"{pdf_path.stem}_{pdf_hash}_chunks"
    return PROCESSED_DIR / f"{stem}.json"


def _load_from_cache(cache_file: Path):
    """Load and return chunks from a previously saved JSON cache file."""
    print(f"[cache] Cache hit! Loading from: {cache_file.name}")
    t = time.time()
    with open(cache_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[cache] Loaded {len(chunks)} chunks in {time.time() - t:.2f}s")
    return chunks


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(pdf_path: str) -> Path:
    """
    Execute the full ingestion pipeline for a single PDF.

    Steps:
        0. Cache check -- return immediately if already processed
        1. Load        -- validate and open the PDF
        2. Parse       -- extract text (+ optionally tables/images)
        3. Chunk       -- split into overlapping character-based chunks
        4. Save        -- write JSON to data/processed/

    Args:
        pdf_path: Path to the input PDF (relative or absolute).

    Returns:
        Path to the saved (or cached) output JSON file.
    """
    print("=" * 60)
    print("  AgenticRag -- PDF Ingestion Pipeline (v2 Optimized)")
    print("=" * 60)
    t_pipeline_start = time.time()

    # ── Step 1: Load ─────────────────────────────────────────────────
    print("\n[Step 1/3] Loading PDF ...")
    t0 = time.time()
    path = load_pdf(pdf_path)
    print(f"           -> Done in {time.time() - t0:.2f}s")

    # ── Step 0: Cache check ───────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path_for(path)

    if cache_file.exists():
        chunks = _load_from_cache(cache_file)
        t_total = time.time() - t_pipeline_start
        print("\n" + "=" * 60)
        print(f"  [OK] Pipeline complete (cache) in {t_total:.2f}s")
        print(f"  Output  : {cache_file}")
        print(f"  Chunks  : {len(chunks)}")
        print("=" * 60)
        return cache_file

    # ── Step 2: Parse ─────────────────────────────────────────────────
    print("\n[Step 2/3] Parsing PDF ...")
    t0 = time.time()
    elements = parse_pdf(path)
    print(f"           -> {len(elements)} elements extracted in {time.time() - t0:.2f}s")

    # ── Step 3: Chunk ─────────────────────────────────────────────────
    print("\n[Step 3/3] Chunking content ...")
    t0 = time.time()
    chunks = chunk_elements(elements)
    print(f"           -> {len(chunks)} chunks created in {time.time() - t0:.2f}s")

    # ── Save ──────────────────────────────────────────────────────────
    t0 = time.time()
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n[save] Written to {cache_file.name} in {time.time() - t0:.2f}s")

    t_total = time.time() - t_pipeline_start

    print("\n" + "=" * 60)
    print(f"  [OK] Pipeline complete in {t_total:.2f}s")
    print(f"  Output  : {cache_file}")
    print(f"  Chunks  : {len(chunks)}")
    print(f"  Elements: {len(elements)}")
    print("=" * 60)

    return cache_file


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/ingestion/run_ingestion.py <path_to_pdf>")
        print("Example: python src/ingestion/run_ingestion.py data/raw_pdfs/sample.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        output = run_pipeline(pdf_path)
        print(f"\n[[OK]] Saved to: {output}")
    except FileNotFoundError as e:
        print(f"\n[[FAIL]] File error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[[FAIL]] Value error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[[FAIL]] Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
