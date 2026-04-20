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
"""

import sys
import os
import json
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ingestion.loader   import load_pdf
from src.ingestion.parser   import parse_pdf
from src.ingestion.chunking import chunk_elements

# Processed output directory (relative to project root)
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def run_pipeline(pdf_path: str) -> Path:
    """
    Execute the full ingestion pipeline for a single PDF.

    Steps:
        1. Load    — validate and open the PDF
        2. Parse   — extract text, tables, images
        3. Chunk   — split into overlapping chunks
        4. Save    — write JSON to data/processed/

    Args:
        pdf_path: Path to the input PDF (relative or absolute).

    Returns:
        Path to the saved output JSON file.
    """
    print("=" * 60)
    print("  AgenticRag — PDF Ingestion Pipeline")
    print("=" * 60)
    t_start = time.time()

    # ── Step 1: Load ────────────────────────────────────────────
    print("\n[Step 1/3] Loading PDF ...")
    path = load_pdf(pdf_path)

    # ── Step 2: Parse ───────────────────────────────────────────
    print("\n[Step 2/3] Parsing PDF ...")
    elements = parse_pdf(path)
    print(f"           → {len(elements)} elements extracted")

    # ── Step 3: Chunk ───────────────────────────────────────────
    print("\n[Step 3/3] Chunking content ...")
    chunks = chunk_elements(elements)
    print(f"           → {len(chunks)} chunks created")

    # ── Save ─────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DIR / f"{path.stem}_chunks.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    t_elapsed = time.time() - t_start

    print("\n" + "=" * 60)
    print(f"  ✓ Pipeline complete in {t_elapsed:.2f}s")
    print(f"  Output  : {output_file}")
    print(f"  Chunks  : {len(chunks)}")
    print(f"  Elements: {len(elements)}")
    print("=" * 60)

    return output_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/ingestion/run_ingestion.py <path_to_pdf>")
        print("Example: python src/ingestion/run_ingestion.py data/raw_pdfs/sample.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        output = run_pipeline(pdf_path)
        print(f"\n[✓] Saved to: {output}")
    except FileNotFoundError as e:
        print(f"\n[✗] File error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[✗] Value error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[✗] Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
