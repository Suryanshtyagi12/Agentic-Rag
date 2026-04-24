"""
parser.py
---------
Extracts text, tables, and images from a PDF.

CHANGES (v2 - Optimized):
  - Text extraction: PyMuPDF ONLY (removed redundant pdfplumber text pass)
  - Table extraction: OPTIONAL via ENABLE_TABLES flag (default: False)
    - Only runs on pages that contain table-like structures
    - Limits extraction to MAX_TABLES_PER_DOC per document
  - Image handling: OCR disabled by default (ENABLE_OCR = False)
    - Replaced with lightweight placeholder string (no pytesseract overhead)
  - Parallel processing: pages processed via ThreadPoolExecutor (max 4 workers)
  - Timing logs added for ingestion profiling

Output format per element:
    {
        "content": str,
        "type": "text" | "table" | "image",
        "page": int        (1-indexed)
    }
"""

import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF only — pdfplumber removed for text extraction

# ---------------------------------------------------------------------------
# Feature Flags
# ---------------------------------------------------------------------------
ENABLE_TABLES       = False   # Set True to enable pdfplumber table extraction
ENABLE_OCR          = False   # Set True to enable OCR (very slow — off by default)
MAX_TABLES_PER_DOC  = 10      # Safety cap on total tables extracted per document
MAX_WORKERS         = 4       # Parallel page processing workers


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _page_has_table_structure(page_fitz) -> bool:
    """
    Lightweight heuristic: check if a page likely contains a table
    by looking for many horizontal/vertical lines in the drawing commands.
    Avoids opening pdfplumber on every page.
    """
    paths = page_fitz.get_drawings()
    # Tables tend to produce many rectangular strokes
    rect_count = sum(1 for p in paths if p.get("type") == "re")
    line_count  = sum(1 for p in paths if p.get("type") == "l")
    return (rect_count + line_count) >= 6


def _extract_text(page_fitz, page_num: int) -> List[Dict[str, Any]]:
    """Extract raw text from a PyMuPDF page (fast — no pdfplumber needed)."""
    elements = []
    text = page_fitz.get_text("text").strip()
    if text:
        elements.append({
            "content": text,
            "type":    "text",
            "page":    page_num,
        })
    return elements


def _table_to_markdown(table: list) -> str:
    """Convert a pdfplumber table (list of lists) to plain markdown string."""
    if not table:
        return ""
    rows = []
    for i, row in enumerate(table):
        cleaned = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append(" | ".join(cleaned))
        if i == 0:
            rows.append("-" * len(rows[0]))
    return "\n".join(rows)


def _extract_tables_plumber(pdf_path: str, page_num: int,
                             human_page: int) -> List[Dict[str, Any]]:
    """
    Extract tables from a specific page using pdfplumber (imported lazily
    so it's never loaded when ENABLE_TABLES=False).
    """
    import pdfplumber  # lazy import — only loaded when tables are enabled
    elements = []
    with pdfplumber.open(pdf_path) as plumber_doc:
        page = plumber_doc.pages[page_num]
        tables = page.extract_tables()
        for idx, table in enumerate(tables, start=1):
            md_table = _table_to_markdown(table)
            if md_table.strip():
                print(f"   [parser] Table {idx} found on page {human_page}")
                elements.append({
                    "content": md_table,
                    "type":    "table",
                    "page":    human_page,
                })
    return elements


def _extract_images(page_fitz, page_num: int) -> List[Dict[str, Any]]:
    """
    Handle images on a page.
    - ENABLE_OCR=False (default): emit a lightweight placeholder string.
    - ENABLE_OCR=True           : run pytesseract OCR on the image pixmap.
    """
    elements = []
    image_list = page_fitz.get_images(full=True)

    for idx, img in enumerate(image_list, start=1):
        if not ENABLE_OCR:
            # Fast path: simple human-readable placeholder (no OCR overhead)
            caption = f"Image detected on page {page_num}"
        else:
            # Slow path: rasterise and OCR the image
            try:
                import pytesseract
                from PIL import Image
                import io

                xref   = img[0]
                base   = page_fitz.parent.extract_image(xref)
                pil_img = Image.open(io.BytesIO(base["image"]))
                ocr_text = pytesseract.image_to_string(pil_img).strip()
                caption = ocr_text if ocr_text else f"Image detected on page {page_num}"
            except Exception as e:
                caption = f"Image detected on page {page_num} (OCR failed: {e})"

        elements.append({
            "content": caption,
            "type":    "image",
            "page":    page_num,
        })
    return elements


def _process_page(args: tuple) -> List[Dict[str, Any]]:
    """
    Worker function — processes a single PDF page.
    Designed to be called inside a ThreadPoolExecutor.

    Args:
        args: (fitz_doc_path, page_num, total_pages, table_budget)
              table_budget is a mutable list [remaining_tables] used as a
              cross-thread counter approximation.
    """
    pdf_path_str, page_num, total_pages, table_budget = args
    human_page = page_num + 1
    page_elements: List[Dict[str, Any]] = []

    # Open a per-thread fitz document (fitz docs are not thread-safe)
    fitz_doc = fitz.open(pdf_path_str)
    fitz_page = fitz_doc[page_num]

    # 1. Text (PyMuPDF only — fast)
    page_elements.extend(_extract_text(fitz_page, human_page))

    # 2. Tables (optional + heuristic-gated)
    if ENABLE_TABLES and table_budget[0] > 0 and _page_has_table_structure(fitz_page):
        tables = _extract_tables_plumber(pdf_path_str, page_num, human_page)
        if tables:
            count = len(tables)
            table_budget[0] = max(0, table_budget[0] - count)
            page_elements.extend(tables)

    # 3. Images (placeholder or OCR)
    page_elements.extend(_extract_images(fitz_page, human_page))

    fitz_doc.close()
    return page_elements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a PDF and return a flat list of content elements.

    Uses parallel page processing (up to MAX_WORKERS threads) for speed.

    Args:
        pdf_path: Resolved pathlib.Path to the PDF file.

    Returns:
        List of dicts: [{"content": str, "type": str, "page": int}, ...]
    """
    t_start = time.time()
    pdf_path_str = str(pdf_path)

    print(f"[parser] Opening: {pdf_path.name}")
    print(f"[parser] Flags — ENABLE_TABLES={ENABLE_TABLES}, ENABLE_OCR={ENABLE_OCR}, "
          f"MAX_WORKERS={MAX_WORKERS}")

    # Quick page count (cheap)
    with fitz.open(pdf_path_str) as probe:
        total_pages = len(probe)
    print(f"[parser] Total pages: {total_pages}")

    # Shared table budget (mutable list acts as a pass-by-reference int)
    table_budget = [MAX_TABLES_PER_DOC]

    # Build per-page argument tuples
    page_args = [
        (pdf_path_str, page_num, total_pages, table_budget)
        for page_num in range(total_pages)
    ]

    # ── Parallel extraction ──────────────────────────────────────────────────
    # Results keyed by page_num so we can re-sort after parallel execution
    results: Dict[int, List[Dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_process_page, args): args[1]   # page_num
            for args in page_args
        }
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                results[page_num] = future.result()
            except Exception as exc:
                print(f"   [parser] WARNING: Page {page_num + 1} failed: {exc}")
                results[page_num] = []

    # Reassemble in page order
    elements: List[Dict[str, Any]] = []
    for page_num in range(total_pages):
        elements.extend(results.get(page_num, []))

    t_elapsed = time.time() - t_start

    text_count  = sum(1 for e in elements if e["type"] == "text")
    table_count = sum(1 for e in elements if e["type"] == "table")
    image_count = sum(1 for e in elements if e["type"] == "image")

    print(
        f"[parser] [OK] Extraction complete in {t_elapsed:.2f}s — "
        f"{text_count} text blocks, {table_count} tables, {image_count} images"
    )
    return elements
