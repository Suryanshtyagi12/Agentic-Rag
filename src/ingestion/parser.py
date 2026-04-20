"""
parser.py
---------
Extracts text, tables, and images from a PDF using:
  - PyMuPDF  (fitz) → text & image extraction
  - pdfplumber       → table extraction

Output format per element:
    {
        "content": str,
        "type": "text" | "table" | "image",
        "page": int        (1-indexed)
    }
"""

from pathlib import Path
from typing import List, Dict, Any

import fitz          # PyMuPDF
import pdfplumber


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_text(page_fitz, page_num: int) -> List[Dict[str, Any]]:
    """Extract raw text blocks from a PyMuPDF page."""
    elements = []
    text = page_fitz.get_text("text").strip()
    if text:
        elements.append({
            "content": text,
            "type": "text",
            "page": page_num,
        })
    return elements


def _table_to_markdown(table: list) -> str:
    """
    Convert a pdfplumber table (list of lists) into a markdown-style
    plain-text string so it flows naturally into a text chunk.
    """
    if not table:
        return ""

    rows = []
    for i, row in enumerate(table):
        cleaned = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append(" | ".join(cleaned))
        if i == 0:
            # header separator
            rows.append("-" * len(rows[0]))

    return "\n".join(rows)


def _extract_tables(page_plumber, page_num: int) -> List[Dict[str, Any]]:
    """Extract tables from a pdfplumber page and convert to structured text."""
    elements = []
    tables = page_plumber.extract_tables()
    for idx, table in enumerate(tables, start=1):
        md_table = _table_to_markdown(table)
        if md_table.strip():
            print(f"   [parser] Table {idx} found on page {page_num}")
            elements.append({
                "content": md_table,
                "type": "table",
                "page": page_num,
            })
    return elements


def _extract_images(page_fitz, page_num: int) -> List[Dict[str, Any]]:
    """
    Detect images on a PyMuPDF page and generate placeholder captions.
    Full image bytes are available via page_fitz.get_pixmap() if needed later.
    """
    elements = []
    image_list = page_fitz.get_images(full=True)
    for idx, img in enumerate(image_list, start=1):
        xref = img[0]
        width = img[2]
        height = img[3]
        color_space = img[5] if len(img) > 5 else "unknown"
        caption = (
            f"[Image {idx} on page {page_num}] "
            f"Dimensions: {width}x{height}px, "
            f"Color space: {color_space}. "
            f"(Visual content — refer to original PDF for details.)"
        )
        print(f"   [parser] Image {idx} found on page {page_num} ({width}x{height})")
        elements.append({
            "content": caption,
            "type": "image",
            "page": page_num,
        })
    return elements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a PDF and return a flat list of content elements (text / table / image).

    Args:
        pdf_path: Resolved pathlib.Path to the PDF file.

    Returns:
        List of dicts: [{"content": str, "type": str, "page": int}, ...]
    """
    elements: List[Dict[str, Any]] = []

    print(f"[parser] Opening: {pdf_path.name}")

    fitz_doc = fitz.open(str(pdf_path))
    total_pages = len(fitz_doc)
    print(f"[parser] Total pages: {total_pages}")

    with pdfplumber.open(str(pdf_path)) as plumber_doc:
        for page_num in range(total_pages):
            human_page = page_num + 1
            print(f"[parser] Processing page {human_page}/{total_pages} ...")

            fitz_page = fitz_doc[page_num]
            plumber_page = plumber_doc.pages[page_num]

            # 1. Text
            elements.extend(_extract_text(fitz_page, human_page))

            # 2. Tables (pdfplumber is superior for structured tables)
            elements.extend(_extract_tables(plumber_page, human_page))

            # 3. Images
            elements.extend(_extract_images(fitz_page, human_page))

    fitz_doc.close()

    text_count  = sum(1 for e in elements if e["type"] == "text")
    table_count = sum(1 for e in elements if e["type"] == "table")
    image_count = sum(1 for e in elements if e["type"] == "image")

    print(
        f"[parser] ✓ Extraction complete — "
        f"{text_count} text blocks, {table_count} tables, {image_count} images"
    )
    return elements
