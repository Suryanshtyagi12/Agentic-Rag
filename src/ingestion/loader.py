"""
loader.py
---------
Responsible for loading a PDF file from disk and validating it
before passing it downstream to the parser.
"""

import os
from pathlib import Path


def load_pdf(pdf_path: str) -> Path:
    """
    Validate and return a resolved Path object for the given PDF.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Resolved pathlib.Path to the PDF.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file is not a PDF.
    """
    path = Path(pdf_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"[loader] PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"[loader] Expected a .pdf file, got: {path.suffix}")

    file_size_kb = path.stat().st_size / 1024
    print(f"[loader] [OK] Loaded: {path.name} ({file_size_kb:.1f} KB)")

    return path
