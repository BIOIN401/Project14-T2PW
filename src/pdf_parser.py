"""
pdf_parser.py

Extract text from scientific PDFs (research papers, pathway descriptions)
for use as input to the PWML extraction pipeline.

Supports:
  - Text-based PDFs via pdfplumber (primary) with pypdf fallback
  - Basic scanned/image PDFs via pytesseract OCR (optional, graceful degradation)
  - Section-aware extraction: detects Abstract, Introduction, Results, etc.
  - Page range filtering
  - Metadata extraction (title, authors, DOI)

Usage (standalone):
  python pdf_parser.py --in paper.pdf --out text.txt
  python pdf_parser.py --in paper.pdf --pages 1-10 --out text.txt

Usage (as module):
  from pdf_parser import extract_text_from_pdf, parse_pdf

  # Simple text extraction
  text = extract_text_from_pdf("paper.pdf")

  # Full result with metadata and section info
  result = parse_pdf("paper.pdf", page_start=1, page_end=20)
  print(result["text"])
  print(result["metadata"])
  print(result["sections"])
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Section headers commonly found in scientific papers
SECTION_HEADER_PATTERNS = re.compile(
    r"^\s*(?:\d[\d.]*\.?\s+)?("
    r"abstract|introduction|background|"
    r"materials?\s+and\s+methods?|methods?\s+and\s+materials?|"
    r"experimental\s+procedures?|methods?|"
    r"results?(?:\s+and\s+discussion)?|"
    r"discussion(?:\s+and\s+conclusions?)?|"
    r"conclusions?|summary|"
    r"acknowledgements?|acknowledgments?|"
    r"references?|bibliography|"
    r"supplementary(?:\s+\w+)*|supporting\s+information"
    r")\s*$",
    re.IGNORECASE,
)

# Sections to exclude by default (save tokens, not relevant to pathway extraction)
SKIP_SECTIONS = {"references", "bibliography", "acknowledgements", "acknowledgments"}

# Maximum characters per page to avoid memory issues on large PDFs
MAX_CHARS_PER_PAGE = 8000


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _check_pdfplumber() -> bool:
    try:
        import pdfplumber  # noqa: F401
        return True
    except ImportError:
        return False


def _check_pypdf() -> bool:
    try:
        from pypdf import PdfReader  # noqa: F401
        return True
    except ImportError:
        return False


def _check_tesseract() -> bool:
    """Check if tesseract + pytesseract are available for OCR."""
    try:
        import pytesseract  # noqa: F401
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata_pdfplumber(pdf_path: str) -> Dict[str, str]:
    """Extract PDF metadata using pdfplumber."""
    meta: Dict[str, str] = {}
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            raw = pdf.metadata or {}
            for key in ["Title", "Author", "Subject", "Creator", "Producer", "CreationDate"]:
                val = raw.get(key) or raw.get(key.lower(), "")
                if val and isinstance(val, str):
                    meta[key.lower()] = val.strip()
    except Exception:
        pass
    return meta


def _extract_doi_from_text(text: str) -> str:
    """Try to find a DOI in the first ~2000 characters of text."""
    snippet = text[:2000]
    match = re.search(r"\b(10\.\d{4,}/[^\s\"'<>]+)", snippet)
    return match.group(1).rstrip(".,;)") if match else ""


# ---------------------------------------------------------------------------
# Text extraction strategies
# ---------------------------------------------------------------------------

def _extract_with_pdfplumber(
    pdf_path: str,
    *,
    page_start: int,
    page_end: Optional[int],
) -> Tuple[List[str], int]:
    """
    Extract text page by page using pdfplumber.

    Returns (page_texts, total_pages).
    page_texts is indexed from 0 (page 0 = PDF page 1).
    """
    import pdfplumber

    page_texts: List[str] = []
    total_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        end = min(page_end, total_pages) if page_end else total_pages
        start_idx = max(0, page_start - 1)

        for i, page in enumerate(pdf.pages):
            if i < start_idx:
                page_texts.append("")
                continue
            if i >= end:
                page_texts.append("")
                continue
            try:
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                page_texts.append(text[:MAX_CHARS_PER_PAGE])
            except Exception:
                page_texts.append("")

    return page_texts, total_pages


def _extract_with_pypdf(
    pdf_path: str,
    *,
    page_start: int,
    page_end: Optional[int],
) -> Tuple[List[str], int]:
    """
    Fallback text extraction using pypdf.
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    end = min(page_end, total_pages) if page_end else total_pages
    start_idx = max(0, page_start - 1)

    page_texts: List[str] = []
    for i, page in enumerate(reader.pages):
        if i < start_idx or i >= end:
            page_texts.append("")
            continue
        try:
            text = page.extract_text() or ""
            page_texts.append(text[:MAX_CHARS_PER_PAGE])
        except Exception:
            page_texts.append("")

    return page_texts, total_pages


def _ocr_page(pdf_path: str, page_number: int, dpi: int = 150) -> str:
    """
    Rasterize a single PDF page and run OCR via pytesseract.
    page_number is 1-based.
    """
    import tempfile
    import os

    try:
        import pytesseract
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "page")
            result = subprocess.run(
                [
                    "pdftoppm",
                    "-jpeg",
                    "-r", str(dpi),
                    "-f", str(page_number),
                    "-l", str(page_number),
                    pdf_path,
                    prefix,
                ],
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                return ""
            # Find the generated file
            images = sorted(Path(tmpdir).glob("page-*.jpg")) + sorted(Path(tmpdir).glob("page-*.jpeg"))
            if not images:
                return ""
            img = Image.open(str(images[0]))
            text = pytesseract.image_to_string(img, lang="eng")
            return text[:MAX_CHARS_PER_PAGE]
    except Exception:
        return ""


def _is_mostly_empty(page_texts: List[str], threshold: float = 0.6) -> bool:
    """Return True if more than `threshold` fraction of pages have < 50 chars of text."""
    if not page_texts:
        return True
    non_empty = [t for t in page_texts if t and len(t.strip()) >= 50]
    return (len(non_empty) / len(page_texts)) < (1.0 - threshold)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def _detect_sections(page_texts: List[str]) -> Dict[str, List[int]]:
    """
    Return a map of section_name -> [page_indices] using the combined text.
    """
    sections: Dict[str, List[int]] = {}
    current = "preamble"
    sections[current] = []

    for page_idx, text in enumerate(page_texts):
        if not text:
            continue
        for line in text.splitlines():
            m = SECTION_HEADER_PATTERNS.match(line)
            if m:
                current = m.group(1).strip().lower()
                current = re.sub(r"\s+", " ", current)
                sections.setdefault(current, [])
        sections.setdefault(current, []).append(page_idx)

    return sections


def _join_page_texts(
    page_texts: List[str],
    *,
    skip_sections: Optional[set] = None,
    section_map: Optional[Dict[str, List[int]]] = None,
) -> str:
    """Join page texts, optionally skipping pages that belong to excluded sections."""
    skip = skip_sections or set()
    if not skip or section_map is None:
        return "\n\n".join(t for t in page_texts if t.strip())

    skipped_pages: set = set()
    for section_name, pages in section_map.items():
        if any(section_name.startswith(s) for s in skip):
            skipped_pages.update(pages)

    parts = []
    for i, text in enumerate(page_texts):
        if i in skipped_pages or not text.strip():
            continue
        parts.append(text)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    pdf_path: str,
    *,
    page_start: int = 1,
    page_end: Optional[int] = None,
    skip_sections: Optional[set] = None,
    enable_ocr_fallback: bool = True,
    ocr_dpi: int = 150,
) -> str:
    """
    Extract plain text from a PDF file.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    page_start : int
        First page to extract (1-based, inclusive). Default 1.
    page_end : int or None
        Last page to extract (1-based, inclusive). None means all pages.
    skip_sections : set or None
        Section names to skip (e.g. {"references", "acknowledgements"}).
        Defaults to SKIP_SECTIONS constant.
    enable_ocr_fallback : bool
        If True and text extraction yields mostly empty pages, attempt OCR
        on those pages (requires pytesseract + tesseract installed).
    ocr_dpi : int
        DPI for OCR rasterization when fallback is triggered.

    Returns
    -------
    str
        Extracted text, ready for the pipeline.
    """
    result = parse_pdf(
        pdf_path,
        page_start=page_start,
        page_end=page_end,
        skip_sections=skip_sections,
        enable_ocr_fallback=enable_ocr_fallback,
        ocr_dpi=ocr_dpi,
    )
    return result["text"]


def parse_pdf(
    pdf_path: str,
    *,
    page_start: int = 1,
    page_end: Optional[int] = None,
    skip_sections: Optional[set] = None,
    enable_ocr_fallback: bool = True,
    ocr_dpi: int = 150,
) -> Dict[str, Any]:
    """
    Full PDF parse returning text + metadata + diagnostics.

    Returns
    -------
    dict with keys:
        text        — extracted text string
        metadata    — {title, author, doi, ...}
        total_pages — total pages in PDF
        pages_used  — number of pages included in output
        sections    — {section_name: [page_indices]}
        method      — 'pdfplumber' | 'pypdf' | 'ocr' | 'mixed'
        warnings    — list of warning strings
        error       — error string if extraction failed, else ''
    """
    skip = skip_sections if skip_sections is not None else SKIP_SECTIONS
    path = str(Path(pdf_path).expanduser().resolve())
    result: Dict[str, Any] = {
        "text": "",
        "metadata": {},
        "total_pages": 0,
        "pages_used": 0,
        "sections": {},
        "method": "",
        "warnings": [],
        "error": "",
    }

    if not Path(path).exists():
        result["error"] = f"File not found: {path}"
        return result

    # --- Metadata ---
    result["metadata"] = _extract_metadata_pdfplumber(path)

    # --- Text extraction ---
    page_texts: List[str] = []
    total_pages = 0
    method = "unknown"

    has_pdfplumber = _check_pdfplumber()
    has_pypdf = _check_pypdf()

    if has_pdfplumber:
        try:
            page_texts, total_pages = _extract_with_pdfplumber(
                path, page_start=page_start, page_end=page_end
            )
            method = "pdfplumber"
        except Exception as exc:
            result["warnings"].append(f"pdfplumber failed: {exc}")
            has_pdfplumber = False

    if not has_pdfplumber and has_pypdf:
        try:
            page_texts, total_pages = _extract_with_pypdf(
                path, page_start=page_start, page_end=page_end
            )
            method = "pypdf"
        except Exception as exc:
            result["error"] = f"Text extraction failed: {exc}"
            return result

    if not has_pdfplumber and not has_pypdf:
        result["error"] = (
            "No PDF reading library available. "
            "Install pdfplumber (`pip install pdfplumber`) or pypdf (`pip install pypdf`)."
        )
        return result

    result["total_pages"] = total_pages

    # --- OCR fallback ---
    if enable_ocr_fallback and _is_mostly_empty(page_texts):
        has_ocr = _check_tesseract()
        if has_ocr:
            result["warnings"].append(
                "Text extraction yielded mostly empty pages; falling back to OCR."
            )
            ocr_texts: List[str] = []
            end = min(page_end, total_pages) if page_end else total_pages
            for page_num in range(page_start, end + 1):
                ocr_texts.append(_ocr_page(path, page_num, dpi=ocr_dpi))
            # Pad to same length as page_texts
            page_texts = ([""] * (page_start - 1)) + ocr_texts
            method = "ocr" if method == "unknown" else "mixed"
        else:
            result["warnings"].append(
                "Text extraction yielded mostly empty pages (possibly scanned PDF). "
                "Install tesseract + pytesseract for OCR support."
            )

    # --- Section detection ---
    section_map = _detect_sections(page_texts)
    result["sections"] = section_map

    # --- Build final text ---
    text = _join_page_texts(page_texts, skip_sections=skip, section_map=section_map)

    # Try to find DOI if not in metadata
    if not result["metadata"].get("doi"):
        doi = _extract_doi_from_text(text)
        if doi:
            result["metadata"]["doi"] = doi

    pages_used = sum(1 for t in page_texts if t.strip())
    result["text"] = text
    result["pages_used"] = pages_used
    result["method"] = method

    return result


def get_pdf_info(pdf_path: str) -> Dict[str, Any]:
    """
    Return quick diagnostic info about a PDF without full extraction.
    Useful for the UI to show page count, estimated extractability, etc.
    """
    info: Dict[str, Any] = {
        "exists": False,
        "total_pages": 0,
        "file_size_kb": 0,
        "has_extractable_text": False,
        "metadata": {},
        "error": "",
    }

    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        info["error"] = f"File not found: {pdf_path}"
        return info

    info["exists"] = True
    info["file_size_kb"] = round(path.stat().st_size / 1024, 1)

    has_pdfplumber = _check_pdfplumber()
    has_pypdf = _check_pypdf()

    if has_pdfplumber:
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                info["total_pages"] = len(pdf.pages)
                info["metadata"] = pdf.metadata or {}
                # Check first page for text
                if pdf.pages:
                    sample = pdf.pages[0].extract_text() or ""
                    info["has_extractable_text"] = len(sample.strip()) > 50
        except Exception as exc:
            info["error"] = str(exc)
    elif has_pypdf:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            info["total_pages"] = len(reader.pages)
            if reader.pages:
                sample = reader.pages[0].extract_text() or ""
                info["has_extractable_text"] = len(sample.strip()) > 50
        except Exception as exc:
            info["error"] = str(exc)
    else:
        info["error"] = "No PDF library available (install pdfplumber or pypdf)."

    return info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF for use as PWML pipeline input."
    )
    parser.add_argument("--in", dest="pdf_path", required=True, help="Input PDF file")
    parser.add_argument(
        "--out",
        dest="out_path",
        default="",
        help="Output text file path (prints to stdout if omitted)",
    )
    parser.add_argument(
        "--pages",
        dest="pages",
        default="",
        help="Page range, e.g. '1-10' or '3' (default: all pages)",
    )
    parser.add_argument(
        "--no-skip-refs",
        action="store_true",
        help="Include References/Acknowledgements sections (skipped by default)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback for scanned PDFs",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print PDF info only (page count, metadata) without extracting text",
    )
    args = parser.parse_args()

    if args.info:
        info = get_pdf_info(args.pdf_path)
        import json
        print(json.dumps(info, indent=2, default=str))
        return

    page_start = 1
    page_end = None
    if args.pages:
        parts = args.pages.strip().split("-")
        try:
            page_start = int(parts[0])
            page_end = int(parts[1]) if len(parts) > 1 else page_start
        except ValueError:
            print(f"Invalid --pages value: {args.pages}", file=sys.stderr)
            sys.exit(1)

    skip = set() if args.no_skip_refs else SKIP_SECTIONS

    result = parse_pdf(
        args.pdf_path,
        page_start=page_start,
        page_end=page_end,
        skip_sections=skip,
        enable_ocr_fallback=not args.no_ocr,
    )

    if result["error"]:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if result["warnings"]:
        for w in result["warnings"]:
            print(f"WARNING: {w}", file=sys.stderr)

    print(
        f"Extracted {result['pages_used']}/{result['total_pages']} pages "
        f"via {result['method']}. "
        f"Sections: {list(result['sections'].keys())}",
        file=sys.stderr,
    )
    if result["metadata"]:
        print(f"Metadata: {result['metadata']}", file=sys.stderr)

    text = result["text"]
    if args.out_path:
        Path(args.out_path).write_text(text, encoding="utf-8")
        print(f"Wrote extracted text to: {args.out_path}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
