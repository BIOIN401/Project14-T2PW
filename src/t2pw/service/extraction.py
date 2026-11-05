"""The "Run pipeline" handler, moved out of ``streamlit_app.py`` step by step (R3).

R3a moved the input assembly: the pasted texts and the uploaded PDFs become the one
``text`` the stages read. R3b-R3d extend this module with the stages themselves.

Every message goes through the ``reporter`` the caller passes (the app passes
``StreamlitReporter(st)``). There is no default: service code never falls back to
Streamlit silently (R2 review F1). Where the app used to call ``st.stop()`` the
function returns a :class:`InputFailure` instead, and the app calls ``st.stop()``.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from t2pw.extraction.pdf_parser import SKIP_SECTIONS, parse_pdf
from t2pw.service.reporter import Reporter

# Failure kinds, one per former ``st.stop()`` in the input-assembly code.
KIND_NO_PDF = "no_pdf_uploaded"
KIND_PDF_FAILED = "pdf_extraction_failed"
KIND_NO_TEXT = "no_text"


@dataclass(frozen=True)
class InputFailure:
    kind: str
    message: str


@dataclass(frozen=True)
class AssembledInputs:
    """``text`` for the stages, or the ``failure`` where the app stops the run."""

    text: str = ""
    failure: Optional[InputFailure] = None


def assemble_inputs(
    text_entries: Iterable[str],
    pdfs: Sequence[Tuple[str, bytes]],
    *,
    uses_pdf_input: bool,
    input_mode: str,
    pdf_page_range: Optional[str],
    pdf_skip_refs: bool,
    pdf_ocr: bool,
    reporter: Reporter,
) -> AssembledInputs:
    """Join the pasted texts and the PDFs' text exactly as the app's submit block did.

    ``pdfs`` are ``(file name, file bytes)`` pairs in upload order. Texts come first,
    then each PDF; parts are stripped, empty ones dropped, and joined with a blank line.
    """

    text_parts = [entry.strip() for entry in text_entries if entry.strip()]

    # PDF extraction runs here — outside the form, so uploaded_pdf is accessible
    if uses_pdf_input:
        if not pdfs:
            if input_mode == "Upload PDF":
                message = "Please upload one or more PDFs using the file uploader above."
                reporter.warning(message)
                return AssembledInputs(failure=InputFailure(KIND_NO_PDF, message))
            pdfs = []

        # Parse page range once and apply it to every uploaded PDF.
        _ps, _pe = 1, None
        _pr = (pdf_page_range or "").strip()
        if _pr:
            _parts = _pr.split("-")
            try:
                _ps = int(_parts[0])
                _pe = int(_parts[1]) if len(_parts) > 1 else _ps
            except ValueError:
                reporter.warning(f"Invalid page range '{_pr}'; extracting all pages.")

        _skip = SKIP_SECTIONS if pdf_skip_refs else set()

        for name, data in pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as _tmp:
                _tmp.write(data)
                _tmp_path = _tmp.name

            try:
                with reporter.spinner(f"Extracting text from {name}..."):
                    _pdf = parse_pdf(
                        _tmp_path,
                        page_start=_ps,
                        page_end=_pe,
                        skip_sections=_skip,
                        enable_ocr_fallback=bool(pdf_ocr),
                    )
            finally:
                try:
                    os.unlink(_tmp_path)
                except Exception:
                    pass

            if _pdf["error"]:
                message = f"PDF extraction failed for {name}: {_pdf['error']}"
                reporter.error(message)
                return AssembledInputs(failure=InputFailure(KIND_PDF_FAILED, message))

            if _pdf["text"].strip():
                text_parts.append(_pdf["text"].strip())

            for _w in _pdf.get("warnings", []):
                reporter.warning(f"{name}: {_w}")

            reporter.success(
                f"{name}: extracted **{_pdf['pages_used']}** of **{_pdf['total_pages']}** pages "
                f"via **{_pdf['method']}**. "
                f"Sections: {', '.join(_pdf['sections'].keys()) or 'none'}"
            )
            _meta = _pdf.get("metadata", {})
            _mp = [p for p in [
                f"Title: {_meta['title']}"     if _meta.get("title")  else "",
                f"Author(s): {_meta['author']}" if _meta.get("author") else "",
                f"DOI: {_meta['doi']}"          if _meta.get("doi")    else "",
            ] if p]
            if _mp:
                reporter.caption(" | ".join(_mp))

            with reporter.expander(f"Preview extracted text from {name} (first 1000 chars)", expanded=False):
                _preview = _pdf["text"][:1000] + ("..." if len(_pdf["text"]) > 1000 else "")
                reporter.text(_preview)

    text = "\n\n".join(text_parts)
    if not text.strip():
        message = "No text to process. Paste text, upload one or more PDFs, or use both."
        reporter.warning(message)
        return AssembledInputs(text=text, failure=InputFailure(KIND_NO_TEXT, message))

    return AssembledInputs(text=text)
