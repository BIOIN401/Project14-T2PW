"""R3a: ``t2pw.service.extraction.assemble_inputs``, the app's input assembly moved out.

NEW TESTS (R3a, new module). The function is a pure move of the start of the app's
``if submit:`` block, so these pin what that block did: which text reaches the stages
(texts first, then PDFs; each part stripped, empty parts dropped, joined by a blank
line), the page-range / skip-refs / OCR arguments handed to ``parse_pdf``, every
message and its order, and the three points where the app stops the run.

The unit tests are offline: ``parse_pdf`` is replaced and a recording reporter stands
in for Streamlit. The last two tests drive the REAL ``streamlit_app.py`` through
``AppTest`` to show the app still shows the warning and stops at the same point. This
file therefore runs in its own bounded process, like every AppTest-driving file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.extraction.pdf_parser import SKIP_SECTIONS  # noqa: E402
from t2pw.service import extraction as ex  # noqa: E402
from t2pw.service.reporter import StreamlitReporter  # noqa: E402

NO_TEXT = "No text to process. Paste text, upload one or more PDFs, or use both."
NO_PDF = "Please upload one or more PDFs using the file uploader above."


class _Ctx:
    def __init__(self, calls: List[tuple], name: str) -> None:
        self._calls, self._name = calls, name

    def __enter__(self) -> None:
        self._calls.append((f"{self._name}:enter",))

    def __exit__(self, *exc: Any) -> None:
        self._calls.append((f"{self._name}:exit",))


class RecordingReporter:
    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def __getattr__(self, name: str) -> Any:
        def _call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs) if kwargs else (name, args))
            return _Ctx(self.calls, name) if name in ("spinner", "expander") else None
        return _call


def _pdf_result(text: str = "PDF BODY", **over: Any) -> Dict[str, Any]:
    res = {"text": text, "metadata": {}, "total_pages": 10, "pages_used": 4,
           "sections": {"abstract": [0], "methods": [1]}, "method": "pdfplumber",
           "warnings": [], "error": ""}
    res.update(over)
    return res


@pytest.fixture
def fake_parse(monkeypatch):
    """Replace ``parse_pdf``; results are popped per call, every call is logged."""

    seen: List[Dict[str, Any]] = []
    queue: List[Dict[str, Any]] = []

    def _parse(path, **kwargs):
        seen.append(dict(kwargs, path=path, data=Path(path).read_bytes()))
        return queue.pop(0) if queue else _pdf_result()

    monkeypatch.setattr(ex, "parse_pdf", _parse)
    return seen, queue


def _run(texts=(), pdfs=(), *, mode="Text + PDF", page_range="", skip_refs=True,
         ocr=False, reporter=None):
    reporter = reporter or RecordingReporter()
    res = ex.assemble_inputs(list(texts), list(pdfs),
                             uses_pdf_input=mode in {"Upload PDF", "Text + PDF"},
                             input_mode=mode, pdf_page_range=page_range,
                             pdf_skip_refs=skip_refs, pdf_ocr=ocr, reporter=reporter)
    return res, getattr(reporter, "calls", None)


# ── text only ────────────────────────────────────────────────────────────────
def test_text_only_strips_drops_empty_and_joins_with_a_blank_line(fake_parse):
    res, calls = _run(["  first  ", "", "   ", "second\n"], mode="Paste text")
    assert res.text == "first\n\nsecond"
    assert res.failure is None
    assert calls == []
    assert fake_parse[0] == []


def test_paste_mode_ignores_pdfs_even_if_given(fake_parse):
    res, calls = _run(["t"], [("a.pdf", b"x")], mode="Paste text")
    assert res.text == "t" and calls == [] and fake_parse[0] == []


# ── PDF only: the exact reporter call sequence ───────────────────────────────
def test_pdf_only_text_and_exact_reporter_sequence(fake_parse):
    seen, queue = fake_parse
    queue.append(_pdf_result("  BODY  ", warnings=["w1", "w2"],
                             metadata={"title": "T", "author": "A", "doi": "D"}))
    res, calls = _run([], [("paper.pdf", b"%PDF-bytes")], mode="Upload PDF")

    assert res.text == "BODY" and res.failure is None
    assert seen[0]["data"] == b"%PDF-bytes"
    assert not os.path.exists(seen[0]["path"])  # temp file removed
    assert seen[0]["path"].endswith(".pdf")
    assert calls == [
        ("spinner", ("Extracting text from paper.pdf...",)),
        ("spinner:enter",),
        ("spinner:exit",),
        ("warning", ("paper.pdf: w1",)),
        ("warning", ("paper.pdf: w2",)),
        ("success", ("paper.pdf: extracted **4** of **10** pages via **pdfplumber**. "
                     "Sections: abstract, methods",)),
        ("caption", ("Title: T | Author(s): A | DOI: D",)),
        ("expander", ("Preview extracted text from paper.pdf (first 1000 chars)",),
         {"expanded": False}),
        ("expander:enter",),
        ("text", ("  BODY  ",)),
        ("expander:exit",),
    ]


def test_caption_lists_only_present_metadata_and_is_skipped_when_none(fake_parse):
    _, queue = fake_parse
    queue.extend([_pdf_result(metadata={"doi": "10.1/x"}),
                  _pdf_result(metadata={}, sections={})])
    _, calls = _run([], [("a.pdf", b"1"), ("b.pdf", b"2")], mode="Upload PDF")
    assert [c for c in calls if c[0] == "caption"] == [("caption", ("DOI: 10.1/x",))]
    successes = [c[1][0] for c in calls if c[0] == "success"]
    assert successes[1] == "b.pdf: extracted **4** of **10** pages via **pdfplumber**. Sections: none"


def test_preview_is_truncated_at_1000_chars(fake_parse):
    _, queue = fake_parse
    queue.extend([_pdf_result("x" * 1001), _pdf_result("y" * 1000)])
    _, calls = _run([], [("a.pdf", b"1"), ("b.pdf", b"2")], mode="Upload PDF")
    previews = [c[1][0] for c in calls if c[0] == "text"]
    assert previews == ["x" * 1000 + "...", "y" * 1000]


# ── mixed ────────────────────────────────────────────────────────────────────
def test_mixed_texts_come_first_then_pdfs_in_upload_order(fake_parse):
    _, queue = fake_parse
    queue.extend([_pdf_result("\nP1\n"), _pdf_result("   "), _pdf_result("P3")])
    res, _ = _run([" T1 ", "T2"], [("1.pdf", b"a"), ("2.pdf", b"b"), ("3.pdf", b"c")])
    assert res.text == "T1\n\nT2\n\nP1\n\nP3"  # the empty-text PDF adds no part
    assert [s["data"] for s in fake_parse[0]] == [b"a", b"b", b"c"]


# ── the three stop points ────────────────────────────────────────────────────
def test_upload_mode_with_no_pdf_stops_before_anything_else(fake_parse):
    res, calls = _run(["ignored"], [], mode="Upload PDF", page_range="bad")
    assert res.failure == ex.InputFailure(ex.KIND_NO_PDF, NO_PDF)
    assert calls == [("warning", (NO_PDF,))]  # page range is not even looked at


def test_text_and_pdf_mode_with_no_pdf_carries_on_with_the_text(fake_parse):
    res, calls = _run(["only text"], [], mode="Text + PDF", page_range="bad")
    assert res.text == "only text" and res.failure is None
    # The page range is still parsed (and warned about) with no PDF to apply it to.
    assert calls == [("warning", ("Invalid page range 'bad'; extracting all pages.",))]


@pytest.mark.parametrize("mode", ["Paste text", "Text + PDF"])
def test_no_text_at_all_is_the_no_text_failure(fake_parse, mode):
    res, calls = _run(["", "  "], [], mode=mode)
    assert res.failure == ex.InputFailure(ex.KIND_NO_TEXT, NO_TEXT)
    assert calls == [("warning", (NO_TEXT,))]


def test_an_empty_pdf_is_dropped_and_then_the_no_text_failure(fake_parse):
    _, queue = fake_parse
    queue.append(_pdf_result("  \n "))
    res, calls = _run([], [("scan.pdf", b"x")], mode="Upload PDF")
    assert res.failure == ex.InputFailure(ex.KIND_NO_TEXT, NO_TEXT)
    assert calls[3][0] == "success"  # the PDF is still reported as extracted
    assert calls[-1] == ("warning", (NO_TEXT,))


def test_an_unreadable_pdf_stops_at_that_pdf(fake_parse):
    seen, queue = fake_parse
    queue.extend([_pdf_result("good"), _pdf_result("", error="cannot open"), _pdf_result("never")])
    res, calls = _run(["t"], [("a.pdf", b"1"), ("b.pdf", b"2"), ("c.pdf", b"3")])
    msg = "PDF extraction failed for b.pdf: cannot open"
    assert res.failure == ex.InputFailure(ex.KIND_PDF_FAILED, msg)
    assert calls[-1] == ("error", (msg,))
    assert calls[-2] == ("spinner:exit",)  # nothing else said about b.pdf, c.pdf not touched
    assert len(seen) == 2
    assert not any(os.path.exists(s["path"]) for s in seen)


def test_the_temp_file_is_removed_even_when_the_parser_raises(monkeypatch):
    paths = []

    def boom(path, **kw):
        paths.append(path)
        raise RuntimeError("parser crashed")

    monkeypatch.setattr(ex, "parse_pdf", boom)
    with pytest.raises(RuntimeError, match="parser crashed"):
        _run([], [("a.pdf", b"1")], mode="Upload PDF")
    assert paths and not os.path.exists(paths[0])


# ── page range, skip refs, OCR ───────────────────────────────────────────────
@pytest.mark.parametrize("page_range,start,end,warned", [
    ("", 1, None, False),
    ("   ", 1, None, False),
    ("3-7", 3, 7, False),
    (" 5 ", 5, 5, False),
    ("x-2", 1, None, True),
    ("2-x", 2, None, True),  # the start is kept when only the end is bad, as before
])
def test_page_range_reaches_parse_pdf(fake_parse, page_range, start, end, warned):
    seen, _ = fake_parse
    _, calls = _run([], [("a.pdf", b"1"), ("b.pdf", b"2")], mode="Upload PDF",
                    page_range=page_range)
    assert [(s["page_start"], s["page_end"]) for s in seen] == [(start, end)] * 2
    warning = ("warning", (f"Invalid page range '{page_range.strip()}'; extracting all pages.",))
    assert (calls[0] == warning) is warned


@pytest.mark.parametrize("skip_refs,ocr", [(True, False), (False, True), (False, 0), (True, "on")])
def test_skip_refs_and_ocr_reach_parse_pdf(fake_parse, skip_refs, ocr):
    seen, _ = fake_parse
    _run([], [("a.pdf", b"1")], mode="Upload PDF", skip_refs=skip_refs, ocr=ocr)
    assert seen[0]["skip_sections"] == (SKIP_SECTIONS if skip_refs else set())
    assert seen[0]["enable_ocr_fallback"] is bool(ocr)


# ── no silent default, no streamlit ──────────────────────────────────────────
def test_the_reporter_is_required():
    with pytest.raises(TypeError):
        ex.assemble_inputs(["t"], [], uses_pdf_input=False, input_mode="Paste text",
                           pdf_page_range="", pdf_skip_refs=True, pdf_ocr=False)


def test_importing_the_module_loads_neither_streamlit_nor_the_app():
    code = ("import sys, t2pw.service.extraction\n"
            "print(sorted(n for n in sys.modules if n.split('.')[0] == 'streamlit'"
            " or n.startswith('t2pw.app')))\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=str(SRC)), cwd=str(ROOT), timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "[]"


def test_streamlit_reporter_forwards_the_new_calls_unchanged():
    calls = []

    class FakeSt:
        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name, a, k)) or f"<{name}>"

    rep = StreamlitReporter(FakeSt())
    assert rep.success("s") == "<success>"
    assert rep.caption("c") == "<caption>"
    assert rep.text("t") == "<text>"
    assert rep.spinner("busy...") == "<spinner>"
    assert rep.expander("Preview", expanded=False) == "<expander>"
    assert calls == [("success", ("s",), {}), ("caption", ("c",), {}), ("text", ("t",), {}),
                     ("spinner", ("busy...",), {}), ("expander", ("Preview",), {"expanded": False})]


def test_logging_reporter_handles_the_new_calls(caplog):
    import logging

    from t2pw.service.reporter import LoggingReporter

    rep = LoggingReporter(logging.getLogger("r3a-test"))
    with caplog.at_level(logging.INFO, logger="r3a-test"):
        res, _ = _run(["hello"], mode="Paste text", reporter=rep)
        with rep.spinner("spin"), rep.expander("exp", expanded=False):
            rep.success("ok")
            rep.caption("cap")
            rep.text("txt")
    assert res.text == "hello"
    said = [r.getMessage() for r in caplog.records if r.name == "r3a-test"]
    assert said == ["spin", "exp", "ok", "cap", "txt"]


# ── the real app: same warning, same stop ────────────────────────────────────
class _ReachedStageZero(Exception):
    pass


@pytest.mark.parametrize("mode,message", [("Paste text", NO_TEXT), ("Upload PDF", NO_PDF)])
def test_real_app_warns_and_stops_before_stage_zero(monkeypatch, mode, message):
    """The app shows the moved warning and still calls ``st.stop()`` there.

    If it did not stop, the next thing the script does is open the extraction
    diagnostics (which hashes the text). That hash is replaced by a raiser, so a
    missing stop shows up as an app exception instead of a real Stage-0 call.
    """

    from streamlit.testing.v1 import AppTest

    import t2pw.pipeline.extraction_diagnostics as diag

    def _reached(*_a, **_k):
        raise _ReachedStageZero("the run went past the input-assembly stop")

    monkeypatch.setattr(diag, "payload_hash", _reached)
    at = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
    assert not at.exception
    at.radio(key="input_mode_radio").set_value(mode).run()
    next(b for b in at.button if b.label == "Run pipeline").click().run()

    assert [str(e.value) for e in at.exception] == []
    assert message in [w.value for w in at.warning]
