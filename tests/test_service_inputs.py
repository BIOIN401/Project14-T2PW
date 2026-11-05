"""NEW ACCEPTANCE TEST (W1, new capability).

Covers ``t2pw.service.inputs``: name + organism -> TopicSpec/RequestedScope,
PMID validation, PMID -> PMCID -> full text (with the "please upload the PDF"
outcome), mixed PMID lists, and the PDF adapter.

No live network: ``fetch_papers`` is replaced by a fake, and the PMID path gets a
fake HTTP client plus a fake full-text function. The PDF adapter test uses a tiny
real PDF generated in ``tmp_path`` (real ``pdf_parser.parse_pdf``, no mock) when
pdfplumber/pypdf can read it; the failure-mode tests mock ``parse_pdf``.
"""

from __future__ import annotations

import pytest

from t2pw.batch.fetch import TopicSpec
from t2pw.rag.eligibility import RequestedScope
from t2pw.service import inputs as si


# -- name mode ----------------------------------------------------------------
def test_name_and_organism_build_spec_and_scope():
    seen = {}

    def fake_fetch(specs, **kw):
        seen["specs"], seen["kw"] = specs, kw
        return ["paper"], []

    out = si.resolve_name("  lipid A   biosynthesis ", "Escherichia coli", 2, fetch_fn=fake_fetch, use_cache=False)
    assert out.spec == TopicSpec(topic="lipid A biosynthesis", organism="Escherichia coli", count=2)
    assert out.scope == RequestedScope("lipid A biosynthesis", "Escherichia coli", False)
    assert seen["specs"] == [out.spec] and seen["kw"]["use_cache"] is False
    assert out.papers == ["paper"] and out.message == ""


def test_name_mode_nothing_found_says_so_and_bad_input_rejected():
    out = si.resolve_name("serine biosynthesis", "", fetch_fn=lambda specs, **kw: ([], [{"reason": "no_full_text"}]))
    assert out.papers == [] and "no open-access full text" in out.message
    assert out.skipped == [{"reason": "no_full_text"}]
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            si.name_to_spec(bad, "E. coli")
    with pytest.raises(ValueError):
        si.name_to_spec("x" * 201)
    with pytest.raises(ValueError):
        si.name_to_spec("x", count=0)


# -- PMID mode ----------------------------------------------------------------
class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


class _Client:
    def __init__(self, records=None, status=200, exc=None):
        self.records, self.status, self.exc, self.calls = records or [], status, exc, []

    def get(self, url, params=None, headers=None):
        self.calls.append((url, params))
        if self.exc:
            raise self.exc
        return _Resp({"status": "ok", "records": self.records}, self.status)


@pytest.mark.parametrize("bad", ["", "abc", "12 34", "0123", "-5", "1.5", "PMC123", "1234567890", None, "12345678; DROP"])
def test_bad_pmid_formats_rejected(bad):
    with pytest.raises(ValueError):
        si.validate_pmid(bad)
    res = si.resolve_pmids([bad], client=_Client())
    assert [(r.status, r.text) for r in res] == [(si.INVALID, "")] and res[0].message


def test_pmid_with_pmcid_returns_text_and_sends_tool_param(monkeypatch):
    monkeypatch.setenv("NCBI_TOOL", "t2pw-test")
    client = _Client([{"pmid": "123456", "pmcid": "PMC999"}])
    res = si.resolve_pmids([" 123456 "], client=client, fetch_text_fn=lambda pmcid: f"full text of {pmcid}")
    assert (res[0].ok, res[0].pmid, res[0].pmcid, res[0].text) == (True, "123456", "PMC999", "full text of PMC999")
    url, params = client.calls[0]
    assert url == si.IDCONV_URL and params["ids"] == "123456" and params["tool"] == "t2pw-test"


def test_pmid_without_pmcid_is_a_user_facing_outcome_not_a_drop():
    client = _Client([{"pmid": "111"}, {"pmid": "222", "status": "error", "errmsg": "invalid article id"}])
    res = si.resolve_pmids(["111", "222", "333"], client=client, fetch_text_fn=lambda p: pytest.fail("no fetch"))
    assert [r.pmid for r in res] == ["111", "222", "333"]
    assert all(r.status == si.NO_PMCID and not r.ok for r in res)
    assert res[0].message == "no open-access full text found for PMID 111; please upload the PDF"


def test_pmcid_but_no_open_access_full_text():
    client = _Client([{"pmid": "444", "pmcid": "PMC1"}])
    res = si.resolve_pmids(["444"], client=client, fetch_text_fn=lambda p: "  ")
    assert res[0].status == si.NO_FULL_TEXT and res[0].pmcid == "PMC1"
    assert res[0].message == "no open-access full text found for PMID 444; please upload the PDF"


def test_mixed_list_preserves_order_dedupes_and_reports_every_entry():
    client = _Client([{"pmid": "1001", "pmcid": "PMC1"}, {"pmid": "1002"}, {"pmid": "1003", "pmcid": "PMC3"}])
    texts = {"PMC1": "alpha", "PMC3": ""}
    res = si.resolve_pmids(["1001", "oops", "1002", "1001", "1003"], client=client, fetch_text_fn=texts.get)
    assert [(r.pmid, r.status) for r in res] == [
        ("1001", si.OK), ("oops", si.INVALID), ("1002", si.NO_PMCID), ("1003", si.NO_FULL_TEXT),
    ]
    assert len(client.calls) == 1 and client.calls[0][1]["ids"] == "1001,1002,1003"


@pytest.mark.parametrize("client", [_Client(exc=RuntimeError("boom")), _Client(status=429)])
def test_idconv_outage_is_lookup_failed_not_crash_or_no_oa(client):
    res = si.resolve_pmids(["555"], client=client)
    assert res[0].status == si.LOOKUP_FAILED and "upload the PDF" in res[0].message
    assert "no open-access" not in res[0].message  # an outage must not be reported as "no OA"


def test_full_text_fetch_error_is_lookup_failed():
    def boom(pmcid):
        raise OSError("down")

    res = si.resolve_pmids(["666"], client=_Client([{"pmid": "666", "pmcid": "PMC6"}]), fetch_text_fn=boom)
    assert res[0].status == si.LOOKUP_FAILED and res[0].pmcid == "PMC6"


# -- PDF mode -----------------------------------------------------------------
def _tiny_pdf(text: str) -> bytes:
    """A one-page PDF with correct xref offsets, so pdfplumber/pypdf read it cleanly."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_pdf_adapter_reads_real_tiny_fixture_pdfs(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(_tiny_pdf("Glycine cleavage enzyme"))
    b.write_bytes(_tiny_pdf("Serine hydroxymethyltransferase"))
    res = si.read_pdfs([a, str(b)])  # real pdf_parser.parse_pdf, no mock
    assert [r.status for r in res] == [si.OK, si.OK]
    assert "Glycine cleavage enzyme" in res[0].text and "Serine hydroxymethyltransferase" in res[1].text
    assert res[0].method in {"pdfplumber", "pypdf"}


def test_pdf_adapter_failure_modes_are_reported_not_dropped(tmp_path):
    good, notpdf = tmp_path / "g.pdf", tmp_path / "x.pdf"
    good.write_bytes(b"%PDF-1.4 stub")
    notpdf.write_text("hello")
    replies = iter([{"text": "", "method": "ocr", "warnings": ["w"]}, {"text": "", "error": "File not found"}])
    res = si.read_pdfs([good, notpdf, tmp_path / "missing.pdf", good], parse_fn=lambda p: next(replies))
    assert [r.status for r in res] == [si.EMPTY, si.ERROR, si.ERROR, si.ERROR]
    assert res[0].warnings == ["w"] and "no text could be extracted" in res[0].message
    assert "not a PDF" in res[1].message and "missing.pdf" in res[2].message
    assert "File not found" in res[3].message


def test_pdf_adapter_parser_exception_is_an_outcome(tmp_path):
    f = tmp_path / "f.pdf"
    f.write_bytes(b"%PDF-1.4")

    def boom(p):
        raise ValueError("corrupt")

    res = si.read_pdfs([f], parse_fn=boom)
    assert res[0].status == si.ERROR and "corrupt" in res[0].message
