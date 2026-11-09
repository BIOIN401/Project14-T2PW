"""W6c: fixes for what the first real-browser walk (E2E-3) found.

Every test here fails on the base commit (cc059750) and passes at the W6c tip:
the Chromium ``Origin: null`` submit, the Attempts-table text for quality-gate kinds,
common-name organism search, and a non-PDF upload costing a daily rate slot.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from t2pw.service import organisms as organism_data
from t2pw.service.pathway import PathwayResult
from t2pw.web import views
from t2pw.web.app import create_app
from t2pw.web.jobs import JobStore, result_to_dict

# Exactly what Chromium 156 sent for its own form POST (docs/webserver/evidence/E2E-3/results.json,
# notes.post_headers[0]); the Host is the test client's.
CHROMIUM_POST = {"Origin": "null", "Sec-Fetch-Site": "same-origin"}
SID_RX = re.compile(r"/job/([0-9a-f]{32})$")


def _pdf() -> bytes:
    """A valid one-page PDF with a real xref table (same shape as test_web_app._pdf)."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


class Site:
    def __init__(self, tmp, **kw):
        self.db, self.work = tmp / "jobs.sqlite", tmp / "work"
        kw.setdefault("rate_limit", 100)
        self.app = create_app(self.db, self.work, "https://t2pw.test", rate_secret=b"k", **kw)
        self.client = TestClient(self.app, follow_redirects=False)
        self.org = organism_data.get_by_label("Homo sapiens")

    def form(self, **over):
        data = {"mode": "name", "name_1": "glycolysis", "organism_1": self.org.label, "consent": "1"}
        data.update(over)
        return data

    def submit(self, data=None, files=None, **kw):
        return self.client.post("/submit", data=self.form() if data is None else data, files=files, **kw)

    def submissions(self) -> int:
        st = JobStore(self.db)
        try:
            return st._db.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        finally:
            st.close()


# -- fix 1: the Chromium submit -------------------------------------------------------


def test_chromium_same_origin_post_with_origin_null_is_accepted(tmp_path):
    s = Site(tmp_path)
    r = s.submit(headers=CHROMIUM_POST)
    assert r.status_code == 303 and SID_RX.search(r.headers["location"]), r.text
    assert r.headers["referrer-policy"] == "no-referrer"  # the job URL stays out of Referer


@pytest.mark.parametrize("headers", [
    {"Origin": "null"},
    {"Origin": "null", "Sec-Fetch-Site": "cross-site"},
    {"Origin": "null", "Sec-Fetch-Site": "same-site"},
    {"Origin": "null", "Sec-Fetch-Site": "none"},
    {"Origin": "https://evil.example", "Sec-Fetch-Site": "same-origin"},
    {"Origin": "NULL", "Sec-Fetch-Site": "same-origin"},
])
def test_origin_null_is_only_accepted_with_exactly_same_origin(tmp_path, headers):
    s = Site(tmp_path)
    r = s.submit(headers=headers)
    assert r.status_code == 403 and "Request blocked" in r.text, headers
    assert s.submissions() == 0


# -- fix 2: quality-gate failures in the Attempts table --------------------------------


@pytest.mark.parametrize("kind,word", [("contract", "quality checks"), ("ambiguous_review_scope", "quality checks"),
                                       ("no_reactions", "no reactions")])
def test_attempts_table_describes_quality_gate_failures(tmp_path, kind, word):
    s = Site(tmp_path)
    sid = SID_RX.search(s.submit().headers["location"]).group(1)
    st = JobStore(s.db)
    item = st.claim_next(3600)
    st.finish(item["id"], item["attempt"],
              result_to_dict(PathwayResult(status="fail", failure_kind=kind, message="gate")), 0, requeue=False)
    st.close()
    page = s.client.get(f"/job/{sid}/0").text
    assert "an unexpected problem occurred" not in page
    assert f"Failed ({views._kind_text(kind)})" in page and word in views._kind_text(kind)


def test_unknown_kind_keeps_the_generic_text():
    assert views._kind_text("something_new") == "an unexpected problem occurred"


# -- fix 3: organism search by common name ----------------------------------------------


def _first(q, n=1):
    return [o.label for o in views.search_organisms(organism_data.load_organisms(), q)[:n]]


def test_common_names_are_searched_and_ranked():
    assert _first("mouse") == ["Mus musculus"]
    assert _first("rat") == ["Rattus norvegicus"]
    assert "Saccharomyces cerevisiae" in _first("yeast", 5)
    assert "Homo sapiens" in _first("human", 6)  # behind misspelt rows that the species file still holds
    assert _first("E. coli")[0] == "Escherichia coli"
    assert _first("e coli")[0] == "Escherichia coli"


def test_search_keeps_the_cap_and_scientific_names():
    orgs = organism_data.load_organisms()
    assert len(views.search_organisms(orgs, "escherichia coli")) == 30
    assert _first("escherichia coli")[0] == "Escherichia coli"
    assert _first("arabidopsis") == ["Arabidopsis thaliana"]
    assert views.search_organisms(orgs, "  ") == []
    assert views.search_organisms(orgs, "zzzzqq") == []


# -- fix 4: a refused upload keeps the daily slot ---------------------------------------


def test_non_pdf_and_empty_uploads_cost_no_rate_slot(tmp_path):
    s = Site(tmp_path, rate_limit=1)
    for body in (b"just some notes\n", b""):
        r = s.submit(s.form(mode="pdfs"), files=[("pdfs_1", ("notes.pdf", body, "application/pdf"))])
        assert r.status_code == 400 and ("is not a PDF file." in r.text or "is empty." in r.text)
    # the single slot is still free: a good upload (read back from the start) is accepted
    r = s.submit(s.form(mode="pdfs"), files=[("pdfs_1", ("ok.pdf", _pdf(), "application/pdf"))])
    assert r.status_code == 303, r.text
    assert s.submissions() == 1
    assert s.submit().status_code == 429  # the slot is spent by the accepted one


# -- fix 5: field limits ------------------------------------------------------------------


def test_name_and_organism_inputs_carry_the_server_limit(tmp_path):
    html = Site(tmp_path).client.get("/").text
    for field in ("name_1", "organism_1"):
        assert re.search(rf'<input[^>]*name="{field}"[^>]*maxlength="200"', html), field
