"""NEW ACCEPTANCE TEST (W6, new capability).

The public pages (FastAPI + Jinja2 + HTMX) over the W3 job store, against the W0 stub:
nothing here existed before, so there is no base-commit failure to show (G9).
Everything runs through ``fastapi.testclient.TestClient`` on a temp DB and work root;
no LLM, no network, no server is started.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from t2pw.service import organisms as organism_data
from t2pw.service.pathway import PathwayResult
from t2pw.web import views
from t2pw.web.app import create_app
from t2pw.web.jobs import JobStore, result_to_dict

SID_RX = re.compile(r"/job/([0-9a-f]{32})$")
HOSTILE = '<script>alert(1)</script>"><img src=x onerror=alert(2)>'
PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
       b"\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7\x9a\xa0\xa0\x00"
       b"\x00\x00\x00IEND\xaeB`\x82")


def _pdf(pages: int = 1) -> bytes:
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode()]
    objs += [b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"] * pages
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
    def __init__(self, tmp: Path, **kw):
        self.db, self.work = tmp / "jobs.sqlite", tmp / "work"
        kw.setdefault("rate_limit", 100)
        self.app = create_app(self.db, self.work, "https://t2pw.test", rate_secret=b"test-secret", **kw)
        self.client = TestClient(self.app, follow_redirects=False)
        self.tmp = tmp
        self.org = organism_data.get_by_label("Homo sapiens")
        assert self.org is not None

    def form(self, **over):
        data = {"mode": "name", "name_1": "glycolysis", "organism_1": self.org.label, "consent": "1"}
        data.update(over)
        return data

    def submit(self, data=None, files=None, **kw):
        return self.client.post("/submit", data=self.form() if data is None else data, files=files, **kw)

    def new_job(self, **over) -> str:
        r = self.submit(self.form(**over))
        assert r.status_code == 303, r.text
        return SID_RX.search(r.headers["location"]).group(1)

    def store(self) -> JobStore:
        return JobStore(self.db)

    def run_to(self, sid: str, result: dict, *, requeue: bool = False, out_files=None) -> dict:
        """Claim the next item and finish it with ``result`` (a PathwayResult field dict)."""
        st = self.store()
        item = st.claim_next(3600)
        assert item is not None
        d = result_to_dict(PathwayResult(**result.pop("fields", {})))
        d.update(result)
        for rel, data in (out_files or {}).items():
            p = self.work / f"item-{item['id']}" / "attempt-1" / "out" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        st.finish(item["id"], item["attempt"], d, 0, requeue=requeue)
        st.close()
        return item

    def out(self, item_id: int, rel: str) -> str:
        return str(self.work / f"item-{item_id}" / "attempt-1" / "out" / rel)


@pytest.fixture
def site(tmp_path):
    return Site(tmp_path)


# ── pages ───────────────────────────────────────────────────────────────────────────


def test_each_public_page_renders(site):
    for path, needle in (("/", "Build a pathway"), ("/examples", "Examples"), ("/help", "Help"),
                         ("/about", "How to cite")):
        r = site.client.get(path)
        assert r.status_code == 200 and needle in r.text, path
        assert r.headers["content-type"].startswith("text/html")
    assert 'aria-current="page"' in site.client.get("/help").text
    missing = site.client.get("/no-such-page")
    assert missing.status_code == 404 and "Page not found" in missing.text


def test_submit_form_controls(site):
    html = site.client.get("/").text
    assert html.count('name="mode"') == 3
    assert re.search(r'name="consent"[^>]*\schecked', html), "PathBank consent must be on by default"
    assert 'type="email"' in html and 'name="email"' in html
    assert html.count('list="organism-options"') == 5 and '<datalist id="organism-options">' in html
    assert 'hx-get="/organisms"' in html
    assert html.count('name="pdfs_') == 5 and 'enctype="multipart/form-data"' in html
    assert "20 to 60 minutes" in html
    assert html.count('<script') == 1 and 'src="/static/js/htmx.min.js"' in html
    assert "<datalist" in html and len(html) < 40_000, "the organism list must not be inlined"


def test_organism_search_is_server_side_and_escaped(site):
    r = site.client.get("/organisms", params={"organism_1": "homo sap"})
    assert r.status_code == 200 and '<option value="Homo sapiens">' in r.text
    assert r.text.count("<option") <= 30
    assert site.client.get("/organisms", params={"organism_1": ""}).text.strip() == ""
    assert "<script" not in site.client.get("/organisms", params={"organism_1": HOSTILE}).text


# ── submission ──────────────────────────────────────────────────────────────────────


def test_good_submission_creates_a_job_and_redirects(site):
    r = site.submit(site.form(email="me@example.org"))
    assert r.status_code == 303 and SID_RX.search(r.headers["location"])
    sid = SID_RX.search(r.headers["location"]).group(1)
    sub = site.store().submission(sid)
    assert sub["consent"] is True and sub["email"] == "me@example.org" and len(sub["items"]) == 1
    req = sub["items"][0]["request"]
    assert req.scope.requested_pathway == "glycolysis"
    assert req.scope.requested_organism == site.org.scope
    assert req.pmids == () and req.pdf_paths == ()
    page = site.client.get(r.headers["location"])
    assert page.status_code == 200 and "Waiting in the queue" in page.text and "Queued" in page.text
    assert "https://t2pw.test/job/" + sid in page.text


def test_consent_off_and_several_pathways(site):
    data = site.form(name_2="TCA cycle", organism_2=site.org.label, name_3="urea cycle", organism_3=site.org.label)
    del data["consent"]
    sid = SID_RX.search(site.submit(data).headers["location"]).group(1)
    sub = site.store().submission(sid)
    assert sub["consent"] is False
    assert [i["request"].scope.requested_pathway for i in sub["items"]] == ["glycolysis", "TCA cycle", "urea cycle"]


def test_pmid_mode_stores_raw_ids_without_resolving(site):
    r = site.submit(site.form(mode="pmids", pmids_1="PMID:123, 456\n123;789"))
    sub = site.store().submission(SID_RX.search(r.headers["location"]).group(1))
    assert sub["items"][0]["request"].pmids == ("123", "456", "789")
    bad = site.submit(site.form(mode="pmids", pmids_1="12x4"))
    assert bad.status_code == 400 and "not PubMed IDs" in bad.text


def test_pdf_mode_saves_uploads_under_the_server_dir(site):
    files = [("pdfs_1", ("../../evil name.pdf", _pdf(), "application/pdf"))]
    r = site.submit(site.form(mode="pdfs"), files=files)
    assert r.status_code == 303, r.text
    req = site.store().submission(SID_RX.search(r.headers["location"]).group(1))["items"][0]["request"]
    (path,) = req.pdf_paths
    assert path.is_file() and path.resolve().is_relative_to((site.work / "uploads").resolve())
    assert "evil" not in path.name
    bad = site.submit(site.form(mode="pdfs"),
                      files=[("pdfs_1", ("x.pdf", b"not a pdf at all", "application/pdf"))])
    assert bad.status_code == 400 and "x.pdf is not a PDF file." in bad.text
    assert str(site.tmp) not in bad.text


@pytest.mark.parametrize("over,needle", [
    ({"name_1": ""}, "Enter the pathway name"),
    ({"organism_1": ""}, "Choose an organism"),
    ({"organism_1": "Homo sapient"}, "not in the organism list"),
    ({"email": "not-an-email"}, "valid email"),
    ({"name_1": "x" * 201}, "at most 200"),
])
def test_validation_errors_are_shown_with_a_400(site, over, needle):
    r = site.submit(site.form(**over))
    assert r.status_code == 400 and needle in r.text
    assert 'role="alert"' in r.text or "field__error" in r.text


def test_nothing_to_submit_is_rejected(site):
    r = site.submit({"mode": "name", "consent": "1"})
    assert r.status_code == 400 and "at least one pathway" in r.text
    assert site.store()._db.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0


@pytest.mark.parametrize("mode,extra", [("pmids", {"pmids_1": "123"}), ("pdfs", {})])
def test_pathway_name_is_required_in_every_mode(site, mode, extra):
    """NEW (BR, PO ruling Q-BR-1): the run uses the organism only with a pathway name."""
    files = [("pdfs_1", ("a.pdf", _pdf(), "application/pdf"))] if mode == "pdfs" else None
    r = site.submit(site.form(mode=mode, name_1="", **extra), files=files)
    assert r.status_code == 400 and "Enter the pathway name these papers describe" in r.text
    assert site.store()._db.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0
    html = site.client.get("/").text
    assert re.search(r'name="name_1"[^>]*\srequired', html), "row 1 name must be marked required"
    assert "Required for every option" in html


def test_unsupported_request_gets_its_own_failure_text(site):
    """NEW (BR, PO ruling Q-BR-4): a bridge refusal says what was wrong; a real quality-gate
    failure keeps the generic contract text."""
    msg = "the organism can only be used together with a pathway name; add the pathway name"
    sid = site.new_job()
    site.run_to(sid, dict(fields=dict(status="error", failure_kind="contract", message=msg,
                                      issue_codes=["bridge_unsupported_request"])))
    html = site.client.get(f"/job/{sid}/0").text
    assert "This pathway was not run because the request cannot be used as given: " + msg + "." in html
    assert "did not pass the quality checks" not in html
    sid2 = site.new_job()
    site.run_to(sid2, dict(fields=dict(status="fail", failure_kind="contract", message="gate")))
    assert "did not pass the quality checks" in site.client.get(f"/job/{sid2}/0").text


def test_rate_limit_and_validation_errors_cost_no_slot(tmp_path):
    s = Site(tmp_path, rate_limit=3)
    for _ in range(6):  # six invalid attempts: none may use a slot
        assert s.submit(s.form(name_1="")).status_code == 400
    for _ in range(3):
        assert s.submit().status_code == 303
    blocked = s.submit()
    assert blocked.status_code == 429 and "limit of" in blocked.text
    assert int(blocked.headers["retry-after"]) > 0
    n = s.store()._db.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
    assert n == 3
    # X-Forwarded-For from an untrusted peer must not buy a fresh bucket
    assert s.submit(headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 429
    # the form is shown again with the user's input kept
    assert 'value="glycolysis"' in blocked.text


def test_queue_full_is_a_503_and_cleans_up_the_upload(tmp_path):
    s = Site(tmp_path)
    for _ in range(4):
        s.new_job(**{"name_2": "a", "organism_2": s.org.label, "name_3": "b", "organism_3": s.org.label,
                     "name_4": "c", "organism_4": s.org.label, "name_5": "d", "organism_5": s.org.label})
    r = s.submit(s.form(mode="pdfs"), files=[("pdfs_1", ("a.pdf", _pdf(), "application/pdf"))])
    assert r.status_code == 503 and "queue is full" in r.text
    left = [p for p in (s.work / "uploads").rglob("*") if p.is_file()] if (s.work / "uploads").exists() else []
    assert left == []


def test_request_body_cap(tmp_path):
    s = Site(tmp_path, max_body_bytes=2000)
    assert s.client.post("/submit", content=b"x" * 5000,
                         headers={"content-type": "application/octet-stream"}).status_code == 413
    chunks = (b"y" * 1000 for _ in range(10))  # no Content-Length: counted as it arrives
    assert s.client.post("/submit", content=chunks,
                         headers={"content-type": "multipart/form-data; boundary=zz"}).status_code == 413
    assert s.client.get("/").status_code == 200


# ── job and result pages ────────────────────────────────────────────────────────────


def test_job_page_shows_every_status(site):
    # queued
    sid = site.new_job()
    html = site.client.get(f"/job/{sid}").text
    assert "Waiting in the queue" in html and 'hx-trigger="every 10s"' in html
    assert '<noscript><meta http-equiv="refresh"' in html
    # running
    st = site.store()
    item = st.claim_next(3600)
    html = site.client.get(f"/job/{sid}").text
    assert "Running (attempt 1 of 3)" in html and "badge--running" in html
    # retrying: attempt 1 failed with an infrastructure problem and was re-queued
    d = result_to_dict(PathwayResult(status="error", failure_kind="llm", message="empty reply"))
    st.finish(item["id"], 1, d, 1, requeue=True)
    html = site.client.get(f"/job/{sid}").text
    assert "Retrying automatically" in html and ">Retrying<" in html
    assert "Attempt 2 is queued" in html and "retried" in html
    # failed after the last attempt
    item = st.claim_next(3600)
    st.finish(item["id"], 2, d, 1)
    html = site.client.get(f"/job/{sid}").text
    assert "Could not build this pathway" in html and "after 2 attempts" in html
    assert "hx-trigger" not in html and "http-equiv" not in html, "polling must stop when all is final"
    assert "2026" in html or "UTC" in html  # attempt table with times
    st.close()


def test_review_required_and_done_and_fragment(site):
    sid = site.new_job()
    item = site.run_to(sid, dict(status="pass", pwml_path=None, release_status={"status": "review_required"},
                                 warnings=["3 reactions have no citation"]))
    html = site.client.get(f"/job/{sid}").text
    assert "badge--review_required" in html and "Needs review" in html
    assert "3 reactions have no citation" in html and "alert--warning" in html
    frag = site.client.get(f"/job/{sid}/status")
    assert frag.status_code == 200 and "<html" not in frag.text and 'id="job-items"' in frag.text
    sid2 = site.new_job()
    site.run_to(sid2, dict(status="pass", release_status={"status": "release_ready"}))
    assert "badge--done" in site.client.get(f"/job/{sid2}").text
    assert item["id"] != 0


def test_result_page_downloads_image_and_reports(site):
    sid = site.new_job()
    st = site.store()
    item = st.claim_next(3600)
    files = {"pathway.pwml": b"<pwml/>", "diagram.png": PNG, "rep.json": b"{}"}
    for rel, data in files.items():
        p = site.work / f"item-{item['id']}" / "attempt-1" / "out" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    d = result_to_dict(PathwayResult(status="pass"))
    d.update(release_status={"status": "release_ready"}, pwml_path=site.out(item["id"], "pathway.pwml"),
             image_path=site.out(item["id"], "diagram.png"),
             report_paths={"rep.json": site.out(item["id"], "rep.json")})
    st.finish(item["id"], 1, d, 0)
    page = site.client.get(f"/job/{sid}/0")
    assert page.status_code == 200
    for needle in ("Download PWML", f"/job/{sid}/0/file/pwml", "<figure", f"/job/{sid}/0/file/image",
                   f"/job/{sid}/0/file/report/rep.json", 'alt="Rendered diagram of glycolysis"'):
        assert needle in page.text, needle
    pw = site.client.get(f"/job/{sid}/0/file/pwml")
    assert pw.status_code == 200 and pw.content == b"<pwml/>"
    assert pw.headers["content-disposition"].startswith("attachment") and "pathway.pwml" in pw.headers["content-disposition"]
    assert "sandbox" in pw.headers["content-security-policy"]
    img = site.client.get(f"/job/{sid}/0/file/image")
    assert img.headers["content-type"] == "image/png" and img.headers["content-disposition"].startswith("inline")
    assert site.client.get(f"/job/{sid}/0/file/report/rep.json").json() == {}
    # the same job page for a not-yet-final item still renders (no files, auto reload)
    st.close()


def test_downloads_are_confined_to_the_items_own_worker_dir(site):
    secret = site.tmp / "secret.txt"
    secret.write_text("TOP SECRET")
    sid = site.new_job(name_2="second", organism_2=site.org.label)
    st = site.store()
    first = st.claim_next(3600)
    other = st.claim_next(3600)
    own_log = site.work / f"item-{first['id']}" / "attempt-1" / "child.log"
    own_log.parent.mkdir(parents=True)
    own_log.write_text("log")
    foreign = site.work / f"item-{other['id']}" / "attempt-1" / "out" / "theirs.txt"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("other item")
    d = result_to_dict(PathwayResult(status="pass"))
    d.update(release_status={"status": "release_ready"}, pwml_path=str(secret),
             image_path=str(site.work / f"item-{first['id']}" / ".." / ".." / "secret.txt"),
             report_paths={"leak.txt": str(secret), "log.txt": str(own_log), "theirs.txt": str(foreign),
                           "gone.txt": site.out(first["id"], "gone.txt")})
    st.finish(first["id"], 1, d, 0)
    base = f"/job/{sid}/0/file"
    for url in (f"{base}/pwml", f"{base}/image", f"{base}/report/leak.txt", f"{base}/report/log.txt",
                f"{base}/report/theirs.txt", f"{base}/report/gone.txt", f"{base}/report/nope",
                f"{base}/report/..%2f..%2fsecret.txt", f"{base}/report/%2e%2e/secret.txt",
                f"{base}/report/C:%5Cwindows%5Cwin.ini", f"{base}/bogus", f"/job/{sid}/9/file/pwml",
                f"/job/{'0' * 32}/0/file/pwml", f"/job/{sid[:-1]}/0/file/pwml", f"/job/{sid}/-1/file/pwml"):
        r = site.client.get(url)
        assert r.status_code == 404, url
        assert "TOP SECRET" not in r.text and "other item" not in r.text, url
    st.close()


def test_diagnostic_only_pwml_is_not_offered(site):
    sid = site.new_job()
    item = site.run_to(sid, dict(status="pass", release_status={"status": "diagnostic_only"}),
                       out_files={"pathway.pwml": b"<x/>"})
    st = site.store()
    st._db.execute("UPDATE items SET result_json = json_set(result_json, '$.pwml_path', ?) WHERE id = ?",
                   (site.out(item["id"], "pathway.pwml"), item["id"]))
    page = site.client.get(f"/job/{sid}/0")
    assert "Download PWML" not in page.text and "No usable PWML" in page.text
    assert site.client.get(f"/job/{sid}/0/file/pwml").status_code == 404
    assert "not offered for download" in page.text


def test_no_server_path_in_any_html(site):
    sid = site.new_job(email="me@example.org")
    leaky = (f"boom in {site.work}\\item-1\\x.py and C:\\Users\\angad\\secret\\k.py and "
             f"/home/t2pw/app/run.py and {site.work.as_posix()}/uploads/a.pdf")
    item = site.run_to(sid, dict(status="error", failure_kind="crash", message=leaky, warnings=[leaky],
                                 report_paths={"rep.json": site.out(1, "rep.json")},
                                 fields={}), out_files={"rep.json": b"{}"})
    pages = ["/", "/examples", "/help", "/about", f"/job/{sid}", f"/job/{sid}/status", f"/job/{sid}/0",
             "/nope", "/organisms?organism_1=homo"]
    pages_html = [site.client.get(p).text for p in pages]
    pages_html.append(site.submit(site.form(name_1=""), ).text)
    for html in pages_html:
        for bad in (str(site.tmp), site.tmp.as_posix(), "C:\\Users", "angad", "/home/t2pw", "item-1/", "item-1\\", "attempt-1",
                    "uploads/", "uploads\\", "x.py", "child.log"):
            assert bad not in html, bad
    assert "[server path]" in site.client.get(f"/job/{sid}/0").text
    assert item["id"] == 1


def test_scrub_unit():
    s = views.scrub(r"a C:\x\y.txt b \\host\share\f c /var/lib/t2pw/x.pdf d and/or e 1/2")
    assert "C:" not in s and "host" not in s and "/var" not in s
    assert "and/or" in s and "1/2" in s
    assert len(views.scrub("z" * 5000)) == 600


# ── security and identity ───────────────────────────────────────────────────────────


def test_security_headers_everywhere(site):
    sid = site.new_job()
    for url in ("/", "/help", f"/job/{sid}", "/static/css/site.css", "/nope", "/organisms?organism_1=a"):
        h = site.client.get(url).headers
        csp = h["content-security-policy"]
        assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
        assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "http" not in csp
        assert h["x-content-type-options"] == "nosniff"
        assert h["referrer-policy"] == "no-referrer" and h["x-frame-options"] == "DENY"
    j = site.client.get(f"/job/{sid}").headers
    assert "no-store" in j["cache-control"] and "noindex" in j["x-robots-tag"]
    assert site.client.post("/submit", data={}).headers["x-content-type-options"] == "nosniff"


def test_hostile_input_is_escaped_everywhere(site):
    sid = site.new_job(name_1=HOSTILE)
    for url in (f"/job/{sid}", f"/job/{sid}/status", f"/job/{sid}/0"):
        html = site.client.get(url).text
        assert "&lt;script&gt;alert(1)" in html, url
        assert "<script>alert" not in html and "<img src=x" not in html, url
        assert html.count("<script") == (1 if "<html" in html else 0)
    redisplayed = site.submit(site.form(name_1=HOSTILE, organism_1=HOSTILE)).text
    assert "<script>alert" not in redisplayed and "<img src=x" not in redisplayed
    assert redisplayed.count("<script") == 1
    assert '"><img' not in redisplayed
    site.run_to(sid, dict(status="error", failure_kind="llm", message=HOSTILE, warnings=[HOSTILE]))
    result = site.client.get(f"/job/{sid}/0").text
    assert "<img src=x" not in result and result.count("<script") == 1


def test_job_urls_are_unguessable_tokens(site):
    sids = [site.new_job() for _ in range(3)]
    assert len(set(sids)) == 3 and all(re.fullmatch(r"[0-9a-f]{32}", s) for s in sids)
    nums = sorted(int(s, 16) for s in sids)
    assert nums[1] - nums[0] > 2**64 and nums[2] - nums[1] > 2**64, "ids must not be sequential"
    assert not any(s.isdigit() for s in sids)
    for guess in ("/job/1", "/job/2", "/job/0", f"/job/{'a' * 32}", "/job/1/0", "/job/1/0/file/pwml"):
        assert site.client.get(guess).status_code == 404, guess
    # W7's current e-mail link shape resolves to the real page until postrun.py uses job_url()
    r = site.client.get(f"/jobs/{sids[0]}")
    assert r.status_code == 303 and r.headers["location"] == f"/job/{sids[0]}"
    assert site.client.get("/jobs/1").status_code == 404


def test_job_url_is_the_one_canonical_builder_and_cheap_to_import(site):
    sid = site.new_job()
    assert views.job_url("https://t2pw.test/", sid) == f"https://t2pw.test/job/{sid}"
    assert site.client.get(f"/job/{sid}").text.count(views.job_url("https://t2pw.test", sid)) >= 1
    with pytest.raises(ValueError):
        views.job_url("https://t2pw.test", "1")
    code = ("import sys; from t2pw.web.views import job_url; "
            "assert 'fastapi' not in sys.modules and 'starlette' not in sys.modules; "
            "print(job_url('', 'a' * 32))")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "/job/" + "a" * 32


# ── packaging and factory ───────────────────────────────────────────────────────────


def test_create_app_has_no_side_effects(tmp_path):
    db, work = tmp_path / "x" / "jobs.sqlite", tmp_path / "x" / "work"
    app = create_app(db, work, "https://t2pw.test", rate_secret=b"k")
    assert app.openapi_url is None and app.docs_url is None
    assert not (tmp_path / "x").exists(), "create_app must not create the DB or work root"


def test_vendored_htmx_is_the_only_script(site):
    js = Path(__file__).resolve().parents[1] / "src" / "t2pw" / "web" / "static" / "js"
    assert [p.name for p in js.iterdir()] == ["htmx.min.js"]
    head = (js / "htmx.min.js").read_text(encoding="utf-8")[:600]
    assert "htmx 2.0.4" in head and "Zero-Clause BSD" in head
    r = site.client.get("/static/js/htmx.min.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]


def test_templates_and_static_ship_as_package_data():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools.package-data]" in text
    web = root / "src" / "t2pw" / "web"
    for pattern in ("templates/*.html", "templates/macros/*.html", "static/css/*.css", "static/js/*.js"):
        assert f'"{pattern}"' in text and list(web.glob(pattern)), pattern
    assert "fastapi" in (root / "requirements.txt").read_text(encoding="utf-8")


# ── review round 1 ──────────────────────────────────────────────────────────────────


def test_layout_has_the_literal_htmx_tag_and_no_jinja_global(site):
    layout = (Path(__file__).resolve().parents[1] / "src" / "t2pw" / "web" / "templates" / "layout.html")
    text = layout.read_text(encoding="utf-8")
    assert text.count('<script src="/static/js/htmx.min.js" defer></script>') == 1
    assert "htmx_script" not in site.app.state.templates.env.globals


def test_cross_site_post_is_refused_and_costs_no_slot(tmp_path):
    s = Site(tmp_path, rate_limit=1)
    for headers in ({"Sec-Fetch-Site": "cross-site"}, {"Origin": "https://evil.example"},
                    {"Origin": "null"}, {"Origin": "https://evil.example", "Sec-Fetch-Site": "same-origin"}):
        r = s.submit(headers=headers)
        assert r.status_code == 403 and "Request blocked" in r.text, headers
    assert s.store()._db.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0
    # same-origin variants are accepted; the single slot was still free, so this one succeeds
    assert s.submit(headers={"Sec-Fetch-Site": "same-origin", "Origin": "http://testserver"}).status_code == 303
    assert s.submit(headers={"Origin": "https://t2pw.test"}).status_code == 429  # base_url host is allowed too: rate-limited, not 403
    assert s.submit().status_code == 429  # no Origin / Sec-Fetch-Site (curl) is not blocked


def test_trailing_newline_in_a_job_id_is_a_404(site):
    sid = site.new_job()
    for url in (f"/job/{sid}%0A", f"/job/{sid}%0A/status", f"/job/{sid}%0A/0", f"/jobs/{sid}%0A",
                f"/job/{sid}/0%0A"):
        assert site.client.get(url).status_code == 404, url
    with pytest.raises(ValueError):
        views.job_url("https://t2pw.test", sid + "\n")


def test_all_x_forwarded_for_lines_are_used(tmp_path):
    s = Site(tmp_path, rate_limit=1, trusted_proxies="10.0.0.0/8")
    s.client = TestClient(s.app, follow_redirects=False, client=("10.0.0.1", 50000))
    first = s.submit(headers=[("x-forwarded-for", "1.1.1.1"), ("x-forwarded-for", "203.0.113.7")])
    assert first.status_code == 303
    # same real client (last hop on the LAST line), different spoofed first line: same bucket
    again = s.submit(headers=[("x-forwarded-for", "2.2.2.2"), ("x-forwarded-for", "203.0.113.7")])
    assert again.status_code == 429
    other = s.submit(headers=[("x-forwarded-for", "2.2.2.2"), ("x-forwarded-for", "203.0.113.8")])
    assert other.status_code == 303
