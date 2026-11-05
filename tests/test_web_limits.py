"""W8 limits: rate limit, upload checks, safe upload saver.

NEW ACCEPTANCE TEST (W8, new capability): every test here covers a module that did not
exist before; there is no base-SHA behaviour to preserve.
"""

import io
import sqlite3

import pytest

from t2pw.web import jobs, limits


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


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


# -- rate limit ----------------------------------------------------------------------
def test_rate_limit_under_at_over_and_rolling_window(tmp_path):
    clock = Clock()
    rl = limits.RateLimiter(tmp_path / "jobs.db", limit=3, window_s=100, clock=clock)
    assert rl.hit("a") == (True, 0.0)            # under
    clock.t += 10
    assert rl.hit("a")[0]
    clock.t += 10
    assert rl.hit("a")[0]                        # exactly at the limit: 3rd is allowed
    clock.t += 10
    ok, wait = rl.hit("a")                       # over
    assert not ok and wait == pytest.approx(100 - 30)
    assert rl.hit("b")[0]                        # other clients unaffected
    clock.t += 69                                # 99 s after the first hit: still blocked
    assert not rl.hit("a")[0]
    clock.t += 2                                 # first hit has rolled out
    assert rl.hit("a")[0]
    ok, wait = rl.hit("a")                       # the 2 s-old... and 3 other hits still count
    assert not ok and 0 < wait <= 100


def test_rate_limit_rejected_hits_do_not_extend_the_block(tmp_path):
    clock = Clock()
    rl = limits.RateLimiter(tmp_path / "jobs.db", limit=1, window_s=100, clock=clock)
    assert rl.hit("a")[0]
    for _ in range(5):
        clock.t += 10
        assert not rl.hit("a")[0]
    clock.t += 51
    assert rl.hit("a")[0]


def test_rate_limit_shares_the_jobs_db_file_and_stores_no_address(tmp_path):
    db = tmp_path / "jobs.db"
    store = jobs.JobStore(db)                     # jobs.py creates its tables first
    rl = limits.RateLimiter(db, limit=1)
    assert rl.hit("203.0.113.9")[0] and not rl.hit("203.0.113.9")[0]
    assert store.submission("nope") is None       # jobs tables still work beside ours
    raw = sqlite3.connect(db).execute("SELECT key FROM rate_events").fetchall()
    assert raw and all("203.0.113.9" not in k for (k,) in raw)
    rl2 = limits.RateLimiter(db, limit=1)         # a second process sees the same counts
    assert not rl2.hit("203.0.113.9")[0]


def test_rate_limit_message_has_no_internals():
    msg = limits.rate_limit_message(90)
    assert "2 minutes" in msg and "limit" in msg
    assert "hours" in limits.rate_limit_message(5 * 3600)


# -- client key / X-Forwarded-For ------------------------------------------------------
def test_client_key_ignores_forwarded_for_unless_peer_is_a_trusted_proxy():
    xff = "198.51.100.7"
    assert limits.client_key("203.0.113.5", xff, trusted_proxies="") == "203.0.113.5"
    assert limits.client_key("203.0.113.5", xff, trusted_proxies="10.0.0.1") == "203.0.113.5"
    assert limits.client_key("10.0.0.1", xff, trusted_proxies="10.0.0.1") == "198.51.100.7"


def test_client_key_walks_right_to_left_so_a_spoofed_entry_is_ignored():
    # the client sent 1.1.1.1; the proxy appended the real peer 198.51.100.7; edge proxy 10.0.0.2
    key = limits.client_key("10.0.0.1", "1.1.1.1, 198.51.100.7, 10.0.0.2", trusted_proxies="10.0.0.0/24")
    assert key == "198.51.100.7"
    assert limits.client_key("10.0.0.1", "1.1.1.1, not-an-ip", trusted_proxies="10.0.0.1") == "10.0.0.1"
    assert limits.client_key("", None, trusted_proxies="") == "unknown"


def test_client_key_ipv6_is_keyed_by_slash_64_and_env_is_the_default(monkeypatch):
    a = limits.client_key("2001:db8:1:2::1")
    b = limits.client_key("2001:db8:1:2:ffff::9")
    assert a == b == "2001:db8:1:2::/64"
    monkeypatch.setenv("T2PW_TRUSTED_PROXIES", "10.0.0.1, bogus")
    assert limits.client_key("10.0.0.1", "198.51.100.7") == "198.51.100.7"


# -- submission checks -----------------------------------------------------------------
def test_pathway_cap_reuses_the_jobs_constant():
    assert limits.MAX_ITEMS_PER_SUBMISSION is jobs.MAX_ITEMS_PER_SUBMISSION
    assert limits.check_pathway_count(jobs.MAX_ITEMS_PER_SUBMISSION) is None
    assert "at most 5" in limits.check_pathway_count(6)
    assert limits.check_pathway_count(0)


def test_pmid_text_and_pdf_count_checks():
    assert limits.check_pmids(["1"] * limits.MAX_PMIDS) is None
    assert "PubMed IDs" in limits.check_pmids(["1"] * (limits.MAX_PMIDS + 1))
    assert limits.check_text(["x" * limits.MAX_TEXT_CHARS]) is None
    assert limits.check_text(["x" * limits.MAX_TEXT_CHARS, "y"])       # total across fields
    assert limits.check_pdf_count(limits.MAX_PDF_FILES) is None
    assert "PDF files" in limits.check_pdf_count(limits.MAX_PDF_FILES + 1)


def test_byte_magic_and_page_checks(tmp_path):
    assert limits.check_pdf_bytes(limits.MAX_PDF_BYTES) is None
    assert "MB" in limits.check_pdf_bytes(limits.MAX_PDF_BYTES + 1, "a.pdf")
    assert limits.check_total_bytes([limits.MAX_TOTAL_PDF_BYTES]) is None
    assert "together" in limits.check_total_bytes([limits.MAX_TOTAL_PDF_BYTES, 1])
    assert limits.check_pdf_magic(b"%PDF-1.7") is None
    assert "not a PDF" in limits.check_pdf_magic(b"<html>")
    p = tmp_path / "p.pdf"
    p.write_bytes(_pdf(3))
    assert limits.count_pdf_pages(p) == 3
    assert limits.check_pdf_pages(p) is None
    p.write_bytes(_pdf(limits.MAX_PDF_PAGES + 1))
    assert f"{limits.MAX_PDF_PAGES + 1} pages" in limits.check_pdf_pages(p)
    p.write_bytes(b"%PDF-1.4 garbage")
    assert limits.count_pdf_pages(p) is None
    assert "could not be read" in limits.check_pdf_pages(p)
    assert limits.count_pdf_pages(tmp_path / "missing.pdf") is None


def test_messages_never_contain_server_paths(tmp_path):
    p = tmp_path / "secret_dir" / "x.pdf"
    p.parent.mkdir()
    p.write_bytes(b"%PDF-1.4 garbage")
    assert str(tmp_path) not in limits.check_pdf_pages(p)


# -- upload saver ----------------------------------------------------------------------
def test_save_upload_roundtrip_uses_a_random_server_name(tmp_path):
    data = _pdf(2)
    s = limits.save_upload(io.BytesIO(data), tmp_path / "up", display_name="My Paper (v2).pdf")
    assert s.path.parent == tmp_path / "up" and s.path.read_bytes() == data and s.size == len(data)
    assert s.path.suffix == ".pdf" and len(s.path.stem) == 32
    assert s.display_name == "My Paper (v2).pdf"
    again = limits.save_upload(io.BytesIO(data), tmp_path / "up")
    assert again.path != s.path and again.display_name == "upload.pdf"


class _Counting(io.BytesIO):
    """Records how much the saver read, to prove the cap bites mid-stream."""
    reads = 0

    def read(self, n=-1):
        chunk = super().read(n)
        self.reads += len(chunk)
        return chunk


def test_save_upload_enforces_byte_cap_mid_stream_and_removes_partial(tmp_path):
    big = _Counting(b"%PDF-1.4\n" + b"0" * (10 * limits._CHUNK))
    with pytest.raises(limits.UploadRejected) as e:
        limits.save_upload(big, tmp_path, display_name="big.pdf", max_bytes=2 * limits._CHUNK)
    assert "big.pdf" in str(e.value) and "MB" in str(e.value) and str(tmp_path) not in str(e.value)
    assert big.reads <= 3 * limits._CHUNK                      # stopped early, did not drain
    assert list(tmp_path.iterdir()) == []                      # partial file removed


def test_save_upload_rejects_non_pdf_empty_and_too_many_pages(tmp_path, monkeypatch):
    for body, word in [(b"<html>hi</html>", "not a PDF"), (b"", "empty")]:
        with pytest.raises(limits.UploadRejected, match=word):
            limits.save_upload(io.BytesIO(body), tmp_path, display_name="a.pdf")
    monkeypatch.setattr(limits, "MAX_PDF_PAGES", 2)
    with pytest.raises(limits.UploadRejected, match="3 pages"):
        limits.save_upload(io.BytesIO(_pdf(3)), tmp_path, display_name="a.pdf")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("evil", [
    "../../etc/passwd.pdf", "..\\..\\Windows\\win.ini", "/abs/path/x.pdf", "C:\\x\\y.pdf",
    "a\x00b.pdf", "..", "", "x\r\nSet-Cookie: a=b.pdf", "<script>.pdf"])
def test_client_filename_never_reaches_the_path(tmp_path, evil):
    up = tmp_path / "up"
    s = limits.save_upload(io.BytesIO(_pdf()), up, display_name=evil)
    assert s.path.parent == up and s.path.resolve().is_relative_to(up.resolve())
    assert len(s.path.stem) == 32 and s.path.suffix == ".pdf"
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == [s.path.name]
    assert not any(c in s.display_name for c in "/\\\r\n\x00<>:")
    assert s.display_name and not s.display_name.startswith(".")


def test_save_uploads_all_or_nothing_count_and_total(tmp_path, monkeypatch):
    def f():
        return io.BytesIO(_pdf())

    ok = limits.save_uploads([("a.pdf", f()), ("b.pdf", f())], tmp_path)
    assert [s.display_name for s in ok] == ["a.pdf", "b.pdf"]
    with pytest.raises(limits.UploadRejected, match="PDF files"):
        limits.save_uploads([("x.pdf", f())] * (limits.MAX_PDF_FILES + 1), tmp_path / "n")
    assert not (tmp_path / "n").exists()
    one = len(_pdf())
    monkeypatch.setattr(limits, "MAX_TOTAL_PDF_BYTES", one + one // 2)
    monkeypatch.setattr(limits, "MAX_PDF_BYTES", one + 1)
    mine = tmp_path / "t"
    with pytest.raises(limits.UploadRejected, match="together"):
        limits.save_uploads([("a.pdf", f()), ("b.pdf", f())], mine)
    assert list(mine.iterdir()) == []                          # the first file was rolled back
    with pytest.raises(limits.UploadRejected, match="not a PDF"):
        limits.save_uploads([("a.pdf", f()), ("b.pdf", io.BytesIO(b"nope"))], mine)
    assert list(mine.iterdir()) == []
