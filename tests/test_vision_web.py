"""Web routes and server runner for scoped views (V10). Offline: FastAPI TestClient, a fake child-process runner.

One test (``test_real_child_process``) runs the real child on a stored fixture; it needs the gated run and is skipped
unless T2PW_VISION_GATED_RUN=1.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from t2pw.service.pathway import PathwayResult
from t2pw.vision import serve, web_routes
from t2pw.web.app import create_app
from t2pw.web.jobs import JobStore, result_to_dict

FIXTURE = Path(__file__).parent / "data" / "vision" / "generate" / "run56_final_mapped.json"
SID_RX = re.compile(r"/job/([0-9a-f]{32})$")


class Site:
    def __init__(self, tmp: Path):
        self.db, self.work, self.tmp = tmp / "jobs.sqlite", tmp / "work", tmp
        self.app = create_app(self.db, self.work, "https://t2pw.test", rate_limit=100, rate_secret=b"k")
        self.client = TestClient(self.app, follow_redirects=False)
        self.calls = []

    def job(self, *, released: bool = True, payload: bool = True, finish: bool = True) -> str:
        r = self.client.post("/submit", data={"mode": "name", "name_1": "glycolysis",
                                              "organism_1": "Homo sapiens", "consent": "1"})
        assert r.status_code == 303, r.text
        sid = SID_RX.search(r.headers["location"]).group(1)
        if not finish:
            return sid
        st = JobStore(self.db)
        item = st.claim_next(3600)
        out = self.work / ("item-%d" % item["id"]) / "attempt-1" / "out"
        out.mkdir(parents=True)
        (out / "pathway.pwml").write_bytes(b"<pwml/>")
        if payload:
            (out / "final_mapped.json").write_bytes(FIXTURE.read_bytes())
        d = result_to_dict(PathwayResult(status="pass"))
        d.update(release_status={"status": "release_ready" if released else "diagnostic_only"},
                 pwml_path=str(out / "pathway.pwml"))
        st.finish(item["id"], 1, d, 0)
        st.close()
        self.out = out
        return sid


@pytest.fixture
def site(tmp_path, monkeypatch):
    s = Site(tmp_path)

    def fake(run_dir, slug, fmt, **kw):  # stands in for the child process: writes the cache file a child would
        s.calls.append((Path(run_dir), slug, fmt))
        p = serve.cache_dir(run_dir) / ("%s.%s" % (serve.fs_slug(slug), fmt))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"FAKE-" + fmt.encode())
        return p

    monkeypatch.setattr(serve, "run_generate", fake)
    monkeypatch.setenv("T2PW_OFFLINE_CURATOR", "1")  # rescope: keyword fallback, no network
    return s


# -- panel on the result page ------------------------------------------------------------------------

def test_result_page_carries_the_panel_loader_and_the_panel_lists_every_format(site):
    sid = site.job()
    page = site.client.get("/job/%s/0" % sid)
    assert page.status_code == 200 and 'hx-get="/job/%s/0/vision"' % sid in page.text
    panel = site.client.get("/job/%s/0/vision" % sid)
    assert panel.status_code == 200 and "Paper and immediate context" in panel.text
    for fmt, (label, _, _) in web_routes.FORMATS.items():
        assert "/job/%s/0/download/%s?scope=default" % (sid, fmt) in panel.text and "Download " + label in panel.text
    assert 'name="message"' in panel.text and 'action="/job/%s/0/rescope"' % sid in panel.text
    assert site.calls == [], "the panel itself must not run a generation"


def test_no_panel_without_a_stored_payload_or_release(site):
    for i, kw in enumerate((dict(payload=False), dict(released=False), dict(finish=False))):
        (site.tmp / ("s%d" % i)).mkdir()
        s = Site(site.tmp / ("s%d" % i))
        sid = s.job(**kw)
        assert s.client.get("/job/%s/0/vision" % sid).text == ""
        for tail in ("view", "download/pwml"):
            assert s.client.get("/job/%s/0/%s" % (sid, tail)).status_code == 404
    assert site.calls == []


def test_unknown_job_position_and_bad_ids_are_404(site):
    sid = site.job()
    for path in ("/job/%s/5/view" % sid, "/job/%s/x/view" % sid, "/job/%s/0/vision" % ("0" * 32),
                 "/job/nothex/0/view", "/job/%s/0/download/pwml" % ("f" * 32)):
        assert site.client.get(path).status_code == 404, path
    assert site.client.post("/job/%s/9/rescope" % sid, data={"scope": "default"}).status_code == 404


# -- view ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scope,slug,label", [(None, "default", "Paper and immediate context"), ("paper", "paper", "Paper only"),
                                              ("full", "full", "Full pathway"), ("hops:3", "hops:3", "3 steps from the paper"),
                                              ("hops:1", "default", "Paper and immediate context")])
def test_view_scopes(site, scope, slug, label):
    sid = site.job()
    r = site.client.get("/job/%s/0/view" % sid, params={"scope": scope} if scope else None)
    assert r.status_code == 200 and label in r.text
    assert site.calls == [(site.out.resolve(), slug, "svg")], "run dir is the job's own last attempt"
    assert "/job/%s/0/download/svg?scope=%s&amp;inline=1" % (sid, slug) in r.text
    assert 'name="scope" value="%s"' % slug in r.text


BAD_SCOPES = ["bogus", "hops:11", "hops:-1", "hops:", "hops:1.5", "paper_only", "Default", " default", "default\n",
              "hops:\u0661", "../../etc", "", "paper;full", "hops:99999999999999999999"]


@pytest.mark.parametrize("bad", BAD_SCOPES)
def test_bad_scope_is_400_everywhere(site, bad):
    sid = site.job()
    assert site.client.get("/job/%s/0/view" % sid, params={"scope": bad}).status_code == 400
    assert site.client.get("/job/%s/0/download/pwml" % sid, params={"scope": bad}).status_code == 400
    assert site.client.post("/job/%s/0/rescope" % sid, data={"scope": bad, "message": ""}).status_code == 400
    assert site.calls == []


# -- downloads -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,media,ext", [("pwml", "application/xml", "pwml"), ("sbml", "application/xml", "sbml.xml"),
                                           ("sbgn", "application/xml", "sbgn"), ("kgml", "application/xml", "kgml"),
                                           ("svg", "image/svg+xml", "svg"), ("html", "text/html", "html")])
def test_download_type_filename_and_headers(site, fmt, media, ext):
    sid = site.job()
    r = site.client.get("/job/%s/0/download/%s" % (sid, fmt), params={"scope": "hops:2"})
    assert r.status_code == 200 and r.content == b"FAKE-" + fmt.encode()
    assert r.headers["content-type"].startswith(media)
    cd = r.headers["content-disposition"]
    assert cd.startswith("attachment") and 'filename="pathway-1-hops2.%s"' % ext in cd
    assert "sandbox" in r.headers["content-security-policy"] and r.headers["x-content-type-options"] == "nosniff"
    assert site.calls == [(site.out.resolve(), "hops:2", fmt)]


def test_svg_inline_only_for_svg(site):
    sid = site.job()
    r = site.client.get("/job/%s/0/download/svg" % sid, params={"inline": "1"})
    assert r.status_code == 200 and r.headers["content-disposition"].startswith("inline")
    for bad in (("pwml", "1"), ("svg", "0"), ("svg", "yes")):
        assert site.client.get("/job/%s/0/download/%s" % (sid, bad[0]), params={"inline": bad[1]}).status_code == 400


@pytest.mark.parametrize("fmt", ["docx", "json", "PWML", "pwml.", "%2e%2e%2f%2e%2e%2fetc%2fpasswd", "..%5c..%5cwin.ini",
                                 "pwml%00", "pwml%0A", "x" * 500, "~"])
def test_unknown_or_traversing_fmt_is_never_served(site, fmt):
    sid = site.job()
    r = site.client.get("/job/%s/0/download/%s" % (sid, fmt))
    assert r.status_code in (400, 404)
    assert site.calls == []


def test_refused_export_gives_a_message_and_no_file(site, monkeypatch):
    def refuse(run_dir, slug, fmt, **kw):
        raise serve.Refused("export refused: no reaction survives the strict gate at C:\\secret\\x.json")

    monkeypatch.setattr(serve, "run_generate", refuse)
    sid = site.job()
    v = site.client.get("/job/%s/0/view" % sid)
    assert v.status_code == 200 and "refused this view" in v.text and "no reaction survives" in v.text
    assert "/download/" not in v.text and "figure__img" not in v.text, "no file and no diagram offered"
    assert "[server path]" in v.text and "secret" not in v.text, "server paths are scrubbed"
    d = site.client.get("/job/%s/0/download/pwml" % sid)
    assert d.status_code == 422 and "no reaction survives" in d.text and "content-disposition" not in d.headers
    monkeypatch.setattr(serve, "run_generate", lambda *a, **k: (_ for _ in ()).throw(serve.ServeError("took too long")))
    assert "could not be generated" in site.client.get("/job/%s/0/view" % sid).text


# -- re-scope ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scope,message,expect", [
    ("default", "make this more detailed", "hops:2"),
    ("default", "only what the paper says", "paper"),
    ("default", "show the whole pathway", "full"),
    ("hops:2", "make it simpler", "default"),
    ("hops:10", "more detail please", "full"),   # past the cap: the whole pathway
    ("paper", "more detail", "default"),
    ("hops:3", "", "hops:3"),                     # nothing to say: same view
    ("hops:3", "   ", "hops:3"),
    ("full", "what a lovely day", "full"),        # unsure: unchanged
])
def test_rescope_redirects_to_the_new_scope(site, scope, message, expect):
    sid = site.job()
    r = site.client.post("/job/%s/0/rescope" % sid, data={"scope": scope, "message": message})
    assert r.status_code == 303 and r.headers["location"] == "/job/%s/0/view?scope=%s" % (sid, expect)
    assert site.client.get(r.headers["location"]).status_code == 200


def test_rescope_hostile_input(site):
    sid = site.job()
    r = site.client.post("/job/%s/0/rescope" % sid, data={"scope": "default", "message": "<script>" + "more " * 400})
    assert r.status_code == 303 and r.headers["location"].startswith("/job/%s/0/view?scope=" % sid)
    assert "<" not in r.headers["location"] and "\n" not in r.headers["location"]
    r = site.client.post("/job/%s/0/rescope" % sid, data={"message": "more detail"})  # no scope field: default
    assert r.headers["location"].endswith("scope=hops:2")


def test_rescope_uses_the_scope_layer_and_survives_its_failure_modes(site, monkeypatch):
    from t2pw.vision.view import PAPER_ONLY
    seen = []
    monkeypatch.setattr(web_routes, "rescope", lambda cur, msg, llm=None: (seen.append((cur, msg, llm)), (PAPER_ONLY, "x"))[1])
    sid = site.job()
    r = site.client.post("/job/%s/0/rescope" % sid, data={"scope": "hops:4", "message": "  strictly the paper  "})
    assert r.headers["location"].endswith("scope=paper")
    assert seen[0][0].max_hops == 4 and seen[0][1] == "strictly the paper"
    with pytest.raises(Exception):
        seen[0][2]("s", "u")  # the llm handed to rescope() always fails: keyword table only


def test_rescope_never_reaches_the_real_llm(site, monkeypatch):
    from t2pw.vision import scope as scope_mod
    monkeypatch.delenv("T2PW_OFFLINE_CURATOR")
    monkeypatch.setattr(scope_mod, "_default_llm", lambda: pytest.fail("the real LLM client was asked for"))
    sid = site.job()
    r = site.client.post("/job/%s/0/rescope" % sid, data={"scope": "default", "message": "make this more detailed"})
    assert r.headers["location"].endswith("scope=hops:2")


def test_generate_offline_false_swaps_no_environment(monkeypatch):
    from types import SimpleNamespace
    from t2pw.pipeline import strict_quarantine
    from t2pw.service import postpipeline
    from t2pw.vision import generate as G
    monkeypatch.setenv("PATHBANK_DB_HOST", "db.example")
    monkeypatch.delenv("T2PW_OFFLINE_CURATOR", raising=False)
    seen = []

    def spy(*a, **k):
        seen.append((os.environ.get("PATHBANK_DB_HOST"), os.environ.get("T2PW_OFFLINE_CURATOR")))
        return {"ok": True, "xml_bytes": b"<x/>", "pwml_ir": {}}

    def spy_q(*a, **k):
        seen.append((os.environ.get("PATHBANK_DB_HOST"), os.environ.get("T2PW_OFFLINE_CURATOR")))
        return SimpleNamespace(ok=True, payload={}, refusal_reasons=[])

    monkeypatch.setattr(postpipeline, "run_pwml_export", spy)
    monkeypatch.setattr(strict_quarantine, "quarantine_and_close", spy_q)
    before = dict(os.environ)
    G._export_pwml({}, {}, offline=False)
    G._kept({}, {}, offline=False)
    assert seen == [("db.example", None)] * 2 and dict(os.environ) == before, "real DB settings, environment untouched"
    seen.clear()
    G._export_pwml({}, {})  # the default is unchanged: DB blanked, curator off, then restored
    G._kept({}, {})
    assert seen == [("", "1")] * 2 and dict(os.environ) == before


def test_rescope_refuses_cross_site_posts(site):
    sid = site.job()
    r = site.client.post("/job/%s/0/rescope" % sid, data={"scope": "default", "message": "more"},
                         headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = site.client.post("/job/%s/0/rescope" % sid, data={"scope": "default", "message": "more"},
                         headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_run_dir_gate(tmp_path):
    out = tmp_path / "item-7" / "attempt-2" / "out"
    out.mkdir(parents=True)
    (out / "final_mapped.json").write_text("{}")
    item = dict(id=7, attempt=2, state="finished", result={"release_status": {"status": "review_required"}})
    assert web_routes.run_dir_of(tmp_path, item) == out.resolve()
    assert web_routes.run_dir_of(tmp_path, dict(item, attempt=1)) is None
    assert web_routes.run_dir_of(tmp_path, dict(item, state="running")) is None
    assert web_routes.run_dir_of(tmp_path, dict(item, result={"release_status": {"status": "diagnostic_only"}})) is None
    assert web_routes.run_dir_of(tmp_path, dict(item, result=None)) is None


def test_scope_slugs_round_trip():
    for slug in ("paper", "default", "full", "hops:0", "hops:2", "hops:10"):
        assert web_routes.scope_arg(slug) == slug
    assert web_routes.scope_arg("hops:1") == "default" and web_routes.scope_arg(None) == "default"


# -- serve.py: child-process runner ---------------------------------------------------------------------

def _fake_child(calls, *, rc=0, delay=0.0, tracker=None, refuse=None):
    def run(cmd, **kw):
        calls.append((cmd, kw))
        if tracker is not None:
            tracker["now"] += 1
            tracker["max"] = max(tracker["max"], tracker["now"])
        time.sleep(delay)
        cache, slug = Path(cmd[5]), serve.fs_slug(cmd[4])
        if refuse:
            (cache / (slug + ".refused")).write_text(refuse, encoding="utf-8")
        elif rc == 0:
            for f in serve.PUBLIC_FORMATS:
                (cache / ("%s.%s" % (slug, f))).write_bytes(b"x-" + f.encode())
        if tracker is not None:
            tracker["now"] -= 1
        return subprocess.CompletedProcess(cmd, 2 if refuse else rc, "", "boom: it broke\n" if rc else "")
    return run


def test_run_generate_spawns_one_child_with_an_explicit_env_and_leaves_os_environ_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("PATHBANK_DB_HOST", "db.example")
    monkeypatch.delenv("T2PW_OFFLINE_CURATOR", raising=False)
    before = dict(os.environ)
    calls = []
    monkeypatch.setattr(serve.subprocess, "run", _fake_child(calls))
    p = serve.run_generate(tmp_path, "hops:2", "sbml", timeout=33)
    assert p == serve.cache_dir(tmp_path) / "hops2.sbml" and p.parent.name.startswith("v") and p.read_bytes() == b"x-sbml"
    (cmd, kw), = calls
    assert cmd[:3] == [sys.executable, "-m", "t2pw.vision.serve"] and cmd[3:] == [str(tmp_path), "hops:2", str(serve.cache_dir(tmp_path))]
    assert kw["timeout"] == 33 and kw["env"] is not os.environ
    assert kw["env"]["PATHBANK_DB_HOST"] == "db.example", "production DB variables pass through as they are"
    assert kw["env"]["T2PW_OFFLINE_CURATOR"] == "1"
    assert Path(kw["env"]["PYTHONPATH"].split(os.pathsep)[0]).joinpath("t2pw").is_dir()
    assert dict(os.environ) == before, "the parent's environment is unchanged"
    # every format came from that one child, and a second request starts nothing
    for f in serve.PUBLIC_FORMATS:
        assert serve.run_generate(tmp_path, "hops:2", f).exists()
    assert len(calls) == 1


def test_run_generate_failure_modes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(serve.subprocess, "run", _fake_child(calls, refuse="strict export refused: nothing to export"))
    with pytest.raises(serve.Refused, match="nothing to export"):
        serve.run_generate(tmp_path, "full", "pwml")
    with pytest.raises(serve.Refused, match="nothing to export"):  # remembered: no second child
        serve.run_generate(tmp_path, "full", "svg")
    assert len(calls) == 1 and not (serve.cache_dir(tmp_path) / "full.pwml").exists()

    monkeypatch.setattr(serve.subprocess, "run", _fake_child([], rc=1))
    with pytest.raises(serve.ServeError, match="boom: it broke") as e:
        serve.run_generate(tmp_path, "paper", "pwml")
    assert not isinstance(e.value, serve.Refused)

    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(serve.subprocess, "run", hang)
    with pytest.raises(serve.ServeError, match="longer than 5 seconds"):
        serve.run_generate(tmp_path, "default", "pwml", timeout=5)
    with pytest.raises(ValueError):
        serve.run_generate(tmp_path, "default", "json")
    assert serve._LOCK.acquire(blocking=False), "every failure path released the lock"
    serve._LOCK.release()


def test_concurrent_requests_run_one_child_at_a_time(tmp_path, monkeypatch):
    tracker, calls = {"now": 0, "max": 0}, []
    monkeypatch.setattr(serve.subprocess, "run", _fake_child(calls, delay=0.05, tracker=tracker))
    errs = []

    def go(scope):
        try:
            serve.run_generate(tmp_path, scope, "pwml")
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=go, args=(s,)) for s in ("paper", "default", "full", "hops:2", "hops:3", "default", "default")]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert not errs and tracker["max"] == 1, "never two children at once"
    assert len(calls) == 5, "waiters for the same scope find the first child's files and start nothing"


def test_two_http_requests_in_flight_are_serialised(site, monkeypatch):
    sid = site.job()
    tracker, calls = {"now": 0, "max": 0}, []
    monkeypatch.undo()  # the real run_generate again, with a fake child under it
    monkeypatch.setenv("T2PW_OFFLINE_CURATOR", "1")
    monkeypatch.setattr(serve.subprocess, "run", _fake_child(calls, delay=0.1, tracker=tracker))
    codes = []

    def get(scope):
        codes.append(site.client.get("/job/%s/0/download/pwml" % sid, params={"scope": scope}).status_code)

    ts = [threading.Thread(target=get, args=(s,)) for s in ("paper", "full")]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert codes == [200, 200] and tracker["max"] == 1 and len(calls) == 2


# -- the real child (gated run only) ----------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("T2PW_VISION_GATED_RUN") != "1", reason="needs the gated run (real export in a child process)")
def test_real_child_process(tmp_path):
    run = tmp_path / "item-1" / "attempt-1" / "out"
    run.mkdir(parents=True)
    (run / "final_mapped.json").write_bytes(FIXTURE.read_bytes())
    before = dict(os.environ)
    try:
        pwml = serve.run_generate(run, "default", "pwml", timeout=600)
    except serve.Refused as e:  # a refused view is a valid verdict: a message, no file
        assert str(e).strip() and (serve.cache_dir(run) / "default.refused").exists()
        assert not (serve.cache_dir(run) / "default.pwml").exists()
    else:
        assert pwml.read_bytes().lstrip().startswith(b"<?xml")
        for f in serve.PUBLIC_FORMATS:
            assert (serve.cache_dir(run) / ("default." + f)).stat().st_size > 0
        assert serve.run_generate(run, "default", "svg").read_bytes().lstrip().startswith(b"<svg")
    assert dict(os.environ) == before
