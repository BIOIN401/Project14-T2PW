"""The public T2PW web site (W6): FastAPI + Jinja2 + HTMX over the W3 job store.

``create_app(db_path, work_root, base_url)`` builds the app; importing this module does
nothing. Run with ``python -m t2pw.web`` (see ``__main__.py``) or
``uvicorn --factory t2pw.web.app:create_app_from_env``.

Request path rule: nothing here resolves a PMID, fetches a paper or calls an LLM. A
submission stores its raw inputs (name + scope, PMIDs, saved PDF paths) in the job store
and the worker's ``run_pathway`` resolves them (R5).

Security shape: no login, so the job URL is the credential (32 hex chars of uuid4 from
``JobStore``); downloads are looked up by key in the job's own result and must resolve
inside that item's worker directory (``views.resolve_download``); templates autoescape
and never see a server path; every response carries CSP / nosniff / no-referrer.
"""

from __future__ import annotations

import logging
import math
import os
import secrets
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import StrictUndefined
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from t2pw.rag.eligibility import RequestedScope
from t2pw.service import organisms as organism_data
from t2pw.service.pathway import PathwayRequest
from t2pw.web import limits, views
from t2pw.web.jobs import MAX_ITEMS_PER_SUBMISSION, JobStore, QueueFull
from t2pw.web.notify import is_valid_email

log = logging.getLogger("t2pw.web.app")

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

MODES = (("name", "Pathway name"), ("pmids", "PubMed IDs"), ("pdfs", "PDF files"))
#: Largest request body: the PDF total plus room for the text fields and multipart framing.
MAX_BODY_BYTES = limits.MAX_TOTAL_PDF_BYTES + 1024 * 1024

CSP = ("default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; "
       "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
FILE_CSP = "sandbox; default-src 'none'"  # a served download can never run script or load anything


# -- ASGI: request body cap ------------------------------------------------------------


class BodyLimit:
    """413 when the body passes ``limit`` bytes: by Content-Length up front, and by
    counting for a chunked body (a client can lie about, or omit, Content-Length)."""

    def __init__(self, app, limit: int) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length", b"")
        over = declared.isdigit() and int(declared) > self.limit

        async def too_big() -> None:
            body = b"Request too large."
            await send({"type": "http.response.start", "status": 413, "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode()), (b"connection", b"close")]})
            await send({"type": "http.response.body", "body": body})

        if over:
            return await too_big()
        seen = 0
        tripped = False

        async def counting_receive():
            nonlocal seen, tripped
            msg = await receive()
            if msg["type"] == "http.request" and not tripped:
                seen += len(msg.get("body", b""))
                if seen > self.limit:
                    tripped = True
                    await too_big()
                    return {"type": "http.disconnect"}
            return msg

        async def guarded_send(msg) -> None:
            if not tripped:
                await send(msg)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except Exception:
            if not tripped:
                raise  # the app failed for its own reasons; after a trip the disconnect is expected


# -- app factory -----------------------------------------------------------------------


def create_app(db_path: Path | str, work_root: Path | str, base_url: str = "", *,
               rate_limit: int = limits.RATE_LIMIT_SUBMISSIONS,
               rate_window_s: float = limits.RATE_LIMIT_WINDOW_S,
               rate_secret: Optional[bytes] = None,
               trusted_proxies: Optional[str] = None,
               organisms_path: Optional[Path] = None,
               max_body_bytes: int = MAX_BODY_BYTES) -> FastAPI:
    """Build the site. Opens no DB, makes no directory, starts nothing."""
    if rate_secret is None and os.environ.get("T2PW_RATE_KEY_SECRET"):
        rate_secret = os.environ["T2PW_RATE_KEY_SECRET"].encode()
    if rate_secret is None:
        log.warning("T2PW_RATE_KEY_SECRET is not set: using a random per-process rate-limit secret")
        rate_secret = secrets.token_bytes(32)

    app = FastAPI(title="T2PW", docs_url=None, redoc_url=None, openapi_url=None)
    s = app.state
    s.db_path, s.work_root = Path(db_path), Path(work_root)
    s.base_url = base_url.rstrip("/")
    s.rate = dict(limit=rate_limit, window_s=rate_window_s, secret=rate_secret)
    s.trusted_proxies = trusted_proxies
    s.organisms_path = organisms_path or organism_data.DEFAULT_PATH

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.undefined = StrictUndefined
    # The one <script> tag on the site, built here (static prefix is server config).
    templates.env.globals["htmx_script"] = lambda: Markup(
        '<script src="/static/js/htmx.min.js" defer></script>')
    templates.env.filters["scrub"] = views.scrub
    s.templates = templates

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.add_middleware(BodyLimit, limit=max_body_bytes)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        h = resp.headers
        h.setdefault("Content-Security-Policy", CSP)
        h["X-Content-Type-Options"] = "nosniff"
        h["Referrer-Policy"] = "no-referrer"  # the job URL is the credential: never leak it
        h.setdefault("X-Frame-Options", "DENY")
        if request.url.path.startswith("/job/"):
            h["Cache-Control"] = "private, no-store"
            h["X-Robots-Tag"] = "noindex, nofollow"
        return resp

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        titles = {404: "Page not found", 413: "Request too large"}
        return _page(request, "error.html", status=exc.status_code,
                     title=titles.get(exc.status_code, "Something went wrong"),
                     message={404: "There is nothing at this address. Check the link you were given.",
                              }.get(exc.status_code, "The request could not be completed."))

    _routes(app)
    return app


def create_app_from_env() -> FastAPI:
    """``uvicorn --factory`` entry: T2PW_DB, T2PW_WORK_ROOT, T2PW_BASE_URL."""
    return create_app(os.environ.get("T2PW_DB", "t2pw_jobs.sqlite"),
                      os.environ.get("T2PW_WORK_ROOT", "t2pw_work"),
                      os.environ.get("T2PW_BASE_URL", ""),
                      trusted_proxies=os.environ.get("T2PW_TRUSTED_PROXIES"))


# -- helpers ---------------------------------------------------------------------------


def _page(request: Request, name: str, status: int = 200, headers: Optional[Dict[str, str]] = None,
          **ctx: Any) -> HTMLResponse:
    ctx.setdefault("active_nav", "")
    ctx.setdefault("static_url", "/static")
    return request.app.state.templates.TemplateResponse(
        request, name, ctx, status_code=status, headers=headers)


@contextmanager
def _open_store(app: FastAPI) -> Iterator[JobStore]:
    """One store per request: a sqlite connection belongs to the thread that made it, and
    sync handlers run in a thread pool."""
    store = JobStore(app.state.db_path)
    try:
        yield store
    finally:
        store.close()


def _organisms(app: FastAPI):
    return organism_data.load_organisms(app.state.organisms_path)  # lru_cached


def _limit_ctx() -> Dict[str, Any]:
    return dict(
        max_items=MAX_ITEMS_PER_SUBMISSION, max_pdf_files=limits.MAX_PDF_FILES,
        max_pdf_mb=limits.MAX_PDF_BYTES // 2**20, max_total_mb=limits.MAX_TOTAL_PDF_BYTES // 2**20,
        max_pdf_pages=limits.MAX_PDF_PAGES, max_pmids=limits.MAX_PMIDS,
        rate_per_day=limits.RATE_LIMIT_SUBMISSIONS)


def _blank_rows(n: int = MAX_ITEMS_PER_SUBMISSION) -> List[Dict[str, Any]]:
    return [dict(n=i, name="", organism="", pmids="", errors={}, open=i == 1) for i in range(1, n + 1)]


def _submit_ctx(**kw: Any) -> Dict[str, Any]:
    ctx = dict(mode="name", rows=_blank_rows(), consent=True, email="", email_error="",
               problems=[], modes=MODES, active_nav="submit", **_limit_ctx())
    ctx.update(kw)
    return ctx


def _sid_ok(sid: str) -> bool:
    return bool(views.SID_RE.match(sid))


# -- submission ------------------------------------------------------------------------


def _field(form, key: str, cap: int) -> Tuple[str, bool]:
    v = form.get(key, "")
    v = v.strip() if isinstance(v, str) else ""
    return v[:cap + 1], len(v) > cap


def _process_submission(app: FastAPI, form, remote: str, forwarded_for: Optional[str]):
    """Validate, rate-limit, save uploads, queue. Returns ``("ok", sid)`` or
    ``("form", status, ctx, headers)``. Order: cheap checks, then the rate limit, then the
    costly upload save (W8 wiring note), so a typo never burns a daily slot."""
    mode = form.get("mode") if isinstance(form.get("mode"), str) else "name"
    mode = mode if mode in dict(MODES) else "name"
    consent = bool(form.get("consent"))
    email, long_email = _field(form, "email", 254)
    rows = _blank_rows()
    problems: List[str] = []
    email_error = ""
    pdf_rows: Dict[int, List[UploadFile]] = {}
    parsed: Dict[int, Dict[str, Any]] = {}

    for row in rows:
        i = row["n"]
        name, name_long = _field(form, f"name_{i}", 200)
        organism, org_long = _field(form, f"organism_{i}", 200)
        pmids_text, pm_long = _field(form, f"pmids_{i}", 2000)
        files = [f for f in form.getlist(f"pdfs_{i}") if isinstance(f, UploadFile) and f.filename]
        row.update(name=name, organism=organism, pmids=pmids_text)
        if not (name or organism or files or pmids_text):
            continue  # an empty row is simply not part of the submission
        row["open"] = True
        err = row["errors"]
        if name_long:
            err["name"] = "Use at most 200 characters."
        if org_long:
            err["organism"] = "Use at most 200 characters."
        org = organism_data.get_by_label(organism, app.state.organisms_path) if organism else None
        if not organism:
            err["organism"] = "Choose an organism."
        elif org is None and "organism" not in err:
            near = views.search_organisms(_organisms(app), organism, 6)
            err["organism"] = ("This is not in the organism list." + (
                " Did you mean: " + "; ".join(o.label for o in near) + "?" if near else
                " Start typing a species name and pick one of the suggestions."))
        ids: List[str] = []
        if mode == "name" and not name:
            err["name"] = "Enter the pathway name."
        elif mode == "pmids":
            if pm_long:
                err["pmids"] = "Too long: enter up to %d PubMed IDs." % limits.MAX_PMIDS
            else:
                ids, bad = views.parse_pmids(pmids_text)
                if not ids and not bad:
                    err["pmids"] = "Enter at least one PubMed ID."
                elif bad:
                    err["pmids"] = "These are not PubMed IDs (digits only): " + ", ".join(bad[:5])
        elif mode == "pdfs":
            if not files:
                err["pdfs"] = "Choose at least one PDF file."
            else:
                for f in files:
                    if f.size is not None and (m := limits.check_pdf_bytes(f.size, views.scrub(f.filename, 80))):
                        err["pdfs"] = m
                        break
        parsed[i] = dict(name=name, org=org, ids=ids)
        pdf_rows[i] = files if mode == "pdfs" else []

    if long_email or (email and not is_valid_email(email)):
        email_error = "Enter a valid email address, or leave this blank."
    active = list(parsed)
    if msg := limits.check_pathway_count(len(active)):
        problems.append(msg)
    all_ids = [x for i in active for x in parsed[i]["ids"]]
    if msg := limits.check_pmids(all_ids):
        problems.append(msg)
    all_files = [f for i in active for f in pdf_rows[i]]
    if msg := limits.check_pdf_count(len(all_files)):
        problems.append(msg)
    sizes = [f.size for f in all_files if f.size is not None]
    if msg := limits.check_total_bytes(sizes):
        problems.append(msg)

    def again(status: int, headers: Optional[Dict[str, str]] = None):
        return ("form", status, _submit_ctx(mode=mode, rows=rows, consent=consent, email=email,
                                            email_error=email_error, problems=problems), headers or {})

    if problems or email_error or any(r["errors"] for r in rows):
        return again(400)

    # The rate limit: counts this attempt; a refused one is not recorded.
    key = limits.client_key(remote, forwarded_for, app.state.trusted_proxies)
    rl = limits.RateLimiter(app.state.db_path, **app.state.rate)
    try:
        ok, wait = rl.hit(key)
    finally:
        rl.close()
    if not ok:
        problems.append(limits.rate_limit_message(wait))
        return again(429, {"Retry-After": str(max(1, math.ceil(wait)))})

    upload_dir = app.state.work_root / "uploads" / uuid.uuid4().hex
    try:
        saved = limits.save_uploads([(f.filename, f.file) for f in all_files], upload_dir) if all_files else []
    except limits.UploadRejected as e:
        problems.append(str(e))
        return again(400)

    it = iter(saved)
    requests = []
    for i in active:
        mine = tuple(next(it).path for _ in pdf_rows[i])
        requests.append(PathwayRequest(
            pmids=tuple(parsed[i]["ids"]), pdf_paths=mine,
            scope=RequestedScope(requested_pathway=parsed[i]["name"],
                                 requested_organism=parsed[i]["org"].scope)))
    try:
        with _open_store(app) as store:
            sid = store.submit(requests, consent=consent, email=email or None)
    except QueueFull:
        _discard(upload_dir, saved)
        problems.append("The queue is full right now. Please try again in a few hours.")
        return again(503, {"Retry-After": "3600"})
    except BaseException:
        _discard(upload_dir, saved)
        raise
    return ("ok", sid)


def _discard(upload_dir: Path, saved) -> None:
    for s in saved:
        s.path.unlink(missing_ok=True)
    try:
        upload_dir.rmdir()
    except OSError:
        pass


# -- routes ----------------------------------------------------------------------------


def _routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return _page(request, "submit.html", **_submit_ctx())

    @app.post("/submit")
    async def submit(request: Request):
        form = await request.form()
        try:
            remote = request.client.host if request.client else ""
            out = await run_in_threadpool(
                _process_submission, request.app, form, remote, request.headers.get("x-forwarded-for"))
        finally:
            await form.close()
        if out[0] == "ok":
            return RedirectResponse(views.job_url("", out[1]), status_code=303)
        _, status, ctx, headers = out
        return _page(request, "submit.html", status=status, headers=headers, **ctx)

    @app.get("/organisms", response_class=HTMLResponse)
    def organism_options(request: Request):
        """Datalist <option>s for the typed text (HTMX sends the input's own name=value)."""
        q = next((v for k, v in request.query_params.items() if k.startswith("organism_")), "")[:100]
        found = views.search_organisms(_organisms(request.app), q)
        return _page(request, "organism_options.html", organisms=found)

    def _load(request: Request, sid: str):
        if not _sid_ok(sid):
            raise StarletteHTTPException(404)
        with _open_store(request.app) as store:
            sub = store.submission(sid)
            if sub is None:
                raise StarletteHTTPException(404)
            views_ = [views.item_view(it, store.attempt_history(it["id"]), sid) for it in sub["items"]]
        return sub, views_

    def _job_ctx(request: Request, sid: str):
        sub, items = _load(request, sid)
        done = all(v["finished"] for v in items)
        return dict(sid=sid, items=items, all_done=done, submitted=views.fmt_time(sub["created_at"]),
                    consent=sub["consent"], has_email=bool(sub["email"]),
                    job_url=views.job_url(request.app.state.base_url, sid), active_nav="")

    @app.get("/job/{sid}", response_class=HTMLResponse)
    def job(request: Request, sid: str):
        return _page(request, "job.html", **_job_ctx(request, sid))

    @app.get("/jobs/{sid}")
    def job_legacy_link(sid: str):
        """W7's email link shape (/jobs/<id>) until postrun.py switches to views.job_url."""
        if not _sid_ok(sid):
            raise StarletteHTTPException(404)
        return RedirectResponse(views.job_url("", sid), status_code=303)

    @app.get("/job/{sid}/status", response_class=HTMLResponse)
    def job_status(request: Request, sid: str):
        return _page(request, "_job_items.html", **_job_ctx(request, sid))

    def _item(request: Request, sid: str, pos: str):
        if not _sid_ok(sid) or not pos.isdigit() or len(pos) > 2:
            raise StarletteHTTPException(404)
        with _open_store(request.app) as store:
            sub = store.submission(sid)
            if sub is None or int(pos) >= len(sub["items"]):
                raise StarletteHTTPException(404)
            item = sub["items"][int(pos)]
            return item, views.item_view(item, store.attempt_history(item["id"]), sid)

    @app.get("/job/{sid}/{pos}", response_class=HTMLResponse)
    def result(request: Request, sid: str, pos: str):
        _, view = _item(request, sid, pos)
        return _page(request, "result.html", sid=sid, it=view, active_nav="")

    @app.get("/job/{sid}/{pos}/file/{kind}")
    @app.get("/job/{sid}/{pos}/file/{kind}/{name}")
    def download(request: Request, sid: str, pos: str, kind: str, name: str = ""):
        item, _ = _item(request, sid, pos)
        found = views.resolve_download(item, kind, name, request.app.state.work_root)
        if found is None:
            raise StarletteHTTPException(404)
        path, filename, media, inline = found
        return FileResponse(path, media_type=media, filename=filename,
                            content_disposition_type="inline" if inline else "attachment",
                            headers={"Content-Security-Policy": FILE_CSP})

    @app.get("/examples", response_class=HTMLResponse)
    def examples(request: Request):
        return _page(request, "examples.html", active_nav="examples")

    @app.get("/help", response_class=HTMLResponse)
    def help_page(request: Request):
        return _page(request, "help.html", active_nav="help", **_limit_ctx())

    @app.get("/about", response_class=HTMLResponse)
    def about(request: Request):
        return _page(request, "about.html", active_nav="about")
