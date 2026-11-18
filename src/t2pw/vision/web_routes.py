"""Web routes for the scoped views (docs/VISION.md section 2 steps 4-5, V10).

``register(app, item_lookup, page, cross_site)`` is called once from ``t2pw.web.app._routes`` with that module's own
helpers, so the job lookup and access rules are exactly those of ``/job/{sid}/{pos}/file/{kind}``: a sid that is not
32 hex, a bad or out-of-range position and an unknown job are 404.

    GET  /job/{sid}/{pos}/vision                   the panel (HTMX loads it into the result page)
    GET  /job/{sid}/{pos}/view?scope=S             the result page with the panel open and a diagram of scope S
    GET  /job/{sid}/{pos}/download/{fmt}?scope=S   pwml | sbml | sbgn | kgml | svg | html
    POST /job/{sid}/{pos}/rescope                  form: message, scope -> 303 to the view with the new scope

S is paper | default | full | hops:N (N 0..10); anything else is 400. Nothing is read from a user-supplied path: the
files come from ``<attempt out dir>/vision_cache`` of the job's own finished attempt (``serve.run_generate``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as HTTPError
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, RedirectResponse

from t2pw.vision import serve
from t2pw.vision.generate import parse_scope
from t2pw.vision.scope import MAX_HOPS, rescope
from t2pw.vision.view import ScopeSpec

FORMATS = {  # fmt -> (label, media type, file extension)
    "pwml": ("PWML", "application/xml", "pwml"),
    "sbml": ("SBML", "application/xml", "sbml.xml"),
    "sbgn": ("SBGN", "application/xml", "sbgn"),
    "kgml": ("KGML", "application/xml", "kgml"),
    "svg": ("SVG", "image/svg+xml", "svg"),
    "html": ("HTML", "text/html", "html"),
}
SCOPES = (("paper", "Paper only"), ("default", "Paper and immediate context"), ("full", "Full pathway"))
_SCOPE_RE = re.compile(r"paper|default|full|hops:[0-9]{1,2}")
MAX_MESSAGE = 300


def _no_llm(system: str, user: str) -> str:
    # No LLM on a request thread (VD-15): rescope() treats a failing llm as "use the keyword table".
    # ponytail: LLM re-scoping can move into the serialised child process later.
    raise RuntimeError("no LLM on the request path")


def slug_of(spec: ScopeSpec) -> str:
    if spec.paper_only:
        return "paper"
    if spec.max_hops is None or spec.max_hops > MAX_HOPS:  # deeper than the cap is the whole pathway
        return "full"
    return "default" if spec.max_hops == 1 else "hops:%d" % spec.max_hops


def label_of(slug: str) -> str:
    if slug.startswith("hops:"):
        n = int(slug[5:])
        return "%d step%s from the paper" % (n, "" if n == 1 else "s")
    return dict(SCOPES)[slug]


def scope_arg(text: Optional[str]) -> str:
    """The canonical slug for a request's scope text; 400 for anything but paper|default|full|hops:N."""
    t = "default" if text is None else text
    if not _SCOPE_RE.fullmatch(t) or (t.startswith("hops:") and int(t[5:]) > MAX_HOPS):
        raise HTTPError(400)
    return slug_of(parse_scope(t))


def run_dir_of(work_root, item) -> Optional[Path]:
    """The output directory of the item's last attempt when it has a stored payload and was released, else None.
    The same release gate as the PWML download; the payload must resolve inside the item's own directory."""
    rs = ((item.get("result") or {}).get("release_status") or {}).get("status")
    if item.get("state") != "finished" or rs not in ("release_ready", "review_required"):
        return None
    base = (Path(work_root) / ("item-%s" % item["id"])).resolve()
    out = base / ("attempt-%s" % item["attempt"]) / "out"
    try:
        (out / "final_mapped.json").resolve(strict=True).relative_to(base)
    except (OSError, ValueError):
        return None
    return out.resolve() if (out / "final_mapped.json").is_file() else None


def register(app, item_lookup: Callable, page: Callable, cross_site: Callable) -> None:
    """Tests replace ``serve.run_generate`` and this module's ``rescope`` (looked up at call time)."""
    from t2pw.web.app import FILE_CSP  # lazy: web.app imports this module's register() during create_app
    from t2pw.web.views import scrub

    def _find(request: Request, sid: str, pos: str):
        item, view = item_lookup(request, sid, pos)  # 404 on a bad sid / position / unknown job
        return view, run_dir_of(request.app.state.work_root, item)

    def _target(request: Request, sid: str, pos: str):
        view, run_dir = _find(request, sid, pos)
        if run_dir is None:
            raise HTTPError(404)  # nothing stored for this job (not finished, not released, or no payload)
        return view, run_dir

    def _panel(view, slug: str, message: str = "", preview: bool = False) -> dict:
        base = view["href"]
        files = [] if message else [
            dict(label=lbl, href="%s/download/%s?scope=%s" % (base, f, slug), name="pathway-%d-%s.%s" % (
                view["number"], serve.fs_slug(slug), ext)) for f, (lbl, _, ext) in FORMATS.items()]
        return dict(scope=slug, label=label_of(slug), message=message, files=files, scopes=SCOPES,
                    view_href="%s/view?scope=%s" % (base, slug), maxlen=MAX_MESSAGE,
                    preview="%s/download/svg?scope=%s&inline=1" % (base, slug) if preview and not message else "")

    @app.get("/job/{sid}/{pos}/vision", response_class=HTMLResponse)
    def vision_panel(request: Request, sid: str, pos: str):
        view, run_dir = _find(request, sid, pos)
        if run_dir is None:
            return HTMLResponse("")  # nothing stored for this job: the loader on the result page just goes away
        return page(request, "_vision_panel.html", it=view, vp=_panel(view, "default"))

    @app.get("/job/{sid}/{pos}/view", response_class=HTMLResponse)
    def vision_view(request: Request, sid: str, pos: str, scope: Optional[str] = None):
        view, run_dir = _target(request, sid, pos)
        slug, message = scope_arg(scope), ""
        try:
            serve.run_generate(run_dir, slug, "svg")
        except serve.Refused as e:
            message = "The PWML export refused this view, so no file was made for it: %s" % e
        except serve.ServeError as e:
            message = "This view could not be generated: %s" % e
        return page(request, "result.html", sid=sid, it=view, vp=_panel(view, slug, message, preview=True))

    @app.get("/job/{sid}/{pos}/download/{fmt}")
    def vision_download(request: Request, sid: str, pos: str, fmt: str, scope: Optional[str] = None,
                        inline: Optional[str] = None):
        view, run_dir = _target(request, sid, pos)
        if fmt not in FORMATS or inline not in (None, "1") or (inline and fmt != "svg"):
            raise HTTPError(400)
        slug = scope_arg(scope)
        try:
            path = serve.run_generate(run_dir, slug, fmt)
        except serve.Busy as e:
            return page(request, "error.html", status=503, headers={"Retry-After": "30"},
                        title="Busy", message=scrub(str(e)))
        except serve.ServeError as e:  # a refused export is a ServeError too: a message, no file
            return page(request, "error.html", status=422, title="No file for this view", message=scrub(str(e)))
        _, media, ext = FORMATS[fmt]
        return FileResponse(path, media_type=media, filename="pathway-%d-%s.%s" % (view["number"], serve.fs_slug(slug), ext),
                            content_disposition_type="inline" if inline else "attachment",
                            headers={"Content-Security-Policy": FILE_CSP})

    def _new_scope(request: Request, sid: str, pos: str, scope: Any, message: Any) -> str:
        _target(request, sid, pos)  # same 404s as the other routes, before the (possibly slow) re-scope
        slug = scope_arg(scope if isinstance(scope, str) else None)
        text = message.strip()[:MAX_MESSAGE] if isinstance(message, str) else ""
        if text:
            slug = slug_of(rescope(parse_scope(slug), text, llm=_no_llm)[0])
        return "/job/%s/%s/view?scope=%s" % (sid, pos, quote(slug, safe=":"))

    @app.post("/job/{sid}/{pos}/rescope")
    async def vision_rescope(request: Request, sid: str, pos: str):
        if cross_site(request):
            raise HTTPError(403)
        form = await request.form()
        try:
            url = await run_in_threadpool(_new_scope, request, sid, pos, form.get("scope"), form.get("message"))
        finally:
            await form.close()
        return RedirectResponse(url, status_code=303)
