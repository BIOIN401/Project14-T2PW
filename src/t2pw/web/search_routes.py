"""Smart-search routes (W10). ``register(app, page, cross_site, rate_hit, open_store)`` is called
once from ``t2pw.web.app._routes`` with that module's helpers.

    GET|POST /search                  one box: classify, then show a PathBank pathway, a short list of
                                      PathBank choices, or a prefilled run form (paper, text, no match)
    GET      /search/run              the run form, prefilled from the query string (no job is made)
    POST     /search/run              queue ONE pipeline job (same rate limit and store as /submit)
    GET      /pathbank/{PW}           one PathBank pathway; no pipeline job
    GET      /pathbank/{PW}/diagram   PathBank's own SVG, else a render of PathBank's PWML (local mirror)
    GET      /pathbank/{PW}/pwml      PathBank's PWML (local mirror)

A PathBank answer never creates a job. The LLM is called only for a sentence-like query that
found nothing deterministically, and at most ``SEARCH_LLM_PER_DAY`` times per visitor per window.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as HTTPError

from t2pw.rag.eligibility import RequestedScope
from t2pw.service import organisms as organism_data
from t2pw.service.pathway import PathwayRequest
from t2pw.web import limits, pathbank, smart_search as ss, views
from t2pw.web.jobs import QueueFull
from t2pw.web.notify import is_valid_email

log = logging.getLogger("t2pw.web.search")

SEARCH_LLM_PER_DAY = 30
GET_QUERY_CAP = 2000  # a long passage must come by POST (URL length)
FILE_CSP = "sandbox; default-src 'none'; img-src data:; style-src 'unsafe-inline'"  # PathBank SVGs embed data: art
CREDIT = ("Pathway data and diagrams: PathBank (Wishart Lab, University of Alberta). "
          "Please cite PathBank when you use this pathway.")


def species_ids(org, organisms) -> Optional[List[int]]:
    """Every PathBank species id whose label starts with ``org``'s binomial (strains included)."""
    if org is None:
        return None
    stem = " ".join(org.label.split()[:2]).casefold()
    return [o.id for o in organisms if o.label.casefold() == stem or o.label.casefold().startswith(stem + " ")]


def _org_label(species_id) -> str:
    o = organism_data.get_by_id(species_id) if species_id is not None else None
    return o.label if o else ""


def _hit_view(h: pathbank.Hit, query: str = "") -> Dict[str, Any]:
    return dict(pw_id=h.pw_id, smp_id=h.smp_id or "", name=h.name, organism=_org_label(h.species_id),
                subject=h.subject or "", organisms=h.organisms, url=h.url or "",
                href=f"/pathbank/{h.pw_id}" + (("?" + urlencode({"q": query[:200]})) if query else ""))


def _run_href(kind: str, name: str = "", organism: str = "", pmids: str = "") -> str:
    return "/search/run?" + urlencode({k: v for k, v in dict(kind=kind, name=name, organism=organism,
                                                                 pmids=pmids).items() if v})


def register(app: FastAPI, page: Callable[..., Any], cross_site: Callable[[Request], bool],
             rate_hit: Callable[..., Any], open_store: Callable[..., Any]) -> None:
    s = app.state

    def organisms():
        return organism_data.load_organisms(s.organisms_path)

    def index() -> pathbank.PathBankIndex:
        return pathbank.PathBankIndex(s.pathbank_index)

    def run_form(request: Request, status: int = 200, **kw):
        ctx = dict(view="run", kind="name", query="", pathway="", organism="", pmids="", text="",
                   consent=True, email="", email_error="", errors={}, problems=[], notice="", choices=[],
                   other_choices=[], generate_href="", organism_unknown="", active_nav="submit")
        ctx.update(kw)
        return page(request, "search.html", status=status, **ctx)

    def llm_allowed(request: Request) -> bool:
        if not s.search_llm:
            return False
        key = "search-llm:" + limits.client_key(request.client.host if request.client else "",
                                                ", ".join(request.headers.getlist("x-forwarded-for")) or None,
                                                s.trusted_proxies)
        rl = limits.RateLimiter(s.db_path, **{**s.rate, "limit": SEARCH_LLM_PER_DAY})
        try:
            return rl.hit(key)[0]
        finally:
            rl.close()

    def pathway_answer(request: Request, route: ss.Route):
        orgs, idx = organisms(), index()
        if not idx.available:
            return run_form(request, kind="name", query=route.query, pathway=route.pathway_name,
                            organism=route.organism.label if route.organism else "",
                            notice="The PathBank lookup is not available on this server right now, "
                                   "so T2PW can build the pathway from the literature instead.")
        hits = idx.search(route.pathway_name, species_ids(route.organism, orgs))
        if not pathbank.is_strong(hits) and ss.wants_llm(route) and llm_allowed(request):
            better = ss.llm_extract(route, orgs, s.search_llm if callable(s.search_llm) else None)
            if better is not None and better.kind == ss.PAPER:
                return paper_answer(request, better)
            if better is not None:
                again = idx.search(better.pathway_name, species_ids(better.organism, orgs))
                if again and (not hits or again[0].score > hits[0].score):
                    route, hits = better, again
                elif not hits:
                    route = better
        if pathbank.is_strong(hits):
            return pathway_page(request, hits[0], route.query, others=hits[1:])
        others = []
        if route.organism is not None:  # nothing for this organism: does PathBank draw it for others?
            others = [h for h in idx.search(route.pathway_name) if h.score >= 0.8]
        org = route.organism.label if route.organism else ""
        return run_form(request, view="choices" if hits else "run", kind="name", query=route.query,
                        pathway=route.pathway_name, organism=org,
                        choices=[_hit_view(h, route.query) for h in hits],
                        other_choices=[_hit_view(h, route.query) for h in others],
                        notice="" if hits else "No curated PathBank pathway matched. T2PW can build it from "
                                               "the literature: it searches open-access papers for this name.",
                        generate_href=_run_href("name", route.pathway_name, org),
                        organism_unknown=route.organism_text)

    def paper_answer(request: Request, route: ss.Route):
        pmids = list(route.pmids)
        problems = []
        if route.pmcids or route.dois:
            got = ss.to_pmids(route.pmcids, route.dois, s.search_idconv)
            for i in (*route.pmcids, *route.dois):
                if got.get(i):
                    pmids.append(got[i])
                else:
                    problems.append(f"{i}: no PubMed record was found for this identifier"
                                    + ("." if i in got else " (the lookup did not answer; try again later).")
                                    + " Give its PubMed ID, paste its text, or upload the PDF on the full form.")
        pmids = list(dict.fromkeys(pmids))
        return run_form(request, kind="paper", query=route.query, pmids=", ".join(pmids), problems=problems,
                        pathway=route.pathway_name, organism=route.organism.label if route.organism else "")

    def pathway_page(request: Request, hit: pathbank.Hit, query: str = "", others=()):
        root = s.pathbank_files
        has_svg = pathbank.local_file(root, "svg", hit.pw_id) is not None
        has_pwml = pathbank.local_file(root, "pwml", hit.pw_id) is not None
        view = _hit_view(hit)
        return page(request, "pathbank.html", hit=view, query=query, others=[_hit_view(h, query) for h in others],
                    diagram=f"/pathbank/{hit.pw_id}/diagram" if (has_svg or has_pwml) else "",
                    diagram_is_render=not has_svg, pwml=f"/pathbank/{hit.pw_id}/pwml" if has_pwml else "",
                    generate_href=_run_href("name", query and _query_name(query, organisms()) or hit.name,
                                            view["organism"]),
                    credit=CREDIT, active_nav="submit")

    def answer(request: Request, q: str):
        route = ss.classify(q, organisms())
        if route.kind == ss.EMPTY:
            return RedirectResponse("/", status_code=303)
        if route.kind == ss.PAPER:
            return paper_answer(request, route)
        if route.kind == ss.TEXT:
            return run_form(request, kind="text", query=q[:80], text=route.query)
        if route.kind == ss.URL_ONLY:
            return run_form(request, view="message", query=route.query,
                            notice="T2PW cannot read a web page directly. Give the paper's PubMed ID, PMCID or "
                                   "DOI, paste its text into the box, or upload the PDF on the full form.")
        return pathway_answer(request, route)

    @app.get("/search")
    def search_get(request: Request, q: str = ""):
        return answer(request, q[:GET_QUERY_CAP])

    @app.post("/search")
    async def search_post(request: Request):
        if cross_site(request):
            raise HTTPError(403)
        form = await request.form()
        try:
            q = form.get("q")
            q = q[:limits.MAX_TEXT_CHARS] if isinstance(q, str) else ""
        finally:
            await form.close()
        return await run_in_threadpool(answer, request, q)

    @app.get("/search/run")
    def run_get(request: Request, kind: str = "name", name: str = "", organism: str = "", pmids: str = ""):
        kind = kind if kind in ("name", "paper", "text") else "name"
        return run_form(request, kind=kind, pathway=name[:200], organism=organism[:200], pmids=pmids[:2000])

    @app.post("/search/run")
    async def run_post(request: Request):
        if cross_site(request):
            raise HTTPError(403)
        form = await request.form()
        try:
            fields = {k: (form.get(k) if isinstance(form.get(k), str) else "") for k in
                      ("kind", "name", "organism_1", "pmids", "text", "email", "consent", "query")}
            fields["organism"] = fields.pop("organism_1")
        finally:
            await form.close()
        remote = request.client.host if request.client else ""
        xff = ", ".join(request.headers.getlist("x-forwarded-for")) or None
        out = await run_in_threadpool(queue_one, fields, remote, xff)
        if out[0] == "ok":
            return RedirectResponse(views.job_url("", out[1]), status_code=303)
        _, status, ctx, headers = out
        return run_form(request, status=status, **ctx) if not headers else _with_headers(
            run_form(request, status=status, **ctx), headers)

    def queue_one(f: Dict[str, str], remote: str, xff: Optional[str]):
        kind = f["kind"] if f["kind"] in ("name", "paper", "text") else "name"
        name, organism, email = f["name"].strip(), f["organism"].strip(), f["email"].strip()
        text, consent = f["text"].strip(), bool(f["consent"])
        errors: Dict[str, str] = {}
        # Same rules as /submit (PO ruling Q-BR-1): a pathway name and a listed organism in every mode.
        if len(name) > 200:
            errors["name"] = "Use at most 200 characters."
        elif not name:
            errors["name"] = ("Enter the pathway name." if kind == "name" else
                              "Enter the pathway name this text or these papers describe; it is needed to use the organism.")
        org = organism_data.get_by_label(organism, s.organisms_path) if organism else None
        if not organism:
            errors["organism"] = "Choose an organism."
        elif org is None:
            near = views.search_organisms(organisms(), organism, 6)
            errors["organism"] = ("This is not in the organism list." + (
                " Did you mean: " + "; ".join(o.label for o in near) + "?" if near else ""))
        ids: List[str] = []
        if kind == "paper":
            ids, bad = views.parse_pmids(f["pmids"][:4000])
            if bad:
                errors["pmids"] = "These are not PubMed IDs (digits only): " + ", ".join(bad[:5])
            elif not ids:
                errors["pmids"] = "Enter at least one PubMed ID."
            elif msg := limits.check_pmids(ids):
                errors["pmids"] = msg
        if kind == "text":
            if not text:
                errors["text"] = "Paste the text to read."
            elif msg := limits.check_text([text]):
                errors["text"] = msg
        email_error = ("Enter a valid email address, or leave this blank."
                       if email and (len(email) > 254 or not is_valid_email(email)) else "")
        ctx = dict(kind=kind, pathway=name[:200], organism=organism[:200], pmids=f["pmids"][:4000],
                   text=text, consent=consent, email=email[:254], errors=errors, email_error=email_error,
                   query=f["query"][:200])
        if errors or email_error:
            return ("form", 400, ctx, {})
        ok, wait = rate_hit(app, remote, xff)
        if not ok:
            return ("form", 429, {**ctx, "problems": [limits.rate_limit_message(wait)]},
                    {"Retry-After": str(max(1, math.ceil(wait)))})
        req = PathwayRequest(texts=(text,) if kind == "text" else (), pmids=tuple(ids),
                             scope=RequestedScope(requested_pathway=name,
                                                  requested_organism=org.scope))
        try:
            with open_store(app) as store:
                sid = store.submit([req], consent=consent, email=email or None)
        except QueueFull:
            return ("form", 503, {**ctx, "problems": ["The queue is full right now. Please try again in a few hours."]},
                    {"Retry-After": "3600"})
        return ("ok", sid)

    @app.get("/pathbank/{pw_id}")
    def pathbank_page(request: Request, pw_id: str, q: str = ""):
        hit = index().get(pw_id)
        if hit is None:
            raise HTTPError(404)
        return pathway_page(request, hit, q[:200])

    @app.get("/pathbank/{pw_id}/diagram")
    def pathbank_diagram(pw_id: str):
        if index().get(pw_id) is None:
            raise HTTPError(404)
        svg = pathbank.local_file(s.pathbank_files, "svg", pw_id)
        if svg is not None:
            return FileResponse(svg, media_type="image/svg+xml", headers={"Content-Security-Policy": FILE_CSP})
        pwml = pathbank.local_file(s.pathbank_files, "pwml", pw_id)
        if pwml is None:
            raise HTTPError(404)
        out = Path(s.work_root) / "pathbank_render" / f"{pw_id}.png"
        if not out.is_file():
            from t2pw.render.pathbank import render_pwml

            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(render_pwml(pwml, "png"))
            tmp.replace(out)
        return FileResponse(out, media_type="image/png", headers={"Content-Security-Policy": FILE_CSP})

    @app.get("/pathbank/{pw_id}/pwml")
    def pathbank_pwml(pw_id: str):
        pwml = pathbank.local_file(s.pathbank_files, "pwml", pw_id) if index().get(pw_id) else None
        if pwml is None:
            raise HTTPError(404)
        return FileResponse(pwml, media_type="application/xml", filename=f"{pw_id}.pwml",
                            headers={"Content-Security-Policy": FILE_CSP})


def _query_name(query: str, orgs) -> str:
    """The pathway part of what the visitor typed, for the "generate instead" form."""
    r = ss.classify(query, orgs)
    return r.pathway_name if r.kind == ss.PATHWAY else ""


def _with_headers(resp: Response, headers: Dict[str, str]) -> Response:
    resp.headers.update(headers)
    return resp
