"""Pure helpers for the public pages (W6): no FastAPI, no I/O except path checks.

* ``scrub``           - strip server paths out of any text shown to a user;
* ``search_organisms``- server-side organism search (HTMX datalist);
* ``parse_pmids``     - PubMed ID field -> ids;
* ``item_view``       - one store item -> the dict the templates render;
* ``resolve_download``- a download request -> a file inside the item's own worker dir.

Everything that reaches a template as text is either a fixed sentence written here or
passes through ``scrub``; user data is left raw for Jinja's autoescape.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from t2pw.service.pathway import (
    KIND_AMBIGUOUS,
    KIND_CONTRACT,
    KIND_CRASH,
    KIND_LLM,
    KIND_NETWORK,
    KIND_NO_REACTIONS,
    KIND_TIMEOUT,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SCOPE_CONFLICT,
    STATUS_TIMEOUT,
)
from t2pw.service.driver_bridge import CODE_UNSUPPORTED
from t2pw.web.retry import MAX_RETRIES

MAX_ATTEMPTS = MAX_RETRIES + 1
SID_RE = re.compile(r"[0-9a-f]{32}")  # always .fullmatch: "$" would accept a trailing newline

def job_url(base_url: str, submission_id: str) -> str:
    """THE canonical link to a job page; use it for every job link (pages, emails).

    ``submission_id`` is the store's submission id: ``JobStore.submit`` returns a random
    uuid4 hex string, so it is unguessable and is the job's credential. (Item integer ids
    are never put in a URL.) Pure: no app, no I/O.
    """
    if not SID_RE.fullmatch(str(submission_id)):
        raise ValueError("not a submission id")
    return f"{base_url.rstrip('/')}/job/{submission_id}"


# -- text hygiene --------------------------------------------------------------------

_WIN_PATH = re.compile(r"[A-Za-z]:[\\/][^\s'\"<>|]*")
_UNC_PATH = re.compile(r"\\\\[^\s'\"<>|]+")
_POSIX_PATH = re.compile(r"(?<![\w/.:~-])/(?:[\w.\-~]+/)+[\w.\-~]*")


def scrub(text: Any, limit: int = 600) -> str:
    """Replace anything that looks like a filesystem path; cap the length."""
    s = "" if text is None else str(text)
    for rx in (_WIN_PATH, _UNC_PATH, _POSIX_PATH):
        s = rx.sub("[server path]", s)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def fmt_time(epoch: Optional[float]) -> str:
    if not epoch:
        return ""
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(epoch))


# -- form helpers --------------------------------------------------------------------

_PMID = re.compile(r"^(?:pmid:?)?(\d{1,9})$", re.I)


def parse_pmids(text: str) -> Tuple[List[str], List[str]]:
    """``(ids, bad_tokens)``; accepts commas, semicolons and whitespace; de-duplicates."""
    ids: List[str] = []
    bad: List[str] = []
    for tok in re.split(r"[\s,;]+", text.strip()):
        if not tok:
            continue
        m = _PMID.match(tok)
        if not m:
            bad.append(tok[:30])
        elif m.group(1) not in ids:
            ids.append(m.group(1))
    return ids, bad


def search_organisms(organisms: Sequence[Any], q: str, limit: int = 30) -> List[Any]:
    """Prefix matches first, then word-start matches, then substring matches."""
    q = q.strip().casefold()
    if not q:
        return []
    prefix, word, sub = [], [], []
    for o in organisms:
        low = o.label.casefold()
        if low.startswith(q):
            prefix.append(o)
        elif (" " + q) in low or ("(" + q) in low:
            word.append(o)
        elif q in low:
            sub.append(o)
        if len(prefix) >= limit:
            break
    return (prefix + word + sub)[:limit]


# -- item view-model -----------------------------------------------------------------

_KIND_TEXT = {
    KIND_TIMEOUT: "the run took too long",
    KIND_LLM: "the language-model service did not return a usable answer",
    KIND_NETWORK: "a network service could not be reached",
    KIND_CRASH: "the run stopped unexpectedly",
}


def _kind_text(kind: str) -> str:
    return _KIND_TEXT.get(kind, "an unexpected problem occurred")


def _badge_for(item: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """(status for ``status_badge``, optional label)."""
    state, res = item["state"], item.get("result")
    if state == "queued":
        return ("queued", "Retrying") if item["attempt"] > 0 else ("queued", None)
    if state == "running":
        return "running", None
    res = res or {}
    rs = (res.get("release_status") or {}).get("status")
    if rs == "review_required":
        return "review_required", None
    if rs == "diagnostic_only":
        return "failed", "Diagnostic only"
    if res.get("status") == STATUS_PASS:
        return "done", None
    if res.get("status") in (STATUS_FAIL, STATUS_TIMEOUT, STATUS_ERROR, STATUS_SCOPE_CONFLICT):
        return res["status"], None
    return "failed", None


def _failure_text(res: Dict[str, Any], attempts: int) -> str:
    status, kind = res.get("status"), res.get("failure_kind") or ""
    if status == STATUS_SCOPE_CONFLICT:
        return ("The papers found appear to describe a different pathway or organism than the one "
                "you asked for, so no PWML was produced.")
    if CODE_UNSUPPORTED in (res.get("issue_codes") or []):  # BR: refused before any run
        return ("This pathway was not run because the request cannot be used as given: "
                f"{scrub(res.get('message')).rstrip('.')}.")
    if kind == KIND_NO_REACTIONS:
        return "No reactions could be extracted from the sources, so no PWML was produced."
    if kind in (KIND_CONTRACT, KIND_AMBIGUOUS):
        return ("The extracted pathway did not pass the quality checks, so no PWML was produced. "
                "A different or richer source may help.")
    if kind in _KIND_TEXT or status in (STATUS_TIMEOUT, STATUS_ERROR):
        return (f"This pathway could not be completed after {attempts} "
                f"attempt{'s' if attempts != 1 else ''} because {_kind_text(kind)}. "
                "This is usually temporary: you can submit it again.")
    return "No PWML was produced for this pathway."


def _title(item: Dict[str, Any]) -> str:
    sc = item["request"].scope
    return sc.requested_pathway.strip() or f"Pathway {item['position'] + 1}"


def _inputs_line(item: Dict[str, Any]) -> str:
    r = item["request"]
    bits = []
    if r.pmids:
        bits.append(f"{len(r.pmids)} PubMed ID{'s' if len(r.pmids) != 1 else ''}")
    if r.pdf_paths:
        bits.append(f"{len(r.pdf_paths)} PDF file{'s' if len(r.pdf_paths) != 1 else ''}")
    return ", ".join(bits) or "pathway name"


def item_view(item: Dict[str, Any], history: List[Dict[str, Any]], sid: str) -> Dict[str, Any]:
    """Everything a template needs for one item. Links are server-built from ``sid`` and
    the item position; no stored path ever reaches the result."""
    state, res = item["state"], item.get("result") or {}
    pos = item["position"]
    base = f"/job/{sid}/{pos}"
    badge, badge_label = _badge_for(item)
    rs = (res.get("release_status") or {}).get("status")
    attempts_done = len([h for h in history if h["finished_at"]])
    view: Dict[str, Any] = {
        "pos": pos, "number": pos + 1, "title": _title(item),
        "organism": item["request"].scope.requested_organism,
        "inputs": _inputs_line(item), "href": base,
        "state": state, "finished": state == "finished",
        "badge": badge, "badge_label": badge_label,
        "attempt": item["attempt"], "max_attempts": MAX_ATTEMPTS,
        "review_required": rs == "review_required",
        "message": scrub(res.get("message")) if res else "",
        "warnings": [scrub(w) for w in res.get("warnings", [])] if res else [],
        "downloads": {"pwml": None, "image": None, "reports": []},
        "failed": False, "headline": "", "detail": "",
    }

    if state == "queued" and item["attempt"] == 0:
        view["headline"] = "Waiting in the queue"
        view["detail"] = ("It starts when earlier pathways finish. Once started, a run takes "
                          "about 20 to 60 minutes, and is stopped after 90 minutes.")
    elif state == "queued":
        last = history[-1] if history else {}
        view["headline"] = "Retrying automatically"
        view["detail"] = (f"Attempt {item['attempt']} of {MAX_ATTEMPTS} ended because "
                          f"{_kind_text(last.get('failure_kind') or '')}. "
                          f"Attempt {item['attempt'] + 1} is queued.")
    elif state == "running":
        view["headline"] = f"Running (attempt {item['attempt']} of {MAX_ATTEMPTS})"
        view["detail"] = ("This takes about 20 to 60 minutes (a run is stopped after 90 minutes). You can close this page and come back "
                          "to the same link.")
    elif rs == "review_required":
        view["headline"] = "Needs review"
        view["detail"] = ("A PWML was produced, but it is incomplete or uncertain in places and must "
                          "be checked by a person before use. Read the warnings below.")
    elif rs == "diagnostic_only":
        view["failed"] = True
        view["headline"] = "No usable PWML"
        view["detail"] = ("The run finished but its result is for diagnosis only, not for use. "
                          "The reports below may show why.")
    elif res.get("status") == STATUS_PASS:
        view["headline"] = "Done"
        view["detail"] = "The pathway was built and passed the strict checks."
    else:
        view["failed"] = True
        view["headline"] = "Could not build this pathway"
        view["detail"] = _failure_text(res, max(attempts_done, 1))

    # lifecycle steps
    if state == "queued":
        steps = [("Queued", "current"), ("Running", "pending"), ("Finished", "pending")]
        if item["attempt"]:
            steps[0] = ("Queued for a retry", "current")
    elif state == "running":
        steps = [("Queued", "done"), ("Running", "current"), ("Finished", "pending")]
    else:
        steps = [("Queued", "done"), ("Running", "done"),
                 ("Finished", "failed" if view["failed"] else "done")]
    view["steps"] = [{"label": a, "state": b} for a, b in steps]

    # attempts table
    rows = []
    for h in history:
        if h["finished_at"] is None:
            outcome = "In progress"
        else:
            outcome = {"pass": "Passed", "fail": "Failed", "timeout": "Timed out",
                       "error": "Error", "scope_conflict": "Scope conflict"}.get(h["status"], "Ended")
            if h["will_retry"] or h["retried"]:
                outcome += ", retried"
            if h["status"] != STATUS_PASS and h.get("failure_kind"):
                outcome += f" ({_kind_text(h['failure_kind'])})"
        rows.append([str(h["attempt"]), fmt_time(h["started_at"]), fmt_time(h["finished_at"]) or "-",
                     outcome])
    view["attempt_rows"] = rows

    # downloads (links only; the files are resolved at request time)
    if view["finished"] and res:
        d = view["downloads"]
        if res.get("pwml_path") and rs in ("release_ready", "review_required"):
            d["pwml"] = {"href": f"{base}/file/pwml", "name": Path(str(res["pwml_path"])).name}
        elif res.get("pwml_path"):
            view["warnings"].append(
                "A PWML file exists but its release status is missing or diagnostic-only, "
                "so it is not offered for download.")
        if res.get("image_path"):
            d["image"] = {"href": f"{base}/file/image"}
        d["reports"] = [{"name": k, "href": f"{base}/file/report/{k}"}
                        for k in sorted(res.get("report_paths", {})) if _SAFE_KEY.match(k)]
        if res.get("status") == STATUS_PASS and not d["pwml"]:
            view["warnings"].append("The run finished without a PWML file.")
    return view


_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")

# -- downloads -----------------------------------------------------------------------

_MEDIA = {
    ".pwml": "application/xml", ".xml": "application/xml", ".json": "application/json",
    ".txt": "text/plain", ".csv": "text/csv", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
}
_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml"}


def resolve_download(item: Dict[str, Any], kind: str, name: str, work_root: Path | str
                     ) -> Optional[Tuple[Path, str, str, bool]]:
    """``(path, download_name, media_type, inline)`` or ``None`` (-> 404).

    The file is looked up by *key* in the item's own persisted result, never by a path
    from the URL, and must then resolve to a regular file under
    ``<work_root>/item-<id>/attempt-<n>/out/``: that item's own worker output directory.
    Anything else (another item's directory, ``..``, a symlink out, the attempt's
    request.json or child.log) is refused.
    """
    res = item.get("result") or {}
    rs = (res.get("release_status") or {}).get("status")
    if kind == "pwml":
        raw = res.get("pwml_path") if rs in ("release_ready", "review_required") else None
    elif kind == "image":
        raw = res.get("image_path")
    elif kind == "report" and _SAFE_KEY.match(name or ""):
        raw = (res.get("report_paths") or {}).get(name)
    else:
        raw = None
    if not raw:
        return None
    try:
        path = Path(str(raw)).resolve(strict=True)
        rel = path.relative_to((Path(work_root) / f"item-{item['id']}").resolve())
    except (OSError, ValueError):
        return None
    if len(rel.parts) < 3 or not rel.parts[0].startswith("attempt-") or rel.parts[1] != "out":
        return None
    if not path.is_file():
        return None
    media = _MEDIA.get(path.suffix.lower(), "application/octet-stream")
    if kind == "image" and media not in _IMAGE_TYPES:
        return None
    safe = re.sub(r"[^\w.\-]", "_", path.name, flags=re.ASCII)[:120] or "download"
    return path, safe, media, kind == "image"
