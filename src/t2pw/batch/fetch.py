"""Paper acquisition for the overnight batch runner.

The batch runner needs ~10 real papers with real full text before it can run the
pipeline twice per paper. Getting them is *already solved* by Stage R1
(``t2pw.rag.acquire``), which owns the literature endpoints, the shared
``HttpClient`` (retry/backoff/throttle), and the on-disk cache. This module is a
thin driver over that: it turns a human-written topics file into search contexts,
calls ``search_candidates`` / ``fetch_full_text``, and normalizes the survivors
into :class:`BatchPaper` records with a filesystem-safe ``slug`` for the
per-paper output folder. No new HTTP code, no new dependency, no second fetcher.

Why the failure handling is so lenient
--------------------------------------
Most literature hits are not open access, so an empty full text is the *normal*
case, not an error: such a paper becomes a ``skipped`` record and the run
continues. Likewise :func:`fetch_papers` never raises — an unattended overnight
job must not die at 2am because one API hiccuped. Every per-paper problem is
recorded as a skip so the morning triage can see exactly what was lost and why.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from t2pw.rag.acquire import (
    CandidatePaper,
    build_query,
    fetch_full_text,
    search_candidates,
)

# How many papers a topic line asks for when it does not say.
DEFAULT_PER_TOPIC = 3

# Slug budget. Long paths break on Windows once the per-paper folder gains
# nested artifact filenames, so the folder name itself is kept short.
_MAX_SLUG_LEN = 60

# Characters Windows forbids in a path component, plus control characters.
_ILLEGAL_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_NON_SLUG_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_DASH_RUN = re.compile(r"-{2,}")

# A bare token is treated as a pinned paper id when it looks like a PMCID, a
# PubMed id (6+ digits), or a DOI (all real DOIs start with the "10." registrant
# prefix). Anything else on a one-field line is a topic.
_PMCID_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)
_PMID_RE = re.compile(r"^\d{6,}$")

_PINNED_SOURCE = "pinned"


@dataclass
class TopicSpec:
    """One line of the topics file.

    Either a literature search (``topic`` + optional ``organism`` + ``count``)
    or a pinned paper (``pinned_id`` set, no search performed).
    """

    topic: str = ""
    organism: str = ""
    count: int = DEFAULT_PER_TOPIC
    pinned_id: str = ""

    @property
    def is_pinned(self) -> bool:
        return bool(self.pinned_id)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchPaper:
    """A paper that has real full text and is ready to be run through the app."""

    paper_id: str
    title: str = ""
    source_uri: str = ""
    year: str = ""
    organism: str = ""
    topic: str = ""
    query: str = ""
    full_text: str = ""
    slug: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        """Render the human-readable body of the paper's ``00_PAPER.txt``.

        The full text is deliberately *not* included: it is saved separately so
        this file stays skimmable during morning triage.
        """
        lines: List[str] = [
            "PAPER",
            "=====",
            f"paper_id   : {self.paper_id}",
            f"title      : {self.title or '(unknown)'}",
            f"year       : {self.year or '(unknown)'}",
            f"organism   : {self.organism or '(unspecified)'}",
            f"source_uri : {self.source_uri or '(none)'}",
            "",
            "HOW IT WAS FOUND",
            "----------------",
            f"topic      : {self.topic or '(pinned id -- no search)'}",
            f"query      : {self.query or '(none)'}",
            "",
            "FULL TEXT",
            "---------",
            f"characters : {len(self.full_text)}",
            f"words      : {len(self.full_text.split())}",
            f"slug       : {self.slug}",
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Topics file parsing.
# ---------------------------------------------------------------------------
def _looks_like_paper_id(token: str) -> bool:
    text = token.strip()
    if not text or " " in text:
        return False
    if _PMCID_RE.match(text):
        return True
    if _PMID_RE.match(text):
        return True
    return text.startswith("10.")


def _to_count(value: str) -> int:
    try:
        count = int(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_PER_TOPIC
    return count if count > 0 else DEFAULT_PER_TOPIC


def parse_topics(text: str) -> List[TopicSpec]:
    """Parse a topics file into specs, ignoring blanks and ``#`` comments.

    Accepted line shapes::

        lipid A biosynthesis | Escherichia coli | 3   # topic|organism|count
        menaquinone biosynthesis | Lactococcus lactis  # topic|organism
        PMC4412817                                     # pinned paper id
        caffeine degradation                           # bare topic

    Malformed lines are skipped rather than raised on: the topics file is
    hand-edited input to an unattended job.
    """
    specs: List[TopicSpec] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [part.strip() for part in line.split("|")]
        if len(fields) == 1:
            token = fields[0]
            if _looks_like_paper_id(token):
                specs.append(TopicSpec(pinned_id=token))
            else:
                specs.append(TopicSpec(topic=token))
            continue
        topic = fields[0]
        if not topic:
            continue
        organism = fields[1] if len(fields) > 1 else ""
        count = _to_count(fields[2]) if len(fields) > 2 else DEFAULT_PER_TOPIC
        specs.append(TopicSpec(topic=topic, organism=organism, count=count))
    return specs


# ---------------------------------------------------------------------------
# Slugs (per-paper output folder names).
# ---------------------------------------------------------------------------
def _safe_component(value: str, *, lower: bool = False) -> str:
    text = str(value or "")
    text = _ILLEGAL_PATH_CHARS.sub("-", text)
    text = re.sub(r"\s+", "-", text.strip())
    text = _NON_SLUG_CHARS.sub("-", text)
    text = _DASH_RUN.sub("-", text).strip("-. ")
    return text.lower() if lower else text


def _trim(text: str, size: int) -> str:
    return text[:size].rstrip("-._ ")


def make_slug(paper_id: str, title: str, used: Optional[set] = None) -> str:
    """Build a unique, Windows-safe ``PAPERID__short-title`` folder name.

    Guarantees: no ``< > : " / \\ | ? *``, no control characters, no trailing dot
    or space, whitespace collapsed to ``-``, at most :data:`_MAX_SLUG_LEN`
    characters, and uniqueness within ``used`` (a content hash is appended when
    two papers would otherwise collide).
    """
    ident = _safe_component(paper_id) or "paper"
    short_title = _safe_component(title, lower=True)
    base = f"{ident}__{short_title}" if short_title else ident
    base = _trim(base, _MAX_SLUG_LEN) or "paper"
    if used is None:
        return base
    if base not in used:
        used.add(base)
        return base
    digest = hashlib.sha1(f"{paper_id}|{title}".encode("utf-8")).hexdigest()[:6]
    stem = _trim(base, max(1, _MAX_SLUG_LEN - len(digest) - 1))
    candidate = f"{stem}-{digest}"
    suffix = 2
    while candidate in used:
        candidate = f"{stem}-{digest}{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Fetching.
# ---------------------------------------------------------------------------
def _skip(
    *,
    paper_id: str,
    reason: str,
    spec: TopicSpec,
    title: str = "",
    detail: str = "",
) -> Dict[str, Any]:
    return {
        "paper_id": paper_id,
        "title": title,
        "topic": spec.topic,
        "organism": spec.organism,
        "pinned": spec.is_pinned,
        "reason": reason,
        "detail": detail,
    }


def _context_for(spec: TopicSpec) -> Dict[str, Any]:
    """Build the minimal seed context ``search_candidates`` needs.

    Only ``pathway_name`` and ``likely_organism`` are set: the batch has no
    upstream Stage-0 extraction to draw compounds/proteins/gap terms from, and an
    organism-scoped topic query is exactly what we want here.
    """
    return {"pathway_name": spec.topic, "likely_organism": spec.organism}


def _as_batch_paper(
    candidate: CandidatePaper,
    *,
    spec: TopicSpec,
    query: str,
    full_text: str,
    used_slugs: set,
) -> BatchPaper:
    paper_id = str(candidate.id or "").strip()
    title = str(candidate.title or "").strip()
    return BatchPaper(
        paper_id=paper_id,
        title=title,
        source_uri=str(candidate.source_uri or "").strip(),
        year=str(candidate.year or "").strip(),
        organism=str(candidate.organism or "").strip() or spec.organism,
        topic=spec.topic,
        query=query,
        full_text=full_text,
        slug=make_slug(paper_id, title, used_slugs),
    )


def fetch_papers(
    specs: Union[str, Iterable[TopicSpec]],
    *,
    limit: Optional[int] = None,
    sources: Optional[Sequence[str]] = None,
    client: Optional[Any] = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    refresh: bool = False,
    search_fn: Optional[Callable[..., List[CandidatePaper]]] = None,
    fetch_text_fn: Optional[Callable[..., str]] = None,
) -> Tuple[List[BatchPaper], List[Dict[str, Any]]]:
    """Fetch the papers the batch will run.

    ``specs`` is the output of :func:`parse_topics` (a raw topics-file string is
    also accepted and parsed for convenience). ``limit`` caps the **total**
    number of papers returned across all topics; each spec additionally
    contributes at most ``spec.count``.

    Returns ``(papers, skipped)``. Every paper in ``papers`` has non-empty
    ``full_text`` and a unique ``slug``. Every rejected candidate appears in
    ``skipped`` with a ``reason``: ``no_full_text`` (not open access — the common
    case), ``duplicate``, ``no_candidates``, ``search_failed``, or
    ``fetch_failed``.

    Never raises. ``search_fn`` / ``fetch_text_fn`` exist so tests can inject
    fakes; they default to the Stage R1 helpers.
    """
    if isinstance(specs, str):
        specs = parse_topics(specs)
    spec_list = [s for s in (specs or []) if isinstance(s, TopicSpec)]

    search = search_fn or search_candidates
    fetch_text = fetch_text_fn or fetch_full_text
    total_cap = limit if isinstance(limit, int) and limit > 0 else None

    papers: List[BatchPaper] = []
    skipped: List[Dict[str, Any]] = []
    used_slugs: set = set()
    seen_ids: set = set()

    def _room_left() -> bool:
        return total_cap is None or len(papers) < total_cap

    def _text_for(candidate: CandidatePaper) -> str:
        return str(
            fetch_text(
                candidate,
                client=client,
                cache_dir=cache_dir,
                use_cache=use_cache,
                refresh=refresh,
            )
            or ""
        )

    for spec in spec_list:
        if not _room_left():
            break

        # ---- pinned id: no search, straight to full text -------------------
        if spec.is_pinned:
            candidate = CandidatePaper(id=spec.pinned_id, source=_PINNED_SOURCE)
            ident = spec.pinned_id.strip()
            if ident.lower() in seen_ids:
                skipped.append(_skip(paper_id=ident, reason="duplicate", spec=spec))
                continue
            try:
                text = _text_for(candidate)
            except Exception as exc:  # noqa: BLE001 - one bad id must not stop the run
                skipped.append(
                    _skip(
                        paper_id=ident,
                        reason="fetch_failed",
                        spec=spec,
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if not text.strip():
                skipped.append(_skip(paper_id=ident, reason="no_full_text", spec=spec))
                continue
            seen_ids.add(ident.lower())
            papers.append(
                _as_batch_paper(
                    candidate,
                    spec=spec,
                    query="",
                    full_text=text,
                    used_slugs=used_slugs,
                )
            )
            continue

        # ---- topic search --------------------------------------------------
        context = _context_for(spec)
        want = max(1, spec.count)
        if total_cap is not None:
            want = min(want, total_cap - len(papers))
        # Over-fetch: most hits are not open access, so asking for exactly
        # ``want`` candidates would almost always come back short after the
        # full-text filter.
        over_fetch = max(want * 3, want + 2)
        status: Dict[str, Any] = {}
        kwargs: Dict[str, Any] = {
            "max_papers": over_fetch,
            "client": client,
            "cache_dir": cache_dir,
            "use_cache": use_cache,
            "refresh": refresh,
            "status": status,
        }
        if sources is not None:
            kwargs["sources"] = sources
        try:
            candidates = list(search(context, **kwargs) or [])
        except Exception as exc:  # noqa: BLE001 - search_candidates shouldn't raise
            skipped.append(
                _skip(
                    paper_id="",
                    reason="search_failed",
                    spec=spec,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        query = str(status.get("query") or "") or build_query(context)
        if not candidates:
            skipped.append(_skip(paper_id="", reason="no_candidates", spec=spec))
            continue

        taken = 0
        for candidate in candidates:
            if taken >= want or not _room_left():
                break
            if not isinstance(candidate, CandidatePaper):
                continue
            ident = str(candidate.id or "").strip()
            title = str(candidate.title or "").strip()
            if not ident:
                continue
            if ident.lower() in seen_ids:
                skipped.append(
                    _skip(paper_id=ident, reason="duplicate", spec=spec, title=title)
                )
                continue
            try:
                text = str(candidate.full_text or "") or _text_for(candidate)
            except Exception as exc:  # noqa: BLE001
                skipped.append(
                    _skip(
                        paper_id=ident,
                        reason="fetch_failed",
                        spec=spec,
                        title=title,
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if not text.strip():
                # Not an error: the paper simply is not open access.
                skipped.append(
                    _skip(
                        paper_id=ident, reason="no_full_text", spec=spec, title=title
                    )
                )
                continue
            seen_ids.add(ident.lower())
            papers.append(
                _as_batch_paper(
                    candidate,
                    spec=spec,
                    query=query,
                    full_text=text,
                    used_slugs=used_slugs,
                )
            )
            taken += 1

    return papers, skipped


__all__ = [
    "BatchPaper",
    "TopicSpec",
    "DEFAULT_PER_TOPIC",
    "parse_topics",
    "make_slug",
    "fetch_papers",
]
