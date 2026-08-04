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

Why a paper is screened before its full text is downloaded
----------------------------------------------------------
A topic query is a blunt instrument. Asking EuropePMC for "lipid A biosynthesis
AND Escherichia coli" returned a Fournier's-gangrene case report and a river
resistome survey; "enterobactin biosynthesis" returned two poultry ESBL
virulence surveys. Each of those cost a full-text download plus two app runs
(strict + research) and then appeared in the morning triage as an *extraction
failure* — sending debugging into the extractor for papers that were never about
the requested pathway.

:func:`fetch_papers` now runs the deterministic title/abstract gate
(``t2pw.rag.eligibility``) **before** ``fetch_full_text``, so an off-topic hit
costs one keyword scan. A screened-out paper is a ``skipped`` record with an
``ineligible_*`` reason and its own eligibility report, never a manifest row, so
it can never be counted as a pipeline failure. Pinned ids bypass the score (a
human named that paper) but still record any mismatch as a warning.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from t2pw.rag.acquire import (
    CandidatePaper,
    build_query,
    fetch_full_text,
    search_candidates,
)
from t2pw.rag.eligibility import (
    ORGANISM_UNKNOWN,
    OUTCOME_DUPLICATE,
    OUTCOME_NO_FULL_TEXT,
    EligibilityDecision,
    EligibilityThresholds,
    RequestedScope,
    screen_candidate,
    thresholds_from_config,
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

    def requested_scope(self) -> RequestedScope:
        """The frozen requested scope this line asks for.

        Frozen on purpose: this is the batch's *request*, and nothing downstream
        (Stage 0 included) may rewrite it to whatever a paper turns out to be
        about. See ``t2pw.rag.eligibility.apply_stage0_observation``.
        """
        return RequestedScope(
            requested_pathway=self.topic,
            requested_organism=self.organism,
            pinned=self.is_pinned,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchPaper:
    """A paper that has real full text and is ready to be run through the app.

    ``requested_*`` is what the topics file asked for; ``observed_*`` is what the
    paper's own title/abstract reports, and ``organism_match`` compares them. A
    paper that names no organism keeps an empty ``observed_organisms`` and
    ``organism_match == "unknown"`` — the requested organism is never copied in
    to fill the gap, because "we asked for E. coli" and "this paper is about
    E. coli" are different facts.

    ``topic`` mirrors ``requested_pathway``; it is kept under its old name
    because the manifest and ``batch.report`` read a ``topic`` column.
    """

    paper_id: str
    title: str = ""
    source_uri: str = ""
    year: str = ""
    requested_pathway: str = ""
    requested_organism: str = ""
    observed_pathways: List[str] = field(default_factory=list)
    observed_organisms: List[str] = field(default_factory=list)
    organism_match: str = ORGANISM_UNKNOWN
    topic: str = ""
    query: str = ""
    full_text: str = ""
    slug: str = ""
    eligibility: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def scope_disagreements(self) -> List[str]:
        """Ways this record's observed fields contradict its eligibility report.

        Always empty in practice -- :func:`_as_batch_paper` reads all three from the
        one decision -- which is exactly why it is worth asserting: a future
        refactor that reintroduces a second source of observed organisms is caught
        here and in ``eligibility_summary`` rather than in a plan a human trusts.
        """
        report = self.eligibility or {}
        if not report:
            return []
        problems: List[str] = []
        if list(report.get("observed_organisms") or []) != list(self.observed_organisms):
            problems.append(
                "observed_organisms "
                f"{self.observed_organisms!r} != eligibility "
                f"{list(report.get('observed_organisms') or [])!r}"
            )
        if list(report.get("observed_pathways") or []) != list(self.observed_pathways):
            problems.append(
                "observed_pathways "
                f"{self.observed_pathways!r} != eligibility "
                f"{list(report.get('observed_pathways') or [])!r}"
            )
        if str(report.get("organism_match") or "") != self.organism_match:
            problems.append(
                f"organism_match {self.organism_match!r} != eligibility "
                f"{report.get('organism_match')!r}"
            )
        if str(report.get("requested_organism") or "") != self.requested_organism:
            problems.append(
                f"requested_organism {self.requested_organism!r} != eligibility "
                f"{report.get('requested_organism')!r}"
            )
        return problems

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
            f"source_uri : {self.source_uri or '(none)'}",
            "",
            "REQUESTED vs OBSERVED",
            "---------------------",
            f"requested pathway  : {self.requested_pathway or '(none)'}",
            f"requested organism : {self.requested_organism or '(none)'}",
            f"observed pathways  : {', '.join(self.observed_pathways) or '(none reported)'}",
            f"observed organisms : {', '.join(self.observed_organisms) or '(none reported)'}",
            f"organism match     : {self.organism_match}",
            "",
            "ELIGIBILITY",
            "-----------",
            f"outcome    : {self.eligibility.get('outcome', '(not screened)')}",
            f"score      : {self.eligibility.get('score', '(n/a)')}"
            f" (threshold {self.eligibility.get('threshold', 'n/a')})",
            f"reason     : {self.eligibility.get('reason', '(none)')}",
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
        PMC4412817 | lipid A biosynthesis | Escherichia coli  # PINNED WITH SCOPE
        PMC4412817                                     # pinned paper id, no scope
        caffeine degradation                           # bare topic

    **Pinned-with-scope** is the shape an acceptance run must use. A bare pinned
    id produces ``TopicSpec(pinned_id=..., topic="", organism="")``, and
    :func:`_as_batch_paper` copies those two empty strings straight into
    ``BatchPaper.requested_pathway`` / ``requested_organism``. Every downstream
    reader of the request then sees nothing: ``driver._extraction_focus`` returns
    ``""`` so extraction is told no scope at all, ``driver._reconcile_stage0_scope``
    has no request to contradict, and the screening record is written against an
    empty pathway (which is why ``runs/2026-08-01_1724`` scored every pinned paper
    ``off_topic`` with ``no_mechanistic_pathway_terms``). The paper is still
    *fetched* correctly -- the defect is invisible in the paper list and fatal in
    the plan.

    Disambiguation is by the FIRST field only: if it looks like a paper id the
    line is pinned and the remaining fields are ``pathway | organism``; otherwise
    the line is a search and they are ``organism | count``. A pathway name can
    never collide with :func:`_looks_like_paper_id`, which requires ``PMC<digits>``,
    6+ bare digits, or a leading ``10.`` DOI prefix.

    Malformed lines are skipped rather than raised on: the topics file is
    hand-edited input to an unattended job.
    """
    specs: List[TopicSpec] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [part.strip() for part in line.split("|")]
        if _looks_like_paper_id(fields[0]):
            specs.append(
                TopicSpec(
                    pinned_id=fields[0],
                    topic=fields[1] if len(fields) > 1 else "",
                    organism=fields[2] if len(fields) > 2 else "",
                    count=1,
                )
            )
            continue
        if len(fields) == 1:
            specs.append(TopicSpec(topic=fields[0]))
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
    decision: Optional[EligibilityDecision] = None,
) -> Dict[str, Any]:
    """One ``skipped.json`` record.

    ``reason`` doubles as the eligibility outcome for the screened-out cases
    (``ineligible_pathway``, ``ineligible_organism``, ``insufficient_metadata``,
    ``duplicate``, ``no_full_text``), so the vocabulary in
    ``t2pw.rag.eligibility`` is the single source of truth for both. ``requested_*``
    keeps the request visible on the skip record; ``eligibility`` carries the
    compact per-paper report when the gate produced one.
    """
    record: Dict[str, Any] = {
        "paper_id": paper_id,
        "title": title,
        "topic": spec.topic,
        "requested_pathway": spec.topic,
        "requested_organism": spec.organism,
        "pinned": spec.is_pinned,
        "reason": reason,
        "detail": detail,
    }
    if decision is not None:
        record["eligibility"] = decision.to_dict()
        record["observed_pathways"] = list(decision.observed_pathways)
        record["observed_organisms"] = list(decision.observed_organisms)
        record["organism_match"] = decision.organism_match
    return record


@dataclass
class _TopicTally:
    """Per-topic acquisition accounting.

    Exists because a selective gate makes "how many did we plan?" an insufficient
    answer. When a topic under-delivers, the useful question is *where* the
    candidates went -- examined but ineligible, eligible but paywalled, or the
    search simply ran dry -- and whether acquisition gave up because it hit the
    candidate ceiling or because there was nothing left to look at.
    """

    topic: str = ""
    organism: str = ""
    query: str = ""
    requested: int = 0
    returned: int = 0
    examined: int = 0
    eligible: int = 0
    ineligible: int = 0
    duplicate: int = 0
    no_full_text: int = 0
    fetch_failed: int = 0
    accepted: int = 0
    search_calls: int = 0
    max_requested_candidates: int = 0
    stop_reason: str = ""
    by_outcome: Dict[str, int] = field(default_factory=dict)

    @property
    def filled(self) -> bool:
        return self.accepted >= self.requested

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "requested_organism": self.organism,
            "query": self.query,
            "requested": self.requested,
            "returned": self.returned,
            "examined": self.examined,
            "eligible": self.eligible,
            "ineligible": self.ineligible,
            "duplicate": self.duplicate,
            "no_full_text": self.no_full_text,
            "fetch_failed": self.fetch_failed,
            "accepted": self.accepted,
            "filled": self.filled,
            "search_calls": self.search_calls,
            "max_requested_candidates": self.max_requested_candidates,
            "stop_reason": self.stop_reason,
            "ineligible_by_outcome": dict(sorted(self.by_outcome.items())),
        }


def _fetch_stats(
    topics: Sequence[Dict[str, Any]],
    papers: Sequence["BatchPaper"],
    skipped: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Roll the per-topic tallies into the run-level acquisition funnel."""

    def _sum(key: str) -> int:
        return sum(int(t.get(key) or 0) for t in topics)

    return {
        "requested": _sum("requested"),
        "examined": _sum("examined"),
        "eligible": _sum("eligible"),
        "ineligible": _sum("ineligible"),
        "duplicate": _sum("duplicate"),
        "no_full_text": _sum("no_full_text"),
        "fetch_failed": _sum("fetch_failed"),
        "accepted": len(papers),
        "skipped_records": len(skipped),
        "topics": [dict(t) for t in topics],
        "topics_filled": sum(1 for t in topics if t.get("filled")),
        "topics_short": [
            {
                "topic": t.get("topic"),
                "requested": t.get("requested"),
                "accepted": t.get("accepted"),
                "stop_reason": t.get("stop_reason"),
            }
            for t in topics
            if not t.get("filled")
        ],
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
    decision: EligibilityDecision,
) -> BatchPaper:
    """Normalize an accepted candidate into a :class:`BatchPaper`.

    **The decision is the single source of the observed fields.** Two earlier
    versions of this function got that wrong: the first wrote
    ``organism=candidate.organism or spec.organism`` (turning "we asked for
    E. coli" into "this paper is about E. coli"), and the second promoted
    ``candidate.organism`` into ``observed_organisms`` unconditionally — which
    re-imported the same false claim from a legacy cache row, whose ``organism``
    *was* the stamped request. Reading only from ``decision`` also guarantees
    ``BatchPaper.observed_organisms``, ``organism_match`` and
    ``eligibility["observed_organisms"]`` cannot disagree; see
    :meth:`BatchPaper.scope_disagreements`.
    """
    paper_id = str(candidate.id or "").strip()
    title = str(candidate.title or "").strip()
    return BatchPaper(
        paper_id=paper_id,
        title=title,
        source_uri=str(candidate.source_uri or "").strip(),
        year=str(candidate.year or "").strip(),
        requested_pathway=spec.topic,
        requested_organism=spec.organism,
        observed_pathways=list(decision.observed_pathways),
        observed_organisms=list(decision.observed_organisms),
        organism_match=decision.organism_match or ORGANISM_UNKNOWN,
        topic=spec.topic,
        query=query,
        full_text=full_text,
        slug=make_slug(paper_id, title, used_slugs),
        eligibility=decision.to_dict(),
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
    screen: Optional[bool] = None,
    thresholds: Optional[EligibilityThresholds] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[List[BatchPaper], List[Dict[str, Any]]]:
    """Fetch the papers the batch will run.

    ``specs`` is the output of :func:`parse_topics` (a raw topics-file string is
    also accepted and parsed for convenience). ``limit`` caps the **total**
    number of papers returned across all topics; each spec additionally
    contributes at most ``spec.count``.

    Returns ``(papers, skipped)``. Every paper in ``papers`` has non-empty
    ``full_text`` and a unique ``slug``. Every rejected candidate appears in
    ``skipped`` with a ``reason``: ``ineligible_pathway`` /
    ``ineligible_organism`` / ``insufficient_metadata`` (screened out before any
    full text was fetched), ``no_full_text`` (not open access — the common case),
    ``duplicate``, ``no_candidates``, ``search_failed``, or ``fetch_failed``.
    None of these are pipeline failures: nothing was run, so nothing failed.

    Screening runs **before** the full-text fetch, so an off-topic hit costs one
    keyword scan instead of a download plus two app runs. ``screen`` defaults to
    ``rag_config()["eligibility_enabled"]``; pass ``screen=False`` to restore the
    old take-everything behavior. ``thresholds`` pins the score bar (default: the
    environment, via :func:`thresholds_from_config`).

    Because the gate is selective, acquisition does **not** settle for one fixed
    over-fetch: per topic it keeps escalating the search until ``spec.count``
    eligible papers with full text are accepted, the source yields nothing new, or
    ``thresholds.candidate_ceiling`` candidates have been examined. Pass a dict as
    ``stats`` to receive the funnel — requested / examined / eligible /
    no_full_text / accepted, per topic and overall, with the stop reason for every
    topic that came up short.

    Never raises. ``search_fn`` / ``fetch_text_fn`` exist so tests can inject
    fakes; they default to the Stage R1 helpers.
    """
    if isinstance(specs, str):
        specs = parse_topics(specs)
    spec_list = [s for s in (specs or []) if isinstance(s, TopicSpec)]

    search = search_fn or search_candidates
    fetch_text = fetch_text_fn or fetch_full_text
    total_cap = limit if isinstance(limit, int) and limit > 0 else None

    limits = thresholds if thresholds is not None else thresholds_from_config()
    if screen is False or (screen is None and not limits.enabled):
        # Explicitly off: keep a thresholds object around so every candidate
        # still gets a recorded (permissive) decision rather than a blank one.
        limits = replace(limits, enabled=False)

    papers: List[BatchPaper] = []
    skipped: List[Dict[str, Any]] = []
    used_slugs: set = set()
    seen_ids: set = set()
    topic_stats: List[Dict[str, Any]] = []

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

        scope = spec.requested_scope()

        # ---- pinned id: no search, straight to full text -------------------
        if spec.is_pinned:
            candidate = CandidatePaper(id=spec.pinned_id, source=_PINNED_SOURCE)
            ident = spec.pinned_id.strip()
            # A pinned id is its own one-candidate "topic" in the funnel, so the
            # run-level accepted/examined counts add up across both branches.
            pinned_tally = _TopicTally(
                topic=f"(pinned {ident})", organism=spec.organism, requested=1
            )
            pinned_tally.returned = 1
            pinned_tally.examined = 1
            pinned_tally.eligible = 1
            topic_stats.append(pinned_tally.to_dict())
            if ident.lower() in seen_ids:
                pinned_tally.duplicate = 1
                pinned_tally.eligible = 0
                pinned_tally.stop_reason = OUTCOME_DUPLICATE
                topic_stats[-1] = pinned_tally.to_dict()
                skipped.append(
                    _skip(paper_id=ident, reason=OUTCOME_DUPLICATE, spec=spec)
                )
                continue
            # A pinned paper is screened for the RECORD only: a human named this
            # id, so the score cannot veto it (outcome == pinned_override), but any
            # pathway/organism mismatch is still computed and carried in
            # ``warnings`` so the morning triage sees it.
            decision = screen_candidate(candidate, scope, thresholds=limits)
            try:
                text = _text_for(candidate)
            except Exception as exc:  # noqa: BLE001 - one bad id must not stop the run
                pinned_tally.fetch_failed = 1
                pinned_tally.stop_reason = "fetch_failed"
                topic_stats[-1] = pinned_tally.to_dict()
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
                pinned_tally.no_full_text = 1
                pinned_tally.stop_reason = OUTCOME_NO_FULL_TEXT
                topic_stats[-1] = pinned_tally.to_dict()
                skipped.append(
                    _skip(paper_id=ident, reason=OUTCOME_NO_FULL_TEXT, spec=spec)
                )
                continue
            seen_ids.add(ident.lower())
            papers.append(
                _as_batch_paper(
                    candidate,
                    spec=spec,
                    query="",
                    full_text=text,
                    used_slugs=used_slugs,
                    decision=decision,
                )
            )
            pinned_tally.accepted = 1
            pinned_tally.stop_reason = "filled"
            topic_stats[-1] = pinned_tally.to_dict()
            continue

        # ---- topic search --------------------------------------------------
        context = _context_for(spec)
        want = max(1, spec.count)
        if total_cap is not None:
            want = min(want, total_cap - len(papers))

        tally = _TopicTally(topic=spec.topic, organism=spec.organism, requested=want)
        query = ""
        # The fixed ``want * 3`` over-fetch was calibrated for a fetcher that took
        # every hit with full text. A selective gate rejects most of what a topic
        # query returns -- 22 of 28 in the 2026-07-28_2122 titles -- so one 3x pull
        # would silently under-deliver every topic. Escalate the request instead:
        # ask for more, examine only the candidates not yet seen, and stop when the
        # requested count is filled, the search stops yielding anything new, or the
        # configured candidate ceiling is reached. Every stop reason is recorded.
        ceiling = max(want, int(limits.candidate_ceiling))
        request_size = min(ceiling, max(want * 3, want + 2))
        examined_ids: set = set()
        search_failed = False

        while True:
            status: Dict[str, Any] = {}
            kwargs: Dict[str, Any] = {
                "max_papers": request_size,
                "client": client,
                "cache_dir": cache_dir,
                "use_cache": use_cache,
                "refresh": refresh,
                "status": status,
            }
            if sources is not None:
                kwargs["sources"] = sources
            tally.search_calls += 1
            tally.max_requested_candidates = max(
                tally.max_requested_candidates, request_size
            )
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
                tally.stop_reason = "search_failed"
                search_failed = True
                break

            query = query or str(status.get("query") or "") or build_query(context)
            tally.returned += len(candidates)

            fresh = [
                candidate
                for candidate in candidates
                if isinstance(candidate, CandidatePaper)
                and str(candidate.id or "").strip()
                and str(candidate.id or "").strip().lower() not in examined_ids
            ]
            if not fresh:
                # Either the very first search matched nothing, or escalating the
                # page size stopped yielding new records: the source is exhausted.
                if tally.examined == 0:
                    skipped.append(
                        _skip(paper_id="", reason="no_candidates", spec=spec)
                    )
                    tally.stop_reason = "no_candidates"
                else:
                    tally.stop_reason = "source_exhausted"
                break

            for candidate in fresh:
                if tally.accepted >= want or not _room_left():
                    break
                if tally.examined >= ceiling:
                    break
                ident = str(candidate.id or "").strip()
                title = str(candidate.title or "").strip()
                examined_ids.add(ident.lower())
                tally.examined += 1

                if ident.lower() in seen_ids:
                    tally.duplicate += 1
                    skipped.append(
                        _skip(
                            paper_id=ident,
                            reason=OUTCOME_DUPLICATE,
                            spec=spec,
                            title=title,
                        )
                    )
                    continue

                # ---- the gate, BEFORE the expensive full-text fetch ----------
                # Also the point where requested vs observed metadata is written
                # onto the candidate. A rejected paper costs one keyword scan.
                decision = screen_candidate(candidate, scope, thresholds=limits)
                if decision.ineligible:
                    tally.ineligible += 1
                    tally.by_outcome[decision.outcome] = (
                        tally.by_outcome.get(decision.outcome, 0) + 1
                    )
                    skipped.append(
                        _skip(
                            paper_id=ident,
                            reason=decision.outcome,
                            spec=spec,
                            title=title,
                            detail=decision.reason,
                            decision=decision,
                        )
                    )
                    continue
                tally.eligible += 1

                try:
                    text = str(candidate.full_text or "") or _text_for(candidate)
                except Exception as exc:  # noqa: BLE001
                    tally.fetch_failed += 1
                    skipped.append(
                        _skip(
                            paper_id=ident,
                            reason="fetch_failed",
                            spec=spec,
                            title=title,
                            detail=f"{type(exc).__name__}: {exc}",
                            decision=decision,
                        )
                    )
                    continue
                if not text.strip():
                    # Not an error: the paper simply is not open access.
                    tally.no_full_text += 1
                    skipped.append(
                        _skip(
                            paper_id=ident,
                            reason=OUTCOME_NO_FULL_TEXT,
                            spec=spec,
                            title=title,
                            decision=decision,
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
                        decision=decision,
                    )
                )
                tally.accepted += 1

            if tally.accepted >= want:
                tally.stop_reason = "filled"
                break
            if not _room_left():
                tally.stop_reason = "run_limit"
                break
            if tally.examined >= ceiling:
                tally.stop_reason = "candidate_ceiling"
                break
            if request_size >= ceiling:
                tally.stop_reason = "candidate_ceiling"
                break
            request_size = min(ceiling, request_size * 2)

        tally.query = query
        if not search_failed and not tally.stop_reason:
            tally.stop_reason = "filled" if tally.accepted >= want else "exhausted"
        topic_stats.append(tally.to_dict())

    if isinstance(stats, dict):
        stats.update(_fetch_stats(topic_stats, papers, skipped))
    return papers, skipped


def eligibility_summary(
    papers: Sequence[BatchPaper],
    skipped: Sequence[Dict[str, Any]],
    *,
    thresholds: Optional[EligibilityThresholds] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A compact, plan-artifact-sized account of what the gate decided.

    Written into ``plan.json`` next to the papers so a run carries the thresholds
    it was screened with, the acquisition funnel, the per-outcome tally, and the
    ids a human should look at -- without having to re-derive any of it.
    """
    limits = thresholds if thresholds is not None else thresholds_from_config()
    counts: Dict[str, int] = {}
    classes: Dict[str, int] = {}
    organism_match: Dict[str, int] = {}
    for paper in papers:
        report = paper.eligibility or {}
        outcome = str(report.get("outcome") or "unscreened")
        counts[outcome] = counts.get(outcome, 0) + 1
        cls = str(report.get("classification") or "")
        if cls:
            classes[cls] = classes.get(cls, 0) + 1
        organism_match[paper.organism_match] = (
            organism_match.get(paper.organism_match, 0) + 1
        )
    for record in skipped:
        reason = str(record.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
        report = record.get("eligibility")
        if isinstance(report, dict):
            cls = str(report.get("classification") or "")
            if cls:
                classes[cls] = classes.get(cls, 0) + 1
    review: List[str] = []
    for paper in papers:
        if (paper.eligibility or {}).get("needs_manual_review"):
            review.append(paper.paper_id)
    for record in skipped:
        report = record.get("eligibility")
        if isinstance(report, dict) and report.get("needs_manual_review"):
            review.append(str(record.get("paper_id") or ""))

    # Invariant surfaced, not assumed: every accepted paper's observed fields must
    # agree with the eligibility report they came from.
    disagreements: List[Dict[str, Any]] = []
    for paper in papers:
        problems = paper.scope_disagreements()
        if problems:
            disagreements.append({"paper_id": paper.paper_id, "problems": problems})

    summary: Dict[str, Any] = {
        "thresholds": limits.to_dict(),
        "outcomes": dict(sorted(counts.items())),
        "classifications": dict(sorted(classes.items())),
        "organism_match": dict(sorted(organism_match.items())),
        "accepted": len(papers),
        "needs_manual_review": [pid for pid in review if pid],
        "scope_disagreements": disagreements,
    }
    if isinstance(stats, dict) and stats:
        summary["acquisition"] = dict(stats)
    return summary


__all__ = [
    "BatchPaper",
    "TopicSpec",
    "DEFAULT_PER_TOPIC",
    "parse_topics",
    "make_slug",
    "fetch_papers",
    "eligibility_summary",
]
