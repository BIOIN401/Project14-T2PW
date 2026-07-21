"""Stage R2 — candidate paper selection / ranking (WP2).

Given the WP1 candidate papers plus the seed pathway context, score each
candidate for relevance to the target pathway + organism, dedupe, and cap the
list so only the genuinely relevant papers reach the (expensive) embedding step
(WP3). Nothing here chunks, embeds, or retrieves — that is WP3/WP4.

Reuse, not reinvention
----------------------
Per the separation invariant (docs/rag/03_separation_invariant.md) and the WP2
brief, this module **reuses the existing preprocessor** rather than building a
second classifier: ``t2pw.pipeline.preprocessor.preprocess`` already returns
``pathway_relevance_score``, ``document_type``, ``likely_organism`` and
``key_compounds`` / ``key_proteins``, and
``is_ambiguous_multi_example_review_context`` already encodes the "multi-example
review with no chosen example" state. We run ``preprocess`` **per candidate** (on
its abstract, or a truncated full text) and score from its output. The name
normalization / safe-access helpers are reused from ``t2pw.mapping.map_ids``.

The dependency arrow points **RAG -> core only**: this module imports from
``t2pw.pipeline`` and ``t2pw.mapping``; nothing in ``t2pw.pipeline`` (or any
stage module) imports ``t2pw.rag``. WP2 changes no pipeline behavior and uses no
core seam — it only filters WP1 output for WP3.

Locality discipline
-------------------
Following ``preprocess_system.txt`` STEP 3, a ``multi_example_review`` candidate
whose examples do not match the seed is **not ingested wholesale**: it is
penalized so it is dropped, or (when one example does match the seed) ranked
below on-topic primary research. Every drop is explained in the
``selection_report``.

Determinism
-----------
Given a fixed ``preprocess`` output the whole module is pure arithmetic and
stable sorting (score desc, then paper id), so results are reproducible. Tests
mock ``preprocess`` and never touch the network / LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from t2pw.config import rag_config
from t2pw.mapping.map_ids import (
    _canonical_name,
    _normalize_name,
    _safe_dict,
    _safe_list,
)
from t2pw.pipeline.preprocessor import (
    is_ambiguous_multi_example_review_context,
    preprocess,
)
from t2pw.rag.acquire import CandidatePaper

# Score component weights (sum to 1.0). Entity overlap dominates because it is
# the strongest signal that a candidate covers the seed pathway; organism match
# and the preprocessor's own pathway-relevance density round it out.
_W_ORGANISM = 0.25
_W_ENTITY = 0.40
_W_RELEVANCE = 0.35

# Penalty applied to a ``multi_example_review`` candidate. If none of its
# examples match the seed the penalty is large enough to drive the score
# negative (-> dropped); if one example matches, the smaller penalty only ranks
# it below equivalent primary research (example-scoped, not ingested wholesale).
_REVIEW_PENALTY_NO_MATCH = 1.0
_REVIEW_PENALTY_MATCH = 0.25

# A candidate is dropped as off-topic when its final score is below this floor.
# Set at 0.0 so only actively-penalized (e.g. non-matching review) or
# zero-signal candidates are dropped; borderline papers are ranked, not dropped.
_MIN_SCORE = 0.0

# Query-shaping cap: how much candidate text to hand the preprocessor. An
# abstract is well under this; a full text is truncated so one huge paper does
# not dominate the (mocked-in-tests, LLM-in-prod) preprocess call.
_MAX_PREPROCESS_CHARS = 6000


# ---------------------------------------------------------------------------
# Result shapes.
# ---------------------------------------------------------------------------
@dataclass
class SelectionScore:
    """The scored breakdown for one candidate (all components in ``[0, 1]``).

    ``score`` is the weighted combination minus ``review_penalty``; it may be
    negative when a non-matching ``multi_example_review`` is penalized.
    ``matched_terms`` and ``notes`` explain the number for the selection report.
    """

    paper_id: str
    score: float
    organism_match: float
    entity_overlap: float
    pathway_relevance: float
    review_penalty: float
    document_type: str
    matched_terms: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "score": round(self.score, 4),
            "organism_match": round(self.organism_match, 4),
            "entity_overlap": round(self.entity_overlap, 4),
            "pathway_relevance": round(self.pathway_relevance, 4),
            "review_penalty": round(self.review_penalty, 4),
            "document_type": self.document_type,
            "matched_terms": list(self.matched_terms),
            "notes": list(self.notes),
        }


@dataclass
class SelectionReportEntry:
    """One line of the selection report — why a candidate was kept or dropped."""

    paper_id: str
    title: str
    source: str
    kept: bool
    score: float
    reason: str
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "source": self.source,
            "kept": self.kept,
            "score": round(self.score, 4),
            "reason": self.reason,
            "rank": self.rank,
        }


@dataclass
class Selection:
    """The selection result: the kept subset, per-paper scores, and the report."""

    selected: List[CandidatePaper] = field(default_factory=list)
    scores: Dict[str, SelectionScore] = field(default_factory=dict)
    selection_report: List[SelectionReportEntry] = field(default_factory=list)

    def kept_entries(self) -> List[SelectionReportEntry]:
        return [e for e in self.selection_report if e.kept]

    def dropped_entries(self) -> List[SelectionReportEntry]:
        return [e for e in self.selection_report if not e.kept]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": [paper.to_dict() for paper in self.selected],
            "scores": {pid: s.to_dict() for pid, s in self.scores.items()},
            "selection_report": [e.to_dict() for e in self.selection_report],
        }


# ---------------------------------------------------------------------------
# Seed / candidate term extraction (reuses map_ids normalization).
# ---------------------------------------------------------------------------
def _norm_terms(*groups: Any) -> List[str]:
    """Flatten, canonicalize, lowercase and de-dup a set of term groups."""
    out: List[str] = []
    seen: set = set()
    for group in groups:
        for item in _safe_list(group):
            term = _normalize_name(_canonical_name(str(item or "")))
            if term and term not in seen:
                seen.add(term)
                out.append(term)
    return out


def _seed_terms(seed_context: Dict[str, Any]) -> Dict[str, List[str]]:
    ctx = _safe_dict(seed_context)
    pathway = _normalize_name(_canonical_name(str(ctx.get("pathway_name") or "")))
    return {
        "compounds": _norm_terms(ctx.get("key_compounds")),
        "proteins": _norm_terms(ctx.get("key_proteins")),
        "gaps": _norm_terms(ctx.get("gap_terms")),
        "pathway": [pathway] if pathway else [],
    }


def _seed_organism(seed_context: Dict[str, Any]) -> str:
    ctx = _safe_dict(seed_context)
    org = str(ctx.get("likely_organism") or ctx.get("organism") or "")
    return _normalize_name(_canonical_name(org))


def _strict_organism_match(seed_org: str, other_org: str) -> bool:
    """Exact or fully-contained organism match (genus-only overlap is not enough)."""
    if not seed_org or not other_org:
        return False
    return seed_org == other_org or seed_org in other_org or other_org in seed_org


def _organism_score(seed_org: str, cand_org: str) -> float:
    """1.0 exact/contained, 0.5 unknown (neutral), 0.0 known mismatch."""
    if not seed_org or not cand_org:
        return 0.5  # unknown on either side -> neither reward nor punish
    if seed_org == cand_org or seed_org in cand_org or cand_org in seed_org:
        return 1.0
    seed_tokens = set(seed_org.split())
    cand_tokens = set(cand_org.split())
    return 1.0 if seed_tokens & cand_tokens else 0.0


def _candidate_haystack(cand_ctx: Dict[str, Any], candidate: CandidatePaper) -> str:
    """A normalized blob of the candidate's entities + title/abstract to match."""
    parts: List[str] = []
    for key in ("pathway_name",):
        parts.append(str(cand_ctx.get(key) or ""))
    for key in ("key_compounds", "key_proteins", "main_subprocesses"):
        parts.extend(str(v or "") for v in _safe_list(cand_ctx.get(key)))
    parts.append(candidate.title)
    parts.append(candidate.abstract)
    return _normalize_name(_canonical_name(" ".join(p for p in parts if p)))


def _example_haystack(example: Dict[str, Any]) -> str:
    ex = _safe_dict(example)
    parts: List[str] = [str(ex.get("name") or ""), str(ex.get("organism") or "")]
    for key in ("key_compounds", "key_proteins", "main_subprocesses"):
        parts.extend(str(v or "") for v in _safe_list(ex.get(key)))
    return _normalize_name(_canonical_name(" ".join(p for p in parts if p)))


def _matched(seed: Dict[str, List[str]], haystack: str) -> List[str]:
    """Which distinct seed terms (compounds/proteins/gaps/pathway) appear."""
    hits: List[str] = []
    seen: set = set()
    for group in ("pathway", "compounds", "proteins", "gaps"):
        for term in seed[group]:
            if term and term in haystack and term not in seen:
                seen.add(term)
                hits.append(term)
    return hits


# ---------------------------------------------------------------------------
# Per-candidate preprocessing (reuse, not a new classifier).
# ---------------------------------------------------------------------------
def _candidate_text(candidate: CandidatePaper) -> str:
    text = (candidate.abstract or "").strip()
    if not text:
        text = (candidate.full_text or "").strip()
    if not text:
        text = (candidate.title or "").strip()
    return text[:_MAX_PREPROCESS_CHARS]


def _candidate_context(candidate: CandidatePaper) -> Dict[str, Any]:
    """Run the reused preprocessor on the candidate; blank text -> empty context.

    Short-circuiting empty text avoids a needless LLM call and keeps a
    text-less candidate deterministic.
    """
    text = _candidate_text(candidate)
    if not text:
        ctx: Dict[str, Any] = {}
    else:
        try:
            ctx = _safe_dict(preprocess(text))
        except Exception:  # noqa: BLE001 - one flaky preprocess (rate limit, timeout)
            # must not sink the whole selection; degrade this candidate to an
            # organism-only context rather than aborting the entire RAG run.
            ctx = {}
    # The candidate's acquisition-time organism is a fallback when the
    # preprocessor could not determine one from the (possibly short) text.
    if not str(ctx.get("likely_organism") or "").strip() and candidate.organism:
        ctx = {**ctx, "likely_organism": candidate.organism}
    return ctx


def _review_assessment(
    cand_ctx: Dict[str, Any],
    seed: Dict[str, List[str]],
    seed_org: str,
) -> Dict[str, Any]:
    """Assess a ``multi_example_review``'s locality vs the seed.

    Returns the penalty to apply and whether any example matches the seed. An
    example matches when its organism matches the seed organism or its local
    entities overlap a seed term (STEP 3 locality discipline).
    """
    doc_type = str(cand_ctx.get("document_type") or "").strip().casefold()
    if doc_type != "multi_example_review":
        return {"penalty": 0.0, "example_match": False, "note": ""}

    examples = _safe_list(cand_ctx.get("candidate_examples"))
    matched_example = ""
    for example in examples:
        hay = _example_haystack(example)
        if not hay:
            continue
        ex_org = _normalize_name(
            _canonical_name(str(_safe_dict(example).get("organism") or ""))
        )
        # A shared genus alone (loose token overlap) must NOT count an example as
        # matching the seed — require a strict organism match (exact / contained)
        # or a concrete seed entity in the example's local content (STEP 3).
        if seed_org and ex_org and _strict_organism_match(seed_org, ex_org):
            matched_example = str(_safe_dict(example).get("name") or "").strip()
            break
        if _matched(seed, hay):
            matched_example = str(_safe_dict(example).get("name") or "").strip()
            break

    if matched_example:
        return {
            "penalty": _REVIEW_PENALTY_MATCH,
            "example_match": True,
            "note": (
                "multi_example_review; example-scoped to seed-matching example "
                f"'{matched_example}' and ranked below primary research"
            ),
        }

    ambiguous = is_ambiguous_multi_example_review_context(cand_ctx)
    detail = "ambiguous (no example selected)" if ambiguous else "no matching example"
    return {
        "penalty": _REVIEW_PENALTY_NO_MATCH,
        "example_match": False,
        "note": (
            f"multi_example_review, {detail}: examples do not match the seed "
            "pathway/organism"
        ),
    }


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def score_candidate(
    candidate: CandidatePaper,
    seed_context: Dict[str, Any],
    *,
    candidate_context: Optional[Dict[str, Any]] = None,
) -> SelectionScore:
    """Score one candidate against the seed context -> :class:`SelectionScore`.

    Combines organism match, compound/protein/gap-term overlap with the seed,
    the preprocessor's ``pathway_relevance_score``, and a penalty for a
    ``multi_example_review`` with no seed-matching example. Pass
    ``candidate_context`` to reuse an already-computed ``preprocess`` result
    (``select`` does this so it preprocesses each candidate exactly once); when
    omitted the reused preprocessor is run here.
    """
    cand_ctx = (
        _safe_dict(candidate_context)
        if candidate_context is not None
        else _candidate_context(candidate)
    )

    seed = _seed_terms(seed_context)
    seed_org = _seed_organism(seed_context)
    total_terms = sum(len(seed[g]) for g in ("pathway", "compounds", "proteins", "gaps"))

    cand_org = _normalize_name(
        _canonical_name(str(cand_ctx.get("likely_organism") or candidate.organism or ""))
    )
    organism_match = _organism_score(seed_org, cand_org)

    haystack = _candidate_haystack(cand_ctx, candidate)
    matched_terms = _matched(seed, haystack)
    entity_overlap = (len(matched_terms) / total_terms) if total_terms else 0.0

    try:
        pathway_relevance = float(cand_ctx.get("pathway_relevance_score") or 0.0)
    except (TypeError, ValueError):
        pathway_relevance = 0.0
    pathway_relevance = max(0.0, min(1.0, pathway_relevance))

    review = _review_assessment(cand_ctx, seed, seed_org)
    penalty = float(review["penalty"])

    base = (
        _W_ORGANISM * organism_match
        + _W_ENTITY * entity_overlap
        + _W_RELEVANCE * pathway_relevance
    )
    score = base - penalty

    notes: List[str] = []
    if review["note"]:
        notes.append(review["note"])
    if seed_org and cand_org and organism_match <= 0.0:
        notes.append(f"organism mismatch (seed '{seed_org}' vs '{cand_org}')")
    if total_terms and not matched_terms:
        notes.append("no seed compound/protein/gap term overlap")

    return SelectionScore(
        paper_id=candidate.id,
        score=score,
        organism_match=organism_match,
        entity_overlap=entity_overlap,
        pathway_relevance=pathway_relevance,
        review_penalty=penalty,
        document_type=str(cand_ctx.get("document_type") or "").strip(),
        matched_terms=matched_terms,
        notes=notes,
    )


def _keep_reason(rank: int, cap: int, score: SelectionScore) -> str:
    detail = f"rank {rank}/{cap}, score {round(score.score, 3)}"
    if score.document_type:
        detail += f", document_type={score.document_type}"
    if score.matched_terms:
        detail += f", matched {len(score.matched_terms)} seed term(s)"
    return f"kept: {detail}"


def _drop_reason_low_score(score: SelectionScore) -> str:
    base = f"dropped: score {round(score.score, 3)} below floor {_MIN_SCORE}"
    if score.notes:
        base += " — " + "; ".join(score.notes)
    return base


def _drop_reason_over_cap(rank: int, cap: int, score: SelectionScore) -> str:
    return (
        f"dropped: rank {rank} exceeds cap {cap} "
        f"(score {round(score.score, 3)}, ranked below the kept papers)"
    )


def select(
    candidates: Sequence[CandidatePaper],
    seed_context: Dict[str, Any],
    *,
    max_papers: Optional[int] = None,
) -> Selection:
    """Rank, dedupe and cap ``candidates`` -> :class:`Selection`.

    Each candidate is preprocessed once (reused classifier), scored via
    :func:`score_candidate`, then the list is sorted (score desc, then paper id
    for a stable order), de-duplicated by PMCID/PMID/DOI/normalized-title (via
    :meth:`CandidatePaper.identity_keys`), and capped at ``max_papers``
    (default ``rag_config()["select_max_papers"]``). The returned
    ``selection_report`` has one entry per candidate and explains **every**
    drop: duplicate, non-matching ``multi_example_review`` (below the score
    floor), or below the cap.
    """
    cap = (
        max_papers
        if isinstance(max_papers, int) and max_papers > 0
        else int(rag_config()["select_max_papers"])
    )

    # 1) Score every candidate (preprocess each exactly once).
    scored: List[tuple] = []
    scores: Dict[str, SelectionScore] = {}
    for idx, candidate in enumerate(candidates):
        cand_ctx = _candidate_context(candidate)
        score = score_candidate(
            candidate, seed_context, candidate_context=cand_ctx
        )
        scores[candidate.id] = score
        # idx is the stable tie-breaker's last resort (preserves input order for
        # candidates with an equal score and equal id).
        scored.append((candidate, score, idx))

    # 2) Deterministic ranking: score desc, then paper id asc, then input order.
    scored.sort(key=lambda t: (-t[1].score, str(t[0].id), t[2]))

    # 3) Walk in ranked order: drop dupes and below-floor, keep up to the cap.
    report: List[SelectionReportEntry] = []
    selected: List[CandidatePaper] = []
    seen_keys: set = set()

    for candidate, score, _idx in scored:
        entry = SelectionReportEntry(
            paper_id=candidate.id,
            title=candidate.title,
            source=candidate.source,
            kept=False,
            score=score.score,
            reason="",
        )

        keys = candidate.identity_keys()
        dup_of = next((k for k in keys if k in seen_keys), None)
        if dup_of is not None:
            entry.reason = f"dropped: duplicate of an already-kept paper ({dup_of})"
            report.append(entry)
            continue

        if score.score < _MIN_SCORE:
            entry.reason = _drop_reason_low_score(score)
            report.append(entry)
            continue

        if len(selected) >= cap:
            entry.reason = _drop_reason_over_cap(len(selected) + 1, cap, score)
            report.append(entry)
            continue

        # Kept.
        seen_keys.update(keys)
        selected.append(candidate)
        entry.kept = True
        entry.rank = len(selected)
        entry.reason = _keep_reason(entry.rank, cap, score)
        report.append(entry)

    return Selection(selected=selected, scores=scores, selection_report=report)


__all__ = [
    "SelectionScore",
    "SelectionReportEntry",
    "Selection",
    "score_candidate",
    "select",
]
