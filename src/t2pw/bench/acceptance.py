"""Score a run directory against a gold set and render the acceptance report.

This is the read-only half of the benchmark. It opens a ``runs/TIMESTAMP/``
directory, matches each stored paper+mode leg to its gold case, runs the semantic
validator over whatever payload the leg actually produced, and reports five
separately-denominated rates, seven separately-counted scientific error classes,
and two failure taxonomies.

It calls no LLM, opens no network connection and writes nothing into the run
directory unless asked to. That is deliberate: an acceptance result you cannot
recompute from the artifacts is not evidence, and one that costs an overnight
batch to recompute will not be recomputed.

Payload precedence
------------------

Identity checks need the *mapped* payload -- ``merged_payload.json`` is
pre-mapping and carries no accessions at all, so every identity check run against
it is vacuously clean. The scorer therefore prefers ``final_mapped.json``, falls
back to ``merged_payload.json``, and **records which one it used** on every
result. A clean identity score sourced from a pre-mapping payload is reported as
``payload_source: merged_payload``, not as a pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from t2pw.bench.goldset import (
    EXPORT_STRICT,
    GoldCase,
    GoldSet,
    RELEVANCE_CONTEXT_ONLY,
    canonical_text,
    load_gold_set,
)
from t2pw.bench.metrics import (
    BLOCKERS_EXTRACTION,
    BLOCKERS_RESEARCH,
    BLOCKERS_STRICT,
    BLOCKER_SCOPES,
    BOUNDARY_EXTRACTION,
    BOUNDARY_LABELS,
    BOUNDARY_NONE,
    BOUNDARY_NOT_ATTEMPTED,
    BOUNDARY_ORDER,
    DENOMINATOR_ORDER,
    DENOM_RELEVANCE,
    DENOM_EXTRACTION,
    DENOM_RESEARCH_CONFIRMED,
    DENOM_RESEARCH_DELIVERABLE,
    DENOM_SEMANTIC,
    DENOM_STRICT,
    RESEARCH_FAILURE_LABELS,
    RESEARCH_FAILURE_ORDER,
    Blocker,
    Denominator,
    ErrorLedger,
    classify_research_failure,
    classify_strict_boundary,
    rank_blockers,
    tally,
)
from t2pw.bench.semantic import (
    ERR_FALSE_REAL_IDENTIFIERS,
    ERR_ORPHANED_REFERENCES,
    ERR_UNSUPPORTED_REACTIONS,
    SemanticReport,
    validate_semantic_coverage,
)
# The runtime verdict's own vocabulary, single-sourced. ``bench`` sits ABOVE
# ``pipeline``, so this is the layering's forward direction, not the inversion
# ``strict_quarantine`` had to declare; ``bench/render.py:23`` already imports
# from here at module level. Only the FAILED constant is taken: the affirmative
# states are deliberately not named in this module (see
# ``ModeResult.runtime_semantic_refuted``).
from t2pw.pipeline.release_status import SEMANTIC_FAILED as _RUNTIME_SEMANTIC_FAILED


MODE_STRICT = "strict"
MODE_RESEARCH = "research"
MODES: Tuple[str, ...] = (MODE_STRICT, MODE_RESEARCH)

#: Preference order. First one present on disk wins; the choice is recorded.
_PAYLOAD_FILES: Tuple[str, ...] = ("final_mapped.json", "merged_payload.json")

_ADMISSION_FILES: Tuple[str, ...] = ("rag_admission_report.json",)
_QUARANTINE_FILES: Tuple[str, ...] = ("quarantine_report.json",)

#: Artifacts whose presence means the strict leg actually emitted an importable
#: file. Mirrors ``batch.runner.REQUIRED_ARTIFACTS["strict"]``, which D-004 splits
#: by release state -- so BOTH names are deliverables here. "Did an importable
#: file land?" and "may this count as strict success?" are different questions,
#: and this constant answers only the first: the second is answered by the frozen
#: record in ``release_status``, never by a filename.
_STRICT_DELIVERABLES: Tuple[str, ...] = ("pathway.pwml", "pathway.review_required.pwml")
_RESEARCH_DELIVERABLE = "research_pathway_report.txt"

#: The exact text handed to the app, written by the batch runner as
#: ``SOURCE_TEXT_NAME``. Gold quotes are verified against THIS file.
_PAPER_TEXT_NAME = "01_source_text.txt"


def _paper_text(run_dir: Path, slug: str) -> str:
    """The stored full text for a paper, or ``""`` when the run did not keep it."""

    if not slug:
        return ""
    path = run_dir / "papers" / slug / _PAPER_TEXT_NAME
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_manifest(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "manifest.jsonl"
    rows: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Per-leg result.
# ---------------------------------------------------------------------------
@dataclass
class ModeResult:
    """One paper in one mode."""

    paper_id: str
    mode: str
    attempted: bool
    status: str = ""
    stage: str = ""
    failure_kind: str = ""
    message: str = ""
    issue_codes: List[str] = field(default_factory=list)
    boundary: str = BOUNDARY_NOT_ATTEMPTED
    boundary_evidence: str = ""
    research_failure: str = ""
    research_failure_evidence: str = ""
    payload_source: str = "none"
    payload_path: str = ""
    deliverable: bool = False
    semantic: Optional[SemanticReport] = None
    seconds: float = 0.0
    #: The release classification the manifest row carried, verbatim, or ``None``
    #: when the row recorded none. Never derived from a filename or from
    #: ``status``, and never re-derived here: this module scores runs, it does not
    #: classify them.
    release_status: Optional[Dict[str, Any]] = None
    #: The PWML filename the leg wrote, as the row reported it.
    pwml_artifact: str = ""

    @property
    def passed(self) -> bool:
        return self.status.casefold() == "pass"

    @property
    def strict_acceptance_eligible(self) -> bool:
        """May this leg count toward the STRICT benchmark denominator?

        AFFIRMATIVE by construction: only a measured ``True`` inside the frozen
        record qualifies. An absent record, a record this reader cannot parse and
        a missing flag are all "not measured", and "not measured" is never
        "eligible" -- so a run whose classification never arrived is excluded
        without anyone having to enumerate the ways a record can go missing
        (D-038 3). ``review_required`` is False by the invariant that builds the
        record (``release_status.py:317``), which is why this reads the flag
        rather than re-testing the state.
        """

        record = self.release_status
        return isinstance(record, dict) and record.get("strict_acceptance_eligible") is True

    @property
    def runtime_semantic_evaluation(self) -> str:
        """The RUNTIME semantic verdict this leg recorded at freeze time, or ``""``.

        Three-valued (``passed`` / ``failed`` / ``not_evaluated``) and read
        verbatim -- this module scores runs, it does not classify them. ``""``
        means the row carried no classification at all, which is a fourth thing
        and never any of the three.
        """

        record = self.release_status
        if not isinstance(record, dict):
            return ""
        return str(record.get("semantic_evaluation") or "")

    @property
    def runtime_semantic_failed_checks(self) -> List[str]:
        """The gating checks the runtime recorded as FAILED, by name.

        C-056b persisted these (``release_status.ReleaseStatus``); before that
        they existed only inside a ``reasons`` string. They are the EVIDENCE
        behind a missed confirmation, never the trigger for one -- see
        :attr:`runtime_semantic_refuted`, which is what the denominators read.
        """

        record = self.release_status
        if not isinstance(record, dict):
            return []
        return [str(name) for name in (record.get("semantic_failed_checks") or []) if name]

    @property
    def runtime_semantic_refuted(self) -> bool:
        """The runtime measured a gating semantic check as FAILED on this leg.

        THERE IS DELIBERATELY NO AFFIRMATIVE TWIN OF THIS PROPERTY, and that
        absence is the whole safeguard. A runtime ``passed`` is NOT evidence that
        a pathway is semantically right: the gating set is closed at four, but
        ``CHECK_RAG_REINTRODUCTION`` is structurally unevaluable at the
        quarantine seam (``quarantine_and_close`` takes no ``admission``
        parameter, D-042 section 4), and ``CHECK_ANCHORS`` / ``CHECK_ORGANISM``
        evaluate only under a derivation that supplies them. Measured on the 32
        committed payload legs (``evidence/c056b_s0_measured.json``): under the
        seam's own ``pathway_context`` derivation **every one of the 32 had
        exactly ONE evaluable gating check**, and 25 of them answered ``passed``
        on that single check. Serialized as a bare string, that ``passed`` is
        indistinguishable from a four-of-four pass.

        So the runtime verdict enters the benchmark in ONE direction only. It
        can REMOVE a success -- a leg the product itself recorded as
        semantically refuted is not confirmed, whatever a re-scoring says -- and
        it can never ADD one. The affirmative case has no accessor to reach for,
        so a later reader cannot turn a one-of-four pass into a numerator by
        picking the convenient property. ``bench.semantic``'s ``confirmed``
        stays the only source of a semantic PASS, and it already requires that
        every check was evaluable.
        """

        return self.runtime_semantic_evaluation == _RUNTIME_SEMANTIC_FAILED

    @property
    def extracted(self) -> bool:
        """Produced a payload with at least one process."""

        if self.semantic is None or not self.semantic.evaluated:
            return False
        return int(self.semantic.graph.get("n_reactions", 0)) > 0

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "paper_id": self.paper_id,
            "mode": self.mode,
            "attempted": self.attempted,
            "status": self.status,
            "stage": self.stage,
            "failure_kind": self.failure_kind,
            "message": self.message,
            "issue_codes": self.issue_codes,
            "boundary": self.boundary,
            "boundary_label": BOUNDARY_LABELS.get(self.boundary, self.boundary),
            "boundary_evidence": self.boundary_evidence,
            "research_failure": self.research_failure,
            "research_failure_label": RESEARCH_FAILURE_LABELS.get(self.research_failure, ""),
            "research_failure_evidence": self.research_failure_evidence,
            "payload_source": self.payload_source,
            "payload_path": self.payload_path,
            "deliverable": self.deliverable,
            "seconds": self.seconds,
            "semantic": self.semantic.to_dict() if self.semantic else None,
        }
        # Conditional, so a leg from a run that recorded neither serializes
        # byte-identically to before -- and an absent classification stays absent
        # rather than becoming a placeholder that reads like a measurement.
        if self.release_status:
            data["release_status"] = dict(self.release_status)
        if self.pwml_artifact:
            data["pwml_artifact"] = self.pwml_artifact
        return data


@dataclass
class PaperResult:
    case: GoldCase
    slug: str = ""
    legs: Dict[str, ModeResult] = field(default_factory=dict)
    #: Whether 01_source_text.txt was on disk, i.e. whether gold quotes could
    #: be verified at all for this paper.
    paper_text_available: bool = False

    @property
    def paper_id(self) -> str:
        return self.case.paper_id

    def leg(self, mode: str) -> Optional[ModeResult]:
        return self.legs.get(mode)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.case.title,
            "slug": self.slug,
            "requested_pathway": self.case.requested_pathway,
            "requested_organism": self.case.requested_organism,
            "actual_organism": self.case.actual_organism,
            "mechanistic_relevance": self.case.mechanistic_relevance,
            "expected_export": self.case.expected_export,
            "is_negative_control": self.case.is_negative_control,
            "paper_text_available": self.paper_text_available,
            "legs": {mode: leg.to_dict() for mode, leg in self.legs.items()},
        }


# ---------------------------------------------------------------------------
# The report.
# ---------------------------------------------------------------------------
@dataclass
class AcceptanceReport:
    run_dir: str
    gold_version: str
    gold_path: str
    papers: List[PaperResult] = field(default_factory=list)
    denominators: Dict[str, Denominator] = field(default_factory=dict)
    errors: ErrorLedger = field(default_factory=ErrorLedger)
    #: Blockers per scope -- strict / research / extraction, never merged.
    blockers: Dict[str, List[Blocker]] = field(default_factory=dict)
    strict_boundaries: Dict[str, int] = field(default_factory=dict)
    research_failures: Dict[str, int] = field(default_factory=dict)
    identity: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # -- coverage -------------------------------------------------------
    def completion(self) -> Dict[str, Any]:
        """How much of the benchmark actually ran, reported on its own.

        Kept apart from every rate on purpose. "0/4 strict" and "1 of 4 strict
        legs never ran" are different facts, and a reader who sees only the rate
        will attribute missing coverage to the exporter.
        """

        planned = len(self.papers)
        papers_attempted = sum(
            1 for p in self.papers if any(leg.attempted for leg in p.legs.values())
        )
        strict_attempted = sum(
            1 for p in self.papers if (p.leg(MODE_STRICT) or ModeResult("", "", False)).attempted
        )
        research_attempted = sum(
            1 for p in self.papers if (p.leg(MODE_RESEARCH) or ModeResult("", "", False)).attempted
        )
        payloads = sum(
            1
            for p in self.papers
            for leg in p.legs.values()
            if leg.payload_source not in ("", "none")
        )
        complete_cases = sum(
            1
            for p in self.papers
            if all((p.leg(m) is not None and p.leg(m).attempted) for m in MODES)
        )
        unevaluated = [
            p.paper_id
            for p in self.papers
            if not any(leg.attempted for leg in p.legs.values())
        ]
        partly = [
            p.paper_id
            for p in self.papers
            if any(leg.attempted for leg in p.legs.values())
            and not all((p.leg(m) is not None and p.leg(m).attempted) for m in MODES)
        ]
        return {
            "planned_gold_cases": planned,
            "papers_attempted": papers_attempted,
            "strict_legs_attempted": strict_attempted,
            "research_legs_attempted": research_attempted,
            "legs_attempted": self.legs_attempted,
            "legs_planned": planned * len(MODES),
            "payloads_available": payloads,
            "semantically_scorable_legs": self.legs_scored,
            "fully_completed_cases": complete_cases,
            "papers_with_no_attempted_leg": unevaluated,
            "papers_with_only_one_mode_attempted": partly,
            "complete": complete_cases == planned and planned > 0,
            "rendered": f"{papers_attempted}/{planned} papers, {self.legs_attempted}/{planned * len(MODES)} legs",
        }

    @property
    def is_complete(self) -> bool:
        """Every planned leg exists. A full benchmark report requires this."""

        return bool(self.completion()["complete"])

    @property
    def legs_attempted(self) -> int:
        """How many paper+mode legs the run actually ran.

        Zero is the trap this exists for: a run that attempted nothing has no
        scientific errors, so acceptance priorities 1-3 all hold vacuously and
        the report reads like a clean bill of health. Callers must gate on this
        before treating a green result as a result.
        """

        return sum(1 for paper in self.papers for leg in paper.legs.values() if leg.attempted)

    @property
    def legs_scored(self) -> int:
        """Legs that produced a payload the semantic validator could evaluate."""

        return sum(
            1
            for paper in self.papers
            for leg in paper.legs.values()
            if leg.semantic is not None and leg.semantic.evaluated
        )

    # -- acceptance priorities -------------------------------------------
    def priorities(self) -> List[Dict[str, Any]]:
        """The user-declared acceptance priorities, in order, as pass/fail.

        Priorities 1-3 are absolute: any non-zero count fails them, regardless of
        how good the rest of the run looks. Priority 4 is a coverage judgement and
        priority 5 is a rate to maximise, so neither is a hard gate.
        """

        totals = self.errors.totals
        semantic = self.denominators.get(DENOM_SEMANTIC)
        strict = self.denominators.get(DENOM_STRICT)

        false_ids = totals.get(ERR_FALSE_REAL_IDENTIFIERS, 0)
        unsupported = totals.get(ERR_UNSUPPORTED_REACTIONS, 0)
        orphans = totals.get(ERR_ORPHANED_REFERENCES, 0)

        return [
            {
                "rank": 1,
                "name": "zero known false real identifiers",
                "ok": false_ids == 0,
                "observed": false_ids,
                "papers": sorted(self.errors.papers_affected.get(ERR_FALSE_REAL_IDENTIFIERS, set())),
            },
            {
                "rank": 2,
                "name": "zero unsupported retained reactions",
                "ok": unsupported == 0,
                "observed": unsupported,
                "papers": sorted(self.errors.papers_affected.get(ERR_UNSUPPORTED_REACTIONS, set())),
            },
            {
                "rank": 3,
                "name": "zero referential-integrity violations",
                "ok": orphans == 0,
                "observed": orphans,
                "papers": sorted(self.errors.papers_affected.get(ERR_ORPHANED_REFERENCES, set())),
            },
            {
                "rank": 4,
                "name": "meaningful requested-pathway coverage",
                "ok": bool(semantic and semantic.rate is not None and semantic.rate > 0),
                "observed": semantic.render() if semantic else "n/a",
                "papers": sorted(semantic.numerator_names) if semantic else [],
            },
            {
                "rank": 5,
                "name": "maximize strict PWML pass rate among eligible/exportable papers",
                "ok": bool(strict and strict.rate is not None and strict.rate > 0),
                "observed": strict.render() if strict else "n/a",
                "papers": sorted(strict.numerator_names) if strict else [],
            },
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "gold_set": {"version": self.gold_version, "path": self.gold_path},
            "legs_attempted": self.legs_attempted,
            "legs_scored": self.legs_scored,
            "completion": self.completion(),
            "acceptance_priorities": self.priorities(),
            "denominators": {
                key: self.denominators[key].to_dict()
                for key in DENOMINATOR_ORDER
                if key in self.denominators
            },
            "scientific_errors": self.errors.to_dict(),
            "strict_failures_by_boundary": self.strict_boundaries,
            "research_failures_by_cause": self.research_failures,
            "identity": self.identity,
            "blockers_by_scope": {
                scope: [b.to_dict() for b in self.blockers.get(scope, [])]
                for scope in BLOCKER_SCOPES
            },
            "papers": [p.to_dict() for p in self.papers],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------
def _find_slug(run_dir: Path, paper_id: str, rows: Sequence[Mapping[str, Any]]) -> str:
    wanted = paper_id.casefold()
    for row in rows:
        if canonical_text(row.get("paper_id")).casefold() == wanted:
            slug = canonical_text(row.get("slug"))
            if slug:
                return slug
    papers_dir = run_dir / "papers"
    if papers_dir.is_dir():
        for child in sorted(papers_dir.iterdir()):
            if child.is_dir() and child.name.split("__")[0].casefold() == wanted:
                return child.name
    return ""


def _first_existing(directory: Path, names: Sequence[str]) -> Tuple[Optional[Any], str, str]:
    for name in names:
        path = directory / name
        if path.is_file():
            blob = _read_json(path)
            if blob is not None:
                return blob, name, str(path)
    return None, "none", ""


def score_run(
    run_dir: Any,
    gold: Optional[GoldSet] = None,
    *,
    gold_path: Optional[Any] = None,
) -> AcceptanceReport:
    """Score every gold case against the artifacts stored in ``run_dir``."""

    run = Path(run_dir)
    gold_set = gold or load_gold_set(gold_path)
    rows = _load_manifest(run)

    report = AcceptanceReport(
        run_dir=str(run),
        gold_version=gold_set.version,
        gold_path=gold_set.source_path,
    )

    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        paper = canonical_text(row.get("paper_id")).casefold()
        mode = canonical_text(row.get("mode")).casefold()
        if paper and mode:
            # Last row wins: a retried leg supersedes its earlier attempt, the
            # same rule batch.report.group_papers uses.
            by_key[(paper, mode)] = row

    for case in gold_set:
        slug = _find_slug(run, case.paper_id, rows)
        paper_result = PaperResult(case=case, slug=slug)
        # The exact string the app was handed. Quote verification must run
        # against this and nothing else: checking a gold quote against a
        # re-fetched copy of the paper would verify a different document than the
        # one the pipeline actually read.
        paper_text = _paper_text(run, slug)
        paper_result.paper_text_available = bool(paper_text)

        for mode in MODES:
            row = by_key.get((case.paper_id.casefold(), mode))
            leg = ModeResult(paper_id=case.paper_id, mode=mode, attempted=row is not None)

            if row is not None:
                leg.status = canonical_text(row.get("status"))
                leg.stage = canonical_text(row.get("stage"))
                leg.failure_kind = canonical_text(row.get("failure_kind"))
                leg.message = str(row.get("message") or "")
                leg.issue_codes = [str(c) for c in (row.get("issue_codes") or []) if c]
                try:
                    leg.seconds = float(row.get("seconds") or 0.0)
                except (TypeError, ValueError):
                    leg.seconds = 0.0
                names = {
                    Path(str(f.get("name", ""))).name
                    for f in (row.get("files") or [])
                    if isinstance(f, dict)
                }
                wanted = (
                    _STRICT_DELIVERABLES
                    if mode == MODE_STRICT
                    else (_RESEARCH_DELIVERABLE,)
                )
                leg.deliverable = any(name in names for name in wanted)
                # D-004's two new row keys, carried across unchanged. A row from
                # before this card has neither, and both stay falsy there.
                record = row.get("release_status")
                leg.release_status = dict(record) if isinstance(record, dict) and record else None
                leg.pwml_artifact = str(row.get("pwml_artifact") or "")

            leg.boundary, leg.boundary_evidence = classify_strict_boundary(row)
            if mode == MODE_RESEARCH and row is not None and not leg.passed:
                leg.research_failure, leg.research_failure_evidence = classify_research_failure(row)

            leg_dir = run / "papers" / slug / mode if slug else None
            payload = admission = quarantine = None
            if leg_dir is not None and leg_dir.is_dir():
                payload, source, path = _first_existing(leg_dir, _PAYLOAD_FILES)
                leg.payload_source = source
                leg.payload_path = path
                admission, _, _ = _first_existing(leg_dir, _ADMISSION_FILES)
                quarantine, _, _ = _first_existing(leg_dir, _QUARANTINE_FILES)

            leg.semantic = validate_semantic_coverage(
                case,
                payload if isinstance(payload, dict) else None,
                mode=mode,
                admission=admission if isinstance(admission, dict) else None,
                quarantine_report=quarantine if isinstance(quarantine, dict) else None,
                paper_text=paper_text,
            )
            if leg.semantic.evaluated:
                report.errors.add(case.paper_id, mode, leg.semantic.scientific_errors)

            paper_result.legs[mode] = leg

        report.papers.append(paper_result)

    _build_denominators(report, gold_set)
    _build_boundaries(report)
    _build_identity(report)
    _build_blockers(report)
    _build_notes(report)
    return report


def _build_denominators(report: AcceptanceReport, gold_set: GoldSet) -> None:
    """Rates over what was ACTUALLY ATTEMPTED, never over what was planned.

    The rule every denominator here obeys: **an unattempted or unscorable paper
    is not a pipeline failure.** Scoring a partial run against the full gold set
    charges the pipeline for legs that never executed -- ``runs/2026-07-28_2122``
    covers 7 of 10 pinned papers, and the previous version reported its strict
    rate as 0/4 when only 1 of those 4 strict-exportable papers had its strict
    leg attempted at all. The other three are *missing coverage*, reported by
    :meth:`AcceptanceReport.completion`, not evidence about the exporter.
    """

    relevant: List[str] = []
    all_papers: List[str] = []
    excluded_relevance: List[Dict[str, str]] = []
    excluded_strict: List[Dict[str, str]] = []
    excluded_extraction: List[Dict[str, str]] = []
    excluded_semantic: List[Dict[str, str]] = []
    excluded_research: List[Dict[str, str]] = []

    extraction_pool: List[str] = []
    extracted: List[str] = []
    semantic_pool: List[str] = []
    semantic_ok: List[str] = []
    strict_pool: List[str] = []
    strict_ok: List[str] = []
    research_pool: List[str] = []
    research_produced: List[str] = []
    research_ok: List[str] = []

    for paper in report.papers:
        case = paper.case
        pid = case.paper_id
        all_papers.append(pid)
        strict_leg = paper.leg(MODE_STRICT)
        research_leg = paper.leg(MODE_RESEARCH)

        is_relevant = case.mechanistic_relevance != RELEVANCE_CONTEXT_ONLY
        if is_relevant:
            relevant.append(pid)
        else:
            excluded_relevance.append(
                {
                    "paper_id": pid,
                    "reason": f"mechanistic_relevance={case.mechanistic_relevance}"
                    + (" (negative control)" if case.is_negative_control else ""),
                }
            )

        # -- extraction: relevant papers whose legs were actually attempted ----
        attempted_any = any(leg.attempted for leg in paper.legs.values())
        if is_relevant:
            if attempted_any:
                extraction_pool.append(pid)
                # The extractor is shared between modes, so a paper that
                # extracted in either mode extracted. Charging it twice would
                # make extraction look mode-dependent when it is not.
                if any(leg.extracted for leg in paper.legs.values()):
                    extracted.append(pid)
            else:
                excluded_extraction.append(
                    {"paper_id": pid, "reason": "no leg was attempted in this run"}
                )

        # -- semantic: graded on the first leg that produced a scorable payload -
        chosen = None
        for mode in (MODE_STRICT, MODE_RESEARCH):
            leg = paper.leg(mode)
            if leg is not None and leg.semantic is not None and leg.semantic.evaluated:
                chosen = leg
                break
        if is_relevant:
            if chosen is not None and chosen.semantic is not None:
                semantic_pool.append(pid)
                # `confirmed`, not `ok`: `ok` is true when every *applicable*
                # check passes, so a leg whose RAG-reintroduction check could not
                # run at all counts as a success. `confirmed` additionally
                # requires that every check WAS evaluable, which is the only
                # honest reading of "this pathway is semantically right".
                #
                # AND the run's own recorded verdict, in the SUBTRACTIVE
                # direction only (C-056b). Two evaluations exist now and they can
                # disagree: this one re-scores the stored payload against the
                # gold case, while `runtime_semantic_refuted` reports what the
                # pipeline measured on the reduced strict graph at freeze time,
                # under its own pinned request derivation. A leg the product
                # itself classified as semantically refuted is not "confirmed",
                # whatever a re-scoring concludes -- PRODUCT_CONTRACT 11 makes
                # the runtime `release_status` the place the semantic states
                # live, and a benchmark that overrode it upward would be scoring
                # a pathway the product declined to vouch for.
                #
                # It can only ever REMOVE a success. A runtime `passed` is not
                # consulted anywhere and has no accessor: measured, every one of
                # the 32 committed payload legs had exactly ONE of the four
                # gating checks evaluable at this seam, so `passed` there means
                # "the one check that could run, ran clean" and reading it as a
                # numerator would inflate this rate outright. `not_evaluated`
                # neither adds nor removes -- it is never `false`.
                #
                # The paper STAYS in the population either way: a refuted leg is
                # a miss, not an exclusion, and moving it out of the denominator
                # would improve the rate by deleting the evidence against it.
                # Which checks the runtime named travels per leg, verbatim, in
                # ``ModeResult.to_dict()["release_status"]["semantic_failed_checks"]``.
                if chosen.semantic.confirmed and not chosen.runtime_semantic_refuted:
                    semantic_ok.append(pid)
            else:
                excluded_semantic.append(
                    {
                        "paper_id": pid,
                        "reason": "no leg produced a payload the validator could score",
                    }
                )

        # -- strict PWML: strict_exportable AND the strict leg was attempted ----
        if case.expected_export != EXPORT_STRICT:
            excluded_strict.append(
                {
                    "paper_id": pid,
                    "reason": f"expected_export={case.expected_export}: "
                    + (case.export_rationale[:150] or "the source cannot support a complete pathway"),
                }
            )
        elif strict_leg is None or not strict_leg.attempted:
            excluded_strict.append(
                {"paper_id": pid, "reason": "strict-exportable, but its strict leg was never attempted"}
            )
        else:
            strict_pool.append(pid)
            # Three separate facts, and none of them implies another: the leg
            # completed, an importable file landed, and the run MEASURED itself
            # release-ready. The third is the one D-004 adds. Without it a
            # ``review_required`` export counts as strict success purely because
            # its file used to be called ``pathway.pwml`` -- TRAP-1, and
            # PRODUCT_CONTRACT 13's "Never strict success". The test is
            # affirmative, so a leg whose classification never arrived is
            # excluded rather than assumed good.
            if strict_leg.passed and strict_leg.deliverable and strict_leg.strict_acceptance_eligible:
                strict_ok.append(pid)

        # -- research: relevant AND the research leg was attempted --------------
        if not is_relevant:
            excluded_research.append(
                {"paper_id": pid, "reason": f"mechanistic_relevance={case.mechanistic_relevance}"}
            )
        elif research_leg is None or not research_leg.attempted:
            excluded_research.append(
                {"paper_id": pid, "reason": "eligible, but its research leg was never attempted"}
            )
        else:
            research_pool.append(pid)
            # Two INDEPENDENT questions, never collapsed into one rate.
            if research_leg.deliverable:
                research_produced.append(pid)
            semantic = research_leg.semantic
            # Same subtractive rule as the semantic denominator above, on the
            # same grounds and in the same direction: a leg the runtime recorded
            # as semantically refuted is not "scientifically confirmed", and a
            # runtime pass is not consulted.
            if (
                research_leg.deliverable
                and semantic is not None
                and semantic.confirmed
                and not research_leg.runtime_semantic_refuted
            ):
                priority_errors = (
                    semantic.scientific_errors.get(ERR_FALSE_REAL_IDENTIFIERS, 0)
                    + semantic.scientific_errors.get(ERR_UNSUPPORTED_REACTIONS, 0)
                    + semantic.scientific_errors.get(ERR_ORPHANED_REFERENCES, 0)
                )
                if priority_errors == 0:
                    research_ok.append(pid)

    report.denominators[DENOM_RELEVANCE] = Denominator(
        key=DENOM_RELEVANCE,
        question=(
            "What fraction of the pinned gold set is mechanistically relevant to its requested "
            "pathway? This is a property of the GOLD SET, not a measurement of the screener."
        ),
        population="every paper in the pinned gold set",
        numerator_names=relevant,
        denominator_names=all_papers,
        excluded=excluded_relevance,
    )
    report.denominators[DENOM_EXTRACTION] = Denominator(
        key=DENOM_EXTRACTION,
        question="Of the relevant papers whose legs were attempted, how many produced a payload with at least one reaction?",
        population="relevant gold cases with at least one attempted leg",
        numerator_names=extracted,
        denominator_names=extraction_pool,
        excluded=excluded_extraction,
    )
    report.denominators[DENOM_SEMANTIC] = Denominator(
        key=DENOM_SEMANTIC,
        question=(
            "Of the relevant papers with a scorable payload, how many were CONFIRMED -- every "
            "check passed, every check was evaluable, and the run itself did not record a "
            "FAILED runtime semantic verdict? The runtime verdict can only REMOVE a "
            "confirmation here; a runtime pass is never counted as one."
        ),
        population="relevant gold cases with at least one scorable payload",
        numerator_names=semantic_ok,
        denominator_names=semantic_pool,
        excluded=excluded_semantic,
    )
    report.denominators[DENOM_STRICT] = Denominator(
        key=DENOM_STRICT,
        question="Of the strict-exportable papers whose strict leg was attempted, how many exported PWML?",
        population="strict_exportable gold cases with an attempted strict leg",
        numerator_names=strict_ok,
        denominator_names=strict_pool,
        excluded=excluded_strict,
    )
    report.denominators[DENOM_RESEARCH_DELIVERABLE] = Denominator(
        key=DENOM_RESEARCH_DELIVERABLE,
        question="Of the relevant papers whose research leg was attempted, how many PRODUCED a reviewable deliverable? (output only -- says nothing about correctness)",
        population="relevant gold cases with an attempted research leg",
        numerator_names=research_produced,
        denominator_names=research_pool,
        excluded=excluded_research,
    )
    report.denominators[DENOM_RESEARCH_CONFIRMED] = Denominator(
        key=DENOM_RESEARCH_CONFIRMED,
        question=(
            "Of those, how many are scientifically CONFIRMED -- deliverable produced, every "
            "semantic check passed AND evaluable, no priority-1..3 error, and no FAILED "
            "runtime semantic verdict recorded by the run itself?"
        ),
        population="relevant gold cases with an attempted research leg",
        numerator_names=research_ok,
        denominator_names=research_pool,
        excluded=excluded_research,
    )


def _build_boundaries(report: AcceptanceReport) -> None:
    strict_boundaries: List[str] = []
    research_causes: List[str] = []
    for paper in report.papers:
        strict = paper.leg(MODE_STRICT)
        if strict is not None and strict.boundary not in (BOUNDARY_NONE,):
            strict_boundaries.append(strict.boundary)
        research = paper.leg(MODE_RESEARCH)
        if research is not None and research.research_failure:
            research_causes.append(research.research_failure)

    counted = tally(strict_boundaries)
    report.strict_boundaries = {key: counted[key] for key in BOUNDARY_ORDER if key in counted}
    counted_research = tally(research_causes)
    report.research_failures = {
        key: counted_research[key] for key in RESEARCH_FAILURE_ORDER if key in counted_research
    }


def _build_identity(report: AcceptanceReport) -> None:
    """Real IDs versus Unknown fallbacks, with the payload each number came from."""

    totals = {"verified": 0, "placeholder": 0, "unresolved": 0, "pathbank_unknown_sentinel": 0}
    per_paper: Dict[str, Dict[str, Any]] = {}
    sources = []

    for paper in report.papers:
        for mode, leg in paper.legs.items():
            if leg.semantic is None or not leg.semantic.evaluated:
                continue
            census = leg.semantic.identity_census
            for key in totals:
                totals[key] += int(census.get(key, 0) or 0)
            per_paper[f"{paper.paper_id}:{mode}"] = {
                "payload_source": leg.payload_source,
                **{key: int(census.get(key, 0) or 0) for key in totals},
                "proteins_total": int(census.get("proteins_total", 0) or 0),
                "proteins_with_organism": int(census.get("proteins_with_organism", 0) or 0),
            }
            sources.append(leg.payload_source)

    report.identity = {
        "totals": totals,
        "by_paper_mode": per_paper,
        "payload_sources": tally(sources),
        "caveat": (
            "Counts sourced from merged_payload.json are pre-mapping: that payload carries no "
            "accessions at all, so 'verified: 0' there means 'mapping had not run yet', NOT "
            "'mapping failed'. Only counts sourced from final_mapped.json measure real resolution."
        ),
    }


def _empty_is_correct(case: GoldCase) -> bool:
    """Whether producing nothing is the RIGHT outcome for this gold case.

    True for a negative control (``max_retained_reactions == 0``) and for any
    ``context_only`` case that declares no minimum core -- both are papers the
    gold set says contain no chemistry of the requested pathway, so an empty
    extraction is a success, not a blocker.
    """

    if case.is_negative_control:
        return True
    return case.mechanistic_relevance == RELEVANCE_CONTEXT_ONLY and case.min_connected_reactions == 0


def _build_blockers(report: AcceptanceReport) -> None:
    """Three separate rankings. Merging them produces an uncollectable number.

    "Papers released if fixed" only means something inside one population. A
    ``partial_only`` paper is not in the strict-PWML denominator, so fixing the
    Stage-3 gate that stops its strict leg cannot raise the strict rate by one
    paper -- yet a merged ranking will happily present it as the top strict
    blocker, which is exactly what happened with PMC12312563.
    """

    strict_failures: List[Dict[str, Any]] = []
    research_failures: List[Dict[str, Any]] = []
    extraction_failures: List[Dict[str, Any]] = []

    for paper in report.papers:
        case = paper.case
        for mode, leg in paper.legs.items():
            if leg.passed or not leg.attempted:
                continue
            entry = {
                "paper_id": paper.paper_id,
                "boundary": leg.boundary,
                "issue_codes": leg.issue_codes,
            }
            # A leg that produced no payload at all is an extraction problem in
            # either mode; it is upstream of both the strict and the research
            # question and belongs in neither of those rankings.
            #
            # UNLESS empty was the right answer. A negative control -- and any
            # context_only case whose ceiling is zero -- is *supposed* to yield
            # nothing, so an empty extraction there is the pipeline behaving
            # correctly. Listing it as an extraction blocker inverts the finding:
            # PMC13231680 appeared as the top `extraction_empty` blocker for
            # declining to invent a lipid A pathway from a paper that contains
            # none. Over-retention on such a paper is still reported -- as
            # unsupported retention, by the ceiling rule in the semantic
            # validator -- and never as a blocker.
            if leg.boundary == BOUNDARY_EXTRACTION or not leg.extracted:
                if _empty_is_correct(case):
                    continue
                extraction_failures.append(entry)
                continue
            if mode == MODE_STRICT:
                if case.expected_export == EXPORT_STRICT:
                    strict_failures.append(entry)
                # else: a partial_only paper's strict failure is real work but
                # cannot move the strict rate, so it is deliberately dropped
                # from the ranking rather than allowed to top it.
            else:
                if case.mechanistic_relevance != RELEVANCE_CONTEXT_ONLY:
                    research_failures.append(entry)

    report.blockers = {
        BLOCKERS_STRICT: rank_blockers(strict_failures),
        BLOCKERS_RESEARCH: rank_blockers(research_failures),
        BLOCKERS_EXTRACTION: rank_blockers(extraction_failures),
    }


def _build_notes(report: AcceptanceReport) -> None:
    completion = report.completion()
    if not completion["complete"]:
        missing = completion["papers_with_no_attempted_leg"]
        partial = completion["papers_with_only_one_mode_attempted"]
        report.notes.append(
            "PARTIAL RUN -- this is NOT a complete benchmark report. "
            f"{completion['rendered']}. Every rate below is computed over what was actually "
            "attempted, so an unattempted paper is never counted as a pipeline failure; but the "
            "run cannot be quoted as a benchmark result until the missing legs exist."
        )
        if missing:
            report.notes.append(
                f"gold cases with NO attempted leg ({len(missing)}): {', '.join(missing)}"
            )
        if partial:
            report.notes.append(
                f"gold cases with only one mode attempted ({len(partial)}): {', '.join(partial)}"
            )
    if not report.legs_attempted:
        report.notes.append(
            "NO LEG WAS ATTEMPTED in this run directory. Acceptance priorities 1-3 hold only "
            "because nothing ran: a run with no output has no scientific errors. Do not read this "
            "as a pass."
        )
    elif not report.legs_scored:
        report.notes.append(
            f"{report.legs_attempted} leg(s) were attempted but none produced a payload the "
            "validator could score, so every error count is 0 for lack of evidence, not for lack "
            "of errors."
        )
    sources = report.identity.get("payload_sources", {})
    if sources.get("merged_payload.json"):
        report.notes.append(
            f"{sources['merged_payload.json']} leg(s) were scored from merged_payload.json, which is "
            "pre-mapping. Their identity and false-identifier counts are floors, not measurements: "
            "run the batch with the updated driver so final_mapped.json is persisted on failure paths."
        )
    if sources.get("none"):
        report.notes.append(
            f"{sources['none']} leg(s) stored no payload at all and could not be scored semantically."
        )
    inapplicable = 0
    for paper in report.papers:
        for leg in paper.legs.values():
            if leg.semantic is not None:
                inapplicable += len(leg.semantic.inapplicable_checks)
    if inapplicable:
        report.notes.append(
            f"{inapplicable} check(s) across all legs could not be evaluated (most likely the RAG "
            "admission report, which the driver did not previously persist). They are excluded from "
            "'confirmed' and are NOT counted as passes."
        )


__all__ = [
    "MODES",
    "MODE_RESEARCH",
    "MODE_STRICT",
    "AcceptanceReport",
    "ModeResult",
    "PaperResult",
    "score_run",
]
