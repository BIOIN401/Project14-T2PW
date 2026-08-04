"""Refuse to launch or score an acceptance run whose scope was lost.

Why this exists
===============

``runs/2026-08-01_1724`` fetched all ten pinned papers, skipped none, wrote a
plan and reported a clean acquisition funnel — and was worthless as an acceptance
run, because every row carried ``requested_pathway: ""`` and
``requested_organism: ""``. A bare pinned id in the topics file produces
``TopicSpec(pinned_id=..., topic="", organism="")``, and ``_as_batch_paper``
copies those empty strings into the plan. Downstream, ``driver._extraction_focus``
hands extraction no scope at all and the screening record is computed against an
empty pathway, which is why every paper scored ``off_topic`` /
``no_mechanistic_pathway_terms`` while being accepted by ``pinned_override``.

Nothing in the batch runner noticed. The paper count was right, the skip count
was zero, and the failure is only visible if you open ``plan.json`` and read a
field that is *supposed* to be empty for an ordinary search topic. So this module
exists to make that specific silence impossible: it compares the plan against the
gold set field by field and refuses on any drift.

It is deliberately a *separate* gate rather than an assertion inside ``fetch``.
The batch runner is a general tool that legitimately accepts scopeless pinned ids
for exploratory runs; it is the **benchmark** that cannot tolerate them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from t2pw.bench.goldset import GoldSet, canonical_text, load_gold_set


#: The acquisition outcome a pinned paper must carry. Anything else means the
#: paper went through the ordinary screening path and its acceptance depended on
#: a score, which is exactly the non-reproducibility a pinned run exists to avoid.
EXPECTED_OUTCOME = "pinned_override"

#: Acquisition fields that prove no literature search happened. ``query`` is the
#: search string the topic branch records; the pinned branch passes ``query=""``.
_SEARCH_EVIDENCE_FIELDS = ("query",)

#: The two modes an acceptance run must plan, exactly. Not a subset and not a
#: superset: the benchmark's whole design is "the same papers through both
#: modes", so a plan missing ``research`` silently halves the run while still
#: producing a directory that scores, and an extra mode invents legs the gold set
#: has no expectations for.
REQUIRED_MODES: Tuple[str, ...] = ("strict", "research")


class PreflightError(RuntimeError):
    """The plan does not match the gold set. Always names every offending case."""


@dataclass
class ScopeFinding:
    paper_id: str
    field_name: str
    expected: str
    observed: str
    reason: str

    def render(self) -> str:
        return (
            f"{self.paper_id}: {self.field_name} is {self.observed!r}, "
            f"expected {self.expected!r} -- {self.reason}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "field": self.field_name,
            "expected": self.expected,
            "observed": self.observed,
            "reason": self.reason,
        }


@dataclass
class PreflightReport:
    """Per-case verdicts plus the findings that produced them."""

    gold_version: str
    source: str
    checked: List[str] = field(default_factory=list)
    findings: List[ScopeFinding] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    search_calls: int = 0
    duplicates: List[str] = field(default_factory=list)
    shape: List[str] = field(default_factory=list)
    triples: List[Dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.findings
            or self.missing
            or self.unexpected
            or self.search_calls
            or self.duplicates
            or self.shape
        )

    def raise_for_status(self) -> None:
        if self.ok:
            return
        lines = ["the acceptance plan does not match the gold set:"]
        for paper_id in self.missing:
            lines.append(f"  MISSING  {paper_id}: gold case has no row in the plan")
        for paper_id in self.unexpected:
            lines.append(f"  EXTRA    {paper_id}: plan row has no gold case")
        if self.search_calls:
            lines.append(
                f"  SEARCH   {self.search_calls} row(s) record a literature query; a pinned "
                "acceptance run must perform zero search calls"
            )
        for finding in self.findings:
            lines.append(f"  SCOPE    {finding.render()}")
        raise PreflightError("\n".join(lines))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gold_version": self.gold_version,
            "source": self.source,
            "ok": self.ok,
            "checked": self.checked,
            "missing_from_plan": self.missing,
            "not_in_gold_set": self.unexpected,
            "search_calls": self.search_calls,
            "duplicate_paper_ids": self.duplicates,
            "shape_violations": self.shape,
            "findings": [f.to_dict() for f in self.findings],
            "triples": self.triples,
        }

    def render(self) -> str:
        lines = [
            f"pinned acceptance preflight against gold set {self.gold_version}",
            f"source: {self.source}",
            f"verdict: {'OK' if self.ok else 'REFUSED'}",
            f"cases checked: {len(self.checked)}   search calls: {self.search_calls}",
            "",
            "paper / pathway / organism triples:",
        ]
        width = max((len(t["paper_id"]) for t in self.triples), default=12)
        for triple in self.triples:
            lines.append(
                f"  {triple['paper_id']:<{width}} | {triple['requested_pathway']} | "
                f"{triple['requested_organism']}   [{triple['outcome']}]"
            )
        if not self.ok:
            lines.append("")
            lines.append("findings:")
            for paper_id in self.missing:
                lines.append(f"  MISSING  {paper_id}")
            for paper_id in self.unexpected:
                lines.append(f"  EXTRA    {paper_id}")
            for finding in self.findings:
                lines.append(f"  SCOPE    {finding.render()}")
        return "\n".join(lines) + "\n"


def _row_scope(row: Mapping[str, Any]) -> Dict[str, str]:
    """Scope as the plan records it, tolerating the pre-2026-07-29 spelling."""

    pathway = canonical_text(row.get("requested_pathway") or row.get("topic"))
    organism = canonical_text(row.get("requested_organism") or row.get("organism"))
    return {"requested_pathway": pathway, "requested_organism": organism}


def _outcome(row: Mapping[str, Any]) -> str:
    eligibility = row.get("eligibility")
    if isinstance(eligibility, dict):
        return canonical_text(eligibility.get("outcome"))
    return ""


def verify_rows(
    rows: Sequence[Mapping[str, Any]],
    gold: GoldSet,
    *,
    source: str = "",
    require_outcome: bool = True,
) -> PreflightReport:
    """Check plan rows (or ``BatchPaper.to_dict()`` output) against the gold set.

    ``require_outcome`` is on for a stored plan, where the acquisition step has
    definitely run and must have recorded ``pinned_override``. It is off only for
    :func:`verify_specs`, which runs *before* any fetch and therefore before any
    outcome exists.
    """

    report = PreflightReport(gold_version=gold.version, source=source)
    by_id: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        paper_id = canonical_text(row.get("paper_id"))
        if not paper_id:
            continue
        key = paper_id.casefold()
        if key in by_id:
            # Previously the later row silently won. Two rows for one paper means
            # the plan is ambiguous about what will run -- and because the batch
            # keys resume state on (slug, mode), a duplicate can also consume a
            # pair slot twice and quietly halve the coverage.
            report.duplicates.append(paper_id)
            report.findings.append(
                ScopeFinding(
                    paper_id=paper_id,
                    field_name="paper_id",
                    expected="exactly one row",
                    observed="more than one row",
                    reason="a duplicate plan row makes the run ambiguous and is never silently kept",
                )
            )
            continue
        by_id[key] = row

    for case in gold:
        key = case.paper_id.casefold()
        row = by_id.pop(key, None)
        if row is None:
            report.missing.append(case.paper_id)
            continue
        report.checked.append(case.paper_id)

        observed = _row_scope(row)
        expected = {
            "requested_pathway": case.requested_pathway,
            "requested_organism": case.requested_organism,
        }
        for field_name, want in expected.items():
            got = observed[field_name]
            if got == want:
                continue
            reason = (
                "the pinned spec carried no scope, so extraction is handed nothing "
                "and the screening record is computed against an empty request"
                if not got
                else "the plan's scope does not match the gold case exactly"
            )
            report.findings.append(
                ScopeFinding(
                    paper_id=case.paper_id,
                    field_name=field_name,
                    expected=want,
                    observed=got,
                    reason=reason,
                )
            )

        outcome = _outcome(row)
        if require_outcome and not outcome:
            # A MISSING outcome used to pass. It must not: the absence of an
            # acquisition record is indistinguishable from a paper that was never
            # screened, and "we cannot tell how this paper got in" is not a
            # property an acceptance run may have.
            report.findings.append(
                ScopeFinding(
                    paper_id=case.paper_id,
                    field_name="eligibility.outcome",
                    expected=EXPECTED_OUTCOME,
                    observed="(absent)",
                    reason="the plan records no acquisition outcome, so admission cannot be audited",
                )
            )
        elif outcome and outcome != EXPECTED_OUTCOME:
            report.findings.append(
                ScopeFinding(
                    paper_id=case.paper_id,
                    field_name="eligibility.outcome",
                    expected=EXPECTED_OUTCOME,
                    observed=outcome,
                    reason="a pinned paper must be admitted by override, never by score",
                )
            )

        for name in _SEARCH_EVIDENCE_FIELDS:
            if canonical_text(row.get(name)):
                report.search_calls += 1
                report.findings.append(
                    ScopeFinding(
                        paper_id=case.paper_id,
                        field_name=name,
                        expected="",
                        observed=canonical_text(row.get(name)),
                        reason="a pinned paper is fetched by id and records no query",
                    )
                )

        report.triples.append(
            {
                "paper_id": case.paper_id,
                "requested_pathway": observed["requested_pathway"],
                "requested_organism": observed["requested_organism"],
                "outcome": outcome or "(not recorded)",
            }
        )

    report.unexpected = sorted(canonical_text(row.get("paper_id")) for row in by_id.values())
    return report


def verify_plan(
    plan_path: Any,
    gold: Optional[GoldSet] = None,
) -> PreflightReport:
    """Check a stored ``plan.json`` against the gold set."""

    path = Path(plan_path)
    if path.is_dir():
        path = path / "plan.json"
    gold_set = gold or load_gold_set()
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PreflightError(f"no plan to verify: {path}") from None
    except json.JSONDecodeError as exc:
        raise PreflightError(f"plan {path} is not valid JSON: {exc}") from None
    if not isinstance(blob, dict):
        raise PreflightError(f"plan {path}: top level must be an object")

    papers = blob.get("papers")
    if not isinstance(papers, list):
        raise PreflightError(f"plan {path}: 'papers' must be a list")

    rows = [p for p in papers if isinstance(p, dict)]
    report = verify_rows(rows, gold_set, source=str(path))

    # -- shape: the run must be exactly the size the gold set describes --------
    #
    # Per-row scope can be perfect while the run as a whole is the wrong shape.
    # A plan carrying only `modes: ["strict"]` halves the benchmark and still
    # produces a directory that scores cleanly on everything it did run; a plan
    # with nine papers is missing a tenth nobody will notice. These are counted
    # once, at the plan level, because they are properties of the run and not of
    # any paper.
    expected_papers = len(gold_set)
    if len(rows) != expected_papers:
        report.shape.append(
            f"plan lists {len(rows)} paper row(s); gold set {gold_set.version} has "
            f"{expected_papers}"
        )

    modes_raw = blob.get("modes")
    modes = [canonical_text(m).casefold() for m in modes_raw] if isinstance(modes_raw, list) else []
    if sorted(modes) != sorted(REQUIRED_MODES):
        report.shape.append(
            f"plan modes are {modes or '(absent)'}; an acceptance run must plan exactly "
            f"{list(REQUIRED_MODES)} -- the benchmark compares the same papers across both"
        )

    expected_pairs = expected_papers * len(REQUIRED_MODES)
    planned_pairs = len(rows) * len(modes)
    if planned_pairs != expected_pairs:
        report.shape.append(
            f"plan schedules {planned_pairs} paper/mode pair(s); gold set "
            f"{gold_set.version} requires {expected_pairs}"
        )

    # A search-derived acquisition would leave per-topic tallies with a non-zero
    # search_calls count even when the individual rows look clean.
    eligibility = blob.get("eligibility")
    if isinstance(eligibility, dict):
        acquisition = eligibility.get("acquisition")
        if isinstance(acquisition, dict):
            for topic in acquisition.get("topics") or ():
                if not isinstance(topic, dict):
                    continue
                try:
                    calls = int(topic.get("search_calls", 0) or 0)
                except (TypeError, ValueError):
                    calls = 0
                report.search_calls += calls
    return report


def verify_specs(specs: Iterable[Any], gold: GoldSet) -> PreflightReport:
    """Check parsed :class:`~t2pw.batch.fetch.TopicSpec` objects before any fetch.

    The cheapest possible gate: it runs before the network is touched, so a
    topics file that lost its scope is refused in milliseconds rather than after
    a fetch that looks entirely successful.
    """

    rows = [
        {
            "paper_id": getattr(spec, "pinned_id", ""),
            "requested_pathway": getattr(spec, "topic", ""),
            "requested_organism": getattr(spec, "organism", ""),
            "query": "",
            "eligibility": {"outcome": EXPECTED_OUTCOME}
            if getattr(spec, "is_pinned", False)
            else {"outcome": "eligible"},
        }
        for spec in specs
    ]
    # `require_outcome=False`: this runs BEFORE any fetch, so no acquisition
    # outcome exists yet and demanding one would refuse every valid topics file.
    return verify_rows(rows, gold, source="parsed topic specs", require_outcome=False)


__all__ = [
    "EXPECTED_OUTCOME",
    "REQUIRED_MODES",
    "PreflightError",
    "PreflightReport",
    "ScopeFinding",
    "verify_plan",
    "verify_rows",
    "verify_specs",
]
