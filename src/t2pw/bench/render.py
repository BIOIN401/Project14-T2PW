"""Render an :class:`~t2pw.bench.acceptance.AcceptanceReport` as plain text.

Kept apart from the scoring so a formatting change can never alter a number, and
so the JSON form stays the machine-readable source of truth. The section order
follows the question order a reader actually has: *did we clear the bar, on what
population, what went wrong, and what should I fix next.*
"""

from __future__ import annotations

from typing import Any, List

from t2pw.bench.acceptance import (
    MODE_RESEARCH,
    MODE_STRICT,
    PRIORITY1_FAIL as _P1_FAIL,
    PRIORITY1_PASS as _P1_PASS,
    PRIORITY1_PASS_WITHIN_VARIANCE as _P1_VARIANCE,
    AcceptanceReport,
    PaperResult,
)
from t2pw.bench.metrics import (
    BLOCKER_SCOPES,
    BLOCKER_SCOPE_LABELS,
    BOUNDARY_LABELS,
    BOUNDARY_NONE,
    DENOMINATOR_ORDER,
    RESEARCH_FAILURE_LABELS,
)
from t2pw.bench.semantic import CHECK_ORDER, ERROR_ORDER
from t2pw.pipeline.release_status import (
    NOT_RECORDED,
    NOT_RELEASE_READY_NOTE,
    describe as _describe_release_status,
)

_WIDTH = 100
_RULE = "=" * _WIDTH
_THIN = "-" * _WIDTH


def _header(title: str) -> List[str]:
    return [_RULE, title, _RULE]


def _label(name: str) -> str:
    return name.replace("_", " ")


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _priorities(report: AcceptanceReport) -> List[str]:
    out = _header("ACCEPTANCE PRIORITIES -- in the declared order")
    out.append("Priorities 1-3 are absolute: any non-zero count fails them however good the rest looks.")
    out.append("NOT EVAL means the run never gathered the evidence. It is not a pass and not a failure;")
    out.append("an absolute priority that was never measured has not been met, only left unproven.")
    out.append("")
    for entry in report.priorities():
        # Three states, not two. A priority the run never gathered evidence for is
        # printed as NOT EVALUATED: rendering it PASS reports an instrument that
        # read nothing as a clean result, and rendering it FAIL invents a
        # violation nobody measured.
        status = "NOT EVAL" if entry.get("evaluated") is False else _mark(bool(entry["ok"]))
        # The rank-1 badge is the ABSOLUTE zero-tolerance answer on the RAW count,
        # which is not D-073's status and never was: at accepted 6 -- an
        # unambiguous PASS -- `ok` is already False. Printing a bare [FAIL] above
        # `accepted: 7 -> PASS_WITHIN_VARIANCE` reads as self-contradiction to the
        # human making the T-107 call, so the badge says which question it answers
        # and points at the other one. Relabelled, NOT rewired: widening `ok` would
        # turn an absolute gate permissive at six and move two pinned baselines.
        if entry["rank"] == 1 and "accepted_status" in entry and entry.get("evaluated") is not False:
            status = f"{status} raw"
        out.append(f"  {entry['rank']}. [{status}] {entry['name']}")
        out.append(f"          observed: {entry['observed']}")
        if entry["papers"] and entry["rank"] <= 3:
            out.append(f"          papers  : {', '.join(entry['papers'])}")
        if entry["rank"] == 1 and "accepted_status" in entry:
            # D-073. The status is spelled out in full because
            # PASS_WITHIN_VARIANCE is not a synonym for PASS and has to be
            # readable as itself by someone who sees only this block.
            adjusted = entry.get("contract_adjusted_rows") or []
            out.append(f"          raw     : {entry['raw']}  (preserved, unchanged in meaning;"
                       " the badge above is absolute zero-tolerance on THIS number)")
            out.append(
                f"          accepted: {entry['accepted']}  -> {entry['accepted_status']}"
                f"   (target {entry['target']}; 0-{entry['target']} {_P1_PASS}, "
                f"{entry['target'] + 1} {_P1_VARIANCE}, {entry['target'] + 2}+ {_P1_FAIL})"
            )
            out.append(
                f"          contract-adjusted out of accepted: {len(adjusted)}"
                + ("" if adjusted else "  (none, and none is POSSIBLE: D-074 licenses"
                   " only the bare sentinel, which can never be a Priority-1 row)")
            )
            for row in entry.get("raw_rows") or []:
                out.append(
                    f"            - {row['paper_id']}:{row['mode']} {row['pointer']} "
                    f"{row['name']} [{row['kind']}]"
                    + ("" if row["accepted"] else f"  EXCUSED by {row['contract_tolerance']}")
                )
            if entry["accepted_status"] == _P1_VARIANCE:
                for line in _wrap(entry.get("variance_note", ""), 88):
                    out.append(f"          ! {line}")
        # D-072. Raw beside contract-accepted on the two priorities the ruling
        # names, never merged into one number, and only where a gold-forbidden
        # term was actually drawn -- a leg with none prints nothing and reads
        # exactly as it did before. Every withheld term is named on its own row,
        # which is guard rail 3: out of the denominator is not out of the record.
        recon = entry.get("requested_core_coverage") if entry["rank"] in (4, 5) else None
        if recon and recon["legs_with_forbidden_terms"]:
            out.append(
                f"          D-072 requested-core reconciliation: "
                f"{recon['forbidden_terms_excluded']} gold-forbidden term(s) withheld from "
                f"{recon['legs_with_forbidden_terms']} of {recon['legs_with_coverage']} scored leg(s)"
            )
            # The rows live once, on the report (F5). The entry carries the
            # counts; this is the one reader that needs the detail.
            for row in report.coverage_reconciliation_corpus["legs"]:
                if not row["excluded_count"]:
                    continue
                raw = "n/a" if row["raw_ratio"] is None else f"{row['raw_ratio']:.3f}"
                acc = "UNDEFINED" if row["accepted_ratio"] is None else f"{row['accepted_ratio']:.3f}"
                out.append(
                    f"            - {row['paper_id']}:{row['mode']}"
                    f"  raw {row['raw_matched']}/{row['raw_denominator']}={raw}"
                    f"  accepted {row['accepted_matched']}/{row['accepted_denominator']}={acc}"
                    + ("  -> CLEARS the unchanged minimum" if row["cleared_by_reconciliation"] else "")
                )
                out.append(
                    "              withheld (still forbidden to export, still recorded): "
                    + ", ".join(f"{e['term']} [{e['forbidden_kind']}]" for e in row["excluded_terms"])
                )
        unmeasured = entry.get("not_evaluated_papers") or []
        if unmeasured and entry["rank"] <= 3:
            out.append(f"          NOT EVALUATED on: {', '.join(unmeasured)}")
    out.append("")
    return out


def _denominators(report: AcceptanceReport) -> List[str]:
    out = _header("SEPARATED DENOMINATORS -- one question each")
    out.append("A rate here never shares a denominator with another rate. That is the whole point:")
    out.append("a retrieval problem and an exporter problem used to move the same number.")
    out.append("")
    out.append("Note the two research rates. 'deliverable produced' is an OUTPUT rate -- the citation")
    out.append("report exists. 'semantically confirmed' is a CORRECTNESS rate. A deliverable built on")
    out.append("missing admission data, identity conflicts, unsupported reactions or orphaned")
    out.append("references is produced but not confirmed, and must never be quoted as a success.")
    out.append("")
    for key in DENOMINATOR_ORDER:
        denominator = report.denominators.get(key)
        if denominator is None:
            continue
        out.append(f"  {_label(key).upper()}")
        out.append(f"     question   : {denominator.question}")
        out.append(f"     population : {denominator.population}")
        out.append(f"     result     : {denominator.render()}")
        if denominator.numerator_names:
            out.append(f"     passing    : {', '.join(sorted(denominator.numerator_names))}")
        missing = sorted(set(denominator.denominator_names) - set(denominator.numerator_names))
        if missing:
            out.append(f"     not passing: {', '.join(missing)}")
        for excluded in denominator.excluded:
            out.append(f"     excluded   : {excluded['paper_id']} -- {excluded['reason']}")
        out.append("")
    return out


def _errors(report: AcceptanceReport) -> List[str]:
    out = _header("SCIENTIFIC ERRORS -- counted separately, never summed")
    out.append("These are different kinds of wrongness, not severities of one kind. One fabricated")
    out.append("accession is worse than forty honest Unknown-backed proteins, so they stay in columns.")
    out.append("")
    totals = report.errors.totals
    affected = report.errors.papers_affected
    width = max((len(name) for name in ERROR_ORDER), default=10)
    for name in ERROR_ORDER:
        count = totals.get(name, 0)
        papers = sorted(affected.get(name, set()))
        line = f"  {_label(name):<{width + 2}} {count:>5}"
        if papers:
            line += f"   papers: {', '.join(papers)}"
        out.append(line)
    out.append("")
    return out


def _identity(report: AcceptanceReport) -> List[str]:
    out = _header("REAL IDENTIFIERS versus UNKNOWN FALLBACKS")
    totals = report.identity.get("totals", {})
    out.append(f"  verified (real external accession) : {totals.get('verified', 0)}")
    out.append(f"  placeholder (Unknown-backed)       : {totals.get('placeholder', 0)}")
    out.append(f"  unresolved (no identity at all)    : {totals.get('unresolved', 0)}")
    out.append(f"  PathBank Unknown sentinel (id 9659): {totals.get('pathbank_unknown_sentinel', 0)}")
    out.append("")
    # D-070's split. Two populations with OPPOSITE dispositions, shown apart and
    # never summed. `other` prints even at zero: hiding the remainder would hide
    # the first row that does not fit. F-141 is a DIFFERENT seam and is never
    # reported as placeholder_backed_proteins.
    out.append("  Unknown-backed placeholders, split (D-070) -- these three partition the count above.")
    out.append("  Neither class is a gold error and none sets placeholder_claims_real_identity:")
    out.append(f"    PathBank sentinel rows, record 9659 (O-1a): {totals.get('placeholder_sentinel_rows', 0)}")
    out.append(f"    generated functional wrappers      (O-1b): {totals.get('placeholder_generated_wrappers', 0)}")
    out.append(f"    other placeholder rows                   : {totals.get('placeholder_other_rows', 0)}")
    out.append("")
    out.append("  Candidate identities NOT shipped (F-141) -- a DIFFERENT seam, never the count above:")
    if totals.get("withheld_identity_evaluated") is False:
        out.append("    NOT EVALUATED -- no leg reached the identity verdict. The absence of a")
        out.append("    measurement, not the absence of withheld identities.")
    else:
        out.append(f"    correctly withheld (species unresolved): {totals.get('withheld_identity_correct', 0)}")
        out.append(f"    lost despite sufficient support       : {totals.get('withheld_identity_recoverable', 0)}")
        out.append(f"    other measured mechanism (unclassified): {totals.get('withheld_identity_other', 0)}")
        out.append("    MEASURED counts, including when they read 0. Withholding a species-specific")
        out.append("    identifier with no species evidence is CORRECT and is not an error elsewhere.")
        out.append("    The third line is D-070 O-1c category 6: a mechanism this reader does not")
        out.append("    recognise is UNCLASSIFIED, never counted as correct withholding.")
    out.append("")
    sources = report.identity.get("payload_sources", {})
    if sources:
        rendered = ", ".join(f"{name}={count}" for name, count in sorted(sources.items()))
        out.append(f"  payload sources scored: {rendered}")
    caveat = report.identity.get("caveat")
    if caveat:
        out.append("")
        for line in _wrap(caveat, 96):
            out.append(f"  ! {line}")
    out.append("")
    return out


def _boundaries(report: AcceptanceReport) -> List[str]:
    out = _header("STRICT FAILURES BY ACTUAL BOUNDARY")
    out.append("Replaces failure_kind='contract', which reported the Stage-3 gate, the PWML")
    out.append("required-field gate and a structural guard as the same thing.")
    out.append("")
    if not report.strict_boundaries:
        out.append("  (no strict failures)")
    for boundary, count in report.strict_boundaries.items():
        out.append(f"  {count:>3}  {boundary}  --  {BOUNDARY_LABELS.get(boundary, boundary)}")
    out.append("")

    out.extend(_header("RESEARCH-MODE FAILURES BY CORRECTED CAUSE"))
    out.append("The old doctrine was 'a research failure can only mean the code is wrong'. It is not:")
    out.append("the structural guards abort in both modes by design, and a deliberate stop is correct.")
    out.append("")
    if not report.research_failures:
        out.append("  (no research-mode failures)")
    for cause, count in report.research_failures.items():
        out.append(f"  {count:>3}  {cause}")
        out.append(f"       {RESEARCH_FAILURE_LABELS.get(cause, '')}")
    out.append("")
    return out


def _wrap(text: str, width: int) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _release(leg: Any) -> List[str]:
    """The leg's RUNTIME release classification, or an explicit "not recorded".

    ``PASS`` on the line above is a TECHNICAL gate result. Printed alone it reads
    as "released", which collapses three of the five states PRODUCT_CONTRACT 11
    requires to stay apart -- and the strict denominator has historically been
    built by equating a strict pass (or a ``pathway.pwml`` on disk) with a
    release. This line states the runtime classification instead, and says so
    plainly when the leg carries none rather than deducing one.
    """

    release = getattr(leg, "release_status", None)
    text = (
        f"{NOT_RECORDED} -- {NOT_RELEASE_READY_NOTE}"
        if release in (None, "")
        else _describe_release_status(release)
    )
    return [
        ("              release : " if index == 0 else "                        ") + line
        for index, line in enumerate(_wrap(text, 74))
    ]


def _paper_block(paper: PaperResult) -> List[str]:
    case = paper.case
    out: List[str] = [_THIN]
    tag = " [NEGATIVE CONTROL]" if case.is_negative_control else ""
    out.append(f"{case.paper_id}{tag}  --  {case.title[:74]}")
    out.append(
        f"   requested : {case.requested_pathway} | {case.requested_organism}"
        + (f"   (paper is actually: {case.actual_organism})" if case.actual_organism else "")
    )
    out.append(
        f"   gold      : relevance={case.mechanistic_relevance}  export={case.expected_export}  "
        f"min_connected={case.min_connected_reactions}  unknown_ok={case.unknown_backed_proteins_acceptable}"
    )

    for mode in (MODE_STRICT, MODE_RESEARCH):
        leg = paper.leg(mode)
        if leg is None:
            continue
        if not leg.attempted:
            out.append(f"   {mode:<9}: NOT ATTEMPTED (no manifest row)")
            continue
        head = f"   {mode:<9}: {leg.status.upper()}"
        if leg.boundary != BOUNDARY_NONE:
            head += f"  boundary={leg.boundary}"
        if leg.research_failure:
            head += f"  cause={leg.research_failure}"
        out.append(head)
        if leg.boundary_evidence:
            for line in _wrap(leg.boundary_evidence, 84):
                out.append(f"              {line}")
        out.extend(_release(leg))

        semantic = leg.semantic
        if semantic is None or not semantic.evaluated:
            reason = semantic.not_evaluated_reason if semantic else "no semantic report"
            out.append(f"              semantic: not evaluated -- {reason}")
            continue
        out.append(
            f"              payload={leg.payload_source}  reactions={semantic.graph.get('n_reactions', 0)}  "
            f"connected_core={semantic.graph.get('largest_core_size', 0)}"
        )
        support = semantic.support or {}
        if support:
            complete = support.get("signature_set_complete")
            rate_label = "precision" if complete else "attribution"
            rate = support.get("precision") if complete else support.get("attribution_rate")
            out.append(
                f"              support: {rate_label}="
                + (f"{rate * 100:.0f}%" if rate is not None else "n/a")
                + f" ({support.get('true_positives', 0)}/{support.get('retained_reactions', 0)} reactions)"
                + "  recall="
                + (
                    f"{support['recall'] * 100:.0f}%"
                    if support.get("recall") is not None
                    else "n/a"
                )
                + f" ({support.get('signatures_matched', 0)}/{support.get('signatures_quote_verified', 0)} paper-stated reactions)"
            )
            if not complete:
                out.append(
                    "                       (signature set is a SUBSET, so unattributed rows are not "
                    "counted as fabrications)"
                )
        for key, label in (
            ("expected_enzymes", "enzyme recall"),
            ("expected_metabolites", "metabolite recall"),
        ):
            detail = semantic.coverage.get(key)
            if detail is None or not detail.expected:
                continue
            missing = ", ".join(m["expected"] for m in detail.missing[:5])
            line = (
                f"              {label}: {detail.found}/{detail.expected} "
                f"({detail.fraction * 100:.0f}%)"
            )
            if missing:
                line += f"   missing: {missing}"
            out.append(line)
        for name in CHECK_ORDER:
            check = semantic.checks.get(name)
            if check is None:
                continue
            if check.ok and check.applicable:
                flag = "ok  "
            elif not check.applicable:
                flag = "n/e "
            else:
                flag = "FAIL"
            out.append(f"              [{flag}] {name}: {check.summary}")
        errors = {k: v for k, v in semantic.scientific_errors.items() if v}
        if errors:
            rendered = ", ".join(f"{_label(k)}={v}" for k, v in errors.items())
            out.append(f"              errors: {rendered}")
    return out


def _papers(report: AcceptanceReport) -> List[str]:
    out = _header("RESULTS PER PAPER")
    for paper in report.papers:
        out.extend(_paper_block(paper))
    out.append("")
    return out


def _completion(report: AcceptanceReport) -> List[str]:
    out = _header("BENCHMARK COMPLETION -- reported separately from every rate")
    out.append("An unattempted or unscorable paper is missing coverage, NOT a pipeline failure.")
    out.append("")
    data = report.completion()
    out.append(f"  planned gold cases          : {data['planned_gold_cases']}")
    out.append(f"  papers attempted            : {data['papers_attempted']}")
    out.append(f"  strict legs attempted       : {data['strict_legs_attempted']}")
    out.append(f"  research legs attempted     : {data['research_legs_attempted']}")
    out.append(f"  legs attempted              : {data['legs_attempted']} of {data['legs_planned']}")
    out.append(f"  payloads available          : {data['payloads_available']}")
    out.append(f"  semantically scorable legs  : {data['semantically_scorable_legs']}")
    out.append(f"  fully completed cases       : {data['fully_completed_cases']}")
    out.append("")
    verdict = "COMPLETE" if data["complete"] else "PARTIAL -- not a quotable benchmark result"
    out.append(f"  status: {verdict}   ({data['rendered']})")
    if data["papers_with_no_attempted_leg"]:
        out.append("")
        out.append("  gold cases with NO attempted leg:")
        for paper_id in data["papers_with_no_attempted_leg"]:
            out.append(f"    - {paper_id}")
    if data["papers_with_only_one_mode_attempted"]:
        out.append("  gold cases with only one mode attempted:")
        for paper_id in data["papers_with_only_one_mode_attempted"]:
            out.append(f"    - {paper_id}")
    out.append("")
    return out


def _blockers(report: AcceptanceReport) -> List[str]:
    out = _header("REMAINING BLOCKERS -- three separate rankings, never merged")
    out.append("Ranked by EXPECTED benefit (papers for which this is the only root cause), then by")
    out.append("total papers touched. 'Papers released' only means something inside one population:")
    out.append("a partial_only paper is not in the strict denominator, so fixing its strict gate")
    out.append("cannot raise the strict rate however real the work is.")
    out.append("")
    for scope in BLOCKER_SCOPES:
        ranked = report.blockers.get(scope) or []
        out.append(_THIN)
        out.append(f"{scope.upper()}")
        out.append(f"  {BLOCKER_SCOPE_LABELS.get(scope, '')}")
        out.append("")
        if not ranked:
            out.append("  (nothing is blocking in this population)")
            out.append("")
            continue
        for index, blocker in enumerate(ranked, start=1):
            out.append(f"  {index}. {blocker.root_cause}")
            out.append(f"       {blocker.explanation}")
            out.append(
                f"       expected benefit: {blocker.expected_benefit} paper(s) released outright; "
                f"touches {len(blocker.papers)} paper(s)"
            )
            if blocker.sole_blocker_for:
                out.append(f"       sole blocker for: {', '.join(sorted(blocker.sole_blocker_for))}")
            out.append(f"       papers          : {', '.join(sorted(blocker.papers))}")
            if blocker.codes:
                out.append(f"       example codes   : {', '.join(sorted(blocker.codes)[:4])}")
            out.append("")
    return out


def _notes(report: AcceptanceReport) -> List[str]:
    if not report.notes:
        return []
    out = _header("READ THIS BEFORE TRUSTING THE NUMBERS ABOVE")
    for note in report.notes:
        for index, line in enumerate(_wrap(note, 94)):
            out.append(("  ! " if index == 0 else "    ") + line)
        out.append("")
    return out


def render_text(report: AcceptanceReport) -> str:
    """The full human-readable acceptance report."""

    lines: List[str] = [
        _RULE,
        "T2PW PINNED ACCEPTANCE EVALUATION",
        _RULE,
        f"run directory : {report.run_dir}",
        f"gold set      : {report.gold_version}  ({report.gold_path})",
        f"papers        : {len(report.papers)}",
        f"legs          : {report.legs_attempted} attempted, {report.legs_scored} scored "
        f"(of {len(report.papers) * 2} possible)",
        "",
    ]
    lines.extend(_completion(report))
    lines.extend(_priorities(report))
    lines.extend(_denominators(report))
    lines.extend(_errors(report))
    lines.extend(_identity(report))
    lines.extend(_boundaries(report))
    lines.extend(_papers(report))
    lines.extend(_blockers(report))
    lines.extend(_notes(report))
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_text"]
