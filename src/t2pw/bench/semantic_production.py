"""Semantic evaluation of a PRODUCED pathway with no gold case (D-006, first half).

``bench.semantic`` can only ask "is this the requested pathway?" about a paper someone
hand-read into a ``GoldCase``. Production has none, so a run can ship as ``release_ready``
having never been semantically evaluated at all. This module asks the subset of those
questions that the *request* and the *payload* answer between them, and reports the rest
as NOT EVALUATED. It is wired to nothing: C-056a merges it into ``release_status``,
C-056b into benchmark denominators. No network, no database, no LLM on any path -- this
runs inside the pipeline's deadline.

``CheckResult`` / ``SemanticReport`` and the ``CHECK_*`` names are reused, never
re-derived, so C-056a merges one verdict shape instead of translating between two. So are
the meanings ``semantic.py:293-299`` gives ``ok`` (every *applicable* check passed) and
``confirmed`` (that, and every check was evaluable): an inapplicable check carries
``ok=True`` plus an ``inapplicable_reason``, so it is excluded from ``confirmed`` and
absent from ``failed_checks`` -- not evaluated is never a pass and never a failure.

C-056a must read THREE facts, not one. ``confirmed`` cannot be ``True`` on a production
run by construction: ``retained_reactions_match_supported_signatures`` needs
quote-verified signatures that exist only in the gold set. Combine ``evaluated``, ``ok``
and ``inapplicable_checks``. In ``scientific_errors`` a zero beside a name in
``inapplicable_checks`` means "not measured", not "clean": most keys are counted from a
partial pass, only the two gold-only keys are omitted, and ``to_dict`` (``semantic.py:324``)
re-fills all nine from ``ERROR_ORDER`` anyway -- so a consumer of the SERIALIZED map must
cross-reference ``inapplicable_checks``. Not this module's behaviour to change.
"""

from __future__ import annotations

import re
from collections.abc import Mapping as _AbcMapping
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# The ONLY t2pw.bench import: the validator whose vocabulary and helpers are reused.
# Deliberately not goldset, acceptance, metrics, render, preflight or any benchmark
# runner -- a runtime that had to import the benchmark harness to classify a run would
# have the dependency backwards. (``t2pw.bench.__init__`` imports ``goldset`` itself; that
# is a pre-existing package-level fact this card neither owns nor can change.)
from t2pw.bench import semantic as _s

CheckResult = _s.CheckResult
SemanticReport = _s.SemanticReport

#: Why a check could not run. Properties of *production*, not of any one payload.
NO_GOLD_SIGNATURES = (
    "reaction-level support can only be audited against hand-read, quote-verified "
    "signatures, which exist only in the gold set; a production run has none"
)
NO_PATHWAY_CONTEXT = "this run supplied no requested-pathway name to look for"
NO_ORGANISM_CONTEXT = "this run supplied no requested-organism context to compare against"
NO_CORE_FLOOR = "this run configured no minimum connected-core size"
NO_PAYLOAD = "the run produced no payload to evaluate"

#: Value of ``goldset.MIN_SUBSTRING_TERM``. A needle shorter than this matches by equality
#: only, so a 2-character name cannot anchor a request by containment.
_MIN_CONTAINMENT = 4


def _not_evaluated(name: str, reason: str) -> CheckResult:
    """A check that could not run. ``ok=False`` here would collapse "not evaluated" into
    "failed"; ``ok=True`` plus a reason is what keeps it out of ``confirmed`` instead."""

    return CheckResult(name=name, ok=True, summary=f"not evaluated: {reason}", inapplicable_reason=reason)


def _contains_token(haystack: str, needle: str) -> bool:
    """LOCAL RESTATEMENT of ``goldset.contains_term`` -- IMPLEMENT-5 bars importing it and
    ``semantic`` does not re-export it, so keep the two in step BY HAND. Both arguments are
    already ``normalize_name``-folded. Whole-token boundaries stop ``ala``
    (5-aminolevulinic acid) anchoring ``alanine``; the floor is measured WITHOUT spaces and
    applies to whichever string is the needle, so a 2-character request cannot anchor
    itself inside a long payload name."""

    if not haystack or not needle or len(needle.replace(" ", "")) < _MIN_CONTAINMENT:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def _mentions(term: str, vocabulary: Sequence[str]) -> str:
    """The payload name that anchors ``term``, or ``""``. Containment is tested in BOTH
    directions -- the request may name the whole pathway ("lipid A biosynthesis") while the
    payload names its core ("lipid A") -- and each direction floors its own needle."""

    needle = _s.normalize_name(term)
    if not needle:
        return ""
    for value in vocabulary:
        candidate = _s.normalize_name(value)
        if candidate == needle or _contains_token(needle, candidate) or _contains_token(candidate, needle):
            return value
    return ""


def _check_requested_pathway(requested_pathway: str, vocabulary: Sequence[str]) -> CheckResult:
    """Is the payload about the pathway that was asked for? The gold version matches
    hand-read anchor terms; production has only the request, which is not gold data -- it
    is the question the user asked."""

    if not _s.normalize_name(requested_pathway):
        return _not_evaluated(_s.CHECK_ANCHORS, NO_PATHWAY_CONTEXT)
    hit = _mentions(requested_pathway, vocabulary)
    summary = (
        f"the requested pathway '{requested_pathway}' is named in the payload (matched '{hit}')" if hit
        else f"nothing in the payload names the requested pathway '{requested_pathway}'"
    )
    findings = [] if hit else [{
        "missing_anchor": requested_pathway,
        "reason": "no entity, process or pathway name in the payload names the requested pathway",
    }]
    return CheckResult(name=_s.CHECK_ANCHORS, ok=bool(hit), summary=summary, findings=findings)


def _check_requested_organism(
    requested_organism: str, processes: Mapping[str, List[Dict[str, Any]]]
) -> Tuple[CheckResult, int]:
    """No retained reaction contradicts the organism the run asked for.

    Reactions only. A protein row legitimately carries the PathBank sentinel species, and
    grading that here would take a side in the standing ``placeholder_backed_proteins``
    disagreement (PRODUCT_CONTRACT section 13 / TRAP-3).
    """

    if not _s.normalize_name(requested_organism):
        return _not_evaluated(_s.CHECK_ORGANISM, NO_ORGANISM_CONTEXT), 0
    wanted = _s.normalize_name(requested_organism)
    findings: List[Dict[str, Any]] = []
    for bucket, rows in processes.items():
        for index, row in enumerate(rows):
            observed = _s._organism_of(row)
            norm = _s.normalize_name(observed)
            # A BLANK organism is not a violation: payloads routinely leave it empty and
            # ``pipeline.propagate_context_organism`` fills it later. Only a stated,
            # conflicting organism is a semantic defect.
            if not norm or norm == wanted or norm.startswith(wanted + " ") or wanted.startswith(norm + " "):
                continue
            findings.append({
                "pointer": f"/processes/{bucket}/{index}", "observed_organism": observed,
                "name": _s.canonical_text(row.get("name")) or "(unnamed)",
                "reason": f"attributed to '{observed}', but this run requested '{requested_organism}'",
            })
    ok = not findings
    summary = (
        f"no retained reaction contradicts the requested organism '{requested_organism}'" if ok
        else f"{len(findings)} retained reaction(s) attributed to an organism other than '{requested_organism}'"
    )
    return CheckResult(name=_s.CHECK_ORGANISM, ok=ok, summary=summary, findings=findings), len(findings)


def _audit_entities(
    entities: Mapping[str, List[Dict[str, Any]]]
) -> Tuple[CheckResult, CheckResult, Dict[str, int], int, int]:
    """One pass over the entity buckets, two verdicts.

    ``no_real_id_or_name_conflict`` asks the two forgeries answerable without gold data: a
    placeholder posing as a real mapping, and one accession claimed by two
    differently-named entities, which silently fuses two molecules in the exported graph.
    The gold-only third -- a *forbidden* identifier carrying an accession -- is not asked.

    ``placeholder_identities_distinguished`` asks only whether an Unknown-backed row
    admits what it is. Whether such a row should be exported at all is a standing policy
    disagreement (PRODUCT_CONTRACT section 13 / TRAP-3) and is deliberately NOT decided
    here: a correctly-marked placeholder passes, and the count is reported so the
    disagreement stays visible without this module adjudicating it.
    """

    from t2pw.pipeline.entity_identity import (
        IDENTITY_PLACEHOLDER, IDENTITY_UNRESOLVED, IDENTITY_VERIFIED,
        identity_status, placeholder_claims_real_identity,
    )

    census: Dict[str, int] = {
        IDENTITY_VERIFIED: 0, IDENTITY_PLACEHOLDER: 0, IDENTITY_UNRESOLVED: 0,
        "proteins_total": 0, "compounds_total": len(entities.get("compounds", [])),
    }
    conflicts: List[Dict[str, Any]] = []
    forged: List[Dict[str, Any]] = []
    holders: Dict[Tuple[str, str], List[str]] = {}
    for bucket in _s._ENTITY_BUCKETS:
        for index, row in enumerate(entities.get(bucket, [])):
            pointer = f"/entities/{bucket}/{index}"
            name = _s.canonical_text(row.get("name"))
            if bucket != "compounds":
                census["proteins_total"] += 1
                status = identity_status(row) or IDENTITY_UNRESOLVED
                census[status] = census.get(status, 0) + 1
            ids = _s._external_ids(row)
            claim = placeholder_claims_real_identity(row)
            if claim:
                conflicts.append({
                    "pointer": pointer, "name": name, "rule": claim, "identifiers": ids,
                    "kind": "placeholder_claims_real_identity",
                    "reason": f"placeholder row is posing as a real mapping ({claim})",
                })
                forged.append({
                    "pointer": pointer, "name": name, "rule": claim,
                    "kind": "placeholder_not_distinguished",
                    "reason": f"placeholder is indistinguishable from a real mapping ({claim})",
                })
            for namespace, accession in ids.items():
                holders.setdefault((namespace, accession.casefold()), []).append(name or pointer)
    for (namespace, accession), names in sorted(holders.items()):
        distinct = sorted({_s.normalize_name(n) for n in names if n})
        if len(distinct) > 1:
            conflicts.append({
                "kind": "accession_claimed_by_multiple_entities", "namespace": namespace,
                "identifier": accession, "entities": sorted(set(names)),
                "reason": f"{namespace}:{accession} is claimed by {len(distinct)} differently-named entities",
            })
    backed = int(census[IDENTITY_PLACEHOLDER])
    breakdown = (
        f"{census[IDENTITY_VERIFIED]} verified / {backed} Unknown-backed placeholder / "
        f"{census[IDENTITY_UNRESOLVED]} unresolved, of {census['proteins_total']} protein row(s)"
    )
    id_check = CheckResult(
        name=_s.CHECK_ID_CONFLICT, ok=not conflicts, findings=conflicts,
        summary="no placeholder forgery and no accession collision" if not conflicts
        else f"{len(forged)} placeholder(s) posing as a real mapping; {len(conflicts)} conflict finding(s)",
    )
    placeholder_check = CheckResult(
        name=_s.CHECK_PLACEHOLDER_IDENTITY, ok=not forged, findings=forged,
        summary=f"{breakdown}; every placeholder admits it is one" if not forged
        else f"{len(forged)} placeholder identity problem(s); {breakdown}",
    )
    return id_check, placeholder_check, census, len(forged), backed


def _check_connected_core(graph: Mapping[str, Any], minimum: Optional[int]) -> CheckResult:
    """Floor on the largest chemically connected core. The floor is a run configuration,
    never guessed: without one the check reports not evaluated rather than inventing a
    threshold."""

    if minimum is None:
        return _not_evaluated(_s.CHECK_CONNECTED_CORE, NO_CORE_FLOOR)
    largest = int(graph.get("largest_core_size", 0))
    ok = largest >= int(minimum)
    findings = [] if ok else [{
        "required": int(minimum), "observed": largest, "n_components": graph.get("n_components", 0),
        "isolated_reactions": graph.get("isolated_reactions", 0),
        "reason": "reactions do not join into one chain through shared non-cofactor metabolites",
    }]
    return CheckResult(
        name=_s.CHECK_CONNECTED_CORE, ok=ok, findings=findings,
        summary=f"largest chemically connected core is {largest} reaction(s); this run requires {minimum}",
    )


def evaluate_production_semantics(
    payload: Optional[Mapping[str, Any]],
    *,
    requested_pathway: str = "",
    requested_organism: str = "",
    paper_id: str = "",
    mode: str = "",
    admission: Optional[Mapping[str, Any]] = None,
    quarantine_report: Optional[Mapping[str, Any]] = None,
    min_connected_reactions: Optional[int] = None,
) -> SemanticReport:
    """Score one produced payload against the run's own request. No gold case.

    Deterministic and offline. ``payload`` may be ``None``: a run that died before
    producing one is reported as *not evaluated*, never as a semantic failure -- those are
    different facts and only the first is the extractor's problem.
    """

    report = SemanticReport(paper_id=paper_id, mode=mode)
    # ``Mapping``, not ``dict``, per the annotation: reporting a MappingProxyType payload
    # as "the run produced no payload" is a false reason on a REAL payload.
    if not isinstance(payload, _AbcMapping) or not payload:
        report.evaluated = False
        report.not_evaluated_reason = NO_PAYLOAD
        report.scientific_errors = {key: 0 for key in _s.ERROR_ORDER}
        return report

    entities = _s._entities(payload)
    processes = _s._processes(payload)
    anchors = _check_requested_pathway(requested_pathway, _s._payload_vocabulary(payload))
    carrier, missing_carrier = _s._check_source_carrier(processes)
    supported = _not_evaluated(_s.CHECK_SUPPORTED_REACTIONS, NO_GOLD_SIGNATURES)
    organism, cross_organism = _check_requested_organism(requested_organism, processes)
    conflicts, placeholder, census, forged, backed = _audit_entities(entities)
    reintroduction = _s._check_rag_reintroduction(processes, admission)
    report.graph = _s._connected_core(processes, _s._cofactor_names())
    connected = _check_connected_core(report.graph, min_connected_reactions)
    report.identity_census = census
    orphans = _s._orphaned_references(payload)

    for result in (anchors, carrier, supported, organism, conflicts, reintroduction, connected, placeholder):
        report.checks[result.name] = result
    report.scientific_errors = {
        _s.ERR_FALSE_REAL_IDENTIFIERS: forged,
        _s.ERR_PLACEHOLDER_BACKED_PROTEINS: backed,
        _s.ERR_MISSING_SOURCE_CARRIER: missing_carrier,
        _s.ERR_CROSS_ORGANISM_REACTIONS: cross_organism,
        _s.ERR_ORPHANED_REFERENCES: len(orphans),
        _s.ERR_MISSING_PATHWAY_ANCHORS: len(anchors.findings),
        _s.ERR_QUARANTINED_PROCESSES: _s._quarantined_count(payload, quarantine_report),
    }
    report.graph["orphaned_reference_examples"] = orphans[:20]
    return report
