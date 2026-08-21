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
NO_ACTOR_SPANS = (
    "no actor sub-entry on any retained reaction carries both a name with an identifying "
    "token and a cited evidence span, so no actor could be compared against its own span"
)

#: Value of ``goldset.MIN_SUBSTRING_TERM``. A needle shorter than this matches by equality
#: only, so a 2-character name cannot anchor a request by containment.
_MIN_CONTAINMENT = 4

#: Sub-entries of a retained reaction that name an ACTOR -- a protein the row asserts is
#: doing the chemistry. ``inputs``/``outputs`` are participants, not actors, and their
#: chemistry is :data:`_s.CHECK_SUPPORTED_REACTIONS`'s gold-only question.
_ACTOR_KEYS: Tuple[str, ...] = ("enzymes", "modifiers", "transporters")

#: Where an actor cites its own span. Measured over the 21 committed ``final_mapped.json``
#: legs: of 221 actor rows, 220 carry a NON-EMPTY ``evidence`` and 19 a non-empty
#: ``source_refs`` (20 carry the key). Both are read and their union is searched, so
#: citing the right sentence in either is enough. The 221st -- ``transports[1]``
#: ``.transporters[1]`` of ``runs_verify/2026-08-04_1358/.../PMC12096016/research``,
#: carrying ``evidence: ""`` beside ``source_refs: [""]`` -- is the ONE not-examined row
#: corpus-wide, and that is the third disposition working rather than a gap: an actor
#: that cites nothing is neither corroborated nor refuted.
_ACTOR_SPAN_KEYS: Tuple[str, ...] = ("source_refs", "evidence")

#: A normalized token shorter than this, or a bare numeral, cannot IDENTIFY a protein:
#: ``normalize_name`` splits ``2,3-DHB`` into ``2 3 dhb``, and without the floor a numeral
#: in a name would be corroborated by any numeral in a span. Deliberately 3 and NOT
#: :data:`_MIN_CONTAINMENT`'s 4 -- three-character symbols are real identifiers (``Fur``
#: is one, flagged in the very leg F-079 was found on), and 4 would silently make every
#: one undecidable, disarming this check on the names most likely to be fabricated.
_MIN_IDENTIFYING_TOKEN = 3


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
    # FUNCTION-LOCAL, and it must STAY function-local (D-039 section 1). ``test_e``
    # in ``tests/test_semantic_production_no_gold.py`` pins this module's import
    # graph as ``set(sys.modules) - package``, i.e. EVERY module including stdlib,
    # so a module-level ``rag`` import would drag ``math`` and others into a pinned
    # exact set. Deferring it to call time keeps that set byte-identical -- the
    # same reason ``_audit_entities`` defers ``t2pw.pipeline.entity_identity``.
    # The PUBLIC wrapper only: ``rag.eligibility``'s private names are not imported.
    from t2pw.rag.admission import ORGANISM_MATCH, compare_organism

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
            # The disjunction above is literal string prefixing, so an ABBREVIATED
            # BINOMIAL is a false positive: ``E. coli`` shares no prefix with
            # ``escherichia coli`` and was reported as a cross-organism reaction.
            # ``rag.admission.compare_organism`` is the established taxonomy
            # wrapper -- consulting it is what stops this check and the paper-level
            # eligibility screen disagreeing about whether two names are one
            # organism, and A0-C3 forbids a competing synonym table of our own.
            #
            # STRICTLY MONOTONE: an extra ``continue`` can only REMOVE a finding,
            # never add one. Only an exact ``match`` clears; ``genus_level`` does
            # not, so ``Escherichia fergusonii`` STAYS a finding, while bare genus
            # ``Escherichia`` stays tolerated by ``wanted.startswith`` above.
            # Both pre-existing behaviours are unchanged (D-039 section 2).
            if compare_organism(requested_organism, observed) == ORGANISM_MATCH:
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


def _cited_spans(actor: Mapping[str, Any]) -> List[str]:
    """The spans an actor row cites AS ITS OWN evidence -- never the reaction's. A
    non-string reference contributes nothing rather than being coerced: guessing which key
    of an unseen mapping shape holds the quote would invent the evidence being read."""

    out: List[str] = []
    for key in _ACTOR_SPAN_KEYS:
        value = actor.get(key)
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, (list, tuple)):
            out.extend(item for item in value if isinstance(item, str))
    return [text for text in out if _s.normalize_name(text)]


def _identifying_tokens(name: str) -> List[str]:
    """The tokens of ``name`` that can identify a protein, in ``normalize_name`` form."""

    return [
        token for token in _s.normalize_name(name).split(" ")
        if len(token) >= _MIN_IDENTIFYING_TOKEN and not token.isdigit()
    ]


def _actor_named_in_span(name: str, spans: Sequence[str]) -> Optional[bool]:
    """THE MATCHING RULE. ``None`` means the row could not be examined at all.

    **An actor is corroborated when its name and its own cited span share at least one
    identifying token, compared on WHOLE-TOKEN boundaries after ``normalize_name``.**

    *Whole tokens, not substrings* -- the half that catches F-079. ``EntE`` is a substring
    of ``enterobactin``, so a raw ``in`` test reports the fabricated transporter
    (``entity: "EntE"`` over *"Enterobactin is produced ... by a TolC-dependent
    process"*) as fully corroborated. Measured over the 21 committed legs a substring test
    accepts **20 of 221** actor rows this rejects, every one an ``Ent*`` symbol swallowed
    by the word ``enterobactin``.

    *One SHARED token, not the whole name* -- the half that stops it over-firing. A bare
    symbol (``EntE``, ``EntA``, ``ALAS2``: the fabrication class) has exactly one
    identifying token, so for those the rule IS "the span must name it". Longer names get
    the looser test because the payload's canonical names are not the paper's words:
    ``Serine hydroxymethyltransferase, mitochondrial`` cites *"catalyzed by serine
    hydroxymethyltransferase"*, ``ALAS2 complex`` cites *"ALAS2 mediates ..."*,
    ``oxidoreductase (entA)`` cites *"EntA (2,3-dihydro-...)"* -- the same protein under a
    DB or wrapper name, and a whole-name rule demotes five of the 21 legs over that.

    So the predicate is deliberately WEAKER than "the span names the actor": it fires only
    on lexical disjointness. It under-reports, which is the correct direction for a check
    joining a CLOSED gating set -- a false positive demotes correct output, a false
    negative only leaves pre-existing behaviour in place.
    """

    wanted = set(_identifying_tokens(name))
    if not wanted:
        return None
    seen = {token for span in spans for token in _s.normalize_name(span).split(" ") if token}
    if not seen:
        return None
    return bool(wanted & seen)


def _check_actor_evidence(processes: Mapping[str, List[Dict[str, Any]]]) -> CheckResult:
    """Does every actor a retained reaction names appear in the span that row cites?

    Three dispositions, and the third is the one that matters:

    * span names the actor -> examined, passes.
    * span names something else -> examined, FINDING.
    * **no cited span, or no identifying token in the name -> NOT EXAMINED**, and counted.
      Not a failure: whether a row carries a citation *at all* is
      :data:`_s.CHECK_SOURCE_CARRIER`'s question, and that check documents itself as
      hygiene and is deliberately non-gating, so failing a row here for a missing span
      would smuggle it into the gate against its own stated meaning. Not a pass either:
      an actor able to evade the gate by DELETING its evidence makes the gate reward less
      evidence.

    When nothing was examinable the check reports NOT EVALUATED rather than ``ok`` -- the
    anti-vacuity property, since a detector that stopped looking and one that looked and
    found nothing agree on ``ok``, and this module's rule (``semantic.py:293-299``) is
    that not evaluated is never a pass. The census is in the summary for the same reason.
    """

    findings: List[Dict[str, Any]] = []
    examined = 0
    not_examined = 0
    for bucket, rows in processes.items():
        for index, row in enumerate(rows):
            reaction = _s.canonical_text(row.get("name")) or "(unnamed)"
            for key in _ACTOR_KEYS:
                actors = row.get(key)
                if not isinstance(actors, (list, tuple)):
                    continue
                for position, actor in enumerate(actors):
                    if not isinstance(actor, _AbcMapping):
                        # A bare string actor carries no citation of its own, so there is
                        # no span to compare it against. Same disposition as a missing one.
                        not_examined += 1
                        continue
                    name = (
                        _s.canonical_text(actor.get("entity"))
                        or _s.canonical_text(actor.get("protein"))
                        or _s.canonical_text(actor.get("name"))
                    )
                    spans = _cited_spans(actor)
                    verdict = _actor_named_in_span(name, spans)
                    if verdict is None:
                        not_examined += 1
                        continue
                    examined += 1
                    if verdict:
                        continue
                    findings.append({
                        "pointer": f"/processes/{bucket}/{index}/{key}/{position}",
                        "entity": name,
                        "name": reaction,
                        "role": _s.canonical_text(actor.get("role")),
                        "cited_span": spans[0][:300],
                        "reason": (
                            f"'{name}' is named as an actor of '{reaction}', but the span "
                            "this row cites as its own evidence does not name it"
                        ),
                    })
    if not examined:
        return _not_evaluated(_s.CHECK_ACTOR_EVIDENCE, NO_ACTOR_SPANS)
    census = f"{examined} examined, {not_examined} carried no comparable name or span"
    ok = not findings
    return CheckResult(
        name=_s.CHECK_ACTOR_EVIDENCE, ok=ok, findings=findings,
        summary=(
            f"every examined actor is named in the span it cites ({census})" if ok
            else f"{len(findings)} actor(s) cite a span that names a different protein ({census})"
        ),
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
    actor_evidence = _check_actor_evidence(processes)
    orphans = _s._orphaned_references(payload)

    for result in (anchors, carrier, supported, organism, conflicts, reintroduction, connected,
                   placeholder, actor_evidence):
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
