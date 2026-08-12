"""Validate ONE RAG round's graph delta against the already-established policy.

Given the payload BEFORE a round, the payload AFTER it and the round's own record, answer
one question deterministically: *is this delta admissible, and if not, exactly which rule
did which added element break?*

VALIDATION ONLY. No loop and no controller (C-043 writes those), no repair, no network,
no LLM, no database, no clock, no randomness -- so a controller drives it and a test
runs it without RAG. Both graphs and the record are READ-ONLY. It neither conforms nor
merges: :mod:`t2pw.rag.conform` already builds the ``{"additions": ...}`` envelope that
``pipeline.merge_additions`` injects. Claim identity is
:func:`t2pw.rag.loop_policy.claim_identity_key` -- the subsystem's ONE notion -- and
provenance is read through :mod:`t2pw.pipeline.lineage`. Every rule below is quoted from
the document that fixes it; none is chosen here.

* ``gap_attribution`` / ``gap_not_detected`` -- §10, a reaction enters "only when it fills
  a *specific detected typed gap*"; an EMPTY detected set refuses every addition.
* ``unclassified`` / ``admission_verdict`` -- §10 "and passes pathway, organism and
  evidence admission": the ``scope_membership`` the gate writes and the payload carries.
* ``rejected_claim`` / ``duplicate_claim`` -- §10 "a rejected RAG claim must not be
  reintroduced by a later stage"; "deduplication is against *all claims ever seen*".
* ``evidence_retention`` -- §10 "supporting passages and source identifiers are retained
  through final export" -- asked of the row, and against what its own claim carried.
* ``provenance`` -- §3 through C-015; lineage is ADDITIVE, so only an ADDED row needs one.
* ``removed_element`` / ``mutated_element`` -- the seam is additive by construction
  (``conform`` builds additions ``merge_additions`` INJECTS, after the incident where
  synthesis REPLACED the rich payload), so a removal or a biology change on a pre-existing
  row is reported and never repaired.
* ``round_reentry`` -- D-008's five stages; one the record cannot state reports
  ``not_evaluated``, which §11 says is NEVER ``false`` and so is never a violation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Sequence, Tuple

from t2pw.pipeline.lineage import LINEAGE_KEY, LineageError, read as read_lineage
# The registry gate's OWN name function, imported exactly as ``rag.conform`` imports it
# so "the same row" means here what the gate means by it and no second notion of entity
# identity is invented. ROW bookkeeping only -- never claim identity.
from t2pw.pipeline.process_normalizer import _normalize as _row_name_norm
from t2pw.rag.admission import SCOPE_ADMITTED
from t2pw.rag.loop_policy import SeenClaims, claim_identity_key

#: Every rule, in report order -- also the primary sort key of the violations. The named
#: constants are UNPACKED from it, so the two can never drift out of the order the sort
#: key (``RULES.index``) depends on.
RULES: Tuple[str, ...] = (
    "removed_element", "mutated_element", "gap_attribution", "gap_not_detected",
    "unclassified", "admission_verdict", "rejected_claim", "duplicate_claim",
    "evidence_retention", "provenance", "round_reentry")
(RULE_REMOVED_ELEMENT, RULE_MUTATED_ELEMENT, RULE_GAP_ATTRIBUTION, RULE_GAP_NOT_DETECTED,
 RULE_UNCLASSIFIED, RULE_ADMISSION_VERDICT, RULE_REJECTED_CLAIM, RULE_DUPLICATE_CLAIM,
 RULE_EVIDENCE_RETENTION, RULE_PROVENANCE, RULE_ROUND_REENTRY) = RULES

#: The five stages D-008 requires EVERY round to re-enter.
REENTRY_STAGES: Tuple[str, ...] = (
    "normalization", "mapping", "gates", "persistence", "classification")

#: Three-valued, never two: ``not_evaluated`` is NEVER ``false`` (§11).
TRUE, FALSE, NOT_EVALUATED = "true", "false", "not_evaluated"

#: The row-bearing payload sections -- exactly the surface ``conform`` writes.
_ENTITIES, _PROCESSES = "entities", "processes"
_REACTIONS = f"{_PROCESSES}.reactions"

#: Fields whose change on a pre-existing row is NOT a biology mutation. Everything else
#: is compared, so a biology field the payload gains later is protected by default -- the
#: opposite direction from an allowlist, which would silently stop protecting it.
NON_BIOLOGICAL: FrozenSet[str] = frozenset({
    LINEAGE_KEY,                                             # §6: lineage is not biology
    "rag_provenance", "evidence", "source_papers", "rag_confidence",  # RAG_ADDITIVE_KEYS
    "source_refs", "provenance", "confidence",         # provenance pointers and scores
    "mapping_meta", "db_status", "raw_name",           # mapping bookkeeping, not an id
    "review_flags", "preservation_status"})            # audit annotations


@dataclass(frozen=True)
class Violation:
    """One rule, the element that broke it, and observed vs required."""

    rule: str
    element: str        # qualified row id, e.g. ``processes.reactions[deacetylation]``
    observed: str
    required: str
    detail: str = ""    # the field, stage or claim key the rule fired on


@dataclass(frozen=True)
class RoundRecord:
    """What the round itself reported -- the only input that is not a graph.

    ``detected_gap_ids`` is its detected-gap set; empty refuses every addition. ``claims``
    are its claims (mappings or ``RagReactionCandidate``-shaped objects), read to detect
    evidence LOST between claim and payload row. ``seen_claims`` is the history as of the
    START of the round, so this round's own claims are not already in it.
    ``rejected_claim_keys`` are :func:`claim_identity_key` keys any round rejected.
    ``reentered`` maps a D-008 stage to ``True``/``False``; one the record cannot state is
    absent, and reports ``not_evaluated``.
    """

    detected_gap_ids: FrozenSet[str] = frozenset()
    claims: Sequence[Any] = ()
    seen_claims: SeenClaims = SeenClaims()
    rejected_claim_keys: FrozenSet[str] = frozenset()
    reentered: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeltaVerdict:
    """Admissible or not, one entry per violation, and the D-008 re-entry record."""

    admissible: bool
    violations: Tuple[Violation, ...] = ()
    reentry: Mapping[str, str] = field(default_factory=dict)
    added: Tuple[str, ...] = ()
    removed: Tuple[str, ...] = ()

    def rules(self) -> Tuple[str, ...]:
        """Every rule broken, deduplicated, in :data:`RULES` order. ``dataclasses.asdict``
        serializes the whole verdict, so no bespoke JSON form is defined here."""
        broken = {v.rule for v in self.violations}
        return tuple(rule for rule in RULES if rule in broken)


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` off a mapping OR an object, as ``claim_identity_key`` itself does."""
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _strs(value: Any) -> List[str]:
    """``value`` as non-empty stripped strings; a scalar string is ONE item, not chars."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [v.strip() for v in value if isinstance(v, str) and v.strip()]
    return []


def _maps(value: Any) -> List[Mapping[str, Any]]:
    """``value`` as a list of mappings; a lone mapping (``source_paper``) is one item."""
    if isinstance(value, Mapping):
        return [value]
    return [v for v in value if isinstance(v, Mapping)] if isinstance(value, list) else []


def _sections(graph: Any) -> Dict[str, List[Mapping[str, Any]]]:
    """``{"<section>.<bucket>": [row, ...]}`` for every row bucket of ``graph``. Generic
    over the buckets rather than naming ``reactions`` and ``compounds``: a bucket this
    walker did not enumerate is one whose removals would go unreported. Sorted, so
    traversal never depends on dict order."""
    top = graph if isinstance(graph, Mapping) else {}
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for section in (_ENTITIES, _PROCESSES):
        buckets = top.get(section)
        for bucket in sorted(buckets if isinstance(buckets, Mapping) else ()):
            rows = buckets.get(bucket)
            if isinstance(rows, list):
                out[f"{section}.{bucket}"] = [r for r in rows if isinstance(r, Mapping)]
    return out


def _index(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    """Rows by a STABLE row identity -- never a claim identity, which is chemistry-derived
    and would read a mutated participant list as one removal plus one addition, so the
    mutation rule could never fire. Duplicates are kept: a removal cannot hide behind one."""
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        locked = str(row.get("locked_reaction_id") or "").strip()
        out.setdefault(locked or _row_name_norm(str(row.get("name") or "")), []).append(row)
    return out


def _canon(value: Any) -> Any:
    """Order-free canonical form: a reordered participant list is not a biology change."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _canon(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(sorted((_canon(v) for v in value), key=repr))
    return value


def _biology(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: _canon(v) for k, v in row.items() if k not in NON_BIOLOGICAL}


def _evidence_facts(obj: Any) -> Tuple[str, str, FrozenSet[str], str]:
    """``(passage text, chunk pointer, source identifiers, exact span)``, read from every
    shape the seam produces -- a synthesized row's ``evidence`` LIST of ``RagEvidence``
    records, the single ``evidence`` STRING ``conform`` coerces it to, the
    ``rag_provenance`` chunk pointer, ``source_papers`` / ``source_paper``, ``source_refs``
    -- so a row is never reported as having lost what it merely restated."""
    prov = _get(obj, "rag_provenance")
    prov = prov if isinstance(prov, Mapping) else {}
    evidence = _get(obj, "evidence")
    texts = _strs(evidence) + [t for m in _maps(evidence) for t in _strs(m.get("text"))]
    ids = set(_strs(prov.get("source_id"))) | set(_strs(_get(obj, "source_refs")))
    for paper in _maps(_get(obj, "source_papers")) + _maps(_get(obj, "source_paper")):
        ids |= set(_strs(paper.get("source_id")))
    return (" ".join(texts), " ".join(_strs(prov.get("chunk_id"))), frozenset(ids),
            " ".join(_strs(_get(obj, "evidence_span"))))


def _gap_ids(row: Mapping[str, Any]) -> FrozenSet[str]:
    """Every gap the row claims to fill: ``rag_provenance.gap_id`` plus the complete
    ``gap_ids`` beside it (``gap_id`` alone is lossy when one canonical claim was
    retrieved for two gaps -- ``provenance.RagProvenance``)."""
    prov = row.get("rag_provenance")
    prov = prov if isinstance(prov, Mapping) else {}
    return frozenset(_strs(prov.get("gap_id")) + _strs(prov.get("gap_ids")))


def _provenance_violations(element: str, row: Mapping[str, Any]) -> List[Violation]:
    """§3 through C-015, asked of ADDED rows only -- lineage is additive."""
    try:
        entries = read_lineage(row).entries
    except LineageError as exc:  # never leaked to the caller: reported as the violation
        return [Violation(RULE_PROVENANCE, element, f"malformed lineage: {exc}",
                          "a lineage whose every entry validates", LINEAGE_KEY)]
    if not entries:
        return [Violation(RULE_PROVENANCE, element, f"no {LINEAGE_KEY}",
                          "an entry naming the stage that introduced it", LINEAGE_KEY)]
    return []


def _evidence_violations(element: str, row: Mapping[str, Any], claim: Any) -> List[Violation]:
    """§10's retention rule, on the row AND against the claim the row came from."""
    text, pointer, ids, span = _evidence_facts(row)
    out: List[Violation] = []
    if not (text or pointer or span):
        out.append(Violation(RULE_EVIDENCE_RETENTION, element, "no supporting passage",
                             "the supporting passage retained", "evidence"))
    if not ids:
        out.append(Violation(RULE_EVIDENCE_RETENTION, element, "no source identifier",
                             "the source identifiers retained", "source_paper"))
    if claim is None:
        return out
    _, _, claim_ids, claim_span = _evidence_facts(claim)
    lost = sorted(claim_ids - ids)
    if lost:
        out.append(Violation(RULE_EVIDENCE_RETENTION, element,
                             "lost source identifier(s) " + ", ".join(lost),
                             "every source identifier its claim carried", "source_paper"))
    if claim_span and claim_span not in f"{span} {text}":
        out.append(Violation(RULE_EVIDENCE_RETENTION, element, "lost the evidence_span",
                             "the claim's evidence_span retained", "evidence_span"))
    return out


def _reaction_violations(element: str, row: Mapping[str, Any],
                         record: RoundRecord) -> List[Violation]:
    """§10's gap-attribution, admission-verdict and dedupe rules -- reactions only. An
    unattributed addition is the broad retrieval ``retrieve.GapContractError`` forbids."""
    out: List[Violation] = []
    gaps = _gap_ids(row)
    unknown = sorted(gaps - frozenset(record.detected_gap_ids))
    if not gaps:
        out.append(Violation(RULE_GAP_ATTRIBUTION, element, "no gap_id",
                             "a gap_id naming the detected gap it fills"))
    elif unknown:
        out.append(Violation(RULE_GAP_NOT_DETECTED, element, ", ".join(unknown),
                             "a gap_id in the round's detected-gap set"))
    scope = " ".join(_strs(row.get("scope_membership")))
    if not scope:
        out.append(Violation(RULE_UNCLASSIFIED, element, "no scope_membership",
                             f"{SCOPE_ADMITTED!r}, written by the admission gate"))
    elif scope != SCOPE_ADMITTED:
        out.append(Violation(RULE_ADMISSION_VERDICT, element, scope, SCOPE_ADMITTED,
                             "scope_membership"))
    key = claim_identity_key(row)
    # A rejected claim is also a seen claim; the more specific rule is reported alone.
    if key in record.rejected_claim_keys:
        out.append(Violation(RULE_REJECTED_CLAIM, element, "the claim was rejected",
                             "a claim no round has rejected", key))
    elif key in record.seen_claims.keys:
        out.append(Violation(RULE_DUPLICATE_CLAIM, element, "the claim was already seen",
                             "a claim novel against every claim ever seen", key))
    return out


def _reentry(record: RoundRecord, stage: str) -> str:
    """``true`` / ``false`` / ``not_evaluated``. What the record cannot state is
    ``not_evaluated``, which §11 forbids collapsing into ``false``."""
    value = _get(record.reentered, stage) if record.reentered is not None else None
    return TRUE if value is True else FALSE if value is False else NOT_EVALUATED


def validate_graph_delta(graph_before: Any, graph_after: Any,
                         record: Any = None) -> DeltaVerdict:
    """Is the delta from ``graph_before`` to ``graph_after`` admissible under §10/§3/D-008?

    Read-only and deterministic: no input is mutated, traversal is sorted throughout and
    violations are ordered by :data:`RULES` then element, so two runs on the same inputs
    are identical. Nothing is repaired -- a real delta that breaks a rule is a finding for
    the caller, never a rule this validator softens.
    """
    record = record if isinstance(record, RoundRecord) else RoundRecord()
    claims: Dict[str, Any] = {}
    for claim in record.claims or ():
        claims.setdefault(claim_identity_key(claim), claim)
    before, after = _sections(graph_before), _sections(graph_after)
    violations: List[Violation] = []
    added: List[str] = []
    removed: List[str] = []
    for section in sorted(set(before) | set(after)):
        old, new = _index(before.get(section, [])), _index(after.get(section, []))
        for key in sorted(set(old) | set(new)):
            olds, news = old.get(key, []), new.get(key, [])
            element = f"{section}[{key}]"
            for _ in range(max(0, len(olds) - len(news))):
                removed.append(element)
                violations.append(Violation(
                    RULE_REMOVED_ELEMENT, element, "absent after the round",
                    "every pre-existing element preserved: the delta is additive"))
            for pre, post in zip(olds, news):
                was, now = _biology(pre), _biology(post)
                violations += [
                    Violation(RULE_MUTATED_ELEMENT, element, _show(now.get(f)),
                              _show(was.get(f)), f)
                    for f in sorted(set(was) | set(now)) if was.get(f) != now.get(f)]
            for row in news[len(olds):]:
                added.append(element)
                violations += _provenance_violations(element, row)
                violations += _evidence_violations(
                    element, row, claims.get(claim_identity_key(row)))
                if section == _REACTIONS:  # only a RETRIEVED REACTION is gap-bound (§10)
                    violations += _reaction_violations(element, row, record)
    reentry = {stage: _reentry(record, stage) for stage in REENTRY_STAGES}
    violations += [Violation(RULE_ROUND_REENTRY, f"round.{stage}", "not re-entered",
                             "re-entered by every round (D-008)", stage)
                   for stage, state in reentry.items() if state == FALSE]
    ordered = tuple(sorted(violations, key=lambda v: (
        RULES.index(v.rule), v.element, v.detail, v.observed)))
    return DeltaVerdict(not ordered, ordered, reentry, tuple(added), tuple(removed))


def _show(value: Any) -> str:
    return "absent" if value is None else repr(value)


__all__ = ["DeltaVerdict", "RoundRecord", "Violation", "validate_graph_delta", "RULES",
           "REENTRY_STAGES", "NON_BIOLOGICAL", "TRUE", "FALSE", "NOT_EVALUATED"]
