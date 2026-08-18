"""Transactional pre-export quarantine and graph closure for strict PWML export.

Strict export is all-or-nothing today: one hallucinated peripheral reaction whose
participant never got declared fails ``validate_required_pwml_contract`` for the
whole pathway, and the nine good reactions beside it are never written. The
recovery that keeps being reached for -- delete the offending *participant* and
re-run -- is worse than the failure: it turns a reaction the paper described into
a different reaction nobody wrote down, and does it silently.

This module implements the other recovery. Every process is admitted or
quarantined *as a unit*, the surviving graph is closed to a fixpoint, and what was
dropped is written out with the original row and the reason attached. The strict
gates downstream are untouched -- they run on a smaller payload, not a laxer one.

Four rules make this safe to put in front of the gates:

1. **The unit of quarantine is the process, never the participant.** If an
   essential participant cannot be represented, the whole process leaves. See
   :func:`_essential_participant_verdict`.
2. **Closure only removes what nothing references.** Entity removal is driven by
   the surviving processes, so it can never strand a reference it did not
   already have. The loop repeats because one round's removals change what the
   next round considers reachable -- a complex that loses its last referencing
   process is dropped, and the subunits it was keeping alive go with it. It ends
   at a fixpoint, and :func:`_revalidate_surviving_processes` is the backstop
   that catches any future removal which stops being reference-driven.

   Note what this does NOT promise: a payload with nothing *quarantined* is not
   necessarily unchanged. Closure still drops unreferenced compounds, degree-zero
   proteins and nameless rows, because each of those is a strict-gate failure on
   its own. What holds is that it never removes a process both gate stacks would
   have accepted, and never edits a surviving one. Everything it does remove is
   in ``removed_entity_report.json``.
3. **A smaller graph still has to be the requested pathway.** An empty graph, or
   one whose survivors are unrelated to what was asked for, fails
   :func:`evaluate_core_coverage` rather than passing because the invalid
   material is gone. Removing everything is not a way to succeed.
4. **Nothing is deleted without a record.** :func:`write_quarantine_artifacts`
   persists the quarantine decisions, the removed entities, every closure
   iteration, and the coverage summary.

Correctly formed Unknown-backed functional complexes are explicitly valid: a
complex whose component is the PathBank ``Unknown`` sentinel is the sanctioned
representation for an enzyme the databases do not carry, and quarantining it for
"no real UniProt ID" would delete the exact biology the fallback exists to keep.
See :func:`_entity_representability` and ``tests/test_pathbank_unknown_fallback.py``.

This module is core-side and imports nothing from the RAG package, per
``docs/rag/03_separation_invariant.md``.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from t2pw.pipeline.entity_identity import (
    has_protein_external_identity,
    is_generated_complex_wrapper,
    is_pathbank_unknown_protein,
    placeholder_claims_real_identity,
    protein_external_identity,
    protein_species_context,
)
from t2pw.pipeline.export_mode import DEFAULT_EXPORT_MODE, ExportMode, coerce_mode, is_research
from t2pw.pipeline.extraction_diagnostics import payload_hash
from t2pw.pipeline.failure_detail import row_digest, scrub_detail

# Imported rather than reimplemented: registry membership here must agree
# character for character with process_normalizer.validate_registry_references,
# or this module would quarantine a process the gate would have accepted (or,
# worse, admit one it rejects). Its _normalize keeps ':' and '+', which
# entity_identity._normalize strips -- a complex named "A:B" resolves under one
# and not the other.
from t2pw.pipeline.process_normalizer import (  # noqa: E402
    _actor_name_from_row as _actor_name,
    _canonical,
    _entity_name_norms,
    _normalize,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CORE_ACCEPTED",
    "AUXILIARY_ACCEPTED",
    "QUARANTINED_UNMAPPED_ENTITY",
    "QUARANTINED_OUT_OF_SCOPE",
    "QUARANTINED_WEAK_EVIDENCE",
    "QUARANTINED_BROKEN_REFERENCE",
    "QUARANTINED_DISCONNECTED",
    "ADMISSION_STATES",
    "ACCEPTED_STATES",
    "QUARANTINE_STATES",
    "StrictQuarantineResult",
    "StrictQuarantineInvariantError",
    "QUARANTINE_REPORT_FILENAME",
    "REMOVED_ENTITY_REPORT_FILENAME",
    "CLOSURE_REPORT_FILENAME",
    "COVERAGE_REPORT_FILENAME",
    "QUARANTINE_HISTORY_DIRNAME",
    "clear_quarantine_artifacts",
    "DEFAULT_MAX_CLOSURE_ITERATIONS",
    "DEFAULT_MIN_CORE_PROCESSES",
    "DEFAULT_MIN_CORE_COVERAGE",
    "RCT_NOT_EXPORTABLE_REASON",
    "RESEARCH_DIAGNOSTIC_NOTE",
    "quarantine_and_close",
    "evaluate_core_coverage",
    "collect_requested_core_terms",
    "write_quarantine_artifacts",
    "admitted_payload_hash",
    "QUARANTINE_POLICY_VERSION",
    "canonical_decision_inputs",
    "decision_input_hash",
    "decision_identifier",
    "decision_inputs_match",
    "decision_matches",
    "quarantine_review_flags",
]


def admitted_payload_hash(payload: Mapping[str, Any]) -> str:
    """The fingerprint a quarantine decision is bound to.

    Canonical, sorted-key JSON via :func:`extraction_diagnostics.payload_hash`, so
    two payloads that differ only in key order hash the same and any content
    change at all hashes differently. Reused rather than reinvented: the
    diagnostics recorder already fingerprints payloads this way, and two hashing
    schemes in one repo is two chances to disagree about whether a payload moved.
    """

    return payload_hash(payload)


# ── Admission states ────────────────────────────────────────────────────────
#
# One vocabulary for "did this process reach the export, and if not, why". A
# reader must never have to infer the answer from which list a row landed in.

#: Survived, and is part of the pathway the user asked for.
CORE_ACCEPTED = "core_accepted"
#: Survived, but is supporting material rather than requested core biology. It
#: exports; it just cannot be what *satisfies* the minimum-core requirement.
AUXILIARY_ACCEPTED = "auxiliary_accepted"
#: An essential participant resolves to nothing declared, or to an entity that
#: strict PWML cannot represent (an unidentified protein, a malformed placeholder).
QUARANTINED_UNMAPPED_ENTITY = "quarantined_unmapped_entity"
#: The extractor gave an explicit ``scope_membership: "out_of_scope"`` verdict.
QUARANTINED_OUT_OF_SCOPE = "quarantined_out_of_scope"
#: Declared confidence below the caller's floor. Off unless a floor is passed.
QUARANTINED_WEAK_EVIDENCE = "quarantined_weak_evidence"
#: Structurally unusable: an empty required side, a non-string participant, a
#: coupled transport naming a process that does not exist, an unresolvable actor.
QUARANTINED_BROKEN_REFERENCE = "quarantined_broken_reference"
#: Admitted, then stranded during closure because an entity it referenced was
#: removed with the process that was keeping that entity alive.
QUARANTINED_DISCONNECTED = "quarantined_disconnected"

ADMISSION_STATES: Tuple[str, ...] = (
    CORE_ACCEPTED,
    AUXILIARY_ACCEPTED,
    QUARANTINED_UNMAPPED_ENTITY,
    QUARANTINED_OUT_OF_SCOPE,
    QUARANTINED_WEAK_EVIDENCE,
    QUARANTINED_BROKEN_REFERENCE,
    QUARANTINED_DISCONNECTED,
)

ACCEPTED_STATES: frozenset[str] = frozenset({CORE_ACCEPTED, AUXILIARY_ACCEPTED})
QUARANTINE_STATES: frozenset[str] = frozenset(ADMISSION_STATES) - ACCEPTED_STATES


QUARANTINE_REPORT_FILENAME = "quarantine_report.json"
REMOVED_ENTITY_REPORT_FILENAME = "removed_entity_report.json"
CLOSURE_REPORT_FILENAME = "graph_closure_iterations.json"
COVERAGE_REPORT_FILENAME = "coverage_summary.json"

#: Where a superseded artifact set is archived, keyed by the full decision
#: identifier (payload hash *and* decision-input hash) it was taken under. One
#: decision per payload version per rule set means several per session, and each
#: is the only record of why that version was admitted under those rules.
QUARANTINE_HISTORY_DIRNAME = "quarantine_history"

#: A closure round can only ever shrink the graph, so the loop terminates on its
#: own. The cap exists to turn a hypothetical non-shrinking round into a loud,
#: recorded failure instead of a hang.
DEFAULT_MAX_CLOSURE_ITERATIONS = 50

#: Minimum surviving ``core_accepted`` processes for strict success. One is not
#: arbitrary: zero is "the graph is empty", which rule 3 above exists to refuse.
DEFAULT_MIN_CORE_PROCESSES = 1

#: Fraction of declared requested-core terms a surviving core process must touch.
DEFAULT_MIN_CORE_COVERAGE = 0.5


#: Bumped whenever a change in this module can change a verdict for an unchanged
#: payload under unchanged controls. It is part of the decision input hash, so a
#: bump invalidates every stored decision instead of letting a report produced by
#: the old rules authorize an export judged by the new ones.
QUARANTINE_POLICY_VERSION = "2026-07-31.1"


def _canonical_requested_core(
    requested_core: Optional[Sequence[str]],
    pathway_context: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """The requested-core controls reduced to what can actually change a verdict.

    Only the keys :func:`collect_requested_core_terms` reads are taken from the
    context, normalized and sorted. Hashing the whole ``pathway_context`` would
    invalidate a perfectly good decision because an unrelated field -- the source
    text, a timestamp -- moved, and every spurious invalidation is a re-run the
    reviewer has to wait through. Hashing the derived terms instead is exact:
    those terms are the only route by which either control reaches a verdict.

    ``requested_core=None`` and ``requested_core=[]`` stay distinguishable
    (``None`` vs a list). They are different regimes -- ``None`` falls through to
    the context, ``[]`` explicitly declares no terms and suppresses it.
    """

    explicit: Optional[List[str]] = None
    if requested_core is not None:
        explicit = sorted(
            {
                _normalize(term)
                for term in collect_requested_core_terms({}, requested_core=requested_core)
            }
            - {""}
        )
    context_terms = sorted(
        {
            _normalize(term)
            for term in collect_requested_core_terms({}, pathway_context=pathway_context)
        }
        - {""}
    )
    return {"requested_core": explicit, "pathway_context_core": context_terms}


def canonical_decision_inputs(
    *,
    mode: ExportMode = DEFAULT_EXPORT_MODE,
    strict_db: bool = True,
    requested_core: Optional[Sequence[str]] = None,
    pathway_context: Optional[Mapping[str, Any]] = None,
    confidence_floor: Optional[float] = None,
    min_core_processes: int = DEFAULT_MIN_CORE_PROCESSES,
    min_core_coverage: float = DEFAULT_MIN_CORE_COVERAGE,
    max_iterations: int = DEFAULT_MAX_CLOSURE_ITERATIONS,
    policy_version: str = QUARANTINE_POLICY_VERSION,
) -> Dict[str, Any]:
    """Every control that can change a verdict, in canonical form.

    This is the complete decision input beside the payload. If a control is
    missing from here, a decision taken under one setting can silently authorize
    an export under another -- and the two settings that matter most, ``mode`` and
    ``strict_db``, are precisely the ones that make quarantine stricter or laxer.
    A research decision quarantines nothing; letting it authorize a strict export
    would ship every unmapped process the strict run exists to stop.
    """

    core = _canonical_requested_core(requested_core, pathway_context)
    return {
        "policy_version": str(policy_version),
        "export_mode": coerce_mode(mode),
        "strict_db": bool(strict_db),
        "requested_core": core["requested_core"],
        "pathway_context_core": core["pathway_context_core"],
        "confidence_floor": None if confidence_floor is None else float(confidence_floor),
        "min_core_processes": int(min_core_processes),
        "min_core_coverage": float(min_core_coverage),
        "max_iterations": int(max_iterations),
    }


def decision_input_hash(**controls: Any) -> str:
    """Canonical fingerprint of :func:`canonical_decision_inputs`."""

    return payload_hash(canonical_decision_inputs(**controls))


def decision_identifier(report: Any) -> str:
    """``<admitted payload hash>.<decision input hash>`` -- the full decision id.

    A decision is identified by *what it judged* and *the rules it judged under*,
    never by the payload alone. Two runs over the same payload -- one strict, one
    research -- are different decisions with opposite conclusions, and keying
    history on the payload hash would let the second overwrite the first.
    """

    if not isinstance(report, Mapping):
        return ""
    admitted = str(report.get("admitted_payload_hash") or "")
    inputs = str(report.get("decision_input_hash") or "")
    if not admitted or not inputs:
        return ""
    return f"{admitted}.{inputs}"


def decision_inputs_match(report: Any, **controls: Any) -> bool:
    """Whether ``report`` was decided under exactly ``controls``.

    Separate from :func:`decision_matches` because the two mismatches need
    different handling. A payload that moved is routine -- re-quarantine the new
    version. Controls that moved are not: re-evaluating from the payload in hand
    would judge an *already reduced strict graph* under research rules and call
    the result a research decision, when research mode's contract is that it never
    reduced anything. The caller has to refuse and start a new run instead.
    """

    if not isinstance(report, Mapping):
        return False
    stored = str(report.get("decision_input_hash") or "")
    return bool(stored) and stored == decision_input_hash(**controls)


def decision_matches(report: Any, payload: Mapping[str, Any], **controls: Any) -> bool:
    """Whether ``report``'s decision covers ``payload`` under ``controls``.

    Both halves are required, and each catches a different way of exporting
    material nothing admitted:

    * **Payload.** Compared against ``resulting_payload_hash``, not
      ``admitted_payload_hash``. That distinction is the whole check: the exporter
      holds the graph quarantine *produced*, so matching it against the graph
      quarantine was *given* fails on every run where anything was actually
      quarantined -- which is every run this feature exists for.
      ``admitted_payload_hash`` stays on the report as provenance: it says which
      version was judged, and it is half of the history archive key.
    * **Controls.** Recording the mode, ``strict_db`` and the thresholds on the
      report is not enough; nothing consults a field nobody compares. Without this
      half, flipping the export mode after the boundary reuses a decision taken
      under the other set of rules.

    A report missing either hash is treated as not matching. Deliberate: a report
    that cannot prove what it judged, or under what rules, proves nothing, and
    re-running quarantine is cheap next to shipping an unadmitted payload.
    """

    if not isinstance(report, Mapping):
        return False
    stored = str(report.get("resulting_payload_hash") or "")
    if not stored or stored != admitted_payload_hash(payload):
        return False
    return decision_inputs_match(report, **controls)


#: Reaction sides. Both are required and every named participant on each is
#: essential. Enzymes and modifiers are NOT essential -- a reaction without its
#: catalyst is still that reaction -- but an *unresolvable* actor is still a
#: broken reference and quarantines the row, because "no unknown references" is a
#: strict gate we are not weakening.
_REACTION_SIDES: Tuple[str, ...] = ("inputs", "outputs")

#: Transport cargo, in the order ``pwml/ir.py`` reads it (:1780-1783):
#: ``transport_elements`` first, then a bare ``cargo`` / ``cargo_complex``.
#: Reading only ``cargo``, as this module first did, quarantined every transport
#: written in the primary shape.
_TRANSPORT_ELEMENT_KEYS: Tuple[str, ...] = ("element", "cargo", "name")

#: Interaction endpoint aliases, in the order production resolves them
#: (``process_normalizer.validate_registry_references``:4188-4189 and
#: ``apply_audit_patch._referenced_entity_norms``:939-940). All three spellings
#: are live in payloads on disk; matching only ``entity_1``/``entity_2`` treats a
#: perfectly valid ``left``/``right`` interaction as having no endpoints at all.
_INTERACTION_LEFT_KEYS: Tuple[str, ...] = ("left", "entity_1", "source")
_INTERACTION_RIGHT_KEYS: Tuple[str, ...] = ("right", "entity_2", "target")

_ACTOR_SLOTS: Dict[str, Tuple[str, ...]] = {
    "reactions": ("enzymes", "modifiers"),
    "reaction_coupled_transports": ("enzymes", "modifiers"),
    "transports": ("transporters",),
    "interactions": (),
    "sub_pathways": (),
}

#: Reason attached to every quarantined ``reaction_coupled_transports`` row.
#: ``build_pwml_ir`` (:1934-1946) builds an RCT with ``left``, ``right`` and
#: ``enzymes`` hard-coded to ``[]`` and never fills them from the raw row, and
#: ``validate_pwml_ir`` (:3018-3022) then raises ``rct_missing_left``,
#: ``rct_missing_right`` and ``rct_missing_enzyme`` for it. There is therefore no
#: exportable representation of a coupled transport today, and admitting one
#: would hand the reviewer a payload that is guaranteed to fail IR validation
#: three errors later. Quarantining it says so at the point the decision is made.
RCT_NOT_EXPORTABLE_REASON = "reaction_coupled_transport_has_no_exportable_ir_representation"

#: Stamped on a research-mode report so no reader mistakes it for a strict one.
RESEARCH_DIAGNOSTIC_NOTE = (
    "research mode: quarantine ran diagnostically. Nothing was removed and the "
    "candidate pathway is unchanged; the decisions below are review flags."
)

_PROCESS_KIND_BY_BUCKET: Dict[str, str] = {
    "reactions": "reaction",
    "reaction_coupled_transports": "reaction_coupled_transport",
    "transports": "transport",
    "interactions": "interaction",
    "sub_pathways": "sub_pathway",
}

#: Entity buckets that participate in the process graph and are therefore
#: subject to closure. Species, subcellular_locations, cell_types and tissues are
#: deliberately absent: they are referenced by biological_states rather than by
#: processes, and pruning them on process references would delete the compartment
#: vocabulary the surviving states are built from.
_GRAPH_ENTITY_BUCKETS: Tuple[str, ...] = (
    "compounds",
    "proteins",
    "protein_complexes",
    "nucleic_acids",
    "element_collections",
)

#: Buckets whose rows must not be exported at degree zero.
_DEGREE_ZERO_BUCKETS: Tuple[str, ...] = ("proteins", "protein_complexes")

#: element_locations bucket -> the field naming the entity it locates.
_LOCATION_BUCKETS: Dict[str, str] = {
    "compound_locations": "compound",
    "protein_locations": "protein",
    "nucleic_acid_locations": "nucleic_acid",
    "element_collection_locations": "element_collection",
}


# ── Small shared helpers ────────────────────────────────────────────────────


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return _canonical(str(value)) if isinstance(value, str) else ""


def _row_name(row: Any) -> str:
    if isinstance(row, dict):
        return _text(row.get("name"))
    if isinstance(row, str):
        return _canonical(row)
    return ""


def _component_name(component: Any) -> str:
    """The protein a protein_complex component names, in either row shape."""

    if isinstance(component, str):
        return _canonical(component)
    if isinstance(component, dict):
        for key in ("protein", "name", "entity", "component"):
            value = component.get(key)
            if isinstance(value, str) and _canonical(value):
                return _canonical(value)
    return ""


def _participant_names(value: Any) -> List[str]:
    """Every name a participant slot holds, in declaration order.

    Accepts the three shapes the payload actually carries in these slots: a bare
    string, a list of strings, and a list of ``{"name"/"entity": ...}`` rows.
    """

    items = [value] if isinstance(value, (str, dict)) else _safe_list(value)
    out: List[str] = []
    for item in items:
        if isinstance(item, str):
            name = _canonical(item)
        elif isinstance(item, dict):
            name = _text(item.get("name") or item.get("entity") or item.get("compound"))
        else:
            name = ""
        if name:
            out.append(name)
    return out


def _transport_cargo_names(process: Mapping[str, Any]) -> Tuple[List[str], bool]:
    """``(names, malformed)`` for a transport, reading all three supported shapes.

    Mirrors ``pwml/ir.py``:1780-1789 exactly: ``transport_elements`` wins when
    present, and each element may be a bare string or a row keyed
    ``element`` / ``cargo`` / ``name``. Only when there are no elements does the
    top-level ``cargo`` / ``cargo_complex`` apply. ``cargo_complex`` takes
    precedence over ``cargo`` when both are non-blank, matching
    ``validate_registry_references``:4173.
    """

    elements = _safe_list(process.get("transport_elements"))
    if elements:
        names: List[str] = []
        malformed = False
        for element in elements:
            if isinstance(element, str):
                name = _canonical(element)
            elif isinstance(element, dict):
                name = ""
                for key in _TRANSPORT_ELEMENT_KEYS:
                    value = element.get(key)
                    if isinstance(value, str) and _canonical(value):
                        name = _canonical(value)
                        break
            else:
                malformed = True
                continue
            if name:
                names.append(name)
            else:
                malformed = True
        return names, malformed

    cargo_complex = process.get("cargo_complex")
    cargo = (
        cargo_complex
        if isinstance(cargo_complex, str) and _canonical(cargo_complex)
        else process.get("cargo")
    )
    names = _participant_names(cargo)
    return names, _has_malformed_participant(cargo) if cargo not in (None, "", []) else False


def _interaction_endpoints(process: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """``(left_names, right_names)`` under every alias production accepts."""

    def first(keys: Tuple[str, ...]) -> List[str]:
        for key in keys:
            value = process.get(key)
            names = _participant_names(value)
            if names:
                return names
        return []

    return first(_INTERACTION_LEFT_KEYS), first(_INTERACTION_RIGHT_KEYS)


def _essential_participants(
    bucket: str, process: Mapping[str, Any]
) -> List[Tuple[str, List[str], bool]]:
    """``[(slot_label, names, malformed)]`` for the slots a process needs.

    One function rather than a per-bucket slot table, because the three process
    kinds do not share a shape: a reaction has two lists, a transport has one
    slot spelled three ways, and an interaction has two endpoints spelled three
    ways each. The table this replaced read ``cargo`` and ``entity_1``/
    ``entity_2`` only, so every transport written with ``transport_elements`` and
    every interaction written with ``left``/``right`` was quarantined as having no
    participants -- a false negative on a completely valid row.
    """

    if bucket == "reactions":
        return [
            (side, _participant_names(process.get(side)), _has_malformed_participant(process.get(side)))
            for side in _REACTION_SIDES
        ]
    if bucket == "transports":
        names, malformed = _transport_cargo_names(process)
        return [("cargo", names, malformed)]
    if bucket == "interactions":
        left, right = _interaction_endpoints(process)
        return [("entity_1", left, False), ("entity_2", right, False)]
    return []


def _has_malformed_participant(value: Any) -> bool:
    """True when a slot holds an entry that carries no usable name at all.

    A blank string or a nameless dict is not a participant the export can place;
    it is a hole the extractor left, and dropping it quietly would change the
    reaction's stoichiometry without saying so.
    """

    items = [value] if isinstance(value, (str, dict)) else _safe_list(value)
    for item in items:
        if isinstance(item, str):
            if not _canonical(item):
                return True
        elif isinstance(item, dict):
            if not _text(item.get("name") or item.get("entity") or item.get("compound")):
                return True
        elif item is not None:
            return True
    return False


# ── Entity registry ─────────────────────────────────────────────────────────


@dataclass
class _Registry:
    """Every declared entity, keyed the way the strict gate keys them."""

    by_norm: Dict[str, Tuple[str, Dict[str, Any]]] = field(default_factory=dict)
    norms: Set[str] = field(default_factory=set)
    proteins_by_identity: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    display_by_norm: Dict[str, str] = field(default_factory=dict)
    #: Names declared under more than one *primary* bucket. A reference to one of
    #: these resolves to whichever declaration was registered first, which is an
    #: arbitrary answer to a question the payload does not actually answer.
    buckets_by_norm: Dict[str, Set[str]] = field(default_factory=dict)

    def resolve(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        return self.by_norm.get(_normalize(_canonical(name)))

    def declared(self, name: str) -> bool:
        return _normalize(_canonical(name)) in self.norms

    def ambiguous(self, name: str) -> Tuple[str, ...]:
        """The buckets a name is primarily declared in, when there is more than one."""

        buckets = self.buckets_by_norm.get(_normalize(_canonical(name)), set())
        return tuple(sorted(buckets)) if len(buckets) > 1 else ()


def _build_registry(payload: Mapping[str, Any]) -> _Registry:
    entities = _safe_dict(payload.get("entities"))
    registry = _Registry()
    for bucket in _GRAPH_ENTITY_BUCKETS:
        rows = _safe_list(entities.get(bucket))
        # Synonyms count, exactly as validate_registry_references counts them: a
        # reaction that says "NAD" resolves against a compound named "NAD+" that
        # lists "NAD" as a synonym.
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = _row_name(row)
            if not name:
                continue
            for norm in _entity_name_norms([row]):
                if norm:
                    registry.by_norm.setdefault(norm, (bucket, row))
                    registry.norms.add(norm)
                    registry.display_by_norm.setdefault(norm, name)
            # Ambiguity is tracked on the row's own name only. Two entities that
            # merely share a *synonym* are a naming overlap the registry is built
            # to tolerate; two entities that share a NAME are two answers to
            # "what is this", and resolve() would silently pick the earlier bucket.
            registry.buckets_by_norm.setdefault(_normalize(name), set()).add(bucket)
            if bucket == "proteins":
                identity = protein_external_identity(row)
                if identity:
                    registry.proteins_by_identity.setdefault(identity.casefold(), row)
    return registry


def _entity_representability(
    bucket: str,
    row: Mapping[str, Any],
    registry: _Registry,
    *,
    strict_db: bool,
    _seen: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """Whether strict PWML can represent this entity, and why not when it cannot.

    Every rule here is one the required-field contract already enforces, matched
    to its own error code and to the *severity* the contract gives it. That
    alignment is the whole point of the function: a rule stricter than the
    contract quarantines a process that would have exported (a false negative on
    real biology), and a rule looser than it lets a row through to a failure that
    was predictable at this point.

    The severity split matters most on protein complexes. The contract makes
    component resolution, component species and component identity ERRORS only
    for a *generated* wrapper (``ir.py``:2246 ``generated_complex``) and mere
    warnings otherwise, because a hand-declared multi-subunit complex legitimately
    names subunits the payload does not carry as separate protein rows. Treating
    those as blocking, as this function first did, would quarantine every reaction
    catalysed by a real complex.

    The Unknown-backed functional complex is the deliberate exception and needs no
    special case: its component IS the PathBank ``Unknown`` protein, which
    :func:`is_pathbank_unknown_protein` accepts as a sanctioned identity, so the
    generic component walk admits it. A placeholder that *forges* an accession is
    still refused, by :func:`placeholder_claims_real_identity`.
    """

    if not _row_name(row):
        return False, f"{bucket.rstrip('s')}_missing_name"

    claim = placeholder_claims_real_identity(row)
    if claim:
        return False, f"malformed_placeholder:{claim}"

    if bucket == "proteins":
        if not strict_db:
            return True, ""
        if not protein_species_context(row):
            return False, "protein_missing_species"
        if is_pathbank_unknown_protein(row):
            return True, ""
        if has_protein_external_identity(row):
            return True, ""
        return False, "protein_missing_external_identity"

    if bucket == "protein_complexes":
        if strict_db and not protein_species_context(row):
            return False, "protein_complex_missing_species"
        generated = is_generated_complex_wrapper(row)
        components = _safe_list(row.get("components"))
        if not components:
            # Blocking only for a generated wrapper, exactly as the contract has
            # it: a declared complex with no listed subunits is a warning there.
            return (False, "protein_complex_missing_components") if generated else (True, "")
        if not generated:
            return True, ""
        seen = set(_seen or ())
        self_norm = _normalize(_row_name(row))
        if self_norm in seen:
            # A complex listed among its own components. Refusing rather than
            # recursing keeps this total; the cycle is itself a broken shape.
            return False, "protein_complex_component_cycle"
        seen.add(self_norm)
        for component in components:
            name = _component_name(component)
            if not name:
                return False, "protein_complex_component_unnamed"
            resolved = registry.resolve(name)
            if resolved is None:
                identity_row = registry.proteins_by_identity.get(_canonical(name).casefold())
                if identity_row is None:
                    return False, f"protein_complex_component_unresolved:{name}"
                resolved = ("proteins", identity_row)
            component_bucket, component_row = resolved
            ok, reason = _entity_representability(
                component_bucket,
                component_row,
                registry,
                strict_db=strict_db,
                _seen=seen,
            )
            if not ok:
                return False, f"protein_complex_component_unrepresentable:{name}:{reason}"
        return True, ""

    return True, ""


# ── Requested-pathway core ──────────────────────────────────────────────────


def collect_requested_core_terms(
    payload: Mapping[str, Any],
    *,
    requested_core: Optional[Sequence[str]] = None,
    pathway_context: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """The terms that define "the pathway the user asked for".

    Priority: an explicit ``requested_core`` argument, then the Stage-0
    ``pathway_context`` the caller passed, then whatever the payload happens to
    carry. ``key_compounds`` and ``key_proteins`` are what the preprocessor
    produced when it read the request, before any extraction could bias them --
    the only description of the goal that is independent of the result.

    **Production must pass the context explicitly.** Payload-only discovery is
    not sufficient and was the bug: the final mapped payload that reaches the
    quarantine boundary commonly carries neither ``metadata`` nor
    ``pathway_context``, so the search finds nothing, ``requested_core_declared``
    comes back False, and :func:`evaluate_core_coverage` drops to the regime where
    only emptiness is refused -- letting an unrelated survivor pass on exactly the
    runs the check exists for.

    Never derived from the surviving graph. Terms taken from what survived would
    match whatever survived, which is not a test.
    """

    out: List[str] = []
    seen: Set[str] = set()

    def add_all(value: Any) -> None:
        for item in _safe_list(value):
            term = _canonical(item) if isinstance(item, str) else _row_name(item)
            norm = _normalize(term)
            if term and norm and norm not in seen:
                seen.add(norm)
                out.append(term)

    if requested_core is not None:
        add_all(list(requested_core))
        return out

    containers = [
        _safe_dict(pathway_context),
        _safe_dict(payload.get("metadata")),
        _safe_dict(payload.get("pathway_context")),
        _safe_dict(_safe_dict(payload.get("metadata")).get("pathway_context")),
    ]
    for container in containers:
        add_all(container.get("requested_core"))
        add_all(container.get("key_compounds"))
        add_all(container.get("key_proteins"))
    return out


def _process_core_terms(bucket: str, process: Mapping[str, Any]) -> Set[str]:
    """Normalized names a process touches, for matching against the core terms."""

    terms: Set[str] = set()
    name = _row_name(process)
    if name:
        terms.add(_normalize(name))
    for _slot, names, _malformed in _essential_participants(bucket, process):
        for participant in names:
            terms.add(_normalize(participant))
    for actor_slot in _ACTOR_SLOTS.get(bucket, ()):
        for actor in _safe_list(process.get(actor_slot)):
            actor_name = _actor_name(actor) if not isinstance(actor, str) else _canonical(actor)
            if actor_name:
                terms.add(_normalize(actor_name))
    terms.discard("")
    return terms


def _term_matches(term_norm: str, process_terms: Set[str]) -> bool:
    """Whether a requested-core term is touched by a process.

    Substring either way, because a request says "glutathione" and the payload
    says "glutathione synthetase" / "reduced glutathione". Exact-only matching
    scored real pathways at zero coverage and would have failed every run.
    """

    if not term_norm:
        return False
    return any(term_norm == other or term_norm in other or other in term_norm for other in process_terms)


def evaluate_core_coverage(
    payload: Mapping[str, Any],
    admissions: Sequence[Mapping[str, Any]],
    *,
    requested_core: Optional[Sequence[str]] = None,
    pathway_context: Optional[Mapping[str, Any]] = None,
    min_core_processes: int = DEFAULT_MIN_CORE_PROCESSES,
    min_core_coverage: float = DEFAULT_MIN_CORE_COVERAGE,
) -> "t2pw.pipeline.release_status.CoverageVerdict":
    """Decide whether what survived is still the pathway that was requested.

    Two independent failures are refused here, and they are the reason this check
    exists at all:

    * **An empty graph.** Quarantining every process leaves a payload that
      violates no gate, because there is nothing left to violate one. Strict
      success has to mean "the pathway exported", not "nothing invalid remained".
    * **A graph of survivors that are not the request.** Coverage is measured
      only against processes marked :data:`CORE_ACCEPTED`, so ten surviving
      reactions from an unrelated part of the paper score zero and cannot carry
      the run to success.

    When no core was declared, relevance is unjudgeable and only the emptiness
    rule applies -- ``requested_core_declared`` says which regime was in force.

    Returns a :class:`~t2pw.pipeline.release_status.CoverageVerdict`, which IS a
    ``dict`` -- same keys, same values, same JSON bytes, same ``==`` against a
    plain dict -- so every existing consumer and every pinned coverage document
    is untouched. What the type adds is a stable shape for the consumers named in
    ``MASTER_PLAN.md:230``: semantic accessors (``below_coverage_minimum``,
    ``has_surviving_core``, ``missing_anchors``, ``completeness``) that answer the
    D-002 questions without string-matching ``reasons``, and one place where
    "undeclared" stops being reported as coverage ``0.0``.

    The import is function-local on purpose: this module is imported by the
    pipeline at large, and the coverage verdict is the only thing here that needs
    the classification vocabulary.
    """

    from t2pw.pipeline.release_status import CoverageVerdict

    terms = collect_requested_core_terms(
        payload, requested_core=requested_core, pathway_context=pathway_context
    )
    declared = bool(terms)

    # Which input actually produced the terms. "payload" is the one worth
    # noticing in a production report: it means nobody passed the Stage-0
    # context and the check is running on whatever the payload happened to keep.
    if requested_core is not None:
        source = "explicit_argument"
    elif not declared:
        source = "none"
    elif collect_requested_core_terms(
        {}, requested_core=None, pathway_context=pathway_context
    ):
        source = "pathway_context"
    else:
        source = "payload"

    accepted = [row for row in admissions if row.get("state") in ACCEPTED_STATES]
    core = [row for row in accepted if row.get("state") == CORE_ACCEPTED]
    auxiliary = [row for row in accepted if row.get("state") == AUXILIARY_ACCEPTED]

    matched_terms: List[str] = []
    unmatched_terms: List[str] = []
    if declared:
        core_term_sets = [set(_safe_list(row.get("core_terms"))) for row in core]
        for term in terms:
            term_norm = _normalize(term)
            if any(_term_matches(term_norm, term_set) for term_set in core_term_sets):
                matched_terms.append(term)
            else:
                unmatched_terms.append(term)

    coverage_ratio = (len(matched_terms) / len(terms)) if declared and terms else 0.0

    reasons: List[str] = []
    if not accepted:
        reasons.append("no_surviving_process")
    # Which survivors count toward the minimum depends on the regime, and getting
    # this wrong is a false negative on a clean pathway. With a declared core it
    # has to be core_accepted only -- that is what stops ten unrelated survivors
    # from carrying the run. With no declared core the split is not meaningful:
    # every accepted process falls to AUXILIARY unless it is a reaction, so a
    # pathway of four interactions and no reactions scored zero and was refused
    # despite failing no gate and having nothing quarantined
    # (runs/2026-07-28_2122 PMC12624714/strict, found by replaying real legs).
    # Undeclared means "relevance unjudgeable", and the only rule left is
    # "not empty".
    countable = core if declared else accepted
    if len(countable) < int(min_core_processes):
        reasons.append(
            f"core_process_count_below_minimum:{len(countable)}<{int(min_core_processes)}"
        )
    if declared and coverage_ratio < float(min_core_coverage):
        reasons.append(
            f"requested_core_coverage_below_minimum:{coverage_ratio:.3f}<{float(min_core_coverage):.3f}"
        )

    return CoverageVerdict({
        "schema_version": 1,
        "requested_core_terms": terms,
        "requested_core_declared": declared,
        # The context exactly as production handed it over, unedited. The terms
        # above are derived; this is the evidence they were derived from, and it
        # is the only way a reviewer can tell "the request had no anchors" from
        # "the anchors were dropped on the way here".
        "requested_context": deepcopy(dict(pathway_context)) if isinstance(pathway_context, Mapping) else None,
        "requested_core_source": source,
        "matched_terms": matched_terms,
        "unmatched_terms": unmatched_terms,
        "coverage_ratio": round(coverage_ratio, 6),
        "core_accepted_processes": len(core),
        "auxiliary_accepted_processes": len(auxiliary),
        "surviving_processes": len(accepted),
        "quarantined_processes": len(admissions) - len(accepted),
        "thresholds": {
            "min_core_processes": int(min_core_processes),
            "min_core_coverage": float(min_core_coverage),
        },
        "minimum_core_satisfied": not reasons,
        "reasons": reasons,
    })


# ── Admission ───────────────────────────────────────────────────────────────


def _essential_participant_verdict(
    bucket: str,
    process: Mapping[str, Any],
    registry: _Registry,
    *,
    strict_db: bool,
) -> Optional[Tuple[str, str, str]]:
    """First essential-participant problem in this process, or ``None``.

    Returns ``(state, reason, participant)``. The whole process is the unit: the
    caller quarantines on any non-``None`` result and never edits the slot. This
    is requirement 2 in the module docstring, and it is the difference between
    "we could not export this reaction" and "we exported a reaction the paper
    does not contain".
    """

    for slot, names, malformed in _essential_participants(bucket, process):
        if malformed:
            return (
                QUARANTINED_BROKEN_REFERENCE,
                f"malformed_participant_in_{slot}",
                "",
            )
        if not names:
            return (
                QUARANTINED_BROKEN_REFERENCE,
                f"missing_{slot}",
                "",
            )
        for name in names:
            ambiguous = registry.ambiguous(name)
            if ambiguous:
                return (
                    QUARANTINED_BROKEN_REFERENCE,
                    f"ambiguous_entity_type:{'+'.join(ambiguous)}",
                    name,
                )
            resolved = registry.resolve(name)
            if resolved is None:
                return (QUARANTINED_UNMAPPED_ENTITY, f"undeclared_entity_in_{slot}", name)
            entity_bucket, row = resolved
            ok, reason = _entity_representability(
                entity_bucket, row, registry, strict_db=strict_db
            )
            if not ok:
                return (QUARANTINED_UNMAPPED_ENTITY, reason, name)
    return None


def _actor_verdict(
    bucket: str,
    process: Mapping[str, Any],
    registry: _Registry,
    *,
    strict_db: bool,
) -> Optional[Tuple[str, str, str]]:
    """First unusable enzyme/modifier/transporter, or ``None``.

    Actors are not essential -- their absence does not change what the reaction
    is -- but an actor naming nothing declared is precisely the "unknown
    protein/modifier reference" the strict gate aborts on, so the process still
    leaves. Quarantining one reaction beats failing the export.
    """

    for slot in _ACTOR_SLOTS.get(bucket, ()):
        for actor in _safe_list(process.get(slot)):
            name = _actor_name(actor) if not isinstance(actor, str) else _canonical(actor)
            if not name:
                return (QUARANTINED_BROKEN_REFERENCE, f"unnamed_actor_in_{slot}", "")
            ambiguous = registry.ambiguous(name)
            if ambiguous:
                return (
                    QUARANTINED_BROKEN_REFERENCE,
                    f"ambiguous_entity_type:{'+'.join(ambiguous)}",
                    name,
                )
            resolved = registry.resolve(name)
            if resolved is None:
                return (QUARANTINED_BROKEN_REFERENCE, f"unresolved_actor_in_{slot}", name)
            entity_bucket, row = resolved
            ok, reason = _entity_representability(
                entity_bucket, row, registry, strict_db=strict_db
            )
            if not ok:
                return (QUARANTINED_UNMAPPED_ENTITY, reason, name)
    return None


def _scope_is_out_of_scope(process: Mapping[str, Any]) -> bool:
    """Only an explicit ``out_of_scope`` verdict counts.

    Deliberately identical to ``pipeline.filter_out_of_scope_reactions``: an
    absent label means nobody classified the row, not that the row is off-pathway,
    and every RAG-imported and Stage-2-added reaction arrives unlabelled. Read
    that function's docstring before loosening this.
    """

    scope = process.get("scope_membership")
    if isinstance(scope, str):
        return scope.strip().casefold() == "out_of_scope"
    if isinstance(scope, dict):
        verdict = scope.get("membership") or scope.get("verdict") or scope.get("status")
        return isinstance(verdict, str) and verdict.strip().casefold() == "out_of_scope"
    return False


def _weak_evidence_verdict(
    process: Mapping[str, Any],
    *,
    confidence_floor: Optional[float],
) -> Optional[str]:
    """Reason string when the row falls below the caller's confidence floor.

    Off by default (``confidence_floor=None``). A missing ``confidence`` is never
    weak evidence: most payloads in this repo carry none, and treating absence as
    zero would quarantine every reaction the moment a caller set a floor.
    """

    if confidence_floor is None:
        return None
    value = process.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if float(value) >= float(confidence_floor):
        return None
    return f"confidence_below_floor:{float(value):.3f}<{float(confidence_floor):.3f}"


def _admit_processes(
    payload: Mapping[str, Any],
    registry: _Registry,
    *,
    strict_db: bool,
    core_terms: Sequence[str],
    confidence_floor: Optional[float],
) -> List[Dict[str, Any]]:
    """One admission record per process, in payload order."""

    processes = _safe_dict(payload.get("processes"))
    core_term_norms = [_normalize(term) for term in core_terms if _normalize(term)]
    records: List[Dict[str, Any]] = []

    for bucket, kind in _PROCESS_KIND_BY_BUCKET.items():
        for index, process in enumerate(_safe_list(processes.get(bucket))):
            pointer = f"/processes/{bucket}/{index}"
            if not isinstance(process, dict):
                records.append(
                    {
                        "process_kind": kind,
                        "bucket": bucket,
                        "index": index,
                        "pointer": pointer,
                        "name": "",
                        "state": QUARANTINED_BROKEN_REFERENCE,
                        "reason": "process_row_not_an_object",
                        "essential_participant": "",
                        "iteration": 0,
                        "core_terms": [],
                        "detail": scrub_detail({"found_type": type(process).__name__}),
                    }
                )
                continue

            name = _row_name(process)
            process_terms = _process_core_terms(bucket, process)
            base: Dict[str, Any] = {
                "process_kind": kind,
                "bucket": bucket,
                "index": index,
                "pointer": pointer,
                "name": name,
                "essential_participant": "",
                "iteration": 0,
                "core_terms": sorted(process_terms),
            }

            verdict: Optional[Tuple[str, str, str]] = None

            if _scope_is_out_of_scope(process):
                verdict = (QUARANTINED_OUT_OF_SCOPE, "explicit_out_of_scope_verdict", "")

            if verdict is None:
                weak = _weak_evidence_verdict(process, confidence_floor=confidence_floor)
                if weak:
                    verdict = (QUARANTINED_WEAK_EVIDENCE, weak, "")

            if verdict is None and bucket == "reaction_coupled_transports":
                # Unconditional, and it does not depend on this row's contents.
                # See RCT_NOT_EXPORTABLE_REASON: the IR cannot represent one at
                # all, so "does its reaction reference resolve?" is a question
                # about a row that could not have shipped either way.
                verdict = (
                    QUARANTINED_BROKEN_REFERENCE,
                    RCT_NOT_EXPORTABLE_REASON,
                    _text(process.get("reaction")),
                )

            if verdict is None:
                verdict = _essential_participant_verdict(
                    bucket, process, registry, strict_db=strict_db
                )

            if verdict is None:
                verdict = _actor_verdict(bucket, process, registry, strict_db=strict_db)

            if verdict is not None:
                state, reason, participant = verdict
                records.append(
                    {
                        **base,
                        "state": state,
                        "reason": reason,
                        "essential_participant": participant,
                        "detail": scrub_detail(
                            {
                                "row": row_digest(process, pointer=pointer),
                                "participant": participant,
                            }
                        ),
                    }
                )
                continue

            is_core = (
                any(_term_matches(term, process_terms) for term in core_term_norms)
                if core_term_norms
                else bucket == "reactions"
            )
            records.append(
                {
                    **base,
                    "state": CORE_ACCEPTED if is_core else AUXILIARY_ACCEPTED,
                    "reason": "admitted",
                    "detail": {},
                }
            )
    return records


# ── Graph closure ───────────────────────────────────────────────────────────


class StrictQuarantineInvariantError(RuntimeError):
    """An admission record did not address the payload it was written against.

    Indices are assigned by enumerating the process lists, so an accepted record
    that stops resolving means the lists moved underneath the decision. Skipping
    it deletes a surviving process's edges silently, which reads downstream as a
    degree-zero entity and refuses an export nothing was wrong with.
    """


def _surviving_processes(
    payload: Mapping[str, Any],
    admissions: Sequence[Mapping[str, Any]],
    *,
    strict: bool = False,
) -> List[Tuple[str, Dict[str, Any]]]:
    """The rows the accepted admission records point at.

    ``strict`` decides what an unresolvable ``(bucket, index)`` means. Skipping is
    right where a row is *expected* to have gone: closure quarantines it through
    :func:`_revalidate_surviving_processes`, which records
    ``process_row_vanished_during_closure`` instead. Where the caller believes
    every record resolves, a skip is an invariant break and must say so.
    """

    processes = _safe_dict(payload.get("processes"))
    out: List[Tuple[str, Dict[str, Any]]] = []
    for record in admissions:
        if record.get("state") not in ACCEPTED_STATES:
            continue
        bucket = str(record.get("bucket") or "")
        rows = _safe_list(processes.get(bucket))
        index = int(record.get("index") or 0)
        if 0 <= index < len(rows) and isinstance(rows[index], dict):
            out.append((bucket, rows[index]))
        elif strict:
            raise StrictQuarantineInvariantError(
                f"admission record {record.get('pointer') or f'/processes/{bucket}/{index}'} "
                f"({record.get('state')}) does not resolve: bucket {bucket!r} holds "
                f"{len(rows)} row(s). The process lists moved after admission."
            )
    return out


def _referenced_entity_norms(
    payload: Mapping[str, Any], admissions: Sequence[Mapping[str, Any]]
) -> Set[str]:
    """Every entity name the surviving processes reach, normalized."""

    found: Set[str] = set()
    for bucket, process in _surviving_processes(payload, admissions):
        for _slot, names, _malformed in _essential_participants(bucket, process):
            for name in names:
                found.add(_normalize(name))
        for slot in _ACTOR_SLOTS.get(bucket, ()):
            for actor in _safe_list(process.get(slot)):
                name = _actor_name(actor) if not isinstance(actor, str) else _canonical(actor)
                if name:
                    found.add(_normalize(name))
    found.discard("")
    return found


def _complex_component_norms(payload: Mapping[str, Any], surviving_complex_norms: Set[str]) -> Set[str]:
    """Components of the complexes that survive.

    Requirement 3: a component protein is kept even at degree zero. It has no
    edge of its own by construction -- the complex carries the edge -- so pruning
    it on connectivity would gut every surviving complex.
    """

    entities = _safe_dict(payload.get("entities"))
    out: Set[str] = set()
    for row in _safe_list(entities.get("protein_complexes")):
        if not isinstance(row, dict):
            continue
        if _normalize(_row_name(row)) not in surviving_complex_norms:
            continue
        for component in _safe_list(row.get("components")):
            norm = _normalize(_component_name(component))
            if norm:
                out.add(norm)
    return out


def _prune_entities(
    payload: Dict[str, Any],
    keep_norms: Set[str],
) -> List[Dict[str, Any]]:
    """Drop graph entities nothing in ``keep_norms`` reaches. Returns removals."""

    entities = _safe_dict(payload.get("entities"))
    removed: List[Dict[str, Any]] = []
    for bucket in _GRAPH_ENTITY_BUCKETS:
        rows = entities.get(bucket)
        if not isinstance(rows, list):
            continue
        kept: List[Any] = []
        for row in rows:
            if not isinstance(row, dict) or not _row_name(row):
                # A row with no usable name cannot be referenced, cannot be
                # exported, and is a guaranteed `*_missing_name` from the
                # required-field contract, which walks every declared row rather
                # than only the reachable ones. Keeping it, as this loop first
                # did, meant closure reported success and the contract then
                # failed on something closure had already looked at.
                removed.append(
                    {
                        "bucket": bucket,
                        "name": _row_name(row) or "<unnamed>",
                        "reason": "malformed_entity_row",
                        "had_external_identity": None,
                    }
                )
                continue
            name = _row_name(row)
            norms = {norm for norm in _entity_name_norms([row]) if norm}
            if norms & keep_norms:
                kept.append(row)
                continue
            reason = (
                "degree_zero_after_quarantine"
                if bucket in _DEGREE_ZERO_BUCKETS
                else "unreferenced_after_quarantine"
            )
            removed.append(
                {
                    "bucket": bucket,
                    "name": name,
                    "reason": reason,
                    "had_external_identity": (
                        has_protein_external_identity(row) if bucket == "proteins" else None
                    ),
                }
            )
        if len(kept) != len(rows):
            entities[bucket] = kept
    return removed


def _prune_locations(payload: Dict[str, Any], declared_norms: Set[str]) -> List[Dict[str, Any]]:
    """Drop element_locations rows whose entity is gone."""

    element_locations = payload.get("element_locations")
    if not isinstance(element_locations, dict):
        return []
    removed: List[Dict[str, Any]] = []
    for bucket, name_field in _LOCATION_BUCKETS.items():
        rows = element_locations.get(bucket)
        if not isinstance(rows, list):
            continue
        kept: List[Any] = []
        for row in rows:
            if not isinstance(row, dict):
                kept.append(row)
                continue
            name = _text(row.get(name_field) or row.get("entity"))
            if name and _normalize(name) not in declared_norms:
                removed.append(
                    {
                        "bucket": f"element_locations/{bucket}",
                        "name": name,
                        "reason": "location_entity_removed",
                        "biological_state": _text(row.get("biological_state")),
                    }
                )
                continue
            kept.append(row)
        if len(kept) != len(rows):
            element_locations[bucket] = kept
    return removed


def _referenced_state_names(
    payload: Mapping[str, Any], admissions: Sequence[Mapping[str, Any]]
) -> Set[str]:
    """Biological-state names anything surviving still points at.

    Walks the ADMITTED processes, not the process lists. Quarantined rows are
    still physically present until the final compaction, so reading the lists
    directly would let a state be kept alive by the very reaction that is on its
    way out -- and the second run of closure over the same payload would then
    remove it, which means the first run never reached a fixpoint.
    """

    found: Set[str] = set()
    for _bucket, row in _surviving_processes(payload, admissions):
        for key in (
            "biological_state",
            "from_biological_state",
            "to_biological_state",
        ):
            name = _text(row.get(key))
            if name:
                found.add(_normalize(name))
        for element in _safe_list(row.get("elements_with_states")):
            if isinstance(element, dict):
                name = _text(element.get("biological_state"))
                if name:
                    found.add(_normalize(name))
    for bucket in _LOCATION_BUCKETS:
        for row in _safe_list(_safe_dict(payload.get("element_locations")).get(bucket)):
            if isinstance(row, dict):
                name = _text(row.get("biological_state"))
                if name:
                    found.add(_normalize(name))
    found.discard("")
    return found


def _prune_biological_states(
    payload: Dict[str, Any], admissions: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Drop states nothing surviving references.

    Left last in the round: it reads the post-prune processes and locations, so a
    state only disappears once every row that pointed at it has already gone. The
    export requires at least one state (``no_biological_states``), which the
    coverage check backstops -- a payload with no surviving process fails there
    first.
    """

    states = payload.get("biological_states")
    if not isinstance(states, list):
        return []
    referenced = _referenced_state_names(payload, admissions)
    kept: List[Any] = []
    removed: List[Dict[str, Any]] = []
    for row in states:
        if not isinstance(row, dict):
            kept.append(row)
            continue
        name = _row_name(row)
        if name and _normalize(name) not in referenced:
            removed.append(
                {
                    "bucket": "biological_states",
                    "name": name,
                    "reason": "state_unreferenced_after_quarantine",
                }
            )
            continue
        kept.append(row)
    if removed:
        payload["biological_states"] = kept
    return removed


def _revalidate_surviving_processes(
    payload: Mapping[str, Any],
    admissions: List[Dict[str, Any]],
    registry: _Registry,
    *,
    strict_db: bool,
    iteration: int,
) -> List[Dict[str, Any]]:
    """Re-check every surviving process against the shrunken graph.

    This is requirement 3's "validate every process reference; repeat until no
    new orphan is produced", and it is a BACKSTOP rather than a routine step.
    Entity removal is driven entirely by what the surviving processes reference,
    so a surviving process's entities are kept by construction and this pass is
    expected to find nothing -- ``quarantined_disconnected`` should stay at zero
    on every real payload.

    It is kept, and its state is kept in the vocabulary, because that guarantee
    is a property of the *current* pruning rules rather than of the design. The
    moment any removal stops being reference-driven -- a contract rule that
    deletes an unexportable row, a future pass that prunes on something other
    than references -- the choice is between a process leaving with a recorded
    reason and a dangling reference reaching the strict gate. This pass makes it
    the former. ``tests/test_strict_quarantine.py`` exercises it directly by
    removing an entity behind closure's back.
    """

    newly_quarantined: List[Dict[str, Any]] = []
    processes = _safe_dict(payload.get("processes"))
    for record in admissions:
        if record.get("state") not in ACCEPTED_STATES:
            continue
        bucket = str(record.get("bucket") or "")
        rows = _safe_list(processes.get(bucket))
        index = int(record.get("index") or 0)
        if not (0 <= index < len(rows)) or not isinstance(rows[index], dict):
            record["state"] = QUARANTINED_DISCONNECTED
            record["reason"] = "process_row_vanished_during_closure"
            record["iteration"] = iteration
            newly_quarantined.append(record)
            continue
        process = rows[index]
        verdict = _essential_participant_verdict(
            bucket, process, registry, strict_db=strict_db
        ) or _actor_verdict(bucket, process, registry, strict_db=strict_db)
        if verdict is None:
            continue
        _state, reason, participant = verdict
        record["state"] = QUARANTINED_DISCONNECTED
        record["reason"] = f"stranded_by_closure:{reason}"
        record["essential_participant"] = participant
        record["iteration"] = iteration
        record["detail"] = scrub_detail(
            {
                "row": row_digest(process, pointer=str(record.get("pointer") or "")),
                "participant": participant,
            }
        )
        newly_quarantined.append(record)
    return newly_quarantined


def _drop_quarantined_processes(
    payload: Dict[str, Any], admissions: Sequence[Mapping[str, Any]]
) -> None:
    """Rewrite the process buckets to hold only the accepted rows.

    Indices in the admission records address the ORIGINAL lists, so this runs
    exactly once, at the end, after every state is final.
    """

    processes = payload.get("processes")
    if not isinstance(processes, dict):
        return
    keep_by_bucket: Dict[str, Set[int]] = {}
    for record in admissions:
        if record.get("state") in ACCEPTED_STATES:
            keep_by_bucket.setdefault(str(record.get("bucket") or ""), set()).add(
                int(record.get("index") or 0)
            )
    for bucket in _PROCESS_KIND_BY_BUCKET:
        rows = processes.get(bucket)
        if not isinstance(rows, list):
            continue
        keep = keep_by_bucket.get(bucket, set())
        processes[bucket] = [row for index, row in enumerate(rows) if index in keep]


def _entity_type_overlaps(payload: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Names declared in more than one graph bucket.

    Requirement 6's "no entity type overlaps" gate, reported rather than
    repaired: closure cannot pick which declaration is the real one, and guessing
    would be exactly the silent edit this module exists to avoid.
    """

    entities = _safe_dict(payload.get("entities"))
    seen: Dict[str, str] = {}
    overlaps: List[Dict[str, str]] = []
    for bucket in _GRAPH_ENTITY_BUCKETS:
        for row in _safe_list(entities.get(bucket)):
            if not isinstance(row, dict):
                continue
            name = _row_name(row)
            norm = _normalize(name)
            if not norm:
                continue
            previous = seen.get(norm)
            if previous is None:
                seen[norm] = bucket
            elif previous != bucket:
                overlaps.append({"name": name, "buckets": f"{previous}+{bucket}"})
    return overlaps


def _unexportable_entities(
    payload: Mapping[str, Any], registry: _Registry, *, strict_db: bool
) -> List[Dict[str, str]]:
    """Surviving entity rows the required-field contract will reject.

    Closure removes what nothing references and admission quarantines the
    processes that reach an unrepresentable entity, so between them almost every
    such row is gone by now. What is left is the case neither can fix: a row kept
    alive as a *component* of a surviving complex. The contract walks every
    declared protein and demands name, species and an accession from each
    (``ir.py``:2206-2231) whether or not a process references it, so a subunit
    without species fails the export while nothing upstream flagged it.

    Reported and refused rather than removed. Deleting a subunit silently
    rewrites what the complex IS, which is the same class of edit as deleting a
    reaction participant -- the thing this module exists to stop.
    """

    entities = _safe_dict(payload.get("entities"))
    out: List[Dict[str, str]] = []
    for bucket in _GRAPH_ENTITY_BUCKETS:
        for row in _safe_list(entities.get(bucket)):
            if not isinstance(row, dict):
                continue
            ok, reason = _entity_representability(bucket, row, registry, strict_db=strict_db)
            if not ok:
                out.append({"bucket": bucket, "name": _row_name(row) or "<unnamed>", "reason": reason})
    return out


def _lock_id(row: Any) -> str:
    """The locked-reaction id on a reaction row or a quarantine record."""

    if not isinstance(row, dict):
        return ""
    value = row.get("locked_reaction_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _reconcile_locked_reactions(
    payload: Dict[str, Any],
    admissions: Sequence[Mapping[str, Any]],
    originals: Mapping[Tuple[str, int], Any],
) -> Dict[str, Any]:
    """Fold quarantined locked reactions into the canonical lock accounting.

    A locked reaction is one the run promised to preserve. Quarantining one is
    permitted -- an unexportable locked reaction is still unexportable -- but
    letting it vanish while ``locked_reaction_filter_report`` goes on reporting it
    under ``exported_locked_reactions`` is not: that report is what
    ``run_strict_post_normalization_gates`` reads (:4510-4546) to decide whether
    every lock is accounted for, and a stale count makes the gate agree that a
    reaction which is no longer in the payload was exported.

    So the report is recomputed from the two sets that exist after closure --
    active locked ids and quarantined locked ids -- in the same shape and by the
    same rule ``dedupe_processes`` uses (:4061-4085). ``locked_reactions_found``
    only ever grows, so a lock this stage never saw still shows up as
    ``unaccounted`` rather than being quietly forgotten.
    """

    processes = _safe_dict(payload.get("processes"))
    active_ids = {
        _lock_id(row)
        for row in _safe_list(processes.get("reactions"))
        if _lock_id(row)
    }

    existing_records = _safe_list(payload.get("quarantined_locked_reactions"))
    known = {_lock_id(record) for record in existing_records if _lock_id(record)}
    new_records: List[Dict[str, Any]] = []
    for record in admissions:
        if record.get("state") not in QUARANTINE_STATES:
            continue
        original = originals.get(
            (str(record.get("bucket") or ""), int(record.get("index") or 0))
        )
        locked_id = _lock_id(original)
        if not locked_id or locked_id in known:
            continue
        known.add(locked_id)
        new_records.append(
            {
                "locked_reaction_id": locked_id,
                "reaction_name": str(record.get("name") or "<unnamed>"),
                # The same two fields the pipeline's own quarantine records carry,
                # so a reader does not have to know which stage wrote the row.
                "reason": str(record.get("reason") or record.get("state") or ""),
                "missing_entities": (
                    [str(record.get("essential_participant"))]
                    if record.get("essential_participant")
                    else []
                ),
                "original_reaction": original,
                "quarantine_stage": "pre_export_strict_quarantine",
                "admission_state": str(record.get("state") or ""),
            }
        )

    all_records = [*existing_records, *new_records]
    quarantined_ids = {_lock_id(record) for record in all_records if _lock_id(record)}

    prior = _safe_dict(payload.get("locked_reaction_filter_report"))
    had_accounting = bool(prior) or bool(all_records) or bool(active_ids)
    if not had_accounting:
        return {"enabled": False, "locked_reactions_found": 0, "unaccounted_locked_reactions": 0}

    if all_records or isinstance(payload.get("quarantined_locked_reactions"), list):
        payload["quarantined_locked_reactions"] = all_records

    locked_found = max(
        int(prior.get("locked_reactions_found") or 0),
        len(active_ids | quarantined_ids),
    )
    unaccounted = max(0, locked_found - len(active_ids | quarantined_ids))
    report = {
        "locked_reactions_found": locked_found,
        "exported_locked_reactions": len(active_ids),
        "quarantined_locked_reactions": len(quarantined_ids),
        "unaccounted_locked_reactions": unaccounted,
    }
    if locked_found:
        payload["locked_reaction_filter_report"] = report
    return {
        "enabled": True,
        "newly_quarantined": [record["locked_reaction_id"] for record in new_records],
        **report,
    }


def _degree_zero_exports(
    payload: Mapping[str, Any],
    admissions: Sequence[Mapping[str, Any]],
    *,
    process_snapshot: Mapping[str, Any],
) -> List[Dict[str, str]]:
    """Surviving proteins/complexes with no surviving reference.

    Components of a surviving complex are exempt -- they have no edge by
    construction, and requirement 3 says to preserve them.

    Two sources, deliberately. Entity rows come from ``payload`` -- the post-drop
    graph, the rows an export would carry. REFERENCES come from
    ``process_snapshot``, the buckets as they were before
    :func:`_drop_quarantined_processes` compacted them, because admission indices
    address the original lists and resolving them against the compacted ones
    silently drops surviving reactions and reports their enzymes as degree-zero.
    ``strict=True``: every accepted record resolves against that snapshot by
    construction, so one that does not makes this answer wrong, not incomplete.
    """

    reference = {**payload, "processes": process_snapshot}
    _surviving_processes(reference, admissions, strict=True)
    referenced = _referenced_entity_norms(reference, admissions)
    surviving_complex_norms = {
        _normalize(_row_name(row))
        for row in _safe_list(_safe_dict(payload.get("entities")).get("protein_complexes"))
        if isinstance(row, dict) and _normalize(_row_name(row)) in referenced
    }
    exempt = _complex_component_norms(payload, surviving_complex_norms)
    out: List[Dict[str, str]] = []
    for bucket in _DEGREE_ZERO_BUCKETS:
        for row in _safe_list(_safe_dict(payload.get("entities")).get(bucket)):
            if not isinstance(row, dict):
                continue
            name = _row_name(row)
            norm = _normalize(name)
            if not norm or norm in referenced or norm in exempt:
                continue
            out.append({"bucket": bucket, "name": name})
    return out


# ── Result ──────────────────────────────────────────────────────────────────


@dataclass
class StrictQuarantineResult:
    """The closed payload plus every artifact explaining how it got that way."""

    payload: Dict[str, Any]
    quarantine_report: Dict[str, Any]
    removed_entity_report: Dict[str, Any]
    closure_report: Dict[str, Any]
    coverage: Dict[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.quarantine_report.get("ok"))

    @property
    def admissions(self) -> List[Dict[str, Any]]:
        return _safe_list(self.quarantine_report.get("admissions"))

    def states(self) -> Dict[str, str]:
        """``process name -> admission state``, for tests and the UI panel."""

        return {
            str(row.get("name") or row.get("pointer")): str(row.get("state"))
            for row in self.admissions
        }

    @property
    def refusal_reasons(self) -> List[str]:
        """Why this refused, one entry per distinct failure. Empty when ``ok``."""

        return _safe_list(self.quarantine_report.get("refusal_reasons"))

    def quarantined_names(self, state: Optional[str] = None) -> List[str]:
        return [
            str(row.get("name") or "")
            for row in self.admissions
            if row.get("state") in QUARANTINE_STATES
            and (state is None or row.get("state") == state)
        ]


def quarantine_and_close(
    payload: Mapping[str, Any],
    *,
    strict_db: bool = True,
    requested_core: Optional[Sequence[str]] = None,
    pathway_context: Optional[Mapping[str, Any]] = None,
    confidence_floor: Optional[float] = None,
    min_core_processes: int = DEFAULT_MIN_CORE_PROCESSES,
    min_core_coverage: float = DEFAULT_MIN_CORE_COVERAGE,
    max_iterations: int = DEFAULT_MAX_CLOSURE_ITERATIONS,
    mode: ExportMode = DEFAULT_EXPORT_MODE,
) -> StrictQuarantineResult:
    """Quarantine unexportable processes, then close the graph to a fixpoint.

    The input is never mutated: everything happens on a deep copy, and the caller
    gets the reduced payload back only alongside the record of what changed. That
    is what makes this transactional -- a caller that does not like the coverage
    verdict still holds the original.

    ``mode="research"`` runs every decision and **applies none of them**. Research
    mode exists for a human reading a novel multi-paper pathway, and its whole
    contract is that findings annotate rather than block (``export_mode.py``). A
    destructive quarantine there would delete the exact rows that mode was built
    to keep -- a genuinely new enzyme has no accession yet, so it is
    ``quarantined_unmapped_entity`` by construction -- and would hand the reviewer
    a smaller strict graph in place of the candidate they asked for. So the
    payload comes back untouched, ``ok`` is True regardless of coverage, and the
    decisions are available as review flags via :func:`quarantine_review_flags`.

    ``pathway_context`` is the Stage-0 context, and production must pass it. The
    final mapped payload that reaches this stage routinely carries no ``metadata``
    and no ``pathway_context`` of its own -- Stage 6 rebuilds rows from field
    whitelists that do not include either -- so payload-only discovery finds
    nothing and the coverage check silently degrades to the undeclared-core
    regime, where an unrelated survivor passes. See
    :func:`collect_requested_core_terms`.

    ``strict_db=False`` drops the protein-identity requirement, matching
    ``validate_required_pwml_contract``'s own switch, so a DB-less run does not
    quarantine every enzyme in the pathway.

    **A coverage shortfall is not a refusal** (D-002, PRODUCT_CONTRACT 7, both
    LOCKED). ``ok`` used to be "no reason of any kind fired", which made a
    surviving core that was merely *smaller than the request* end the run with no
    PWML at all -- the "valid pathway core suppressed because optional peripheral
    material is unresolved" that PRODUCT_CONTRACT 1 lists as an unacceptable
    terminal blocker. It now means "this graph may be frozen": the five
    structural reasons still refuse exactly as before, and a subthreshold graph
    that is structurally valid, internally connected and serializable without
    guessing is classified ``review_required`` on ``quarantine_report["release"]``
    -- exported, flagged, never counted as strict success -- with the shortfall
    recorded under ``review_reasons``. An EMPTY graph is not a shortfall and is
    still refused. Nothing here admits content or raises coverage; the threshold
    value is untouched and no argument can move it.
    """

    working: Dict[str, Any] = deepcopy(dict(payload))
    registry = _build_registry(working)
    core_terms = collect_requested_core_terms(
        working, requested_core=requested_core, pathway_context=pathway_context
    )

    admissions = _admit_processes(
        working,
        registry,
        strict_db=strict_db,
        core_terms=core_terms,
        confidence_floor=confidence_floor,
    )
    # The audit artifact keeps the row itself, not a digest: requirement 2 says
    # the original process is retained, and a reviewer restoring one needs the
    # participants back, not a summary of them.
    originals: Dict[Tuple[str, int], Any] = {}
    processes_in = _safe_dict(working.get("processes"))
    for record in admissions:
        bucket = str(record.get("bucket") or "")
        index = int(record.get("index") or 0)
        rows = _safe_list(processes_in.get(bucket))
        if 0 <= index < len(rows):
            originals[(bucket, index)] = deepcopy(rows[index])

    iterations: List[Dict[str, Any]] = []
    all_removed_entities: List[Dict[str, Any]] = []
    all_removed_locations: List[Dict[str, Any]] = []
    all_removed_states: List[Dict[str, Any]] = []
    converged = False

    for iteration in range(1, int(max_iterations) + 1):
        referenced = _referenced_entity_norms(working, admissions)
        surviving_complex_norms = {
            _normalize(_row_name(row))
            for row in _safe_list(_safe_dict(working.get("entities")).get("protein_complexes"))
            if isinstance(row, dict) and _normalize(_row_name(row)) in referenced
        }
        keep_norms = referenced | _complex_component_norms(working, surviving_complex_norms)

        removed_entities = _prune_entities(working, keep_norms)
        registry = _build_registry(working)
        removed_locations = _prune_locations(working, set(registry.norms))
        newly_quarantined = _revalidate_surviving_processes(
            working, admissions, registry, strict_db=strict_db, iteration=iteration
        )
        removed_states = _prune_biological_states(working, admissions)

        changed = bool(removed_entities or removed_locations or removed_states or newly_quarantined)
        all_removed_entities.extend(
            {**row, "iteration": iteration} for row in removed_entities
        )
        all_removed_locations.extend(
            {**row, "iteration": iteration} for row in removed_locations
        )
        all_removed_states.extend({**row, "iteration": iteration} for row in removed_states)
        iterations.append(
            {
                "iteration": iteration,
                "changed": changed,
                "removed_entities": removed_entities,
                "removed_locations": removed_locations,
                "removed_biological_states": removed_states,
                "quarantined_processes": [
                    {
                        "pointer": row.get("pointer"),
                        "name": row.get("name"),
                        "state": row.get("state"),
                        "reason": row.get("reason"),
                    }
                    for row in newly_quarantined
                ],
            }
        )
        if not changed:
            converged = True
            break

    if not converged:
        logger.warning(
            "quarantine_and_close: graph closure did not converge in %d iterations",
            int(max_iterations),
        )

    # The buckets as closure left them, taken BEFORE the compaction that
    # invalidates every admission index. Deep-copied so the compaction cannot
    # reach it, never written to after: it is the only surviving record of which
    # row each accepted record was written for.
    process_snapshot: Dict[str, Any] = deepcopy(_safe_dict(working.get("processes")))
    _drop_quarantined_processes(working, admissions)
    lock_accounting = _reconcile_locked_reactions(working, admissions, originals)

    coverage = evaluate_core_coverage(
        working,
        admissions,
        requested_core=requested_core,
        pathway_context=pathway_context,
        min_core_processes=min_core_processes,
        min_core_coverage=min_core_coverage,
    )

    registry = _build_registry(working)
    overlaps = _entity_type_overlaps(working)
    degree_zero = _degree_zero_exports(working, admissions, process_snapshot=process_snapshot)
    unexportable = _unexportable_entities(working, registry, strict_db=strict_db)
    unaccounted_locks = int(lock_accounting.get("unaccounted_locked_reactions") or 0)
    invariants = {
        "entity_type_overlaps": overlaps,
        "degree_zero_exports": degree_zero,
        "unexportable_entities": unexportable,
        "unaccounted_locked_reactions": unaccounted_locks,
        "closure_converged": converged,
        "ok": not overlaps
        and not degree_zero
        and not unexportable
        and not unaccounted_locks
        and converged,
    }

    # One reason per distinct failure, never a single "quarantine failed". A run
    # refused for empty coverage and a run refused for a stranded lock need
    # different fixes, and a caller that renders only "issues: 3" tells the
    # reviewer nothing about which.
    #
    # D-002 and PRODUCT_CONTRACT 7 (both LOCKED) split these six into two kinds,
    # and this is the only place the split is drawn.
    #
    # Five of them say the graph is WRONG or UNSERIALIZABLE: a type
    # contradiction, an export with no connectivity, an entity that cannot be
    # written without inventing it, a stranded lock, a closure that never
    # settled. Every one of those still refuses, unchanged and untouched.
    #
    # The sixth says only that the surviving core is SMALLER than the request. A
    # shortfall in SIZE is not a defect in CORRECTNESS, and the locked decision
    # is that "the threshold blocks release-ready status, not PWML production" --
    # so a subthreshold but structurally valid, internally connected,
    # serializable-without-guessing graph stops being refused here and is
    # recorded as a REVIEW reason instead. Nothing is admitted to make that
    # happen: ``working``, ``admissions`` and ``coverage`` are all already final
    # above this line and no code below touches them, so the exported graph for
    # any payload is byte for byte the graph the strict rules produced. The
    # threshold value does not move and nothing here can move it.
    #
    # The one coverage reason that still REFUSES is the one that is not a
    # shortfall at all: nothing survived. An empty graph has no defensible
    # connected core to review, so it stays diagnostic_only -- exactly the
    # distinction ``release_status.py`` already draws between
    # ``COVERAGE_REASON_EMPTY`` and ``COVERAGE_REASON_BELOW_MINIMUM``. That test
    # is consumed from there (``has_surviving_core``) rather than re-derived, so
    # this seam and the classifier can never disagree about what "defensible"
    # means.
    #
    # Function-local for the same reason ``evaluate_core_coverage``'s is: this
    # module is imported by the pipeline at large and the classification
    # vocabulary is needed at this seam only.
    from t2pw.pipeline.release_status import (
        classify_release_status,
        coverage_verdict,
        semantic_verdict,
    )

    coverage_reasons = [
        f"minimum_core:{reason}" for reason in _safe_list(coverage.get("reasons"))
    ]
    structural_reasons: List[str] = []
    if overlaps:
        structural_reasons.append(f"entity_type_overlap:{len(overlaps)}")
    if degree_zero:
        structural_reasons.append(f"degree_zero_export:{len(degree_zero)}")
    if unexportable:
        structural_reasons.append(f"unexportable_entity:{len(unexportable)}")
    if unaccounted_locks:
        structural_reasons.append(f"unaccounted_locked_reactions:{unaccounted_locks}")
    if not converged:
        structural_reasons.append(f"closure_not_converged:{int(max_iterations)}")

    verdict = coverage_verdict(coverage)
    defensible_core = bool(verdict is not None and verdict.has_surviving_core)
    # ``refusal_reasons`` keeps meaning exactly what its name and its docstring
    # say -- why this REFUSED -- so ``ok`` and it stay in exact agreement. A
    # reason that no longer blocks is therefore not silently left in it: it moves
    # to ``review_reasons``, same string, same class prefix, recorded either way.
    review_reasons: List[str] = coverage_reasons if defensible_core else []
    refusal_reasons: List[str] = (
        [] if defensible_core else list(coverage_reasons)
    ) + structural_reasons
    # What the STRICT rules would have refused before D-002 was applied, in the
    # pre-D-002 order. Research mode reports exactly this and is otherwise
    # untouched by any of the above.
    would_have_refused: List[str] = coverage_reasons + structural_reasons

    # D-002's required record, built through C-041's factory rather than beside
    # it: the invariant ``strict_acceptance_eligible == (status ==
    # release_ready)`` lives only in that factory (FINDINGS M-8), so going
    # through it is the only way this seam cannot contradict it. ``unexportable``
    # is passed as the SERIALIZATION input rather than folded into the technical
    # gates because that is precisely what it means -- the entity cannot be
    # written without inventing it -- and it makes the recorded reason say so.
    quarantined_reason_counts: Dict[str, int] = {}
    for record in admissions:
        if record.get("state") in QUARANTINE_STATES:
            key = str(record.get("state") or "quarantined")
            quarantined_reason_counts[key] = quarantined_reason_counts.get(key, 0) + 1
    # What retrieval this seam can actually SEE. No retrieval-round counter
    # reaches here -- the gap and audit round counts live in
    # ``extraction_diagnostics`` and are never written onto the payload -- so
    # what is recorded is the retrieval provenance carried by the surviving rows,
    # and it is labelled by its source so a reader can never read it as a round
    # count.
    retrieval_rows = 0
    for _group in ("entities", "processes"):
        for _rows in _safe_dict(working.get(_group)).values():
            for _row in _safe_list(_rows):
                if isinstance(_row, dict) and _row.get("rag_provenance"):
                    retrieval_rows += 1
    unmatched = _safe_list(coverage.get("unmatched_terms"))
    expansion_blocked_parts: List[str] = []
    if unmatched:
        expansion_blocked_parts.append(
            f"{len(unmatched)} requested-core anchor(s) matched no admitted process: "
            + ", ".join(str(term) for term in unmatched[:12])
        )
    if quarantined_reason_counts:
        expansion_blocked_parts.append(
            "candidate processes withheld by strict admission ("
            + ", ".join(
                f"{state}:{count}"
                for state, count in sorted(quarantined_reason_counts.items())
            )
            + "); admitting them would require unsupported biology"
        )
    if not expansion_blocked_parts:
        expansion_blocked_parts.append(
            "no further supported content remained at the freeze seam"
        )
    # ── Semantic evaluation reaches the runtime release_status ───────────────
    # PRODUCT_CONTRACT 11: "Semantic checks must affect the runtime
    # ``release_status``. Wiring them only into benchmark denominators is
    # insufficient." Until here nothing in ``src/`` produced ``SEMANTIC_FAILED``
    # and ``semantic_evaluation`` was hardcoded, so a run could ship
    # ``release_ready`` having never been semantically evaluated at all.
    #
    # LAYERING INVERSION, authorized narrowly by D-039 section 6 and NOT to be
    # "fixed" by restructuring packages. This is the FIRST ``t2pw.pipeline`` ->
    # ``t2pw.bench`` import in the codebase (measured: zero before it) and it
    # inverts the layering ``bench/__init__.py`` declares, where bench sits above
    # every layer and may import anything. It is authorized because the semantic
    # vocabulary is needed at THIS seam only, and it is function-local for exactly
    # the reason the ``release_status`` import above it is. It is not a cycle: the
    # forward chain was traced and nothing under ``bench/`` or ``rag/`` imports
    # ``strict_quarantine``.
    from t2pw.bench.semantic_production import evaluate_production_semantics

    # PINNED DERIVATION (D-042 section 2). The request is read through
    # ``entity_admission.pathway_context_from_stage_zero`` and through nothing
    # else. This matters more than it looks: the semantic verdict is a property of
    # HOW the request is derived, not only of the payload. That factory is the
    # codebase's single-sourced derivation (``pathway_name`` /
    # ``likely_organism``|``organism``), it is already ``t2pw.pipeline`` so it
    # costs no cross-layer import, and reusing it is what stops this seam, the
    # eligibility screen and the admission gate reading the same run's request
    # three different ways. ``metadata.pathway_subject`` is NOT a legitimate
    # source -- it is a PathWhiz CATEGORY ("Metabolic"), not a pathway name, and
    # deriving anchors from it fails CHECK_ANCHORS on essentially every payload.
    # ``tests/test_semantic_release_gating.py`` locks this choice.
    from t2pw.pipeline.entity_admission import pathway_context_from_stage_zero

    requested = pathway_context_from_stage_zero(pathway_context)
    # ``working`` is the reduced strict graph and is already final here -- the same
    # object the coverage verdict above was computed from -- so the semantic
    # verdict and the coverage verdict describe one graph, never two. No
    # ``admission`` report exists at this seam (D-042 section 4), so
    # CHECK_RAG_REINTRODUCTION is structurally inapplicable and reports itself as
    # not evaluated rather than as a pass.
    semantic_report = evaluate_production_semantics(
        working,
        requested_pathway=requested.requested_pathway,
        requested_organism=requested.organism,
        mode=coerce_mode(mode),
        min_connected_reactions=min_core_processes,
    )
    # ``semantic_evaluable`` is F-053's carrier (C-056c): per gating check, whether
    # it was applicable here and, when not, the reason the check itself gave. At
    # THIS seam that is never four of four -- the comment above says why
    # CHECK_RAG_REINTRODUCTION cannot be evaluated without an admission report --
    # so the record travels rather than the shortfall staying invisible behind a
    # bare ``passed``. Nothing below branches on it.
    semantic_state, semantic_reason, semantic_failed, semantic_evaluable = semantic_verdict(
        semantic_report
    )

    release = classify_release_status(
        coverage,
        pipeline_executed=True,
        strict_gates_passed=(
            not overlaps and not degree_zero and not unaccounted_locks and converged
        ),
        serializable_without_invention=not unexportable,
        retrieval_attempts=retrieval_rows,
        expansion_blocked_reason="; ".join(expansion_blocked_parts),
        semantic_evaluation=semantic_state,
        semantic_not_evaluated_reason=semantic_reason,
        semantic_failed_checks=semantic_failed,
        semantic_check_evaluability=semantic_evaluable,
    )

    # ── Research mode: decide, then apply nothing ────────────────────────────
    # Everything above ran, so the flags are real findings against the real
    # payload. What changes here is only whether they are acted on. The reduced
    # graph is discarded and the caller gets its candidate back byte for byte;
    # refusal reasons become review flags rather than a block.
    research = is_research(mode)
    if research:
        research_flags = [
            {
                "pointer": record.get("pointer"),
                "name": record.get("name"),
                "process_kind": record.get("process_kind"),
                "state": record.get("state"),
                "reason": record.get("reason"),
                "essential_participant": record.get("essential_participant"),
                "research_mode": "review_flag",
                "effect": "annotate_only",
            }
            for record in admissions
            if record.get("state") in QUARANTINE_STATES
        ]
        working = deepcopy(dict(payload))
        all_removed_entities = []
        all_removed_locations = []
        all_removed_states = []
    else:
        research_flags = []

    counts = {state: 0 for state in ADMISSION_STATES}
    for record in admissions:
        state = str(record.get("state") or "")
        if state in counts:
            counts[state] += 1

    quarantined_records: List[Dict[str, Any]] = []
    accepted_records: List[Dict[str, Any]] = []
    for record in admissions:
        if record.get("state") in QUARANTINE_STATES:
            quarantined_records.append(
                {
                    **record,
                    "original_process": originals.get(
                        (str(record.get("bucket") or ""), int(record.get("index") or 0))
                    ),
                }
            )
        else:
            accepted_records.append(dict(record))

    decision_inputs = canonical_decision_inputs(
        mode=mode,
        strict_db=strict_db,
        requested_core=requested_core,
        pathway_context=pathway_context,
        confidence_floor=confidence_floor,
        min_core_processes=min_core_processes,
        min_core_coverage=min_core_coverage,
        max_iterations=max_iterations,
    )
    quarantine_report = {
        # 4: a coverage shortfall stopped being a refusal (D-002) and the report
        # grew ``review_reasons`` and the ``release`` classification beside the
        # ``ok`` it can no longer be read off. Additive: every schema-3 key keeps
        # its name, its type and its meaning.
        #
        # 5 (C-056b, D-039 section 5): ``release`` grew ONE key,
        # ``semantic_failed_checks`` -- a list, empty unless
        # ``semantic_evaluation == "failed"``. Additive in exactly the same sense:
        # no schema-4 key changed name, type or meaning, and ``decision_id``
        # (``decision_identifier``, hashing only ``admitted_payload_hash`` +
        # ``decision_input_hash``) does not move. The version still bumps, because
        # the house rule that produced 4 for an additive change is what lets a
        # reader trust the number at all.
        #
        # 6 (C-056c, F-053 / D-054 section 8): ``release`` grew ONE key,
        # ``semantic_check_evaluability`` -- a list of one record per gating
        # check, empty only when no evaluation ran. Additive in exactly the same
        # sense again: no schema-5 key changed name, type or meaning, and
        # ``decision_id`` does not move, because it hashes only
        # ``admitted_payload_hash`` + ``decision_input_hash`` and neither is a
        # function of this key. The bump is the house rule above applied a third
        # time: an additive change still bumps, or the number stops meaning
        # anything. What it buys: ``semantic_evaluation: "passed"`` on this
        # report could not previously be told apart from a four-of-four pass,
        # and here it is never four of four.
        "schema_version": 6,
        "stage": "pre_export_strict_quarantine",
        "strict_db": bool(strict_db),
        "export_mode": coerce_mode(mode),
        "policy_version": QUARANTINE_POLICY_VERSION,
        # The payload this decision was made ABOUT, not the one it produced.
        # Reuse is gated on the caller's current payload hashing to the RESULTING
        # hash and on the inputs below hashing to decision_input_hash; see
        # decision_matches for why one decision per payload version *under one set
        # of rules*, and not one per session, is the invariant.
        "admitted_payload_hash": admitted_payload_hash(payload),
        "resulting_payload_hash": admitted_payload_hash(working),
        # Every control that can change this verdict, and its fingerprint. Both
        # are on the report: the hash is what reuse compares, the expansion is
        # what tells a reviewer *which* control moved when reuse is refused.
        "decision_inputs": decision_inputs,
        "decision_input_hash": payload_hash(decision_inputs),
        # Research mode annotates and never blocks, so the run continues; the
        # refusal reasons stay recorded below as flags rather than as a verdict.
        "ok": True if research else not refusal_reasons,
        "counts": counts,
        "admissions": [dict(record) for record in admissions],
        "quarantined": quarantined_records,
        "accepted": accepted_records,
        "strict_invariants": invariants,
        "refusal_reasons": [] if research else refusal_reasons,
        # Recorded, never blocking. Empty in research mode for the same reason
        # ``refusal_reasons`` is: research applies nothing, and its findings live
        # under ``research_mode`` where they cannot be read as a verdict.
        "review_reasons": [] if research else review_reasons,
        # The D-002 classification, so no consumer has to re-derive "is there
        # anything releasable?" from ``ok``, from an exit status or from a
        # filename. ``ok`` says whether the graph may be frozen; this says what
        # the run IS -- and ``strict_acceptance_eligible`` is False for every
        # review_required run, so a below-threshold export can never be counted
        # as strict success (TRAP-1 / PRODUCT_CONTRACT 13).
        "release": {
            **release.to_dict(),
            # Research decides and applies nothing, so there this is a FINDING,
            # exactly like ``research_mode.would_have_refused`` -- not a verdict
            # anything acted on.
            "applied": not research,
            "review_reasons": list(review_reasons),
            # Named by its source. This is NOT a retrieval-round counter; see
            # where it is computed.
            "retrieval_attempts_source": "surviving_rows_carrying_rag_provenance",
        },
        "locked_reactions": lock_accounting,
        "coverage": coverage,
        "closure": {
            "iterations": len(iterations),
            "converged": converged,
            "max_iterations": int(max_iterations),
        },
    }
    quarantine_report["decision_id"] = decision_identifier(quarantine_report)
    if research:
        quarantine_report["research_mode"] = {
            "note": RESEARCH_DIAGNOSTIC_NOTE,
            "applied": False,
            "review_flags": research_flags,
            "would_have_quarantined": len(research_flags),
            # The pre-D-002 strict list, in the pre-D-002 order. Research mode's
            # observable contract is unchanged by this card: it reports what the
            # strict rules would have refused, including the coverage shortfall
            # that no longer refuses a strict run.
            "would_have_refused": would_have_refused,
        }

    removed_entity_report = {
        "schema_version": 1,
        "stage": "pre_export_strict_quarantine",
        "removed_entities": all_removed_entities,
        "removed_locations": all_removed_locations,
        "removed_biological_states": all_removed_states,
        "counts": {
            "entities": len(all_removed_entities),
            "locations": len(all_removed_locations),
            "biological_states": len(all_removed_states),
        },
    }

    closure_report = {
        "schema_version": 1,
        "stage": "pre_export_strict_quarantine",
        "converged": converged,
        "stable_after": len(iterations) if converged else None,
        "iteration_count": len(iterations),
        "max_iterations": int(max_iterations),
        "iterations": iterations,
    }

    return StrictQuarantineResult(
        payload=working,
        quarantine_report=quarantine_report,
        removed_entity_report=removed_entity_report,
        closure_report=closure_report,
        coverage=coverage,
    )


def quarantine_review_flags(report: Any) -> List[Dict[str, Any]]:
    """Research-mode review flags from a report, or ``[]`` for a strict one."""

    if not isinstance(report, Mapping):
        return []
    return [
        dict(flag)
        for flag in _safe_list(_safe_dict(report.get("research_mode")).get("review_flags"))
        if isinstance(flag, dict)
    ]


def write_quarantine_artifacts(
    result: StrictQuarantineResult,
    out_dir: Path | str,
) -> Dict[str, str]:
    """Persist the four artifacts and return ``name -> path``.

    All four are always written, including when nothing was quarantined: an
    absent ``quarantine_report.json`` is indistinguishable from a run that never
    reached this stage, and "nothing was dropped" is the answer a reviewer most
    often needs recorded.

    A pre-existing set is not overwritten but **archived** first, into
    ``quarantine_history/<decision id>/``. There is one decision per payload
    version *per set of rules* and a session can legitimately produce several --
    the boundary's, then one after refinement edits the graph, then one after
    grounding rewrites identifiers -- and each is the only record of why that
    version was admitted. Overwriting would leave the artifacts describing the
    last payload while the reviewer is still asking about the one they approved.

    Keyed on the full decision identifier, not on ``admitted_payload_hash`` alone.
    A strict and a research decision over the *same* payload share that hash and
    reach opposite conclusions, so a payload-keyed archive would file the second
    on top of the first and lose the one that said "this would not export".
    """

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        QUARANTINE_REPORT_FILENAME: result.quarantine_report,
        REMOVED_ENTITY_REPORT_FILENAME: result.removed_entity_report,
        CLOSURE_REPORT_FILENAME: result.closure_report,
        COVERAGE_REPORT_FILENAME: result.coverage,
    }

    existing = directory / QUARANTINE_REPORT_FILENAME
    if existing.exists():
        try:
            prior = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # pragma: no cover - unreadable prior report
            prior = {}
        prior_id = decision_identifier(_safe_dict(prior))
        new_id = decision_identifier(result.quarantine_report)
        if prior_id and prior_id != new_id:
            archive = directory / QUARANTINE_HISTORY_DIRNAME / prior_id.replace(":", "_")
            archive.mkdir(parents=True, exist_ok=True)
            for filename in payloads:
                source = directory / filename
                if source.exists():
                    (archive / filename).write_text(
                        source.read_text(encoding="utf-8"), encoding="utf-8"
                    )

    written: Dict[str, str] = {}
    for filename, document in payloads.items():
        path = directory / filename
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        written[filename] = str(path)
    return written


def clear_quarantine_artifacts(out_dir: Path | str) -> List[str]:
    """Delete the current artifact set. Returns the paths removed.

    Called when a new pipeline run starts. A stale ``quarantine_report.json`` on
    disk outlives the session that wrote it, so a run that dies before reaching
    the boundary would otherwise display -- and a careless caller could export
    under -- the previous run's decision about a completely different pathway.
    History under ``quarantine_history/`` is left alone; that is the archive.
    """

    directory = Path(out_dir)
    removed: List[str] = []
    for filename in (
        QUARANTINE_REPORT_FILENAME,
        REMOVED_ENTITY_REPORT_FILENAME,
        CLOSURE_REPORT_FILENAME,
        COVERAGE_REPORT_FILENAME,
    ):
        path = directory / filename
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed
