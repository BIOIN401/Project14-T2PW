"""C-060 -- the assay-reagent / assay-reporter admission gate at the Stage-2 merge.

R-003 found four mechanisms behind nine PRODUCT_CONTRACT violations. This module
owns exactly one of them, **M1**: an assay reagent or reporter retained as a
pathway member, including a *synthesized assay-composite reaction*. PRODUCT_CONTRACT
section 2 states the rule in terms -- "no assay reporters as pathway members".

A0-C5, THE ORDERING, IS THE POINT
---------------------------------
Two phases, in this order, and the order is load-bearing rather than cosmetic:

1. **The hallucination gate runs FIRST and INDEPENDENTLY.** A row whose evidence
   span is not locatable in the seed paper text and is not carried by an
   **admitted** RAG record is removed *before any* ``cofactor_policy``
   consultation. Nothing in :func:`_screen_hallucinations` reads a verdict, so a
   ``participant`` verdict can never protect a row here.

   This is measured, not theoretical. R-003 executed the merged C-018 classifier
   against the run's entities in their own gold contexts and ``succinyl-CoA`` in
   heme biosynthesis comes back ``participant`` / **high** -- the single worst
   fabrication in the run (zero occurrences of "succinyl" in the 67,304-character
   source). Consulting the classifier first would have PROTECTED that row.

2. **Only then**, for survivors, :func:`classify_entity` is consulted
   **advisorily** to separate pathway membership from a cofactor/currency role.

What this gate may do
---------------------
**It only ever REMOVES or DEMOTES.** It never adds, never repairs, never renames,
never invents. Every surviving row is byte-identical to the row it was handed;
demotion is recorded in the ledger and mutates no row, precisely so that property
holds unconditionally (merge rule 6: it cannot weaken a biological gate to
increase PWML production, because it can only subtract).

It runs inside ``pipeline.merge_additions``, **pre-freeze** at the Stage-2 merge
and before ``apply_post_merge_cleanup`` -- so it is not an exporter repairing
biology after the canonical graph is frozen (merge rule 8).

Every removal and demotion is auditable in the ledger. When the gate's own
removals empty the reaction set, the payload is **flagged**, never dropped:
``review_required`` plus a stated reason, with every surviving row still present
(merge rule 7).

Deliberate non-goals
--------------------
* **It never touches a protein row.** ``placeholder_backed_proteins`` is standing
  open question O-1; the gate's removals are confined to ``entities.compounds``
  and ``processes.*``, so it cannot reach protein export policy.
* **It never reads a RAG report's ``rejected`` list.** This is a POSITIVE
  grounding rule -- locatable in the seed text, or carried by an ADMITTED record.
  Section-10 rejected-claim enforcement already exists in ``rag/graph_delta.py``
  (``RULE_REJECTED_CLAIM``, C-021) and is C-055's to wire, not C-060's to rebuild.
* **It fails open.** No seed text means the hallucination rule cannot be
  evaluated, and "not evaluated" is never "false" (PRODUCT_CONTRACT section 8).
  No requested pathway means the advisory phase abstains: ``unknown`` is not a
  rejection, by C-018's own docstring.

Pure, offline, deterministic: no LLM, no network, no database, no clock, no I/O.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from t2pw.pipeline.cofactor_policy import (
    ASSAY_REPORTER,
    COFACTOR_OR_CURRENCY,
    VOCABULARY,
    PathwayContext,
    classify_entity,
)

__all__ = [
    "LEDGER_KEY", "SCHEMA_VERSION", "PHASE_HALLUCINATION", "PHASE_ADVISORY",
    "RULE_REAGENT_DUPLICATE", "RULE_UNLOCATABLE_EVIDENCE", "RULE_ASSAY_COMPOSITE",
    "RULE_REFERENCES_REMOVED_ROW", "RULE_CURRENCY_NOT_SUBJECT", "RULE_REPORTER_NOT_MEMBER",
    "REASON_NO_SURVIVING_REACTION", "COFACTOR_OR_CURRENCY", "ASSAY_REPORTER",
    "PathwayContext", "classify_entity", "screen_additions", "carry_forward",
]

#: Where :func:`pipeline.merge_additions` attaches the ledger. MEASURED, not
#: assumed: the key survives ``map_ids.map_payload`` and ``quarantine_and_close``
#: (both deep-copy) and sits inside the ``deepcopy`` that
#: ``freeze_canonical_payload`` hashes, so it reaches ``final.canonical.json``
#: and **participates in** ``canonical_payload_hash`` -- the hash differs with and
#: without it. Attaching the ledger is chartered; altering that hash's composition
#: is not. Excluding it from the projection is C-030 / C-052 territory, disclosed
#: to the product owner and deliberately not done here.
LEDGER_KEY = "entity_admission_report"
SCHEMA_VERSION = 1

PHASE_HALLUCINATION = "hallucination_gate"
PHASE_ADVISORY = "cofactor_policy_advisory"

RULE_REAGENT_DUPLICATE = "reagent_form_of_a_species_already_present"
RULE_UNLOCATABLE_EVIDENCE = "evidence_span_not_locatable_and_not_admitted"
RULE_ASSAY_COMPOSITE = "assay_readout_fused_into_a_duplicate_reaction"
RULE_REFERENCES_REMOVED_ROW = "references_a_removed_row"
RULE_CURRENCY_NOT_SUBJECT = "currency_not_subject_of_the_requested_pathway"
RULE_REPORTER_NOT_MEMBER = "assay_reporter_not_pathway_member"

REASON_NO_SURVIVING_REACTION = "entity_admission_removed_the_last_reaction"

#: Reagent / preparation forms that are the SAME chemical identity as a
#: physiological species, per R-003 F1 and the gold's own reason for ``hemin``:
#: "Reagent and cofactor forms of heme used in the binding assays. The same
#: chemical identity as heme, not separate pathway metabolites." The rule fires
#: ONLY when the physiological form is also present -- the duplication is the
#: violation, not the accession -- so an unaccompanied reagent spelling is left
#: exactly where it is. Curated and deliberately small: a name this table does
#: not know is not a claim about that name.
_REAGENT_FORMS: Dict[str, str] = {
    "hemin": "heme", "haemin": "heme", "heminchloride": "heme",
    "ferricchlorideheme": "heme", "hematin": "heme", "haematin": "heme",
    "hemeb": "heme", "haemb": "heme",
}

#: Assay-readout vocabulary. A reaction is a composite only when a marker fires
#: AND it duplicates a marker-free sibling's chemistry (see :func:`_assay_marker`).
_ASSAY_MARKERS: Tuple[str, ...] = (
    "coupled assay", "coupled-assay", "coupling assay", "coupled enzymatic assay",
    "coupled spectrophotometric", "ldh-catalyzed", "ldh-catalysed", "ldh-coupled",
    "lactate dehydrogenase-coupled", "od 340", "od340", "a340", "absorbance at 340",
    "loss of od", "reporter assay", "colorimetric assay", "spectrophotometric assay",
    "continuous spectrophotometric", "fluorimetric assay", "assay readout",
)

#: Currency spellings, read from C-018's VOCABULARY *table* -- a data table, never
#: a verdict. Phase 1 uses it only to compare two reactions' non-currency cores;
#: it never calls :func:`classify_entity`, so A0-C5 holds by construction.
_CURRENCY_KEYS = frozenset(
    key for key, (_family, verdict) in VOCABULARY.items() if verdict == COFACTOR_OR_CURRENCY
)

#: Below this a span is a fragment, not a citation: too short to locate honestly.
_MIN_SPAN_CHARS = 24

_PARTICIPANT_KEYS: Tuple[str, ...] = (
    "inputs", "outputs", "reactants", "products", "substrates", "enzymes", "modifiers",
    "participants", "components", "entity_1", "entity_2", "cargo", "transporter",
)
_KIND_BY_BUCKET: Dict[str, str] = {
    "compounds": "compound", "reactions": "reaction", "interactions": "interaction",
    "transports": "transport", "regulations": "regulation",
}
_STRIP_RE = re.compile(r"[^a-z0-9]")
_SPACE_RE = re.compile(r"\s+")


def _norm(value: Any) -> str:
    """Fold a name to a comparison key -- the same fold C-018 uses."""
    return _STRIP_RE.sub("", str(value or "").casefold())


def _squash(value: Any) -> str:
    """Fold free text for containment: case and whitespace only, never punctuation."""
    return _SPACE_RE.sub(" ", str(value or "").casefold()).strip()


def _rows(section: Any, bucket: str) -> List[Any]:
    if not isinstance(section, dict):
        return []
    rows = section.get(bucket)
    return rows if isinstance(rows, list) else []


def _buckets(section: Any) -> List[str]:
    return [k for k, v in section.items() if isinstance(v, list)] if isinstance(section, dict) else []


def _participants(row: Any) -> Iterator[str]:
    if not isinstance(row, dict):
        return
    for key in _PARTICIPANT_KEYS:
        value = row.get(key)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and item.strip():
                yield item
            elif isinstance(item, dict):
                for field in ("name", "entity", "protein"):
                    candidate = item.get(field)
                    if isinstance(candidate, str) and candidate.strip():
                        yield candidate
                        break


def _spans(row: Any) -> List[str]:
    """Every evidence span a row carries, in a stable order."""
    out: List[str] = []
    if not isinstance(row, dict):
        return out
    for key in ("evidence", "source_refs", "evidence_quote"):
        value = row.get(key)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and item.strip():
                out.append(item)
            elif isinstance(item, dict):
                inner = item.get("span") or item.get("text")
                if isinstance(inner, str) and inner.strip():
                    out.append(inner)
    return out


def _admitted_spans(report: Any) -> List[str]:
    """Squashed text of every ADMITTED record. ``rejected`` is never read here:
    that list belongs to ``rag/graph_delta.py``'s section-10 rule (C-055)."""
    out: List[str] = []
    if not isinstance(report, dict):
        return out
    stack: List[Any] = [report.get("accepted")]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            squashed = _squash(node)
            if squashed:
                out.append(squashed)
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return out


def _unlocatable(row: Any, seed: str, admitted: List[str]) -> str:
    """The span that condemns *row*, or ``""`` when the rule does not fire.

    Fails open in both directions: no seed text, or no span long enough to
    locate honestly, and the row is simply not evaluated.
    """
    if not seed:
        return ""
    candidates = [s for s in _spans(row) if len(s.strip()) >= _MIN_SPAN_CHARS]
    if not candidates:
        return ""
    for span in candidates:
        squashed = _squash(span)
        if squashed in seed or any(squashed in record for record in admitted):
            return ""
    return candidates[0]


def _assay_marker(row: Any) -> str:
    haystack = _squash(" ".join([str((row or {}).get("name") or "")] + _spans(row)))
    return next((marker for marker in _ASSAY_MARKERS if marker in haystack), "")


def _core(row: Any) -> frozenset:
    """A reaction's non-currency chemistry -- what makes two rows the same claim."""
    if not isinstance(row, dict):
        return frozenset()
    names = set()
    for key in ("inputs", "outputs", "reactants", "products"):
        value = row.get(key)
        for item in value if isinstance(value, list) else []:
            key_norm = _norm(item if isinstance(item, str) else (item or {}).get("name"))
            if key_norm and key_norm not in _CURRENCY_KEYS:
                names.add(key_norm)
    return frozenset(names)


def _assay_composite(row: Any, siblings: List[Any]) -> str:
    """*row* is an assay composite when an assay marker fires **and** it restates
    a marker-free sibling's chemistry with extra species fused in. Both halves are
    required: a genuine reaction merely mentioned in a Methods sentence keeps its
    place, and a duplicate without an assay marker is the merge's own business."""
    marker = _assay_marker(row)
    if not marker:
        return ""
    core = _core(row)
    if not core:
        return ""
    for other in siblings:
        if other is row or _assay_marker(other):
            continue
        other_core = _core(other)
        if other_core and other_core <= core:
            return marker
    return ""


#: The fields that CONSTITUTE a ledger entry, and therefore its identity. There
#: is no nonce, counter or timestamp here: identity is content-derived, exactly as
#: PACK2-SHARED S10 requires of a lineage writer.
_ENTRY_FIELDS: Tuple[str, ...] = ("phase", "rule", "kind", "entity", "evidence")


def _entry(phase: str, rule: str, kind: str, entity: Any, evidence: Any) -> Dict[str, str]:
    return {
        "phase": phase, "rule": rule, "kind": kind,
        "entity": str(entity or ""), "evidence": str(evidence or ""),
    }


def _entry_key(entry: Any) -> Tuple[str, ...]:
    return tuple(str((entry or {}).get(field, "")) for field in _ENTRY_FIELDS)


def carry_forward(existing: Any, fresh: Dict[str, Any]) -> Dict[str, Any]:
    """Fold *fresh* into the ledger the payload already carries.

    ``merge_additions`` runs more than once on a RAG leg: Stage 2 first, then
    again with the conformed RAG envelope, whose merge BASE is the first merge's
    own output. Overwriting the key there erased the first pass's removals from
    the record while leaving the rows themselves gone -- a silent removal, which
    A4 calls a reject.

    Inbound entries are preserved and always come first; the ledger is never
    replaced, only extended. A fresh entry is appended unless an entry with the
    same content-derived identity is already present -- S10's discipline, and for
    S10's reason: nothing downstream dedups for us. Identity is the FULL
    constituting tuple, so the same entity removed under a different rule, in a
    different phase, or on different evidence is a genuinely distinct second fact
    and is kept. ``review_required`` is sticky: a pass that once emptied the
    reaction set stays on the record even if a later pass adds reactions back.
    """
    if not isinstance(existing, dict):
        return fresh
    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "removed": [], "demoted": [],
        "counts": {"removed": 0, "demoted": 0, "admitted": 0},
        "review_required": bool(existing.get("review_required") or fresh.get("review_required")),
        "review_reasons": [],
    }
    for bucket in ("removed", "demoted"):
        seen: set = set()
        for source in (existing.get(bucket), fresh.get(bucket)):
            for entry in source if isinstance(source, list) else []:
                key = _entry_key(entry)
                if key in seen:
                    continue
                seen.add(key)
                out[bucket].append(entry)
        out["counts"][bucket] = len(out[bucket])
    for source in (existing.get("review_reasons"), fresh.get("review_reasons")):
        for reason in source if isinstance(source, list) else []:
            if reason not in out["review_reasons"]:
                out["review_reasons"].append(reason)
    return out


def _reaction_count(payload: Any) -> int:
    return len(_rows((payload or {}).get("processes"), "reactions")) if isinstance(payload, dict) else 0


def screen_additions(
    payload: Any,
    *,
    seed_text: str = "",
    admission_report: Optional[Dict[str, Any]] = None,
    context: Optional[PathwayContext] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Screen a merged Stage-1 + Stage-2 payload. Returns ``(kept, ledger)``.

    Total: a payload that is not a dict is returned untouched with an empty
    ledger. Every branch below either drops a row or leaves it exactly as it was;
    no branch constructs, renames or repairs one.
    """
    ledger: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "removed": [], "demoted": [],
        "counts": {"removed": 0, "demoted": 0, "admitted": 0},
        "review_required": False, "review_reasons": [],
    }
    if not isinstance(payload, dict):
        return payload, ledger

    reactions_before = _reaction_count(payload)
    orphaned = _screen_hallucinations(payload, seed_text, admission_report, ledger)
    _screen_advisory(payload, context, orphaned, ledger)

    ledger["counts"]["removed"] = len(ledger["removed"])
    ledger["counts"]["demoted"] = len(ledger["demoted"])
    if reactions_before and not _reaction_count(payload) and ledger["removed"]:
        # Merge rule 7: flagged, never dropped. Every surviving row is still on
        # the payload; the reason travels with it for release_status.py to read.
        ledger["review_required"] = True
        ledger["review_reasons"] = [REASON_NO_SURVIVING_REACTION]
    return payload, ledger


def _screen_hallucinations(
    payload: Dict[str, Any],
    seed_text: str,
    admission_report: Optional[Dict[str, Any]],
    ledger: Dict[str, Any],
) -> set:
    """PHASE 1. Nothing here consults ``cofactor_policy``'s verdicts -- that is
    A0-C5, and it is why a ``participant`` verdict cannot protect a row.

    Returns the normalized names that appeared on rows this phase removed: the
    only entities phase 2 is allowed to treat as newly stranded. A pre-existing
    orphan is somebody else's finding.
    """
    seed = _squash(seed_text)
    admitted = _admitted_spans(admission_report)
    entities, processes = payload.get("entities"), payload.get("processes")

    # --- compounds only. Proteins are O-1 territory and are never touched. ---
    compounds = _rows(entities, "compounds")
    present = {_norm(row.get("name")) for row in compounds if isinstance(row, dict)}
    removed_keys: set = set()
    if compounds:
        kept: List[Any] = []
        for row in compounds:
            name = row.get("name") if isinstance(row, dict) else None
            canonical = _REAGENT_FORMS.get(_norm(name), "")
            if canonical and _norm(canonical) in present and _norm(canonical) != _norm(name):
                ledger["removed"].append(
                    _entry(PHASE_HALLUCINATION, RULE_REAGENT_DUPLICATE, "compound", name, canonical)
                )
                removed_keys.add(_norm(name))
                continue
            span = _unlocatable(row, seed, admitted)
            if span:
                ledger["removed"].append(
                    _entry(PHASE_HALLUCINATION, RULE_UNLOCATABLE_EVIDENCE, "compound", name, span)
                )
                removed_keys.add(_norm(name))
                continue
            kept.append(row)
        entities["compounds"] = kept

    # --- process rows: unlocatable evidence, assay composites, and the cascade ---
    orphaned: set = set(removed_keys)
    for bucket in _buckets(processes):
        rows = _rows(processes, bucket)
        siblings = list(rows)
        kept_rows: List[Any] = []
        for row in rows:
            kind = _KIND_BY_BUCKET.get(bucket, bucket)
            name = row.get("name") if isinstance(row, dict) else None
            stranded = next(
                (p for p in _participants(row) if _norm(p) in removed_keys), ""
            )
            span = _unlocatable(row, seed, admitted)
            marker = _assay_composite(row, siblings) if bucket == "reactions" else ""
            if stranded:
                rule, evidence, entity = RULE_REFERENCES_REMOVED_ROW, stranded, stranded
            elif span:
                rule, evidence, entity = RULE_UNLOCATABLE_EVIDENCE, span, name
            elif marker:
                rule, evidence, entity = RULE_ASSAY_COMPOSITE, marker, name
            else:
                kept_rows.append(row)
                continue
            ledger["removed"].append(_entry(PHASE_HALLUCINATION, rule, kind, entity, evidence))
            orphaned.update(_norm(p) for p in _participants(row))
        processes[bucket] = kept_rows
    return orphaned


def _screen_advisory(
    payload: Dict[str, Any],
    context: Optional[PathwayContext],
    orphaned: set,
    ledger: Dict[str, Any],
) -> None:
    """PHASE 2. Survivors only, and advisory: ``classify_entity`` is consulted to
    separate membership from a cofactor/currency role.

    A currency species that is no longer a participant of any surviving reaction
    **because this gate removed the row that carried it** is not a pathway member
    and goes. One that still participates is DEMOTED -- recorded here, and the row
    itself left untouched. ``hub`` and ``unknown`` are retained: C-018's own
    docstring says ``unknown`` is NOT a rejection.
    """
    entities = payload.get("entities")
    compounds = _rows(entities, "compounds")
    if not compounds:
        return
    ctx = context if isinstance(context, PathwayContext) else PathwayContext("")
    referenced = {
        _norm(participant)
        for bucket in _buckets(payload.get("processes"))
        for row in _rows(payload.get("processes"), bucket)
        for participant in _participants(row)
    }

    kept: List[Any] = []
    for row in compounds:
        name = row.get("name") if isinstance(row, dict) else None
        verdict = classify_entity(name, ctx)
        if verdict.verdict not in (COFACTOR_OR_CURRENCY, ASSAY_REPORTER):
            kept.append(row)
            continue
        rule = (
            RULE_CURRENCY_NOT_SUBJECT
            if verdict.verdict == COFACTOR_OR_CURRENCY
            else RULE_REPORTER_NOT_MEMBER
        )
        key = _norm(name)
        if key in referenced:
            ledger["demoted"].append(
                _entry(PHASE_ADVISORY, rule, "compound", name, verdict.reason)
            )
            kept.append(row)
        elif key in orphaned:
            ledger["removed"].append(
                _entry(PHASE_ADVISORY, rule, "compound", name, verdict.reason)
            )
        else:
            kept.append(row)  # a pre-existing degree-zero row is not this gate's finding
    entities["compounds"] = kept
