"""Additive RAG provenance field definitions (WP0).

The RAG subsystem attaches four **optional, additive** keys to entities and
processes so every synthesized element stays traceable to a source:

* ``rag_provenance`` — the source pointer backing the element.
* ``evidence``       — the retrieved passage(s) supporting it.
* ``source_papers``  — the papers/records the evidence came from.
* ``rag_confidence`` — the subsystem's confidence in the addition.

These are **definitions only**. Per the separation invariant
(docs/rag/03_separation_invariant.md) and the additive-metadata rule, these keys
are optional and are ignored by every existing stage that does not know about
them. This module intentionally does **not** edit or import ``t2pw.schema`` — the
core payload contracts stay untouched; these TypedDicts document the extra shape
RAG layers on top, exactly as ``t2pw.schema`` documents (without enforcing) the
core shapes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict


class RagProvenance(TypedDict, total=False):
    """A resolvable source pointer for a synthesized entity/process.

    ``source_id`` + ``source_uri`` are carried through from the retrieved
    ``Chunk`` that justified the element (see ``t2pw.rag.store.Chunk``).
    """

    source_id: str  # PMID / PMCID / DOI / pathbank id / reference filename
    source_title: str
    source_type: str  # "paper" | "pathbank" | "kegg" | "pwml_example"
    source_uri: str
    section: str
    organism: str
    chunk_id: str


class RagEvidence(TypedDict, total=False):
    """A single retrieved passage supporting an element."""

    text: str
    source_id: str
    source_uri: str
    section: str
    score: float
    chunk_id: str


class RagSourcePaper(TypedDict, total=False):
    """A paper or database record that contributed evidence."""

    source_id: str
    title: str
    uri: str
    source_type: str
    organism: str


# The bundle of additive keys RAG may attach to an entity or process. Every key
# is optional (``total=False``); an element with none of them is a plain core
# payload record. The four keys match docs/rag/03_separation_invariant.md.
RagAdditiveMetadata = TypedDict(
    "RagAdditiveMetadata",
    {
        "rag_provenance": RagProvenance,
        "evidence": List[RagEvidence],
        "source_papers": List[RagSourcePaper],
        "rag_confidence": float,
    },
    total=False,
)


# The exact key names RAG is permitted to add (for assertions/validation in
# later WPs). Nothing outside this set may be attached. ``rag_provenance`` is
# deliberately namespaced (not ``provenance``) so it never collides with the
# core ``PayloadCommonRecord.provenance`` string field.
RAG_ADDITIVE_KEYS = ("rag_provenance", "evidence", "source_papers", "rag_confidence")


# ---------------------------------------------------------------------------
# WP6 — Provenance validation & stripping.
# ---------------------------------------------------------------------------
# The three entity buckets whose members are synthesized *chemistry* and must
# therefore be evidence-bound. ``species`` / ``subcellular_locations`` are
# contextual scaffolding (organism / compartment), not reaction participants, so
# they are exempt — they are never emitted by RAG synthesis.
_EVIDENCE_ENTITY_BUCKETS = ("compounds", "proteins", "protein_complexes")


def _cofactor_names() -> frozenset:
    """The cofactor exemption set, reused from WP5 synthesis (single source).

    Imported lazily so this module stays import-cheap and free of any circular
    dependency (``synthesize`` imports :data:`RAG_ADDITIVE_KEYS` from here).
    Falls back to an empty set if synthesis is unavailable, in which case every
    compound is simply held to the provenance requirement (never *weakened*).
    """
    try:
        from t2pw.rag.synthesize import COFACTOR_NAMES

        return frozenset(COFACTOR_NAMES)
    except Exception:  # pragma: no cover - defensive; synthesize is a sibling
        return frozenset()


@dataclass
class ProvenanceIssue:
    """A single element that failed the evidence-bound requirement."""

    kind: str  # "reaction" | "compound" | "protein" | "protein_complex"
    label: str
    pointer: str  # JSON-pointer-ish path into the payload
    reason: str


@dataclass
class ProvenanceReport:
    """Result of :func:`validate_provenance`.

    ``ok`` is ``True`` only when **every** reaction and every non-cofactor entity
    carries at least one resolvable source pointer. ``issues`` lists every
    element that did not (empty when ``ok``). The counts record how many elements
    were examined.
    """

    ok: bool
    issues: List[ProvenanceIssue] = field(default_factory=list)
    checked_reactions: int = 0
    checked_entities: int = 0


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _has_resolvable_source(row: Dict[str, Any]) -> bool:
    """True when ``row`` carries at least one usable ``source_id``/``source_uri``.

    A source is *resolvable* if any of the additive provenance carriers points at
    a non-empty identifier or URI: the primary ``rag_provenance`` pointer, any
    ``source_papers`` entry, any ``evidence`` passage, or a core-typed
    ``source_refs`` string. This mirrors exactly the pointers WP5 attaches.
    """
    prov = row.get("rag_provenance")
    if isinstance(prov, dict) and (_text(prov.get("source_id")) or _text(prov.get("source_uri"))):
        return True
    for paper in _safe_list(row.get("source_papers")):
        if isinstance(paper, dict) and (_text(paper.get("source_id")) or _text(paper.get("uri"))):
            return True
    for ev in _safe_list(row.get("evidence")):
        if isinstance(ev, dict) and (_text(ev.get("source_id")) or _text(ev.get("source_uri"))):
            return True
    for ref in _safe_list(row.get("source_refs")):
        if isinstance(ref, str) and ref.strip():
            return True
    return False


def _reaction_label(row: Dict[str, Any], idx: int) -> str:
    name = _text(row.get("name"))
    return name or f"reaction[{idx}]"


def validate_provenance(payload: Any) -> ProvenanceReport:
    """Assert the evidence-bound guarantee: **no element without evidence**.

    Every reaction (``processes.reactions``) and every non-cofactor entity
    (``entities.{compounds,proteins,protein_complexes}``) must carry at least one
    resolvable ``source_id`` / ``source_uri``. Any element that does not is
    flagged in the returned :class:`ProvenanceReport` and makes ``ok`` ``False``.

    This is a **read-only** check over an additive payload — it never mutates the
    payload and never touches a core gate. It is the WP6 tripwire that proves WP5
    synthesis omitted (rather than invented) any element it could not source.
    """
    issues: List[ProvenanceIssue] = []
    payload_dict = _safe_dict(payload)
    cofactors = _cofactor_names()

    processes = _safe_dict(payload_dict.get("processes"))
    reactions = _safe_list(processes.get("reactions"))
    checked_reactions = 0
    for idx, row in enumerate(reactions):
        if not isinstance(row, dict):
            continue
        checked_reactions += 1
        if not _has_resolvable_source(row):
            issues.append(
                ProvenanceIssue(
                    kind="reaction",
                    label=_reaction_label(row, idx),
                    pointer=f"/processes/reactions/{idx}",
                    reason="reaction has no resolvable source_id/source_uri",
                )
            )

    entities = _safe_dict(payload_dict.get("entities"))
    checked_entities = 0
    for bucket in _EVIDENCE_ENTITY_BUCKETS:
        kind = bucket[:-1] if bucket.endswith("s") else bucket
        for idx, row in enumerate(_safe_list(entities.get(bucket))):
            if not isinstance(row, dict):
                continue
            name = _text(row.get("name"))
            if bucket == "compounds" and name.casefold() in cofactors:
                # Ubiquitous cofactors are exempt (no pathway-specific evidence).
                continue
            checked_entities += 1
            if not _has_resolvable_source(row):
                issues.append(
                    ProvenanceIssue(
                        kind=kind,
                        label=name or f"{bucket}[{idx}]",
                        pointer=f"/entities/{bucket}/{idx}",
                        reason="entity has no resolvable source_id/source_uri",
                    )
                )

    return ProvenanceReport(
        ok=not issues,
        issues=issues,
        checked_reactions=checked_reactions,
        checked_entities=checked_entities,
    )


def _strip_in_place(value: Any) -> Any:
    """Recursively drop every :data:`RAG_ADDITIVE_KEYS` key from ``value``."""
    if isinstance(value, dict):
        for key in RAG_ADDITIVE_KEYS:
            value.pop(key, None)
        for sub in value.values():
            _strip_in_place(sub)
    elif isinstance(value, list):
        for item in value:
            _strip_in_place(item)
    return value


def strip_provenance(payload: Any) -> Dict[str, Any]:
    """Return a provenance-free deep copy of ``payload``.

    Removes every additive RAG key (:data:`RAG_ADDITIVE_KEYS`) from every element
    at any depth, leaving a plain core payload — exactly what a RAG-unaware (or
    RAG-off) stage sees. The input is **not** mutated. Used to prove the core
    gates pass on the plain payload too, i.e. that the provenance keys are purely
    additive and ignored.
    """
    return _strip_in_place(deepcopy(_safe_dict(payload)))


__all__ = [
    "RagProvenance",
    "RagEvidence",
    "RagSourcePaper",
    "RagAdditiveMetadata",
    "RAG_ADDITIVE_KEYS",
    "ProvenanceIssue",
    "ProvenanceReport",
    "validate_provenance",
    "strip_provenance",
]
