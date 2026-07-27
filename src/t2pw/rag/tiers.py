"""Evidence tiering for research-mode pathways — an OFF-PAYLOAD side-car.

A novel multi-paper pathway is not uniformly trustworthy: one enzyme has a
reviewed UniProt accession, the next is a name that appeared once in the seed
paper's discussion. Strict PWML mode papers over that difference (it either
passes the gate or it does not); a human reviewing a *candidate* pathway needs
the difference spelled out per element, loudly, with nothing dropped.

This module assigns each entity and reaction one of four tiers and returns a
:class:`TierReport` keyed by JSON pointer. It **never mutates the payload** —
tiers live beside it because ``synthesize`` asserts a closed set of row keys,
and because the merged payload is a different object (below).

**CORROBORATION CONFIRMS THE ENTITY EXISTS. IT DOES NOT INVENT A UNIPROT ID. A
Tier B entity still has no accession. Tiering is a claim about EVIDENCE, never
about IDENTITY.**

Three traps this is built around, all verified in the current tree:

* **Tier the PRE-MERGE payload.** ``pipeline._clean_processes`` /
  ``_clean_enzymes`` rebuild rows from a whitelist and drop every RAG carrier,
  so the merged payload has no provenance left to tier. Pass
  ``rag_result.payload`` (== ``SynthesisResult.payload``).
* **A ``source_id`` is not a paper.** ``synthesize._seed_row_provenance`` turns
  each ``source_refs`` string into a ``source_id``, and Stage 1 fills
  ``source_refs`` with verbatim quotes — so a quoted sentence masquerades as a
  paper id. Only a source with a non-empty ``chunk_id`` (or an id present in the
  selection scores) counts as a genuine retrieved paper; every other pointer
  collapses into one "seed / unverified" bucket so it cannot inflate a
  distinct-paper count.
* **"mapped" is not grounded.** Two sites stamp a fake identity — the PathBank
  ``Unknown`` sentinel and ``best_effort_fallback``. Both are refused Tier A
  *with a recorded reason*, so the reviewer sees why.

Whether a corroborating review actually cites the seed is **not knowable here**:
``ingest`` discards reference sections and ``CandidatePaper`` has no DOI, so
there is no citation graph. A review corroborator is therefore labelled
"review (may restate the seed)" and can never, alone, satisfy Tier B.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from t2pw.pipeline.entity_identity import (
    PATHBANK_UNKNOWN_FALLBACK_RULE,
    PATHBANK_UNKNOWN_PROTEIN_ID,
    is_pathbank_unknown_protein,
    protein_external_identity,
)
from t2pw.pipeline.export_mode import DEFAULT_EXPORT_MODE, coerce_mode, is_research
from t2pw.rag.provenance import SEED_SOURCE_ID
from t2pw.rag.synthesize import canonical_name

TIER_A = "A"
TIER_B = "B"
TIER_C = "C"
TIER_D = "D"

TIER_LABELS: Dict[str, str] = {
    TIER_A: "DB-grounded",
    TIER_B: "corroborated by a second paper (no identifier)",
    TIER_C: "single-source (review)",
    TIER_D: "UNSOURCED (review)",
}

#: Tiers a human must look at before this pathway is believed.
REVIEW_REQUIRED_TIERS: Tuple[str, ...] = (TIER_C, TIER_D)

#: ``SelectionScore.document_type`` value meaning "the corroborator is a review".
REVIEW_DOCUMENT_TYPE = "multi_example_review"
REVIEW_SOURCE_LABEL = "review (may restate the seed)"

#: Buckets holding synthesized chemistry. Scaffolding buckets (species, tissues,
#: ...) are carried from the seed, never evidence-built, so they are not tiered.
TIERED_ENTITY_BUCKETS: Tuple[str, ...] = ("compounds", "proteins", "protein_complexes")

#: Accession strings that look like an identifier but ground nothing.
_UNUSABLE_IDS = frozenset({"", "unknown", "n/a", "none", "null", "-", "?"})
_UNIPROT_KEYS = ("uniprot", "uniprot_id", "uniprot-id", "drugbank", "drugbank_id", "drugbank-id")
_DB_IDENTIFIER_KEYS = (
    "pathbank_id",
    "pathbank_compound_id",
    "pathbank_protein_id",  # the Unknown sentinel 9659 is rejected separately
    "pathwhiz_id",
    "pathwhiz_reaction_id",
    "pw_id",
    "locked_reaction_id",
)
_GUESS_RULES = frozenset({"best_effort_fallback", PATHBANK_UNKNOWN_FALLBACK_RULE})
_SOURCE_FIELDS = ("title", "uri", "section", "quote", "chunk_id")

#: ``(row key, is_list, {side-car field: carrier key})``. The key drift is real
#: and silent: ``source_papers`` uses ``title``/``uri`` while ``rag_provenance``
#: uses ``source_title``/``source_uri``, and an evidence record carries no title
#: at all — a title can only be joined back from one of the other two.
_CARRIER_MAPS: Tuple[Tuple[str, bool, Dict[str, str]], ...] = (
    (
        "rag_provenance",
        False,
        {"title": "source_title", "uri": "source_uri", "section": "section", "chunk_id": "chunk_id"},
    ),
    ("source_papers", True, {"title": "title", "uri": "uri"}),
    (
        "evidence",
        True,
        {"uri": "source_uri", "section": "section", "chunk_id": "chunk_id", "quote": "text"},
    ),
)

#: ``resolver(name, organism) -> Dict[str, Any]`` shaped like ``map_protein_uniprot``.
Resolver = Callable[[str, str], Dict[str, Any]]


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _name_key(name: Any) -> str:
    """Tolerant join key: the same synonym collapse synthesis itself keys on.

    ``canonical_name`` rewrites only *known* aliases, so distinct names stay
    distinct; everything else is a plain casefold.
    """

    return canonical_name(name).casefold()


def _participant_name(participant: Any) -> str:
    return _text(participant.get("name")) if isinstance(participant, dict) else _text(participant)


# ---------------------------------------------------------------------------
# Identity — Tier A, with both poison sites refused out loud.
# ---------------------------------------------------------------------------
def _identity_rejections(row: Dict[str, Any]) -> List[str]:
    """Why this row's "mapped" identity must not be believed (empty = believable)."""

    meta = _as_dict(row.get("mapping_meta"))
    out: List[str] = []
    if is_pathbank_unknown_protein(row):
        out.append("identity rejected: PathBank Unknown-protein sentinel")
    if row.get("best_effort") is True or meta.get("best_effort") is True:
        out.append("identity rejected: best_effort guess (top candidate accepted unconditionally)")
    rule = _text(meta.get("chosen_rule") or row.get("chosen_rule")).casefold()
    reason = _text(meta.get("reason") or row.get("reason")).casefold()
    if rule in _GUESS_RULES or reason in _GUESS_RULES:
        out.append(f"identity rejected: mapping rule {rule or reason} is a guess, not a match")
    if meta.get("cross_species_placeholder") is True or row.get("cross_species_placeholder") is True:
        out.append("identity rejected: cross-species placeholder")
    for container in (row, _as_dict(row.get("mapped_ids")), _as_dict(row.get("ids")), meta):
        try:
            grounded = int(container.get("pathbank_protein_id") or 0) != PATHBANK_UNKNOWN_PROTEIN_ID
        except (TypeError, ValueError):
            continue
        if not grounded:
            out.append("identity rejected: PathBank Unknown-protein id 9659")
            break
    return out


def _usable_identity(row: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """``(identifier, source_key, rejections)``; identifier is "" unless grounded.

    Reuses :func:`protein_external_identity` for UniProt/DrugBank rather than
    re-deriving it; the key scan exists only to name the field for the reviewer.
    """

    containers = (
        row,
        _as_dict(row.get("mapped_ids")),
        _as_dict(row.get("ids")),
        _as_dict(row.get("mapping_meta")),
    )
    identifier, key = protein_external_identity(row), ""
    if identifier:
        key = next(
            (k for c in containers for k in _UNIPROT_KEYS if _text(c.get(k)) == identifier),
            "uniprot",
        )
    else:
        for candidate_key in _DB_IDENTIFIER_KEYS:
            found = next((_text(c.get(candidate_key)) for c in containers if _text(c.get(candidate_key))), "")
            if found:
                identifier, key = found, candidate_key
                break

    rejections = _identity_rejections(row)
    if identifier and identifier.casefold() in _UNUSABLE_IDS:
        rejections.append(f"identity rejected: placeholder identifier {identifier!r}")
    return ("" if rejections else identifier), key, rejections


# ---------------------------------------------------------------------------
# Sources — distinct-paper counting off the RAG carriers.
# ---------------------------------------------------------------------------
def _merge_source(bucket: Dict[str, Dict[str, Any]], source_id: str, **fields: Any) -> None:
    """Fold one pointer into the per-source record; first non-empty value wins."""

    if not source_id:
        return
    record = bucket.setdefault(source_id, {"source_id": source_id, **{f: "" for f in _SOURCE_FIELDS}})
    for key, value in fields.items():
        if _text(value) and not record.get(key):
            record[key] = _text(value)


def _collect_sources(rows: List[Dict[str, Any]], doc_types: Dict[str, str]) -> List[Dict[str, Any]]:
    """Every distinct source backing ``rows`` (the row itself plus joined rows)."""

    bucket: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for key, is_list, field_map in _CARRIER_MAPS:
            carriers = _as_list(row.get(key)) if is_list else [row.get(key)]
            for carrier in carriers:
                carrier = _as_dict(carrier)
                _merge_source(
                    bucket,
                    _text(carrier.get("source_id")),
                    **{dest: carrier.get(src) for dest, src in field_map.items()},
                )
        for ref in _as_list(row.get("source_refs")):
            _merge_source(bucket, _text(ref))

    for record in bucket.values():
        doc_type = doc_types.get(record["source_id"], "")
        record["document_type"] = doc_type
        record["is_review"] = doc_type.casefold() == REVIEW_DOCUMENT_TYPE
        # A verbatim quote can masquerade as a source_id; only a real chunk (or a
        # scored selection candidate) proves a retrieved paper.
        record["retrieved"] = bool(record["chunk_id"]) or record["source_id"] in doc_types
        record["label"] = (
            REVIEW_SOURCE_LABEL
            if record["is_review"]
            else ("retrieved paper" if record["retrieved"] else "seed / unverified pointer")
        )
    return sorted(bucket.values(), key=lambda r: (not r["retrieved"], r["source_id"]))


def _tier_from_sources(sources: List[Dict[str, Any]]) -> Tuple[str, str, int, bool, List[str]]:
    """``(tier, reason, distinct_paper_count, is_review_only, notes)`` — evidence only."""

    notes: List[str] = []
    retrieved = [s for s in sources if s["retrieved"]]
    corroborators = [s for s in retrieved if not s["is_review"]]
    quoted = [s for s in corroborators if s["quote"]]

    # ``distinct`` counts only sources we could actually identify as a paper --
    # a non-empty chunk_id or a scored selection candidate. Unverified pointers
    # are NOT added to it: Stage 1 fills ``source_refs`` with verbatim quotes,
    # so a quoted sentence arrives shaped exactly like a source_id and would
    # otherwise buy a free +1 -- precisely the increment that crosses C into B.
    distinct = len({s["source_id"] for s in retrieved})
    unverified = len(sources) - len(retrieved)
    review_only = bool(retrieved) and not corroborators

    if not sources:
        return TIER_D, "no provenance of any kind on this element", 0, False, notes

    if quoted:
        # Tier B needs a second *independent* statement: either a second
        # identified paper, or the seed (which the unverified pointers stand
        # for). One identified paper and nothing else is still single-source.
        independent = len({s["source_id"] for s in quoted})
        if independent >= 2 or unverified:
            cited = ", ".join(s["title"] or s["source_id"] for s in quoted[:3])
            if independent >= 2:
                reason = (
                    f"stated in {independent} identified papers, each with its own cited "
                    f"passage: {cited}"
                )
            else:
                reason = (
                    f"stated in the seed and independently in {cited}, which carries its "
                    "own cited passage"
                )
                notes.append(
                    "the corroborating statement is one identified paper plus the seed; "
                    "the seed pointer itself is a quotation, not a resolvable paper id"
                )
            return TIER_B, reason, distinct, False, notes
        return (
            TIER_C,
            "one identified paper with a cited passage and no second, independent statement",
            distinct,
            review_only,
            notes,
        )

    if corroborators:
        notes.append("corroborating paper carries no cited passage of its own")
        return (
            TIER_C,
            "second source has no cited passage of its own, so corroboration is unverified",
            distinct,
            review_only,
            notes,
        )

    if retrieved:
        return (
            TIER_C,
            "only corroborator is a review (may restate the seed); a review alone cannot "
            "establish independence and never satisfies Tier B",
            distinct,
            True,
            notes,
        )

    # Distinguish the seed sentinel from a genuine quote-as-source_id: the seed
    # IS a resolvable document, it simply has no indexed passage to cite. Saying
    # "unresolvable quotation" about it reads as though the source were bogus.
    if any(str(s.get("source_id")) == SEED_SOURCE_ID for s in sources):
        notes.append(
            "stated in the uploaded seed paper; no retrieved paper corroborates it"
        )
    else:
        notes.append(
            "every pointer on this element is an unresolvable quotation rather than a paper id"
        )
    return TIER_C, "stated in a single source", distinct, review_only, notes


# ---------------------------------------------------------------------------
# Optional, opt-in, never-blocking UniProt retry (READ-ONLY side-car lookup).
# ---------------------------------------------------------------------------
def default_resolver() -> Resolver:
    """A ``map_protein_uniprot`` resolver, imported lazily so RAG-off stays hermetic.

    It reuses the existing query plan, alias expansion and organism scoping
    wholesale — nothing here reimplements HTTP or query construction. The result
    is consumed read-only; no payload field is ever written.
    """

    state: Dict[str, Any] = {"client": None}

    def _lookup(name: str, organism: str) -> Dict[str, Any]:
        from t2pw.mapping.map_ids import HttpClient, map_protein_uniprot

        if state["client"] is None:
            state["client"] = HttpClient()
        return map_protein_uniprot(state["client"], name, organism)

    return _lookup


def _accept_lookup(result: Any) -> Tuple[str, str]:
    """``(accession, note)`` — accepts only a genuinely matched, non-guess mapping."""

    data = _as_dict(result)
    reason = _text(data.get("reason")).casefold()
    if _text(data.get("status")) != "mapped":
        return "", f"UniProt retry found no confident match ({reason or 'unmapped'})"
    if data.get("best_effort") is True or reason in _GUESS_RULES:
        return "", "UniProt retry rejected: best_effort guess, not a match"
    if _text(data.get("chosen_rule")).casefold() in _GUESS_RULES:
        return "", "UniProt retry rejected: fallback rule, not a match"
    accession = _text(_as_dict(data.get("mapped_ids")).get("uniprot"))
    if not accession or accession.casefold() in _UNUSABLE_IDS:
        return "", "UniProt retry rejected: placeholder accession"
    return accession, ""


def _online_identity(
    lookup: Resolver,
    memo: Dict[Tuple[str, str], Tuple[str, str]],
    name: str,
    organism: str,
) -> Tuple[str, str]:
    """Best-effort, never-blocking accession retry. Returns ``(accession, note)``."""

    key = (name.casefold(), organism.casefold())
    if key not in memo:
        try:
            memo[key] = _accept_lookup(lookup(name, organism))
        except Exception as exc:  # network/LLM/parse — never allowed to break tiering
            memo[key] = ("", f"UniProt retry failed ({type(exc).__name__}: {exc}); tiered on evidence alone")
    return memo[key]


# ---------------------------------------------------------------------------
# Public results.
# ---------------------------------------------------------------------------
@dataclass
class TierAssignment:
    """One tiered element. ``pointer`` is the JSON pointer into the payload."""

    pointer: str
    kind: str  # "entity" | "reaction"
    name: str
    tier: str
    label: str
    reason: str
    identifier: str = ""
    identifier_source: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    distinct_paper_count: int = 0
    is_review_only: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TierReport:
    """The off-payload side-car: assignments plus lookup indexes."""

    assignments: List[TierAssignment] = field(default_factory=list)
    mode: str = DEFAULT_EXPORT_MODE
    online: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def by_pointer(self) -> Dict[str, TierAssignment]:
        return {a.pointer: a for a in self.assignments}

    @property
    def counts(self) -> Dict[str, int]:
        counts = {tier: 0 for tier in (TIER_A, TIER_B, TIER_C, TIER_D)}
        for assignment in self.assignments:
            counts[assignment.tier] = counts.get(assignment.tier, 0) + 1
        return counts

    @property
    def review_required(self) -> bool:
        return any(a.tier in REVIEW_REQUIRED_TIERS for a in self.assignments)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "online": self.online,
            "counts": self.counts,
            "tier_labels": dict(TIER_LABELS),
            "review_required": self.review_required,
            "assignments": [a.to_dict() for a in self.assignments],
            "notes": list(self.notes),
        }


def _document_types(rag_result: Any) -> Dict[str, str]:
    """``source_id -> document_type`` from the selection scores (empty when absent).

    Membership is itself meaningful: db/corpus chunks and ``seed_paper`` have no
    entry, so a missing key means "unknown document type", never "primary research".
    """

    scores = getattr(getattr(rag_result, "selection", None), "scores", None)
    if not isinstance(scores, dict):
        return {}
    return {_text(k): _text(getattr(v, "document_type", "")) for k, v in scores.items() if _text(k)}


def _join_index(reactions: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """``name key -> rows carrying that name's evidence``.

    Entity rows never get an ``evidence`` list (``synthesize`` attaches an empty
    one), so an entity's own cited passage is only reachable by joining its name
    to every reaction naming it in inputs/outputs/enzymes.
    """

    index: Dict[str, List[Dict[str, Any]]] = {}
    for row in reactions:
        row = _as_dict(row)
        participants = _as_list(row.get("inputs")) + _as_list(row.get("outputs"))
        carriers = [(_participant_name(p), row) for p in participants]
        for actor in _as_list(row.get("enzymes")):
            actor = _as_dict(actor)
            carriers.append((_text(actor.get("entity") or actor.get("name")), actor))
        for name, carrier in carriers:
            if name:
                index.setdefault(_name_key(name), []).append(carrier)
    return index


def _identity_index(identity_source: Any) -> Dict[str, Dict[str, Any]]:
    """``name -> entity row`` from a *mapped* payload, for Tier A lookups.

    The pre-merge synthesized payload cannot carry an identifier: every row it
    emits is filtered through ``synthesize._ALLOWED_ROW_KEYS``, which has no
    ``mapped_ids`` / ``ids`` / ``mapping_meta``. So identity has to come from the
    mapped payload, whose entity rows survive ``pipeline._clean_entities``
    unchanged. Without this bridge no element could ever reach Tier A on real
    data and every genuinely UniProt-mapped enzyme would be under-reported.
    """

    index: Dict[str, Dict[str, Any]] = {}
    entities = _as_dict(_as_dict(identity_source).get("entities"))
    for bucket in TIERED_ENTITY_BUCKETS:
        for row in _as_list(entities.get(bucket)):
            row = _as_dict(row)
            key = _name_key(row.get("name"))
            if key and key not in index:
                index[key] = row
    return index


def assign_tiers(
    payload: Any,
    *,
    rag_result: Any = None,
    resolver: Optional[Resolver] = None,
    online: bool = False,
    mode: Any = DEFAULT_EXPORT_MODE,
    identity_source: Any = None,
) -> TierReport:
    """Tier every entity and reaction of a **pre-merge** synthesized payload.

    ``payload`` is read only — nothing is written back, ever. ``online`` gates the
    UniProt retry and defaults to ``False`` so unit tests and RAG-off runs never
    touch the network; ``resolver`` is the injection point (default
    :func:`default_resolver`). Every network failure is swallowed into a note.

    ``identity_source`` is the *mapped* payload, consulted only for Tier A
    identifiers (see :func:`_identity_index`). Provenance is always read from
    ``payload``, because the merge strips the RAG carriers off every reaction.
    """

    payload_dict = _as_dict(payload)
    identity_by_name = _identity_index(identity_source)
    report = TierReport(mode=coerce_mode(mode), online=bool(online))
    doc_types = _document_types(rag_result)
    reactions = _as_list(_as_dict(payload_dict.get("processes")).get("reactions"))
    entities = _as_dict(payload_dict.get("entities"))
    index = _join_index(reactions)
    organism = next(
        (n for row in _as_list(entities.get("species")) for n in [_text(_as_dict(row).get("name"))] if n),
        "",
    )
    lookup: Optional[Resolver] = (resolver or default_resolver()) if online else None
    memo: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def _emit(pointer: str, kind: str, name: str, row: Dict[str, Any], joined: List[Dict[str, Any]], protein: bool) -> None:
        identifier, identifier_source, rejections = _usable_identity(row)
        if not identifier and not rejections:
            # The synthesized row can never carry an id; consult the mapped
            # payload for this name. Rejections there (Unknown sentinel,
            # best-effort guess) are surfaced exactly as if they were local.
            mapped_row = identity_by_name.get(_name_key(name))
            if mapped_row is not None:
                identifier, identifier_source, rejections = _usable_identity(mapped_row)
        sources = _collect_sources([row, *joined], doc_types)
        evidence_tier, reason, distinct, review_only, notes = _tier_from_sources(sources)
        notes = [*rejections, *notes]
        tier = evidence_tier
        if identifier:
            tier, reason = TIER_A, f"external identifier {identifier} ({identifier_source})"
        else:
            if rejections:
                reason = f"{reason}; {'; '.join(rejections)}"
            if lookup is not None and protein and name:
                row_organism = _text(_as_dict(row.get("rag_provenance")).get("organism")) or organism
                accession, note = _online_identity(lookup, memo, name, row_organism)
                if note:
                    notes.append(note)
                if accession:
                    tier, identifier, identifier_source = TIER_A, accession, "uniprot:online_retry"
                    reason = (
                        f"external identifier {accession} recovered by read-only UniProt retry "
                        "(side-car only; payload mapping untouched)"
                    )
        report.assignments.append(
            TierAssignment(
                pointer=pointer,
                kind=kind,
                name=name,
                tier=tier,
                label=TIER_LABELS[tier],
                reason=reason,
                identifier=identifier,
                identifier_source=identifier_source if identifier else "",
                sources=sources,
                distinct_paper_count=distinct,
                is_review_only=review_only,
                notes=notes,
            )
        )

    for bucket in TIERED_ENTITY_BUCKETS:
        for idx, row in enumerate(_as_list(entities.get(bucket))):
            if isinstance(row, dict):
                name = _text(row.get("name"))
                pointer = f"/entities/{bucket}/{idx}"
                _emit(pointer, "entity", name, row, index.get(_name_key(name), []), bucket != "compounds")

    for idx, row in enumerate(reactions):
        if not isinstance(row, dict):
            continue
        label = _text(row.get("name")) or f"reaction[{idx}]"
        _emit(f"/processes/reactions/{idx}", "reaction", label, row, [], False)
        for actor_idx, actor in enumerate(_as_list(row.get("enzymes"))):
            if isinstance(actor, dict):
                pointer = f"/processes/reactions/{idx}/enzymes/{actor_idx}"
                _emit(pointer, "entity", _text(actor.get("entity") or actor.get("name")), actor, [row], True)

    if is_research(report.mode) and report.review_required:
        report.notes.append(
            "research mode: Tier C/D elements need human review before this pathway is believed"
        )
    return report


__all__ = [
    "TIER_A", "TIER_B", "TIER_C", "TIER_D", "TIER_LABELS", "REVIEW_REQUIRED_TIERS",
    "REVIEW_DOCUMENT_TYPE", "REVIEW_SOURCE_LABEL", "TIERED_ENTITY_BUCKETS", "Resolver",
    "TierAssignment", "TierReport", "assign_tiers", "default_resolver",
]
