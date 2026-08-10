"""Typed, append-only provenance lineage for payload entities and processes.

``PRODUCT_CONTRACT`` § 3 requires six distinguishable categories of content, and six
facts per externally added element. :class:`LineageEntry` carries all six -- which stage
introduced it (``stage``), why (``reason``), what backs it (``sources``), whether the
paper stated it (``paper_explicit``), its evidence/support level (``support``), and any
uncertainty or review requirement (``review_required`` + ``uncertainty``). Attribution
rides two orthogonal CLOSED vocabularies, :data:`STAGES` (*which* stage) and
:data:`ORIGINS` (*what kind* of provenance -- the six § 3 categories, mapped by
:data:`CONTRACT_CATEGORIES`). Free strings would get seven spellings from seven writers,
leaving "which stage introduced this false identifier?" unanswerable.

Rules a writer must not work around. **Append-only**: :class:`Lineage` is immutable,
:meth:`Lineage.append` returns a new value, and :func:`record` -- the only supported
writer -- re-emits every entry it read, so no stage erases an earlier attribution.
**Loud rejection**: fields validate on construction and :class:`LineageError` names the
offending field; a silently dropped entry reads as "this content was original".
**Order-independent**: equality and serialization both go through
:meth:`Lineage.canonical`, so entry order never matters; duplicates are kept, because
dedup removes. **Evidence-bound**: ``direct``/``indirect`` support requires a named
source. **Pointer, not copy**: :class:`LineageSource` names a record, the retrieved
payload stays in ``rag_provenance`` / ``db_row``; this duplicates no existing carrier,
and ``source_id`` is checked only for non-emptiness -- accepting an identifier for its
*format* is a standing invariant violation. **Leaf**: imports no stage module.
**Three-valued** ``paper_explicit``: a stage that did not evaluate it records
``not_evaluated``, which is never ``not_explicit``.

:data:`LINEAGE_KEY` is namespaced like ``rag_provenance`` because plain ``lineage`` is
taken -- ``curation/gap_resolver.py:465`` reads ``row["lineage"]`` off
``entities.species`` as an NCBI *taxonomic* lineage. It is additive: a row without it is
a plain payload row, and a row needs review when any entry sets ``review_required``.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import MISSING, asdict, astuple, dataclass, fields
from typing import Any, Dict, List, Tuple

#: Payload key holding a row's serialized lineage (a list of entry mappings).
LINEAGE_KEY = "provenance_lineage"

#: Every stage allowed to introduce or change biological content, in pipeline order --
#: also the canonical sort order of a lineage. Deliberately NO exporter stage: an
#: exporter may not change biology after the canonical graph is frozen.
STAGES: Tuple[str, ...] = (
    "preprocessing", "paper_extraction", "normalization", "deterministic_inference",
    "identifier_mapping", "entity_enrichment", "rag_retrieval", "rag_admission",
    "gap_resolution", "audit_repair", "manual_curation", "quarantine",
)

#: The six § 3 categories, with the sixth ("unresolved or excluded") split: "could not
#: be grounded" and "deliberately removed" are different facts that must not merge.
ORIGINS: Tuple[str, ...] = (
    "paper_stated", "rag_literature", "database_grounded", "inferred",
    "audit_modified", "unresolved", "excluded",
)

#: Evidence/support level, weakest first (compare with ``SUPPORT_LEVELS.index``):
#: ``direct`` a named source states this fact · ``indirect`` a named source supports it
#: only by implication · ``derived`` deterministically derived from supported content ·
#: ``unsupported`` neither, so the content is provisional.
SUPPORT_LEVELS: Tuple[str, ...] = ("unsupported", "derived", "indirect", "direct")

#: Whether the supplied paper stated it; ``not_evaluated`` is never ``not_explicit``.
PAPER_EXPLICIT_VALUES: Tuple[str, ...] = ("explicit", "not_explicit", "not_evaluated")

#: ``PRODUCT_CONTRACT`` § 3 category -> the origins that represent it.
CONTRACT_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "paper_explicit": ("paper_stated",),
    "rag_external_literature": ("rag_literature",),
    "database_grounded": ("database_grounded",),
    "deterministic_inference": ("inferred",),
    "audit_repair": ("audit_modified",),
    "unresolved_or_excluded": ("unresolved", "excluded"),
}


class LineageError(ValueError):
    """A malformed lineage record. The message always names the offending field."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise LineageError(f"{name}: expected a string, got {type(value).__name__}")
    return value.strip()


def _choice(value: Any, allowed: Tuple[str, ...], name: str) -> str:
    text = _text(value, name)
    if text not in allowed:
        raise LineageError(f"{name}: {text!r} is not one of {list(allowed)}")
    return text


@dataclass(frozen=True)
class LineageSource:
    """A pointer to the paper, passage or database record backing an entry."""

    source_id: str = ""     # PMCID / DOI / accession / PathBank id / filename
    source_type: str = ""   # advisory: "paper" | "pathbank" | "uniprot" | ...
    uri: str = ""
    locator: str = ""       # section, chunk id, row id -- whatever resolves it


def _source(value: Any) -> LineageSource:
    if isinstance(value, LineageSource):
        return value
    if not isinstance(value, Mapping):
        raise LineageError(f"sources: expected a mapping, got {type(value).__name__}")
    unknown = sorted(set(value) - _SOURCE_FIELDS)
    if unknown:
        raise LineageError(f"sources: unknown field(s) {unknown}")
    source = LineageSource(**{k: _text(value[k], f"sources.{k}") for k in value})
    if not (source.source_id or source.uri):
        raise LineageError("sources: each source needs a source_id or a uri")
    return source


@dataclass(frozen=True)
class LineageEntry:
    """One stage's attribution for one element. Validated on construction."""

    stage: str             # which stage introduced or changed it
    origin: str            # what kind of provenance it has
    support: str           # its evidence/support level
    paper_explicit: str    # whether the supplied paper stated it
    reason: str            # why this stage introduced or changed it
    review_required: bool  # whether it needs human review
    uncertainty: str = ""  # what is uncertain about it
    sources: Tuple[LineageSource, ...] = ()  # the records backing it

    def __post_init__(self) -> None:
        put = object.__setattr__
        for name, allowed in _CLOSED_FIELDS:
            put(self, name, _choice(getattr(self, name), allowed, name))
        reason = _text(self.reason, "reason")
        if not reason:
            raise LineageError("reason: must say why this stage introduced or changed it")
        put(self, "reason", reason)
        if not isinstance(self.review_required, bool):
            raise LineageError("review_required: expected a bool, got "
                               f"{type(self.review_required).__name__}")
        uncertainty = _text(self.uncertainty, "uncertainty")
        if self.review_required and not uncertainty:
            raise LineageError("uncertainty: required whenever review_required is true")
        put(self, "uncertainty", uncertainty)
        if not isinstance(self.sources, (list, tuple)):
            raise LineageError(f"sources: expected a list, got {type(self.sources).__name__}")
        parsed = tuple(sorted((_source(s) for s in self.sources), key=astuple))
        if self.support in ("direct", "indirect") and not parsed:
            raise LineageError(f"sources: support={self.support!r} requires a named source")
        put(self, "sources", parsed)

    def as_dict(self) -> Dict[str, Any]:  # plain JSON types; ``sources`` as a list
        return dict(asdict(self), sources=[asdict(s) for s in self.sources])


_CLOSED_FIELDS = (("stage", STAGES), ("origin", ORIGINS), ("support", SUPPORT_LEVELS),
                  ("paper_explicit", PAPER_EXPLICIT_VALUES))
_SOURCE_FIELDS = frozenset(f.name for f in fields(LineageSource))
_ENTRY_FIELDS = frozenset(f.name for f in fields(LineageEntry))
_REQUIRED_FIELDS = tuple(f.name for f in fields(LineageEntry) if f.default is MISSING)


def _entry(value: Any) -> LineageEntry:
    if isinstance(value, LineageEntry):
        return value
    if not isinstance(value, Mapping):
        raise LineageError(f"entry: expected a mapping, got {type(value).__name__}")
    bad = [n for n in _REQUIRED_FIELDS if n not in value] + sorted(set(value) - _ENTRY_FIELDS)
    if bad:
        raise LineageError(f"entry: missing or unknown field(s) {bad}")
    return LineageEntry(**dict(value))


def _entry_key(entry: LineageEntry) -> Tuple[Any, ...]:  # content-derived, stable
    return (STAGES.index(entry.stage), entry.origin, entry.support, entry.paper_explicit,
            entry.reason, entry.review_required, entry.uncertainty,
            tuple(astuple(s) for s in entry.sources))


@dataclass(frozen=True, eq=False)
class Lineage:
    """An append-only history of :class:`LineageEntry` records for one element."""

    entries: Tuple[LineageEntry, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entries, (list, tuple)):
            raise LineageError(f"{LINEAGE_KEY}: expected a list, not "
                               f"{type(self.entries).__name__}")
        object.__setattr__(self, "entries", tuple(_entry(item) for item in self.entries))

    def canonical(self) -> Tuple[LineageEntry, ...]:
        """The entries in canonical order, independent of the order appended."""
        return tuple(sorted(self.entries, key=_entry_key))

    def append(self, entry: Any) -> "Lineage":
        """A NEW lineage with ``entry`` added. ``self`` is unchanged."""
        return Lineage(self.entries + (_entry(entry),))

    def as_list(self) -> List[Dict[str, Any]]:
        """Stable JSON form: canonical order, plain types, no entry lost."""
        return [entry.as_dict() for entry in self.canonical()]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Lineage):
            return NotImplemented
        return self.canonical() == other.canonical()

    def __hash__(self) -> int:
        return hash(self.canonical())


def read(row: Mapping[str, Any]) -> Lineage:
    """``row``'s lineage. Raises :class:`LineageError` if what is stored is malformed."""
    return Lineage(row.get(LINEAGE_KEY) or ())


def record(row: MutableMapping[str, Any], entry: Any) -> Lineage:
    """Append ``entry`` to ``row``'s lineage in place; return the new lineage. Every
    entry already there is parsed and re-emitted, so no stage using this can erase
    another stage's attribution."""
    lineage = read(row).append(entry)
    row[LINEAGE_KEY] = lineage.as_list()
    return lineage
