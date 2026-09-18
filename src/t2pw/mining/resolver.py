"""The deferred external-identifier resolver seam.

Version 1 admits a reaction on **paper evidence alone**. PathBank, PWC, HMDB,
KEGG, ChEBI, PubChem, UniProt and every other biochemical identifier service are
deliberately NOT consulted, are not a prerequisite for admission, and are not
queried during mining rounds.

This module is the place a later, separately authorized database-validation
layer attaches without a schema migration. Today it is a no-op that answers
``not_evaluated`` for everything, and the shape of that answer is the whole
point: a reader of ``primary_graph.json`` sees an explicit "not evaluated"
rather than an absent field they might read as "checked and clean".

Literature acquisition services (EuropePMC, NCBI, Crossref, ...) are a different
thing and are still used -- the restriction is on biochemical *fact* validation,
not on paper *discovery*.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Protocol, runtime_checkable

STATUS_NOT_EVALUATED = "not_evaluated"
STATUS_RESOLVED = "resolved"
STATUS_UNRESOLVED = "unresolved"
STATUS_ERROR = "error"


@dataclass
class ResolutionResult:
    """What a resolver has to say about one entity or reaction.

    ``status`` is :data:`STATUS_NOT_EVALUATED` in version 1, always. ``refs`` is
    reserved for a future ``{"chebi": "CHEBI:15422", ...}`` mapping and stays
    empty until a database layer is authorized.
    """

    status: str = STATUS_NOT_EVALUATED
    refs: Dict[str, str] = field(default_factory=dict)
    reason: str = "external biochemical database validation is deferred in version 1"
    resolver: str = "noop"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@runtime_checkable
class EntityResolver(Protocol):
    """The optional interface a future database layer implements."""

    name: str

    def resolve_entity(self, name: str, entity_type: str = "") -> ResolutionResult: ...

    def resolve_reaction(
        self, substrates: Iterable[str], products: Iterable[str]
    ) -> ResolutionResult: ...


class NoopResolver:
    """The version-1 resolver. Answers ``not_evaluated`` and touches nothing.

    It performs no I/O, holds no credentials and has no cache, so enabling
    pathway mining cannot accidentally start a database crawl.
    """

    name = "noop"

    def resolve_entity(self, name: str, entity_type: str = "") -> ResolutionResult:  # noqa: ARG002
        return ResolutionResult()

    def resolve_reaction(
        self, substrates: Iterable[str], products: Iterable[str]
    ) -> ResolutionResult:  # noqa: ARG002
        return ResolutionResult()


_DEFAULT = NoopResolver()


def default_resolver() -> EntityResolver:
    """The resolver a mining run uses unless a caller injects another one."""
    return _DEFAULT


def annotate(entity_or_reaction: Any, result: ResolutionResult) -> None:
    """Attach a resolution result to whichever field the object carries."""
    payload = result.to_dict()
    if hasattr(entity_or_reaction, "external_refs"):
        entity_or_reaction.external_refs = payload
    elif hasattr(entity_or_reaction, "external_validation"):
        entity_or_reaction.external_validation = payload


def deferred_note() -> List[str]:
    """The sentence every report prints about database validation."""
    return [
        "External biochemical database validation (PathBank, PWC, HMDB, KEGG, ChEBI, "
        "PubChem, UniProt) is DEFERRED in version 1. Every entity and reaction carries "
        "status 'not_evaluated'. Admission was on paper evidence alone; no identifier in "
        "this run has been checked against any external database.",
    ]


__all__ = [
    "STATUS_NOT_EVALUATED",
    "STATUS_RESOLVED",
    "STATUS_UNRESOLVED",
    "STATUS_ERROR",
    "ResolutionResult",
    "EntityResolver",
    "NoopResolver",
    "default_resolver",
    "annotate",
    "deferred_note",
]
