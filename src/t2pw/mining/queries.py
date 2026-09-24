"""Literature query formulation, traceable back to the frontier item that asked.

Every query this module produces records the frontier item, the kind of question,
and the exact terms that went into it, so a paper in the run can always be traced
to the question that found it.

Reuse, not reimplementation: the query *string* is rendered by
``t2pw.rag.acquire.build_query``, the same renderer ``search_candidates`` runs,
from a context dict this module assembles. So the recorded query is exactly the
executed one. What mining decides is the content: the frontier item's own terms
and the species, all REQUIRED. The topic and generic words are not OR'd in; in
pilot-001 they made the item's own term optional, and 60 queries returned the
same 20 papers (finding F2).

Search terms vs biochemistry
----------------------------
A bridge query deliberately puts two metabolite names in the same search. Two
papers coming back for that query are **not** thereby connected, and the
metabolites are **not** thereby a reaction. Search terms find candidate
literature; only an extracted, evidenced reaction can connect anything.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from t2pw.mining.config import MiningConfig
from t2pw.mining.ids import query_id as make_query_id
from t2pw.mining.models import (
    FRONTIER_BRANCH,
    FRONTIER_BRIDGE,
    FRONTIER_COFACTOR,
    FRONTIER_COMPARTMENT,
    FRONTIER_CONFLICT,
    FRONTIER_DIRECTION,
    FRONTIER_DOWNSTREAM,
    FRONTIER_ENZYME,
    FRONTIER_PATHWAY_REFERENCE,
    FRONTIER_SEED,
    FRONTIER_SPECIES,
    FRONTIER_UPSTREAM,
    FrontierItem,
    LiteratureQuery,
)
from t2pw.mining.normalize import is_cofactor
from t2pw.mining.species import SPECIES_TARGET_SUPPORTED

#: Extra free-text terms per frontier kind. These steer *retrieval*; they never
#: appear in a claim and never affect admission.
_KIND_TERMS: Dict[str, Sequence[str]] = {
    FRONTIER_DOWNSTREAM: ("metabolism", "conversion", "catabolism", "downstream product"),
    FRONTIER_UPSTREAM: ("biosynthesis", "precursor", "formed from", "upstream"),
    FRONTIER_BRANCH: ("reaction", "enzyme", "metabolic fate"),
    FRONTIER_ENZYME: ("substrate specificity", "catalyzes", "enzyme activity"),
    FRONTIER_COFACTOR: ("cofactor", "co-substrate", "requirement"),
    FRONTIER_PATHWAY_REFERENCE: ("pathway", "biosynthetic pathway", "steps"),
    FRONTIER_BRIDGE: ("intermediate", "converted to", "links", "connecting step"),
    FRONTIER_CONFLICT: ("evidence", "characterization", "kinetics"),
    FRONTIER_DIRECTION: ("reversible", "equilibrium", "direction"),
    FRONTIER_COMPARTMENT: ("localization", "subcellular", "compartment"),
    FRONTIER_SPECIES: ("in vivo", "expressed in", "characterized in"),
    FRONTIER_SEED: ("biosynthesis", "metabolism", "pathway"),
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: Frontier kinds whose target is a reaction label ("A + B -> C").
_REACTION_KINDS = {FRONTIER_CONFLICT, FRONTIER_DIRECTION, FRONTIER_COMPARTMENT, FRONTIER_SPECIES}
_ARROW = re.compile(r"\s*(?:<=>|->)\s*")
#: A trailing parenthetical note, e.g. "HDA (16:2n6)" or "allene oxide (12,13-EOT)".
#: As a phrase search it almost never matches, so the note is dropped.
_TRAILING_NOTE = re.compile(r"\s+\([^()]*\)$")


def _clean(term: str) -> str:
    return _TRAILING_NOTE.sub("", str(term or "").replace('"', "").strip()).strip()


def target_terms(item: FrontierItem, config: MiningConfig) -> List[str]:
    """The frontier item's own search terms. Every one is REQUIRED in the query.

    A bridge requires both of its sides; a reaction label requires its first
    substrate and its first product; anything else requires its target. The
    species is never a term here: it is always required separately.
    """
    target = str(item.target or "")
    if item.kind == FRONTIER_CONFLICT and ": " in target:
        target = target.split(": ", 1)[1]  # "<attribute>: <reaction label>"
    if item.kind == FRONTIER_BRIDGE:
        parts = target.split("|")
    elif item.kind in _REACTION_KINDS:
        parts = [side.split(" + ")[0] for side in _ARROW.split(target)]
    else:
        parts = [target]
    species = config.target_species.casefold()
    out: List[str] = []
    for part in parts:
        term = _clean(part)
        if term and term != "?" and term.casefold() != species and term.casefold() not in {
            t.casefold() for t in out
        }:
            out.append(term)
    return out


def formulation(item: FrontierItem, config: MiningConfig, attempt: int = 1) -> Optional[List[str]]:
    """The required terms for query ``attempt`` of ``item``; ``None`` when there is none.

    Attempt 1 requires the item's own terms and the species. Attempt 2 also
    requires the run topic: a narrower query, re-ranked, which reaches papers
    past attempt 1's result window. It exists only when it differs from
    attempt 1. Because attempt 2 only narrows attempt 1, a zero-result attempt 1
    means attempt 2 would also return nothing. When it does not exist, the
    controller may ask attempt 1's question again over a wider window
    (:func:`build_query` with ``deeper=True``).
    """
    terms = target_terms(item, config)
    if not terms and item.kind != FRONTIER_SEED:
        return None  # e.g. "? -> ?": a species-only search would ask nothing of this item
    if attempt <= 1:
        return terms
    topic = _clean(config.topic)
    if attempt == 2 and topic and topic.casefold() not in {t.casefold() for t in terms}:
        return [*terms, topic]
    return None


def build_context(item: FrontierItem, config: MiningConfig, *, attempt: int = 1) -> Optional[Dict[str, Any]]:
    """The acquisition context for one query attempt, or ``None`` if there is no such attempt.

    ``t2pw.rag.acquire.build_query`` renders a context as an OR-list AND the
    organism: ``("a" OR "b") AND "<organism>"``. Mining needs every term
    required, so the terms go in as ONE entry joined with ``" AND "``. That entry
    renders as ``"a" AND "b" AND "<organism>"``, and as
    ``("a" AND "b") AND <organism>[Organism]`` for NCBI. No topic and no generic
    words are OR'd in. ``test_mining_search.py`` pins the rendered string, so a
    change to rag's quoting fails a test instead of silently changing the search.
    """
    terms = formulation(item, config, attempt)
    if terms is None:
        return None
    return {
        "pathway_name": "",
        "likely_organism": config.target_species,
        "key_compounds": ['" AND "'.join(terms)] if terms else [],
        "key_proteins": [],
        "gap_terms": [],
    }


def build_query(
    item: FrontierItem,
    config: MiningConfig,
    *,
    round_index: int,
    attempt: int = 1,
    deeper: bool = False,
) -> Optional[LiteratureQuery]:
    """One traceable :class:`LiteratureQuery` for one attempt, or ``None``.

    ``query_text`` is exactly the EuropePMC query ``search_candidates`` runs for
    :func:`build_context`'s context. With ``deeper``, attempt ``attempt`` asks
    attempt 1's question again (same terms, same text, its own ``query_id``);
    the controller gives it a wider result window.
    """
    shape = 1 if deeper else attempt
    context = build_context(item, config, attempt=shape)
    if context is None:
        return None
    terms = formulation(item, config, shape) or []
    try:
        from t2pw.rag.acquire import build_query as rag_build_query

        text = rag_build_query(context)
    except Exception:  # noqa: BLE001 - never let query rendering break a round
        text = " AND ".join(f'"{t}"' for t in [*terms, config.target_species])

    return LiteratureQuery(
        query_id=make_query_id(f"{text} [deeper]" if deeper else text, round_index),
        query_text=text,
        frontier_item_id=item.frontier_item_id,
        frontier_kind=item.kind,
        round_index=int(round_index),
        sources=list(config.acquisition_sources),
        source_terms=list(terms),
        executed_at=_utcnow(),
        attempt=int(attempt),
    )


def rescue_terms(config: MiningConfig, reactions: Iterable[Any] = ()) -> List[str]:
    """The screen rescue's vocabulary: the declared seeds, plus the substrate,
    product and enzyme names of this run's ``target_species_supported`` reactions.

    Nothing comes from the frontier. In pilot-001 every round-2 frontier item came
    from quarantined Marchantia claims, and their names (EPA, ARA, ALA, ...)
    rescued off-topic Arabidopsis papers. Names are kept as written, including a
    disambiguating note ("ALA (18:3n3)"). The topic, the species and cofactors
    are left out. ``Acquirer.rescue`` also ignores short names and matches
    acronyms case-sensitively.
    """
    skip = {config.topic.strip().casefold(), config.target_species.casefold()}
    names = [str(seed.value or seed.raw) for seed in config.seed_entities]
    for reaction in reactions:
        if getattr(reaction, "species_state", "") == SPECIES_TARGET_SUPPORTED:
            names += [*reaction.substrate_names, *reaction.product_names, *reaction.enzyme_names]
    out: List[str] = []
    for name in names:
        term = " ".join(str(name or "").replace('"', "").split())
        if term and term.casefold() not in skip and not is_cofactor(term) and term not in out:
            out.append(term)
    return out


def retrieval_query_text(item: FrontierItem, config: MiningConfig) -> str:
    """The natural-language string used for *passage* retrieval, not paper search.

    Deliberately different from the boolean literature query: the vector store
    responds to prose, not to ``("x" OR "y") AND "z"``.
    """
    parts = [item.target.replace("|", " and ")]
    parts.extend(_KIND_TERMS.get(item.kind, ())[:2])
    if config.topic:
        parts.append(config.topic)
    parts.append(config.target_species)
    return " ".join(p for p in parts if p)


__all__ = ["target_terms", "formulation", "build_context", "build_query", "rescue_terms", "retrieval_query_text"]
