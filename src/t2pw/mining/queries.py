"""Literature query formulation, traceable back to the frontier item that asked.

Every query this module produces records the frontier item, the kind of question,
and the exact terms that went into it, so a paper in the run can always be traced
to the question that found it.

Reuse, not reimplementation: the query *string* is built by
``t2pw.rag.acquire.build_query`` -- the same organism-scoped OR-clause the
production acquisition path uses -- from a context dict this module assembles.
That keeps one query grammar in the repository instead of two that drift.

Search terms vs biochemistry
----------------------------
A bridge query deliberately puts two metabolite names in the same search. Two
papers coming back for that query are **not** thereby connected, and the
metabolites are **not** thereby a reaction. Search terms find candidate
literature; only an extracted, evidenced reaction can connect anything.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from t2pw.mining.config import MiningConfig
from t2pw.mining.graph import ResearchGraph
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


def _split_bridge_target(target: str) -> List[str]:
    return [part.strip() for part in str(target or "").split("|") if part.strip()]


def build_context(
    item: FrontierItem,
    config: MiningConfig,
    *,
    graph: Optional[ResearchGraph] = None,
) -> Dict[str, Any]:
    """Assemble the acquisition context dict for one frontier item.

    The shape is exactly what ``t2pw.rag.acquire`` already understands:
    ``pathway_name``, ``likely_organism``, ``key_compounds``, ``key_proteins``,
    ``gap_terms``.
    """
    compounds: List[str] = []
    proteins: List[str] = []

    if item.kind == FRONTIER_BRIDGE:
        compounds.extend(_split_bridge_target(item.target))
    elif item.kind in {FRONTIER_ENZYME}:
        proteins.append(item.target)
    elif item.kind in {FRONTIER_CONFLICT, FRONTIER_DIRECTION, FRONTIER_COMPARTMENT, FRONTIER_SPECIES}:
        compounds.extend(
            part.strip()
            for part in str(item.target).replace("->", "|").replace("+", "|").split("|")
            if part.strip()
        )
    else:
        compounds.append(item.target)

    if graph is not None:
        for eid in item.entity_ids:
            entity = graph.registry.get(eid)
            if entity is None:
                continue
            bucket = proteins if entity.is_catalytic else compounds
            if entity.name not in bucket:
                bucket.append(entity.name)

    pathway = config.topic
    if item.kind == FRONTIER_PATHWAY_REFERENCE:
        pathway = item.target

    return {
        "pathway_name": pathway,
        "likely_organism": config.target_species,
        "key_compounds": [c for c in dict.fromkeys(compounds) if c][:6],
        "key_proteins": [p for p in dict.fromkeys(proteins) if p][:4],
        "gap_terms": list(_KIND_TERMS.get(item.kind, ()))[:4],
    }


def build_query(
    item: FrontierItem,
    config: MiningConfig,
    *,
    round_index: int,
    graph: Optional[ResearchGraph] = None,
) -> LiteratureQuery:
    """One traceable :class:`LiteratureQuery` for one frontier item."""
    context = build_context(item, config, graph=graph)
    try:
        from t2pw.rag.acquire import build_query as rag_build_query

        text = rag_build_query(context)
    except Exception:  # noqa: BLE001 - never let query rendering break a round
        core = [*context["key_compounds"], *context["key_proteins"], context["pathway_name"]]
        joined = " OR ".join(f'"{t}"' for t in core if t)
        text = f'({joined}) AND "{config.target_species}"' if joined else f'"{config.target_species}"'

    terms = [
        *context["key_compounds"],
        *context["key_proteins"],
        *context["gap_terms"],
    ]
    if context["pathway_name"]:
        terms.insert(0, context["pathway_name"])

    return LiteratureQuery(
        query_id=make_query_id(text, round_index),
        query_text=text,
        frontier_item_id=item.frontier_item_id,
        frontier_kind=item.kind,
        round_index=int(round_index),
        sources=list(config.acquisition_sources),
        source_terms=[t for t in dict.fromkeys(terms) if t],
        executed_at=_utcnow(),
    )


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


__all__ = ["build_context", "build_query", "retrieval_query_text"]
