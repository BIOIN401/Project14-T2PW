"""Serialization for the two research views, plus DOT export.

Two views, deliberately different graphs
----------------------------------------
**Reaction network** -- metabolites, reaction nodes and enzyme annotations.
Edges here are *biochemical*: substrate into a reaction, reaction out to a
product. An enzyme edge is drawn as an annotation and is typed
``annotation``/``catalyzes`` so no renderer can treat it as a path through the
chemistry.

**Paper / evidence provenance** -- papers, evidence spans, claims and the
reactions they support. Edges here are *bibliographic*. Two reactions that share
a paper are **not** connected; the payload carries
``provenance_edges_are_not_chemical_connectivity: true`` and every renderer in
this package states it on screen.

Both payloads are plain JSON with no library dependency, so they serialize into
the run directory, render in Streamlit, and can be fed to anything else later.
Multi-substrate and multi-product reactions keep one reaction node with all its
incident edges -- never a chain of pairwise metabolite edges.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from t2pw.mining.graph import (
    GRAPH_PRIMARY,
    ResearchGraph,
    ResearchReaction,
)
from t2pw.mining.models import (
    ClaimDecision,
    ConflictRecord,
    EvidenceSpan,
    ExtractedClaim,
    PaperRecord,
)
from t2pw.mining.normalize import CATALYTIC_TYPES

VIEW_SCHEMA_VERSION = "t2pw.mining.view/1.0.0"

NODE_METABOLITE = "metabolite"
NODE_REACTION = "reaction"
NODE_ENZYME = "enzyme"
NODE_PAPER = "paper"
NODE_EVIDENCE = "evidence"
NODE_CLAIM = "claim"

EDGE_SUBSTRATE = "substrate_of"
EDGE_PRODUCT = "product_of"
EDGE_CATALYZES = "catalyzes"
EDGE_SUPPORTS = "supports"
EDGE_CITES = "cites"
EDGE_EVIDENCES = "evidences"


def _reaction_node(
    reaction: ResearchReaction,
    *,
    conflicts: Mapping[str, ConflictRecord],
    papers: Mapping[str, PaperRecord],
) -> Dict[str, Any]:
    supporting = [
        {
            "paper_id": pid,
            "title": papers[pid].title if pid in papers else "",
            "year": papers[pid].year if pid in papers else "",
            "evidence_type": papers[pid].evidence_type if pid in papers else "unknown",
        }
        for pid in reaction.paper_ids
    ]
    return {
        "id": reaction.reaction_id,
        "type": NODE_REACTION,
        "label": reaction.label(),
        "graph_status": reaction.graph_status,
        "component_id": reaction.component_id,
        "discovery_round": reaction.discovery_round,
        "promoted_round": reaction.promoted_round,
        "direction": reaction.direction,
        "reversible": reaction.reversible,
        "compartment": reaction.compartment,
        "conditions": reaction.conditions,
        "tissue": reaction.tissue,
        "species": reaction.species_text,
        "species_state": reaction.species_state,
        "enzymes": list(reaction.enzyme_names),
        "gene": reaction.gene,
        "pathway_names": list(reaction.pathway_names),
        "decision_status": reaction.decision_status,
        "decision_reason": reaction.decision_reason,
        "evidence_level": reaction.evidence_level,
        "hypothesis": reaction.hypothesis,
        "multi_paper_supported": reaction.multi_paper_supported,
        "paper_count": reaction.independent_paper_count,
        "supporting_papers": supporting,
        "claim_ids": list(reaction.claim_ids),
        "evidence_ids": list(reaction.evidence_ids),
        "conflicts": [
            conflicts[c].to_dict() for c in reaction.conflict_ids if c in conflicts
        ],
        "external_validation": dict(reaction.external_validation),
        "history": [h.to_dict() for h in reaction.history],
    }


def reaction_network(
    graph: ResearchGraph,
    *,
    papers: Optional[Mapping[str, PaperRecord]] = None,
    conflicts: Optional[Mapping[str, ConflictRecord]] = None,
) -> Dict[str, Any]:
    """The biochemical view. Reactions are nodes; enzymes are annotations."""
    papers = dict(papers or {})
    conflicts = dict(conflicts or {})
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_entities: set = set()

    def entity_node(entity_id: str, role: str) -> Optional[Dict[str, Any]]:
        entity = graph.registry.get(entity_id)
        if entity is None:
            return None
        if entity_id in seen_entities:
            return None
        seen_entities.add(entity_id)
        node_type = NODE_ENZYME if entity.entity_type in CATALYTIC_TYPES else NODE_METABOLITE
        node = {
            "id": entity.entity_id,
            "type": node_type,
            "label": entity.name,
            "entity_type": entity.entity_type,
            "aliases": list(entity.aliases),
            "first_round": entity.first_round,
            "paper_ids": list(entity.paper_ids),
            "evidence_ids": list(entity.evidence_ids),
            "external_refs": dict(entity.external_refs),
            "role_hint": role,
        }
        nodes.append(node)
        return node

    for reaction in sorted(graph.reactions.values(), key=lambda r: r.reaction_id):
        nodes.append(_reaction_node(reaction, conflicts=conflicts, papers=papers))
        for eid in reaction.substrate_ids:
            entity_node(eid, "substrate")
            edges.append(
                {
                    "source": eid,
                    "target": reaction.reaction_id,
                    "type": EDGE_SUBSTRATE,
                    "chemical": True,
                    "graph_status": reaction.graph_status,
                    "round": reaction.discovery_round,
                }
            )
        for eid in reaction.product_ids:
            entity_node(eid, "product")
            edges.append(
                {
                    "source": reaction.reaction_id,
                    "target": eid,
                    "type": EDGE_PRODUCT,
                    "chemical": True,
                    "graph_status": reaction.graph_status,
                    "round": reaction.discovery_round,
                }
            )
        for eid in reaction.enzyme_entity_ids:
            entity_node(eid, "enzyme")
            edges.append(
                {
                    "source": eid,
                    "target": reaction.reaction_id,
                    "type": EDGE_CATALYZES,
                    # NOT a chemical edge. An enzyme never bridges two reactions.
                    "chemical": False,
                    "annotation": True,
                    "graph_status": reaction.graph_status,
                    "round": reaction.discovery_round,
                }
            )

    rounds = sorted({r.discovery_round for r in graph.reactions.values()} | {0})
    return {
        "schema_version": VIEW_SCHEMA_VERSION,
        "view": "reaction_network",
        "note": (
            "Reaction nodes carry all their substrates and products; there are no pairwise "
            "metabolite edges. Enzyme edges are annotations (chemical=false) and never "
            "connect two reactions."
        ),
        "primary_component_id": graph.primary_component_id,
        "primary_selection_rationale": graph.primary_selection_rationale,
        "rounds": rounds,
        "nodes": nodes,
        "edges": edges,
        "connections": [c.to_dict() for c in graph.connections],
        "components": {
            cid: {
                "component_id": cid,
                "promoted": component.promoted,
                "created_round": component.created_round,
                "promoted_round": component.promoted_round,
                "relevance_score": component.relevance_score,
                "rationale": component.rationale,
                "promotion_history": list(component.promotion_history),
                "bridge_reaction_ids": list(component.bridge_reaction_ids),
                "is_primary": cid == graph.primary_component_id,
            }
            for cid, component in sorted(graph.components.items())
        },
        "counts": {
            "reactions": len(graph.reactions),
            "primary_reactions": len(graph.primary_reactions),
            "candidate_reactions": len(graph.candidate_reactions),
            "metabolites": sum(1 for n in nodes if n["type"] == NODE_METABOLITE),
            "enzymes": sum(1 for n in nodes if n["type"] == NODE_ENZYME),
        },
    }


def provenance_network(
    *,
    graph: ResearchGraph,
    papers: Mapping[str, PaperRecord],
    claims: Mapping[str, ExtractedClaim],
    evidence: Mapping[str, EvidenceSpan],
    conflicts: Optional[Mapping[str, ConflictRecord]] = None,
    include_excluded: bool = True,
) -> Dict[str, Any]:
    """The bibliographic view: paper -> evidence span -> claim -> reaction."""
    conflicts = dict(conflicts or {})
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    reaction_by_claim: Dict[str, ResearchReaction] = {}
    for reaction in graph.reactions.values():
        for cid in reaction.claim_ids:
            reaction_by_claim[cid] = reaction

    first_paper_for: Dict[str, str] = {}
    for reaction in graph.reactions.values():
        earliest = None
        for cid in reaction.claim_ids:
            claim = claims.get(cid)
            if claim is None:
                continue
            if earliest is None or claim.first_seen_round < earliest.first_seen_round:
                earliest = claim
        if earliest is not None:
            first_paper_for[reaction.reaction_id] = earliest.paper_id

    bridge_papers = {
        pid
        for component in graph.components.values()
        for pid in component.bridge_paper_ids
    }
    candidate_papers = {
        pid
        for component in graph.candidate_components
        for rid in component.reaction_ids
        if rid in graph.reactions
        for pid in graph.reactions[rid].paper_ids
    }

    for paper_id, paper in sorted(papers.items()):
        excluded = bool(paper.exclusion_reasons)
        if excluded and not include_excluded:
            continue
        nodes.append(
            {
                "id": paper_id,
                "type": NODE_PAPER,
                "label": paper.title or paper_id,
                "pmid": paper.pmid,
                "pmcid": paper.pmcid,
                "doi": paper.doi,
                "url": paper.url,
                "journal": paper.journal,
                "year": paper.year,
                "authors": list(paper.authors),
                "evidence_type": paper.evidence_type,
                "evidence_type_basis": paper.evidence_type_basis,
                "acquisition_source": paper.acquisition_source,
                "discovered_round": paper.discovered_round,
                "discovered_via_query_id": paper.discovered_via_query_id,
                "discovered_via_frontier_id": paper.discovered_via_frontier_id,
                "screening_outcome": paper.screening_outcome,
                "inclusion_reasons": list(paper.inclusion_reasons),
                "exclusion_reasons": list(paper.exclusion_reasons),
                "excluded": excluded,
                "file_sha256": paper.file_sha256,
                "is_bridge_paper": paper_id in bridge_papers,
                "supports_candidate_subgraph": paper_id in candidate_papers,
            }
        )

    for eid, span in sorted(evidence.items()):
        if span.paper_id not in papers and not include_excluded:
            continue
        nodes.append(
            {
                "id": eid,
                "type": NODE_EVIDENCE,
                "label": (span.text[:120] + "...") if len(span.text) > 120 else span.text,
                "text": span.text,
                "paper_id": span.paper_id,
                "section": span.section,
                "page": span.page,
                "paragraph": span.paragraph,
                "locator": span.locator,
                "chunk_id": span.chunk_id,
                "char_start": span.char_start,
                "char_end": span.char_end,
                "round": span.extraction_round,
                "retrieval_query_id": span.retrieval_query_id,
                "frontier_item_id": span.frontier_item_id,
            }
        )
        edges.append(
            {"source": span.paper_id, "target": eid, "type": EDGE_CITES, "chemical": False}
        )

    for cid, claim in sorted(claims.items()):
        reaction = reaction_by_claim.get(cid)
        nodes.append(
            {
                "id": cid,
                "type": NODE_CLAIM,
                "label": f"{claim.subject} -> {claim.object}",
                "paper_id": claim.paper_id,
                "claim_identity": claim.claim_identity,
                "species_state": claim.species_state,
                "species_reason": claim.species_reason,
                "extraction_status": claim.extraction_status,
                "hypothesis": claim.hypothesis,
                "round": claim.first_seen_round,
                "model": claim.model,
                "actual_model": claim.actual_model,
                "prompt_version": claim.prompt_version,
                "field_evidence": {k: list(v) for k, v in claim.field_evidence.items()},
                "conflicts": [
                    conflicts[c].to_dict() for c in claim.conflicts_with if c in conflicts
                ],
                "reaction_id": reaction.reaction_id if reaction else "",
            }
        )
        for eid in claim.evidence_ids:
            edges.append(
                {"source": eid, "target": cid, "type": EDGE_EVIDENCES, "chemical": False}
            )
        if reaction is not None:
            edges.append(
                {
                    "source": cid,
                    "target": reaction.reaction_id,
                    "type": EDGE_SUPPORTS,
                    "chemical": False,
                    "first_support": first_paper_for.get(reaction.reaction_id) == claim.paper_id,
                }
            )

    for reaction in sorted(graph.reactions.values(), key=lambda r: r.reaction_id):
        nodes.append(_reaction_node(reaction, conflicts=conflicts, papers=papers))

    return {
        "schema_version": VIEW_SCHEMA_VERSION,
        "view": "paper_evidence_provenance",
        "provenance_edges_are_not_chemical_connectivity": True,
        "note": (
            "Edges in this view are bibliographic: paper -> evidence span -> claim -> "
            "reaction. Two reactions sharing a paper are NOT chemically connected. Only the "
            "reaction network view shows biochemical connectivity."
        ),
        "nodes": nodes,
        "edges": edges,
        "first_support": first_paper_for,
        "counts": {
            "papers": sum(1 for n in nodes if n["type"] == NODE_PAPER),
            "papers_excluded": sum(
                1 for n in nodes if n["type"] == NODE_PAPER and n.get("excluded")
            ),
            "evidence_spans": sum(1 for n in nodes if n["type"] == NODE_EVIDENCE),
            "claims": sum(1 for n in nodes if n["type"] == NODE_CLAIM),
            "reactions": sum(1 for n in nodes if n["type"] == NODE_REACTION),
        },
    }


def evidence_table(
    *,
    graph: ResearchGraph,
    papers: Mapping[str, PaperRecord],
    claims: Mapping[str, ExtractedClaim],
    evidence: Mapping[str, EvidenceSpan],
    decisions: Sequence[ClaimDecision] = (),
) -> List[Dict[str, Any]]:
    """The flat audit table. One row per (reaction, claim, evidence span).

    A diagram is not enough for exact auditing; this is the thing a reviewer
    actually reads.
    """
    latest: Dict[str, ClaimDecision] = {}
    for decision in decisions:
        latest[decision.claim_identity] = decision

    rows: List[Dict[str, Any]] = []
    for reaction in sorted(graph.reactions.values(), key=lambda r: r.reaction_id):
        decision = latest.get(reaction.claim_identity)
        for cid in reaction.claim_ids:
            claim = claims.get(cid)
            if claim is None:
                continue
            paper = papers.get(claim.paper_id)
            for eid in claim.evidence_ids:
                span = evidence.get(eid)
                rows.append(
                    {
                        "reaction_id": reaction.reaction_id,
                        "reaction": reaction.label(),
                        "graph_status": reaction.graph_status,
                        "component_id": reaction.component_id,
                        "discovery_round": reaction.discovery_round,
                        "promoted_round": reaction.promoted_round,
                        "species": reaction.species_text,
                        "species_state": reaction.species_state,
                        "enzyme": "; ".join(reaction.enzyme_names),
                        "compartment": reaction.compartment,
                        "direction": reaction.direction,
                        "reversible": reaction.reversible,
                        "decision_status": decision.status if decision else reaction.decision_status,
                        "decision_reason": decision.reason if decision else reaction.decision_reason,
                        "evidence_level": decision.evidence_level if decision else "",
                        "claim_id": cid,
                        "paper_id": claim.paper_id,
                        "paper_title": paper.title if paper else "",
                        "paper_year": paper.year if paper else "",
                        "evidence_type": paper.evidence_type if paper else "",
                        "source_identifier": (
                            paper.pmcid or paper.pmid or paper.doi or paper.url or paper.file_sha256
                            if paper
                            else ""
                        ),
                        "evidence_id": eid,
                        "section": span.section if span else "",
                        "locator": span.locator if span else "",
                        "quote": span.text if span else "",
                        "retrieval_query_id": span.retrieval_query_id if span else "",
                        "frontier_item_id": span.frontier_item_id if span else "",
                        "conflicts": "; ".join(reaction.conflict_ids),
                        "external_validation": reaction.external_validation.get("status", ""),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# DOT export
# ---------------------------------------------------------------------------
def _dot_escape(text: Any) -> str:
    return str(text or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")[:120]


def reaction_network_dot(view: Mapping[str, Any]) -> str:
    """Graphviz DOT for the reaction view. Text only; no ``dot`` binary needed."""
    lines = [
        "digraph reaction_network {",
        '  graph [rankdir=LR, labelloc="t", '
        'label="Reaction network - solid = chemical, dashed = enzyme annotation"];',
        "  node [fontname=\"Helvetica\", fontsize=10];",
    ]
    for node in view.get("nodes", []):
        nid = _dot_escape(node.get("id"))
        label = _dot_escape(node.get("label"))
        ntype = node.get("type")
        if ntype == NODE_REACTION:
            primary = node.get("graph_status") == GRAPH_PRIMARY
            style = "filled" if primary else "filled,dashed"
            fill = "#cfe8d5" if primary else "#f3e2c7"
            lines.append(
                f'  "{nid}" [shape=box, style="{style}", fillcolor="{fill}", '
                f'label="{label}\\nround {node.get("discovery_round")}"];'
            )
        elif ntype == NODE_ENZYME:
            lines.append(
                f'  "{nid}" [shape=hexagon, style=filled, fillcolor="#e6e0f8", label="{label}"];'
            )
        else:
            lines.append(
                f'  "{nid}" [shape=ellipse, style=filled, fillcolor="#e8f0fb", label="{label}"];'
            )
    for edge in view.get("edges", []):
        source = _dot_escape(edge.get("source"))
        target = _dot_escape(edge.get("target"))
        if edge.get("type") == EDGE_CATALYZES:
            lines.append(
                f'  "{source}" -> "{target}" [style=dashed, color="#7a6fb0", '
                'arrowhead=odot, label="catalyzes"];'
            )
        else:
            lines.append(f'  "{source}" -> "{target}" [color="#34506b"];')
    lines.append("}")
    return "\n".join(lines)


def provenance_network_dot(view: Mapping[str, Any]) -> str:
    """Graphviz DOT for the provenance view, with the caveat in the graph label."""
    lines = [
        "digraph provenance_network {",
        '  graph [rankdir=LR, labelloc="t", label="Paper -> evidence -> claim -> reaction. '
        'These edges are bibliographic; they are NOT chemical connectivity."];',
        "  node [fontname=\"Helvetica\", fontsize=10];",
    ]
    shapes = {
        NODE_PAPER: ("note", "#fdf0d5"),
        NODE_EVIDENCE: ("parallelogram", "#eef7ea"),
        NODE_CLAIM: ("diamond", "#f6e7f2"),
        NODE_REACTION: ("box", "#cfe8d5"),
    }
    for node in view.get("nodes", []):
        shape, fill = shapes.get(str(node.get("type")), ("ellipse", "#eeeeee"))
        style = "filled,dashed" if node.get("excluded") else "filled"
        lines.append(
            f'  "{_dot_escape(node.get("id"))}" [shape={shape}, style="{style}", '
            f'fillcolor="{fill}", label="{_dot_escape(node.get("label"))}"];'
        )
    for edge in view.get("edges", []):
        lines.append(
            f'  "{_dot_escape(edge.get("source"))}" -> "{_dot_escape(edge.get("target"))}" '
            f'[label="{_dot_escape(edge.get("type"))}", color="#8a8a8a"];'
        )
    lines.append("}")
    return "\n".join(lines)


def filter_reaction_network(
    view: Mapping[str, Any],
    *,
    max_round: Optional[int] = None,
    graph_status: Optional[str] = None,
    paper_id: str = "",
    enzyme: str = "",
    species_state: str = "",
    only_multi_paper: bool = False,
) -> Dict[str, Any]:
    """Apply the view filters the UI exposes, server-side and testable.

    ``max_round`` is the round slider: it yields the network as it stood at the
    end of that round, so growth over time is inspectable.
    """
    nodes = list(view.get("nodes", []))
    reactions = [n for n in nodes if n.get("type") == NODE_REACTION]

    def keep(reaction: Mapping[str, Any]) -> bool:
        if max_round is not None and int(reaction.get("discovery_round", 0)) > int(max_round):
            return False
        if graph_status:
            status = reaction.get("graph_status")
            promoted = reaction.get("promoted_round", -1)
            if max_round is not None and status == GRAPH_PRIMARY and promoted > int(max_round):
                status = "candidate"
            if status != graph_status:
                return False
        if paper_id and paper_id not in [p.get("paper_id") for p in reaction.get("supporting_papers", [])]:
            return False
        if enzyme and enzyme.lower() not in " ".join(reaction.get("enzymes", [])).lower():
            return False
        if species_state and reaction.get("species_state") != species_state:
            return False
        if only_multi_paper and not reaction.get("multi_paper_supported"):
            return False
        return True

    kept = {r["id"] for r in reactions if keep(r)}
    edges = [
        e
        for e in view.get("edges", [])
        if e.get("source") in kept or e.get("target") in kept
    ]
    touched = kept | {e.get("source") for e in edges} | {e.get("target") for e in edges}
    out_nodes = [n for n in nodes if n.get("id") in touched]
    return {
        **{k: v for k, v in view.items() if k not in {"nodes", "edges"}},
        "nodes": out_nodes,
        "edges": edges,
        "filtered": True,
        "filter": {
            "max_round": max_round,
            "graph_status": graph_status,
            "paper_id": paper_id,
            "enzyme": enzyme,
            "species_state": species_state,
            "only_multi_paper": only_multi_paper,
        },
    }


__all__ = [
    "VIEW_SCHEMA_VERSION",
    "NODE_METABOLITE",
    "NODE_REACTION",
    "NODE_ENZYME",
    "NODE_PAPER",
    "NODE_EVIDENCE",
    "NODE_CLAIM",
    "EDGE_SUBSTRATE",
    "EDGE_PRODUCT",
    "EDGE_CATALYZES",
    "EDGE_SUPPORTS",
    "EDGE_CITES",
    "EDGE_EVIDENCES",
    "reaction_network",
    "provenance_network",
    "evidence_table",
    "reaction_network_dot",
    "provenance_network_dot",
    "filter_reaction_network",
]
