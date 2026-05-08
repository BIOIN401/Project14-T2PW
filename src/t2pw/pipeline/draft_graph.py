"""
draft_graph.py

Converts the merged Stage-1 + Stage-2 JSON payload into a DraftGraph — a
clean intermediate representation that all subsequent pipeline stages
(normalization, QA, enrichment, SBML) can operate on.

Nodes represent:
  - Individual entities (compound, protein, protein_complex, nucleic_acid,
    element_collection)
  - Processes (reaction, transport, reaction_coupled_transport, interaction)

Edges represent directed participation roles:
  reactant, product, catalyst, modifier, transporter, cargo, participant
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DraftNode:
    id: str           # normalized key, e.g. "compound:atp"
    label: str        # display name as extracted
    kind: str         # compound | protein | protein_complex | nucleic_acid |
                      # element_collection | reaction | transport |
                      # reaction_coupled_transport | interaction
    cls: str          # from entity .class field (empty string when absent)
    compartment: str  # primary compartment from element_locations (empty when absent)
    confidence: float
    provenance: str   # "extracted" | "inferred"


@dataclass
class DraftEdge:
    source: str       # DraftNode.id
    target: str       # DraftNode.id
    role: str         # reactant | product | catalyst | modifier | transporter | cargo | participant
    confidence: float


@dataclass
class DraftGraph:
    nodes: List[DraftNode] = field(default_factory=list)
    edges: List[DraftEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DraftGraph":
        nodes = [DraftNode(**n) for n in data.get("nodes", [])]
        edges = [DraftEdge(**e) for e in data.get("edges", [])]
        return cls(nodes=nodes, edges=edges, metadata=data.get("metadata", {}))

    def orphan_nodes(self) -> List[DraftNode]:
        """Return nodes that appear in no edge (either as source or target)."""
        connected: Set[str] = set()
        for e in self.edges:
            connected.add(e.source)
            connected.add(e.target)
        return [n for n in self.nodes if n.id not in connected]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_id(name: str) -> str:
    """Lowercase + collapse whitespace + strip non-alphanumeric (except spaces/hyphens)."""
    text = (name or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _entity_node_id(kind: str, name: str) -> str:
    return f"{kind}:{_normalize_id(name)}"


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_str(value: Any) -> str:
    return (value or "").strip() if isinstance(value, str) else ""


def _resolve_actor_name(row: Any) -> str:
    """Extract entity name from an enzyme/transporter/modifier dict."""
    if not isinstance(row, dict):
        return ""
    for key in ["protein", "protein_complex", "entity", "name"]:
        candidate = row.get(key, "")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _resolve_actor_kind(
    name: str,
    proteins: Set[str],
    complexes: Set[str],
    compounds: Set[str],
) -> str:
    n = name.strip()
    if n in complexes:
        return "protein_complex"
    if n in proteins:
        return "protein"
    if n in compounds:
        return "compound"
    # heuristic: complex-style names contain "/" or ":"
    if "/" in n or ":" in n:
        return "protein_complex"
    return "compound"


# ---------------------------------------------------------------------------
# Compartment index builder
# ---------------------------------------------------------------------------

def _build_compartment_index(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns a map from _normalize_id(entity_name) → compartment label.
    Uses element_locations if available; falls back to biological_state on
    entities themselves.
    """
    index: Dict[str, str] = {}
    locations = payload.get("element_locations", {})
    if not isinstance(locations, dict):
        return index

    key_to_entity_field = {
        "compound_locations": "compound",
        "element_collection_locations": "element_collection",
        "nucleic_acid_locations": "nucleic_acid",
        "protein_locations": "protein",
    }

    for loc_key, entity_field in key_to_entity_field.items():
        for item in _safe_list(locations.get(loc_key, [])):
            if not isinstance(item, dict):
                continue
            entity_name = _safe_str(item.get(entity_field))
            bio_state = _safe_str(item.get("biological_state"))
            if entity_name and bio_state:
                norm = _normalize_id(entity_name)
                if norm not in index:  # keep first/primary entry
                    index[norm] = bio_state

    return index


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_draft_graph(merged_json: Dict[str, Any]) -> DraftGraph:
    """
    Convert the merged Stage-1 + Stage-2 JSON payload into a DraftGraph.

    Parameters
    ----------
    merged_json:
        The output of ``pipeline.merge_additions(stage_one, stage_two_additions)``.

    Returns
    -------
    DraftGraph
        Populated with DraftNode and DraftEdge objects.
    """
    graph = DraftGraph()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    pathway_context = merged_json.get("pathway_context") or {}
    graph.metadata = {
        "pathway_name": _safe_str(pathway_context.get("name")),
        "organism": _safe_str(pathway_context.get("organism")),
        "node_count": 0,   # filled after building
        "edge_count": 0,
    }

    entities_raw = merged_json.get("entities") or {}
    processes_raw = merged_json.get("processes") or {}

    # ------------------------------------------------------------------
    # Collect entity name sets for kind resolution
    # ------------------------------------------------------------------
    proteins: Set[str] = set()
    complexes: Set[str] = set()
    compounds: Set[str] = set()

    for item in _safe_list(entities_raw.get("proteins", [])):
        if isinstance(item, dict) and _safe_str(item.get("name")):
            proteins.add(item["name"].strip())
    for item in _safe_list(entities_raw.get("protein_complexes", [])):
        if isinstance(item, dict) and _safe_str(item.get("name")):
            complexes.add(item["name"].strip())
    for item in _safe_list(entities_raw.get("compounds", [])):
        if isinstance(item, dict) and _safe_str(item.get("name")):
            compounds.add(item["name"].strip())

    # ------------------------------------------------------------------
    # Compartment index
    # ------------------------------------------------------------------
    compartment_index = _build_compartment_index(merged_json)

    # ------------------------------------------------------------------
    # Track which node IDs have been added (dedupe)
    # ------------------------------------------------------------------
    node_ids: Set[str] = set()

    def _add_entity_node(
        kind: str,
        name: str,
        cls: str = "",
        confidence: float = 1.0,
        provenance: str = "extracted",
    ) -> str:
        nid = _entity_node_id(kind, name)
        if nid not in node_ids:
            node_ids.add(nid)
            compartment = compartment_index.get(_normalize_id(name), "")
            graph.nodes.append(DraftNode(
                id=nid,
                label=name,
                kind=kind,
                cls=cls,
                compartment=compartment,
                confidence=confidence,
                provenance=provenance,
            ))
        return nid

    def _add_edge(source: str, target: str, role: str, confidence: float = 1.0) -> None:
        graph.edges.append(DraftEdge(source=source, target=target, role=role, confidence=confidence))

    # ------------------------------------------------------------------
    # Entity nodes
    # ------------------------------------------------------------------
    entity_kind_map = {
        "compounds": "compound",
        "proteins": "protein",
        "protein_complexes": "protein_complex",
        "nucleic_acids": "nucleic_acid",
        "element_collections": "element_collection",
    }

    for entities_key, kind in entity_kind_map.items():
        for item in _safe_list(entities_raw.get(entities_key, [])):
            if not isinstance(item, dict):
                continue
            name = _safe_str(item.get("name"))
            if not name:
                continue
            cls = _safe_str(item.get("class"))
            _add_entity_node(kind, name, cls=cls)

    # ------------------------------------------------------------------
    # Process nodes + edges
    # ------------------------------------------------------------------

    # --- Reactions ---
    reactions = _safe_list(processes_raw.get("reactions", []))
    for i, r in enumerate(reactions):
        if not isinstance(r, dict):
            continue
        rxn_name = _safe_str(r.get("name")) or f"reaction_{i + 1}"
        rxn_id = f"reaction:{_normalize_id(rxn_name)}"
        # Make reaction node ID unique when name is duplicated
        while rxn_id in node_ids:
            rxn_id = f"{rxn_id}_{i + 1}"
        node_ids.add(rxn_id)

        bio_state = _safe_str(r.get("biological_state"))
        graph.nodes.append(DraftNode(
            id=rxn_id,
            label=rxn_name,
            kind="reaction",
            cls="",
            compartment=bio_state,
            confidence=1.0,
            provenance="extracted",
        ))

        for inp in _safe_list(r.get("inputs", [])):
            inp = _safe_str(inp)
            if not inp:
                continue
            entity_kind = _resolve_actor_kind(inp, proteins, complexes, compounds)
            src_id = _add_entity_node(entity_kind, inp)
            _add_edge(src_id, rxn_id, "reactant")

        for out in _safe_list(r.get("outputs", [])):
            out = _safe_str(out)
            if not out:
                continue
            entity_kind = _resolve_actor_kind(out, proteins, complexes, compounds)
            tgt_id = _add_entity_node(entity_kind, out)
            _add_edge(rxn_id, tgt_id, "product")

        for enz in _safe_list(r.get("enzymes", [])) + _safe_list(r.get("modifiers", [])):
            actor_name = _resolve_actor_name(enz)
            if not actor_name:
                continue
            actor_kind = _resolve_actor_kind(actor_name, proteins, complexes, compounds)
            actor_id = _add_entity_node(actor_kind, actor_name, provenance="extracted")
            role = _safe_str((enz or {}).get("role")) or "catalyst"
            confidence = float((enz or {}).get("confidence", 1.0)) if isinstance(enz, dict) else 1.0
            _add_edge(actor_id, rxn_id, role, confidence=confidence)

    # --- Transports ---
    transports = _safe_list(processes_raw.get("transports", []))
    for i, t in enumerate(transports):
        if not isinstance(t, dict):
            continue
        t_name = _safe_str(t.get("name")) or f"transport_{i + 1}"
        t_id = f"transport:{_normalize_id(t_name)}"
        while t_id in node_ids:
            t_id = f"{t_id}_{i + 1}"
        node_ids.add(t_id)

        from_state = _safe_str(t.get("from_biological_state")) or "unspecified"
        to_state = _safe_str(t.get("to_biological_state")) or "unspecified"
        graph.nodes.append(DraftNode(
            id=t_id,
            label=t_name,
            kind="transport",
            cls="",
            compartment=f"{from_state} → {to_state}",
            confidence=1.0,
            provenance="extracted",
        ))

        cargo_name = _safe_str(t.get("cargo") or t.get("cargo_complex"))
        if not cargo_name:
            for ews in _safe_list(t.get("elements_with_states", [])):
                if isinstance(ews, dict):
                    cargo_name = _safe_str(ews.get("element"))
                    if cargo_name:
                        break
        if cargo_name:
            cargo_kind = _resolve_actor_kind(cargo_name, proteins, complexes, compounds)
            cargo_id = _add_entity_node(cargo_kind, cargo_name)
            _add_edge(cargo_id, t_id, "cargo")

        for tr in _safe_list(t.get("transporters", [])):
            actor_name = _resolve_actor_name(tr)
            if not actor_name:
                continue
            actor_kind = _resolve_actor_kind(actor_name, proteins, complexes, compounds)
            actor_id = _add_entity_node(actor_kind, actor_name)
            _add_edge(actor_id, t_id, "transporter")

    # --- Reaction-coupled transports ---
    rcts = _safe_list(processes_raw.get("reaction_coupled_transports", []))
    for i, rct in enumerate(rcts):
        if not isinstance(rct, dict):
            continue
        rct_name = _safe_str(rct.get("name")) or f"rct_{i + 1}"
        rct_id = f"reaction_coupled_transport:{_normalize_id(rct_name)}"
        while rct_id in node_ids:
            rct_id = f"{rct_id}_{i + 1}"
        node_ids.add(rct_id)

        graph.nodes.append(DraftNode(
            id=rct_id,
            label=rct_name,
            kind="reaction_coupled_transport",
            cls="",
            compartment="",
            confidence=1.0,
            provenance="extracted",
        ))

        for enz in _safe_list(rct.get("enzymes", [])):
            actor_name = _resolve_actor_name(enz)
            if not actor_name:
                continue
            actor_kind = _resolve_actor_kind(actor_name, proteins, complexes, compounds)
            actor_id = _add_entity_node(actor_kind, actor_name)
            _add_edge(actor_id, rct_id, "catalyst")

        for ews in _safe_list(rct.get("elements_with_states", [])):
            if isinstance(ews, dict):
                el = _safe_str(ews.get("element"))
                if el:
                    el_kind = _resolve_actor_kind(el, proteins, complexes, compounds)
                    el_id = _add_entity_node(el_kind, el)
                    _add_edge(el_id, rct_id, "participant")

    # --- Interactions ---
    interactions = _safe_list(processes_raw.get("interactions", []))
    for i, inter in enumerate(interactions):
        if not isinstance(inter, dict):
            continue
        inter_name = _safe_str(inter.get("name")) or f"interaction_{i + 1}"
        inter_id = f"interaction:{_normalize_id(inter_name)}"
        while inter_id in node_ids:
            inter_id = f"{inter_id}_{i + 1}"
        node_ids.add(inter_id)

        rel = _safe_str(inter.get("relationship")) or "interaction"
        graph.nodes.append(DraftNode(
            id=inter_id,
            label=inter_name,
            kind="interaction",
            cls=rel,
            compartment="",
            confidence=1.0,
            provenance="extracted",
        ))

        e1 = _safe_str(inter.get("entity_1"))
        e2 = _safe_str(inter.get("entity_2"))
        if e1:
            k1 = _resolve_actor_kind(e1, proteins, complexes, compounds)
            id1 = _add_entity_node(k1, e1)
            _add_edge(id1, inter_id, "participant")
        if e2:
            k2 = _resolve_actor_kind(e2, proteins, complexes, compounds)
            id2 = _add_entity_node(k2, e2)
            _add_edge(inter_id, id2, "participant")

    # ------------------------------------------------------------------
    # Finalise metadata counts
    # ------------------------------------------------------------------
    graph.metadata["node_count"] = len(graph.nodes)
    graph.metadata["edge_count"] = len(graph.edges)

    return graph
