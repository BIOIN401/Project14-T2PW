"""
reaction_summary.py

Generates a human-readable plain-text summary of a DraftGraph's reactions and
transports, annotated with QA flags from the QA report.

This summary is consumed by:
  - The enrichment agent (gap_resolver._run_enrichment_agent) as additional
    context alongside the raw JSON payload.
  - The Streamlit UI "Pathway Summary" tab for manual inspection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_reaction_summary(draft_graph: Any, qa_report: Dict[str, Any]) -> str:
    """
    Convert a DraftGraph + QA report into a human-readable pathway summary.

    Parameters
    ----------
    draft_graph:
        A ``DraftGraph`` instance (from draft_graph.py).
    qa_report:
        The dict produced by ``qa_graph.generate_qa_report()``.

    Returns
    -------
    str
        Plain text formatted as shown in the example below::

            === PATHWAY: Glycolysis ===
            Organism: Homo sapiens

            REACTION 1: glucose phosphorylation
              Reactants: glucose [cytosol], ATP [cytosol]
              Products:  glucose-6-phosphate [cytosol], ADP [cytosol]
              Modifier:  hexokinase [cytosol] (role=catalyst, confidence=1.0)
              QA flags:  none

            TRANSPORT 1: pyruvate import
              Cargo: pyruvate
              From: cytosol
              To:   mitochondrial_matrix
              Transporter: MPC1/MPC2 complex [plasma_membrane] (role=transporter)
              QA flags:  possible_complex — MPC1/MPC2 not represented as protein_complex
    """
    if not isinstance(qa_report, dict):
        qa_report = {}

    lines: List[str] = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    metadata = getattr(draft_graph, "metadata", {}) or {}
    pathway_name = (metadata.get("pathway_name") or "").strip() or "Unnamed Pathway"
    organism = (metadata.get("organism") or "").strip() or "Unknown Organism"

    lines.append(f"=== PATHWAY: {pathway_name} ===")
    lines.append(f"Organism: {organism}")
    lines.append("")

    # ------------------------------------------------------------------
    # Build lookup structures
    # ------------------------------------------------------------------
    nodes = list(getattr(draft_graph, "nodes", []))
    edges = list(getattr(draft_graph, "edges", []))

    node_by_id: Dict[str, Any] = {n.id: n for n in nodes}

    # For each process node: edges whose *target* is the process (incoming)
    incoming: Dict[str, List[Tuple[Any, str, float]]] = {}
    # For each process node: edges whose *source* is the process (outgoing)
    outgoing: Dict[str, List[Tuple[Any, str, float]]] = {}

    PROCESS_KINDS = {"reaction", "transport", "reaction_coupled_transport", "interaction"}

    for edge in edges:
        src = node_by_id.get(edge.source)
        tgt = node_by_id.get(edge.target)
        if src is None or tgt is None:
            continue
        conf = float(edge.confidence) if edge.confidence is not None else 1.0
        if tgt.kind in PROCESS_KINDS:
            incoming.setdefault(tgt.id, []).append((src, edge.role, conf))
        if src.kind in PROCESS_KINDS:
            outgoing.setdefault(src.id, []).append((tgt, edge.role, conf))

    # ------------------------------------------------------------------
    # QA flag indexes (keyed on lowercased names)
    # ------------------------------------------------------------------
    flags = qa_report.get("flags", {}) if isinstance(qa_report, dict) else {}

    # missing_modifier reactions
    missing_modifier_reactions: Dict[str, str] = {
        (item.get("reaction") or "").lower(): (item.get("reason") or "no enzyme found for this reaction")
        for item in (flags.get("missing_modifiers") or [])
        if isinstance(item, dict) and item.get("reaction")
    }

    # empty reactions
    empty_reaction_names = {
        (item.get("reaction") or "").lower()
        for item in (flags.get("empty_reactions") or [])
        if isinstance(item, dict)
    }

    # transport-like reactions (flagged as potential transports)
    transport_like_reactions: Dict[str, str] = {
        (item.get("reaction") or "").lower(): (item.get("reason") or "reaction name implies transport")
        for item in (flags.get("transport_like_reactions") or [])
        if isinstance(item, dict) and item.get("reaction")
    }

    # possible complexes (entity names)
    possible_complex_entities = {
        (item.get("entity") or "").lower()
        for item in (flags.get("possible_complexes") or [])
        if isinstance(item, dict)
    }

    # ------------------------------------------------------------------
    # Helper: format a single entity node for display
    # ------------------------------------------------------------------
    def _entity_str(node: Any, role: Optional[str] = None, confidence: float = 1.0) -> str:
        label = (node.label or "").strip() or node.id
        compartment = (node.compartment or "").strip() or "unknown"
        parts = f"{label} [{compartment}]"

        if role and role not in ("reactant", "product", "cargo"):
            parts += f" (role={role}, confidence={confidence:.1f})"

        if confidence < 1.0:
            parts += " [inferred]"

        return parts

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------
    reactions = [n for n in nodes if n.kind == "reaction"]
    for i, rxn in enumerate(reactions, 1):
        rxn_key = (rxn.label or "").lower()
        lines.append(f"REACTION {i}: {rxn.label}")

        in_edges = incoming.get(rxn.id, [])
        out_edges = outgoing.get(rxn.id, [])

        reactants = [(n, conf) for n, role, conf in in_edges if role == "reactant"]
        products = [(n, conf) for n, role, conf in out_edges if role == "product"]
        modifiers = [(n, role, conf) for n, role, conf in in_edges
                     if role in ("catalyst", "modifier")]

        # Reactants
        if reactants:
            react_strs = [_entity_str(n, confidence=conf) for n, conf in reactants]
            lines.append(f"  Reactants: {', '.join(react_strs)}")
        else:
            lines.append("  Reactants: MISSING")

        # Products
        if products:
            prod_strs = [_entity_str(n, confidence=conf) for n, conf in products]
            lines.append(f"  Products:  {', '.join(prod_strs)}")
        else:
            lines.append("  Products:  MISSING")

        # Modifiers / catalysts
        if modifiers:
            mod_strs = []
            for n, role, conf in modifiers:
                s = _entity_str(n, role=role, confidence=conf)
                if n.label.lower() in possible_complex_entities:
                    s += " [possible_complex]"
                mod_strs.append(s)
            lines.append(f"  Modifier:  {', '.join(mod_strs)}")
        else:
            lines.append("  Modifier:  MISSING")

        # QA flags
        rxn_flags: List[str] = []
        if rxn_key in missing_modifier_reactions:
            rxn_flags.append(f"missing_modifier — {missing_modifier_reactions[rxn_key]}")
        if rxn_key in empty_reaction_names:
            rxn_flags.append("empty_reaction — no inputs or outputs")
        if rxn_key in transport_like_reactions:
            rxn_flags.append(f"transport_like — {transport_like_reactions[rxn_key]}")

        lines.append(f"  QA flags:  {'; '.join(rxn_flags) if rxn_flags else 'none'}")
        lines.append("")

    # ------------------------------------------------------------------
    # Transports
    # ------------------------------------------------------------------
    transports = [n for n in nodes if n.kind == "transport"]
    for i, trn in enumerate(transports, 1):
        lines.append(f"TRANSPORT {i}: {trn.label}")

        # Compartment is stored as "from_state → to_state" by build_draft_graph
        comp_str = (trn.compartment or "").strip()
        if " → " in comp_str:
            from_comp, to_comp = comp_str.split(" → ", 1)
        else:
            from_comp = to_comp = comp_str or "unspecified"

        lines.append(f"  From: {from_comp}")
        lines.append(f"  To:   {to_comp}")

        in_edges = incoming.get(trn.id, [])

        cargos = [(n, conf) for n, role, conf in in_edges if role == "cargo"]
        transporters = [(n, conf) for n, role, conf in in_edges if role == "transporter"]

        if cargos:
            cargo_strs = [_entity_str(n, confidence=conf) for n, conf in cargos]
            lines.append(f"  Cargo: {', '.join(cargo_strs)}")
        else:
            lines.append("  Cargo: MISSING")

        if transporters:
            tr_strs = [_entity_str(n, role="transporter", confidence=conf)
                       for n, conf in transporters]
            lines.append(f"  Transporter: {', '.join(tr_strs)}")
        else:
            lines.append("  Transporter: MISSING")

        # QA flags — flag transporters that may be unresolved complexes
        tr_flags: List[str] = []
        for n, _conf in transporters:
            if n.label.lower() in possible_complex_entities:
                tr_flags.append(
                    f"possible_complex — {n.label} not represented as protein_complex"
                )

        lines.append(f"  QA flags:  {'; '.join(tr_flags) if tr_flags else 'none'}")
        lines.append("")

    return "\n".join(lines)
