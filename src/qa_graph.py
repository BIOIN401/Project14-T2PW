# qa_graph.py
# Graph QA for PWML-extracted JSON:
# - Build a graph of compounds/proteins/reactions (and optionally transports/interactions)
# - Find connected components
# - Flag orphan components (not in main component)
# - Flag dangling nodes (low degree)
# - Heuristic hints: "present but unused" for entities not connected to any process

import json
import re
import sys
from collections import defaultdict, deque
from typing import Any, Dict, List, Set, Tuple

BINOMIAL_RE = re.compile(r"^[A-Z][a-z]+(\.|)\s+[a-z][a-z-]+$")  # e.g., Arabidopsis thaliana OR A. thaliana

def node(kind: str, name: str) -> str:
    return f"{kind}:{name}"



def add_edge(adj: Dict[str, Set[str]], a: str, b: str) -> None:
    if not a or not b:
        return
    adj[a].add(b)
    adj[b].add(a)

def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []



def get_entities(extracted: Dict[str, Any]) -> Dict[str, Set[str]]:
    ents = extracted.get("entities", {})
    out = {
        "compounds": set(),
        "proteins": set(),
        "nucleic_acids": set(),
        "element_collections": set(),
        "protein_complexes": set(),
        "species": set(),
        "cell_types": set(),
        "tissues": set(),
        "subcellular_locations": set(),
    }

    for k, target in [
        ("compounds", "compounds"),
        ("proteins", "proteins"),
        ("nucleic_acids", "nucleic_acids"),
        ("element_collections", "element_collections"),
        ("protein_complexes", "protein_complexes"),
        ("species", "species"),
        ("cell_types", "cell_types"),
        ("tissues", "tissues"),
        ("subcellular_locations", "subcellular_locations"),
    ]:
        for item in safe_list(ents.get(k, [])):
            name = (item or {}).get("name", "")
            if isinstance(name, str) and name.strip():
                out[target].add(name.strip())

    return out

def build_graph(extracted: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    """
    Graph model:
      - Nodes: compound/protein/protein_complex/reaction/transport/interaction (+ state nodes for transport)
      - Edges:
          entity -- reaction   (inputs/outputs)
          enzyme/transporter -- reaction/transport
          entity@from_state -- transport -- entity@to_state
          interaction entity_1 -- interaction -- entity_2
      - Never creates cargo:* nodes.
    """
    adj: Dict[str, Set[str]] = defaultdict(set)
    ents = get_entities(extracted)
    compounds = set(ents.get("compounds", set()))
    proteins = set(ents.get("proteins", set()))
    complexes = set(ents.get("protein_complexes", set()))

    def resolve_kind(name: str) -> str:
        n = (name or "").strip()
        if not n:
            return "entity"
        if n in proteins:
            return "protein"
        if n in complexes:
            return "protein_complex"
        if n in compounds:
            return "compound"
        if ":" in n:
            return "protein_complex"
        return "compound"

    processes = extracted.get("processes", {})
    reactions = safe_list(processes.get("reactions", []))
    transports = safe_list(processes.get("transports", []))
    interactions = safe_list(processes.get("interactions", []))
    rcts = safe_list(processes.get("reaction_coupled_transports", []))

    # --- Reactions ---
    for i, r in enumerate(reactions):
        rname = (r or {}).get("name", "").strip()
        rid = node("reaction", rname if rname else f"#{i+1}")

        inputs = [x for x in safe_list((r or {}).get("inputs", [])) if isinstance(x, str) and x.strip()]
        outputs = [x for x in safe_list((r or {}).get("outputs", [])) if isinstance(x, str) and x.strip()]

        for c in inputs + outputs:
            c_name = c.strip()
            add_edge(adj, node(resolve_kind(c_name), c_name), rid)

        for enz in safe_list((r or {}).get("enzymes", [])):
            if isinstance(enz, dict):
                p_name = ""
                for key in ["protein", "protein_complex", "name"]:
                    candidate = (enz.get(key) or "").strip() if isinstance(enz.get(key), str) else ""
                    if candidate:
                        p_name = candidate
                        break
                if p_name:
                    add_edge(adj, node(resolve_kind(p_name), p_name), rid)

    # --- Reaction-coupled transports ---
    for i, rct in enumerate(rcts):
        rctname = (rct or {}).get("name", "").strip()
        rctid = node("reaction_coupled_transport", rctname if rctname else f"#{i+1}")

        # connect referenced reaction/transport ids by name if present
        rx = (rct or {}).get("reaction", "").strip()
        tx = (rct or {}).get("transport", "").strip()
        if rx:
            add_edge(adj, rctid, node("reaction", rx))
        if tx:
            add_edge(adj, rctid, node("transport", tx))

        for enz in safe_list((rct or {}).get("enzymes", [])):
            if isinstance(enz, dict):
                pc = (enz.get("protein_complex") or "").strip()
                if pc:
                    add_edge(adj, node("protein_complex", pc), rctid)

        for ews in safe_list((rct or {}).get("elements_with_states", [])):
            if isinstance(ews, dict):
                el = (ews.get("element") or "").strip()
                if el:
                    # We don't know element type; attach as generic element
                    add_edge(adj, node("element", el), rctid)

    # --- Transports ---
    for i, t in enumerate(transports):
        tname = (t or {}).get("name", "").strip()
        tid = node("transport", tname if tname else f"#{i+1}")

        cargo = ""
        cargo_complex = (t or {}).get("cargo_complex", "")
        if isinstance(cargo_complex, str) and cargo_complex.strip():
            cargo = cargo_complex.strip()
        else:
            cargo_raw = (t or {}).get("cargo", "")
            if isinstance(cargo_raw, str) and cargo_raw.strip():
                cargo = cargo_raw.strip()
        if not cargo:
            for ews in safe_list((t or {}).get("elements_with_states", [])):
                if not isinstance(ews, dict):
                    continue
                el = (ews.get("element") or "").strip() if isinstance(ews.get("element"), str) else ""
                if el:
                    cargo = el
                    break

        from_state = (t or {}).get("from_biological_state", "")
        to_state = (t or {}).get("to_biological_state", "")
        from_state_name = from_state.strip() if isinstance(from_state, str) and from_state.strip() else "unspecified"
        to_state_name = to_state.strip() if isinstance(to_state, str) and to_state.strip() else "unspecified"
        if cargo:
            cargo_kind = resolve_kind(cargo)
            base_entity = node(cargo_kind, cargo)
            source_entity = node(f"{cargo_kind}_state", f"{cargo}@{from_state_name}")
            dest_entity = node(f"{cargo_kind}_state", f"{cargo}@{to_state_name}")
            add_edge(adj, base_entity, source_entity)
            add_edge(adj, base_entity, dest_entity)
            add_edge(adj, source_entity, tid)
            add_edge(adj, tid, dest_entity)

        for tr in safe_list((t or {}).get("transporters", [])):
            if isinstance(tr, dict):
                p_name = ""
                for key in ["protein", "protein_complex", "name"]:
                    candidate = (tr.get(key) or "").strip() if isinstance(tr.get(key), str) else ""
                    if candidate:
                        p_name = candidate
                        break
                if p_name:
                    add_edge(adj, node(resolve_kind(p_name), p_name), tid)

        for ews in safe_list((t or {}).get("elements_with_states", [])):
            if isinstance(ews, dict):
                el = (ews.get("element") or "").strip()
                if el:
                    add_edge(adj, node(resolve_kind(el), el), tid)

    # --- Interactions ---
    for i, inter in enumerate(interactions):
        iname = (inter or {}).get("name", "").strip()
        iid = node("interaction", iname if iname else f"#{i+1}")

        e1 = (inter or {}).get("entity_1", "")
        e2 = (inter or {}).get("entity_2", "")
        if isinstance(e1, str) and e1.strip():
            add_edge(adj, node("entity", e1.strip()), iid)
        if isinstance(e2, str) and e2.strip():
            add_edge(adj, node("entity", e2.strip()), iid)

    meta = {
        "n_reactions": len(reactions),
        "n_transports": len(transports),
        "n_interactions": len(interactions),
        "n_reaction_coupled_transports": len(rcts),
    }
    return adj, meta
def connected_components(adj: Dict[str, Set[str]]) -> List[Set[str]]:
    visited: Set[str] = set()
    comps: List[Set[str]] = []

    for start in adj.keys():
        if start in visited:
            continue
        q = deque([start])
        visited.add(start)
        comp = {start}
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
                    comp.add(v)
        comps.append(comp)

    # also include isolated nodes not present as keys (rare with our add_edge design)
    return comps


def degrees(adj: Dict[str, Set[str]]) -> Dict[str, int]:
    return {k: len(v) for k, v in adj.items()}

def main():
    if len(sys.argv) < 2:
        print("Usage: python qa_graph.py <extracted.json> [qa_report.json]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else "qa_report.json"

    with open(in_path, "r", encoding="utf-8") as f:
        extracted = json.load(f)

    ents = get_entities(extracted)
    adj, meta = build_graph(extracted)

    # Ensure entity nodes exist even if disconnected (so we can flag unused entities)
    for c in ents["compounds"]:
        adj.setdefault(node("compound", c), set())
    for p in ents["proteins"]:
        adj.setdefault(node("protein", p), set())
    for na in ents["nucleic_acids"]:
        adj.setdefault(node("nucleic_acid", na), set())
    for ec in ents["element_collections"]:
        adj.setdefault(node("element_collection", ec), set())
    for pc in ents["protein_complexes"]:
        adj.setdefault(node("protein_complex", pc), set())

    comps = connected_components(adj)
    comps_sorted = sorted(comps, key=lambda s: len(s), reverse=True)
    main_comp = comps_sorted[0] if comps_sorted else set()

    deg = degrees(adj)

    # Orphans = components not equal to main component (size>=1). Often you may want size>=2 threshold.
    orphan_components = []
    for comp in comps_sorted[1:]:
        orphan_components.append({
            "size": len(comp),
            "nodes": sorted(comp),
        })

    dangling = [{"node": n, "degree": d} for n, d in deg.items() if d <= 1]
    dangling_sorted = sorted(dangling, key=lambda x: (x["degree"], x["node"]))

    # Heuristic hints: entities with degree 0 (totally unused)
    missing_links = []
    for kind, names in [
        ("compound", ents["compounds"]),
        ("protein", ents["proteins"]),
        ("nucleic_acid", ents["nucleic_acids"]),
        ("element_collection", ents["element_collections"]),
        ("protein_complex", ents["protein_complexes"]),
    ]:
        for name_ in names:
            n = node(kind, name_)
            if deg.get(n, 0) == 0:
                missing_links.append({
                    "hint": f"{kind} appears in entities but is not connected to any process/location",
                    "node": n
                })

    # Species QA: flag any species not binomial
    bad_species = []
    for s in ents["species"]:
        if not BINOMIAL_RE.match(s.strip()):
            bad_species.append(s)

    report = {
        "meta": meta,
        "n_nodes": len(adj),
        "n_edges": sum(len(v) for v in adj.values()) // 2,
        "n_components": len(comps_sorted),
        "main_component_size": len(main_comp),
        "orphan_components": orphan_components,
        "dangling_nodes": dangling_sorted[:200],  # cap
        "missing_links_suspected": missing_links[:200],
        "species_violations": bad_species,
        "notes": [
            "Orphan components are any connected component not equal to the largest component.",
            "Dangling nodes are degree <= 1 and often indicate missing links or incomplete extraction.",
        ]
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote QA report: {out_path}")
    print(f"Components: {report['n_components']} | Main size: {report['main_component_size']} | Orphans: {len(report['orphan_components'])}")


if __name__ == "__main__":
    main()

