"""
draft_graph_render.py

Render a DraftGraph dict to a PNG via graphviz (dot).

Layout: top-to-bottom, reactions as labeled boxes with enzyme as italic subtitle.
"Through" metabolites (connect one reaction to the next) flow on the main vertical
axis. Side compounds (cofactors, terminal inputs/outputs) hang off with
constraint=false so they don't pollute the main flow. Enzyme/protein nodes are
suppressed entirely — their names appear inside the reaction box.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set

# Kinds that represent real metabolic steps (laid out top-to-bottom)
_RXN_KINDS        = {"reaction", "transport", "reaction_coupled_transport", "interaction"}
_FLOW_RXN_KINDS   = {"reaction", "transport", "reaction_coupled_transport"}  # excluded: interaction
_METABOLITE_KINDS = {"compound", "nucleic_acid"}
_ENZYME_KINDS     = {"protein", "protein_complex", "element_collection"}

# Reaction box fill / border colours by kind
_RXN_FILL  = {"reaction": "#e8f4fd", "transport": "#e8f5e9",
               "reaction_coupled_transport": "#f3e5f5", "interaction": "#fff8e1"}
_RXN_COLOR = {"reaction": "#1565c0", "transport": "#2e7d32",
              "reaction_coupled_transport": "#6a1b9a",  "interaction": "#e65100"}


def _dot_id(node_id: str) -> str:
    return '"' + node_id.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _dot_label(text: str, max_chars: int = 20) -> str:
    """Word-wrap and escape for a plain DOT quoted string."""
    text = text.strip()
    if len(text) <= max_chars:
        return text.replace('"', '\\"')
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip() if cur else w
    if cur:
        lines.append(cur)
    return "\\n".join(ln.replace('"', '\\"') for ln in lines)


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _topo_sort(rxn_ids: List[str], successors: Dict[str, Set[str]]) -> List[str]:
    in_deg: Dict[str, int] = {r: 0 for r in rxn_ids}
    for r in rxn_ids:
        for s in successors.get(r, set()):
            if s in in_deg:
                in_deg[s] += 1
    queue = deque(r for r in rxn_ids if in_deg[r] == 0)
    order: List[str] = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for s in successors.get(n, set()):
            if s not in in_deg:
                continue
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    # Append anything left (cycles / disconnected)
    seen = set(order)
    for r in rxn_ids:
        if r not in seen:
            order.append(r)
    return order


def _detect_cycle_order(
    flow_rxn_ids: List[str],
    compound_producers: Dict[str, Set[str]],
    compound_consumers: Dict[str, Set[str]],
    metabolite_ids: Set[str],
) -> Optional[List[str]]:
    """Return reaction IDs in Hamiltonian-cycle order if all flow reactions form
    a single directed cycle, else None.

    Ported from sbml_add_pathwhiz_layout._detect_cycle_order; adapted to work
    with plain string IDs instead of RxnNode objects.
    """
    n = len(flow_rxn_ids)
    if n < 3:
        return None

    idx = {rid: i for i, rid in enumerate(flow_rxn_ids)}

    successors: Dict[int, Set[int]] = defaultdict(set)
    in_degree = [0] * n

    for cid in metabolite_ids:
        if not compound_producers[cid] or not compound_consumers[cid]:
            continue
        for prod_r in compound_producers[cid]:
            if prod_r not in idx:
                continue
            for cons_r in compound_consumers[cid]:
                if cons_r not in idx:
                    continue
                pi, ci = idx[prod_r], idx[cons_r]
                if pi != ci and ci not in successors[pi]:
                    successors[pi].add(ci)
                    in_degree[ci] += 1

    # Any reaction with no predecessor means there is a source → not a pure cycle
    if any(d == 0 for d in in_degree):
        return None

    # DFS with Warnsdorff heuristic: prefer candidates with fewer onward options
    path: List[int] = [0]
    visited: Set[int] = {0}

    def _dfs() -> bool:
        if len(path) == n:
            return 0 in successors[path[-1]]
        remaining = set(range(n)) - visited
        candidates = sorted(
            successors[path[-1]] - visited,
            key=lambda j: len(successors[j] & remaining),
        )
        for nxt in candidates:
            visited.add(nxt)
            path.append(nxt)
            if _dfs():
                return True
            path.pop()
            visited.remove(nxt)
        return False

    return [flow_rxn_ids[i] for i in path] if _dfs() else None


def _build_dot(nodes: List[Dict], edges: List[Dict]) -> str:
    node_by_id = {n["id"]: n for n in nodes}

    # Only metabolic steps go in the main layout — interactions (regulatory) are skipped
    flow_rxn_ids = [n["id"] for n in nodes if n["kind"] in _FLOW_RXN_KINDS]
    flow_rxn_set = set(flow_rxn_ids)
    metabolite_ids = {n["id"] for n in nodes if n["kind"] in _METABOLITE_KINDS}

    # ── Collect per-reaction roles ─────────────────────────────────────────────
    compound_producers: Dict[str, Set[str]] = defaultdict(set)  # cid -> {rxn_id}
    compound_consumers: Dict[str, Set[str]] = defaultdict(set)  # cid -> {rxn_id}
    rxn_catalysts:      Dict[str, List[str]] = defaultdict(list) # rxn_id -> [label]

    for e in edges:
        src, tgt, role = e["source"], e["target"], e.get("role", "")
        src_node = node_by_id.get(src)
        tgt_node = node_by_id.get(tgt)

        # Only track flow reactions (not interactions)
        if role == "product" and src in flow_rxn_set:
            compound_producers[tgt].add(src)
        elif role in ("reactant", "cargo") and tgt in flow_rxn_set:
            compound_consumers[src].add(tgt)
        elif role in ("catalyst", "modifier", "transporter", "participant"):
            if src_node and src_node["kind"] in _ENZYME_KINDS and tgt in flow_rxn_set:
                rxn_catalysts[tgt].append(src_node.get("label", src))

    # ── Build reaction successor graph (any metabolite connecting two reactions) ─
    rxn_successors: Dict[str, Set[str]] = defaultdict(set)
    for cid in metabolite_ids:
        if compound_producers[cid] and compound_consumers[cid]:
            for prod_r in compound_producers[cid]:
                for cons_r in compound_consumers[cid]:
                    if prod_r != cons_r:
                        rxn_successors[prod_r].add(cons_r)

    # ── Order reactions: cycle-aware ──────────────────────────────────────────
    cycle_order = _detect_cycle_order(
        flow_rxn_ids, compound_producers, compound_consumers, metabolite_ids
    )
    if cycle_order is not None:
        ordered_rxns = cycle_order
        is_cycle = True
    else:
        ordered_rxns = _topo_sort(flow_rxn_ids, rxn_successors)
        is_cycle = False

    # ── Classify metabolites: consecutive-pair intersection ───────────────────
    # "Through" = products of rxn[i] ∩ reactants of rxn[i+1] for each adjacent pair.
    # This prevents cofactors/carriers (NADH, H2O, CoA-SH) that happen to appear in
    # multiple reactions from being misclassified as backbone metabolites.
    through: Set[str] = set()
    pairs = list(zip(ordered_rxns, ordered_rxns[1:]))
    if is_cycle and ordered_rxns:
        pairs.append((ordered_rxns[-1], ordered_rxns[0]))  # close the ring
    for rxn_a, rxn_b in pairs:
        prods_a = {cid for cid in metabolite_ids if rxn_a in compound_producers[cid]}
        reacts_b = {cid for cid in metabolite_ids if rxn_b in compound_consumers[cid]}
        through |= prods_a & reacts_b

    # "Side" = everything else (terminal inputs, cofactors, byproducts)
    side = metabolite_ids - through

    # ── DOT output ────────────────────────────────────────────────────────────
    lines: List[str] = [
        "digraph pathway {",
        "  rankdir=TB;",
        "  splines=polyline;",
        "  nodesep=0.9;",
        "  ranksep=1.1;",
        "  bgcolor=white;",
        "  pad=0.6;",
    ]

    # Reaction boxes — name bold, enzyme italic subtitle
    for nid in ordered_rxns:
        n = node_by_id[nid]
        kind = n.get("kind", "reaction")
        name_raw = n.get("label", nid)
        cats = list(dict.fromkeys(rxn_catalysts.get(nid, [])))

        fill  = _RXN_FILL.get(kind, "#e8f4fd")
        color = _RXN_COLOR.get(kind, "#1565c0")
        name_html = _html_escape(_dot_label(name_raw, max_chars=28)).replace("\\n", "<BR/>")

        if cats:
            enz_html = _html_escape(", ".join(cats))
            label = f'<<B>{name_html}</B><BR/><FONT POINT-SIZE="9"><I>{enz_html}</I></FONT>>'
        else:
            label = f'<<B>{name_html}</B>>'

        lines.append(
            f'  {_dot_id(nid)} [shape=box style="rounded,filled" '
            f'fillcolor="{fill}" color="{color}" penwidth=2.0 '
            f'fontsize=11 fontname="Helvetica" margin="0.25,0.15" label={label}];'
        )

    # Through metabolite nodes — on the main vertical axis
    for cid in sorted(through):
        n = node_by_id[cid]
        label = _dot_label(n.get("label", cid), max_chars=14)
        lines.append(
            f'  {_dot_id(cid)} [shape=ellipse style="filled" fillcolor=white '
            f'color="#444444" penwidth=1.5 width=1.1 height=0.65 fixedsize=true '
            f'fontsize=10 fontname="Helvetica" label="{label}"];'
        )

    # Side metabolite nodes — small; forced to same rank as their reaction
    # so they appear beside it rather than floating elsewhere
    side_to_rxn: Dict[str, str] = {}
    for e in edges:
        src, tgt, role = e["source"], e["target"], e.get("role", "")
        if role in ("catalyst", "modifier", "transporter", "participant"):
            continue
        if src in side and tgt in flow_rxn_set and src not in side_to_rxn:
            side_to_rxn[src] = tgt
        elif tgt in side and src in flow_rxn_set and tgt not in side_to_rxn:
            side_to_rxn[tgt] = src

    for cid in sorted(side):
        n = node_by_id[cid]
        label = _dot_label(n.get("label", cid), max_chars=12)
        lines.append(
            f'  {_dot_id(cid)} [shape=ellipse style="filled" fillcolor="#f5f5f5" '
            f'color="#bbbbbb" penwidth=1.0 width=0.8 height=0.5 fixedsize=true '
            f'fontsize=8 fontname="Helvetica" label="{label}"];'
        )

    rxn_to_side: Dict[str, List[str]] = defaultdict(list)
    for cid, rid in side_to_rxn.items():
        rxn_to_side[rid].append(cid)
    for rid, cids in rxn_to_side.items():
        members = " ".join(_dot_id(c) for c in cids)
        lines.append(f'  {{ rank=same; {_dot_id(rid)}; {members}; }}')

    # ── Edges ─────────────────────────────────────────────────────────────────
    skip_roles = {"catalyst", "modifier", "transporter", "participant"}

    for e in edges:
        src, tgt, role = e["source"], e["target"], e.get("role", "")
        if role in skip_roles:
            continue

        src_node = node_by_id.get(src)
        tgt_node = node_by_id.get(tgt)

        # Skip enzyme nodes (folded into box) and interaction nodes
        if src_node and src_node["kind"] in _ENZYME_KINDS:
            continue
        if tgt_node and tgt_node["kind"] in _ENZYME_KINDS:
            continue
        if src_node and src_node["kind"] == "interaction":
            continue
        if tgt_node and tgt_node["kind"] == "interaction":
            continue

        # Skip edges that involve non-flow reactions
        if src in flow_rxn_set or tgt in flow_rxn_set:
            pass  # at least one end is a flow reaction — keep it
        elif src_node and src_node["kind"] in _RXN_KINDS:
            continue  # both ends are non-flow reactions
        elif tgt_node and tgt_node["kind"] in _RXN_KINDS:
            continue

        is_side_edge = (src in side) or (tgt in side)

        if is_side_edge:
            lines.append(
                f'  {_dot_id(src)} -> {_dot_id(tgt)} '
                f'[color="#cccccc" penwidth=1.0 arrowsize=0.55 '
                f'constraint=false style=dashed];'
            )
        elif role == "reactant":
            lines.append(
                f'  {_dot_id(src)} -> {_dot_id(tgt)} '
                f'[color="#1565c0" penwidth=2.0 arrowsize=0.8];'
            )
        elif role in ("product", "cargo"):
            lines.append(
                f'  {_dot_id(src)} -> {_dot_id(tgt)} '
                f'[color="#2e7d32" penwidth=2.0 arrowsize=0.8];'
            )
        else:
            lines.append(
                f'  {_dot_id(src)} -> {_dot_id(tgt)} '
                f'[color="#888888" penwidth=1.4 arrowsize=0.7];'
            )

    lines.append("}")
    return "\n".join(lines)


# ── Graphviz discovery ────────────────────────────────────────────────────────

_FALLBACK_DOT_PATHS = [
    r"C:\Program Files\Graphviz\bin\dot.exe",
    r"C:\Program Files (x86)\Graphviz\bin\dot.exe",
    r"C:\Program Files\Graphviz2.38\bin\dot.exe",
]


def _find_dot() -> str:
    found = shutil.which("dot")
    if found:
        return found
    for p in _FALLBACK_DOT_PATHS:
        if os.path.isfile(p):
            return p
    raise RuntimeError(
        "graphviz 'dot' not found on PATH. Install graphviz to enable graph rendering."
    )


def render_draft_graph_to_png_bytes(draft_graph_dict: Dict[str, Any], dpi: int = 160) -> bytes:
    nodes: List[Dict] = draft_graph_dict.get("nodes", [])
    edges: List[Dict] = draft_graph_dict.get("edges", [])

    if not nodes:
        raise RuntimeError("DraftGraph has no nodes — nothing to render.")

    dot_exe = _find_dot()
    dot_src = _build_dot(nodes, edges)

    result = subprocess.run(
        [dot_exe, f"-Gdpi={dpi}", "-Tpng"],
        input=dot_src.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"graphviz dot failed: {result.stderr.decode('utf-8', errors='replace')}")

    return result.stdout
