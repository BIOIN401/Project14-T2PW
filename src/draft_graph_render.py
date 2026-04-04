"""
draft_graph_render.py

Render a DraftGraph dict to a PNG via graphviz (dot).

Layout: rankdir=LR — reactions ranked between their reactants (left) and
products (right); catalysts/modifiers come in from above via constraint=false
edges so they don't distort the main left-right flow.

Compartments are visualised as labelled subgraph clusters.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, List, Set

_RXN_KINDS = {"reaction", "transport", "reaction_coupled_transport", "interaction"}

# ── Visual style ─────────────────────────────────────────────────────────────

_NODE_ATTRS: Dict[str, str] = {
    "compound":      'shape=circle style="filled" fillcolor=white      color="#333333" penwidth=1.8 width=0.75 height=0.75 fixedsize=true fontsize=10',
    "nucleic_acid":  'shape=circle style="filled" fillcolor="#fffde7"  color="#333333" penwidth=1.8 width=0.75 height=0.75 fixedsize=true fontsize=10',
    "protein":       'shape=box   style="rounded,filled" fillcolor="#e3f2fd" color="#1565c0" penwidth=1.6 fontsize=10',
    "protein_complex": 'shape=box style="rounded,filled" fillcolor="#e8eaf6" color="#283593" penwidth=1.6 fontsize=10',
    "element_collection": 'shape=box style="rounded,filled" fillcolor="#f3e5f5" color="#6a1b9a" penwidth=1.6 fontsize=10',
    # reaction-like nodes: tiny filled black square, no label shown on node itself
    "reaction":                      'shape=square style="filled" fillcolor=black color=black width=0.22 height=0.22 fixedsize=true label=""',
    "transport":                     'shape=square style="filled" fillcolor="#424242" color=black width=0.22 height=0.22 fixedsize=true label=""',
    "reaction_coupled_transport":    'shape=square style="filled" fillcolor="#616161" color=black width=0.22 height=0.22 fixedsize=true label=""',
    "interaction":                   'shape=diamond style="filled" fillcolor=black color=black width=0.30 height=0.22 fixedsize=true label=""',
}

# Edge style per role.  "constraint" controls whether the edge contributes to
# graphviz rank assignment (i.e. left-right ordering).
_EDGE_ATTRS: Dict[str, str] = {
    "reactant":    'color="#1565c0" penwidth=2.0 arrowhead=normal',
    "product":     'color="#2e7d32" penwidth=2.0 arrowhead=normal',
    "catalyst":    'color="#e65100" penwidth=1.4 style=dashed arrowhead=open constraint=false',
    "modifier":    'color="#6a1b9a" penwidth=1.4 style=dotted arrowhead=open constraint=false',
    "transporter": 'color="#00695c" penwidth=1.8 arrowhead=normal',
    "cargo":       'color="#558b2f" penwidth=1.8 arrowhead=normal',
    "participant": 'color="#37474f" penwidth=1.2 style=dashed arrowhead=open constraint=false',
}
_DEFAULT_EDGE = 'color="#555555" penwidth=1.2'


def _dot_id(node_id: str) -> str:
    """Escape a node id for DOT."""
    return '"' + node_id.replace('"', '\\"') + '"'


def _dot_label(text: str, max_chars: int = 20) -> str:
    """Wrap long labels and escape for DOT."""
    text = text.strip()
    if len(text) <= max_chars:
        return text.replace('"', '\\"')
    # simple word-wrap
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


def _build_dot(nodes: List[Dict], edges: List[Dict]) -> str:
    lines: List[str] = [
        "digraph pathway {",
        "  rankdir=LR;",
        "  splines=true;",
        "  nodesep=0.55;",
        "  ranksep=1.1;",
        "  bgcolor=white;",
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica" fontsize=8];',
    ]

    node_by_id = {n["id"]: n for n in nodes}

    # Group entity nodes by compartment for cluster subgraphs
    compartment_members: Dict[str, List[str]] = {}
    for n in nodes:
        if n["kind"] in _RXN_KINDS:
            continue  # reactions not clustered
        comp = (n.get("compartment") or "").strip()
        compartment_members.setdefault(comp, []).append(n["id"])

    # Emit compartment clusters (skip the empty-string "unknown" compartment)
    emitted_in_cluster: Set[str] = set()
    cluster_idx = 0
    for comp, members in compartment_members.items():
        if not comp:
            continue
        lines.append(f'  subgraph cluster_{cluster_idx} {{')
        lines.append(f'    label="{comp}";')
        lines.append('    style="rounded,filled";')
        lines.append('    fillcolor="#f5f5f5";')
        lines.append('    color="#aaaaaa";')
        lines.append('    penwidth=1.2;')
        for nid in members:
            n = node_by_id[nid]
            kind = n.get("kind", "compound")
            attrs = _NODE_ATTRS.get(kind, _NODE_ATTRS["compound"])
            label = _dot_label(n.get("label", nid))
            lines.append(f'    {_dot_id(nid)} [{attrs} label="{label}"];')
            emitted_in_cluster.add(nid)
        lines.append("  }")
        cluster_idx += 1

    # Emit nodes not in any cluster (no compartment, or reactions)
    for n in nodes:
        nid = n["id"]
        if nid in emitted_in_cluster:
            continue
        kind = n.get("kind", "compound")
        attrs = _NODE_ATTRS.get(kind, _NODE_ATTRS["compound"])
        is_rxn = kind in _RXN_KINDS
        if is_rxn:
            # Reaction: node has no visible label; we add an external label via
            # a separate invisible helper node or just rely on the tooltip.
            # Simpler: emit the node, then add a label node linked by invisible edge.
            rxn_dot_id = _dot_id(nid)
            lines.append(f'  {rxn_dot_id} [{attrs}];')
            # Floating label node
            label_node_id = _dot_id(nid + "__lbl")
            label = _dot_label(n.get("label", nid), max_chars=18)
            lines.append(f'  {label_node_id} [shape=none label="{label}" fontsize=8 width=0 height=0];')
            lines.append(f'  {rxn_dot_id} -> {label_node_id} [style=invis weight=0];')
        else:
            label = _dot_label(n.get("label", nid))
            lines.append(f'  {_dot_id(nid)} [{attrs} label="{label}"];')

    # Emit edges
    for e in edges:
        src, tgt, role = e["source"], e["target"], e.get("role", "")
        attrs = _EDGE_ATTRS.get(role, _DEFAULT_EDGE)
        lines.append(f'  {_dot_id(src)} -> {_dot_id(tgt)} [{attrs}];')

    lines.append("}")
    return "\n".join(lines)


def render_draft_graph_to_png_bytes(draft_graph_dict: Dict[str, Any], dpi: int = 150) -> bytes:
    """
    Render a DraftGraph dict to PNG bytes using graphviz dot.
    Raises RuntimeError if graphviz is not installed or graph is empty.
    """
    nodes: List[Dict] = draft_graph_dict.get("nodes", [])
    edges: List[Dict] = draft_graph_dict.get("edges", [])

    if not nodes:
        raise RuntimeError("DraftGraph has no nodes — nothing to render.")

    if not shutil.which("dot"):
        raise RuntimeError(
            "graphviz 'dot' not found on PATH. Install graphviz to enable graph rendering."
        )

    dot_src = _build_dot(nodes, edges)
    result = subprocess.run(
        ["dot", f"-Gdpi={dpi}", "-Tpng"],
        input=dot_src.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"graphviz dot failed: {stderr}")

    return result.stdout
