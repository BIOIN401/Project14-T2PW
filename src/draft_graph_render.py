"""
draft_graph_render.py

Render a DraftGraph dict to a PNG via graphviz (dot).

Layout: rankdir=LR with newrank=true so compartment clusters don't fight the
left-right rank ordering. Reactions sit between their reactants and products.
Catalysts/modifiers attach from above via constraint=false edges.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Set

_RXN_KINDS = {"reaction", "transport", "reaction_coupled_transport", "interaction"}

# ── Visual style ──────────────────────────────────────────────────────────────

# Compound / entity nodes
_COMPOUND_ATTRS  = 'shape=ellipse style="filled" fillcolor=white color="#444444" penwidth=2.0 width=1.1 height=1.1 fixedsize=true fontsize=11 fontname="Helvetica"'
_NUCLEIC_ATTRS   = 'shape=ellipse style="filled" fillcolor="#fff9c4" color="#444444" penwidth=2.0 width=1.1 height=1.1 fixedsize=true fontsize=11 fontname="Helvetica"'
_PROTEIN_ATTRS   = 'shape=box style="rounded,filled" fillcolor="#ddeeff" color="#1a5fa8" penwidth=2.0 fontsize=11 fontname="Helvetica" margin="0.18,0.12"'
_COMPLEX_ATTRS   = 'shape=box style="rounded,filled" fillcolor="#e8eaf6" color="#283593" penwidth=2.0 fontsize=11 fontname="Helvetica" margin="0.18,0.12"'
_ECOLL_ATTRS     = 'shape=box style="rounded,filled" fillcolor="#f3e5f5" color="#6a1b9a" penwidth=2.0 fontsize=11 fontname="Helvetica" margin="0.18,0.12"'

# Reaction nodes: small filled square, label rendered outside via xlabel
_RXN_ATTRS  = 'shape=square style="filled" fillcolor="#111111" color="#111111" width=0.28 height=0.28 fixedsize=true label="" fontsize=10 fontname="Helvetica Bold"'
_TRAN_ATTRS = 'shape=square style="filled" fillcolor="#1a5fa8" color="#1a5fa8" width=0.28 height=0.28 fixedsize=true label="" fontsize=10 fontname="Helvetica Bold"'
_RCT_ATTRS  = 'shape=square style="filled" fillcolor="#555555" color="#555555" width=0.28 height=0.28 fixedsize=true label="" fontsize=10 fontname="Helvetica Bold"'
_INT_ATTRS  = 'shape=diamond style="filled" fillcolor="#111111" color="#111111" width=0.36 height=0.28 fixedsize=true label="" fontsize=10 fontname="Helvetica Bold"'

_NODE_ATTRS: Dict[str, str] = {
    "compound": _COMPOUND_ATTRS,
    "nucleic_acid": _NUCLEIC_ATTRS,
    "protein": _PROTEIN_ATTRS,
    "protein_complex": _COMPLEX_ATTRS,
    "element_collection": _ECOLL_ATTRS,
    "reaction": _RXN_ATTRS,
    "transport": _TRAN_ATTRS,
    "reaction_coupled_transport": _RCT_ATTRS,
    "interaction": _INT_ATTRS,
}

# Edges — constraint=false on catalyst/modifier/participant so they don't
# push reaction nodes off the main left-right flow axis.
_EDGE_ATTRS: Dict[str, str] = {
    "reactant":    'color="#1a5fa8" penwidth=2.2 arrowhead=normal arrowsize=0.9',
    "product":     'color="#2e7d32" penwidth=2.2 arrowhead=normal arrowsize=0.9',
    "catalyst":    'color="#c84b00" penwidth=1.6 style=dashed arrowhead=open arrowsize=1.0 constraint=false',
    "modifier":    'color="#7b1fa2" penwidth=1.4 style=dotted arrowhead=open arrowsize=1.0 constraint=false',
    "transporter": 'color="#00695c" penwidth=2.2 arrowhead=normal arrowsize=0.9',
    "cargo":       'color="#558b2f" penwidth=2.0 arrowhead=normal arrowsize=0.9',
    "participant": 'color="#37474f" penwidth=1.2 style=dashed arrowhead=open arrowsize=0.8 constraint=false',
}
_DEFAULT_EDGE = 'color="#777777" penwidth=1.4'

# Compartment cluster fill colours (cycle through these)
_CLUSTER_FILLS  = ["#f0f7ff", "#fff8f0", "#f0fff4", "#fdf0ff", "#fffff0"]
_CLUSTER_COLORS = ["#90b8d8", "#d8b890", "#90d8a8", "#c890d8", "#d8d890"]


def _dot_id(node_id: str) -> str:
    return '"' + node_id.replace('"', '\\"') + '"'


def _dot_label(text: str, max_chars: int = 16) -> str:
    """Word-wrap and escape for DOT string."""
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


def _build_dot(nodes: List[Dict], edges: List[Dict]) -> str:
    lines: List[str] = [
        "digraph pathway {",
        "  newrank=true;",       # allow proper LR ranking inside clusters
        "  rankdir=LR;",
        "  splines=true;",
        "  nodesep=0.65;",
        "  ranksep=1.4;",
        "  bgcolor=white;",
        "  pad=0.4;",
    ]

    node_by_id = {n["id"]: n for n in nodes}

    # ── Compartment clusters (entity nodes only) ──────────────────────────────
    compartment_members: Dict[str, List[str]] = {}
    for n in nodes:
        if n["kind"] in _RXN_KINDS:
            continue
        comp = (n.get("compartment") or "").strip()
        compartment_members.setdefault(comp, []).append(n["id"])

    emitted_in_cluster: Set[str] = set()
    for ci, (comp, members) in enumerate(compartment_members.items()):
        if not comp:
            continue
        fill  = _CLUSTER_FILLS[ci % len(_CLUSTER_FILLS)]
        color = _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]
        lines.append(f'  subgraph cluster_{ci} {{')
        lines.append(f'    label="{comp}";')
        lines.append(f'    style="rounded,filled";')
        lines.append(f'    fillcolor="{fill}";')
        lines.append(f'    color="{color}";')
        lines.append(f'    penwidth=1.8;')
        lines.append(f'    fontname="Helvetica Bold";')
        lines.append(f'    fontsize=12;')
        for nid in members:
            n = node_by_id[nid]
            kind = n.get("kind", "compound")
            attrs = _NODE_ATTRS.get(kind, _COMPOUND_ATTRS)
            label = _dot_label(n.get("label", nid))
            lines.append(f'    {_dot_id(nid)} [{attrs} label="{label}"];')
            emitted_in_cluster.add(nid)
        lines.append("  }")

    # ── Reaction nodes (outside clusters, use xlabel for label) ───────────────
    for n in nodes:
        nid = n["id"]
        kind = n.get("kind", "compound")
        if kind not in _RXN_KINDS:
            if nid not in emitted_in_cluster:
                # entity with no compartment
                attrs = _NODE_ATTRS.get(kind, _COMPOUND_ATTRS)
                label = _dot_label(n.get("label", nid))
                lines.append(f'  {_dot_id(nid)} [{attrs} label="{label}"];')
            continue
        attrs = _NODE_ATTRS.get(kind, _RXN_ATTRS)
        xlabel = _dot_label(n.get("label", nid), max_chars=18)
        lines.append(f'  {_dot_id(nid)} [{attrs} xlabel="{xlabel}"];')

    # ── Edges ─────────────────────────────────────────────────────────────────
    for e in edges:
        src, tgt, role = e["source"], e["target"], e.get("role", "")
        attrs = _EDGE_ATTRS.get(role, _DEFAULT_EDGE)
        lines.append(f'  {_dot_id(src)} -> {_dot_id(tgt)} [{attrs}];')

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
