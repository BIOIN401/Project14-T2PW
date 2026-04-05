"""
draft_graph_render.py

Render a DraftGraph dict to a PNG using matplotlib.
Each reaction is drawn as a self-contained mini-diagram:
  (reactant circles) → [■] → (product circles)
with the enzyme name in a rounded rectangle above the reaction dot.
Reactions are tiled left-to-right in rows.
"""
from __future__ import annotations

import io
import textwrap
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

_RXN_KINDS        = {"reaction", "transport", "reaction_coupled_transport", "interaction"}
_FLOW_RXN_KINDS   = {"reaction", "transport", "reaction_coupled_transport"}
_METABOLITE_KINDS = {"compound", "nucleic_acid"}
_ENZYME_KINDS     = {"protein", "protein_complex", "element_collection"}

# Layout constants (matplotlib data units = pixels at 1:1)
_BLOCK_W   = 400   # width of one reaction block
_BLOCK_H   = 340   # height of one reaction block
_COLS      = 4     # reactions per row
_CIRC_R    = 32    # compound circle radius
_RECT_W    = 175   # enzyme rectangle width
_RECT_H    = 48    # enzyme rectangle height
_DOT       = 10    # reaction-center square side length
_PAD       = 20    # inner padding
_FONT      = 8     # base font size


def _wrap(text: str, width: int = 12) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


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
    seen = set(order)
    for r in rxn_ids:
        if r not in seen:
            order.append(r)
    return order


def _parse_graph(
    nodes: List[Dict], edges: List[Dict]
) -> Tuple[List[str], Dict[str, Dict], Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (ordered_rxn_ids, node_by_id, rxn_reactants, rxn_products, rxn_enzymes)."""
    node_by_id: Dict[str, Dict] = {n["id"]: n for n in nodes}

    flow_rxn_ids = [n["id"] for n in nodes if n["kind"] in _FLOW_RXN_KINDS]
    flow_rxn_set = set(flow_rxn_ids)
    metabolite_ids = {n["id"] for n in nodes if n["kind"] in _METABOLITE_KINDS}

    rxn_reactants: Dict[str, List[str]] = defaultdict(list)
    rxn_products:  Dict[str, List[str]] = defaultdict(list)
    rxn_enzymes:   Dict[str, List[str]] = defaultdict(list)

    compound_producers: Dict[str, Set[str]] = defaultdict(set)
    compound_consumers: Dict[str, Set[str]] = defaultdict(set)

    for e in edges:
        src, tgt, role = e["source"], e["target"], e.get("role", "")
        src_node = node_by_id.get(src)
        tgt_node = node_by_id.get(tgt)

        if role == "product" and src in flow_rxn_set and tgt in metabolite_ids:
            rxn_products[src].append(tgt)
            compound_producers[tgt].add(src)
        elif role in ("reactant", "cargo") and tgt in flow_rxn_set and src in metabolite_ids:
            rxn_reactants[tgt].append(src)
            compound_consumers[src].add(tgt)
        elif role in ("catalyst", "modifier", "transporter", "participant"):
            if src_node and src_node["kind"] in _ENZYME_KINDS and tgt in flow_rxn_set:
                rxn_enzymes[tgt].append(src_node.get("label", src))

    # Topo-sort reactions
    rxn_successors: Dict[str, Set[str]] = defaultdict(set)
    for cid in metabolite_ids:
        for prod_r in compound_producers[cid]:
            for cons_r in compound_consumers[cid]:
                if prod_r != cons_r:
                    rxn_successors[prod_r].add(cons_r)

    ordered = _topo_sort(flow_rxn_ids, rxn_successors)
    return ordered, node_by_id, rxn_reactants, rxn_products, rxn_enzymes


def _draw_arrow(ax: Any, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=1.6,
            mutation_scale=14,
        ),
        zorder=10,
    )


def _draw_dashed(ax: Any, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.plot(
        [x0, x1], [y0, y1],
        color="black", lw=1.2, linestyle=(0, (4, 5)), zorder=8,
    )


def _draw_reaction_block(
    ax: Any,
    ox: float,
    oy: float,
    rxn_label: str,
    reactant_labels: List[str],
    product_labels: List[str],
    enzyme_labels: List[str],
) -> None:
    """Draw one reaction sub-diagram at block origin (ox, oy)."""
    import matplotlib.patches as mpatches

    cx = ox + _BLOCK_W / 2
    cy = oy + _BLOCK_H / 2

    # ── Enzyme rectangle ──────────────────────────────────────────────────────
    if enzyme_labels:
        enz_text = ", ".join(dict.fromkeys(enzyme_labels))  # deduplicate
        rect = mpatches.FancyBboxPatch(
            (cx - _RECT_W / 2, oy + _PAD),
            _RECT_W,
            _RECT_H,
            boxstyle="round,pad=2,rounding_size=8",
            linewidth=1.8,
            edgecolor="black",
            facecolor="white",
            zorder=6,
        )
        ax.add_patch(rect)
        ax.text(
            cx, oy + _PAD + _RECT_H / 2,
            _wrap(enz_text, width=22),
            fontsize=_FONT, ha="center", va="center", zorder=7,
        )
        # Dashed line from enzyme rect bottom to reaction dot
        _draw_dashed(ax, cx, oy + _PAD + _RECT_H, cx, cy)

    # ── Reaction dot (small black square) ─────────────────────────────────────
    dot = mpatches.Rectangle(
        (cx - _DOT / 2, cy - _DOT / 2),
        _DOT, _DOT,
        linewidth=0, facecolor="black", zorder=9,
    )
    ax.add_patch(dot)

    # ── Reactant circles ──────────────────────────────────────────────────────
    n_r = len(reactant_labels)
    spacing_r = max(_CIRC_R * 2 + 6, (_BLOCK_H - _PAD * 2) / max(n_r, 1))
    start_y_r = cy - (n_r - 1) * spacing_r / 2
    rx = ox + _PAD + _CIRC_R

    for i, label in enumerate(reactant_labels):
        ry = start_y_r + i * spacing_r
        circ = mpatches.Circle(
            (rx, ry), _CIRC_R,
            linewidth=1.8, edgecolor="black", facecolor="white", zorder=6,
        )
        ax.add_patch(circ)
        ax.text(
            rx, ry, _wrap(label, width=10),
            fontsize=_FONT - 1, ha="center", va="center", zorder=7,
        )
        # Arrow from circle edge to dot
        _draw_arrow(ax, rx + _CIRC_R, ry, cx - _DOT / 2, cy)

    # ── Product circles ───────────────────────────────────────────────────────
    n_p = len(product_labels)
    spacing_p = max(_CIRC_R * 2 + 6, (_BLOCK_H - _PAD * 2) / max(n_p, 1))
    start_y_p = cy - (n_p - 1) * spacing_p / 2
    px_coord = ox + _BLOCK_W - _PAD - _CIRC_R

    for i, label in enumerate(product_labels):
        py = start_y_p + i * spacing_p
        circ = mpatches.Circle(
            (px_coord, py), _CIRC_R,
            linewidth=1.8, edgecolor="black", facecolor="white", zorder=6,
        )
        ax.add_patch(circ)
        ax.text(
            px_coord, py, _wrap(label, width=10),
            fontsize=_FONT - 1, ha="center", va="center", zorder=7,
        )
        # Arrow from dot to circle edge
        _draw_arrow(ax, cx + _DOT / 2, cy, px_coord - _CIRC_R, py)

    # ── Reaction name (small, bottom of block) ────────────────────────────────
    ax.text(
        cx, oy + _BLOCK_H - 8,
        _wrap(rxn_label, width=30),
        fontsize=_FONT - 2, ha="center", va="bottom",
        color="#555555", zorder=5,
    )


def render_draft_graph_to_png_bytes(draft_graph_dict: Dict[str, Any], dpi: int = 160) -> bytes:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for graph rendering.") from exc

    nodes: List[Dict] = draft_graph_dict.get("nodes", [])
    edges: List[Dict] = draft_graph_dict.get("edges", [])

    if not nodes:
        raise RuntimeError("DraftGraph has no nodes — nothing to render.")

    ordered, node_by_id, rxn_reactants, rxn_products, rxn_enzymes = _parse_graph(nodes, edges)

    if not ordered:
        raise RuntimeError("DraftGraph has no reaction nodes — nothing to render.")

    n_rxns = len(ordered)
    cols = min(_COLS, n_rxns)
    rows = (n_rxns + cols - 1) // cols

    canvas_w = cols * _BLOCK_W
    canvas_h = rows * _BLOCK_H

    fig_w = max(8.0, canvas_w / 96)
    fig_h = max(4.0, canvas_h / 96)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)   # y increases downward
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for idx, rxn_id in enumerate(ordered):
        col = idx % cols
        row = idx // cols
        ox = col * _BLOCK_W
        oy = row * _BLOCK_H

        rxn_node = node_by_id.get(rxn_id, {})
        rxn_label = rxn_node.get("label", rxn_id)

        reactant_ids = rxn_reactants.get(rxn_id, [])
        product_ids  = rxn_products.get(rxn_id, [])
        enzyme_names = rxn_enzymes.get(rxn_id, [])

        reactant_labels = [node_by_id[c].get("label", c) for c in reactant_ids if c in node_by_id]
        product_labels  = [node_by_id[c].get("label", c) for c in product_ids  if c in node_by_id]

        _draw_reaction_block(ax, ox, oy, rxn_label, reactant_labels, product_labels, enzyme_names)

    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
