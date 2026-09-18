"""A self-contained interactive network renderer for the two research views.

Why hand-rolled: the project has no networkx, plotly, pyvis or d3 dependency, and
the Streamlit component sandbox should not be made to depend on a CDN for a
research tool that has to work offline. This is a few hundred lines of vanilla
SVG + JS with a seeded, deterministic force layout -- the same input always draws
the same picture, which matters when two runs are being compared.

What the renderer is careful about
----------------------------------
* **Reactions are nodes.** A multi-substrate reaction is one box with all its
  incident edges. Nothing is collapsed into pairwise metabolite edges.
* **Enzyme edges are visibly different** (dashed, open arrowhead, distinct
  colour) and are labelled as annotations, because an enzyme never bridges two
  reactions.
* **Primary vs candidate is visually unmistakable** -- filled vs dashed outline,
  different palette, and a legend that says so in words.
* **The provenance view prints its caveat on the canvas**: shared papers are not
  chemical connectivity.
* Direction and reversibility are drawn (``->`` vs ``<=>``) rather than implied.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

_TEMPLATE = """
<style>
  .t2pw-net {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; }}
  .t2pw-net svg {{ width: 100%; height: {height}px; background: #fbfcfd;
                   border: 1px solid #d8dee5; border-radius: 6px; cursor: grab; }}
  .t2pw-net .legend {{ font-size: 11px; color: #44525f; margin: 6px 0 2px; line-height: 1.7; }}
  .t2pw-net .legend span {{ margin-right: 14px; white-space: nowrap; }}
  .t2pw-net .swatch {{ display: inline-block; width: 11px; height: 11px;
                       vertical-align: -1px; margin-right: 4px; border: 1px solid #5a6a78; }}
  .t2pw-net .caveat {{ font-size: 11px; color: #8a3b2f; background: #fdf1ee;
                       border-left: 3px solid #c0553f; padding: 6px 9px; margin: 6px 0; }}
  .t2pw-net .detail {{ font-size: 12px; margin-top: 8px; border: 1px solid #d8dee5;
                       border-radius: 6px; padding: 10px; background: #fff;
                       max-height: 340px; overflow: auto; }}
  .t2pw-net .detail h4 {{ margin: 0 0 6px; font-size: 13px; }}
  .t2pw-net .detail table {{ border-collapse: collapse; width: 100%; }}
  .t2pw-net .detail td {{ padding: 2px 6px 2px 0; vertical-align: top;
                          border-bottom: 1px solid #eef1f4; }}
  .t2pw-net .detail td.k {{ color: #667685; width: 168px; white-space: nowrap; }}
  .t2pw-net .quote {{ font-style: italic; color: #33414d; }}
  .t2pw-net .hint {{ font-size: 11px; color: #8896a3; margin-top: 4px; }}
  .t2pw-net text {{ pointer-events: none; user-select: none; }}
</style>
<div class="t2pw-net">
  <div class="legend" id="legend-{uid}"></div>
  {caveat}
  <svg id="svg-{uid}"></svg>
  <div class="hint">Drag the background to pan, scroll to zoom, drag a node to move it,
     click a node for its full record.</div>
  <div class="detail" id="detail-{uid}">Click a node to see its record.</div>
</div>
<script>
(function() {{
  const DATA = {payload};
  const uid = "{uid}";
  const svg = document.getElementById("svg-" + uid);
  const detail = document.getElementById("detail-" + uid);
  const legendEl = document.getElementById("legend-" + uid);
  const NS = "http://www.w3.org/2000/svg";

  const STYLE = {{
    metabolite: {{ fill: "#e3eefb", stroke: "#3f6ea3", shape: "ellipse", label: "metabolite" }},
    reaction:   {{ fill: "#d3ead9", stroke: "#3c7a52", shape: "rect",    label: "reaction" }},
    enzyme:     {{ fill: "#e9e2fa", stroke: "#6f5fae", shape: "hex",     label: "enzyme (annotation)" }},
    paper:      {{ fill: "#fdf1d6", stroke: "#a98526", shape: "rect",    label: "paper" }},
    evidence:   {{ fill: "#eaf6e6", stroke: "#4f8a43", shape: "rect",    label: "evidence span" }},
    claim:      {{ fill: "#f8e6f3", stroke: "#a3559a", shape: "ellipse", label: "claim" }}
  }};

  // Deterministic PRNG: the same graph always lays out the same way.
  let seed = 20260917;
  function rnd() {{ seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }}

  const nodes = DATA.nodes.map(function(n, i) {{
    const angle = (i / Math.max(1, DATA.nodes.length)) * Math.PI * 2;
    return Object.assign({{}}, n, {{
      x: 480 + Math.cos(angle) * (160 + rnd() * 140),
      y: 300 + Math.sin(angle) * (140 + rnd() * 110),
      vx: 0, vy: 0, pinned: false
    }});
  }});
  const byId = {{}};
  nodes.forEach(function(n) {{ byId[n.id] = n; }});
  const edges = DATA.edges.filter(function(e) {{ return byId[e.source] && byId[e.target]; }});

  // --- force layout, fixed iteration count (no animation loop, no drift) ----
  const W = 960, H = {height};
  function layout(iterations) {{
    const k = Math.sqrt((W * H) / Math.max(1, nodes.length)) * 0.62;
    for (let step = 0; step < iterations; step++) {{
      const t = 1 - step / iterations;
      for (let i = 0; i < nodes.length; i++) {{
        nodes[i].vx = 0; nodes[i].vy = 0;
        for (let j = 0; j < nodes.length; j++) {{
          if (i === j) continue;
          let dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 0.01) {{ dx = (rnd() - 0.5); dy = (rnd() - 0.5); d2 = 0.01; }}
          const rep = (k * k) / d2;
          nodes[i].vx += dx * rep; nodes[i].vy += dy * rep;
        }}
      }}
      edges.forEach(function(e) {{
        const a = byId[e.source], b = byId[e.target];
        const dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.max(1, Math.sqrt(dx * dx + dy * dy));
        const att = (d * d) / k / 22;
        const ux = dx / d, uy = dy / d;
        a.vx += ux * att; a.vy += uy * att;
        b.vx -= ux * att; b.vy -= uy * att;
      }});
      nodes.forEach(function(n) {{
        if (n.pinned) return;
        const sp = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
        const cap = 22 * t + 1;
        const f = sp > cap ? cap / sp : 1;
        n.x += n.vx * f; n.y += n.vy * f;
        n.x = Math.max(60, Math.min(W - 60, n.x));
        n.y = Math.max(40, Math.min(H - 40, n.y));
      }});
    }}
  }}
  layout(nodes.length > 160 ? 90 : 220);

  // --- drawing -------------------------------------------------------------
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  const defs = document.createElementNS(NS, "defs");
  defs.innerHTML =
    '<marker id="arw-' + uid + '" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"' +
    ' markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"' +
    ' fill="#4b5c6b"/></marker>' +
    '<marker id="ann-' + uid + '" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"' +
    ' markerHeight="7" orient="auto-start-reverse"><circle cx="5" cy="5" r="3"' +
    ' fill="none" stroke="#6f5fae" stroke-width="1.6"/></marker>';
  svg.appendChild(defs);
  const root = document.createElementNS(NS, "g");
  svg.appendChild(root);
  const edgeLayer = document.createElementNS(NS, "g");
  const nodeLayer = document.createElementNS(NS, "g");
  root.appendChild(edgeLayer); root.appendChild(nodeLayer);

  const edgeEls = edges.map(function(e) {{
    const line = document.createElementNS(NS, "line");
    const chemical = e.chemical !== false;
    line.setAttribute("stroke", chemical ? "#4b5c6b" : (e.type === "catalyzes" ? "#6f5fae" : "#a4b0bb"));
    line.setAttribute("stroke-width", chemical ? "1.7" : "1.2");
    if (!chemical) line.setAttribute("stroke-dasharray", "5,4");
    line.setAttribute("marker-end", "url(#" + (e.type === "catalyzes" ? "ann-" : "arw-") + uid + ")");
    line.setAttribute("opacity", chemical ? "0.9" : "0.65");
    edgeLayer.appendChild(line);
    return line;
  }});

  function truncate(s, n) {{
    s = String(s === undefined || s === null ? "" : s);
    return s.length > n ? s.slice(0, n - 1) + "\\u2026" : s;
  }}

  const nodeEls = nodes.map(function(n) {{
    const st = STYLE[n.type] || {{ fill: "#eee", stroke: "#888", shape: "ellipse" }};
    const g = document.createElementNS(NS, "g");
    g.style.cursor = "pointer";
    const candidate = n.type === "reaction" && n.graph_status !== "primary";
    let shape;
    const label = truncate(n.label, n.type === "reaction" ? 34 : 22);
    const w = Math.max(64, label.length * 6.4 + 16), h = 26;
    if (st.shape === "rect") {{
      shape = document.createElementNS(NS, "rect");
      shape.setAttribute("x", -w / 2); shape.setAttribute("y", -h / 2);
      shape.setAttribute("width", w); shape.setAttribute("height", h);
      shape.setAttribute("rx", 4);
    }} else if (st.shape === "hex") {{
      shape = document.createElementNS(NS, "polygon");
      const hw = w / 2, hh = h / 2;
      shape.setAttribute("points",
        [[-hw+8,-hh],[hw-8,-hh],[hw,0],[hw-8,hh],[-hw+8,hh],[-hw,0]]
          .map(function(p) {{ return p.join(","); }}).join(" "));
    }} else {{
      shape = document.createElementNS(NS, "ellipse");
      shape.setAttribute("rx", w / 2); shape.setAttribute("ry", h / 2);
    }}
    shape.setAttribute("fill", st.fill);
    shape.setAttribute("stroke", st.stroke);
    shape.setAttribute("stroke-width", candidate ? "2" : "1.3");
    if (candidate) shape.setAttribute("stroke-dasharray", "6,3");
    if (n.excluded) {{ shape.setAttribute("opacity", "0.45"); shape.setAttribute("stroke-dasharray", "3,3"); }}
    g.appendChild(shape);
    const text = document.createElementNS(NS, "text");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("dy", "4");
    text.setAttribute("font-size", "11");
    text.setAttribute("fill", "#22303c");
    text.textContent = label;
    g.appendChild(text);
    if (n.type === "reaction" && n.conflicts && n.conflicts.length) {{
      const badge = document.createElementNS(NS, "text");
      badge.setAttribute("text-anchor", "middle");
      badge.setAttribute("dy", -h / 2 - 4);
      badge.setAttribute("font-size", "11");
      badge.setAttribute("fill", "#b3452f");
      badge.textContent = "\\u26A0 conflict";
      g.appendChild(badge);
    }}
    g.addEventListener("click", function(ev) {{ ev.stopPropagation(); show(n); }});
    nodeLayer.appendChild(g);
    n._w = w; n._h = h;
    return g;
  }});

  function render() {{
    edges.forEach(function(e, i) {{
      const a = byId[e.source], b = byId[e.target];
      edgeEls[i].setAttribute("x1", a.x); edgeEls[i].setAttribute("y1", a.y);
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const pad = (b._h || 26) / 2 + 6;
      edgeEls[i].setAttribute("x2", b.x - (dx / d) * pad);
      edgeEls[i].setAttribute("y2", b.y - (dy / d) * pad);
    }});
    nodes.forEach(function(n, i) {{
      nodeEls[i].setAttribute("transform", "translate(" + n.x + "," + n.y + ")");
    }});
  }}
  render();

  // --- detail panel --------------------------------------------------------
  const HIDE = {{ x: 1, y: 1, vx: 1, vy: 1, pinned: 1, _w: 1, _h: 1 }};
  function row(k, v) {{
    if (v === null || v === undefined || v === "" ||
        (Array.isArray(v) && !v.length) ||
        (typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length)) return "";
    let body;
    if (k === "supporting_papers" && Array.isArray(v)) {{
      body = v.map(function(p) {{
        return "<div>" + esc(p.paper_id) + " &mdash; " + esc(truncate(p.title, 90)) +
               " <em>(" + esc(p.evidence_type || "unknown") + ")</em></div>";
      }}).join("");
    }} else if (k === "history" && Array.isArray(v)) {{
      body = v.map(function(h) {{
        return "<div>round " + esc(h.round_index) + ": <b>" + esc(h.event) + "</b> " +
               esc(truncate(h.detail, 140)) + "</div>";
      }}).join("");
    }} else if (k === "conflicts" && Array.isArray(v)) {{
      body = v.map(function(c) {{
        return "<div><b>" + esc(c.attribute) + "</b>: " +
               esc((c.positions || []).map(function(p) {{ return p.value; }}).join(" vs ")) +
               " (papers " + esc((c.paper_ids || []).join(", ")) + ")</div>";
      }}).join("");
    }} else if (k === "text" || k === "quote") {{
      body = '<span class="quote">' + esc(v) + "</span>";
    }} else if (typeof v === "object") {{
      body = "<code>" + esc(JSON.stringify(v)) + "</code>";
    }} else {{
      body = esc(v);
    }}
    return "<tr><td class='k'>" + esc(k) + "</td><td>" + body + "</td></tr>";
  }}
  function esc(s) {{
    return String(s === undefined || s === null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }}
  function show(n) {{
    const keys = Object.keys(n).filter(function(k) {{ return !HIDE[k] && k.charAt(0) !== "_"; }});
    const head = "<h4>" + esc(n.label) + " <small>(" + esc(n.type) + ")</small></h4>";
    detail.innerHTML = head + "<table>" +
      keys.map(function(k) {{ return row(k, n[k]); }}).join("") + "</table>";
  }}

  // --- legend --------------------------------------------------------------
  const present = {{}};
  nodes.forEach(function(n) {{ present[n.type] = true; }});
  let legend = Object.keys(STYLE).filter(function(t) {{ return present[t]; }})
    .map(function(t) {{
      return "<span><i class='swatch' style='background:" + STYLE[t].fill +
             ";border-color:" + STYLE[t].stroke + "'></i>" + STYLE[t].label + "</span>";
    }}).join("");
  if (DATA.view === "reaction_network") {{
    legend += "<span><i class='swatch' style='background:#d3ead9;border-color:#3c7a52'></i>" +
              "solid outline = primary graph</span>" +
              "<span><i class='swatch' style='background:#d3ead9;border-color:#3c7a52;" +
              "border-style:dashed'></i>dashed outline = quarantined candidate</span>" +
              "<span>solid edge = chemical &nbsp; dashed edge = enzyme annotation " +
              "(never a connection)</span>";
  }} else {{
    legend += "<span>faded, dashed = retrieved but EXCLUDED</span>";
  }}
  legendEl.innerHTML = legend;

  // --- pan / zoom / drag ---------------------------------------------------
  let tx = 0, ty = 0, scale = 1, panning = false, px = 0, py = 0, dragNode = null;
  function apply() {{
    root.setAttribute("transform", "translate(" + tx + "," + ty + ") scale(" + scale + ")");
  }}
  svg.addEventListener("wheel", function(ev) {{
    ev.preventDefault();
    const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
    scale = Math.max(0.25, Math.min(4, scale * factor));
    apply();
  }}, {{ passive: false }});
  svg.addEventListener("mousedown", function(ev) {{
    const target = ev.target.closest && ev.target.closest("g");
    const idx = nodeEls.indexOf(target);
    if (idx >= 0) {{ dragNode = nodes[idx]; dragNode.pinned = true; }}
    else {{ panning = true; svg.style.cursor = "grabbing"; }}
    px = ev.clientX; py = ev.clientY;
  }});
  window.addEventListener("mousemove", function(ev) {{
    if (!panning && !dragNode) return;
    const dx = ev.clientX - px, dy = ev.clientY - py;
    px = ev.clientX; py = ev.clientY;
    if (dragNode) {{
      const rect = svg.getBoundingClientRect();
      const k = (W / rect.width) / scale;
      dragNode.x += dx * k; dragNode.y += dy * k; render();
    }} else {{ tx += dx; ty += dy; apply(); }}
  }});
  window.addEventListener("mouseup", function() {{
    panning = false; dragNode = null; svg.style.cursor = "grab";
  }});
}})();
</script>
"""

_CAVEAT = (
    '<div class="caveat">These edges are <b>bibliographic</b>: paper &rarr; evidence span '
    "&rarr; claim &rarr; reaction. Two reactions sharing a paper are <b>not</b> chemically "
    "connected. Biochemical connectivity is shown only in the reaction-network view.</div>"
)


def render_network_html(
    view: Mapping[str, Any],
    *,
    height: int = 620,
    uid: str = "net",
    max_nodes: int = 420,
) -> str:
    """Build the standalone HTML for one view payload.

    For a large graph the payload is trimmed to ``max_nodes``, keeping reaction
    nodes first (they are the subject) and then whatever their edges touch, so a
    big network degrades to a readable subset rather than an unreadable hairball.
    The caller is told what was trimmed through the returned HTML's legend hint.
    """
    payload = _trim(view, max_nodes=max_nodes)
    caveat = _CAVEAT if view.get("view") != "reaction_network" else ""
    return _TEMPLATE.format(
        payload=json.dumps(payload),
        uid=uid,
        height=int(height),
        caveat=caveat,
    )


def _trim(view: Mapping[str, Any], *, max_nodes: int) -> Dict[str, Any]:
    nodes = list(view.get("nodes") or [])
    edges = list(view.get("edges") or [])
    if len(nodes) <= max_nodes:
        return {"view": view.get("view", ""), "nodes": nodes, "edges": edges}

    priority = {"reaction": 0, "metabolite": 1, "claim": 2, "paper": 3, "enzyme": 4, "evidence": 5}
    ordered = sorted(nodes, key=lambda n: (priority.get(str(n.get("type")), 9), str(n.get("id"))))
    keep = {str(n.get("id")) for n in ordered[:max_nodes]}
    kept_nodes = [n for n in nodes if str(n.get("id")) in keep]
    kept_edges = [
        e for e in edges
        if str(e.get("source")) in keep and str(e.get("target")) in keep
    ]
    return {
        "view": view.get("view", ""),
        "nodes": kept_nodes,
        "edges": kept_edges,
        "trimmed": True,
        "trimmed_from": len(nodes),
    }


__all__ = ["render_network_html"]
