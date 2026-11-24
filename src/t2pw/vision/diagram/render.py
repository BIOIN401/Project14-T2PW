"""T2PW pathway diagram prototype: one automatic layout engine, three styles.

Input : a finished run's pwml_ir.json (read only).  Output: SVG strings.
Everything is derived from the data. No per-paper coordinates, no per-paper cases.
Run:  python t2pw_diagrams.py   (see build_page.py for the page builder)
"""
import heapq
import math
import re
from collections import defaultdict, Counter
from xml.sax.saxutils import escape

from PIL import ImageFont

from t2pw.vision.currency import CURRENCY, is_currency  # noqa: F401  one list with checks/layout_metrics


def _font(size):
    """Text-width measuring font. The prototype used Segoe UI; layout widths depend on it, so output is only
    byte-identical to the prototype where that font exists. Order: T2PW_DIAGRAM_FONT, Segoe UI if present, the bundled
    DejaVu Sans (VD-9, so Linux output is fixed), PIL's default."""
    import os
    for p in (os.environ.get("T2PW_DIAGRAM_FONT"), "C:/Windows/Fonts/segoeui.ttf",
              os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")):
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:  # exists but is not a font: try the next one
                continue
    return ImageFont.load_default(size)


_FONT = _font(12)
_FONT_S = _font(10)


def tw(s, small=False):
    return (_FONT_S if small else _FONT).getlength(s) * 1.06  # 6% safety for other fonts


def wrap(text, maxw, maxlines=3):
    """Greedy wrap on spaces and hyphens. Returns (lines, truncated)."""
    toks = re.findall(r"[^\s-]+-?|\s+", text.strip())
    toks = [t for t in re.split(r"(?<=[ -])", text.strip()) if t != ""]
    lines, cur = [], ""
    for t in toks:
        if cur and tw((cur + t).rstrip()) > maxw:
            lines.append(cur.rstrip())
            cur = t
        else:
            cur += t
    if cur.strip():
        lines.append(cur.rstrip())
    trunc = False
    if len(lines) > maxlines:
        lines = lines[:maxlines]
        lines[-1] = lines[-1].rstrip() + "\u2026"
        trunc = True
    return lines, trunc


# --------------------------------------------------------------------------- model
ENT_KEYS = {"compound": "compounds", "protein": "proteins", "protein_complex": "protein_complexes",
            "nucleic_acid": "nucleic_acids", "element_collection": "element_collections", "bound": "bounds"}
KIND = {"compound": "compound", "protein": "protein", "protein_complex": "complex", "nucleic_acid": "nucleic"}


def norm_name(s):
    s = re.sub(r"[^a-z0-9 ]", "", (s or "").lower())
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in s.split())


def comp_role(name, auto=False):
    n = name.lower()
    if auto:
        return "auto", -1.0
    if n in ("cell", "whole cell"):
        return "cell", 3.6
    if "membrane" in n or "leaflet" in n:
        if "outer" in n:
            return "membrane", 1.0
        if "periplasmic" in n:
            return "membrane", 2.9
        if "cytoplasmic" in n:
            return "membrane", 3.1
        return "membrane", 3.0
    if "periplasm" in n:
        return "periplasm", 2.0
    if any(k in n for k in ("extracellular", "saliva", "blood", "lumen", "cleft", "medium", "serum", "outside",
                            "apoplast", "interstitial", "urine", "bile", "csf", "space")) or n == "plasma":
        return "extra", 0.0
    if n.startswith("cyto") or n in ("cytosol", "cytoplasm"):
        return "cytosol", 4.0
    return "organelle", 5.0


def icls(t):
    t = (t or "").lower()
    if any(k in t for k in ("inhibit", "repress", "downregul", "suppress")):
        return "inh"
    if any(k in t for k in ("activat", "stimulat", "upregul", "induc")):
        return "act"
    if "catalyz" in t:
        return "enz"
    return "assoc"


# interaction types that name a direction (left acts on right) but carry no activate/inhibit sign: drawn with an arrowhead
DIRECTED = ("regulat", "modul", "modif", "substrate", "transport", "compet", "component", "cofactor_of", "cofactor of",
            "phosphopantetheinyl", "degrad", "provides", "aids")


def is_directed(t):
    t = (t or "").lower()
    return any(k in t for k in DIRECTED)


class Node:
    def __init__(self, nid, kind, label, comp=None, ent=None, cls="", full=None):
        self.id, self.kind, self.label, self.comp, self.ent, self.cls = nid, kind, label, comp, ent, cls
        self.full = full or label
        self.layer = 0
        self.x = self.y = 0.0
        self.region = None
        self.row = None
        self.tag = ""
        self.note = ""
        self.et = ""
        self.merge_note = ""
        self.truncated = False
        self.measure()

    def measure(self):
        if self.kind == "junction":
            tw_ = tw(self.label, True) + 12
            self.lines = [self.label]
            self.w, self.h = max(30, tw_), 20
            return
        maxw = 150 if self.kind != "compound" else 128
        self.lines, self.truncated = wrap(self.label, maxw)
        lw = max(tw(l) for l in self.lines)
        self.h = 14 * len(self.lines) + 14
        pad = {"compound": 24, "protein": 18, "complex": 34, "nucleic": 36}[self.kind]
        self.w = lw + pad

    def rect(self, pad=0):
        return (self.x - self.w / 2 - pad, self.y - self.h / 2 - pad, self.x + self.w / 2 + pad, self.y + self.h / 2 + pad)


class Edge:
    def __init__(self, a, b, kind, tip="", badge="", withheld=False, rev=False):
        self.a, self.b, self.kind, self.tip, self.badge, self.withheld, self.rev = a, b, kind, tip, badge, withheld, rev
        self.pts = None


class Model:
    pass


def build(ir):
    m = Model()
    ent = {}
    for t, k in ENT_KEYS.items():
        for e in ir["entities"].get(k, []):
            ent[(t, e["key"])] = e
    ctype = {c["key"]: c["name"] for c in ir.get("cell_types", [])}
    loc = {l["key"]: l["name"] for l in ir.get("subcellular_locations", [])}
    comps = {}
    state2comp = {}
    for s in ir.get("biological_states", []):
        auto = s["name"].startswith("__auto")
        lname = loc.get(s.get("subcellular_location_key"), s["name"])
        role, rank = comp_role(lname, auto)
        grp = ctype.get(s.get("cell_type_key"))
        ck = ("auto",) if auto else (s.get("subcellular_location_key"), s.get("cell_type_key"))
        if ck not in comps:
            comps[ck] = dict(key=ck, name="unplaced (auto state)" if auto else lname, role=role, rank=rank, group=grp,
                             nodes=[], states=[])
        comps[ck]["states"].append(s["key"])
        state2comp[s["key"]] = ck
    if ("none",) not in comps:
        comps[("none",)] = dict(key=("none",), name="unplaced (no state)", role="auto", rank=-1.0, group=None, nodes=[], states=[])
    m.comps, m.state2comp = comps, state2comp
    m.nodes, m.edges, m.procs = {}, [], []
    m.skipped = 0
    m.stats = Counter()

    def cof(state):
        return state2comp.get(state, ("none",))

    def inst(etype, ekey, ck, sfx=None):
        key = (etype, ekey, ck) if sfx is None else (etype, ekey, ck, sfx)
        if key in m.nodes:
            return m.nodes[key]
        e = ent.get((etype, ekey))
        if e is None or etype not in KIND:
            return None
        name = e.get("name") or ekey
        if name.strip().lower() == "unknown":
            name = "Unknown protein"
        cls = e.get("class") or e.get("type") or ""
        n = Node(key, KIND[etype], name, ck, ekey, cls if etype == "compound" else "")
        n.et = etype
        m.nodes[key] = n
        comps[ck]["nodes"].append(n)
        return n

    def junction(label, ck, kind_):
        n = Node(("J", label), "junction", label, ck)
        n.jkind = kind_
        m.nodes[n.id] = n
        comps[ck]["nodes"].append(n)
        return n

    P = ir.get("processes", {})
    pw_name = ((ir.get("pathway") or {}).get("name") or "").lower()
    uses = Counter((x["entity_type"], x["entity_key"], cof(r.get("biological_state_key")))
                   for r in P.get("reactions", []) for x in {(y["entity_type"], y["entity_key"]): y for y in r["left"] + r["right"]}.values())
    # transport cargo keeps its single drawing in each compartment it moves between: a local copy would cut the link
    # between the transport and the reactions that use what it delivers (H+ pumped into the matrix, then used there)
    cargo = {(el["entity_type"], el["entity_key"], cof(el.get(side)))
             for t in P.get("transports", []) for el in t.get("transport_elements", [])
             for side in ("left_biological_state_key", "right_biological_state_key")}
    sigs = defaultdict(list)
    # reactions
    msig = {}
    for i, r in enumerate(P.get("reactions", [])):
        ck = cof(r.get("biological_state_key"))
        tag = "R" + re.sub(r"\D", "", r["key"])
        # I3: identical steps (same compartment, same substrates and products, same direction) are drawn once;
        # every enzyme of every copy is kept and the data is untouched.
        ms = (frozenset((x["entity_type"], x["entity_key"]) for x in r["left"]),
              frozenset((x["entity_type"], x["entity_key"]) for x in r["right"]), ck, r.get("direction") or ">")
        J0 = msig.get(ms)
        if J0 is not None:
            for x in r.get("enzymes", []):
                z = inst(x["entity_type"], x["entity_key"], ck, J0.proc["key"])
                if z and z not in J0.enz:
                    J0.enz.append(z)
                    m.edges.append(Edge(z, J0, "cat"))
            J0.merged.append(r)
            J0.mtags.append(tag)
            J0.tip += " | also stated as " + tag + ": " + r.get("name", "")
            J0.merge_note = f"×{len(J0.merged)} in data"
            continue
        J = junction(tag, ck, "reaction")
        J.merged = [r]
        J.mtags = [tag]
        J.merge_note = ""
        msig[ms] = J
        J.proc = r
        # currency molecules (H+, water, ATP, CoA ...) get a local copy per reaction, as curated maps draw them
        def cur(x):  # CoA in a CoA pathway is the story, not currency
            nm = (ent.get(("compound", x["entity_key"])) or {}).get("name") or ""
            return (x["entity_type"] == "compound" and is_currency(nm) and nm.strip().lower() not in pw_name
                    and uses[(x["entity_type"], x["entity_key"], ck)] > 1  # only a hub needs local copies
                    and (x["entity_type"], x["entity_key"], ck) not in cargo)
        subs = [inst(x["entity_type"], x["entity_key"], ck, ("cur", r["key"]) if cur(x) else None) for x in r["left"]]
        prods = [inst(x["entity_type"], x["entity_key"], ck, ("cur", r["key"]) if cur(x) else None) for x in r["right"]]
        for n in subs + prods:
            if n is not None and len(n.id) == 4 and n.id[3][0] == "cur":
                n.cur = True
        # I2: one enzyme drawing per catalysed reaction (suffix = the reaction key), so it sits beside its reaction
        enz = [inst(x["entity_type"], x["entity_key"], ck, r["key"]) for x in r.get("enzymes", [])]
        subs, prods = [n for n in subs if n], [n for n in prods if n]
        enz = list(dict.fromkeys(n for n in enz if n))
        rev = "<" in (r.get("direction") or ">")
        for s in subs:
            m.edges.append(Edge(s, J, "sub", rev=rev))
        for p in prods:
            m.edges.append(Edge(J, p, "prod"))
        for k, z in enumerate(enz):
            m.edges.append(Edge(z, J, "cat"))
        J.enz = enz
        J.subs, J.prods = subs, prods
        J.tip = r.get("name", "")
        sigs[("rx", frozenset(norm_name(s.label) for s in subs), frozenset(norm_name(p.label) for p in prods), ck)].append(J)
        m.procs.append(dict(tag=tag, kind="reaction", name=r.get("name", ""), where=comps[ck]["name"], node=J,
                            spont=r.get("spontaneous"), det=" + ".join(s.label for s in subs) + " → " + " + ".join(p.label for p in prods),
                            enz=[z.label for z in enz]))
    # transports
    for t in P.get("transports", []):
        tag = "T" + re.sub(r"\D", "", t["key"])
        pairs = []
        for el in t.get("transport_elements", []):
            a = cof(el.get("left_biological_state_key"))
            b = cof(el.get("right_biological_state_key"))
            pairs.append((el, a, b))
        trs = t.get("transporters", [])
        J = Node(("J", tag), "junction", tag, None)
        J.jkind = "transport"
        J.proc = t
        J.pairs = pairs
        J.trs = trs
        J.tip = t.get("name", "")
        J.srcs, J.dsts = [], []
        m.nodes[J.id] = J
        # comps assigned in layout stage (membrane choice); keep ends here
        for el, a, b in pairs:
            s = inst(el["entity_type"], el["entity_key"], a)
            d = inst(el["entity_type"], el["entity_key"], b)
            if s and d:
                J.srcs.append(s)
                J.dsts.append(d)
                m.edges.append(Edge(s, J, "xin"))
                m.edges.append(Edge(J, d, "xout"))
        J.trn_ents = [(x["entity_type"], x["entity_key"], cof(x.get("biological_state_key"))) for x in trs]
        sigs[("tr", frozenset(norm_name(ent[(e[0], e[1])]["name"]) for e in J.trn_ents if (e[0], e[1]) in ent),
              frozenset(norm_name(s.label) for s in J.srcs)) ].append(J)
        m.procs.append(dict(tag=tag, kind="transport", name=t.get("name", ""),
                            where=" | ".join(f"{comps[a]['name']} \u2192 {comps[b]['name']}" for _, a, b in pairs), node=J,
                            det=", ".join(s.label for s in J.srcs),
                            enz=[ent[(e[0], e[1])]["name"] for e in J.trn_ents if (e[0], e[1]) in ent],
                            notr=not trs))
    # interactions
    ni = 0
    for it in P.get("interactions", []):
        ck = cof(it.get("biological_state_key"))
        l, r = it["left"], it["right"]
        a = inst(l["entity_type"], l["entity_key"], ck)
        b = inst(r["entity_type"], r["entity_key"], ck)
        if not a or not b:
            m.skipped += 1
            continue
        ni += 1
        tag = "I" + re.sub(r"\D", "", it["key"])
        cl = icls(it.get("interaction_type"))
        wh = it.get("export_status") == "withheld"
        e = Edge(a, b, cl, tip=f"{it.get('name', '')} ({it.get('interaction_type')})", withheld=wh)
        e.tag = tag
        e.key = it["key"]
        e.directed = cl == "assoc" and is_directed(it.get("interaction_type"))
        m.edges.append(e)
        sigs[("int", cl, norm_name(a.label), norm_name(b.label), ck)].append(e)
        m.procs.append(dict(tag=tag, kind="interaction", name=it.get("name", ""), where=comps[ck]["name"], edge=e,
                            itype=it.get("interaction_type"), cls=cl, withheld=wh, det=f"{a.label} \u2192 {b.label}"))
    # duplicate marking: same signature -> draw both, add a note
    m.dups = []
    for sig, items in sigs.items():
        if len(items) > 1:
            tags = [(x.label if isinstance(x, Node) else x.tag) for x in items]
            for x in items:
                others = [t for t in tags if t != (x.label if isinstance(x, Node) else x.tag)]
                pre = "same as "
                if isinstance(x, Node):
                    x.note = pre + ", ".join(others)
                else:
                    x.badge = pre + ", ".join(others)
            m.dups.append(tags)
    for p in m.procs:
        for g in m.dups:
            if p["tag"] in g:
                p["dup"] = [t for t in g if t != p["tag"]]
    # sign unknown regulation count
    m.stats.update(t=len(P.get("transports", [])), r=len(P.get("reactions", [])), i=ni)
    # drop comps without content
    return m


def finalize_comps(m):
    """Transport junctions are placed later; remove empty compartments afterwards."""
    pass


# --------------------------------------------------------------------------- layering
def assign_layers(m, style=None):
    nodes = list(m.nodes.values())
    cons = []  # (u, v, gap)
    proc_kinds = ("sub", "prod", "xin", "xout", "act", "inh", "cat")
    has_proc = set()
    for e in m.edges:
        if e.kind in proc_kinds:
            has_proc.add(e.a.id)
            has_proc.add(e.b.id)
    # satellites: an enzyme-like molecule whose only links are 'acts on / binds' links to a compound or to a
    # molecule that is part of a process. It is drawn next to that partner instead of in its own column.
    for n in nodes:
        n.sat = None
    for e in m.edges:
        if e.kind in ("assoc", "enz"):
            for p, q in ((e.a, e.b), (e.b, e.a)):
                if (p.kind in ("protein", "complex") and p.id not in has_proc and p.sat is None
                        and (q.kind == "compound" or q.id in has_proc) and q.kind != "nucleic"):
                    p.sat = q
    for e in m.edges:
        if e.kind == "cat":
            continue
        if e.kind in ("assoc", "enz") and (e.a.sat is e.b or e.b.sat is e.a):
            continue
        if e.kind in ("assoc", "enz") and (e.a.sat is not None or e.b.sat is not None):
            continue
        # a transport is drawn straight across the membrane: cargo, transport and cargo copy share one column
        cons.append((e.a, e.b, 0 if e.kind in ("xin", "xout") and style != "spine" else 1))  # spine pins the chain on one row
    # copies of one molecule in two compartments with no transport between them: the copy nothing produces is drawn
    # no further left than a produced copy, so a downstream step continues the flow instead of restarting at the left
    has_in = {v.id for u, v, g in cons}
    cargo = {x.id for e in m.edges if e.kind in ("xin", "xout") for x in (e.a, e.b)}
    copies = defaultdict(list)
    for n in nodes:
        if n.kind != "junction" and not getattr(n, "cur", False) and len(n.id) == 3:
            copies[(n.et, n.ent)].append(n)
    for cs in copies.values():
        made = [c for c in cs if c.id in has_in]
        for c in cs:
            if made and c.id not in has_in and c.id not in cargo and c.comp != made[0].comp:
                cons.append((made[0], c, 0))
    adj = defaultdict(list)
    indeg = Counter()
    # break cycles by DFS back-edge removal (deterministic order)
    out = defaultdict(list)
    for u, v, g in cons:
        out[u.id].append((v, g))
    state = {}
    keep = []

    def dfs(start):
        stack = [(start, iter(out[start.id]))]
        state[start.id] = 1
        while stack:
            u, it = stack[-1]
            for v, g in it:
                if state.get(v.id) == 1:
                    continue  # back edge, dropped
                keep.append((u, v, g))
                if state.get(v.id) is None:
                    state[v.id] = 1
                    stack.append((v, iter(out[v.id])))
                    break
            else:
                state[u.id] = 2
                stack.pop()

    for n in nodes:
        if state.get(n.id) is None:
            dfs(n)
    import sys
    sys.setrecursionlimit(10000)

    def longest(keep):
        preds = defaultdict(list)
        for u, v, g in keep:
            preds[v.id].append((u, g))
        layer = {}

        def L(n):
            if n.id in layer:
                return layer[n.id]
            layer[n.id] = 0
            best = 0
            for u, g in preds[n.id]:
                best = max(best, L(u) + g)
            layer[n.id] = best
            return best

        for n in nodes:
            L(n)
        return layer

    layer = longest(keep)
    # two transports in one membrane band must not share a column (their straight lines would lie on top of each other)
    def reaches(a, b, out):
        todo, done = [a.id], set()
        while todo:
            x = todo.pop()
            if x == b.id:
                return True
            if x not in done:
                done.add(x)
                todo += [v.id for v in out.get(x, ())]
        return False

    for _ in range(8):
        seen, extra = {}, []
        out = defaultdict(list)
        for u, v, g in keep:
            out[u.id].append(v)
        for n in sorted((n for n in nodes if n.kind == "junction" and n.jkind == "transport" and n.comp is not None),
                        key=lambda n: (layer[n.id], str(n.id))):
            k = (name_tokens(m.comps[n.comp]["name"]) if n.comp in m.comps else n.comp, layer[n.id])  # spellings share a lane
            if k in seen:
                # these edges come after cycle breaking: never close a cycle. When n already leads to the transport
                # seen first (chained transports through one membrane), order them the way the flow runs instead.
                extra.append((n, seen[k], 1) if reaches(n, seen[k], out) else (seen[k], n, 1))
                out[extra[-1][0].id].append(extra[-1][1])
            seen[k] = n
        if not extra:
            break
        keep += extra
        layer = longest(keep)
    # sources hang off the step that uses them: a node whose every input is such a source, and that feeds exactly one
    # node, is drawn just before it instead of in column 0
    succ, pred_of = defaultdict(list), defaultdict(list)
    for u, v, g in keep:
        succ[u.id].append((v, g))
        pred_of[v.id].append(u)
    indeg = Counter(v.id for u, v, g in keep)
    topo, queue = [], [n for n in nodes if not indeg[n.id]]
    while queue:
        n = queue.pop(0)
        topo.append(n)
        for v, g in succ[n.id]:
            indeg[v.id] -= 1
            if not indeg[v.id]:
                queue.append(v)
    pulled = {}
    for n in topo:  # a pure source with several users goes just before the first of them
        pulled[n.id] = bool(succ[n.id]) and (not pred_of[n.id] or len(succ[n.id]) == 1 and all(pulled.get(p.id) for p in pred_of[n.id]))
    for n in reversed(topo):
        if pulled[n.id]:
            layer[n.id] = max(layer[n.id], min(layer[v.id] - g for v, g in succ[n.id]))
    for n in nodes:
        n.layer = layer[n.id]
        n.stub_of = None
    # side substrates (a source used only by this step, e.g. a cofactor or a local currency copy) are short stubs beside
    # their step, in its column, on a side its enzymes leave free
    src = {n.id for n in nodes if not pred_of[n.id]}
    for j in nodes:
        if j.kind != "junction" or j.jkind != "reaction":
            continue
        side = [s for s in j.subs if s.id in src and len(succ[s.id]) == 1 and s.kind == "compound"]
        main = [s for s in j.subs if s not in side] or [s for s in side if not getattr(s, "cur", False)][:1] or side[:1]
        slots = {0: 2, 1: 1}.get(len(j.enz), 0)
        j.stubs = [s for s in side if s not in main][:slots]
        for s in j.stubs:
            s.stub_of = j
            s.layer = j.layer
    for _ in range(3):
        for n in nodes:
            if n.sat is not None:
                n.layer = n.sat.layer
    # catalysts sit in the layer of (the first of) their reaction junction
    first = {}
    for e in m.edges:
        if e.kind == "cat":
            first[e.a.id] = min(first.get(e.a.id, 10 ** 9), e.b.layer)
    for e in m.edges:
        if e.kind == "cat":
            e.a.layer = first[e.a.id]
    # transporter pills: layer of their junction
    for n in nodes:
        if n.kind == "junction" and n.jkind == "transport":
            pass
    m.keep = keep


# --------------------------------------------------------------------------- routing (grid A*)
CELL = 10


class Router:
    def __init__(self, W, H):
        self.gw, self.gh = int(W // CELL) + 2, int(H // CELL) + 2
        self.block = bytearray(self.gw * self.gh)
        self.uh = defaultdict(int)
        self.uv = defaultdict(int)

    def add_rect(self, r, val=1):
        x0, y0, x1, y1 = r
        for gx in range(max(0, int(x0 // CELL)), min(self.gw, int(x1 // CELL) + 1)):
            for gy in range(max(0, int(y0 // CELL)), min(self.gh, int(y1 // CELL) + 1)):
                self.block[gy * self.gw + gx] += val

    def route(self, a, b, own):
        """a, b nodes; own = rects (cell ranges) allowed to be crossed."""
        gw, gh = self.gw, self.gh
        sx, sy = int(a.x // CELL), int(a.y // CELL)
        tx, ty = int(b.x // CELL), int(b.y // CELL)
        allowed = set()
        for n in (a, b):
            x0, y0, x1, y1 = n.rect(7)
            for gx in range(int(x0 // CELL), int(x1 // CELL) + 1):
                for gy in range(int(y0 // CELL), int(y1 // CELL) + 1):
                    allowed.add((gx, gy))
        blk = self.block
        start = (sx, sy, -1)
        pq = [(0, 0, start)]
        best = {start: 0}
        prev = {}
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        goal = None
        n_exp = 0
        while pq:
            f, g, s = heapq.heappop(pq)
            if g > best.get(s, 1e18):
                continue
            x, y, d = s
            if (x, y) == (tx, ty):
                goal = s
                break
            n_exp += 1
            if n_exp > 60000:
                break
            for nd, (dx, dy) in enumerate(dirs):
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= gw or ny >= gh:
                    continue
                if blk[ny * gw + nx] and (nx, ny) not in allowed:
                    continue
                c = 1
                if d != -1 and d != nd:
                    c += 5
                if dy == 0 and self.uh.get((nx, ny)):
                    c += 3
                if dx == 0 and self.uv.get((nx, ny)):
                    c += 3
                ng = g + c
                ns = (nx, ny, nd)
                if ng < best.get(ns, 1e18):
                    best[ns] = ng
                    prev[ns] = s
                    h = abs(nx - tx) + abs(ny - ty)
                    heapq.heappush(pq, (ng + h, ng, ns))
        if goal is None:
            return None
        path = [goal]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        path.reverse()
        cells = [(p[0], p[1]) for p in path]
        for i in range(1, len(cells)):
            (x0, y0), (x1, y1) = cells[i - 1], cells[i]
            if y0 == y1:
                self.uh[(x1, y1)] += 1
                self.uh[(x0, y0)] += 1
            else:
                self.uv[(x1, y1)] += 1
                self.uv[(x0, y0)] += 1
        pts = [(c[0] * CELL + CELL / 2, c[1] * CELL + CELL / 2) for c in cells]
        return simplify(pts)


def simplify(pts):
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
            continue
        out.append(b)
    out.append(pts[-1])
    return out


def clip_start(pts, r):
    """Drop the part of the polyline inside rect r at its start."""
    x0, y0, x1, y1 = r
    inside = lambda p: x0 <= p[0] <= x1 and y0 <= p[1] <= y1
    k = 0
    while k < len(pts) and inside(pts[k]):
        k += 1
    if k == 0 or k >= len(pts):
        return pts
    a, b = pts[k - 1], pts[k]
    if a[0] == b[0]:  # vertical
        y = y1 if b[1] > a[1] else y0
        c = (a[0], y)
    else:
        x = x1 if b[0] > a[0] else x0
        c = (x, a[1])
    return [c] + pts[k:]


def clip_end(pts, r):
    return list(reversed(clip_start(list(reversed(pts)), r)))


def path_d(pts, rad=9):
    if len(pts) < 2:
        return ""
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        l1 = math.dist(p0, p1)
        l2 = math.dist(p1, p2)
        r = min(rad, l1 / 2, l2 / 2)
        a = (p1[0] + (p0[0] - p1[0]) * r / l1, p1[1] + (p0[1] - p1[1]) * r / l1)
        b = (p1[0] + (p2[0] - p1[0]) * r / l2, p1[1] + (p2[1] - p1[1]) * r / l2)
        d += f" L{a[0]:.1f},{a[1]:.1f} Q{p1[0]:.1f},{p1[1]:.1f} {b[0]:.1f},{b[1]:.1f}"
    d += f" L{pts[-1][0]:.1f},{pts[-1][1]:.1f}"
    return d


# --------------------------------------------------------------------------- regions / layout
ROWGAP, COLGAP, TIERGAP = 22, 56, 40  # DRAW-RENDERER: more room so arrows run straight (was 16, 40)


class Region:
    def __init__(self, key, label, role, sub="", kind="band"):
        self.key, self.label, self.role, self.sub, self.kind = key, label, role, sub, kind
        self.nodes = []
        self.x = self.y = self.w = self.h = 0
        self.nrows = 1
        self.minrow = 0


def membrane_for(m, ca, cb):
    """Pick a membrane compartment for a transport between compartments ca and cb, or None."""
    A, B = m.comps[ca], m.comps[cb]
    for c in (B, A):
        if c["role"] == "membrane":
            return c["key"]
    groups = {c["group"] for c in (A, B) if c["role"] not in ("extra", "auto")}
    cands = [c for c in m.comps.values() if c["role"] == "membrane" and c["nodes"] is not None
             and (c["group"] in groups or not groups)]
    if cands:
        lo, hi = sorted((A["rank"], B["rank"]))
        btw = [c for c in cands if lo <= c["rank"] <= hi]
        return (btw or cands)[0]["key"]
    return None


def place_transports(m, style):
    """Give every transport junction (and its transporter pills) a home compartment or an interface band."""
    m.ifaces = {}
    for n in list(m.nodes.values()):
        if n.kind != "junction" or n.jkind != "transport":
            continue
        pairs = n.pairs
        if style == "spine":
            n.comp = None
            n.iface = None
            n.pills = [(et, ek, ck) for et, ek, ck in n.trn_ents]
            continue
        ca, cb = pairs[0][1], pairs[0][2]
        home = membrane_for(m, ca, cb)
        n.iface = None
        if home is None:
            # boundary glyph: an interface band (no compartment is invented, it is only a crossing line)
            A, B = m.comps[ca], m.comps[cb]
            if style == "cell":
                g = None
                for c in (B, A):
                    if c["role"] not in ("extra", "auto"):
                        g = c["group"] or "cell"
                        break
                if A["role"] in ("extra", "auto") or B["role"] in ("extra", "auto"):
                    key = ("iface", "cell", g)
                else:
                    key = ("iface", "in", ca if A["role"] == "organelle" else cb)
            else:
                lo = ca if A["rank"] <= B["rank"] else cb
                key = ("iface", lo, ca, cb) if False else ("iface", lo)
            n.iface = key
            home = key
        n.comp = home
        n.pills = [(et, ek, ck) for et, ek, ck in n.trn_ents]
    return


def make_pills(m, ent_lookup):
    pass


def layout(m, style, tiers=1, order="flow"):
    """Returns canvas dict with regions, boxes, size. Mutates node x,y."""
    place_transports(m, style)
    # attach transporter pill nodes
    ent = m._ent
    pill_edges = []
    for n in list(m.nodes.values()):
        if n.kind == "junction" and n.jkind == "transport":
            n.pillnodes = []
            for et, ek, pck in n.pills:
                if ("pill", n.id, et, ek) in m.nodes:  # F1: same transporter listed twice (idx 128 tr_2): draw one pill
                    continue
                e = ent[(et, ek)]
                pn = Node(("pill", n.id, et, ek), KIND[et], e.get("name") or ek, pck if style == "spine" else n.comp, ek)
                pn.attached = n
                pn.et = et
                pn.stated = m.comps[pck]['name'] if pck in m.comps else ''
                m.nodes[pn.id] = pn
                n.pillnodes.append(pn)
                m.edges.append(Edge(pn, n, "cat"))
    assign_layers(m, style)
    # transporter pills share the layer of their junction
    for n in m.nodes.values():
        if n.kind == "junction" and n.jkind == "transport":
            for pn in n.pillnodes:
                pn.layer = n.layer
    # reaction junction compartments / enzymes stay with their reaction compartment
    # collect region membership
    if style == "spine":
        return layout_spine(m, tiers)
    if style == "bands":
        return layout_bands(m, tiers, order)
    return layout_cell(m)


def node_region_key(m, n):
    return n.comp


def compartments_with_content(m):
    used = defaultdict(list)
    for n in m.nodes.values():
        used[n.comp].append(n)
    return used


def order_key(c):
    return (c["rank"], c["group"] or "", c["name"])


def scope_columns(nodes_by_region, x0):
    layers = sorted({n.layer for ns in nodes_by_region for n in ns})
    colw = {}
    for ns in nodes_by_region:
        for n in ns:
            colw[n.layer] = max(colw.get(n.layer, 0), n.w)
    cx = {}
    x = x0
    for L in layers:
        cx[L] = x + colw[L] / 2
        x += colw[L] + COLGAP
    return cx, x - COLGAP - x0 if layers else 0


def assign_rows(nodes, rowrelax=True):
    """Place nodes (all in one region) into integer rows per layer, near their predecessors."""
    occ = set()
    preds = defaultdict(list)
    for e in nodes[0].m_edges if False else []:
        pass


def rows_for(region_nodes, edges_in, scope_nodes_all):
    """Assign .row for nodes of one region."""
    order = sorted(region_nodes, key=lambda n: (n.layer, 0 if n.kind == "junction" else 1 if n.kind == "compound" else 2, str(n.id)))
    occ = set()
    pred = defaultdict(list)
    for a, b in edges_in:
        pred[b.id].append(a)
    jcat = defaultdict(list)
    for a, b in edges_in:
        pass
    rs = set(n.id for n in region_nodes)
    for n in order:
        pref = None
        # catalysts: next to their junction
        att = getattr(n, "attached", None)
        if att is not None and att.row is not None and att.id in rs:
            k = att.pillnodes.index(n)
            off = (k // 2 + 1) * (-1 if k % 2 == 0 else 1)
            pref = att.row + off
        elif n.kind != "junction":
            # catalyst of a junction in this region?
            for j in getattr(n, "cat_of", []):
                if j.row is not None and j.id in rs:
                    k = j.enz.index(n) if hasattr(j, "enz") and n in j.enz else 0
                    off = (k // 2 + 1) * (-1 if k % 2 == 0 else 1)
                    pref = j.row + off
                    break
        if pref is None:
            ps = [p.row for p in pred[n.id] if p.row is not None and p.id in rs and getattr(p, "attached", None) is None
                  and not (p.kind != "junction" and n.kind == "junction" and False)]
            # exclude catalyst predecessors for junction/compound preference
            ps = [p.row for p in pred[n.id] if p.row is not None and p.id in rs and p.id not in getattr(n, "catids", ())]
            pref = int(round(sum(ps) / len(ps))) if ps else 0
        r, step = pref, 0
        while (n.layer, r) in occ:
            step += 1
            r = pref + ((step + 1) // 2) * (1 if step % 2 else -1)
        occ.add((n.layer, r))
        n.row = r


def prep_edges(m):
    for n in m.nodes.values():
        n.cat_of = []
        n.catids = set()
    for e in m.edges:
        if e.kind == "cat":
            e.a.cat_of.append(e.b)
            e.b.catids.add(e.a.id)
    pred = defaultdict(list)
    for e in m.edges:
        if e.kind != "cat":
            pred[e.b.id].append(e.a)
        else:
            pred[e.b.id]  # junction ignores catalysts for preference
    return pred


def region_rows(m, nodes, pred_list, pinned=()):
    """Row assignment for the nodes of one region (rows are integers, collisions resolved per layer)."""
    rs = {n.id: n for n in nodes}
    order = sorted(nodes, key=lambda n: (n.layer, 0 if n.kind == "junction" else 1 if n.kind == "compound" else 2, str(n.id)))
    # I2: a reaction's own enzyme copies take the rows next to it before the next reaction does
    seq, done = [], set()
    for n in order:
        if n.id in done:
            continue
        seq.append(n)
        done.add(n.id)
        for z in getattr(n, "enz", []) if n.kind == "junction" else []:
            if z.id in rs and z.id not in done and len(z.cat_of) == 1:
                seq.append(z)
                done.add(z.id)
        for z in getattr(n, "stubs", []) if n.kind == "junction" else []:
            if z.id in rs and z.id not in done:
                seq.append(z)
                done.add(z.id)
    order = seq
    occ = set()
    for n in nodes:
        if n.id in pinned:
            occ.add((n.layer, n.row))
    sat_count = defaultdict(int)

    def alt(k):
        return (k // 2 + 1) * (-1 if k % 2 == 0 else 1)

    for n in order:
        if n.id in pinned:
            continue
        pref = None
        att = getattr(n, "attached", None)
        sat = getattr(n, "sat", None)
        stub = getattr(n, "stub_of", None)
        if att is not None and att.row is not None and att.id in rs:
            pref = att.row + alt(att.pillnodes.index(n))
        elif stub is not None and stub.row is not None and stub.id in rs:
            taken = {alt(k) for k in range(len(stub.enz))}
            pref = stub.row + [d for d in (1, -1, 2, -2, 3, -3) if d not in taken][stub.stubs.index(n)]
        elif sat is not None and sat.row is not None and sat.id in rs:
            pref = sat.row + alt(sat_count[sat.id])
            sat_count[sat.id] += 1
        else:
            for j in n.cat_of:
                if j.row is not None and j.id in rs and hasattr(j, "enz") and n in j.enz:
                    pref = j.row + alt(j.enz.index(n))
                    break
        if pref is None:
            ps = [p.row for p in pred_list[n.id] if p.row is not None and p.id in rs]
            pref = int(round(sum(ps) / len(ps))) if ps else 0
        r, step = pref, 0
        while (n.layer, r) in occ:
            step += 1
            r = pref + ((step + 1) // 2) * (1 if step % 2 else -1)
        occ.add((n.layer, r))
        n.row = r


def finish_region(reg, rowh, pad_top, pad_bot):
    rows = [n.row for n in reg.nodes]
    reg.minrow = min(rows) if rows else 0
    reg.nrows = (max(rows) - reg.minrow + 1) if rows else 1
    reg.h = reg.nrows * rowh + pad_top + pad_bot
    return reg.h


def layout_bands(m, tiers=1, order="flow"):
    pred = prep_edges(m)
    used = compartments_with_content(m)
    regs = {}
    for ck, ns in used.items():
        if isinstance(ck, tuple) and ck and ck[0] == "iface":
            regs[ck] = Region(ck, "boundary (no membrane compartment recorded)", "iface", kind="iface")
        else:
            c = m.comps[ck]
            sub = f"{c['group']}" if c["group"] else ""
            regs[ck] = Region(ck, c["name"], c["role"], sub)
        regs[ck].nodes = ns
    # order bands
    def okey(r):
        if r.kind == "iface":
            lo = r.key[1]
            c = m.comps[lo]
            return (c["rank"] + 0.05, c["group"] or "", "~")
        c = m.comps[r.key]
        return order_key(c)
    # I7: lanes follow the flow. The lane the main chain mostly runs in comes first, the others in the order the
    # flow first reaches them (layer of their first process); lanes without a process keep the fixed outside-to-inside order.
    chain = main_chain(m)
    m.chain = chain
    flow = {}
    if len(chain) >= 2:
        for ck, ns in used.items():
            ls = [n.layer for n in ns if n.kind == "junction"]
            if ls:
                flow[ck] = min(ls)
        cnt = Counter(j.comp for j in chain)
        first = {}
        for k, j in enumerate(chain):
            first.setdefault(j.comp, k)
        main = max(cnt, key=lambda c: (cnt[c], -first[c]))
    def fkey(r):
        if not flow or order == "anat":  # anat: outside to inside, so a transport crosses into the neighbouring band
            return (0, 0, okey(r))
        return (0 if r.key == main else 1 if r.key in flow else 2, flow.get(r.key, 0) if r.key != main else 0, okey(r))
    # DRAW-RENDERER (PO-approved display grouping): spellings of one compartment share a lane, and the leaflets of one
    # membrane (or near-synonyms such as cytoplasm / intracellular) are lanes of one band. Every verbatim name stays on
    # the drawing and in each node's data-comp; nothing is merged in the data.
    lanes, band_of = {}, {}
    for r in sorted(regs.values(), key=fkey):
        if r.kind == "iface":
            lk = fk = r.key
        else:
            c = m.comps[r.key]
            lk, fk = (name_tokens(c["name"]), c["group"]), (display_family(c["name"], c["role"]), c["group"])
        if lk not in lanes:
            lanes[lk] = Region(lk, r.label, r.role, r.sub, kind=r.kind)
            lanes[lk].names, lanes[lk].ranks = [], []
            band_of[lk] = fk
        L = lanes[lk]
        L.nodes = L.nodes + r.nodes
        L.names.append(r.label)
        L.ranks.append(okey(r))
    groups = {}
    for lk, L in lanes.items():  # bands keep the flow order of their first lane
        groups.setdefault(band_of[lk], []).append(L)
    bands = []
    for fk, ls in groups.items():
        ls.sort(key=lambda L: min(L.ranks))
        bands.append(dict(key=fk, lanes=ls, role=ls[0].role, kind=ls[0].kind))
    for b in bands:
        for L in b["lanes"]:
            region_rows(m, L.nodes, pred)
    rowh = max(n.h for n in m.nodes.values()) + ROWGAP
    GUT = 190
    alln = [n for b in bands for L in b["lanes"] for n in L.nodes]
    tier, span = place_tiers(alln, tiers, GUT + 50)
    W = GUT + 50 + span + 60
    y = 20
    m.bands, m.regions = [], []
    for t in range(tiers):  # a wrapped drawing repeats only the bands that hold something in that tier
        y += TIERGAP if t else 0
        for b0 in bands:
            ls = []
            for L0 in b0["lanes"]:
                ns = [n for n in L0.nodes if tier[n.layer] == t]
                if ns:
                    L = Region(L0.key, L0.label, L0.role, L0.sub, kind=L0.kind)
                    L.names, L.nodes = L0.names, ns
                    ls.append(L)
            if not ls:
                continue
            b = dict(b0, lanes=ls, y=y, tier=t)
            multi, mem = len(ls) > 1, b["role"] == "membrane"
            for k, L in enumerate(ls):
                top = 22 if mem and k == 0 else 14 if not multi else 8
                bot = 22 if mem and k == len(ls) - 1 else 14 if not multi else 8
                finish_region(L, rowh, top, bot)
                L.text = lane_text(L, GUT - 36)
                L.h = max(L.h, (70 if L.kind != "iface" else 56) if not multi else 40, 16 * len(L.text) + top + bot)
                L.x, L.y, L.w = 0, y, W
                for n in L.nodes:
                    n.y = L.y + (L.h - L.nrows * rowh) / 2 + (n.row - L.minrow + 0.5) * rowh
                y += L.h
            b["h"] = y - b["y"]
            m.bands.append(b)
            m.regions += ls
    H = y + 20
    return dict(W=W, H=H, kind="bands", gut=GUT, tiers=tiers, order=order)


def place_tiers(nodes, tiers, x0):
    """Shared columns for ``nodes`` from x0; with tiers > 1 the columns are cut into that many consecutive runs of about
    equal width, stacked top to bottom as a snake (every second run reads right to left, so a run ends above the start
    of the next one, as curated maps wrap long chains). Sets n.x; returns ({layer: tier}, widest run)."""
    colw = {}
    for n in nodes:
        colw[n.layer] = max(colw.get(n.layer, 0), n.w)
    layers = sorted(colw)
    total = sum(colw[L] + COLGAP for L in layers) or 1
    tier, acc = {}, 0.0
    for L in layers:
        tier[L] = min(tiers - 1, int((acc + (colw[L] + COLGAP) / 2) * tiers / total))
        acc += colw[L] + COLGAP
    cx, span = {}, 0.0
    for t in range(tiers):
        x = x0
        mine = [L for L in layers if tier[L] == t]
        for L in mine:
            cx[L] = x + colw[L] / 2
            x += colw[L] + COLGAP
        w = x - COLGAP - x0 if mine else 0
        if t % 2:
            for L in mine:
                cx[L] = x0 + w - (cx[L] - x0)
        span = max(span, w)
    for n in nodes:
        n.x = cx[n.layer]
    return tier, span


LEAF_WORDS = {"leaflet", "periplasmic", "cytoplasmic", "cytosolic", "exoplasmic", "luminal", "lumenal", "extracellular",
              "intracellular", "side", "face"}
STOP_WORDS = {"of", "the", "a", "an"}
SYNONYM_FAMILY = {"cytoplasm": "cytoplasm", "cytosol": "cytoplasm", "intracellular": "cytoplasm", "cytoplasmic": "cytoplasm",
                  "intracellular space": "cytoplasm", "extracellular": "extracellular", "extracellular space": "extracellular",
                  "extracellular region": "extracellular", "extracellular medium": "extracellular", "medium": "extracellular"}


def name_tokens(name):
    """Order-free spelling key: 'periplasmic leaflet of the inner membrane' == 'inner membrane periplasmic leaflet'."""
    return frozenset(w for w in re.findall(r"[a-z0-9]+", (name or "").lower()) if w not in STOP_WORDS)


def display_family(name, role):
    """Which band a compartment is drawn in: a membrane's leaflets join the membrane; known near-synonyms join."""
    n = re.sub(r"\s+", " ", (name or "").strip().lower())
    if n in SYNONYM_FAMILY:
        return ("syn", SYNONYM_FAMILY[n])
    if role == "membrane":
        return ("mem", frozenset(name_tokens(n) - LEAF_WORDS) or name_tokens(n))
    return ("lane", name_tokens(n))


def lane_text(L, maxw):
    """Gutter lines of one lane: its first name, then every other spelling ('= …'), then the cell type."""
    out = [(l, "blabel") for l in wrap(L.names[0], maxw, 3)[0]]
    for nm in L.names[1:]:
        out += [(l, "bsub") for l in wrap("= " + nm, maxw, 3)[0]]
    if L.sub:
        out += [(l, "bsub") for l in wrap(L.sub, maxw, 2)[0]]
    return out


def layout_cell(m):
    pred = prep_edges(m)
    used = compartments_with_content(m)
    regs = {}
    for ck, ns in used.items():
        if ck and ck[0] == "iface":
            kind_ = ck[1]
            label = "cell boundary" if kind_ == "cell" else "organelle boundary"
            regs[ck] = Region(ck, label, "iface", kind="iface")
        else:
            c = m.comps[ck]
            regs[ck] = Region(ck, c["name"], c["role"], c["group"] or "")
        regs[ck].nodes = ns
    for r in regs.values():
        region_rows(m, r.nodes, pred)
    rowh = max(n.h for n in m.nodes.values()) + ROWGAP
    # group into cells
    named = {m.comps[k]["group"] for k in regs if k[0] != "iface" and m.comps[k]["role"] not in ("extra", "auto") and m.comps[k]["group"]}
    def grp(r):
        if r.kind == "iface":
            g0 = r.key[2] if r.key[1] == "cell" else m.comps[r.key[2]]["group"]
            return g0 or (next(iter(named)) if len(named) == 1 else "cell")
        c = m.comps[r.key]
        if c["role"] in ("extra",):
            return "__out"
        if c["role"] == "auto":
            return "__auto"
        g = c["group"]
        if not g:
            g = next(iter(named)) if len(named) == 1 else "cell"
        return g
    top, cells = [], defaultdict(list)
    for r in regs.values():
        g = grp(r)
        if g in ("__out", "__auto"):
            top.append(r)
        else:
            cells[g].append(r)
    # shared columns within each scope
    PADX = 34
    def lay_scope(rs, x0):
        cx, span = scope_columns([r.nodes for r in rs], x0)
        for r in rs:
            for n in r.nodes:
                n.x = cx[n.layer]
        return span
    def rank_of(r):
        if r.kind == "iface":
            return -0.5
        return m.comps[r.key]["rank"]
    LABEL_W = 0
    boxes = []
    # extracellular strip
    top.sort(key=lambda r: (rank_of(r), r.label))
    top_span = lay_scope(top, 40) if top else 0
    cellinfo = []
    joint = False
    for g in sorted(cells, key=lambda g: min(n.layer for r in cells[g] for n in r.nodes) if any(r.nodes for r in cells[g]) else 0):
        rs = cells[g]
        stack = sorted([r for r in rs if r.role != "organelle"], key=lambda r: (rank_of(r), r.label))
        orgs = sorted([r for r in rs if r.role == "organelle"], key=lambda r: r.label)
        joint = len(cells) == 1 and bool(top)  # one cell: the outside strip shares its columns, so transports run straight
        span = lay_scope(stack + (top if joint else []), 0) if stack or joint else 0
        # organelles: own scope each
        info = dict(name=g, stack=stack, orgs=orgs, span=span)
        cellinfo.append(info)
    # vertical sizes
    y = 24
    for r in top:
        finish_region(r, rowh, 26, 14)
        r.h = max(r.h, 70)
        r.y = y
        for n in r.nodes:
            n.y = r.y + (r.h - r.nrows * rowh) / 2 + (n.row - r.minrow + 0.5) * rowh
        y += r.h + 6
    y_cells = y + 26
    # horizontal arrangement of cells
    xcur = 30
    maxh_cells = 0
    cell_boxes = []
    for info in cellinfo:
        stack, orgs = info["stack"], info["orgs"]
        # organelle widths
        org_w = []
        for o in orgs:
            span = lay_scope([o], 0)
            org_w.append(span + 2 * 30)
        inner_w = max([info["span"] + 2 * PADX] + [sum(org_w) + 24 * (len(orgs) - 1) + 2 * 20 if orgs else 0])
        cw = inner_w
        # shift stack columns
        ox = xcur + (cw - info["span"]) / 2
        for r in stack + (top if joint else []):
            for n in r.nodes:
                n.x += ox
        yy = y_cells + 46
        for r in stack:
            finish_region(r, rowh, 28 if r.role == "membrane" else 16, 28 if r.role == "membrane" else 16)
            r.h = max(r.h, 62 if r.kind != "iface" else 50)
            r.x, r.w = xcur, cw
            r.y = yy
            for n in r.nodes:
                n.y = r.y + (r.h - r.nrows * rowh) / 2 + (n.row - r.minrow + 0.5) * rowh
            yy += r.h
        if orgs:
            oy = yy + 22
            oh = 0
            xo = xcur + (cw - (sum(org_w) + 24 * (len(orgs) - 1))) / 2
            for o, wv in zip(orgs, org_w):
                finish_region(o, rowh, 40, 22)
                o.h = max(o.h, 100)
                oh = max(oh, o.h)
            for o, wv in zip(orgs, org_w):
                for n in o.nodes:
                    n.x += xo + 30
                o.x, o.w, o.y = xo, wv, oy
                o.h = oh
                for n in o.nodes:
                    n.y = o.y + 22 + (o.h - 22 - 18 - o.nrows * rowh) / 2 + (n.row - o.minrow + 0.5) * rowh
                xo += wv + 24
            yy = oy + oh + 22
        cell_boxes.append(dict(x=xcur, y=y_cells, w=cw, h=yy - y_cells + 8, name=info["name"]))
        xcur += cw + 36
        maxh_cells = max(maxh_cells, yy - y_cells + 8)
    # extracellular strip regions widen to canvas
    W = max(xcur + 10, 40 + (0 if joint else top_span) + 60, 400)
    for r in top:
        r.x, r.w = 0, W
    H = y_cells + maxh_cells + 30 if cellinfo else y + 20
    # nodes are placed relative to scope origin 0 for stack; shift extracellular strip nodes already at x0=40
    m.regions = top + [r for info in cellinfo for r in info["stack"]] + [o for info in cellinfo for o in info["orgs"]]
    m.cell_boxes = cell_boxes
    return dict(W=W, H=H, kind="cell")


# ---- spine
def layout_spine(m, tiers=1):
    pred = prep_edges(m)
    nodes = list(m.nodes.values())
    J = [n for n in nodes if n.kind == "junction"]
    chain = main_chain(m)
    m.chain = chain
    # pin the main chain on row 0: its junctions and the compounds that link consecutive steps
    for n in nodes:
        n.row = None
    pin = set()
    for j in chain:
        j.row = 0
        pin.add(j.id)
    outs = defaultdict(list)
    ins = defaultdict(list)
    for e in m.edges:
        if e.kind in ("prod", "xout"):
            outs[e.a.id].append(e.b)
        if e.kind in ("sub", "xin"):
            ins[e.b.id].append(e.a)
    for k, j in enumerate(chain):
        if k + 1 < len(chain):
            link = [p for p in outs[j.id] if any(q is chain[k + 1] for q in [e.b for e in m.edges if e.kind in ("sub", "xin") and e.a is p])]
            for p in link[:1]:
                p.row = 0
                pin.add(p.id)
    if chain:
        for n in ([x for x in ins[chain[0].id] if getattr(x, "stub_of", None) is None][:1] + outs[chain[-1].id][:1]):
            n.row = 0
            pin.add(n.id)
    # no repeated captions: the compartment most molecules are in is named once, in the main-chain label; only
    # molecules elsewhere carry a caption
    cnt = Counter(n.comp for n in nodes if n.kind != "junction" and n.comp in m.comps)
    m.main_comp = max(cnt, key=lambda c: (cnt[c], str(c))) if cnt else None
    for n in nodes:
        n.chip = m.comps[n.comp]["name"] if (n.kind != "junction" and n.comp in m.comps and n.comp != m.main_comp) else None
        if n.chip:
            n.h += 14
            n.w = max(n.w, tw(n.chip, True) + 10)
    region_rows(m, nodes, pred, pinned=pin)
    rowh = max(n.h for n in nodes) + ROWGAP
    tier, span = place_tiers(nodes, tiers, 100)
    y, spine_rows = 46, []
    for t in range(tiers):
        ns = [n for n in nodes if tier[n.layer] == t]
        if not ns:
            continue
        y += TIERGAP if spine_rows else 0
        lo, hi = min(min(n.row for n in ns), 0), max(max(n.row for n in ns), 0)
        for n in ns:
            n.y = y + (n.row - lo + 0.5) * rowh
        if any(j in ns for j in chain):
            spine_rows.append(y + (0 - lo + 0.5) * rowh)
        y += (hi - lo + 1) * rowh
    W = 100 + span + 70
    H = y + 30
    m.regions = []
    return dict(W=W, H=H, kind="spine", rowh=rowh, spine_rows=spine_rows, tiers=tiers,
                main_comp=m.comps[m.main_comp]["name"] if m.main_comp else "")


def main_chain(m):
    """Longest chain of processes (junctions) linked by a shared molecule instance. Data only."""
    J = [n for n in m.nodes.values() if n.kind == "junction"]
    outs = defaultdict(list)
    ins = defaultdict(list)
    for e in m.edges:
        if e.kind in ("prod", "xout"):
            outs[e.a.id].append(e.b.id)
        if e.kind in ("sub", "xin"):
            ins[e.a.id].append(e.b)
    byid = {j.id: j for j in J}
    nxt = defaultdict(list)
    for j in J:
        for p in outs[j.id]:
            for k in ins[p]:
                if k.layer > j.layer:
                    nxt[j.id].append(k)
    memo = {}

    def lp(j):
        if j.id in memo:
            return memo[j.id]
        best = (1, None)
        for k in nxt[j.id]:
            c = lp(k)[0] + 1
            if c > best[0]:
                best = (c, k)
        memo[j.id] = best
        return best

    if not J:
        return []
    start = max(J, key=lambda j: (lp(j)[0], -j.layer))
    chain = [start]
    while memo[chain[-1].id][1] is not None:
        chain.append(memo[chain[-1].id][1])
    return chain


# --------------------------------------------------------------------------- drawing
def draw_view(m, lay, vcls):
    """Route and draw one model. Returns an SVG <g class=vcls> string (canvas, regions, edges, nodes, legend)."""
    W, H = lay["W"], lay["H"]
    style = lay["kind"]
    nodes = list(m.nodes.values())
    for n in nodes:  # snap centres to the routing grid so edges meet nodes orthogonally
        n.x = math.floor(n.x / CELL) * CELL + CELL / 2
        n.y = math.floor(n.y / CELL) * CELL + CELL / 2
    R = Router(W, H)
    obstacles = []
    for n in nodes:
        R.add_rect(n.rect(4))
        obstacles.append(n)
    # region labels (cell/bands): reserve
    labels = []
    if style == "bands":
        pass
    else:
        for r in getattr(m, "regions", []):
            lw = tw(r.label, True) + 16
            labels.append((r.x + 14, r.y + 4, r.x + 14 + lw, r.y + (36 if r.role == "membrane" else 26)))
        for lb in labels:
            R.add_rect(lb)
    # route short edges first
    order = sorted(m.edges, key=lambda e: abs(e.a.x - e.b.x) + abs(e.a.y - e.b.y))
    nfail = 0
    for e in order:
        if e.a is e.b:  # an interaction of a molecule with itself: a small loop over the node
            n = e.a
            x0, y0 = n.x, n.y - n.h / 2
            e.pts = [(x0 + n.w / 4, y0), (x0 + n.w / 4, y0 - 16), (x0 - n.w / 4, y0 - 16), (x0 - n.w / 4, y0)]
            continue
        p = R.route(e.a, e.b, None)
        if p is None:
            nfail += 1
            p = [(e.a.x, e.a.y), (e.a.x, e.b.y), (e.b.x, e.b.y)] if e.a.x != e.b.x and e.a.y != e.b.y else [(e.a.x, e.a.y), (e.b.x, e.b.y)]
        p = [(e.a.x, e.a.y)] + p[1:-1] + [(e.b.x, e.b.y)] if len(p) >= 2 else p
        p = simplify(p)
        p = clip_start(p, e.a.rect(1))
        p = clip_end(p, e.b.rect(3 if e.kind in ("prod", "xout", "act", "inh", "assoc", "enz") else 1))
        e.pts = p
    m.route_fail = nfail
    # which steps each molecule drawing belongs to (hover highlight)
    pids = defaultdict(list)
    for e in m.edges:
        if e.kind in PROC_KINDS:
            jn, mol = (e.b, e.a) if e.b.kind == "junction" else (e.a, e.b)
            if jn.label not in pids[id(mol)]:
                pids[id(mol)].append(jn.label)
    for n in nodes:
        n.pids = " ".join(pids.get(id(n), []))
        if style == "spine" and n.kind == "junction" and n.jkind == "transport":
            n.cross = cross_caption(m, n)
    out = [f'<g class="{vcls}">']
    out.append(f'<rect class="canvas" x="0" y="0" width="{W:.0f}" height="{H:.0f}"/>')
    # regions
    if style == "bands":
        gut = lay["gut"]

        def labels(L, y0, h):
            ty = y0 + h / 2 - (len(L.text) - 1) * 8 + 4
            return "".join(f'<text class="{c}" x="18" y="{ty + i * 16:.1f}">{escape(l)}</text>' for i, (l, c) in enumerate(L.text))

        for b in m.bands:
            ls, y0, h = b["lanes"], b["y"], b["h"]
            mt = membrane_type(" ".join(n for L in ls for n in L.names))
            out.append(f'<g class="band {b["role"]}{" " + mt if mt else ""}" data-comp="{escape(ls[0].names[0])}">'
                       f'<rect class="bg" x="0" y="{y0:.1f}" width="{W:.1f}" height="{h:.1f}"/>')
            if b["role"] == "membrane":
                out.append(f'<rect x="0" y="{y0:.1f}" width="{W:.1f}" height="14" fill="url(#lipidTop)"/>')
                out.append(f'<rect x="0" y="{y0 + h - 14:.1f}" width="{W:.1f}" height="14" fill="url(#lipidBot)"/>')
            elif mt in DOUBLE:  # nucleus, mitochondrion, plastid: an envelope of two membranes
                for yy in (y0 + 2, y0 + 6, y0 + h - 6, y0 + h - 2):
                    out.append(f'<line class="mline2" x1="0" x2="{W:.1f}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
            elif mt:
                for yy in (y0 + 2, y0 + h - 2):
                    out.append(f'<line class="mline1" x1="0" x2="{W:.1f}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
            out.append(f'<line class="sep" x1="0" x2="{W:.1f}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
            if len(ls) == 1:
                out.append(labels(ls[0], y0, h))
            out.append("</g>")
            if len(ls) > 1:
                for k, L in enumerate(ls):
                    out.append(f'<g class="lane" data-comp="{escape(L.names[0])}"><rect class="bg" x="0" y="{L.y:.1f}" width="{W:.1f}" height="{L.h:.1f}"/>'
                               + (f'<line class="lanesep" x1="{gut}" x2="{W:.1f}" y1="{L.y:.1f}" y2="{L.y:.1f}"/>' if k else "")
                               + labels(L, L.y, L.h) + "</g>")
        out.append(f'<line class="gutline" x1="{gut}" x2="{gut}" y1="0" y2="{lay.get("Hc", H):.0f}"/>')
    elif style == "cell":
        for r in [x for x in m.regions if x.role in ("extra", "auto")]:
            out.append(f'<g class="band {r.role}"><rect class="bg" x="0" y="{r.y:.1f}" width="{r.w:.1f}" height="{r.h:.1f}" rx="0"/></g>')
        for cb in m.cell_boxes:
            out.append(f'<g class="cellbox"><rect class="cellfill" x="{cb["x"]:.1f}" y="{cb["y"]:.1f}" width="{cb["w"]:.1f}" height="{cb["h"]:.1f}" rx="34"/></g>')
        for r in m.regions:
            if r.role in ("extra", "auto"):
                continue
            if r.role == "organelle":
                mt = membrane_type(r.label)
                out.append(f'<rect class="orgblob {mt}" x="{r.x:.1f}" y="{r.y:.1f}" width="{r.w:.1f}" height="{r.h:.1f}" rx="26"/>')
                if mt in DOUBLE:
                    out.append(f'<rect class="orgline2 {mt}" x="{r.x + 4:.1f}" y="{r.y + 4:.1f}" width="{r.w - 8:.1f}" height="{r.h - 8:.1f}" rx="22"/>')
            elif r.role == "membrane":
                out.append(f'<g class="band membrane {membrane_type(r.label)}"><rect class="bg" x="{r.x:.1f}" y="{r.y:.1f}" width="{r.w:.1f}" height="{r.h:.1f}"/>'
                           f'<rect x="{r.x:.1f}" y="{r.y:.1f}" width="{r.w:.1f}" height="14" fill="url(#lipidTop)"/>'
                           f'<rect x="{r.x:.1f}" y="{r.y + r.h - 14:.1f}" width="{r.w:.1f}" height="14" fill="url(#lipidBot)"/></g>')
            elif r.role == "iface":
                out.append(f'<line class="ifaceline" x1="{r.x + 20:.1f}" x2="{r.x + r.w - 20:.1f}" y1="{r.y + r.h / 2:.1f}" y2="{r.y + r.h / 2:.1f}"/>')
            else:
                out.append(f'<g class="band {r.role}"><rect class="bg" x="{r.x + 10:.1f}" y="{r.y:.1f}" width="{r.w - 20:.1f}" height="{r.h:.1f}" rx="10"/></g>')
        for cb in m.cell_boxes:
            out.append(f'<rect class="cellline" x="{cb["x"]:.1f}" y="{cb["y"]:.1f}" width="{cb["w"]:.1f}" height="{cb["h"]:.1f}" rx="34"/>')
            out.append(f'<text class="celltitle" x="{cb["x"] + 34:.1f}" y="{cb["y"] + 30:.1f}">cell \u00b7 {escape(cb["name"] if cb["name"] != "cell" else "cell type not stated")}</text>')
        for r in m.regions:
            if r.kind == "iface":
                continue
            lab = r.label
            out.append(f'<text class="rlabel {r.role}" x="{r.x + 16:.1f}" y="{r.y + (17 if r.role not in ("organelle", "membrane") else 21 if r.role == "organelle" else 31):.1f}">{escape(lab)}</text>')
            if r.sub and r.role in ("extra",):
                pass
        for r in m.regions:
            if r.kind == "iface":
                out.append(f'<text class="rlabel ifacelbl" x="{r.x + 22:.1f}" y="{r.y + r.h / 2 - 5:.1f}">{escape(r.label)}</text>')
    else:
        ch = getattr(m, "chain", [])
        for k, sy in enumerate(lay["spine_rows"] if ch else []):
            out.append(f'<rect class="spine" x="0" y="{sy - lay["rowh"] / 2:.1f}" width="{W:.1f}" height="{lay["rowh"]:.1f}"/>')
            if k == 0:
                out.append(f'<text class="spinelbl" x="10" y="{sy - lay["rowh"] / 2 + 14:.1f}">main chain</text>')
        if lay.get("main_comp"):  # named once here instead of under every molecule
            out.append(f'<text class="spinecomp" x="10" y="20">all in {escape(lay["main_comp"])} unless labelled</text>')
    # edges
    out.append('<g class="edges">')
    for e in m.edges:
        if not e.pts or len(e.pts) < 2:
            continue
        cls = f"e {e.kind}" + (" wh" if e.withheld else "")
        mk = ""
        if e.kind in ("prod", "xout"):
            mk = ' marker-end="url(#arr)"'
        elif e.kind == "sub" and e.rev:
            mk = ' marker-start="url(#arrs)"'
        elif e.kind == "act":
            mk = ' marker-end="url(#act)"'
        elif e.kind == "inh":
            mk = ' marker-end="url(#inh)"'
        elif e.kind in ("cat", "enz"):
            mk = ' marker-end="url(#cat)"'
        elif e.kind == "assoc" and getattr(e, "directed", False):
            cls += " dir"
            mk = ' marker-end="url(#arrd)"'
        elif e.kind == "assoc":
            mk = ' marker-end="url(#dot)" marker-start="url(#dot)"'
        tip = f"<title>{escape(e.tip)}</title>" if e.tip else ""
        if e.kind in PROC_KINDS:
            jn, mol = (e.b, e.a) if e.b.kind == "junction" else (e.a, e.b)
            da = f' data-p="{escape(jn.label)}" data-ek="{e.kind}" data-e="{mol.et}:{mol.ent}"'
        else:
            da = f' data-ik="{escape(getattr(e, "key", ""))}" data-ea="{e.a.et}:{e.a.ent}" data-eb="{e.b.et}:{e.b.ent}"'
        out.append(f'<path class="{cls}"{da} d="{path_d(e.pts)}"{mk}>{tip}</path>')
        if e.badge:
            segs = list(zip(e.pts[:-1], e.pts[1:]))
            (p0, p1) = max(segs, key=lambda sg: math.dist(*sg))
            mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            if p0[1] == p1[1]:
                mid = (mid[0], mid[1] - 13)
            else:
                mid = (mid[0] + 22, mid[1])
            bw = tw(e.badge, True) + 10
            out.append(f'<g class="badge"><rect x="{mid[0] - bw / 2:.1f}" y="{mid[1] - 8:.1f}" width="{bw:.1f}" height="16" rx="8"/><text x="{mid[0]:.1f}" y="{mid[1] + 3.5:.1f}" text-anchor="middle">{escape(e.badge)}</text></g>')
    out.append("</g>")
    out.append(hops_svg(m, lay))
    # nodes
    out.append('<g class="nodes">')
    for n in nodes:
        out.append(node_svg(n, style, m))
    out.append("</g>")
    out.append(legend_svg(m, lay))
    out.append("</g>")
    return "\n".join(out)


PROC_KINDS = ("sub", "prod", "xin", "xout", "cat")
HOP = 5


def _bg_at(m, lay, x, y):
    """CSS class of the background under (x, y), so a line bridge's gap is painted in the colour it sits on."""
    if lay["kind"] == "bands":
        for b in m.bands:
            if b["y"] <= y <= b["y"] + b["h"]:
                return f'g-{b["role"]} ' + membrane_type(" ".join(n for L in b["lanes"] for n in L.names))
        return ""
    if lay["kind"] == "spine":
        return "g-spine" if any(abs(y - sy) <= lay["rowh"] / 2 for sy in lay["spine_rows"]) else ""
    best = None
    for r in m.regions:
        if r.role != "iface" and r.x <= x <= r.x + r.w and r.y <= y <= r.y + r.h and (best is None or r.w * r.h < best.w * best.h):
            best = r
    if best is not None:
        return f"g-{best.role} " + (membrane_type(best.label) if best.role in ("organelle", "membrane") else "")
    return "g-cell" if any(c["x"] <= x <= c["x"] + c["w"] and c["y"] <= y <= c["y"] + c["h"] for c in m.cell_boxes) else ""


def hops_svg(m, lay):
    """'Speed bumps': where a horizontal line still crosses a vertical one, the horizontal line bridges over it.
    Drawn as an overlay; the edge paths (and every metric read from them) are unchanged."""
    H, V = [], []
    for e in m.edges:
        for (x0, y0), (x1, y1) in zip(e.pts or [], (e.pts or [])[1:]):
            if abs(y0 - y1) < 0.5 and abs(x0 - x1) > 0.5:
                H.append((min(x0, x1), max(x0, x1), y0, e))
            elif abs(x0 - x1) < 0.5 and abs(y0 - y1) > 0.5:
                V.append((min(y0, y1), max(y0, y1), x0, e))
    out = []
    for hx0, hx1, hy, he in H:
        for vy0, vy1, vx, ve in V:
            if he is not ve and hx0 + HOP + 1 < vx < hx1 - HOP - 1 and vy0 + HOP < hy < vy1 - HOP:
                bg = _bg_at(m, lay, vx, hy)
                out.append(f'<path class="hopgap {bg}" d="M{vx - HOP - 1:.1f},{hy:.1f} L{vx + HOP + 1:.1f},{hy:.1f}"/>'
                           f'<path class="hopv k-{ve.kind}" d="M{vx:.1f},{hy - 4:.1f} L{vx:.1f},{hy + 4:.1f}"/>'
                           f'<path class="hop k-{he.kind}" d="M{vx - HOP:.1f},{hy:.1f} A{HOP},{HOP} 0 0 1 {vx + HOP:.1f},{hy:.1f}"/>')
    return '<g class="hops">' + "".join(out) + "</g>"

# membrane / compartment type -> CSS class (colour); DOUBLE = envelopes of two membranes. Nucleus words only: the
# bacterial nucleoid and the nucleolus have no membrane.
MTYPES = (("mt-er", ("endoplasmic", "reticulum", " er ", "sarcoplasmic")), ("mt-golgi", ("golgi",)),
          ("mt-mito", ("mitochond",)), ("mt-nuc", ("nucleus", "nuclear", "nucleoplasm")), ("mt-plastid", ("chloroplast", "plastid", "thylakoid", "stroma")),
          ("mt-vac", ("vacuol", "tonoplast")), ("mt-lyso", ("lysosom", "endosom")), ("mt-perox", ("peroxisom", "glyoxysom")),
          ("mt-outer", ("outer membrane",)), ("mt-plasma", ("plasma membrane", "cell membrane", "apical", "basolateral")))
DOUBLE = ("mt-mito", "mt-nuc", "mt-plastid")


def membrane_type(names):
    n = " " + re.sub(r"[^a-z]+", " ", (names or "").lower()) + " "
    if "thylakoid" in n:  # the thylakoid is a single membrane inside the plastid
        return "mt-plastid1"
    return next((c for c, keys in MTYPES if any(k in n for k in keys)), "")


def cross_caption(m, j):
    """I9: spine style draws no compartments, so a transport says which two it joins."""
    seen = []
    for el, a, b in j.pairs:
        t = (m.comps[a]["name"], m.comps[b]["name"])
        if t not in seen:
            seen.append(t)
    if not seen:
        return ""
    a, b = seen[0]
    cut = lambda x: x if len(x) <= 18 else x[:17] + "\u2026"
    return f"{cut(a)} \u2192 {cut(b)}" + (f" +{len(seen) - 1}" if len(seen) > 1 else "")


ROLE_LABEL = {}


def node_svg(n, style, m=None):
    x, y, w, h = n.x, n.y, n.w, n.h
    chip = getattr(n, "chip", None)
    body_h = h - (14 if chip else 0)
    by = y - h / 2
    if n.kind == "junction":
        cls = "jn " + n.jkind
        merged = getattr(n, "merged", None) or [n.proc]
        keys = " ".join(r["key"] for r in merged)
        tipt = getattr(n, "tip", "")
        if n.jkind == "reaction" and len(merged) > 1:
            names = list(dict.fromkeys(z.label for z in n.enz))
            tipt += f"  |  {len(merged)} identical steps in the data ({', '.join(n.mtags)}); enzymes: " + (", ".join(names) or "none stated")
        tip = f"<title>{escape(tipt)}</title>"
        da = f' data-p="{escape(n.label)}" data-k="{escape(keys)}"'
        if n.jkind == "transport":
            s = f'<g class="{cls}"{da}>{tip}<rect x="{x - w / 2:.1f}" y="{y - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" rx="10"/>' \
                f'<rect class="inner" x="{x - w / 2 + 3:.1f}" y="{y - h / 2 + 3:.1f}" width="{w - 6:.1f}" height="{h - 6:.1f}" rx="7"/>'
        else:
            s = f'<g class="{cls}"{da}>{tip}<rect x="{x - w / 2:.1f}" y="{y - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" rx="10"/>'
        s += f'<text x="{x:.1f}" y="{y + 3.6:.1f}" text-anchor="middle">{escape(n.label)}</text>'
        py = y + h / 2 + 1
        for bt, bc in ((n.merge_note, " merged"), (n.note, "")):
            if bt:
                s += f'<g class="badge{bc}"><rect x="{x - w / 2:.1f}" y="{py:.1f}" width="{tw(bt, True) + 10:.1f}" height="14" rx="7"/>' \
                     f'<text x="{x - w / 2 + 5:.1f}" y="{py + 10.5:.1f}">{escape(bt)}</text></g>'
                py += 15
        cross = getattr(n, "cross", "")
        if cross:
            s += f'<text class="xcap" x="{x:.1f}" y="{py + 9:.1f}" text-anchor="middle">{escape(cross)}</text>'
        return s + "</g>"
    tip = f"<title>{escape(n.full)}{(' [' + n.cls + ']') if n.cls else ''}</title>"
    cx0 = y - h / 2 + body_h / 2
    shape = ""
    if n.kind == "compound":
        cl = "n compound" + (" small" if n.cls in ("ion", "cofactor") else "") + (" cur" if getattr(n, "cur", False) else "")
        shape = f'<rect x="{x - w / 2:.1f}" y="{by:.1f}" width="{w:.1f}" height="{body_h:.1f}" rx="{body_h / 2:.1f}"/>'
    elif n.kind == "protein":
        cl = "n protein"
        shape = f'<rect x="{x - w / 2:.1f}" y="{by:.1f}" width="{w:.1f}" height="{body_h:.1f}" rx="3"/>'
    elif n.kind == "complex":
        cl = "n complex"
        k = min(12, body_h / 2)
        x0, x1, y0, y1 = x - w / 2, x + w / 2, by, by + body_h
        pts = f"{x0 + k:.1f},{y0:.1f} {x1 - k:.1f},{y0:.1f} {x1:.1f},{(y0 + y1) / 2:.1f} {x1 - k:.1f},{y1:.1f} {x0 + k:.1f},{y1:.1f} {x0:.1f},{(y0 + y1) / 2:.1f}"
        shape = f'<polygon points="{pts}"/>'
    else:
        cl = "n nucleic"
        k = 10
        x0, x1, y0, y1 = x - w / 2, x + w / 2, by, by + body_h
        pts = f"{x0 + k:.1f},{y0:.1f} {x1:.1f},{y0:.1f} {x1 - k:.1f},{y1:.1f} {x0:.1f},{y1:.1f}"
        shape = f'<polygon points="{pts}"/>'
    cname = getattr(n, "stated", None) or (m.comps[n.comp]["name"] if (m is not None and n.comp in m.comps) else "")
    s = f'<g class="{cl}" data-ent="{n.et}:{n.ent}" data-comp="{escape(cname)}" data-p="{n.pids}">{tip}{shape}'
    ly = cx0 - (len(n.lines) - 1) * 7 + 4
    for i, l in enumerate(n.lines):
        s += f'<text x="{x:.1f}" y="{ly + i * 14:.1f}" text-anchor="middle">{escape(l)}</text>'
    if chip:
        s += f'<text class="chip" x="{x:.1f}" y="{y + h / 2 - 2:.1f}" text-anchor="middle">{escape(chip if tw(chip, True) < w + 20 else chip[:int(w / 5)] + chr(8230))}</text>'
    return s + "</g>"


DEFS = """<defs>
<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path class="mk-fill" d="M0,0 L10,5 L0,10 z"/></marker>
<marker id="arrs" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path class="mk-fill" d="M10,0 L0,5 L10,10 z"/></marker>
<marker id="act" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="10" markerHeight="10" orient="auto"><path class="mk-act" d="M1,1.5 L11,6 L1,10.5 z"/></marker>
<marker id="inh" viewBox="0 0 12 12" refX="8" refY="6" markerWidth="10" markerHeight="10" orient="auto"><path class="mk-inh" d="M8,0.5 L8,11.5"/></marker>
<marker id="cat" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><circle class="mk-cat" cx="5" cy="5" r="3.2"/></marker>
<marker id="arrd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path class="mk-dir" d="M0,0 L10,5 L0,10 z"/></marker>
<marker id="dot" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="5" markerHeight="5"><circle class="mk-dot" cx="4" cy="4" r="3"/></marker>
<pattern id="lipidTop" width="12" height="14" patternUnits="userSpaceOnUse"><circle class="lipid-h" cx="6" cy="4" r="3"/><path class="lipid-t" d="M4.6,7 L4.6,13 M7.4,7 L7.4,13"/></pattern>
<pattern id="lipidBot" width="12" height="14" patternUnits="userSpaceOnUse"><circle class="lipid-h" cx="6" cy="10" r="3"/><path class="lipid-t" d="M4.6,1 L4.6,7 M7.4,1 L7.4,7"/></pattern>
</defs>"""


# --------------------------------------------------------------------------- checks + stats
def checks(m):
    ns = list(m.nodes.values())
    ov = 0
    for i in range(len(ns)):
        a = ns[i].rect()
        for j in range(i + 1, len(ns)):
            b = ns[j].rect()
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                ov += 1
    thru = 0
    for e in m.edges:
        if not e.pts:
            continue
        for k in range(len(e.pts) - 1):
            (x0, y0), (x1, y1) = e.pts[k], e.pts[k + 1]
            for n in ns:
                if n is e.a or n is e.b:
                    continue
                r = n.rect(-1)
                if max(x0, x1) > r[0] and min(x0, x1) < r[2] and max(y0, y1) > r[1] and min(y0, y1) < r[3]:
                    thru += 1
                    break
    return dict(node_overlaps=ov, edge_through_node=thru, route_fail=m.route_fail,
                truncated=sum(1 for n in ns if getattr(n, "truncated", False)))


def metrics(ir):
    """Numbers used by the auto-pick rule, from the data alone."""
    P = ir["processes"]
    m = build(ir)
    used = {n.comp for n in m.nodes.values() if n.kind != "junction"} | {ck for p in m.procs if p["kind"] == "transport" for _, a, b in p["node"].pairs for ck in (a, b)}
    placed = [ck for ck in used if m.comps[ck]["role"] != "auto"]
    # longest chain of processes (substrate -> product links), data only
    prods = defaultdict(list)
    for e in m.edges:
        pass
    J = [n for n in m.nodes.values() if n.kind == "junction"]
    out_ = defaultdict(list)
    for e in m.edges:
        if e.kind in ("prod", "xout"):
            out_[e.a.id].append(e.b)
    nxt = defaultdict(set)
    ins = defaultdict(list)
    for e in m.edges:
        if e.kind in ("sub", "xin"):
            ins[e.a.id].append(e.b.id)
    for j in J:
        for p in out_[j.id]:
            for k in ins[p.id]:
                nxt[j.id].add(k)
    memo = {}

    def lp(j, stack=()):
        if j in memo:
            return memo[j]
        if j in stack:
            return 1
        b = 1
        for k in nxt[j]:
            b = max(b, 1 + lp(k, stack + (j,)))
        memo[j] = b
        return b

    chain = max([lp(j.id) for j in J], default=0)
    reg = sum(1 for e in m.edges if e.kind in ("act", "inh"))
    n_ent = sum(1 for n in m.nodes.values() if n.kind != "junction")
    return dict(n_transport=len(P.get("transports", [])), n_reaction=len(P.get("reactions", [])),
                n_interaction=len(P.get("interactions", [])), n_regulation=reg, n_compartments=len(placed),
                chain=chain, n_nodes=n_ent)


def pick_style(k):
    """Auto-pick rule: computed from counts only."""
    if k["chain"] >= 4 and k["n_compartments"] <= 3:
        return "spine", f"longest substrate\u2192product chain is {k['chain']} steps in only {k['n_compartments']} compartment(s): one main chain carries the story"
    if k["n_compartments"] >= 3 and k["n_transport"] >= 1 and k["n_nodes"] <= 30:
        return "cell", f"{k['n_compartments']} compartments and {k['n_transport']} transport(s) with {k['n_nodes']} molecule drawings: small enough that a drawn cell stays readable"
    return "bands", f"{k['n_compartments']} compartments, {k['n_nodes']} molecule drawings, chain of {k['chain']}: too large or too flat for a cell drawing, so bands keep compartments legible"


REG_KINDS = ("act", "inh", "assoc", "enz")


def hide_regulation(m):
    """I1: move regulation/binding links, and molecules that exist only for them, out of the default view.
    Nothing is deleted from the data: the keys and counts are returned so the page can say what is hidden."""
    hid_e = [e for e in m.edges if e.kind in REG_KINDS]
    m.edges = [e for e in m.edges if e.kind not in REG_KINDS]
    used = {id(x) for e in m.edges for x in (e.a, e.b)}
    reg_nodes = {id(x) for e in hid_e for x in (e.a, e.b)}
    hid_n = [n for n in m.nodes.values() if n.kind != "junction" and id(n) not in used and id(n) in reg_nodes]
    for n in hid_n:
        del m.nodes[n.id]
        m.comps[n.comp]["nodes"].remove(n)
    shown = {(n.et, n.ent) for n in m.nodes.values() if n.kind != "junction"}
    only = {(n.et, n.ent) for n in hid_n} - shown
    return dict(edges=len(hid_e), keys=[e.key for e in hid_e], nodes=len(hid_n), molecules=len(only),
                act=sum(e.kind == "act" for e in hid_e), inh=sum(e.kind == "inh" for e in hid_e),
                other=sum(e.kind in ("assoc", "enz") for e in hid_e),
                exported=sum(not e.withheld for e in hid_e), withheld=sum(e.withheld for e in hid_e))


# ---------------------------------------------------------------- legend (I4) and regulation toggle (I1)
LEG_ROW = 24
SYM_W = 32

TOGGLE_JS = ("var s=this.ownerSVGElement,f=s.classList.toggle('show-reg'),d=s.getAttribute(f?'data-full':'data-core').split(' ');"
             "s.setAttribute('viewBox','0 0 '+d[0]+' '+d[1]);s.setAttribute('width',d[0]);s.setAttribute('height',d[1])")
TOGGLE_KEY_JS = "if(event.key==='Enter'||event.key===' '){event.preventDefault();this.dispatchEvent(new MouseEvent('click'))}"
HOVER_JS = ("var s=this,j=event.target.closest('.nodes .jn');s.querySelectorAll('.on').forEach(function(e){e.classList.remove('on')});"
            "s.classList.toggle('hl',!!j);if(j){s.querySelectorAll('[data-p~='+j.dataset.p+']').forEach(function(e){e.classList.add('on')})}")


def _sym(kind, x, y):
    """Small legend drawing, left edge x, vertical centre y."""
    if kind == "compound":
        return f'<g class="n compound"><rect x="{x}" y="{y - 7}" width="26" height="14" rx="7"/></g>'
    if kind == "cur":
        return f'<g class="n compound cur"><rect x="{x}" y="{y - 7}" width="26" height="14" rx="7"/></g>'
    if kind == "protein":
        return f'<g class="n protein"><rect x="{x}" y="{y - 7}" width="26" height="14" rx="3"/></g>'
    if kind == "complex":
        return f'<g class="n complex"><polygon points="{x + 6},{y - 7} {x + 20},{y - 7} {x + 26},{y} {x + 20},{y + 7} {x + 6},{y + 7} {x},{y}"/></g>'
    if kind == "nucleic":
        return f'<g class="n nucleic"><polygon points="{x + 5},{y - 7} {x + 26},{y - 7} {x + 21},{y + 7} {x},{y + 7}"/></g>'
    if kind == "reaction":
        return f'<g class="jn reaction"><rect x="{x}" y="{y - 8}" width="28" height="16" rx="8"/><text x="{x + 14}" y="{y + 3.6}" text-anchor="middle">R#</text></g>'
    if kind == "transport":
        return (f'<g class="jn transport"><rect x="{x}" y="{y - 8}" width="28" height="16" rx="8"/><rect class="inner" x="{x + 3}" y="{y - 5}" '
                f'width="22" height="10" rx="5"/><text x="{x + 14}" y="{y + 3.6}" text-anchor="middle">T#</text></g>')
    if kind == "merged":
        return f'<g class="badge"><rect x="{x}" y="{y - 7}" width="30" height="14" rx="7"/><text x="{x + 15}" y="{y + 3.5}" text-anchor="middle">×N</text></g>'
    if kind == "same":
        return f'<g class="badge"><rect x="{x}" y="{y - 7}" width="30" height="14" rx="7"/><text x="{x + 15}" y="{y + 3.5}" text-anchor="middle">same</text></g>'
    if kind == "xcap":
        return f'<text class="xcap" x="{x + 14}" y="{y + 3.5}" text-anchor="middle">a → b</text>'
    cls, mk, extra = {
        "flow": ("e prod", "arr", ""), "cat": ("e cat", "cat", ""), "act": ("e act", "act", ""), "inh": ("e inh", "inh", ""),
        "assoc": ("e assoc", "dot", ""), "dir": ("e assoc dir", "arrd", ""), "wh": ("e assoc wh", "", ""),
    }[kind]
    mm = f' marker-end="url(#{mk})"' + (' marker-start="url(#dot)"' if mk == "dot" else "") if mk else ""
    return f'<path class="{cls}" d="M{x},{y} h{SYM_W - 4}"{mm}/>'


def legend_plan(m, lay):
    """Work out the legend rows for canvas width lay['W'] and extend lay['H'] by their height (rows wrap on narrow canvases)."""
    W = lay["W"]
    kinds = {n.kind for n in m.nodes.values()}
    jk = {n.jkind for n in m.nodes.values() if n.kind == "junction"}
    ek = {e.kind for e in m.edges}
    items = []
    if "compound" in kinds:
        items.append(("compound", "molecule"))
    if "protein" in kinds:
        items.append(("protein", "protein / enzyme"))
    if "complex" in kinds:
        items.append(("complex", "complex"))
    if "nucleic" in kinds:
        items.append(("nucleic", "nucleic acid"))
    if "reaction" in jk:
        items.append(("reaction", "reaction step (R + number from the data)"))
    if "transport" in jk:
        items.append(("transport", "transport"))
    if ek & {"sub", "prod", "xin", "xout"}:
        items.append(("flow", "substrate → product"))
    if "cat" in ek:
        items.append(("cat", "catalyses"))
    if "act" in ek:
        items.append(("act", "activates"))
    if "inh" in ek:
        items.append(("inh", "inhibits"))
    if any(e.kind == "assoc" and not getattr(e, "directed", False) for e in m.edges):
        items.append(("assoc", "binds / other link"))
    if any(e.kind == "assoc" and getattr(e, "directed", False) for e in m.edges):
        items.append(("dir", "acts on (direction stated)"))
    if any(e.withheld for e in m.edges):
        items.append(("wh", "paler = in the text, withheld from the PWML export"))
    if any(getattr(n, "merge_note", "") for n in m.nodes.values()):
        items.append(("merged", "×N in data = N identical steps drawn once"))
    if any(n.note for n in m.nodes.values() if n.kind == "junction") or any(e.badge for e in m.edges):
        items.append(("same", "same as R# = same molecules, listed as a separate step"))
    if any(getattr(n, "cur", False) for n in m.nodes.values()):
        items.append(("cur", "dashed = local copy of a common molecule (H+, water, ATP ...)"))
    if lay["kind"] == "spine" and "transport" in jk:
        items.append(("xcap", "compartments a transport joins"))
    rows, x, y, row_h = [], 16, 0, LEG_ROW
    pos = []
    for k, label in items:
        wd = SYM_W + 6 + tw(label) + 22
        if x > 16 and x + wd - 22 > W - 16:
            x, y = 16, y + row_h
        pos.append((k, label, x, y))
        x += wd
    h = (y + row_h if items else 0) + 12
    reg = getattr(m, "reg", None)
    if reg and reg["edges"]:
        h += 2 * 20 + 4
    lay["Hc"] = lay["H"]
    lay["leg"] = dict(pos=pos, h=h)
    lay["H"] = lay["H"] + h


def legend_svg(m, lay):
    leg = lay["leg"]
    y0, W = lay["Hc"], lay["W"]
    out = [f'<g class="legend"><rect class="legendbg" x="0" y="{y0:.1f}" width="{W:.1f}" height="{leg["h"]:.1f}"/>'
           f'<line class="sep" x1="0" x2="{W:.1f}" y1="{y0:.1f}" y2="{y0:.1f}"/>']
    for k, label, x, y in leg["pos"]:
        cy = y0 + 18 + y
        out.append(_sym(k, x, cy))
        out.append(f'<text x="{x + SYM_W + 6}" y="{cy + 4}">{escape(label)}</text>')
    reg = getattr(m, "reg", None)
    if reg and reg["edges"]:
        n = reg["edges"]
        word = "link" if n == 1 else "links"
        full = getattr(m, "view", "") == "full"
        ly = y0 + leg["h"] - 2 * 20 - 4 + 16
        head = f"Showing all {n} regulation {word}" if full else f"{n} regulation {word} hidden"
        btn = "Hide regulation links" if full else "Show regulation links"
        hw = tw(head) + 6
        bw = tw(btn, True) + 22
        out.append(f'<text class="reghead" x="16" y="{ly:.1f}">{escape(head)}</text>')
        out.append(f'<g class="regtoggle" role="button" tabindex="0" aria-label="{btn}" onclick="{TOGGLE_JS}" onkeydown="{TOGGLE_KEY_JS}">'
                   f'<rect x="{16 + hw + 10:.1f}" y="{ly - 14:.1f}" width="{bw:.1f}" height="20" rx="10"/>'
                   f'<text x="{16 + hw + 10 + bw / 2:.1f}" y="{ly:.1f}" text-anchor="middle">{btn}</text></g>')
        bits = []
        if reg["act"]:
            bits.append(f'{reg["act"]} activate')
        if reg["inh"]:
            bits.append(f'{reg["inh"]} inhibit')
        if reg["other"]:
            bits.append(f'{reg["other"]} bind or act on')
        det = ", ".join(bits)
        det += f'; {reg["exported"]} written to the PWML, {reg["withheld"]} withheld'
        if reg["molecules"]:
            det += f'; {reg["molecules"]} molecule{"s" if reg["molecules"] != 1 else ""} appear only in them'
        out.append(f'<text class="regdet" x="16" y="{ly + 20:.1f}">{escape(det)}</text>')
    out.append("</g>")
    return "".join(out)


def render(ir, style, ident, hide_reg=True, tiers=1, order="flow"):
    """Returns (svg, model of the default view, layout of the default view).
    With regulation links present (and some reaction or transport to show) the default view hides them; the SVG holds a
    second, complete view that the 'Show regulation links' button switches on. model.full / model.lay_full hold that view."""
    ent = {}
    for t, k in ENT_KEYS.items():
        for e in ir["entities"].get(k, []):
            ent[(t, e["key"])] = e
    P = ir.get("processes", {})
    nproc = len(P.get("reactions", [])) + len(P.get("transports", []))

    def make(hide):
        m = build(ir)
        m._ent = ent
        m.reg = hide_regulation(m) if hide else None
        return m

    mc = make(False)
    hideable = hide_reg and nproc > 0 and any(e.kind in REG_KINDS for e in mc.edges)
    mf = None
    if hideable:
        mf = mc
        mc = make(True)
        mf.reg = dict(mc.reg)
        mf.view, mc.view = "full", "core"
    else:
        mc.view = "plain"
    layc = layout(mc, style, tiers, order)
    legend_plan(mc, layc)
    views = [draw_view(mc, layc, "view-core")]
    layf = None
    if mf is not None:
        layf = layout(mf, style, tiers, order)
        legend_plan(mf, layf)
        views.append(draw_view(mf, layf, "view-full"))
    mc.full, mc.lay_full = mf, layf
    W, H = layc["W"], layc["H"]
    hid = mc.reg or {"keys": []}
    attrs = f'data-style="{style}" data-tiers="{layc.get("tiers", 1)}" data-order="{layc.get("order", "")}" data-core="{W:.0f} {H:.0f}"'
    if layf:
        attrs += f' data-full="{layf["W"]:.0f} {layf["H"]:.0f}" data-hidden="{" ".join(hid["keys"])}"'
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" class="dg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" '
           f'role="img" aria-labelledby="{ident}-t" {attrs} onmouseover="{HOVER_JS}" onmouseleave="this.classList.remove(\'hl\')">\n'
           f'<title id="{ident}-t">{escape(ident)}</title>\n{DEFS}\n' + "\n".join(views) + "\n</svg>")
    return svg, mc, layc
