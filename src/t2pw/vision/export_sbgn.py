"""SBGN-ML 0.3 Process Description exporter for a payload in the final_mapped.json schema.

Pure: no network, no DB, no LLM. Coordinates come from the PWML IR when given, else a deterministic grid.

    sbgn_export(payload, ir=None) -> (text, report)
    sbgn_text(payload, *, ir=None) -> str
    write_sbgn(payload, path, *, ir=None)

Mapping: one compartment glyph per subcellular location used ("unspecified" when a state has none);
one entity glyph per (entity, compartment) used; each reaction / transport -> one process glyph with
consumption, production, catalysis and modulation arcs. An interaction becomes a stimulation / inhibition /
modulation arc from entity_1 to the process(es) it resolves to (SBGN PD modulation arcs must end at a process):
a process name, or the processes entity_2 itself catalyses or transports (no complex-membership inference).
Only a closed list of regulatory relationships gets an arc; binding, component_of, SAME_AS etc. do not.
Anything else is omitted and counted in the report; placements the payload did not state are counted in
report["notes"], merged compartments in report["compartment_collisions"].
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

NS = "http://sbgn.org/libsbgn/0.3"
XNS = "urn:t2pw:sbgn-ext"  # our annotations live in <extension> (schema: any content, skipped)
PD = "http://identifiers.org/combine.specifications/sbgn.pd.level-1.version-1.3"
# (kind, payload bucket, SBGN class)
KINDS = (
    ("compound", "compounds", "simple chemical"),
    ("protein", "proteins", "macromolecule"),
    ("protein_complex", "protein_complexes", "complex"),
    ("nucleic_acid", "nucleic_acids", "nucleic acid feature"),
)
SBGN_CLASS = {k: c for k, _, c in KINDS}
SIZE = {"compound": (40, 40), "protein": (80, 40), "nucleic_acid": (80, 40), "protein_complex": (100, 50)}
PROC = 20
CELL_W, CELL_H, COLS, PAD = 200, 110, 8, 40
# Namespaces are emitted as literal xmlns attributes on the root, never via ET.register_namespace
# (its prefix registry is process-wide, so output would depend on import order).


def _cf(s) -> str:
    return str(s).strip().casefold()


# Closed word lists, matched on word prefixes after lower-casing and treating "_" as a space. Anything not
# listed is NOT regulation and gets no arc (binding, component_of, SAME_AS, substrate_of, ...).
# ponytail: deactivat*/inactivat* are read as inhibition (the paper states the target loses activity).
_INHIBIT = ("inhibit", "downregulat", "repress", "deactivat", "inactivat")
_STIMULATE = ("activat", "stimulat", "upregulat")
_MODULATE = ("modif", "regulat", "modulat", "cofactor")


def _rel_class(text) -> str | None:
    words = re.sub(r"[_\s]+", " ", str(text or "")).strip().casefold().split()
    for cls, prefixes in (("inhibition", _INHIBIT), ("stimulation", _STIMULATE), ("modulation", _MODULATE)):
        if any(w.startswith(p) for w in words for p in prefixes):
            return cls
    return None


def _role_class(role) -> str | None:
    """Modifier role -> arc class; catalyst / enzyme / empty -> catalysis; unknown role -> None (no arc)."""
    return "catalysis" if _cf(role or "") in ("", "catalyst", "enzyme") else _rel_class(role)


def _ids(row) -> dict:
    out = {}
    for k, v in sorted((row.get("mapped_ids") or {}).items()):
        if isinstance(v, (str, int)) and str(v).strip():
            out[re.sub(r"\W", "_", str(k)).lstrip("0123456789_") or "id"] = str(v)
    return out


class _Model:
    def __init__(self, payload: dict, ir: dict | None):
        ent = payload.get("entities") or {}
        self.idx = {k: {} for k, _, _ in KINDS}
        for k, bucket, _ in KINDS:
            for r in ent.get(bucket) or []:
                if isinstance(r.get("name"), str) and r["name"].strip():
                    self.idx[k].setdefault(_cf(r["name"]), r)
        self.bs_loc = {}
        self.bs_ctx = {}  # cf state name -> (species, cell_type, tissue)
        for b in payload.get("biological_states") or []:
            if isinstance(b.get("name"), str) and isinstance(b.get("subcellular_location"), str) and b["subcellular_location"].strip():
                self.bs_loc[_cf(b["name"])] = b["subcellular_location"].strip()
                self.bs_ctx[_cf(b["name"])] = tuple(str(b.get(k) or "") for k in ("species", "cell_type", "tissue"))
        self.comp_ctx = defaultdict(set)  # compartment -> {(species, cell_type, tissue)} of the states that map to it
        self.notes = Counter()  # things the payload did not state, or that were merged: reported, not hidden
        self.comps = {}  # cf location -> {id,label}
        self.glyphs = {}  # (kind, cf name, comp) -> glyph dict
        self.procs = []
        self.proc_by_name = {}
        self.catalysed = defaultdict(list)  # (kind, cf name) -> [proc]
        self.omit = defaultdict(Counter)
        self.n = Counter()
        self.ir = _ir_index(ir, self.bs_loc) if ir else None
        self._build(payload.get("processes") or {})

    # --- lookups ---------------------------------------------------------------------------
    def comp(self, state=None, loc=None) -> str:
        if not loc and state:
            loc = self.bs_loc.get(_cf(state))
        key = _cf(loc) if loc else ""
        if state and _cf(state) in self.bs_ctx and key:
            self.comp_ctx[key].add(self.bs_ctx[_cf(state)])
        if key not in self.comps:
            self.comps[key] = {"id": "c%d" % (len(self.comps) + 1), "label": loc or "unspecified"}
        return key

    def find(self, name, hint=None):
        if not isinstance(name, str):
            return None
        for k in ([hint] if hint in self.idx else []) + [k for k, _, _ in KINDS]:
            if _cf(name) in self.idx[k]:
                return k, self.idx[k][_cf(name)]
        return None

    def glyph(self, kind, row, comp) -> dict:
        key = (kind, _cf(row["name"]), comp)
        if key not in self.glyphs:
            members = []
            for c in row.get("components") or [] if kind == "protein_complex" else []:
                m = c if isinstance(c, dict) else {"name": c}
                if isinstance(m.get("name"), str) and m["name"].strip():
                    members.append(m)
            w, h = SIZE[kind]
            if members:
                w, h = max(w, 10 + 2 * 75), max(h, 10 + 35 * ((len(members) + 1) // 2))
            self.glyphs[key] = {"id": "g%d" % (len(self.glyphs) + 1), "kind": kind, "row": row, "comp": comp,
                                "members": members, "w": w, "h": h}
        return self.glyphs[key]

    def add_proc(self, ptype, name, comp, ins, outs, mods, hint_xy_key=None) -> dict:
        p = {"id": "p%d" % (len(self.procs) + 1), "type": ptype, "name": name or "", "comp": comp,
             "in": ins, "out": outs, "mods": mods}
        self.procs.append(p)
        if name:
            self.proc_by_name.setdefault(_cf(name), p)
        for cls, g in mods:
            if cls == "catalysis":
                self.catalysed[(g["kind"], _cf(g["row"]["name"]))].append(p)
        return p

    # --- payload -> model -----------------------------------------------------------------
    def _members(self, items, comp, kind_label, hint_key="entity"):
        """Resolve entity names to glyphs; unresolved ones are counted. Returns glyph list (deduped)."""
        out = []
        for it in items or []:
            d = it if isinstance(it, dict) else {"entity": it}
            nm = d.get("entity") or d.get("name") or d.get("protein")
            hit = self.find(nm, d.get("entity_type"))
            if not hit:
                self.omit[kind_label]["unresolved_entity"] += 1
                continue
            g = self.glyph(hit[0], hit[1], comp)
            if g not in out:
                out.append(g)
        return out

    def _mods(self, rx, comp, label):
        mods = []
        for e in list(rx.get("enzymes") or []) + list(rx.get("modifiers") or rx.get("transporters") or []):
            if not isinstance(e, dict):
                e = {"entity": e}
            hit = self.find(e.get("entity") or e.get("protein"), e.get("entity_type"))
            if not hit:
                self.omit[label]["unresolved_entity"] += 1
                continue
            cls = _role_class(e.get("role"))
            if cls is None:
                self.omit[label]["role_not_regulatory"] += 1
                continue
            m = (cls, self.glyph(hit[0], hit[1], comp))
            if not any(m[0] == c and m[1] is g for c, g in mods):
                mods.append(m)
        return mods

    def _build(self, pr: dict):
        for rx in pr.get("reactions") or []:
            name = rx.get("name") or ""
            if rx.get("biological_state") and _cf(rx["biological_state"]) in self.bs_loc:
                c = self.comp(rx["biological_state"])
            elif isinstance(rx.get("subcellular_location"), str) and rx["subcellular_location"].strip():
                c = self.comp(loc=rx["subcellular_location"].strip())
            elif self.ir and _cf(name) in self.ir["rxn_state"]:
                c = self.comp(self.ir["rxn_state"][_cf(name)])
                self.notes["compartment_assumed:reaction_state_from_ir"] += 1
            else:
                c = self.comp()
            ins = self._members(rx.get("inputs"), c, "reaction_arcs")
            outs = self._members(rx.get("outputs"), c, "reaction_arcs")
            self.add_proc("reaction", name, c, ins, outs, self._mods({"enzymes": rx.get("enzymes"), "modifiers": rx.get("modifiers")}, c, "reaction_arcs"))
            self.n["reactions"] += 1
        for tr in pr.get("transports") or []:
            sides = {e.get("side"): e.get("biological_state") for e in tr.get("elements_with_states") or []
                     if isinstance(e, dict) and _cf(e.get("element") or "") == _cf(tr.get("cargo") or "")}
            fs = tr.get("from_biological_state") or sides.get("left")
            ts = tr.get("to_biological_state") or sides.get("right")
            hit = self.find(tr.get("cargo"))
            if not hit:
                self.omit["transports"]["unresolved_cargo"] += 1
                continue
            c1, c2 = self.comp(fs), self.comp(ts)
            mods = self._mods({"enzymes": tr.get("transporters")}, c1, "transport_arcs")
            self.add_proc("transport", tr.get("name"), c1, [self.glyph(hit[0], hit[1], c1)], [self.glyph(hit[0], hit[1], c2)], mods)
            self.n["transports"] += 1
            if c1 == c2:
                self.notes["transport_same_compartment"] += 1
            self.notes["compartment_assumed:transporter_in_from_compartment"] += len(mods)
        for ix in pr.get("interactions") or []:
            self._interaction(ix)

    def _interaction(self, ix):
        e1, e2 = ix.get("entity_1"), ix.get("entity_2")
        cls = _rel_class(ix.get("relationship"))
        if cls is None:  # binding, component_of, SAME_AS, ...: not regulation, so no arc
            self.omit["interactions"]["relationship_not_regulatory"] += 1
            return
        src, dst = self.find(e1), self.find(e2)
        if not src:
            self.omit["interactions"]["source_is_process" if _cf(e1 or "") in self.proc_by_name else "unresolved_endpoint"] += 1
            return
        if dst:
            targets = list(self.catalysed.get((dst[0], _cf(dst[1]["name"]))) or [])  # processes the entity itself catalyses
            reason = "target_entity_catalyses_no_process"
        else:
            targets = [self.proc_by_name[_cf(e2 or "")]] if _cf(e2 or "") in self.proc_by_name else []
            reason = "unresolved_endpoint"
        if not targets:
            self.omit["interactions"][reason] += 1
            return
        if ix.get("biological_state") and _cf(ix["biological_state"]) in self.bs_loc:
            c = self.comp(ix["biological_state"])
        else:
            c = targets[0]["comp"]
            self.notes["compartment_assumed:interaction_source_in_target_compartment"] += 1
        g = self.glyph(src[0], src[1], c)
        arcs = 0
        for p in targets:
            if not any(cls == c2 and g is g2 for c2, g2 in p["mods"]):
                p["mods"].append((cls, g))
                arcs += 1
        self.n["interactions"] += 1
        self.n["interactions_multi_arc"] += arcs > 1


def _ir_index(ir: dict, bs_loc: dict) -> dict:
    ents = {}
    for k, bucket, _ in KINDS:
        for e in (ir.get("entities") or {}).get(bucket) or []:
            ents[(k, e.get("key"))] = e.get("name")
    bsn = {b.get("key"): b.get("name") for b in ir.get("biological_states") or []}
    out = {"xy": {}, "box": {}, "proc": defaultdict(list), "rxn_state": {}}
    for loc in ir.get("locations") or []:
        nm, lc = ents.get((loc.get("entity_type"), loc.get("entity_key"))), bs_loc.get(_cf(bsn.get(loc.get("biological_state_key")) or ""))
        if nm and lc and loc.get("x") is not None and loc.get("y") is not None:
            out["xy"].setdefault((loc["entity_type"], _cf(nm), _cf(lc)), (loc["x"], loc["y"]))
    for bv in ir.get("bound_visualizations") or []:
        lc = bs_loc.get(_cf(bsn.get(bv.get("biological_state_key")) or ""))
        if lc and None not in (bv.get("x"), bv.get("y"), bv.get("width"), bv.get("height")):
            x0, y0, x1, y1 = bv["x"], bv["y"], bv["x"] + bv["width"], bv["y"] + bv["height"]
            o = out["box"].get(_cf(lc))
            out["box"][_cf(lc)] = (x0, y0, x1, y1) if not o else (min(o[0], x0), min(o[1], y0), max(o[2], x1), max(o[3], y1))
    pn = {}
    for bucket in ("reactions", "transports"):
        for p in (ir.get("processes") or {}).get(bucket) or []:
            pn[p.get("key")] = p.get("name") or ""
            if bucket == "reactions" and bsn.get(p.get("biological_state_key")):
                out["rxn_state"].setdefault(_cf(p.get("name") or ""), bsn[p["biological_state_key"]])
    for pv in ir.get("process_visualizations") or []:
        nm = pn.get(pv.get("process_key"))
        if nm is not None and pv.get("x") is not None and pv.get("y") is not None:
            out["proc"][(pv["type"].split("_")[0], _cf(nm))].append((pv["x"], pv["y"]))
    return out


def _layout(m: _Model) -> dict:
    """Return {comp: (x0,y0,x1,y1)}; sets x,y on every glyph and process dict."""
    ir = m.ir
    items = defaultdict(list)
    for g in m.glyphs.values():
        items[g["comp"]].append(g)
    for p in m.procs:
        items[p["comp"]].append(p)
    ystack = max([b[3] for b in ir["box"].values()] + [0]) if ir else 0
    boxes, queues = {}, {k: list(v) for k, v in (ir["proc"].items() if ir else [])}
    for ck in m.comps:
        origin = ir["box"][ck][:2] if ir and ck in ir["box"] else (0, ystack)
        slots = 0
        for it in items[ck]:
            xy = None
            if ir:
                if "kind" in it:
                    xy = ir["xy"].get((it["kind"], _cf(it["row"]["name"]), ck))
                else:
                    q = queues.get((it["type"], _cf(it["name"])))
                    xy = q.pop(0) if q else None
            m.n["ir_positioned" if xy else "grid_positioned"] += 1
            if not xy:
                xy = (origin[0] + PAD + (slots % COLS) * CELL_W, origin[1] + PAD + 30 + (slots // COLS) * CELL_H)
                slots += 1
            it["x"], it["y"] = round(xy[0]), round(xy[1])
            it.setdefault("w", PROC), it.setdefault("h", PROC)
        x0, y0 = origin
        x1, y1 = origin
        for it in items[ck]:
            x0, y0 = min(x0, it["x"] - PAD), min(y0, it["y"] - PAD - 20)
            x1, y1 = max(x1, it["x"] + it["w"] + PAD), max(y1, it["y"] + it["h"] + PAD)
        if ir and ck in ir["box"]:
            b = ir["box"][ck]
            x0, y0, x1, y1 = min(x0, b[0]), min(y0, b[1]), max(x1, b[2]), max(y1, b[3])
        boxes[ck] = tuple(round(v) for v in (x0, y0, x1, y1))
        if not (ir and ck in ir["box"]):
            ystack = max(ystack, boxes[ck][3])
    return boxes


def _x(s) -> str:  # drop characters XML 1.0 cannot carry
    return re.sub("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]", "", str(s))


def _el(parent, tag, **a):
    return ET.SubElement(parent, tag, {k.rstrip("_"): _x(v) for k, v in a.items()})


def _bbox(parent, x, y, w, h):
    _el(parent, "bbox", x=x, y=y, w=w, h=h)


def _ids_ext(parent, row, tag="ids"):
    ids = _ids(row)
    if ids:
        ET.SubElement(_el(parent, "extension"), "t2pw:" + tag, {k: _x(v) for k, v in ids.items()})


def sbgn_export(payload: dict, ir: dict | None = None) -> tuple[str, dict]:
    m = _Model(payload or {}, ir)
    boxes = _layout(m)
    root = ET.Element("sbgn", {"xmlns": NS, "xmlns:t2pw": XNS})
    mp = _el(root, "map", language="process description", version=PD, id="map1")
    for ck, c in m.comps.items():
        x0, y0, x1, y1 = boxes[ck]
        g = _el(mp, "glyph", **{"class": "compartment", "id": c["id"]})
        _el(g, "label", text=c["label"])
        _bbox(g, x0, y0, x1 - x0, y1 - y0)
    for g in m.glyphs.values():
        e = _el(mp, "glyph", **{"class": SBGN_CLASS[g["kind"]], "id": g["id"], "compartmentRef": m.comps[g["comp"]]["id"]})
        _ids_ext(e, g["row"])
        _el(e, "label", text=g["row"]["name"].strip())
        _bbox(e, g["x"], g["y"], g["w"], g["h"])
        for i, mem in enumerate(g["members"]):
            me = _el(e, "glyph", **{"class": "macromolecule", "id": "%sm%d" % (g["id"], i + 1)})
            _ids_ext(me, mem)
            _el(me, "label", text=mem["name"].strip())
            _bbox(me, g["x"] + 5 + (i % 2) * 75, g["y"] + 5 + (i // 2) * 35, 70, 30)
    for p in m.procs:
        e = _el(mp, "glyph", **{"class": "process", "id": p["id"], "compartmentRef": m.comps[p["comp"]]["id"]})
        ET.SubElement(_el(e, "extension"), "t2pw:process", {"kind": p["type"], "name": _x(p["name"])})
        _bbox(e, p["x"], p["y"], PROC, PROC)
        _el(e, "port", id=p["id"] + "_in", x=p["x"] - 10, y=p["y"] + PROC // 2)
        _el(e, "port", id=p["id"] + "_out", x=p["x"] + PROC + 10, y=p["y"] + PROC // 2)
    n_arc = 0

    def arc(cls, src, tgt, s, t):
        nonlocal n_arc
        n_arc += 1
        a = _el(mp, "arc", **{"class": cls, "id": "a%d" % n_arc, "source": src, "target": tgt})
        _el(a, "start", x=s[0], y=s[1])
        _el(a, "end", x=t[0], y=t[1])

    ctr = lambda o: (o["x"] + o["w"] // 2, o["y"] + o["h"] // 2)  # noqa: E731
    for p in m.procs:
        pin, pout = (p["x"] - 10, p["y"] + PROC // 2), (p["x"] + PROC + 10, p["y"] + PROC // 2)
        for g in p["in"]:
            arc("consumption", g["id"], p["id"] + "_in", ctr(g), pin)
        for g in p["out"]:
            arc("production", p["id"] + "_out", g["id"], pout, ctr(g))
        for cls, g in p["mods"]:
            arc(cls, g["id"], p["id"], ctr(g), (p["x"] + PROC // 2, p["y"]))
    ET.indent(root)
    text = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"
    n = m.n
    report = {
        "exported": {"compartments": len(m.comps), "entity_glyphs": len(m.glyphs),
                     "complex_members": sum(len(g["members"]) for g in m.glyphs.values()),
                     "reactions": n["reactions"], "transports": n["transports"], "interactions": n["interactions"],
                     "process_glyphs": len(m.procs), "arcs": n_arc},
        "omitted": {k: dict(v) for k, v in sorted(m.omit.items())},
        "layout": {"ir_positioned": n["ir_positioned"], "grid_positioned": n["grid_positioned"]},
        "unspecified_compartment": "" in m.comps,
        # not omissions: things merged or placed without the payload saying so
        "notes": dict(sorted(m.notes.items())),
        # compartment label -> number of distinct (species, cell type, tissue) states merged into it (only if > 1)
        "compartment_collisions": {m.comps[k]["label"]: len(v) for k, v in sorted(m.comp_ctx.items()) if len(v) > 1},
    }
    report["exported"]["interactions_multi_arc"] = n["interactions_multi_arc"]
    return text, report


def sbgn_text(payload: dict, *, ir: dict | None = None) -> str:
    return sbgn_export(payload, ir)[0]


def write_sbgn(payload: dict, path, *, ir: dict | None = None) -> None:
    Path(path).write_text(sbgn_text(payload, ir=ir), encoding="utf-8", newline="\n")
