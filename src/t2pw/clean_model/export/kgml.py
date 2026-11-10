"""Clean model -> KGML text (KGML v0.7.2 structure). Stdlib only, deterministic (same model, same bytes), no network.

This is NOT a KEGG map: ``name/org/number`` are the documented placeholders ``undefined/undefined/00000`` and the
layout is a plain grid (layout only, not biology). Nothing is invented:
* KEGG ids (``cpd:C#####``, KEGG gene ids, ``rn:R#####``) only when the model already holds them (paper/db labelled);
* else the non-KEGG prefixed id the model holds (``chebi:``, ``uniprot:``, ``rhea:``), stated in an XML comment;
* else the KGML placeholder ``undefined`` with the paper label in ``<graphics name>``, listed in the omissions dict.
A class or complex never gets a member's accession. Unsigned interactions get no ``<subtype>`` (P-18).
KGML has no slot for compartments, stoichiometry, catalysts or evidence quotes; those losses are counted, not hidden.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from ..model import CleanModel
from .sbml import _Exporter, _str, _esc

UNDEF = "undefined"
NONKEGG = ("chebi", "uniprot")  # the only non-KEGG prefixes written
_KEGG_TYPES = ("kegg.compound", "kegg.genes")
_ETYPE = {"compound": "compound", "protein": "gene", "nucleic_acid": "gene", "complex": "group"}

# interaction kind -> (KGML subtype name, value). Kinds not listed are written without a subtype.
_SUBTYPE = {
    "phosphorylation": ("phosphorylation", "+p"), "dephosphorylation": ("dephosphorylation", "-p"),
    "binding": ("binding/association", "---"), "association": ("binding/association", "---"),
    "complex_formation": ("binding/association", "---"), "ubiquitination": ("ubiquination", "+u"),
}
_SIGNED = {("activation", "positive"): ("activation", "-->"), ("inhibition", "negative"): ("inhibition", "--|"),
           ("expression_regulation", "positive"): ("expression", "-->"), ("expression_regulation", "negative"): ("repression", "--|")}


def _c(s: Any) -> str:  # comment-safe: no "--"
    return re.sub(r"-{2,}", "-", re.sub("[^\t\n\r -퟿-�]", "", str(s)))


def _a(**kw: Any) -> str:
    return "".join(' %s="%s"' % (k, _esc(v, True)) for k, v in kw.items() if v is not None)


class _Kgml:
    def __init__(self, model: CleanModel, strict: bool) -> None:
        self.m, self.strict = model, strict
        self.x = _Exporter(model)
        self.om: Dict[str, Any] = {
            "unmapped_entities": [], "no_slot_in_format": {}, "direction_placeholder": {"directionStated": False, "count": 0, "names": []},
            "skipped_reactions": [], "skipped_interactions": [], "skipped_group_members": [], "sources_used": [],
            "non_kegg_prefixes_used": [], "stats": {}}
        self.prefixes: set = set()
        self.eid: Dict[int, int] = {}  # _Ent.n -> entry id
        self.stats = {k: 0 for k in ("entries", "entries_kegg", "entries_non_kegg", "entries_undefined", "reactions", "reactions_rn",
                                     "reactions_rhea", "reactions_undefined", "relations", "relations_signed", "relations_unsigned")}

    # names ------------------------------------------------------------------------------------------------------
    def _ids(self, kind: str, name: str, obj: Any) -> List[Tuple[str, str]]:
        return [(t, i) for t, i, _ in self.x.ids_for(kind, name, obj).valid]

    def entry_name(self, e: Any) -> Tuple[str, str]:
        """(name attr, class kegg|non_kegg|undefined)."""
        if e.kind == "complex":
            return UNDEF, "undefined"  # a complex never carries a member accession
        ids = self._ids(e.kind, e.name, e.obj)
        kegg = [("cpd:" + i) if t == "kegg.compound" else i for t, i in ids if t in _KEGG_TYPES]
        if kegg:
            return " ".join(dict.fromkeys(kegg)), "kegg"
        if not self.strict:
            for pre in NONKEGG:
                got = [i for t, i in ids if t == pre]
                if got:
                    self.prefixes.add(pre)
                    return " ".join(dict.fromkeys(pre + ":" + re.sub(r"(?i)^chebi:", "", i) for i in got)), "non_kegg"
        return UNDEF, "undefined"

    def reaction_name(self, rx: Any) -> Tuple[str, str]:
        ids = self._ids("reaction", _str(rx.name), rx)
        rn = [i for t, i in ids if t == "kegg.reaction"]
        if rn:
            return " ".join("rn:" + i for i in dict.fromkeys(rn)), "rn"
        rh = [i for t, i in ids if t == "rhea"]
        if rh and not self.strict:
            self.prefixes.add("rhea")
            return " ".join("rhea:" + i for i in dict.fromkeys(rh)), "rhea"
        return UNDEF, "undefined"

    # document ---------------------------------------------------------------------------------------------------
    def build(self) -> str:
        m, pw, x = self.m, self.m.pathway, self.x
        ents = [e for e in x.ents if e.kind != "undeclared"]
        for i, e in enumerate(ents, 1):
            self.eid[e.n] = i
        nid = len(ents)
        cols = max(1, math.ceil(math.sqrt(len(ents))))
        out: List[str] = []
        for i, e in enumerate(ents):
            name, cls = self.entry_name(e)
            self.stats["entries"] += 1
            self.stats["entries_" + cls] += 1
            if cls == "undefined":
                self.om["unmapped_entities"].append({"name": e.name, "kind": e.kind})
            comp = ""
            if e.kind == "complex":
                ids = []
                for ref in e.obj.members or []:
                    me = x.entity(ref.entity.value, create=False) if isinstance(ref.entity.value, str) and not ref.unknown else None
                    if me is None or me.kind == "undeclared" or me.n not in self.eid:
                        self.om["skipped_group_members"].append({"complex": e.name, "member": ref.entity.value if isinstance(ref.entity.value, str) else None})
                    elif self.eid[me.n] not in ids and me.n != e.n:
                        ids.append(self.eid[me.n])
                comp = "".join('      <component id="%d"/>\n' % c for c in ids)
            box = e.kind != "compound"
            g = _a(name=e.name, fgcolor="#000000", bgcolor={"compound": "#FFFFFF", "group": "#BFBFFF"}.get(_ETYPE[e.kind], "#BFFFBF"),
                   type="rectangle" if box else "circle", x=70 + (i % cols) * 140, y=35 + (i // cols) * 70,
                   width=100 if box else 20, height=40 if box else 20)
            out.append('  <entry id="%d"%s>\n    <graphics%s/>\n%s  </entry>\n' % (i + 1, _a(name=name, type=_ETYPE[e.kind]), g, comp))
        out = ["".join(out)]

        rels, nrel = [], 0
        lost = {"quotes": 0, "stoichiometry_participants": 0, "catalysts_and_modifiers": 0, "compartments": len(m.locations or []),
                "transport_locations": []}
        for ix in m.interactions or []:
            es, et = x.entity(_str(ix.source), create=False), x.entity(_str(ix.target), create=False)
            nm = {"source": _str(ix.source), "target": _str(ix.target), "relationship": _str(ix.relationship)}
            if es is None or et is None or es.n not in self.eid or et.n not in self.eid:
                self.om["skipped_interactions"].append(dict(nm, reason="endpoint not a declared entity"))
                continue
            kind, sign = _str(ix.kind) or "other_unsigned", _str(ix.sign) or "unsigned"
            if es.kind == "compound" and et.kind == "compound":
                self.om["skipped_interactions"].append(dict(nm, reason="compound-compound relation has no KGML relation type"))
                continue
            rtype = ("GErel" if kind == "expression_regulation" or "nucleic_acid" in (es.kind, et.kind)
                     else "PCrel" if "compound" in (es.kind, et.kind) else "PPrel")
            sub = _SIGNED.get((kind, sign)) or _SUBTYPE.get(kind)
            if ix.evidence is not None and _str(ix.evidence):
                lost["quotes"] += 1
            body = ('    <subtype name="%s" value="%s"/>\n' % (sub[0], sub[1])) if sub else "    <!-- unsigned: no subtype, sign not stated -->\n"
            rels.append('  <relation%s>\n%s  </relation>\n' % (_a(entry1=self.eid[es.n], entry2=self.eid[et.n], type=rtype), body))
            self.stats["relations"] += 1
            self.stats["relations_signed" if sub and sub[0] in ("activation", "inhibition", "expression", "repression") else "relations_unsigned"] += 1

        rxs: List[str] = []

        def add_reaction(label: str, name: str, subs: List[Any], prods: List[Any], stated: bool, rev: bool, cls: str) -> None:
            nonlocal nid
            nid += 1
            self.stats["reactions"] += 1
            self.stats["reactions_" + cls] += 1
            note = "" if stated else "    <!-- directionStated=false: type irreversible is the required placeholder, the paper did not state a direction -->\n"
            if not stated:
                self.om["direction_placeholder"]["count"] += 1
                self.om["direction_placeholder"]["names"].append(label)
            sp = "".join('    <substrate id="%d" name="%s"/>\n' % (self.eid[e.n], _esc(self.names[e.n], True)) for e in subs)
            pp = "".join('    <product id="%d" name="%s"/>\n' % (self.eid[e.n], _esc(self.names[e.n], True)) for e in prods)
            rxs.append('  <reaction%s>\n%s%s%s  </reaction>\n' % (_a(id=nid, name=name, type="reversible" if rev else "irreversible"), note, sp, pp))

        self.names = {e.n: self.entry_name(e)[0] for e in ents}
        for rx in m.reactions or []:
            label = _str(rx.name) or "(unnamed reaction)"
            sides, dropped = [], []
            for lst in (rx.reactants, rx.products):
                es = []
                for ref in lst or []:
                    lost["stoichiometry_participants"] += 1
                    e = x.entity(ref.entity.value, create=False) if isinstance(ref.entity.value, str) and not ref.unknown else None
                    if e is None or e.n not in self.eid:
                        dropped.append(ref.entity.value if isinstance(ref.entity.value, str) else None)
                    elif e not in es:
                        es.append(e)
                sides.append(es)
            lost["catalysts_and_modifiers"] += len((rx.catalysts or [])) + len((rx.modifiers or []))
            lost["quotes"] += 1 if _str(rx.evidence) else 0
            if not sides[0] or not sides[1]:
                self.om["skipped_reactions"].append({"name": label, "reason": "no declared %s%s" % (
                    "substrate" if not sides[0] else "product", "; undeclared participants: %s" % dropped if dropped else "")})
                continue
            if dropped:
                self.om["skipped_reactions"].append({"name": label, "reason": "written without undeclared participants: %s" % dropped, "written": True})
            rev, stated = x._direction(rx)
            name, cls = self.reaction_name(rx)
            add_reaction(label, name, sides[0], sides[1], stated, rev, cls)
        for t in m.transports or []:
            label = _str(t.name) or "(unnamed transport)"
            cargo = x.entity(t.cargo.value, create=False) if isinstance(t.cargo.value, str) else None
            lost["quotes"] += 1 if _str(t.evidence) else 0
            lost["catalysts_and_modifiers"] += len(t.transporters or [])
            if cargo is None or cargo.n not in self.eid or cargo.kind != "compound":
                self.om["skipped_reactions"].append({"name": label, "reason": "transport cargo is not a declared compound"})
                continue
            lost["transport_locations"].append({"name": label, "from": _str(t.from_location), "to": _str(t.to_location)})
            add_reaction(label, UNDEF, [cargo], [cargo], False, False, "undefined")

        om = self.om
        om["no_slot_in_format"] = lost
        om["sources_used"] = [f.value for f in pw.licences if isinstance(f.value, str)]
        om["non_kegg_prefixes_used"] = sorted(self.prefixes)
        om["stats"] = self.stats
        title = _str(pw.title) or _str(pw.pmcid)
        head = ["This file is not a KEGG map. It is a KGML-structured export of a clean model extracted from one paper: name, org and number are",
                "the documented placeholders (undefined, undefined, 00000), no KEGG layout, names or map content is used, and the x/y positions",
                "are a plain grid (layout only, not biology). KEGG ids appear only where the model already held them.",
                "KGML has no slot for compartments, stoichiometry, catalysts or evidence quotes; those are reported in the sidecar omissions file."]
        if self.strict:
            head.append("Strict mode: every entity or reaction without a KEGG id is written as undefined.")
        elif self.prefixes:
            head.append("Non-KEGG ids used in name attributes (%s): consumers that only resolve KEGG ids may not resolve them." % ", ".join(sorted(self.prefixes)))
        if om["sources_used"]:
            head.append("Database attribution: " + " | ".join(om["sources_used"]))
        return ('<?xml version="1.0" encoding="UTF-8"?>\n<!--\n%s\n-->\n<pathway%s>\n%s%s%s</pathway>\n' % (
            "\n".join(_c(h) for h in head), _a(name=UNDEF, org=UNDEF, number="00000", title=title),
            out[0], "".join(rels), "".join(rxs)))


def kgml_text(model: CleanModel, strict_kegg: bool = False) -> Tuple[str, Dict[str, Any]]:
    k = _Kgml(model, strict_kegg)
    text = k.build()
    return text, k.om


def write_kgml(model: CleanModel, path: str, strict_kegg: bool = False) -> Dict[str, Any]:
    text, om = kgml_text(model, strict_kegg)
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))
    with open(path + ".omissions.json", "wb") as fh:
        fh.write((json.dumps(om, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    return om


def kgml_for_run(run_dir: str, fill: bool = False, strict_kegg: bool = False) -> Tuple[str, Dict[str, Any]]:
    from ..builder import build_from_run
    model = build_from_run(run_dir)
    if fill:
        from ..fill import fill_model
        fill_model(model)
    return kgml_text(model, strict_kegg)
