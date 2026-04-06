"""
json_to_sbml.py — PathWhiz-compatible SBML Level 3 builder.

Handles all reaction types seen in real PathWhiz SBMLs:
  - reaction     (SBO:0000176, reversible or not)
  - transport    (SBO:0000185, reactant sboTerm SBO:0000010, product SBO:0000011)
  - interaction  (SBO:0000342)
  - sub_pathway  (SBO:0000375)

Species types:
  - compound           (SBO:0000247)
  - protein            (SBO:0000245)
  - protein_complex    (SBO:0000245)
  - element_collection (SBO:0000247, id prefix ElementCollection)

Modifier sboTerm:
  - SBO:0000460 for enzyme/catalyst (reaction)
  - SBO:0000019 for transporter (transport)

Layout: per-speciesReference <pathwhiz:location> annotations with
compound_location + edge location_elements. Uses layout_engine.py for
circular (TCA-style) or linear arrangement of any pathway.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sys
import uuid
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from layout_engine import (
    build_pathway_layout as _build_pathway_layout,
    _norm as _le_norm,
    _is_cofactor as _le_is_cofactor,
    COMPOUND_W as _LE_CW,
    COMPOUND_H as _LE_CH,
    PROTEIN_W  as _PROTEIN_W,
    PROTEIN_H  as _PROTEIN_H,
    CANVAS_W, CANVAS_H,
)

# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_list(v: Any) -> List:   return v if isinstance(v, list) else []
def _safe_dict(v: Any) -> Dict:   return v if isinstance(v, dict) else {}

def _normalize(name: str) -> str:
    s = re.sub(r"\s+", " ", (name or "").strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", "", s)

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _html_name(name: str) -> str:
    return (name
            .replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))

def _arrow_options() -> str:
    """Escaped JSON options string for product edges (start_arrow=true)."""
    raw = ('{"start_arrow":true,'
           '"start_arrow_path":"M 25.9 13.3 L 11 12 L 17.4 25.6",'
           '"start_flat_arrow":false,"start_flat_arrow_path":null}')
    return raw.replace('"', "&quot;")

def _end_arrow_options() -> str:
    raw = ('{"end_arrow":true,'
           '"end_arrow_path":"M 25.9 13.3 L 11 12 L 17.4 25.6",'
           '"end_flat_arrow":false,"end_flat_arrow_path":null}')
    return raw.replace('"', "&quot;")

# ── ID counters ───────────────────────────────────────────────────────────────

class _Ctr:
    def __init__(self, start: int = 1) -> None:
        self._n = start
    def next(self) -> int:
        v = self._n; self._n += 1; return v

# ── Geometry helpers ──────────────────────────────────────────────────────────

_COMPOUND_W  = _LE_CW
_COMPOUND_H  = _LE_CH
_TMPL_COMP   = "3"
_TMPL_EDGE   = "5"
_TMPL_PROTO  = "6"
_TMPL_ECOLL  = "37"
_MODIFIER_ABOVE = 220.0    # how far above reaction square enzyme boxes sit


def _edge_path(x1: float, y1: float, x2: float, y2: float) -> str:
    """Cubic bezier between two points with a gentle midpoint bulge."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    return (f"M{x1:.0f} {y1:.0f} "
            f"C{mx:.0f} {y1:.0f} {mx:.0f} {y2:.0f} "
            f"{x2:.0f} {y2:.0f} ")


def _modifier_positions(
    cx: float, cy: float, count: int
) -> List[Tuple[float, float, float, float]]:
    """
    Return (node_cx, node_cy, node_x, node_y) for each modifier box,
    placed above the reaction square in a horizontal row.
    """
    results = []
    for j in range(count):
        offset_x = (j - (count - 1) / 2) * (_PROTEIN_W + 20.0)
        node_cx  = cx + offset_x
        node_cy  = cy - _MODIFIER_ABOVE
        node_x   = node_cx - _PROTEIN_W / 2
        node_y   = node_cy - _PROTEIN_H / 2
        results.append((node_cx, node_cy, node_x, node_y))
    return results


# ── SBML emission helpers ─────────────────────────────────────────────────────

def _rdf_xrefs(metaid: str, mapped_ids: Dict[str, Any]) -> List[str]:
    DB_URN = {
        "hmdb":     "urn:miriam:hmdb:",
        "kegg":     "urn:miriam:kegg.compound:",
        "chebi":    "urn:miriam:chebi:",
        "pubchem":  "urn:miriam:pubchem.compound:",
        "drugbank": "urn:miriam:drugbank:",
        "uniprot":  "urn:miriam:uniprot:",
    }
    uris = []
    for k, v in sorted(mapped_ids.items()):
        if not isinstance(v, str) or not v.strip():
            continue
        prefix = DB_URN.get(k.lower().strip())
        if prefix:
            uris.append(f"{prefix}{v.strip()}")
    if not uris:
        return []
    return [
        f'\t<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        f' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">',
        f'\t<rdf:Description rdf:about="#{metaid}">',
        "              <bqbiol:is>", "\t<rdf:Bag>",
        *[f'\t<rdf:li rdf:resource="{u}"/>' for u in uris],
        "\t</rdf:Bag>", "\t</bqbiol:is>",
        "\t</rdf:Description>", "\t</rdf:RDF>",
    ]


def _rdf_has_part(metaid: str, uniprot_ids: List[str]) -> List[str]:
    return [
        f'\t<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        f' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">',
        f'\t<rdf:Description rdf:about="#{metaid}">',
        "\t<bqbiol:hasPart>", "\t<rdf:Bag>",
        *[f'\t<rdf:li rdf:resource="urn:miriam:uniprot:{u}"/>' for u in uniprot_ids],
        "\t</rdf:Bag>", "\t</bqbiol:hasPart>",
        "\t</rdf:Description>", "\t</rdf:RDF>",
    ]


def _location_element_compound(
    eid: str, loc_id: int, x: float, y: float,
    w: float = _COMPOUND_W, h: float = _COMPOUND_H,
    tmpl: str = _TMPL_COMP, zindex: int = 10,
) -> str:
    return (
        f'<pathwhiz:location_element'
        f' pathwhiz:element_type="compound_location"'
        f' pathwhiz:element_id="{eid}"'
        f' pathwhiz:location_id="{loc_id}"'
        f' pathwhiz:x="{x:.1f}" pathwhiz:y="{y:.1f}"'
        f' pathwhiz:visualization_template_id="{tmpl}"'
        f' pathwhiz:width="{w:.1f}" pathwhiz:height="{h:.1f}"'
        f' pathwhiz:zindex="{zindex}" pathwhiz:hidden="false"/>'
    )


def _location_element_ecollection(
    eid: str, loc_id: int, x: float, y: float,
    w: float = 100.0, h: float = 90.0,
) -> str:
    return (
        f'<pathwhiz:location_element'
        f' pathwhiz:element_type="element_collection_location"'
        f' pathwhiz:element_id="{eid}"'
        f' pathwhiz:location_id="{loc_id}"'
        f' pathwhiz:x="{x:.1f}" pathwhiz:y="{y:.1f}"'
        f' pathwhiz:visualization_template_id="{_TMPL_ECOLL}"'
        f' pathwhiz:width="{w:.1f}" pathwhiz:height="{h:.1f}"'
        f' pathwhiz:zindex="12" pathwhiz:hidden="false"/>'
    )


def _location_element_edge(
    path: str, opts: str = '{}',
    tmpl: str = _TMPL_EDGE, zindex: int = 18,
) -> str:
    return (
        f'<pathwhiz:location_element'
        f' pathwhiz:element_type="edge"'
        f' pathwhiz:path="{path}"'
        f' pathwhiz:visualization_template_id="{tmpl}"'
        f' pathwhiz:options="{opts}"'
        f' pathwhiz:zindex="{zindex}" pathwhiz:hidden="false"/>'
    )


def _location_element_protein(
    eid: str, loc_id: int, x: float, y: float,
    w: float = _PROTEIN_W, h: float = _PROTEIN_H,
    tmpl: str = _TMPL_PROTO,
) -> str:
    return (
        f'<pathwhiz:location_element'
        f' pathwhiz:element_type="protein_location"'
        f' pathwhiz:element_id="{eid}"'
        f' pathwhiz:location_id="{loc_id}"'
        f' pathwhiz:x="{x:.1f}" pathwhiz:y="{y:.1f}"'
        f' pathwhiz:visualization_template_id="{tmpl}"'
        f' pathwhiz:width="{w:.1f}" pathwhiz:height="{h:.1f}"'
        f' pathwhiz:zindex="8" pathwhiz:hidden="false"/>'
    )


# ── Core builder ──────────────────────────────────────────────────────────────

def build_sbml(
    input_path: Path,
    sbml_path: Path,
    report_json_path: Path,
    report_txt_path: Path,
    *,
    default_compartment_name: str = "cell",
    db_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be a dict.")

    report: Dict[str, Any] = {
        "hard_errors": [], "warnings": [], "defaults_applied": [],
        "counts": {"compartments": 0, "species": 0, "reactions": 0},
        "pathwhiz_id_stats": {
            "mysql_connected": False,
            "compounds_matched": 0, "proteins_matched": 0, "species_no_id": 0,
        },
        "validation": {
            "check_count": 0, "error_count": 0,
            "has_errors": False, "messages": [],
        },
    }

    data = deepcopy(payload)
    entities  = _safe_dict(data.get("entities"))
    processes = _safe_dict(data.get("processes"))

    # ── ID counters ───────────────────────────────────────────────────────────
    state_ctr   = _Ctr(10000)
    comp_ctr    = _Ctr(20000)
    prot_ctr    = _Ctr(30000)
    cmplx_ctr   = _Ctr(40000)
    ecoll_ctr   = _Ctr(50000)
    rxn_ctr     = _Ctr(60000)
    loc_ctr     = _Ctr(100000)

    # ── Compartments ──────────────────────────────────────────────────────────
    state_name_to_id:   Dict[str, int] = {}
    state_name_to_sbml: Dict[str, str] = {}

    def _register_state(name: str) -> str:
        nm = (name or "").strip() or default_compartment_name
        if nm not in state_name_to_id:
            sid = state_ctr.next()
            state_name_to_id[nm]   = sid
            state_name_to_sbml[nm] = f"BiologicalState{sid}"
        return state_name_to_sbml[nm]

    for bs in _safe_list(data.get("biological_states")):
        if isinstance(bs, dict):
            nm = (bs.get("name") or "").strip()
            if nm:
                _register_state(nm)
    _register_state(default_compartment_name)

    # ── Entity registries ─────────────────────────────────────────────────────
    compound_by_norm: Dict[str, Dict] = {}
    protein_by_norm:  Dict[str, Dict] = {}
    complex_by_norm:  Dict[str, Dict] = {}
    ecoll_by_norm:    Dict[str, Dict] = {}

    def _register_compound(row: Dict) -> Optional[Dict]:
        name = (row.get("name") or "").strip()
        if not name:
            return None
        norm = _normalize(name)
        if norm in compound_by_norm:
            return compound_by_norm[norm]
        pw_id = None
        for k in ["pathbank_compound_id", "pw_compound_id"]:
            v = row.get(k) or _safe_dict(row.get("mapping_meta", {})).get(k)
            if v:
                try: pw_id = int(v); break
                except: pass
        if pw_id is None:
            pw_id = comp_ctr.next()
        rec = {"name": name, "norm": norm, "pw_id": pw_id,
               "sbml_id": f"Compound{pw_id}",
               "mapped_ids": _safe_dict(row.get("mapped_ids")),
               "stype": "compound"}
        compound_by_norm[norm] = rec
        return rec

    def _register_protein(row: Dict) -> Optional[Dict]:
        name = (row.get("name") or "").strip()
        if not name:
            return None
        norm = _normalize(name)
        if norm in protein_by_norm:
            return protein_by_norm[norm]
        pw_id = None
        for k in ["pathbank_protein_id", "pw_protein_id"]:
            v = row.get(k) or _safe_dict(row.get("mapping_meta", {})).get(k)
            if v:
                try: pw_id = int(v); break
                except: pass
        if pw_id is None:
            pw_id = prot_ctr.next()
        rec = {"name": name, "norm": norm, "pw_id": pw_id,
               "sbml_id": f"Protein{pw_id}",
               "mapped_ids": _safe_dict(row.get("mapped_ids")),
               "stype": "protein"}
        protein_by_norm[norm] = rec
        return rec

    def _register_complex(row: Dict) -> Optional[Dict]:
        name = (row.get("name") or "").strip()
        if not name:
            return None
        norm = _normalize(name)
        if norm in complex_by_norm:
            return complex_by_norm[norm]
        pw_id = None
        for k in ["pathbank_complex_id", "pw_complex_id"]:
            v = row.get(k) or _safe_dict(row.get("mapping_meta", {})).get(k)
            if v:
                try: pw_id = int(v); break
                except: pass
        if pw_id is None:
            pw_id = cmplx_ctr.next()
        components = [c.strip() for c in _safe_list(row.get("components"))
                      if isinstance(c, str) and c.strip()]
        rec = {"name": name, "norm": norm, "pw_id": pw_id,
               "sbml_id": f"ProteinComplex{pw_id}",
               "components": components,
               "mapped_ids": _safe_dict(row.get("mapped_ids")),
               "stype": "protein_complex"}
        complex_by_norm[norm] = rec
        return rec

    def _register_ecollection(row: Dict) -> Optional[Dict]:
        name = (row.get("name") or "").strip()
        if not name:
            return None
        norm = _normalize(name)
        if norm in ecoll_by_norm:
            return ecoll_by_norm[norm]
        pw_id = None
        for k in ["pathbank_id", "pw_id"]:
            v = row.get(k)
            if v:
                try: pw_id = int(v); break
                except: pass
        if pw_id is None:
            pw_id = ecoll_ctr.next()
        rec = {"name": name, "norm": norm, "pw_id": pw_id,
               "sbml_id": f"ElementCollection{pw_id}",
               "mapped_ids": {}, "stype": "element_collection"}
        ecoll_by_norm[norm] = rec
        return rec

    for row in _safe_list(entities.get("compounds")):
        if isinstance(row, dict): _register_compound(row)
    for row in _safe_list(entities.get("proteins")):
        if isinstance(row, dict): _register_protein(row)
    for row in _safe_list(entities.get("protein_complexes")):
        if isinstance(row, dict): _register_complex(row)
    for row in _safe_list(entities.get("element_collections")):
        if isinstance(row, dict): _register_ecollection(row)

    _protein_component_norms: Set[str] = set()
    for crec in complex_by_norm.values():
        for cn in crec.get("components", []):
            if isinstance(cn, str) and cn.strip():
                _protein_component_norms.add(_normalize(cn))

    def _resolve_entity(name: str) -> Tuple[str, Optional[Dict]]:
        norm = _normalize(name)
        for registry, stype in [
            (complex_by_norm,   "protein_complex"),
            (ecoll_by_norm,     "element_collection"),
            (protein_by_norm,   "protein"),
            (compound_by_norm,  "compound"),
        ]:
            if norm in registry:
                return stype, registry[norm]
        # Auto-register
        if norm in _protein_component_norms:
            new_id = prot_ctr.next()
            rec = {"name": name, "norm": norm, "pw_id": new_id,
                   "sbml_id": f"Protein{new_id}", "mapped_ids": {},
                   "stype": "protein"}
            protein_by_norm[norm] = rec
            return "protein", rec
        new_id = comp_ctr.next()
        rec = {"name": name, "norm": norm, "pw_id": new_id,
               "sbml_id": f"Compound{new_id}", "mapped_ids": {},
               "stype": "compound"}
        compound_by_norm[norm] = rec
        return "compound", rec

    # ── Compartment assignment from element_locations ─────────────────────────
    element_locs  = _safe_dict(data.get("element_locations"))
    entity_states: Dict[str, List[str]] = defaultdict(list)
    for loc_key, entity_field in [
        ("compound_locations",           "compound"),
        ("protein_locations",            "protein"),
        ("element_collection_locations", "element_collection"),
    ]:
        for row in _safe_list(element_locs.get(loc_key)):
            if not isinstance(row, dict):
                continue
            ename = (row.get(entity_field) or "").strip()
            state = (row.get("biological_state") or "").strip()
            if ename and state:
                entity_states[_normalize(ename)].append(state)
                _register_state(state)

    def _primary_state(entity_norm: str) -> str:
        states = entity_states.get(entity_norm, [])
        return states[0] if states else default_compartment_name

    def _compartment_sbml(entity_norm: str) -> str:
        return _register_state(_primary_state(entity_norm))

    # ── Build reaction plans ──────────────────────────────────────────────────
    seen_rxn_keys: Set[str] = set()
    reaction_plans: List[Dict] = []

    def _rxn_name(inputs: List[str], outputs: List[str],
                  reversible: bool = False, kind: str = "reaction") -> str:
        sep = " &harr; " if reversible else " &rarr; "
        lhs = " + ".join(inputs) if inputs else "?"
        rhs = " + ".join(outputs) if outputs else "?"
        return f"{lhs}{sep}{rhs}"

    # Standard reactions
    for rxn in _safe_list(processes.get("reactions")):
        if not isinstance(rxn, dict):
            continue
        inputs  = [x.strip() for x in _safe_list(rxn.get("inputs"))
                   if isinstance(x, str) and x.strip()]
        outputs = [x.strip() for x in _safe_list(rxn.get("outputs"))
                   if isinstance(x, str) and x.strip()]
        if not inputs or not outputs:
            report["hard_errors"].append(
                {"path": "/processes/reactions", "reason": "Missing inputs/outputs"})
            continue
        key = "|".join(sorted(inputs)) + "→" + "|".join(sorted(outputs))
        if key in seen_rxn_keys:
            report["warnings"].append(
                {"path": "/processes/reactions", "reason": f"Duplicate collapsed: {key[:60]}"})
            continue
        seen_rxn_keys.add(key)

        reversible  = bool(rxn.get("reversible", False))
        state_name  = (rxn.get("biological_state") or "").strip()
        if not state_name:
            for inp in inputs:
                s = _primary_state(_normalize(inp))
                if s != default_compartment_name:
                    state_name = s
                    break
        if not state_name:
            state_name = default_compartment_name
        compartment = _register_state(state_name)

        modifiers = []
        for mod in _safe_list(rxn.get("modifiers")) + _safe_list(rxn.get("enzymes")):
            if not isinstance(mod, dict):
                continue
            actor = ""
            for field in ["entity", "protein", "protein_complex", "name"]:
                v = (mod.get(field) or "").strip()
                if v:
                    actor = v
                    break
            if not actor:
                continue
            stype, rec = _resolve_entity(actor)
            if rec:
                modifiers.append({"stype": stype, "rec": rec})

        pw_id = rxn_ctr.next()
        reaction_plans.append({
            "pw_id": pw_id, "sbml_id": f"Reaction{pw_id}",
            "name": _rxn_name(inputs, outputs, reversible),
            "compartment": compartment,
            "inputs": inputs, "outputs": outputs,
            "modifiers": modifiers,
            "kind": "reaction",
            "reversible": reversible,
            "sbo": "SBO:0000176",
        })

    # Transports
    for tr in _safe_list(processes.get("transports")):
        if not isinstance(tr, dict):
            continue
        cargo      = (tr.get("cargo") or "").strip()
        from_state = (tr.get("from_biological_state") or "").strip() or default_compartment_name
        to_state   = (tr.get("to_biological_state")   or "").strip() or default_compartment_name
        if not cargo:
            continue
        if from_state == to_state:
            report["warnings"].append(
                {"path": "/processes/transports",
                 "reason": f"Degenerate transport '{cargo}' skipped"})
            continue

        _register_state(from_state)
        _register_state(to_state)
        key = f"transport|{_normalize(cargo)}|{_normalize(from_state)}|{_normalize(to_state)}"
        if key in seen_rxn_keys:
            continue
        seen_rxn_keys.add(key)

        modifiers = []
        for trec in _safe_list(tr.get("transporters")):
            if not isinstance(trec, dict):
                continue
            actor = ""
            for field in ["protein", "protein_complex", "name"]:
                v = (trec.get(field) or "").strip()
                if v:
                    actor = v
                    break
            if not actor:
                continue
            stype, rec = _resolve_entity(actor)
            if rec:
                modifiers.append({"stype": stype, "rec": rec})

        cargo_disp = f"{_html_name(cargo)} (&rarr;)"
        tr_name    = (f"{cargo_disp} Transport: {from_state} to {to_state}")
        pw_id      = rxn_ctr.next()
        comp_sbml  = state_name_to_sbml.get(from_state, _register_state(from_state))

        reaction_plans.append({
            "pw_id": pw_id,
            "sbml_id": f"Transport{pw_id}",
            "name": tr_name,
            "compartment": comp_sbml,
            "inputs":  [cargo],
            "outputs": [cargo],
            "modifiers": modifiers,
            "kind": "transport",
            "reversible": False,
            "sbo": "SBO:0000185",
            "from_state": from_state,
            "to_state":   to_state,
        })

    # Interactions (activations, inhibitions, etc.)
    for inter in _safe_list(processes.get("interactions")):
        if not isinstance(inter, dict):
            continue
        actor_name  = (inter.get("actor") or inter.get("from") or "").strip()
        target_name = (inter.get("target") or inter.get("to") or "").strip()
        if not actor_name or not target_name:
            continue
        key = f"interaction|{_normalize(actor_name)}|{_normalize(target_name)}"
        if key in seen_rxn_keys:
            continue
        seen_rxn_keys.add(key)

        state_name = (inter.get("biological_state") or "").strip() or default_compartment_name
        compartment = _register_state(state_name)

        iname = inter.get("name") or f"{actor_name} activates {target_name}"
        pw_id = rxn_ctr.next()
        reaction_plans.append({
            "pw_id": pw_id,
            "sbml_id": f"Interaction{pw_id}",
            "name": iname,
            "compartment": compartment,
            "inputs":  [actor_name],
            "outputs": [target_name],
            "modifiers": [],
            "kind": "interaction",
            "reversible": False,
            "sbo": "SBO:0000342",
        })

    # Sub-pathways
    for sp in _safe_list(processes.get("sub_pathways")):
        if not isinstance(sp, dict):
            continue
        sp_name = (sp.get("name") or "").strip()
        if not sp_name:
            continue
        key = f"subpathway|{_normalize(sp_name)}"
        if key in seen_rxn_keys:
            continue
        seen_rxn_keys.add(key)

        outputs = [x.strip() for x in _safe_list(sp.get("outputs"))
                   if isinstance(x, str) and x.strip()]
        compartment = _register_state(
            (sp.get("biological_state") or "").strip() or default_compartment_name)
        pw_id = rxn_ctr.next()
        reaction_plans.append({
            "pw_id": pw_id,
            "sbml_id": f"SubPathway{pw_id}",
            "name": sp_name,
            "compartment": compartment,
            "inputs":  [],
            "outputs": outputs,
            "modifiers": [],
            "kind": "sub_pathway",
            "reversible": False,
            "sbo": "SBO:0000375",
            "sp_link_id": sp.get("link_id"),
        })

    # ── Layout ────────────────────────────────────────────────────────────────
    _species_pos, _rxn_centers = _build_pathway_layout(reaction_plans)

    def _get_node_pos(
        name: str, fallback_side: str, fallback_idx: int, fallback_total: int,
        cx: float, cy: float,
    ) -> Tuple[float, float, float, float]:
        """Return (node_cx, node_cy, node_x, node_y)."""
        nn = _normalize(name)
        if nn in _species_pos:
            ncx, ncy = _species_pos[nn]
            return ncx, ncy, ncx - _COMPOUND_W / 2, ncy - _COMPOUND_H / 2
        # Fallback: place left/right of reaction centre
        sign = -1 if fallback_side == "reactant" else 1
        offset_y = (fallback_idx - (fallback_total - 1) / 2) * 110.0
        ncx = cx + sign * 280.0
        ncy = cy + offset_y
        return ncx, ncy, ncx - _COMPOUND_W / 2, ncy - _COMPOUND_H / 2

    # ── Write SBML ─────────────────────────────────────────────────────────────
    L: List[str] = []
    a = L.append

    a("<?xml version='1.0' encoding='UTF-8' standalone='no'?>")
    a('<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"'
      ' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"'
      ' xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
      ' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
      ' level="3" version="1">')

    model_mid = _new_uuid()
    a(f'  <model metaid="{model_mid}" id="PathwayModel">')
    a('    <annotation>')
    a(f'  <pathwhiz:dimensions xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
      f' pathwhiz:width="{int(CANVAS_W)}" pathwhiz:height="{int(CANVAS_H)}"/>')
    a('    </annotation>')

    # Unit
    a('      <listOfUnitDefinitions>')
    a('      <unitDefinition name="Mole" id="Unit1">')
    a('        <listOfUnits>')
    a('          <unit scale="0" kind="mole" multiplier="1" exponent="1"/>')
    a('        </listOfUnits>')
    a('      </unitDefinition>')
    a('    </listOfUnitDefinitions>')

    # Compartments
    a('    <listOfCompartments>')
    for nm, sbml_id in sorted(state_name_to_sbml.items(), key=lambda kv: kv[1]):
        pw_id = state_name_to_id[nm]
        a(f'      <compartment name="{_html_name(nm)}" constant="false"'
          f' id="{sbml_id}" sboTerm="SBO:0000240">')
        a('        <annotation>')
        a(f'  <pathwhiz:compartment xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:compartment_id="{pw_id}"'
          f' pathwhiz:compartment_type="biological_state"/>')
        a('        </annotation>')
        a('            </compartment>')
    a('    </listOfCompartments>')

    # ── Species ───────────────────────────────────────────────────────────────
    a('    <listOfSpecies>')
    emitted_proteins: Set[str] = set()

    def _emit_protein(prec: Dict, lines: List[str]) -> None:
        if prec["sbml_id"] in emitted_proteins:
            return
        emitted_proteins.add(prec["sbml_id"])
        mid = _new_uuid()
        comp = _compartment_sbml(prec["norm"])
        uniprot = prec["mapped_ids"].get("uniprot", "Unknown") or "Unknown"
        lines.append(
            f'      <species boundaryCondition="false" constant="false"'
            f' substanceUnits="Unit1" metaid="{mid}"'
            f' hasOnlySubstanceUnits="true" initialAmount="1"'
            f' sboTerm="SBO:0000245" compartment="{comp}"'
            f' name="{_html_name(prec["name"])}" id="{prec["sbml_id"]}">')
        lines.append('        <annotation>')
        lines.append(
            f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
            f' pathwhiz:species_id="{prec["pw_id"]}" pathwhiz:species_type="protein"/>')
        for rl in _rdf_xrefs(mid, prec["mapped_ids"]):
            lines.append(rl)
        lines.append('\t</annotation>')
        lines.append('            </species>')

    # Protein complexes
    for crec in sorted(complex_by_norm.values(), key=lambda r: r["pw_id"]):
        mid   = _new_uuid()
        comp  = _compartment_sbml(crec["norm"])
        member_ids: List[str] = []
        for cn in crec.get("components", []):
            _, prec = _resolve_entity(cn)
            if prec:
                member_ids.append(prec["sbml_id"])
        if not member_ids:
            unk_id = prot_ctr.next()
            unk = {"name": "Unknown", "norm": "unknown", "pw_id": unk_id,
                   "sbml_id": f"Protein{unk_id}", "mapped_ids": {}, "stype": "protein"}
            protein_by_norm["unknown"] = unk
            member_ids.append(unk["sbml_id"])

        uniprots = []
        for pid in member_ids:
            pr = next((r for r in protein_by_norm.values() if r["sbml_id"] == pid), None)
            uniprots.append((pr["mapped_ids"].get("uniprot") or "Unknown") if pr else "Unknown")

        a(f'      <species boundaryCondition="false" constant="false"'
          f' substanceUnits="Unit1" metaid="{mid}"'
          f' hasOnlySubstanceUnits="true" initialAmount="1"'
          f' sboTerm="SBO:0000245" compartment="{comp}"'
          f' name="{_html_name(crec["name"])}" id="{crec["sbml_id"]}">')
        a('        <annotation>')
        pw_ann = (
            f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
            f' pathwhiz:species_id="{crec["pw_id"]}"'
            f' pathwhiz:species_type="protein_complex">\n'
            f'            <pathwhiz:protein_associations>\n'
            f'              <pathwhiz:protein_complex_proteins>'
        )
        for pid in member_ids:
            pw_ann += f"\n                <pathwhiz:protein>{pid}</pathwhiz:protein>"
        pw_ann += ("</pathwhiz:protein_complex_proteins>"
                   "</pathwhiz:protein_associations></pathwhiz:species>")
        a(pw_ann)
        for rl in _rdf_has_part(mid, uniprots):
            a(rl)
        a('\t</annotation>')
        a('            </species>')

        # Emit member proteins
        for pid in member_ids:
            pr = next((r for r in protein_by_norm.values() if r["sbml_id"] == pid), None)
            if pr:
                _emit_protein(pr, L)

    # Standalone proteins
    for prec in sorted(protein_by_norm.values(), key=lambda r: r["pw_id"]):
        _emit_protein(prec, L)

    # Element collections
    for erec in sorted(ecoll_by_norm.values(), key=lambda r: r["pw_id"]):
        mid  = _new_uuid()
        comp = _compartment_sbml(erec["norm"])
        a(f'      <species boundaryCondition="false" constant="false"'
          f' substanceUnits="Unit1" metaid="{mid}"'
          f' hasOnlySubstanceUnits="true" initialAmount="1"'
          f' sboTerm="SBO:0000247" compartment="{comp}"'
          f' name="{_html_name(erec["name"])}" id="{erec["sbml_id"]}">')
        a('        <annotation>')
        a(f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:species_id="{erec["pw_id"]}" pathwhiz:species_type="element_collection"/>')
        a('        </annotation>')
        a('            </species>')

    # Compounds
    for crec in sorted(compound_by_norm.values(), key=lambda r: r["pw_id"]):
        mid  = _new_uuid()
        comp = _compartment_sbml(crec["norm"])
        a(f'      <species boundaryCondition="false" constant="false"'
          f' substanceUnits="Unit1" metaid="{mid}"'
          f' hasOnlySubstanceUnits="true" initialAmount="1"'
          f' sboTerm="SBO:0000247" compartment="{comp}"'
          f' name="{_html_name(crec["name"])}" id="{crec["sbml_id"]}">')
        a('        <annotation>')
        a(f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:species_id="{crec["pw_id"]}" pathwhiz:species_type="compound"/>')
        for rl in _rdf_xrefs(mid, crec["mapped_ids"]):
            a(rl)
        a('\t</annotation>')
        a('            </species>')
        if crec["mapped_ids"]:
            report["pathwhiz_id_stats"]["compounds_matched"] += 1
        else:
            report["pathwhiz_id_stats"]["species_no_id"] += 1

    a('    </listOfSpecies>')

    # ── Reactions ─────────────────────────────────────────────────────────────
    a('    <listOfReactions>')

    for rxn_idx, plan in enumerate(reaction_plans):
        cx, cy = _rxn_centers.get(rxn_idx, (300.0 + rxn_idx * 420.0, 600.0))
        kind       = plan["kind"]
        reversible = plan.get("reversible", False)
        sbo        = plan["sbo"]
        sbml_id    = plan["sbml_id"]
        pw_id      = plan["pw_id"]

        # Sub-pathway: annotation goes on the reaction element itself
        if kind == "sub_pathway":
            sp_link = plan.get("sp_link_id") or pw_id
            # The reaction square position (cx, cy) acts as the sub_pathway box
            sx, sy = cx - 75, cy - 35
            a(f'      <reaction fast="false" reversible="false" sboTerm="{sbo}"'
              f' compartment="{plan["compartment"]}"'
              f' name="{_html_name(plan["name"])}" id="{sbml_id}">')
            a('        <annotation>')
            a(f'  <pathwhiz:reaction xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
              f' pathwhiz:reaction_id="{pw_id}" pathwhiz:reaction_type="sub_pathway">')
            a(f'            <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
              f' pathwhiz:location_type="sub_pathway">')
            a(f'              <pathwhiz:location_element'
              f' pathwhiz:element_type="sub_pathway"'
              f' pathwhiz:element_id="SubPathway{sp_link}"'
              f' pathwhiz:location_id="{pw_id}"'
              f' pathwhiz:x="{sx:.1f}" pathwhiz:y="{sy:.1f}"'
              f' pathwhiz:visualization_template_id="14"'
              f' pathwhiz:width="150.0" pathwhiz:height="70.0"'
              f' pathwhiz:zindex="16" pathwhiz:hidden="false"/>'
              f'</pathwhiz:location></pathwhiz:reaction>')
            a('        </annotation>')
            # Products (output species linked to the sub-pathway box)
            if plan["outputs"]:
                a('              <listOfProducts>')
                n_out = len(plan["outputs"])
                for j, out_name in enumerate(plan["outputs"]):
                    stype2, rec2 = _resolve_entity(out_name)
                    if rec2 is None:
                        continue
                    ncx, ncy, nx, ny = _get_node_pos(out_name, "product", j, n_out, cx, cy)
                    path   = _edge_path(cx + 75, cy, ncx, ncy)
                    loc_id = loc_ctr.next()
                    loc_type = rec2.get("stype", "compound")
                    a(f'          <speciesReference stoichiometry="1" constant="false"'
                      f' species="{rec2["sbml_id"]}" sboTerm="SBO:0000011">')
                    a('            <annotation>')
                    a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                      f' pathwhiz:location_type="{loc_type}">')
                    le_comp = _location_element_compound(rec2["sbml_id"], loc_id, nx, ny)
                    le_edge = _location_element_edge(path, _arrow_options())
                    a(f'                {le_comp}{le_edge}</pathwhiz:location>')
                    a('            </annotation>')
                    a('                    </speciesReference>')
                a('        </listOfProducts>')
            a('      </reaction>')
            continue

        # Normal reaction / transport / interaction
        a(f'      <reaction fast="false"'
          f' reversible="{str(reversible).lower()}"'
          f' sboTerm="{sbo}"'
          f' compartment="{plan["compartment"]}"'
          f' name="{_html_name(plan["name"])}" id="{sbml_id}">')
        a('        <annotation>')
        a(f'  <pathwhiz:reaction xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:reaction_id="{pw_id}" pathwhiz:reaction_type="{kind}"/>')
        a('        </annotation>')

        # Reactant sboTerm: SBO:0000010 for transports, SBO:0000015 for reactions
        reactant_sbo = "SBO:0000010" if kind == "transport" else "SBO:0000015"
        from_state   = plan.get("from_state", "")
        to_state     = plan.get("to_state", "")

        # listOfReactants
        inputs = plan["inputs"]
        if inputs:
            a('              <listOfReactants>')
            n_in = len(inputs)
            for j, inp_name in enumerate(inputs):
                stype2, rec2 = _resolve_entity(inp_name)
                if rec2 is None:
                    continue
                ncx, ncy, nx, ny = _get_node_pos(inp_name, "reactant", j, n_in, cx, cy)
                path   = _edge_path(ncx, ncy, cx, cy)
                loc_id = loc_ctr.next()
                loc_type = rec2.get("stype", "compound")

                a(f'          <speciesReference stoichiometry="1" constant="false"'
                  f' species="{rec2["sbml_id"]}" sboTerm="{reactant_sbo}">')
                a('            <annotation>')
                a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                  f' pathwhiz:location_type="{loc_type}">')

                if loc_type == "element_collection":
                    le_comp = _location_element_ecollection(rec2["sbml_id"], loc_id, nx, ny)
                    le_edge = _location_element_edge(path)
                    a(f'                {le_comp}{le_edge}</pathwhiz:location>')
                elif loc_type == "protein_complex":
                    # protein_complex as a reactant: wrap in protein_complex_visualization
                    a(f'                <pathwhiz:location_element'
                      f' pathwhiz:element_type="protein_complex_visualization"'
                      f' pathwhiz:element_id="{rec2["sbml_id"]}">')
                    for cn in rec2.get("components", []):
                        _, prec = _resolve_entity(cn)
                        if prec:
                            ploc = loc_ctr.next()
                            a(f'                  {_location_element_protein(prec["sbml_id"], ploc, nx, ny)}')
                    a(f'</pathwhiz:location_element>'
                      f'{_location_element_edge(path)}</pathwhiz:location>')
                else:
                    le_comp = _location_element_compound(rec2["sbml_id"], loc_id, nx, ny)
                    le_edge = _location_element_edge(path)
                    a(f'                {le_comp}{le_edge}</pathwhiz:location>')

                a('            </annotation>')
                a('                    </speciesReference>')
            a('        </listOfReactants>')

        # listOfProducts
        outputs = plan["outputs"]
        if outputs:
            a('        <listOfProducts>')
            n_out = len(outputs)
            for j, out_name in enumerate(outputs):
                stype2, rec2 = _resolve_entity(out_name)
                if rec2 is None:
                    continue
                ncx, ncy, nx, ny = _get_node_pos(out_name, "product", j, n_out, cx, cy)
                path     = _edge_path(cx, cy, ncx, ncy)
                opts     = _arrow_options()
                loc_id   = loc_ctr.next()
                loc_type = rec2.get("stype", "compound")

                a(f'          <speciesReference stoichiometry="1" constant="false"'
                  f' species="{rec2["sbml_id"]}" sboTerm="SBO:0000011">')
                a('            <annotation>')
                a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                  f' pathwhiz:location_type="{loc_type}">')

                if loc_type == "element_collection":
                    le_comp = _location_element_ecollection(rec2["sbml_id"], loc_id, nx, ny)
                    le_edge = _location_element_edge(path, opts)
                    a(f'                {le_comp}{le_edge}</pathwhiz:location>')
                elif loc_type == "protein_complex":
                    a(f'                <pathwhiz:location_element'
                      f' pathwhiz:element_type="protein_complex_visualization"'
                      f' pathwhiz:element_id="{rec2["sbml_id"]}">')
                    for cn in rec2.get("components", []):
                        _, prec = _resolve_entity(cn)
                        if prec:
                            ploc = loc_ctr.next()
                            a(f'                  {_location_element_protein(prec["sbml_id"], ploc, nx, ny)}')
                    a(f'</pathwhiz:location_element>'
                      f'{_location_element_edge(path, opts)}</pathwhiz:location>')
                else:
                    le_comp = _location_element_compound(rec2["sbml_id"], loc_id, nx, ny)
                    le_edge = _location_element_edge(path, opts)
                    a(f'                {le_comp}{le_edge}</pathwhiz:location>')

                a('            </annotation>')
                a('                    </speciesReference>')
            a('        </listOfProducts>')

        # listOfModifiers
        modifiers = plan.get("modifiers", [])
        if modifiers:
            # Modifier sboTerm: SBO:0000019 for transporters, SBO:0000460 for enzymes
            mod_sbo = "SBO:0000019" if kind == "transport" else "SBO:0000460"
            a('        <listOfModifiers>')
            mod_positions = _modifier_positions(cx, cy, len(modifiers))
            for j, mod in enumerate(modifiers):
                mrec     = mod["rec"]
                mstype   = mod["stype"]
                _, mod_cy, mod_x, mod_y = mod_positions[j]
                loc_id   = loc_ctr.next()

                a(f'          <modifierSpeciesReference'
                  f' species="{mrec["sbml_id"]}" sboTerm="{mod_sbo}">')
                a('            <annotation>')
                a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                  f' pathwhiz:location_type="{mstype}">')

                if mstype == "protein_complex":
                    a(f'                <pathwhiz:location_element'
                      f' pathwhiz:element_type="protein_complex_visualization"'
                      f' pathwhiz:element_id="{mrec["sbml_id"]}">')
                    for cn in mrec.get("components", []):
                        _, prec = _resolve_entity(cn)
                        if prec:
                            ploc = loc_ctr.next()
                            a(f'                  {_location_element_protein(prec["sbml_id"], ploc, mod_x, mod_y)}')
                            mod_x += _PROTEIN_W + 5  # offset each subunit
                    a('</pathwhiz:location_element></pathwhiz:location>')
                else:
                    a(f'                {_location_element_protein(mrec["sbml_id"], loc_id, mod_x, mod_y)}'
                      f'</pathwhiz:location>')

                a('            </annotation>')
                a('                    </modifierSpeciesReference>')
            a('        </listOfModifiers>')

        a('      </reaction>')

    a('    </listOfReactions>')
    a('  </model>')
    a('</sbml>')

    sbml_text = "\n".join(L)
    sbml_path.write_text(sbml_text, encoding="utf-8")

    report["counts"]["compartments"] = len(state_name_to_id)
    report["counts"]["species"]      = (len(compound_by_norm) + len(protein_by_norm)
                                        + len(complex_by_norm) + len(ecoll_by_norm))
    report["counts"]["reactions"]    = len(reaction_plans)

    report_json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_txt_path.write_text(
        "\n".join([
            f"SBML: {sbml_path}",
            f"Compartments: {report['counts']['compartments']}",
            f"Species:      {report['counts']['species']}",
            f"Reactions:    {report['counts']['reactions']}",
            f"Hard errors:  {len(report['hard_errors'])}",
            f"Warnings:     {len(report['warnings'])}",
        ]), encoding="utf-8")

    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build PathWhiz-compatible SBML from mapped pathway JSON.")
    ap.add_argument("--in",          dest="input_path",       required=True)
    ap.add_argument("--out",         dest="sbml_path",        default="pathway.sbml")
    ap.add_argument("--report-json", dest="report_json_path", default="sbml_report.json")
    ap.add_argument("--report-txt",  dest="report_txt_path",  default="sbml_report.txt")
    ap.add_argument("--default-compartment", dest="default_compartment", default="cell")
    args = ap.parse_args()

    report = build_sbml(
        Path(args.input_path),
        Path(args.sbml_path),
        Path(args.report_json_path),
        Path(args.report_txt_path),
        default_compartment_name=args.default_compartment,
    )
    print(f"SBML written: {args.sbml_path}")
    print(f"  Compartments {report['counts']['compartments']} | "
          f"Species {report['counts']['species']} | "
          f"Reactions {report['counts']['reactions']}")
    if report["hard_errors"]:
        print(f"  ERRORS: {len(report['hard_errors'])}")
        for e in report["hard_errors"][:5]:
            print(f"    {e}")


if __name__ == "__main__":
    main()
