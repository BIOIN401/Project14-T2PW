from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ── Layout engine (circular TCA-aware layout) ─────────────────────────────────
from layout_engine import build_pathway_layout as _build_pathway_layout

def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", "", lowered)


def _short_hash(value: str, length: int = 8) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# PathWhiz-style layout constants
# ---------------------------------------------------------------------------
_CANVAS_W = 2400.0
_CANVAS_H = 2400.0
_MARGIN = 120.0
_REACTION_GAP_X = 420.0
_REACTION_GAP_Y = 380.0
_NODE_OFFSET_X = 200.0
_NODE_SPACING_Y = 110.0
_MODIFIER_OFFSET_Y = 120.0

_COFACTOR_NORMS_LAYOUT: Set[str] = {
    "nad", "nad+", "nadh", "nadp", "nadp+", "nadph",
    "fad", "fadh", "fadh2",
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "ctp", "utp",
    "h2o", "water", "h+", "h", "proton", "oh",
    "o2", "oxygen", "co2", "carbon dioxide",
    "pi", "ppi", "pyrophosphate", "phosphate", "inorganic phosphate",
    "coa", "coa sh", "coash", "coa-sh", "coenzyme a", "acetyl coa",
    "hco3", "bicarbonate",
    "ubiquinone", "ubiquinol", "ubiquinone q", "ubiquinol qh2",
    "h2o2", "hydrogen peroxide",
    "fe2+", "fe3+", "cu+", "cu2+", "zn2+", "mg2+", "mn2+", "ca2+",
}

def _is_layout_cofactor(name: str) -> bool:
    n = _normalize(name)
    return not n or len(n) <= 1 or n in _COFACTOR_NORMS_LAYOUT


def _is_cofactor(name: str) -> bool:
    return _is_layout_cofactor(name)


# ---------------------------------------------------------------------------
# ID allocation
# ---------------------------------------------------------------------------
class _IdCounter:
    def __init__(self, start: int = 1) -> None:
        self._n = start

    def next(self) -> int:
        v = self._n
        self._n += 1
        return v


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
_COMPOUND_W = 78.0
_COMPOUND_H = 78.0
_PROTEIN_W = 160.0
_PROTEIN_H = 80.0
_TMPL_COMPOUND = "3"
_TMPL_EDGE = "5"
_TMPL_PROTEIN = "6"


def _reaction_center(rxn_idx: int) -> Tuple[float, float]:
    cx = _MARGIN + _NODE_OFFSET_X + rxn_idx * _REACTION_GAP_X
    cy = _MARGIN + _MODIFIER_OFFSET_Y
    return cx, cy


def _node_pos_for_side(
    cx: float, cy: float, idx: int, total: int, side: str
) -> Tuple[float, float, float, float]:
    if side == "reactant":
        x_center = cx - _NODE_OFFSET_X
    else:
        x_center = cx + _NODE_OFFSET_X
    y_offset = (idx - (total - 1) / 2.0) * _NODE_SPACING_Y
    node_cy = cy + y_offset
    node_x = x_center - _COMPOUND_W / 2
    node_y = node_cy - _COMPOUND_H / 2
    return x_center, node_cy, node_x, node_y


def _edge_path_between(x1: float, y1: float, x2: float, y2: float) -> str:
    dx = x2 - x1
    dy = y2 - y1
    ctrl_x = (x1 + x2) / 2
    ctrl_y = (y1 + y2) / 2
    return (f"M{x1:.1f} {y1:.1f} "
            f"C{ctrl_x:.1f} {y1:.1f} {ctrl_x:.1f} {y2:.1f} "
            f"{x2:.1f} {y2:.1f} ")


def _edge_path_reactant(node_cx: float, node_cy: float, cx: float, cy: float) -> str:
    return _edge_path_between(node_cx, node_cy, cx, cy)


def _edge_path_product(cx: float, cy: float, node_cx: float, node_cy: float) -> str:
    return _edge_path_between(cx, cy, node_cx, node_cy)


def _modifier_pos(cx: float, cy: float, idx: int, total: int) -> Tuple[float, float, float, float]:
    x_off = (idx - (total - 1) / 2.0) * (_PROTEIN_W + 20.0)
    node_cx = cx + x_off
    node_cy = cy - _MODIFIER_OFFSET_Y
    node_x = node_cx - _PROTEIN_W / 2
    node_y = node_cy - _PROTEIN_H / 2
    return node_cx, node_cy, node_x, node_y


def _html_name(name: str) -> str:
    return name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# Core SBML builder
# ---------------------------------------------------------------------------

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
        raise ValueError("Input JSON must be an object.")

    report: Dict[str, Any] = {
        "hard_errors": [],
        "warnings": [],
        "defaults_applied": [],
        "counts": {"compartments": 0, "species": 0, "reactions": 0},
        "pathwhiz_id_stats": {
            "mysql_connected": False,
            "compounds_matched": 0, "proteins_matched": 0, "species_no_id": 0
        },
        "validation": {
            "check_count": 0,
            "error_count": 0,
            "has_errors": False,
            "messages": [],
        },
    }

    data = deepcopy(payload)
    entities = _safe_dict(data.get("entities"))
    processes = _safe_dict(data.get("processes"))

    # ------------------------------------------------------------------
    # ID counters
    # ------------------------------------------------------------------
    state_id_ctr    = _IdCounter(10000)
    compound_id_ctr = _IdCounter(20000)
    protein_id_ctr  = _IdCounter(30000)
    complex_id_ctr  = _IdCounter(40000)
    reaction_id_ctr = _IdCounter(50000)
    location_id_ctr = _IdCounter(100000)

    # ------------------------------------------------------------------
    # 1. Compartments
    # ------------------------------------------------------------------
    states_raw = _safe_list(data.get("biological_states"))
    state_name_to_id:   Dict[str, int] = {}
    state_name_to_sbml: Dict[str, str] = {}
    state_display:      Dict[str, str] = {}

    def _register_state(name: str) -> str:
        nm = name.strip()
        if not nm:
            nm = default_compartment_name
        if nm not in state_name_to_id:
            sid = state_id_ctr.next()
            state_name_to_id[nm]   = sid
            state_name_to_sbml[nm] = f"BiologicalState{sid}"
            state_display[nm]      = nm
        return state_name_to_sbml[nm]

    for bs in states_raw:
        if not isinstance(bs, dict):
            continue
        nm = (bs.get("name") or "").strip()
        if nm:
            _register_state(nm)

    _register_state(default_compartment_name)

    # ------------------------------------------------------------------
    # 2. Entity registries
    # ------------------------------------------------------------------
    compound_rows = _safe_list(entities.get("compounds"))
    protein_rows  = _safe_list(entities.get("proteins"))
    complex_rows  = _safe_list(entities.get("protein_complexes"))

    compound_by_norm: Dict[str, Dict[str, Any]] = {}
    protein_by_norm:  Dict[str, Dict[str, Any]] = {}
    complex_by_norm:  Dict[str, Dict[str, Any]] = {}

    def _register_compound(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        name = (row.get("name") or "").strip()
        if not name:
            return None
        norm = _normalize(name)
        if norm in compound_by_norm:
            return compound_by_norm[norm]
        pw_id = None
        for k in ["pathbank_compound_id", "pw_compound_id"]:
            v = row.get(k) or _safe_dict(row.get("mapping_meta")).get(k)
            if v:
                try:
                    pw_id = int(v); break
                except (ValueError, TypeError):
                    pass
        if pw_id is None:
            pw_id = compound_id_ctr.next()
        sbml_id = f"Compound{pw_id}"
        mapped = _safe_dict(row.get("mapped_ids"))
        rec = {"name": name, "norm": norm, "pw_id": pw_id,
               "sbml_id": sbml_id, "mapped_ids": mapped, "row": row}
        compound_by_norm[norm] = rec
        return rec

    def _register_protein(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        name = (row.get("name") or "").strip()
        if not name:
            return None
        norm = _normalize(name)
        if norm in protein_by_norm:
            return protein_by_norm[norm]
        pw_id = None
        for k in ["pathbank_protein_id", "pw_protein_id"]:
            v = row.get(k) or _safe_dict(row.get("mapping_meta")).get(k)
            if v:
                try:
                    pw_id = int(v); break
                except (ValueError, TypeError):
                    pass
        if pw_id is None:
            pw_id = protein_id_ctr.next()
        sbml_id = f"Protein{pw_id}"
        mapped = _safe_dict(row.get("mapped_ids"))
        rec = {"name": name, "norm": norm, "pw_id": pw_id,
               "sbml_id": sbml_id, "mapped_ids": mapped, "row": row}
        protein_by_norm[norm] = rec
        return rec

    def _register_complex(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        name = (row.get("name") or "").strip()
        if not name:
            return None
        norm = _normalize(name)
        if norm in complex_by_norm:
            return complex_by_norm[norm]
        pw_id = None
        for k in ["pathbank_complex_id", "pw_complex_id"]:
            v = row.get(k) or _safe_dict(row.get("mapping_meta")).get(k)
            if v:
                try:
                    pw_id = int(v); break
                except (ValueError, TypeError):
                    pass
        if pw_id is None:
            pw_id = complex_id_ctr.next()
        sbml_id = f"ProteinComplex{pw_id}"
        components = [c.strip() for c in _safe_list(row.get("components"))
                      if isinstance(c, str) and c.strip()]
        mapped = _safe_dict(row.get("mapped_ids"))
        rec = {"name": name, "norm": norm, "pw_id": pw_id,
               "sbml_id": sbml_id, "components": components,
               "mapped_ids": mapped, "row": row}
        complex_by_norm[norm] = rec
        return rec

    for row in compound_rows:
        if isinstance(row, dict):
            _register_compound(row)
    for row in protein_rows:
        if isinstance(row, dict):
            _register_protein(row)
    for row in complex_rows:
        if isinstance(row, dict):
            _register_complex(row)

    _protein_component_norms: Set[str] = set()
    for crec_pre in list(complex_by_norm.values()):
        for comp_name in _safe_list(crec_pre.get("components", [])):
            if isinstance(comp_name, str) and comp_name.strip():
                _protein_component_norms.add(_normalize(comp_name))

    def _resolve_entity(name: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        norm = _normalize(name)
        if norm in complex_by_norm:
            return "complex", complex_by_norm[norm]
        if norm in protein_by_norm:
            return "protein", protein_by_norm[norm]
        if norm in compound_by_norm:
            return "compound", compound_by_norm[norm]
        if norm in _protein_component_norms:
            report["warnings"].append({"path": "/entities",
                "reason": f"Unknown entity '{name}' auto-registered as protein"})
            new_id = protein_id_ctr.next()
            rec = {"name": name, "norm": norm, "pw_id": new_id,
                   "sbml_id": f"Protein{new_id}", "mapped_ids": {}, "row": {}}
            protein_by_norm[norm] = rec
            return "protein", rec
        report["warnings"].append({"path": "/entities",
            "reason": f"Unknown entity '{name}' auto-registered as compound"})
        new_id = compound_id_ctr.next()
        rec = {"name": name, "norm": norm, "pw_id": new_id,
               "sbml_id": f"Compound{new_id}", "mapped_ids": {}, "row": {}}
        compound_by_norm[norm] = rec
        return "compound", rec

    # ------------------------------------------------------------------
    # 3. Compartment assignment
    # ------------------------------------------------------------------
    element_locs = _safe_dict(data.get("element_locations"))
    entity_states: Dict[str, List[str]] = defaultdict(list)
    for loc_key, entity_field in [
        ("compound_locations",            "compound"),
        ("protein_locations",             "protein"),
        ("element_collection_locations",  "element_collection"),
    ]:
        for row in _safe_list(element_locs.get(loc_key)):
            if not isinstance(row, dict):
                continue
            ename = (row.get(entity_field) or "").strip()
            state = (row.get("biological_state") or "").strip()
            if ename and state:
                entity_states[_normalize(ename)].append(state)

    def _primary_state(entity_norm: str) -> str:
        states = entity_states.get(entity_norm, [])
        return states[0] if states else default_compartment_name

    for states in entity_states.values():
        for s in states:
            _register_state(s)

    def _compartment_sbml_id(entity_norm: str) -> str:
        return _register_state(_primary_state(entity_norm))

    # ------------------------------------------------------------------
    # 4. Build reaction plan
    # ------------------------------------------------------------------
    seen_reaction_keys: Set[str] = set()
    reaction_plans: List[Dict[str, Any]] = []

    def _build_reaction_name(inputs: List[str], outputs: List[str]) -> str:
        lhs = " + ".join(inputs) if inputs else "?"
        rhs = " + ".join(outputs) if outputs else "?"
        return f"{lhs} → {rhs}"

    for rxn in _safe_list(processes.get("reactions")):
        if not isinstance(rxn, dict):
            continue
        inputs  = [x.strip() for x in _safe_list(rxn.get("inputs"))
                   if isinstance(x, str) and x.strip()]
        outputs = [x.strip() for x in _safe_list(rxn.get("outputs"))
                   if isinstance(x, str) and x.strip()]
        if not inputs or not outputs:
            report["hard_errors"].append({"path": "/processes/reactions",
                "reason": "Missing inputs or outputs."})
            continue
        key = "|".join(sorted(inputs)) + "→" + "|".join(sorted(outputs))
        if key in seen_reaction_keys:
            report["warnings"].append({"path": "/processes/reactions",
                "reason": "Duplicate reaction collapsed."})
            continue
        seen_reaction_keys.add(key)

        state_name = (rxn.get("biological_state") or "").strip()
        if not state_name:
            for inp in inputs:
                s = _primary_state(_normalize(inp))
                if s != default_compartment_name:
                    state_name = s
                    break
        if not state_name:
            state_name = default_compartment_name
        compartment_sbml = _register_state(state_name)

        modifiers: List[Dict[str, Any]] = []
        for mod in (_safe_list(rxn.get("modifiers")) +
                    _safe_list(rxn.get("enzymes"))):
            if not isinstance(mod, dict):
                continue
            actor_name = ""
            for field in ["entity", "protein", "protein_complex", "name"]:
                v = (mod.get(field) or "").strip()
                if v:
                    actor_name = v
                    break
            if not actor_name:
                continue
            kind, rec = _resolve_entity(actor_name)
            if rec:
                modifiers.append({"kind": kind, "rec": rec})

        pw_rxn_id = reaction_id_ctr.next()
        rxn_name  = _build_reaction_name(inputs, outputs)

        reaction_plans.append({
            "pw_id":           pw_rxn_id,
            "sbml_id":         f"Reaction{pw_rxn_id}",
            "name":            rxn_name,
            "compartment_sbml": compartment_sbml,
            "inputs":          inputs,
            "outputs":         outputs,
            "modifiers":       modifiers,
            "kind":            "reaction",
        })

    for tr in _safe_list(processes.get("transports")):
        if not isinstance(tr, dict):
            continue
        cargo = (tr.get("cargo") or "").strip()
        if not cargo:
            continue
        from_state = (tr.get("from_biological_state") or "").strip() or default_compartment_name
        to_state   = (tr.get("to_biological_state")   or "").strip() or default_compartment_name
        if from_state == to_state:
            report["warnings"].append({"path": "/processes/transports",
                "reason": f"Degenerate transport for '{cargo}' skipped."})
            continue
        _register_state(from_state)
        _register_state(to_state)

        key = f"transport|{_normalize(cargo)}|{_normalize(from_state)}|{_normalize(to_state)}"
        if key in seen_reaction_keys:
            continue
        seen_reaction_keys.add(key)

        modifiers = []
        for tr_rec in _safe_list(tr.get("transporters")):
            if not isinstance(tr_rec, dict):
                continue
            actor_name = ""
            for field in ["protein", "protein_complex", "name"]:
                v = (tr_rec.get(field) or "").strip()
                if v:
                    actor_name = v
                    break
            if not actor_name:
                continue
            kind, rec = _resolve_entity(actor_name)
            if rec:
                modifiers.append({"kind": kind, "rec": rec})

        pw_rxn_id = reaction_id_ctr.next()
        rxn_name  = f"{cargo} transport: {from_state} → {to_state}"
        reaction_plans.append({
            "pw_id":           pw_rxn_id,
            "sbml_id":         f"Reaction{pw_rxn_id}",
            "name":            rxn_name,
            "compartment_sbml": state_name_to_sbml.get(
                from_state, _register_state(from_state)),
            "inputs":          [cargo],
            "outputs":         [cargo],
            "inputs_state":    from_state,
            "outputs_state":   to_state,
            "modifiers":       modifiers,
            "kind":            "transport",
        })

    # ------------------------------------------------------------------
    # 5. Compute layout via circular engine
    # ------------------------------------------------------------------
    _species_pos, _rxn_centers = _build_pathway_layout(reaction_plans)

    # ------------------------------------------------------------------
    # 6. Write SBML
    # ------------------------------------------------------------------
    lines: List[str] = []
    a = lines.append

    a("<?xml version='1.0' encoding='UTF-8' standalone='no'?>")
    a('<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"'
      ' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"'
      ' xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
      ' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
      ' level="3" version="1">')

    model_metaid = _new_uuid()
    a(f'  <model metaid="{model_metaid}" id="PathwayModel">')
    a('    <annotation>')
    a(f'  <pathwhiz:dimensions xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
      f' pathwhiz:width="{int(_CANVAS_W)}" pathwhiz:height="{int(_CANVAS_H)}"/>')
    a('    </annotation>')
    a('      <listOfUnitDefinitions>')
    a('      <unitDefinition name="Mole" id="Unit1">')
    a('        <listOfUnits>')
    a('          <unit scale="0" kind="mole" multiplier="1" exponent="1"/>')
    a('        </listOfUnits>')
    a('      </unitDefinition>')
    a('    </listOfUnitDefinitions>')

    # Compartments
    a('    <listOfCompartments>')
    for state_name, sbml_id in sorted(state_name_to_sbml.items(),
                                       key=lambda kv: kv[1]):
        pw_id   = state_name_to_id[state_name]
        display = state_display.get(state_name, state_name)
        a(f'      <compartment name="{_html_name(display)}" constant="false"'
          f' id="{sbml_id}" sboTerm="SBO:0000240">')
        a('        <annotation>')
        a(f'  <pathwhiz:compartment xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:compartment_id="{pw_id}"'
          f' pathwhiz:compartment_type="biological_state"/>')
        a('        </annotation>')
        a('            </compartment>')
    a('    </listOfCompartments>')

    # Species
    a('    <listOfSpecies>')

    def _rdf_xrefs(metaid: str, mapped_ids: Dict[str, Any]) -> List[str]:
        DB_URN: Dict[str, str] = {
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
        out = [
            f'\t<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
            f' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">',
            f'\t<rdf:Description rdf:about="#{metaid}">',
            "              <bqbiol:is>", "\t<rdf:Bag>",
        ]
        for uri in uris:
            out.append(f'\t<rdf:li rdf:resource="{uri}"/>')
        out += ["\t</rdf:Bag>", "\t</bqbiol:is>",
                "\t</rdf:Description>", "\t</rdf:RDF>"]
        return out

    def _rdf_hasPart(metaid: str, protein_sbml_ids: List[str]) -> List[str]:
        out = [
            f'\t<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
            f' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">',
            f'\t<rdf:Description rdf:about="#{metaid}">',
            "\t<bqbiol:hasPart>", "\t<rdf:Bag>",
        ]
        for pid in protein_sbml_ids:
            prec = next((r for r in protein_by_norm.values()
                         if r["sbml_id"] == pid), None)
            uniprot = (prec["mapped_ids"].get("uniprot", "Unknown")
                       if prec else "Unknown") or "Unknown"
            out.append(f'\t<rdf:li rdf:resource="urn:miriam:uniprot:{uniprot}"/>')
        out += ["\t</rdf:Bag>", "\t</bqbiol:hasPart>",
                "\t</rdf:Description>", "\t</rdf:RDF>"]
        return out

    emitted_proteins: Set[str] = set()

    # Protein complexes
    for crec in sorted(complex_by_norm.values(), key=lambda r: r["pw_id"]):
        compartment_sbml = _compartment_sbml_id(crec["norm"])
        metaid = _new_uuid()

        member_protein_ids: List[str] = []
        for comp_name in crec["components"]:
            kind, prec = _resolve_entity(comp_name)
            if prec and kind == "protein":
                member_protein_ids.append(prec["sbml_id"])
            elif prec and kind == "complex":
                member_protein_ids.append(prec["sbml_id"])
            else:
                pnorm = _normalize(comp_name)
                if pnorm not in protein_by_norm:
                    pid = protein_id_ctr.next()
                    protein_by_norm[pnorm] = {
                        "name": comp_name, "norm": pnorm, "pw_id": pid,
                        "sbml_id": f"Protein{pid}", "mapped_ids": {}, "row": {},
                    }
                member_protein_ids.append(protein_by_norm[pnorm]["sbml_id"])

        if not member_protein_ids:
            unknown_norm = _normalize("Unknown")
            if unknown_norm not in protein_by_norm:
                uid = protein_id_ctr.next()
                protein_by_norm[unknown_norm] = {
                    "name": "Unknown", "norm": unknown_norm, "pw_id": uid,
                    "sbml_id": f"Protein{uid}", "mapped_ids": {}, "row": {},
                }
            member_protein_ids.append(protein_by_norm[unknown_norm]["sbml_id"])

        a(f'      <species boundaryCondition="false" constant="false"'
          f' substanceUnits="Unit1" metaid="{metaid}"'
          f' hasOnlySubstanceUnits="true" initialAmount="1"'
          f' sboTerm="SBO:0000245" compartment="{compartment_sbml}"'
          f' name="{_html_name(crec["name"])}" id="{crec["sbml_id"]}">')
        a('        <annotation>')
        pw_ann = (
            f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
            f' pathwhiz:species_id="{crec["pw_id"]}" pathwhiz:species_type="protein_complex">'
            f'\n            <pathwhiz:protein_associations>'
            f'\n              <pathwhiz:protein_complex_proteins>'
        )
        for pid_str in member_protein_ids:
            pw_ann += f"\n                <pathwhiz:protein>{pid_str}</pathwhiz:protein>"
        pw_ann += ('</pathwhiz:protein_complex_proteins>'
                   '</pathwhiz:protein_associations>'
                   '</pathwhiz:species>')
        a(pw_ann)
        for rl in _rdf_hasPart(metaid, member_protein_ids):
            a(rl)
        a('\t</annotation>')
        a('            </species>')

        for pid_str in member_protein_ids:
            if pid_str in emitted_proteins:
                continue
            emitted_proteins.add(pid_str)
            prec = next((r for r in protein_by_norm.values()
                         if r["sbml_id"] == pid_str), None)
            if prec is None:
                continue
            p_metaid = _new_uuid()
            p_comp   = _compartment_sbml_id(prec["norm"])
            uniprot  = prec["mapped_ids"].get("uniprot", "Unknown") or "Unknown"
            a(f'      <species boundaryCondition="false" constant="false"'
              f' substanceUnits="Unit1" metaid="{p_metaid}"'
              f' hasOnlySubstanceUnits="true" initialAmount="1"'
              f' sboTerm="SBO:0000245" compartment="{p_comp}"'
              f' name="{_html_name(prec["name"])}" id="{prec["sbml_id"]}">')
            a('        <annotation>')
            a(f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
              f' pathwhiz:species_id="{prec["pw_id"]}" pathwhiz:species_type="protein"/>')
            a(f'\t<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
              f' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">')
            a(f'\t<rdf:Description rdf:about="#{p_metaid}">')
            a('\t<bqbiol:is>\n\t<rdf:Bag>')
            a(f'\t<rdf:li rdf:resource="urn:miriam:uniprot:{uniprot}"/>')
            a('\t</rdf:Bag>\n\t</bqbiol:is>')
            a('\t</rdf:Description>\n\t</rdf:RDF>')
            a('\t</annotation>')
            a('            </species>')

    # Standalone proteins
    for prec in sorted(protein_by_norm.values(), key=lambda r: r["pw_id"]):
        if prec["sbml_id"] in emitted_proteins:
            continue
        emitted_proteins.add(prec["sbml_id"])
        p_metaid = _new_uuid()
        p_comp   = _compartment_sbml_id(prec["norm"])
        uniprot  = prec["mapped_ids"].get("uniprot", "Unknown") or "Unknown"
        a(f'      <species boundaryCondition="false" constant="false"'
          f' substanceUnits="Unit1" metaid="{p_metaid}"'
          f' hasOnlySubstanceUnits="true" initialAmount="1"'
          f' sboTerm="SBO:0000245" compartment="{p_comp}"'
          f' name="{_html_name(prec["name"])}" id="{prec["sbml_id"]}">')
        a('        <annotation>')
        a(f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:species_id="{prec["pw_id"]}" pathwhiz:species_type="protein"/>')
        a(f'\t<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
          f' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">')
        a(f'\t<rdf:Description rdf:about="#{p_metaid}">')
        a('\t<bqbiol:is>\n\t<rdf:Bag>')
        a(f'\t<rdf:li rdf:resource="urn:miriam:uniprot:{uniprot}"/>')
        a('\t</rdf:Bag>\n\t</bqbiol:is>')
        a('\t</rdf:Description>\n\t</rdf:RDF>')
        a('\t</annotation>')
        a('            </species>')
        if uniprot and uniprot != "Unknown":
            report["pathwhiz_id_stats"]["proteins_matched"] += 1
        else:
            report["pathwhiz_id_stats"]["species_no_id"] += 1

    # Compounds
    for crec in sorted(compound_by_norm.values(), key=lambda r: r["pw_id"]):
        c_metaid = _new_uuid()
        c_comp   = _compartment_sbml_id(crec["norm"])
        mapped   = crec["mapped_ids"]
        a(f'      <species boundaryCondition="false" constant="false"'
          f' substanceUnits="Unit1" metaid="{c_metaid}"'
          f' hasOnlySubstanceUnits="true" initialAmount="1"'
          f' sboTerm="SBO:0000247" compartment="{c_comp}"'
          f' name="{_html_name(crec["name"])}" id="{crec["sbml_id"]}">')
        a('        <annotation>')
        a(f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:species_id="{crec["pw_id"]}" pathwhiz:species_type="compound"/>')
        for rl in _rdf_xrefs(c_metaid, mapped):
            a(rl)
        a('\t</annotation>')
        a('            </species>')
        if mapped:
            report["pathwhiz_id_stats"]["compounds_matched"] += 1
        else:
            report["pathwhiz_id_stats"]["species_no_id"] += 1

    a('    </listOfSpecies>')

    # ------------------------------------------------------------------
    # Reactions — emit with PathWhiz layout annotations
    # ------------------------------------------------------------------
    a('    <listOfReactions>')

    for rxn_idx, plan in enumerate(reaction_plans):
        # Get reaction centre from layout engine
        cx, cy = _rxn_centers.get(rxn_idx, (300.0 + rxn_idx * 420.0, 300.0))

        def _get_node_pos(name: str, fallback_side: str,
                          fallback_idx: int, fallback_total: int):
            norm = _normalize(name)
            if norm in _species_pos:
                nx, ny = _species_pos[norm]
                return nx, ny, nx - _COMPOUND_W / 2, ny - _COMPOUND_H / 2
            return _node_pos_for_side(cx, cy, fallback_idx, fallback_total,
                                      fallback_side)

        sbml_rxn_id       = plan["sbml_id"]
        pw_rxn_id         = plan["pw_id"]
        compartment_sbml  = plan["compartment_sbml"]
        rxn_name_html     = _html_name(plan["name"])
        sbo = "SBO:0000176" if plan["kind"] == "reaction" else "SBO:0000185"

        a(f'      <reaction fast="false" reversible="false" sboTerm="{sbo}"'
          f' compartment="{compartment_sbml}"'
          f' name="{rxn_name_html}" id="{sbml_rxn_id}">')
        a('        <annotation>')
        a(f'  <pathwhiz:reaction xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:reaction_id="{pw_rxn_id}"'
          f' pathwhiz:reaction_type="{plan["kind"]}"/>')
        a('        </annotation>')

        inputs_state  = plan.get("inputs_state",  "")
        outputs_state = plan.get("outputs_state", "")

        # listOfReactants
        reactants = plan["inputs"]
        if reactants:
            a('              <listOfReactants>')
            n_react = len(reactants)
            for j, inp_name in enumerate(reactants):
                kind2, rec2 = _resolve_entity(inp_name)
                if rec2 is None:
                    continue
                r_comp = (_register_state(inputs_state)
                          if inputs_state
                          else _compartment_sbml_id(rec2["norm"]))
                node_cx, node_cy, node_x, node_y = _get_node_pos(
                    inp_name, "reactant", j, n_react)
                edge_path = _edge_path_reactant(node_cx, node_cy, cx, cy)
                loc_id   = location_id_ctr.next()
                loc_type = ("protein_complex" if kind2 == "complex"
                            else ("protein" if kind2 == "protein" else "compound"))

                a(f'          <speciesReference stoichiometry="1" constant="false"'
                  f' species="{rec2["sbml_id"]}" sboTerm="SBO:0000015">')
                a('            <annotation>')
                a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                  f' pathwhiz:location_type="{loc_type}">')
                a(f'                <pathwhiz:location_element'
                  f' pathwhiz:element_type="compound_location"'
                  f' pathwhiz:element_id="{rec2["sbml_id"]}"'
                  f' pathwhiz:location_id="{loc_id}"'
                  f' pathwhiz:x="{node_x:.1f}" pathwhiz:y="{node_y:.1f}"'
                  f' pathwhiz:visualization_template_id="{_TMPL_COMPOUND}"'
                  f' pathwhiz:width="{_COMPOUND_W:.1f}"'
                  f' pathwhiz:height="{_COMPOUND_H:.1f}"'
                  f' pathwhiz:zindex="10" pathwhiz:hidden="false"/>'
                  f'<pathwhiz:location_element pathwhiz:element_type="edge"'
                  f' pathwhiz:path="{edge_path}"'
                  f' pathwhiz:visualization_template_id="{_TMPL_EDGE}"'
                  f' pathwhiz:options="{{}}"'
                  f' pathwhiz:zindex="18" pathwhiz:hidden="false"/>'
                  f'</pathwhiz:location>')
                a('            </annotation>')
                a('                    </speciesReference>')
            a('        </listOfReactants>')

        # listOfProducts
        products = plan["outputs"]
        if products:
            a('        <listOfProducts>')
            n_prod = len(products)
            arrow_opt_escaped = (
                '{"start_arrow":true,'
                '"start_arrow_path":"M 25.9 13.3 L 11 12 L 17.4 25.6",'
                '"start_flat_arrow":false,"start_flat_arrow_path":null}'
            ).replace('"', "&quot;")
            for j, out_name in enumerate(products):
                kind2, rec2 = _resolve_entity(out_name)
                if rec2 is None:
                    continue
                p_comp = (_register_state(outputs_state)
                          if outputs_state
                          else _compartment_sbml_id(rec2["norm"]))
                node_cx, node_cy, node_x, node_y = _get_node_pos(
                    out_name, "product", j, n_prod)
                edge_path = _edge_path_product(cx, cy, node_cx, node_cy)
                loc_id   = location_id_ctr.next()
                loc_type = ("protein_complex" if kind2 == "complex"
                            else ("protein" if kind2 == "protein" else "compound"))

                a(f'          <speciesReference stoichiometry="1" constant="false"'
                  f' species="{rec2["sbml_id"]}" sboTerm="SBO:0000011">')
                a('            <annotation>')
                a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                  f' pathwhiz:location_type="{loc_type}">')
                a(f'                <pathwhiz:location_element'
                  f' pathwhiz:element_type="compound_location"'
                  f' pathwhiz:element_id="{rec2["sbml_id"]}"'
                  f' pathwhiz:location_id="{loc_id}"'
                  f' pathwhiz:x="{node_x:.1f}" pathwhiz:y="{node_y:.1f}"'
                  f' pathwhiz:visualization_template_id="{_TMPL_COMPOUND}"'
                  f' pathwhiz:width="{_COMPOUND_W:.1f}"'
                  f' pathwhiz:height="{_COMPOUND_H:.1f}"'
                  f' pathwhiz:zindex="10" pathwhiz:hidden="false"/>'
                  f'<pathwhiz:location_element pathwhiz:element_type="edge"'
                  f' pathwhiz:path="{edge_path}"'
                  f' pathwhiz:visualization_template_id="{_TMPL_EDGE}"'
                  f' pathwhiz:options="{arrow_opt_escaped}"'
                  f' pathwhiz:zindex="18" pathwhiz:hidden="false"/>'
                  f'</pathwhiz:location>')
                a('            </annotation>')
                a('                    </speciesReference>')
            a('        </listOfProducts>')

        # listOfModifiers (enzymes)
        modifiers = plan.get("modifiers", [])
        if modifiers:
            a('        <listOfModifiers>')
            n_mod = len(modifiers)
            for j, mod in enumerate(modifiers):
                mrec  = mod["rec"]
                mkind = mod["kind"]
                node_cx, node_cy, node_x, node_y = _modifier_pos(cx, cy, j, n_mod)
                loc_id = location_id_ctr.next()

                if mkind == "complex":
                    member_ids = []
                    for comp_name in mrec.get("components", []):
                        _, prec = _resolve_entity(comp_name)
                        if prec:
                            member_ids.append(prec["sbml_id"])
                    if not member_ids:
                        unknown_norm = _normalize("Unknown")
                        if unknown_norm in protein_by_norm:
                            member_ids.append(
                                protein_by_norm[unknown_norm]["sbml_id"])

                    a(f'          <modifierSpeciesReference'
                      f' species="{mrec["sbml_id"]}" sboTerm="SBO:0000460">')
                    a('            <annotation>')
                    a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                      f' pathwhiz:location_type="protein_complex">')
                    a(f'                <pathwhiz:location_element'
                      f' pathwhiz:element_type="protein_complex_visualization"'
                      f' pathwhiz:element_id="{mrec["sbml_id"]}">')
                    for pid_str in member_ids:
                        p_loc_id = location_id_ctr.next()
                        a(f'                  <pathwhiz:location_element'
                          f' pathwhiz:element_type="protein_location"'
                          f' pathwhiz:element_id="{pid_str}"'
                          f' pathwhiz:location_id="{p_loc_id}"'
                          f' pathwhiz:x="{node_x:.1f}" pathwhiz:y="{node_y:.1f}"'
                          f' pathwhiz:visualization_template_id="{_TMPL_PROTEIN}"'
                          f' pathwhiz:width="{_PROTEIN_W:.1f}"'
                          f' pathwhiz:height="{_PROTEIN_H:.1f}"'
                          f' pathwhiz:zindex="8" pathwhiz:hidden="false"/>')
                    a('</pathwhiz:location_element></pathwhiz:location>')
                    a('            </annotation>')
                    a('                    </modifierSpeciesReference>')
                else:
                    a(f'          <modifierSpeciesReference'
                      f' species="{mrec["sbml_id"]}" sboTerm="SBO:0000460">')
                    a('            <annotation>')
                    a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                      f' pathwhiz:location_type="protein">')
                    a(f'                <pathwhiz:location_element'
                      f' pathwhiz:element_type="protein_location"'
                      f' pathwhiz:element_id="{mrec["sbml_id"]}"'
                      f' pathwhiz:location_id="{loc_id}"'
                      f' pathwhiz:x="{node_x:.1f}" pathwhiz:y="{node_y:.1f}"'
                      f' pathwhiz:visualization_template_id="{_TMPL_PROTEIN}"'
                      f' pathwhiz:width="{_PROTEIN_W:.1f}"'
                      f' pathwhiz:height="{_PROTEIN_H:.1f}"'
                      f' pathwhiz:zindex="8" pathwhiz:hidden="false"/>'
                      f'</pathwhiz:location>')
                    a('            </annotation>')
                    a('                    </modifierSpeciesReference>')

            a('        </listOfModifiers>')

        a('      </reaction>')

    a('    </listOfReactions>')
    a('  </model>')
    a('</sbml>')

    # Write SBML
    sbml_text = "\n".join(lines)
    sbml_path.write_text(sbml_text, encoding="utf-8")

    report["counts"]["compartments"] = len(state_name_to_id)
    report["counts"]["species"]      = (len(compound_by_norm) +
                                        len(protein_by_norm) +
                                        len(complex_by_norm))
    report["counts"]["reactions"]    = len(reaction_plans)

    report_json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    txt_lines = [
        f"SBML file: {sbml_path}",
        f"Compartments: {report['counts']['compartments']}",
        f"Species: {report['counts']['species']}",
        f"Reactions: {report['counts']['reactions']}",
        f"Hard errors: {len(report['hard_errors'])}",
        f"Warnings: {len(report['warnings'])}",
    ]
    if report["hard_errors"]:
        txt_lines.append("\nHard errors:")
        for item in report["hard_errors"][:50]:
            txt_lines.append(f"- {item.get('path','')}: {item.get('reason','')}")
    report_txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PathWhiz-compatible SBML Level 3 from mapped pathway JSON.")
    parser.add_argument("--in",  dest="input_path",    required=True)
    parser.add_argument("--out", dest="sbml_path",     default="pathway.sbml")
    parser.add_argument("--report-json", dest="report_json_path",
                        default="sbml_validation_report.json")
    parser.add_argument("--report-txt",  dest="report_txt_path",
                        default="sbml_validation_report.txt")
    parser.add_argument("--default-compartment", dest="default_compartment",
                        default="cell")
    args = parser.parse_args()

    report = build_sbml(
        Path(args.input_path),
        Path(args.sbml_path),
        Path(args.report_json_path),
        Path(args.report_txt_path),
        default_compartment_name=str(args.default_compartment),
    )
    print(f"Wrote SBML: {args.sbml_path}")
    print(f"Compartments: {report['counts']['compartments']} | "
          f"Species: {report['counts']['species']} | "
          f"Reactions: {report['counts']['reactions']}")


if __name__ == "__main__":
    main()
