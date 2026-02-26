from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", "", lowered)


def sanitize_sbml_id(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", (raw or "").strip())
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        token = "x"
    if not re.match(r"^[A-Za-z_]", token):
        token = f"x_{token}"
    return token


def _short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _mapped_entity_id(item: Dict[str, Any], preferred_keys: Sequence[str]) -> str:
    mapped = _safe_dict(item.get("mapped_ids"))
    for key in preferred_keys:
        val = mapped.get(key)
        if isinstance(val, str) and val.strip():
            return sanitize_sbml_id(val.strip())
    return ""


def _extract_state_compartments(payload: Dict[str, Any]) -> Dict[str, str]:
    states: Dict[str, str] = {}
    for state in _safe_list(payload.get("biological_states")):
        if not isinstance(state, dict):
            continue
        name = (state.get("name") or "").strip() if isinstance(state.get("name"), str) else ""
        loc = (state.get("subcellular_location") or "").strip() if isinstance(state.get("subcellular_location"), str) else ""
        if name and loc:
            states[name] = loc
    return states


def _entity_compartment_map(
    payload: Dict[str, Any],
    states_to_loc: Dict[str, str],
    *,
    default_compartment_name: str,
    report: Dict[str, Any],
) -> Dict[Tuple[str, str], Set[str]]:
    out: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    element_locations = _safe_dict(payload.get("element_locations"))
    for list_key, name_key, kind in [
        ("compound_locations", "compound", "compound"),
        ("protein_locations", "protein", "protein"),
    ]:
        for idx, row in enumerate(_safe_list(element_locations.get(list_key))):
            if not isinstance(row, dict):
                continue
            name = (row.get(name_key) or "").strip() if isinstance(row.get(name_key), str) else ""
            if not name:
                continue
            state_name = (row.get("biological_state") or "").strip() if isinstance(row.get("biological_state"), str) else ""
            if state_name and state_name in states_to_loc:
                out[(kind, name)].add(states_to_loc[state_name])
            else:
                out[(kind, name)].add(default_compartment_name)
                report["defaults_applied"].append(
                    {
                        "type": "missing_biological_state_location",
                        "json_pointer": f"/element_locations/{list_key}/{idx}",
                        "name": name,
                        "default_compartment": default_compartment_name,
                    }
                )
    return out


def _known_entity_names(payload: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    entities = _safe_dict(payload.get("entities"))
    compounds = {
        (item.get("name") or "").strip()
        for item in _safe_list(entities.get("compounds"))
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name").strip()
    }
    proteins = {
        (item.get("name") or "").strip()
        for item in _safe_list(entities.get("proteins"))
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name").strip()
    }
    return compounds, proteins


def _usage_profile(processes: Dict[str, Any]) -> Dict[str, Set[str]]:
    io_names: Set[str] = set()
    enzyme_names: Set[str] = set()

    for reaction in _safe_list(processes.get("reactions")):
        if not isinstance(reaction, dict):
            continue
        for side in ["inputs", "outputs"]:
            for name in _safe_list(reaction.get(side)):
                if isinstance(name, str) and name.strip():
                    io_names.add(name.strip())
        for enzyme in _safe_list(reaction.get("enzymes")):
            if not isinstance(enzyme, dict):
                continue
            pname = _first_string([enzyme.get("protein"), enzyme.get("protein_complex"), enzyme.get("name")])
            if pname:
                enzyme_names.add(pname)

    for transport in _safe_list(processes.get("transports")):
        if not isinstance(transport, dict):
            continue
        cargo = (transport.get("cargo") or "").strip() if isinstance(transport.get("cargo"), str) else ""
        if cargo:
            io_names.add(cargo)
        for transporter in _safe_list(transport.get("transporters")):
            if not isinstance(transporter, dict):
                continue
            pname = _first_string([transporter.get("protein"), transporter.get("protein_complex"), transporter.get("name")])
            if pname:
                enzyme_names.add(pname)

    return {"io_names": io_names, "enzyme_names": enzyme_names}


def _resolve_cross_type_name_conflicts(
    compounds: Set[str],
    proteins: Set[str],
    processes: Dict[str, Any],
    report: Dict[str, Any],
) -> Tuple[Set[str], Set[str]]:
    usage = _usage_profile(processes)
    io_names = usage["io_names"]
    enzyme_names = usage["enzyme_names"]

    comp_out = set(compounds)
    prot_out = set(proteins)

    compounds_by_norm: Dict[str, List[str]] = defaultdict(list)
    proteins_by_norm: Dict[str, List[str]] = defaultdict(list)
    for name in compounds:
        compounds_by_norm[_normalize(name)].append(name)
    for name in proteins:
        proteins_by_norm[_normalize(name)].append(name)

    for norm in sorted(set(compounds_by_norm.keys()) & set(proteins_by_norm.keys())):
        c_names = compounds_by_norm[norm]
        p_names = proteins_by_norm[norm]
        all_names = c_names + p_names
        used_as_io = any(name in io_names for name in all_names)
        used_as_enzyme = any(name in enzyme_names for name in all_names)

        if used_as_io and not used_as_enzyme:
            for p in p_names:
                prot_out.discard(p)
            report["warnings"].append(
                {
                    "path": "/entities",
                    "reason": f"Name appears as compound+protein; keeping compound role for '{c_names[0]}'.",
                }
            )
        elif used_as_enzyme and not used_as_io:
            for c in c_names:
                comp_out.discard(c)
            report["warnings"].append(
                {
                    "path": "/entities",
                    "reason": f"Name appears as compound+protein; keeping protein role for '{p_names[0]}'.",
                }
            )
        else:
            report["warnings"].append(
                {
                    "path": "/entities",
                    "reason": f"Name '{all_names[0]}' appears as both compound and protein with dual/unclear role.",
                }
            )
    return comp_out, prot_out


def _pick_reaction_compartment(
    reaction: Dict[str, Any],
    states_to_loc: Dict[str, str],
    entity_compartments: Dict[Tuple[str, str], Set[str]],
    *,
    default_compartment_name: str,
    report: Dict[str, Any],
    pointer: str,
) -> str:
    state_name = (reaction.get("biological_state") or "").strip() if isinstance(reaction.get("biological_state"), str) else ""
    if state_name and state_name in states_to_loc:
        return states_to_loc[state_name]

    for side in ["inputs", "outputs"]:
        for name in _safe_list(reaction.get(side)):
            if not isinstance(name, str) or not name.strip():
                continue
            options = sorted(entity_compartments.get(("compound", name.strip()), []))
            if options:
                return options[0]

    report["defaults_applied"].append(
        {
            "type": "reaction_missing_compartment",
            "json_pointer": pointer,
            "default_compartment": default_compartment_name,
        }
    )
    return default_compartment_name


def _reaction_id(reactants: Sequence[str], products: Sequence[str], modifiers: Sequence[str], compartments: Sequence[str], kind: str = "reaction") -> str:
    payload = "|".join(
        [
            kind,
            ",".join(sorted(reactants)),
            ",".join(sorted(products)),
            ",".join(sorted(modifiers)),
            ",".join(sorted(compartments)),
        ]
    )
    return f"r_{_short_hash(payload)}"


def _same_multiset(values_a: Sequence[str], values_b: Sequence[str]) -> bool:
    return sorted(values_a) == sorted(values_b)


def _first_string(values: Sequence[Any]) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_sbml(
    input_path: Path,
    sbml_path: Path,
    report_json_path: Path,
    report_txt_path: Path,
    *,
    default_compartment_name: str = "cell",
) -> Dict[str, Any]:
    try:
        import libsbml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("python-libsbml is required. Install python-libsbml.") from exc

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mapped input JSON must be an object.")

    report: Dict[str, Any] = {
        "hard_errors": [],
        "warnings": [],
        "defaults_applied": [],
        "counts": {"compartments": 0, "species": 0, "reactions": 0},
    }
    data = deepcopy(payload)
    entities = _safe_dict(data.get("entities"))
    processes = _safe_dict(data.get("processes"))

    states_to_loc = _extract_state_compartments(data)
    raw_compounds, raw_proteins = _known_entity_names(data)
    known_compounds, known_proteins = _resolve_cross_type_name_conflicts(
        raw_compounds,
        raw_proteins,
        processes,
        report,
    )

    # Compartments from explicit location entities + biological states + default.
    location_names: Set[str] = set()
    for item in _safe_list(entities.get("subcellular_locations")):
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name").strip():
            location_names.add(item["name"].strip())
    for loc in states_to_loc.values():
        if loc:
            location_names.add(loc)
    location_names.add(default_compartment_name)

    compartment_by_name: Dict[str, str] = {}
    for loc_name in sorted(location_names, key=lambda x: sanitize_sbml_id(x)):
        cid = f"c_{sanitize_sbml_id(_normalize(loc_name) or loc_name)}"
        if cid in compartment_by_name.values():
            cid = f"{cid}_{_short_hash(loc_name, 6)}"
        compartment_by_name[loc_name] = cid

    entity_compartments = _entity_compartment_map(
        data,
        states_to_loc,
        default_compartment_name=default_compartment_name,
        report=report,
    )

    # Ensure every compound/protein has at least one compartment.
    for comp in known_compounds:
        if not entity_compartments.get(("compound", comp)):
            entity_compartments[("compound", comp)].add(default_compartment_name)
            report["defaults_applied"].append(
                {
                    "type": "compound_missing_location",
                    "name": comp,
                    "default_compartment": default_compartment_name,
                }
            )
    for prot in known_proteins:
        if not entity_compartments.get(("protein", prot)):
            entity_compartments[("protein", prot)].add(default_compartment_name)
            report["defaults_applied"].append(
                {
                    "type": "protein_missing_location",
                    "name": prot,
                    "default_compartment": default_compartment_name,
                }
            )

    species_registry: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    species_id_to_meta: Dict[str, Dict[str, Any]] = {}

    for item in _safe_list(entities.get("compounds")):
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip() if isinstance(item.get("name"), str) else ""
        if not name:
            continue
        if name not in known_compounds:
            continue
        preferred = _mapped_entity_id(item, ["chebi", "hmdb", "kegg"])
        for loc in sorted(entity_compartments.get(("compound", name), {default_compartment_name})):
            cid = compartment_by_name.get(loc, compartment_by_name[default_compartment_name])
            if preferred:
                sid = sanitize_sbml_id(f"m_{preferred}__{cid}")
            else:
                sid = sanitize_sbml_id(f"m_unmapped_{_short_hash(name + '|' + loc)}__{cid}")
            key = ("compound", name, cid)
            if sid in species_id_to_meta and species_id_to_meta[sid] != {"kind": "compound", "name": name, "compartment_id": cid}:
                sid = sanitize_sbml_id(f"{sid}_{_short_hash(name + cid, 6)}")
            species_registry[key] = {"id": sid, "name": name, "kind": "compound", "compartment_id": cid}
            species_id_to_meta[sid] = {"kind": "compound", "name": name, "compartment_id": cid}

    for item in _safe_list(entities.get("proteins")):
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip() if isinstance(item.get("name"), str) else ""
        if not name:
            continue
        if name not in known_proteins:
            continue
        preferred = _mapped_entity_id(item, ["uniprot"])
        for loc in sorted(entity_compartments.get(("protein", name), {default_compartment_name})):
            cid = compartment_by_name.get(loc, compartment_by_name[default_compartment_name])
            if preferred:
                sid = sanitize_sbml_id(f"p_{preferred}__{cid}")
            else:
                sid = sanitize_sbml_id(f"p_unmapped_{_short_hash(name + '|' + loc)}__{cid}")
            key = ("protein", name, cid)
            if sid in species_id_to_meta and species_id_to_meta[sid] != {"kind": "protein", "name": name, "compartment_id": cid}:
                sid = sanitize_sbml_id(f"{sid}_{_short_hash(name + cid, 6)}")
            species_registry[key] = {"id": sid, "name": name, "kind": "protein", "compartment_id": cid}
            species_id_to_meta[sid] = {"kind": "protein", "name": name, "compartment_id": cid}

    # Build reaction plans first, then write sorted by reaction ID.
    reaction_plans: List[Dict[str, Any]] = []
    seen_rids: Set[str] = set()
    reactions = _safe_list(processes.get("reactions"))
    for ridx, reaction in enumerate(reactions):
        pointer = f"/processes/reactions/{ridx}"
        if not isinstance(reaction, dict):
            report["hard_errors"].append({"path": pointer, "reason": "Reaction item is not an object."})
            continue
        inputs = [x.strip() for x in _safe_list(reaction.get("inputs")) if isinstance(x, str) and x.strip()]
        outputs = [x.strip() for x in _safe_list(reaction.get("outputs")) if isinstance(x, str) and x.strip()]
        if not inputs or not outputs:
            report["hard_errors"].append({"path": pointer, "reason": "Reaction missing inputs or outputs."})
            continue

        loc_name = _pick_reaction_compartment(
            reaction,
            states_to_loc,
            entity_compartments,
            default_compartment_name=default_compartment_name,
            report=report,
            pointer=pointer,
        )
        compartment_id = compartment_by_name.get(loc_name, compartment_by_name[default_compartment_name])

        reactant_ids: List[str] = []
        product_ids: List[str] = []
        unresolved = False

        for name in inputs:
            if name not in known_compounds:
                report["hard_errors"].append(
                    {"path": f"{pointer}/inputs", "reason": f"Unknown input compound '{name}'."}
                )
                unresolved = True
                continue
            key = ("compound", name, compartment_id)
            if key not in species_registry:
                # Deterministic best effort: create missing per-compartment species instance.
                sid = sanitize_sbml_id(f"m_unmapped_{_short_hash(name + '|' + compartment_id)}__{compartment_id}")
                species_registry[key] = {"id": sid, "name": name, "kind": "compound", "compartment_id": compartment_id}
                species_id_to_meta[sid] = {"kind": "compound", "name": name, "compartment_id": compartment_id}
                report["warnings"].append(
                    {
                        "path": f"{pointer}/inputs",
                        "reason": f"Created missing species instance for input '{name}' in compartment '{compartment_id}'.",
                    }
                )
            reactant_ids.append(species_registry[key]["id"])

        for name in outputs:
            if name not in known_compounds:
                report["hard_errors"].append(
                    {"path": f"{pointer}/outputs", "reason": f"Unknown output compound '{name}'."}
                )
                unresolved = True
                continue
            key = ("compound", name, compartment_id)
            if key not in species_registry:
                sid = sanitize_sbml_id(f"m_unmapped_{_short_hash(name + '|' + compartment_id)}__{compartment_id}")
                species_registry[key] = {"id": sid, "name": name, "kind": "compound", "compartment_id": compartment_id}
                species_id_to_meta[sid] = {"kind": "compound", "name": name, "compartment_id": compartment_id}
                report["warnings"].append(
                    {
                        "path": f"{pointer}/outputs",
                        "reason": f"Created missing species instance for output '{name}' in compartment '{compartment_id}'.",
                    }
                )
            product_ids.append(species_registry[key]["id"])

        if unresolved:
            continue

        modifier_ids: List[str] = []
        for enzyme in _safe_list(reaction.get("enzymes")):
            if not isinstance(enzyme, dict):
                continue
            pname = _first_string([enzyme.get("protein"), enzyme.get("protein_complex"), enzyme.get("name")])
            if pname and pname in known_proteins:
                pkey = ("protein", pname, compartment_id)
                if pkey in species_registry:
                    modifier_ids.append(species_registry[pkey]["id"])

        if _same_multiset(reactant_ids, product_ids):
            report["warnings"].append(
                {
                    "path": pointer,
                    "reason": "Dropped degenerate reaction with identical reactants and products.",
                }
            )
            continue

        reaction_id = _reaction_id(reactant_ids, product_ids, modifier_ids, [compartment_id], kind="reaction")
        if reaction_id in seen_rids:
            report["warnings"].append({"path": pointer, "reason": f"Duplicate reaction collapsed: {reaction_id}"})
            continue
        seen_rids.add(reaction_id)
        reaction_plans.append(
            {
                "id": reaction_id,
                "name": (reaction.get("name") or "").strip() if isinstance(reaction.get("name"), str) else "",
                "reactants": sorted(reactant_ids),
                "products": sorted(product_ids),
                "modifiers": sorted(set(modifier_ids)),
                "compartment_id": compartment_id,
            }
        )

    transports = _safe_list(processes.get("transports"))
    for tidx, transport in enumerate(transports):
        pointer = f"/processes/transports/{tidx}"
        if not isinstance(transport, dict):
            report["hard_errors"].append({"path": pointer, "reason": "Transport item is not an object."})
            continue
        cargo = (transport.get("cargo") or "").strip() if isinstance(transport.get("cargo"), str) else ""
        if not cargo:
            # try side-based fallback
            left = ""
            right = ""
            for elem in _safe_list(transport.get("elements_with_states")):
                if not isinstance(elem, dict):
                    continue
                side = (elem.get("side") or "").strip().lower() if isinstance(elem.get("side"), str) else ""
                entity = (elem.get("element") or "").strip() if isinstance(elem.get("element"), str) else ""
                if side == "left" and entity:
                    left = entity
                if side == "right" and entity:
                    right = entity
            cargo = left or right
        if not cargo:
            report["hard_errors"].append({"path": pointer, "reason": "Transport missing cargo."})
            continue
        if cargo not in known_compounds and cargo not in known_proteins:
            report["hard_errors"].append({"path": f"{pointer}/cargo", "reason": f"Unknown cargo '{cargo}'."})
            continue

        from_state = (
            transport.get("from_biological_state") or ""
            if isinstance(transport.get("from_biological_state"), str)
            else ""
        )
        to_state = (
            transport.get("to_biological_state") or ""
            if isinstance(transport.get("to_biological_state"), str)
            else ""
        )
        from_loc = states_to_loc.get(from_state.strip(), "") if from_state else ""
        to_loc = states_to_loc.get(to_state.strip(), "") if to_state else ""
        if not from_loc:
            from_loc = default_compartment_name
            report["defaults_applied"].append(
                {
                    "type": "transport_missing_source",
                    "json_pointer": pointer,
                    "default_compartment": default_compartment_name,
                }
            )
        if not to_loc:
            to_loc = default_compartment_name
            report["defaults_applied"].append(
                {
                    "type": "transport_missing_destination",
                    "json_pointer": pointer,
                    "default_compartment": default_compartment_name,
                }
            )

        source_cid = compartment_by_name.get(from_loc, compartment_by_name[default_compartment_name])
        dest_cid = compartment_by_name.get(to_loc, compartment_by_name[default_compartment_name])
        kind = "compound" if cargo in known_compounds else "protein"
        source_key = (kind, cargo, source_cid)
        dest_key = (kind, cargo, dest_cid)

        if source_key not in species_registry:
            prefix = "m" if kind == "compound" else "p"
            sid = sanitize_sbml_id(f"{prefix}_unmapped_{_short_hash(cargo + '|' + source_cid)}__{source_cid}")
            species_registry[source_key] = {"id": sid, "name": cargo, "kind": kind, "compartment_id": source_cid}
            species_id_to_meta[sid] = {"kind": kind, "name": cargo, "compartment_id": source_cid}
        if dest_key not in species_registry:
            prefix = "m" if kind == "compound" else "p"
            sid = sanitize_sbml_id(f"{prefix}_unmapped_{_short_hash(cargo + '|' + dest_cid)}__{dest_cid}")
            species_registry[dest_key] = {"id": sid, "name": cargo, "kind": kind, "compartment_id": dest_cid}
            species_id_to_meta[sid] = {"kind": kind, "name": cargo, "compartment_id": dest_cid}

        if source_key == dest_key:
            report["warnings"].append(
                {
                    "path": pointer,
                    "reason": "Dropped degenerate transport with identical source and destination species.",
                }
            )
            continue

        modifiers: List[str] = []
        for transporter in _safe_list(transport.get("transporters")):
            if not isinstance(transporter, dict):
                continue
            pname = _first_string([transporter.get("protein"), transporter.get("protein_complex"), transporter.get("name")])
            if pname and pname in known_proteins:
                key = ("protein", pname, source_cid)
                if key in species_registry:
                    modifiers.append(species_registry[key]["id"])

        reaction_id = _reaction_id(
            [species_registry[source_key]["id"]],
            [species_registry[dest_key]["id"]],
            modifiers,
            [source_cid, dest_cid],
            kind="transport",
        )
        if reaction_id in seen_rids:
            report["warnings"].append({"path": pointer, "reason": f"Duplicate transport collapsed: {reaction_id}"})
            continue
        seen_rids.add(reaction_id)
        reaction_plans.append(
            {
                "id": reaction_id,
                "name": (transport.get("name") or f"transport_{cargo}").strip() if isinstance(transport.get("name"), str) else f"transport_{cargo}",
                "reactants": [species_registry[source_key]["id"]],
                "products": [species_registry[dest_key]["id"]],
                "modifiers": sorted(set(modifiers)),
                "compartment_id": source_cid,
            }
        )

    # Build SBML document.
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId("pathway_model")
    model.setName("Pathway model generated from mapped JSON")
    model.setSubstanceUnits("Unit1")
    model.setExtentUnits("Unit1")
    model.setVolumeUnits("UnitVol")

    unit_def = model.createUnitDefinition()
    unit_def.setId("Unit1")
    unit_def.setName("Mole")
    unit = unit_def.createUnit()
    unit.setKind(libsbml.UNIT_KIND_MOLE)
    unit.setExponent(1)
    unit.setScale(0)
    unit.setMultiplier(1.0)

    volume_unit_def = model.createUnitDefinition()
    volume_unit_def.setId("UnitVol")
    volume_unit_def.setName("Litre")
    vunit = volume_unit_def.createUnit()
    vunit.setKind(libsbml.UNIT_KIND_LITRE)
    vunit.setExponent(1)
    vunit.setScale(0)
    vunit.setMultiplier(1.0)

    for loc_name, cid in sorted(compartment_by_name.items(), key=lambda kv: kv[1]):
        comp = model.createCompartment()
        comp.setId(cid)
        comp.setName(loc_name)
        comp.setConstant(True)
        comp.setSize(1.0)
        comp.setSpatialDimensions(3)
        comp.setUnits("UnitVol")

    species_items = sorted(species_registry.values(), key=lambda item: item["id"])
    for item in species_items:
        sp = model.createSpecies()
        sp.setId(item["id"])
        sp.setName(item["name"])
        sp.setCompartment(item["compartment_id"])
        sp.setBoundaryCondition(False)
        sp.setHasOnlySubstanceUnits(True)
        sp.setConstant(False)
        sp.setInitialAmount(0.0)
        sp.setSubstanceUnits("Unit1")

    for plan in sorted(reaction_plans, key=lambda item: item["id"]):
        rxn = model.createReaction()
        rxn.setId(plan["id"])
        if plan["name"]:
            rxn.setName(plan["name"])
        rxn.setReversible(False)
        rxn.setFast(False)
        for sid in plan["reactants"]:
            ref = rxn.createReactant()
            ref.setSpecies(sid)
            ref.setConstant(True)
            ref.setStoichiometry(1.0)
        for sid in plan["products"]:
            ref = rxn.createProduct()
            ref.setSpecies(sid)
            ref.setConstant(True)
            ref.setStoichiometry(1.0)
        for sid in plan["modifiers"]:
            ref = rxn.createModifier()
            ref.setSpecies(sid)

    report["counts"]["compartments"] = model.getNumCompartments()
    report["counts"]["species"] = model.getNumSpecies()
    report["counts"]["reactions"] = model.getNumReactions()

    libsbml.writeSBMLToFile(doc, str(sbml_path))
    n_errors = doc.checkConsistency()
    validation_errors: List[Dict[str, Any]] = []
    has_validation_errors = False
    for idx in range(doc.getNumErrors()):
        err = doc.getError(idx)
        entry = {
            "severity": int(err.getSeverity()),
            "category": int(err.getCategory()),
            "message": err.getMessage(),
            "line": int(err.getLine()),
        }
        validation_errors.append(entry)
        # Severity >= 2 typically maps to error/fatal.
        if entry["severity"] >= 2:
            has_validation_errors = True

    report["validation"] = {
        "check_count": int(n_errors),
        "error_count": len(validation_errors),
        "has_errors": has_validation_errors,
        "messages": validation_errors,
    }

    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_lines = [
        f"SBML file: {sbml_path}",
        f"Compartments: {report['counts']['compartments']}",
        f"Species: {report['counts']['species']}",
        f"Reactions: {report['counts']['reactions']}",
        f"Hard errors (skipped items): {len(report['hard_errors'])}",
        f"Warnings: {len(report['warnings'])}",
        f"Defaults applied: {len(report['defaults_applied'])}",
        f"Validation messages: {report['validation']['error_count']}",
        f"Validation has errors: {report['validation']['has_errors']}",
    ]
    if report["hard_errors"]:
        txt_lines.append("")
        txt_lines.append("Hard errors:")
        for item in report["hard_errors"][:100]:
            txt_lines.append(f"- {item.get('path', '')}: {item.get('reason', '')}")
    if report["validation"]["messages"]:
        txt_lines.append("")
        txt_lines.append("Validation messages:")
        for msg in report["validation"]["messages"][:100]:
            txt_lines.append(f"- [sev={msg['severity']}] line {msg['line']}: {msg['message']}")
    report_txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SBML Level 3 Core from mapped pathway JSON.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input mapped JSON path")
    parser.add_argument("--out", dest="sbml_path", default="pathway.sbml", help="Output SBML file path")
    parser.add_argument(
        "--report-json",
        dest="report_json_path",
        default="sbml_validation_report.json",
        help="Validation report JSON path",
    )
    parser.add_argument(
        "--report-txt",
        dest="report_txt_path",
        default="sbml_validation_report.txt",
        help="Validation report text path",
    )
    parser.add_argument(
        "--default-compartment",
        dest="default_compartment",
        default="cell",
        help="Default compartment name used when missing",
    )
    args = parser.parse_args()

    report = build_sbml(
        Path(args.input_path),
        Path(args.sbml_path),
        Path(args.report_json_path),
        Path(args.report_txt_path),
        default_compartment_name=str(args.default_compartment),
    )
    print(f"Wrote SBML: {args.sbml_path}")
    print(f"Wrote validation report: {args.report_json_path} and {args.report_txt_path}")
    if report.get("validation", {}).get("has_errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
