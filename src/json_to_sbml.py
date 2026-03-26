from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
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


def _dedupe_preserve_strings(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _mapped_ids_signature(mapped_ids: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    pairs: List[Tuple[str, str]] = []
    for key, value in sorted(_safe_dict(mapped_ids).items()):
        key_text = str(key).strip().casefold()
        value_text = str(value).strip()
        if not key_text or not value_text:
            continue
        pairs.append((key_text, value_text))
    return tuple(pairs)


def _mapped_ids_group_key(row: Dict[str, Any], preferred_keys: Sequence[str]) -> Tuple[str, Any]:
    primary = _mapped_entity_id(row, preferred_keys)
    if primary:
        return ("primary", primary)
    signature = _mapped_ids_signature(_safe_dict(row.get("mapped_ids")))
    if signature:
        return ("full", signature)
    return ("", "")


def _merge_deduped_entity_rows(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target_aliases = _dedupe_preserve_strings(
        _safe_list(target.get("_dedupe_aliases")) + _safe_list(source.get("_dedupe_aliases"))
    )
    if target_aliases:
        target["_dedupe_aliases"] = target_aliases

    target_mapped = _safe_dict(target.get("mapped_ids"))
    source_mapped = _safe_dict(source.get("mapped_ids"))
    merged_mapped = dict(source_mapped)
    merged_mapped.update(target_mapped)
    target["mapped_ids"] = merged_mapped


def _unique_entity_rows(rows_by_name: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_ids: Set[int] = set()
    for row in rows_by_name.values():
        marker = id(row)
        if marker in seen_ids:
            continue
        seen_ids.add(marker)
        out.append(row)
    return out


def _mapped_entity_id(item: Dict[str, Any], preferred_keys: Sequence[str]) -> str:
    mapped = _safe_dict(item.get("mapped_ids"))
    for key in preferred_keys:
        val = mapped.get(key)
        if isinstance(val, str) and val.strip():
            return sanitize_sbml_id(val.strip())
    return ""


def sbml_species_id(entity: Dict[str, Any], compartment: str) -> str:
    kind = str(entity.get("kind") or "").strip().lower()
    name = str(entity.get("name") or "").strip()
    mapped = _safe_dict(entity.get("mapped_ids"))
    cpt = sanitize_sbml_id(compartment)

    is_protein_like = kind in {"protein", "protein_complex", "complex"}
    if kind == "element_collection":
        prefix = "ec_"
        preferred_keys: List[str] = []
    elif is_protein_like:
        prefix = "p_"
        preferred_keys = ["uniprot"]
    else:
        prefix = "m_"
        preferred_keys = ["chebi", "hmdb", "kegg", "pubchem", "drugbank"]

    primary = ""
    for key in preferred_keys:
        value = mapped.get(key)
        if isinstance(value, str) and value.strip():
            v = value.strip()
            if key == "chebi" and v and not v.upper().startswith("CHEBI:"):
                v = f"CHEBI:{v}"
            primary = sanitize_sbml_id(v)
            break
    if primary:
        return sanitize_sbml_id(f"{prefix}{primary}__{cpt}")

    name_hash = _short_hash(_normalize(name) or name)
    return sanitize_sbml_id(f"{prefix}unmapped_{name_hash}__{cpt}")


def _dedupe_entity_rows(
    rows: Sequence[Any],
    *,
    preferred_mapped_keys: Sequence[str] = (),
) -> Dict[str, Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = (row.get("name") or "").strip() if isinstance(row.get("name"), str) else ""
        if not name:
            continue
        norm = _normalize(name)
        if norm not in by_name:
            by_name[norm] = deepcopy(row)
            by_name[norm]["name"] = name
            by_name[norm]["_dedupe_aliases"] = [name]
            continue
        incoming = deepcopy(row)
        incoming["name"] = name
        incoming["_dedupe_aliases"] = [name]
        _merge_deduped_entity_rows(by_name[norm], incoming)
    out: Dict[str, Dict[str, Any]] = {}
    by_mapped_ids: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for norm, row in by_name.items():
        group_key = _mapped_ids_group_key(row, preferred_mapped_keys)
        if group_key == ("", ""):
            out[norm] = row
            continue
        canonical = by_mapped_ids.get(group_key)
        if canonical is None:
            by_mapped_ids[group_key] = row
            out[norm] = row
            continue
        _merge_deduped_entity_rows(canonical, row)
        out[norm] = canonical
    return out


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
        ("element_collection_locations", "element_collection", "element_collection"),
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


def _known_entity_names(payload: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str]]:
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
    proteins.update(
        {
            (item.get("name") or "").strip()
            for item in _safe_list(entities.get("protein_complexes"))
            if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name").strip()
        }
    )
    element_collections = {
        (item.get("name") or "").strip()
        for item in _safe_list(entities.get("element_collections"))
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name").strip()
    }
    return compounds, proteins, element_collections


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
        cargo = (
            transport.get("cargo_complex")
            if isinstance(transport.get("cargo_complex"), str)
            else transport.get("cargo")
        )
        cargo = (cargo or "").strip() if isinstance(cargo, str) else ""
        if cargo:
            io_names.add(cargo)
        for transporter in _safe_list(transport.get("transporters")):
            if not isinstance(transporter, dict):
                continue
            pname = _first_string([transporter.get("protein"), transporter.get("protein_complex"), transporter.get("name")])
            if pname:
                enzyme_names.add(pname)

    return {"io_names": io_names, "enzyme_names": enzyme_names}


def _split_composite_name(value: str) -> List[str]:
    text = (value or "").strip()
    if not text:
        return []
    parts = re.split(r"\s*\+\s*", text)
    out = [p.strip() for p in parts if p and p.strip()]
    return out


def _is_composite_name(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    return "+" in text


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


def _augment_entity_sets_from_processes(
    compounds: Set[str],
    proteins: Set[str],
    processes: Dict[str, Any],
    report: Dict[str, Any],
) -> Tuple[Set[str], Set[str]]:
    comp = set(compounds)
    prot = set(proteins)

    for ridx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        for side in ["inputs", "outputs"]:
            for name in _safe_list(reaction.get(side)):
                if not isinstance(name, str) or not name.strip():
                    continue
                n = name.strip()
                if _is_composite_name(n):
                    report["warnings"].append(
                        {
                            "path": f"/processes/reactions/{ridx}/{side}",
                            "reason": f"Composite reaction token '{n}' detected; splitting into individual names.",
                        }
                    )
                    for sub in _split_composite_name(n):
                        if sub and sub not in prot:
                            comp.add(sub)
                    continue
                if n in prot:
                    continue
                comp.add(n)

    for tidx, transport in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(transport, dict):
            continue
        cargo = (
            transport.get("cargo_complex")
            if isinstance(transport.get("cargo_complex"), str)
            else transport.get("cargo")
        )
        cargo = (cargo or "").strip() if isinstance(cargo, str) else ""
        if cargo:
            if _is_composite_name(cargo):
                report["warnings"].append(
                    {
                        "path": f"/processes/transports/{tidx}/cargo",
                        "reason": f"Composite transport cargo '{cargo}' detected; transport will be split by cargo tokens.",
                    }
                )
                for sub in _split_composite_name(cargo):
                    if sub in prot:
                        continue
                    comp.add(sub)
            else:
                if cargo not in prot:
                    comp.add(cargo)
        for transporter in _safe_list(transport.get("transporters")):
            if not isinstance(transporter, dict):
                continue
            pname = _first_string([transporter.get("protein"), transporter.get("protein_complex"), transporter.get("name")])
            if pname:
                prot.add(pname)

    return comp, prot


def _drop_composite_entities(
    compounds: Set[str],
    proteins: Set[str],
    report: Dict[str, Any],
) -> Tuple[Set[str], Set[str]]:
    comp_out = set(compounds)
    prot_out = set(proteins)

    dropped_compounds = sorted([name for name in comp_out if _is_composite_name(name)])
    dropped_proteins = sorted([name for name in prot_out if _is_composite_name(name)])

    for name in dropped_compounds:
        comp_out.discard(name)
    for name in dropped_proteins:
        prot_out.discard(name)

    if dropped_compounds:
        report["warnings"].append(
            {
                "path": "/entities/compounds",
                "reason": "Dropped composite compound names from species registry.",
                "evidence": ", ".join(dropped_compounds[:8]),
            }
        )
    if dropped_proteins:
        report["warnings"].append(
            {
                "path": "/entities/proteins",
                "reason": "Dropped composite protein names from species registry.",
                "evidence": ", ".join(dropped_proteins[:8]),
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
            for kind in ["compound", "protein", "element_collection"]:
                options = sorted(entity_compartments.get((kind, name.strip()), []))
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


def _load_pathwhiz_db(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load PathWhiz ID mappings from MySQL, falling back to pathwhiz_id_db.json."""
    db: Dict[str, Any] = {
        "compounds": {"hmdb": {}, "kegg": {}, "chebi": {}, "pubchem": {}, "drugbank": {}},
        "proteins": {"uniprot": {}},
        "biological_states": {"by_location_name": {}},
        "reactions": {"by_elements": {}},
        "element_collections": {},
    }

    host = os.environ.get("PATHBANK_DB_HOST", "")
    user = os.environ.get("PATHBANK_DB_USER", "")
    password = os.environ.get("PATHBANK_DB_PASSWORD", "")
    schema = os.environ.get("PATHBANK_DB_SCHEMA", "pathbank")
    port = int(os.environ.get("PATHBANK_DB_PORT", "3306"))

    if not host or not user:
        print(
            "  INFO: PATHBANK_DB_HOST/PATHBANK_DB_USER not set — skipping MySQL PathWhiz lookup",
            file=sys.stderr,
        )
    if host and user:
        try:
            import pymysql  # type: ignore[import-not-found]
            print(f"  INFO: Connecting to PathWhiz MySQL at {host}:{port} schema={schema}", file=sys.stderr)
            conn = pymysql.connect(
                host=host, port=port, user=user, password=password,
                database=schema, charset="utf8mb4", connect_timeout=10,
                cursorclass=pymysql.cursors.DictCursor, autocommit=True,
            )
            with conn:
                with conn.cursor() as cur:
                    # Compounds
                    cur.execute(
                        "SELECT id, hmdb_id, kegg_id, chebi_id, pubchem_cid, drugbank_id"
                        " FROM compounds"
                    )
                    for row in cur.fetchall():
                        cid = row["id"]
                        if row.get("hmdb_id"):
                            db["compounds"]["hmdb"][row["hmdb_id"]] = cid
                        if row.get("kegg_id"):
                            db["compounds"]["kegg"][row["kegg_id"]] = cid
                        if row.get("chebi_id"):
                            chebi = row["chebi_id"]
                            bare = chebi.replace("CHEBI:", "").strip()
                            db["compounds"]["chebi"][chebi] = cid
                            db["compounds"]["chebi"][bare] = cid
                            db["compounds"]["chebi"][f"CHEBI:{bare}"] = cid
                        if row.get("pubchem_cid"):
                            db["compounds"]["pubchem"][str(row["pubchem_cid"])] = cid
                        if row.get("drugbank_id"):
                            db["compounds"]["drugbank"][row["drugbank_id"]] = cid

                    # Proteins
                    cur.execute(
                        "SELECT id, uniprot_id FROM proteins"
                        " WHERE uniprot_id IS NOT NULL AND uniprot_id != ''"
                    )
                    for row in cur.fetchall():
                        db["proteins"]["uniprot"][row["uniprot_id"]] = row["id"]

                    # Biological states → subcellular location name
                    try:
                        cur.execute(
                            "SELECT bs.id, sl.name"
                            " FROM biological_states bs"
                            " JOIN subcellular_locations sl"
                            "   ON bs.subcellular_location_id = sl.id"
                            " WHERE sl.name IS NOT NULL"
                        )
                        for row in cur.fetchall():
                            db["biological_states"]["by_location_name"][
                                row["name"].lower()
                            ] = row["id"]
                    except Exception:
                        pass

                    # Reactions — index by sorted left/right compound element IDs
                    try:
                        cur.execute(
                            "SELECT reaction_id, element_id, type"
                            " FROM reaction_elements"
                            " WHERE element_type = 'Compound'"
                        )
                        rxn_sides: Dict[int, Dict[str, List[int]]] = {}
                        LEFT_VALS = {"left", "substrate", "reactant", "reactionleftelement"}
                        for row in cur.fetchall():
                            rid = row["reaction_id"]
                            if rid not in rxn_sides:
                                rxn_sides[rid] = {"left": [], "right": []}
                            side = "left" if (row["type"] or "").lower() in LEFT_VALS else "right"
                            rxn_sides[rid][side].append(row["element_id"])
                        for rid, sides in rxn_sides.items():
                            lk = ",".join(sorted(str(i) for i in sides["left"]))
                            rk = ",".join(sorted(str(i) for i in sides["right"]))
                            db["reactions"]["by_elements"][f"{lk}→{rk}"] = rid
                    except Exception:
                        pass

            print(
                f"  INFO: PathWhiz MySQL loaded — "
                f"compounds={len(db['compounds']['hmdb'])} "
                f"proteins={len(db['proteins']['uniprot'])} "
                f"bio_states={len(db['biological_states']['by_location_name'])} "
                f"reactions={len(db['reactions']['by_elements'])}",
                file=sys.stderr,
            )
            return db
        except Exception as exc:
            print(
                f"  WARNING: MySQL PathWhiz DB unavailable ({exc}), falling back to JSON",
                file=sys.stderr,
            )

    # Fallback: JSON file built from PWML dump
    candidates: List[Path] = []
    if db_path:
        candidates.append(db_path)
    here = Path(__file__).parent
    for d in [here, here.parent, Path.cwd(), Path.cwd().parent]:
        candidates.append(d / "pathwhiz_id_db.json")
        candidates.append(d / "data" / "pathwhiz_id_db.json")
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return db


def _lookup_compound_id(db: Dict[str, Any], mapped_ids: Dict[str, Any]) -> Optional[int]:
    """Return PathWhiz compound/element_collection integer ID from a species' mapped_ids dict."""
    compounds = db.get("compounds", {})
    ecs = db.get("element_collections", {})
    for field, index_key in [("hmdb", "hmdb"), ("kegg", "kegg"), ("chebi", "chebi"),
                              ("pubchem", "pubchem"), ("drugbank", "drugbank")]:
        val = mapped_ids.get(field, "")
        if not isinstance(val, str) or not val.strip():
            continue
        v = val.strip()
        # Normalise variants to try
        variants = [v]
        if field == "chebi":
            bare = v.replace("CHEBI:", "").strip()
            variants += [bare, f"CHEBI:{bare}"]
        elif field == "hmdb":
            # HMDB IDs can be stored as HMDB0001234 (7 digits) or HMDB00001234 (8 digits)
            digits = re.sub(r"^HMDB0*", "", v)
            variants += [f"HMDB{digits.zfill(7)}", f"HMDB{digits.zfill(8)}", f"HMDB0{digits.zfill(6)}"]
        for candidate in dict.fromkeys(variants):  # dedup preserving order
            hit = compounds.get(index_key, {}).get(candidate)
            if hit:
                return int(hit)
            # Element collections use "kegg_compound" and "chebi" as index keys
            ec_key = "kegg_compound" if index_key == "kegg" else index_key
            hit = ecs.get(ec_key, {}).get(candidate)
            if hit:
                return int(hit)
    return None


def _lookup_protein_id(db: Dict[str, Any], mapped_ids: Dict[str, Any]) -> Optional[int]:
    """Return PathWhiz protein integer ID from a species' mapped_ids dict."""
    uniprot = mapped_ids.get("uniprot", "")
    if isinstance(uniprot, str) and uniprot.strip():
        hit = db.get("proteins", {}).get("uniprot", {}).get(uniprot.strip())
        if hit:
            return int(hit)
    return None


def _lookup_bs_id(db: Dict[str, Any], compartment_name: str) -> Optional[int]:
    """Return PathWhiz biological-state integer ID from a compartment name."""
    hit = db.get("biological_states", {}).get("by_location_name", {}).get(
        compartment_name.lower()
    )
    return int(hit) if hit else None


def _lookup_reaction_id(
    db: Dict[str, Any],
    reactant_pw_ids: List[int],
    product_pw_ids: List[int],
) -> Optional[int]:
    """Return PathWhiz reaction integer ID by matching left/right element IDs."""
    left_key = ",".join(sorted(str(i) for i in reactant_pw_ids))
    right_key = ",".join(sorted(str(i) for i in product_pw_ids))
    key = f"{left_key}→{right_key}"
    hit = db.get("reactions", {}).get("by_elements", {}).get(key)
    return int(hit) if hit else None


def _add_cv_terms(element: Any, mapped_ids: Dict[str, Any], libsbml: Any) -> None:
    """Add bqbiol:is CVTerm entries for each known database xref."""
    DB_URN: Dict[str, str] = {
        "hmdb": "urn:miriam:hmdb:",
        "kegg": "urn:miriam:kegg.compound:",
        "chebi": "urn:miriam:chebi:",
        "pubchem": "urn:miriam:pubchem.compound:",
        "drugbank": "urn:miriam:drugbank:",
        "uniprot": "urn:miriam:uniprot:",
    }
    resources = []
    for db_key, val in sorted(mapped_ids.items()):
        if not isinstance(val, str) or not val.strip():
            continue
        prefix = DB_URN.get(db_key.lower().strip())
        if prefix:
            resources.append(f"{prefix}{val.strip()}")
    if not resources:
        return
    cv = libsbml.CVTerm()
    cv.setQualifierType(libsbml.BIOLOGICAL_QUALIFIER)
    cv.setBiologicalQualifierType(libsbml.BQB_IS)
    for uri in resources:
        cv.addResource(uri)
    element.addCVTerm(cv)


def _inject_root_namespaces(sbml_path: Path) -> None:
    """Post-process the written SBML to add xmlns declarations PathWhiz expects on <sbml>."""
    BQ_URI = "http://biomodels.net/biology-qualifiers/"
    PW_URI = "http://www.spmdb.ca/pathwhiz"
    RDF_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

    text = sbml_path.read_text(encoding="utf-8")

    # libsbml may assign an auto-prefix (e.g. ns3) to the biology-qualifiers namespace.
    # Detect the actual prefix and rename ALL occurrences to bqbiol using str.replace
    # (more reliable than regex on libsbml's deterministic double-quoted output).
    m = re.search(r'xmlns:([A-Za-z][A-Za-z0-9_]*)\s*=\s*["\']' + re.escape(BQ_URI) + r'["\']', text)
    if m:
        pfx = m.group(1)
        if pfx != "bqbiol":
            # libsbml always writes double-quoted, no spaces around '='
            text = text.replace(f'xmlns:{pfx}="{BQ_URI}"', f'xmlns:bqbiol="{BQ_URI}"')
            # also handle single-quoted variant just in case
            text = text.replace(f"xmlns:{pfx}='{BQ_URI}'", f'xmlns:bqbiol="{BQ_URI}"')
            text = text.replace(f'<{pfx}:', '<bqbiol:')
            text = text.replace(f'</{pfx}:', '</bqbiol:')

    # Add any missing namespace declarations directly onto the <sbml> root tag.
    def _patch_sbml_tag(m2: re.Match) -> str:
        tag = m2.group(0)
        additions = ""
        if 'xmlns:pathwhiz=' not in tag:
            additions += f' xmlns:pathwhiz="{PW_URI}"'
        if 'xmlns:bqbiol=' not in tag:
            additions += f' xmlns:bqbiol="{BQ_URI}"'
        if 'xmlns:rdf=' not in tag:
            additions += f' xmlns:rdf="{RDF_URI}"'
        if not additions:
            return tag
        # Insert before the closing '>' of the opening tag
        return tag.rstrip(">").rstrip() + additions + ">"

    text = re.sub(r'<sbml\b[^>]*>', _patch_sbml_tag, text, count=1)
    sbml_path.write_text(text, encoding="utf-8")


def build_sbml(
    input_path: Path,
    sbml_path: Path,
    report_json_path: Path,
    report_txt_path: Path,
    *,
    default_compartment_name: str = "cell",
    db_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        import libsbml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("python-libsbml is required. Install python-libsbml.") from exc

    # Push DB credentials into env so _load_pathwhiz_db can pick them up
    if db_config:
        for env_key, cfg_key in [
            ("PATHBANK_DB_HOST", "host"),
            ("PATHBANK_DB_PORT", "port"),
            ("PATHBANK_DB_USER", "user"),
            ("PATHBANK_DB_PASSWORD", "password"),
            ("PATHBANK_DB_SCHEMA", "schema"),
        ]:
            val = db_config.get(cfg_key)
            if val is not None and not os.environ.get(env_key):
                os.environ[env_key] = str(val)

    pw_db = _load_pathwhiz_db()

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mapped input JSON must be an object.")

    report: Dict[str, Any] = {
        "hard_errors": [],
        "warnings": [],
        "defaults_applied": [],
        "counts": {"compartments": 0, "species": 0, "reactions": 0},
        "pathwhiz_id_stats": {"mysql_connected": bool(os.environ.get("PATHBANK_DB_HOST")), "compounds_matched": 0, "proteins_matched": 0, "species_no_id": 0},
    }
    data = deepcopy(payload)
    entities = _safe_dict(data.get("entities"))
    processes = _safe_dict(data.get("processes"))

    states_to_loc = _extract_state_compartments(data)
    raw_compounds, raw_proteins, known_element_collections = _known_entity_names(data)
    known_compounds, known_proteins = _resolve_cross_type_name_conflicts(
        raw_compounds,
        raw_proteins,
        processes,
        report,
    )
    known_compounds, known_proteins = _augment_entity_sets_from_processes(
        known_compounds,
        known_proteins,
        processes,
        report,
    )
    known_compounds, known_proteins = _drop_composite_entities(
        known_compounds,
        known_proteins,
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
    for ec in known_element_collections:
        if not entity_compartments.get(("element_collection", ec)):
            entity_compartments[("element_collection", ec)].add(default_compartment_name)
            report["defaults_applied"].append(
                {
                    "type": "element_collection_missing_location",
                    "name": ec,
                    "default_compartment": default_compartment_name,
                }
            )

    compound_rows = _dedupe_entity_rows(
        _safe_list(entities.get("compounds")),
        preferred_mapped_keys=["chebi", "hmdb", "kegg", "pubchem", "drugbank"],
    )
    protein_rows = _dedupe_entity_rows(
        list(_safe_list(entities.get("proteins"))) + list(_safe_list(entities.get("protein_complexes"))),
        preferred_mapped_keys=["uniprot"],
    )
    element_collection_rows = _dedupe_entity_rows(
        _safe_list(entities.get("element_collections")),
        preferred_mapped_keys=[],
    )
    entity_row_lookups: Dict[str, Dict[str, Dict[str, Any]]] = {
        "compound": compound_rows,
        "protein": protein_rows,
        "element_collection": element_collection_rows,
    }

    species_registry: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    species_id_to_meta: Dict[str, Dict[str, Any]] = {}

    def _register_species(
        *,
        kind: str,
        name: str,
        compartment_id: str,
        mapped_ids: Optional[Dict[str, Any]] = None,
    ) -> None:
        deduped_row = _safe_dict(_safe_dict(entity_row_lookups.get(kind)).get(_normalize(name)))
        canonical_name = (
            deduped_row.get("name").strip()
            if isinstance(deduped_row.get("name"), str) and deduped_row.get("name").strip()
            else name
        )
        canonical_mapped_ids = _safe_dict(deduped_row.get("mapped_ids")) or _safe_dict(mapped_ids)
        key = (kind, name, compartment_id)
        canonical_key = (kind, canonical_name, compartment_id)
        if canonical_key in species_registry:
            species_registry[key] = species_registry[canonical_key]
            return
        entity_meta = {"kind": kind, "name": canonical_name, "mapped_ids": canonical_mapped_ids}
        sid = sbml_species_id(entity_meta, compartment_id)
        sid_meta = {"kind": kind, "name": canonical_name, "compartment_id": compartment_id}
        if sid in species_id_to_meta and species_id_to_meta[sid] != sid_meta:
            sid = sanitize_sbml_id(f"{sid}_{_short_hash(canonical_name + compartment_id, 6)}")
        entity_class = deduped_row.get("class") or ""
        entity_provenance = deduped_row.get("provenance") or ""
        record = {"id": sid, "name": canonical_name, "kind": kind, "compartment_id": compartment_id, "mapped_ids": canonical_mapped_ids, "class": entity_class, "provenance": entity_provenance}
        species_registry[canonical_key] = record
        species_registry[key] = record
        species_id_to_meta[sid] = sid_meta

    for row in _unique_entity_rows(compound_rows):
        name = (row.get("name") or "").strip() if isinstance(row.get("name"), str) else ""
        if not name:
            continue
        aliases = _dedupe_preserve_strings(_safe_list(row.get("_dedupe_aliases")) + [name])
        locations: Set[str] = set()
        for alias in aliases:
            locations.update(entity_compartments.get(("compound", alias), set()))
        for loc in sorted(locations or {default_compartment_name}):
            cid = compartment_by_name.get(loc, compartment_by_name[default_compartment_name])
            _register_species(kind="compound", name=name, compartment_id=cid)

    for row in _unique_entity_rows(protein_rows):
        name = (row.get("name") or "").strip() if isinstance(row.get("name"), str) else ""
        if not name:
            continue
        aliases = _dedupe_preserve_strings(_safe_list(row.get("_dedupe_aliases")) + [name])
        locations: Set[str] = set()
        for alias in aliases:
            locations.update(entity_compartments.get(("protein", alias), set()))
        for loc in sorted(locations or {default_compartment_name}):
            cid = compartment_by_name.get(loc, compartment_by_name[default_compartment_name])
            _register_species(kind="protein", name=name, compartment_id=cid)

    for row in _unique_entity_rows(element_collection_rows):
        name = (row.get("name") or "").strip() if isinstance(row.get("name"), str) else ""
        if not name:
            continue
        aliases = _dedupe_preserve_strings(_safe_list(row.get("_dedupe_aliases")) + [name])
        locations_ec: Set[str] = set()
        for alias in aliases:
            locations_ec.update(entity_compartments.get(("element_collection", alias), set()))
        for loc in sorted(locations_ec or {default_compartment_name}):
            cid = compartment_by_name.get(loc, compartment_by_name[default_compartment_name])
            _register_species(kind="element_collection", name=name, compartment_id=cid)

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

        expanded_inputs: List[str] = []
        for name in inputs:
            if _is_composite_name(name):
                expanded_inputs.extend(_split_composite_name(name))
            else:
                expanded_inputs.append(name)
        expanded_outputs: List[str] = []
        for name in outputs:
            if _is_composite_name(name):
                expanded_outputs.extend(_split_composite_name(name))
            else:
                expanded_outputs.append(name)

        for name in expanded_inputs:
            kind = ""
            if name in known_compounds:
                kind = "compound"
            elif name in known_proteins:
                kind = "protein"
            elif name in known_element_collections:
                kind = "element_collection"
            if not kind:
                report["hard_errors"].append(
                    {"path": f"{pointer}/inputs", "reason": f"Unknown input entity '{name}'."}
                )
                unresolved = True
                continue
            key = (kind, name, compartment_id)
            if key not in species_registry:
                _register_species(kind=kind, name=name, compartment_id=compartment_id)
                report["warnings"].append(
                    {
                        "path": f"{pointer}/inputs",
                        "reason": f"Created missing species instance for input '{name}' ({kind}) in compartment '{compartment_id}'.",
                    }
                )
            reactant_ids.append(species_registry[key]["id"])

        for name in expanded_outputs:
            kind = ""
            if name in known_compounds:
                kind = "compound"
            elif name in known_proteins:
                kind = "protein"
            elif name in known_element_collections:
                kind = "element_collection"
            if not kind:
                report["hard_errors"].append(
                    {"path": f"{pointer}/outputs", "reason": f"Unknown output entity '{name}'."}
                )
                unresolved = True
                continue
            key = (kind, name, compartment_id)
            if key not in species_registry:
                _register_species(kind=kind, name=name, compartment_id=compartment_id)
                report["warnings"].append(
                    {
                        "path": f"{pointer}/outputs",
                        "reason": f"Created missing species instance for output '{name}' ({kind}) in compartment '{compartment_id}'.",
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
                "kind": "reaction",
                "class": reaction.get("class") or "biochemical_reaction",
                "provenance": reaction.get("provenance") or "",
            }
        )

    transports = _safe_list(processes.get("transports"))
    for tidx, transport in enumerate(transports):
        pointer = f"/processes/transports/{tidx}"
        if not isinstance(transport, dict):
            report["hard_errors"].append({"path": pointer, "reason": "Transport item is not an object."})
            continue
        cargo = (
            transport.get("cargo_complex")
            if isinstance(transport.get("cargo_complex"), str)
            else transport.get("cargo")
        )
        cargo = (cargo or "").strip() if isinstance(cargo, str) else ""
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
        cargo_items = _split_composite_name(cargo) if _is_composite_name(cargo) else [cargo]
        if not cargo_items:
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

        modifiers: List[str] = []
        for transporter in _safe_list(transport.get("transporters")):
            if not isinstance(transporter, dict):
                continue
            pname = _first_string([transporter.get("protein"), transporter.get("protein_complex"), transporter.get("name")])
            if pname and pname in known_proteins:
                source_key = ("protein", pname, source_cid)
                dest_key = ("protein", pname, dest_cid)
                chosen_key = source_key if source_key in species_registry else dest_key
                if chosen_key not in species_registry:
                    _register_species(kind="protein", name=pname, compartment_id=source_cid)
                    chosen_key = ("protein", pname, source_cid)
                if chosen_key in species_registry:
                    modifiers.append(species_registry[chosen_key]["id"])

        for cargo_item in cargo_items:
            if cargo_item not in known_compounds and cargo_item not in known_proteins and cargo_item not in known_element_collections:
                report["hard_errors"].append(
                    {"path": f"{pointer}/cargo", "reason": f"Unknown cargo '{cargo_item}'."}
                )
                continue

            if cargo_item in known_compounds:
                kind = "compound"
            elif cargo_item in known_element_collections:
                kind = "element_collection"
            else:
                kind = "protein"
            source_key = (kind, cargo_item, source_cid)
            dest_key = (kind, cargo_item, dest_cid)

            if source_key not in species_registry:
                _register_species(kind=kind, name=cargo_item, compartment_id=source_cid)
            if dest_key not in species_registry:
                _register_species(kind=kind, name=cargo_item, compartment_id=dest_cid)

            if source_key == dest_key:
                report["warnings"].append(
                    {
                        "path": pointer,
                        "reason": "Dropped degenerate transport with identical source and destination species.",
                    }
                )
                continue

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
                    "name": (
                        (transport.get("name") or f"transport_{cargo_item}").strip()
                        if isinstance(transport.get("name"), str)
                        else f"transport_{cargo_item}"
                    ),
                    "reactants": [species_registry[source_key]["id"]],
                    "products": [species_registry[dest_key]["id"]],
                    "modifiers": sorted(set(modifiers)),
                    "compartment_id": source_cid,
                    "kind": "transport",
                    "class": transport.get("class") or "transport_reaction",
                    "provenance": transport.get("provenance") or "",
                }
            )

    # 2b: Suppress orphan compartments — remove any compartment with no species assigned
    # (run after all reaction/transport building so on-the-fly species are counted).
    used_cids: Set[str] = {item["compartment_id"] for item in species_registry.values()}
    used_cids.add(compartment_by_name[default_compartment_name])
    orphan_locs = [loc for loc, cid in compartment_by_name.items() if cid not in used_cids]
    for loc in orphan_locs:
        report["warnings"].append({"path": "/compartments", "reason": f"Removed orphan compartment '{loc}' (no species assigned)."})
        del compartment_by_name[loc]

    # Build SBML document.
    # PathWhiz exports are commonly Level 3 Version 1; keep this for compatibility.
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("PathwayModel")
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
        # PathWhiz-produced SBML commonly sets compartment constant=false.
        comp.setConstant(False)
        comp.setSize(1.0)
        comp.setSpatialDimensions(3)
        comp.setUnits("UnitVol")
        comp.setSBOTerm(240)  # SBO:0000240 — physical compartment
        bs_id = _lookup_bs_id(pw_db, loc_name)
        bs_id_attr = f' pathwhiz:compartment_id="{bs_id}"' if bs_id is not None else ""
        comp.setAnnotation(
            '<annotation><pathwhiz:compartment xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
            f'{bs_id_attr} pathwhiz:compartment_type="biological_state"/></annotation>'
        )

    species_items_by_id: Dict[str, Dict[str, Any]] = {}
    for item in species_registry.values():
        species_items_by_id.setdefault(item["id"], item)
    species_items = sorted(species_items_by_id.values(), key=lambda item: item["id"])
    # Build sbml_species_id → PathWhiz entity ID map (used for reaction matching below)
    pw_species_id_map: Dict[str, int] = {}
    for item in species_items_by_id.values():
        mapped = item.get("mapped_ids") or {}
        is_compound = item.get("kind") == "compound"
        pw_id = (_lookup_compound_id(pw_db, mapped) if is_compound
                 else _lookup_protein_id(pw_db, mapped))
        if pw_id is not None:
            pw_species_id_map[item["id"]] = pw_id

    for item in species_items:
        sp = model.createSpecies()
        sid = item["id"]
        sp.setId(sid)
        sp.setName(item["name"])
        sp.setCompartment(item["compartment_id"])
        sp.setBoundaryCondition(False)
        sp.setHasOnlySubstanceUnits(True)
        sp.setConstant(False)
        sp.setInitialAmount(1.0)
        sp.setSubstanceUnits("Unit1")
        # metaid required for RDF CVTerm annotations
        sp.setMetaId(f"_meta_{sid}")
        # SBO:0000247 simple chemical, SBO:0000245 macromolecule (protein/EC)
        is_compound = item.get("kind") == "compound"
        sp.setSBOTerm(247 if is_compound else 245)
        mapped = item.get("mapped_ids") or {}
        pw_sp_id = (_lookup_compound_id(pw_db, mapped) if is_compound
                    else _lookup_protein_id(pw_db, mapped))
        sp_type = "compound" if is_compound else "protein"
        if pw_sp_id is not None:
            if is_compound:
                report["pathwhiz_id_stats"]["compounds_matched"] += 1
            else:
                report["pathwhiz_id_stats"]["proteins_matched"] += 1
        else:
            report["pathwhiz_id_stats"]["species_no_id"] += 1
        # Set pathwhiz:species annotation (pathwhiz block only — RDF added via CVTerm below)
        pw_id_attr = f' pathwhiz:species_id="{pw_sp_id}"' if pw_sp_id is not None else ""
        sp_class = item.get("class") or sp_type
        sp_provenance = item.get("provenance") or ""
        sp_class_attr = f' pathwhiz:class="{sp_class}"'
        sp_prov_attr = f' pathwhiz:provenance="{sp_provenance}"' if sp_provenance else ""
        sp.setAnnotation(
            f'<annotation><pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
            f'{pw_id_attr} pathwhiz:species_type="{sp_type}"{sp_class_attr}{sp_prov_attr}/></annotation>'
        )
        # Add cross-database xrefs via CVTerm (requires metaid)
        _add_cv_terms(sp, mapped, libsbml)

    for plan in sorted(reaction_plans, key=lambda item: item["id"]):
        rxn = model.createReaction()
        rxn.setId(plan["id"])
        if plan["name"]:
            rxn.setName(plan["name"])
        if plan.get("compartment_id"):
            rxn.setCompartment(plan["compartment_id"])
        rxn.setReversible(False)
        # SBML Level 3 Version 1 requires an explicit 'fast' attribute on reactions.
        rxn.setFast(False)
        # SBO terms: SBO:0000176 biochemical reaction, SBO:0000185 transport
        rxn_kind = plan.get("kind", "reaction")
        rxn.setSBOTerm(176 if rxn_kind == "reaction" else 185)
        r_pw_ids = [pw_species_id_map[s] for s in plan["reactants"] if s in pw_species_id_map]
        p_pw_ids = [pw_species_id_map[s] for s in plan["products"] if s in pw_species_id_map]
        pw_rxn_id = _lookup_reaction_id(pw_db, r_pw_ids, p_pw_ids)
        rxn_id_attr = f' pathwhiz:reaction_id="{pw_rxn_id}"' if pw_rxn_id is not None else ""
        rxn_class = plan.get("class") or rxn_kind
        rxn_provenance = plan.get("provenance") or ""
        rxn_class_attr = f' pathwhiz:class="{rxn_class}"'
        rxn_prov_attr = f' pathwhiz:provenance="{rxn_provenance}"' if rxn_provenance else ""
        rxn.setAnnotation(
            f'<annotation><pathwhiz:reaction xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
            f'{rxn_id_attr} pathwhiz:reaction_type="{rxn_kind}"{rxn_class_attr}{rxn_prov_attr}/></annotation>'
        )
        reactant_counts = Counter(plan["reactants"])
        for sid, stoich in sorted(reactant_counts.items()):
            ref = rxn.createReactant()
            ref.setSpecies(sid)
            ref.setConstant(False)
            ref.setStoichiometry(float(stoich))
            ref.setSBOTerm(15)  # SBO:0000015 substrate
        product_counts = Counter(plan["products"])
        for sid, stoich in sorted(product_counts.items()):
            ref = rxn.createProduct()
            ref.setSpecies(sid)
            ref.setConstant(False)
            ref.setStoichiometry(float(stoich))
            ref.setSBOTerm(11)  # SBO:0000011 product
        for sid in plan["modifiers"]:
            ref = rxn.createModifier()
            ref.setSpecies(sid)
            ref.setSBOTerm(460)  # SBO:0000460 enzymatic catalyst

    report["counts"]["compartments"] = model.getNumCompartments()
    report["counts"]["species"] = model.getNumSpecies()
    report["counts"]["reactions"] = model.getNumReactions()

    libsbml.writeSBMLToFile(doc, str(sbml_path))

    # Inject root-level namespace declarations that PathWhiz expects on <sbml>
    _inject_root_namespaces(sbml_path)

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
