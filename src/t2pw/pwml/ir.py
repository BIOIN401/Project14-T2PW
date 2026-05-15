from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ENTITY_BUCKETS = {
    "compound": "compounds",
    "protein": "proteins",
    "nucleic_acid": "nucleic_acids",
    "element_collection": "element_collections",
    "protein_complex": "protein_complexes",
    "bound": "bounds",
}

PROCESS_BUCKETS = {
    "reaction": "reactions",
    "reaction_coupled_transport": "reaction_coupled_transports",
    "transport": "transports",
    "interaction": "interactions",
    "sub_pathway": "sub_pathways",
}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _norm(value: Any) -> str:
    text = _canonical(value).casefold()
    return re.sub(r"[^a-z0-9:+ ]+", " ", text).strip()


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _first_nonempty(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    meta = _safe_dict(row.get("mapping_meta"))
    for key in keys:
        if key in meta and meta.get(key) not in (None, ""):
            return meta.get(key)
    mapped = _safe_dict(row.get("mapped_ids"))
    for key in keys:
        if key in mapped and mapped.get(key) not in (None, ""):
            return mapped.get(key)
    candidates = _safe_list(meta.get("candidates"))
    if candidates and isinstance(candidates[0], dict):
        top = candidates[0]
        for key in keys:
            if key in top and top.get(key) not in (None, ""):
                return top.get(key)
    return None


def _db_id(row: Dict[str, Any], keys: Sequence[str]) -> Optional[int]:
    return _to_int(_first_nonempty(row, keys))


def _new_report() -> Dict[str, Any]:
    return {
        "ok": True,
        "errors": [],
        "warnings": [],
        "unresolved": {
            "db_identities": [],
            "entity_references": [],
            "biological_state_references": [],
        },
        "counts": {},
        "generated_default_state": False,
    }


def _add_issue(
    report: Dict[str, Any],
    severity: str,
    code: str,
    message: str,
    *,
    pointer: str = "",
    **extra: Any,
) -> None:
    issue = {"code": code, "message": message}
    if pointer:
        issue["pointer"] = pointer
    issue.update(extra)
    bucket = "errors" if severity == "error" else "warnings"
    report.setdefault(bucket, []).append(issue)
    if severity == "error":
        report["ok"] = False


def _empty_ir(
    pathway_name: str,
    pathway_subject: str,
    width: int,
    height: int,
) -> Dict[str, Any]:
    return {
        "pathway": {
            "key": "pathway_1",
            "name": pathway_name,
            "subject": pathway_subject,
            "width": int(width),
            "height": int(height),
        },
        "species": [],
        "subcellular_locations": [],
        "cell_types": [],
        "tissues": [],
        "biological_states": [],
        "entities": {
            "compounds": [],
            "proteins": [],
            "nucleic_acids": [],
            "element_collections": [],
            "protein_complexes": [],
            "bounds": [],
        },
        "processes": {
            "reactions": [],
            "reaction_coupled_transports": [],
            "transports": [],
            "interactions": [],
            "sub_pathways": [],
        },
        "locations": [],
        "protein_complex_visualizations": [],
        "bound_visualizations": [],
        "edges": [],
        "process_visualizations": [],
    }


def is_pwml_ir(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        isinstance(payload.get("pathway"), dict)
        and isinstance(payload.get("entities"), dict)
        and isinstance(payload.get("processes"), dict)
        and isinstance(payload.get("locations"), list)
        and isinstance(payload.get("process_visualizations"), list)
    )


class _KeyFactory:
    def __init__(self) -> None:
        self._counters: Dict[str, int] = defaultdict(int)

    def next(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}_{self._counters[prefix]}"


def _dedupe_named_rows(
    rows: Iterable[Any],
    *,
    key_prefix: str,
    report: Dict[str, Any],
    pointer_prefix: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    by_norm: Dict[str, Dict[str, Any]] = {}
    idx = 0
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        name = _canonical(raw.get("name"))
        if not name:
            continue
        n = _norm(name)
        if n in by_norm:
            _add_issue(
                report,
                "warning",
                "duplicate_named_record",
                f"Duplicate record for '{name}' ignored in PWML IR.",
                pointer=f"{pointer_prefix}/{idx}",
            )
            idx += 1
            continue
        record = dict(raw)
        record["name"] = name
        record["key"] = f"{key_prefix}_{len(out) + 1}"
        out.append(record)
        by_norm[n] = record
        idx += 1
    return out, by_norm


def _copy_common_entity_fields(record: Dict[str, Any], raw: Dict[str, Any]) -> None:
    for key in [
        "mapped_ids",
        "mapping_meta",
        "class",
        "confidence",
        "provenance",
        "source_refs",
        "evidence",
        "organism",
        "species_id",
        "pathbank_species_id",
        "hmdb_id",
        "kegg_id",
        "chebi_id",
        "pubchem_cid",
        "cas",
        "biocyc_id",
        "chemspider_id",
        "moldb_inchi",
        "moldb_inchikey",
        "short_name",
        "synonyms",
        "type",
        "compound_class",
        "compound_type",
        "uniprot",
        "gene_name",
        "ec_number",
        "ec_numbers",
        "element_states",
        "components",
    ]:
        if key in raw:
            record[key] = raw[key]


def _component_record(raw: Dict[str, Any], key: str, db_keys: Sequence[str]) -> Dict[str, Any]:
    record: Dict[str, Any] = {"key": key, "name": _canonical(raw.get("name"))}
    pathwhiz_id = _db_id(raw, db_keys)
    if pathwhiz_id is not None:
        record["pathwhiz_id"] = pathwhiz_id
    for src, dst in [
        ("taxonomy_id", "taxonomy_id"),
        ("taxonomy-id", "taxonomy_id"),
        ("ontology_id", "ontology_id"),
        ("ontology-id", "ontology_id"),
        ("classification", "classification"),
        ("common_name", "common_name"),
        ("common-name", "common_name"),
    ]:
        value = raw.get(src)
        if value not in (None, ""):
            record[dst] = value
    return record


def _entity_record(
    raw: Dict[str, Any],
    key: str,
    db_keys: Sequence[str],
    db_field: str,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"key": key, "name": _canonical(raw.get("name"))}
    db_value = _db_id(raw, db_keys)
    if db_value is not None:
        record["pathwhiz_id"] = db_value
        record[db_field] = db_value
    _copy_common_entity_fields(record, raw)
    return record


def _direction(value: Any) -> str:
    text = _canonical(value).casefold()
    if text in {"<", "left", "reverse"}:
        return "<"
    if text in {"<>", "<=>", "both", "reversible"}:
        return "<>"
    return ">"


def _coerce_participants(value: Any) -> List[Dict[str, Any]]:
    items = value if isinstance(value, list) else []
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            name = _canonical(item)
            if name:
                out.append({"name": name, "stoichiometry": 1})
        elif isinstance(item, dict):
            name = _canonical(
                item.get("name")
                or item.get("compound")
                or item.get("protein")
                or item.get("protein_complex")
                or item.get("element")
                or item.get("entity")
            )
            if not name:
                continue
            stoich = _to_int(item.get("stoichiometry") or item.get("coefficient")) or 1
            out.append({"name": name, "stoichiometry": max(1, stoich)})
    return out


def _component_protein_name(component: Any) -> str:
    if isinstance(component, str):
        return _canonical(component)
    if not isinstance(component, dict):
        return ""
    return _canonical(
        component.get("protein")
        or component.get("protein_name")
        or component.get("name")
        or component.get("component")
        or component.get("entity")
    )


def _component_stoichiometry(component: Any) -> Optional[int]:
    if isinstance(component, str):
        return 1
    if not isinstance(component, dict):
        return None
    stoich = _to_int(component.get("stoichiometry") or component.get("coefficient"))
    if stoich is None or stoich < 1:
        return None
    return stoich


def _actor_name_and_hint(item: Any) -> Tuple[str, str, str]:
    if isinstance(item, str):
        return _canonical(item), "", ""
    if not isinstance(item, dict):
        return "", "", ""
    for field, hint in [
        ("protein_complex", "protein_complex"),
        ("protein-complex", "protein_complex"),
        ("protein", "protein"),
        ("entity", _canonical(item.get("entity_type")).lower()),
        ("name", _canonical(item.get("entity_type")).lower()),
    ]:
        value = _canonical(item.get(field))
        if value:
            return value, hint, _canonical(item.get("role")).lower()
    return "", "", _canonical(item.get("role")).lower()


def _entity_type_from_location_bucket(bucket: str) -> Tuple[str, str]:
    return {
        "compound_locations": ("compound", "compound"),
        "protein_locations": ("protein", "protein"),
        "nucleic_acid_locations": ("nucleic_acid", "nucleic_acid"),
        "element_collection_locations": ("element_collection", "element_collection"),
    }.get(bucket, ("", ""))


def build_pwml_ir(
    payload: dict,
    *,
    pathway_name: str = "Generated Pathway",
    pathway_subject: str = "Metabolic",
    strict_db: bool = True,
    width: int = 3200,
    height: int = 1400,
) -> tuple[dict, dict]:
    report = _new_report()
    if not isinstance(payload, dict):
        ir = _empty_ir(pathway_name, pathway_subject, width, height)
        _add_issue(report, "error", "invalid_payload", "PWML IR input payload must be an object.")
        return ir, report

    ir = _empty_ir(pathway_name, pathway_subject, width, height)
    keys = _KeyFactory()
    entities = _safe_dict(payload.get("entities"))
    processes = _safe_dict(payload.get("processes"))

    pathway_meta = _safe_dict(payload.get("metadata"))
    if pathway_meta.get("pathway_name") and pathway_name == "Generated Pathway":
        ir["pathway"]["name"] = _canonical(pathway_meta.get("pathway_name"))
    if pathway_meta.get("pathway_subject") and pathway_subject == "Metabolic":
        ir["pathway"]["subject"] = _canonical(pathway_meta.get("pathway_subject"))

    component_specs = [
        ("species", "species", "sp", ["pathbank_species_id", "pw_species_id", "pathwhiz_id"]),
        (
            "subcellular_locations",
            "subcellular_locations",
            "scl",
            ["pathbank_subcellular_location_id", "pw_subcellular_location_id", "pathwhiz_id"],
        ),
        ("cell_types", "cell_types", "ct", ["pathbank_cell_type_id", "pw_cell_type_id", "pathwhiz_id"]),
        ("tissues", "tissues", "tis", ["pathbank_tissue_id", "pw_tissue_id", "pathwhiz_id"]),
    ]
    component_by_name: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for source_key, ir_key, prefix, db_keys in component_specs:
        rows, lookup = _dedupe_named_rows(
            _safe_list(entities.get(source_key)),
            key_prefix=prefix,
            report=report,
            pointer_prefix=f"/entities/{source_key}",
        )
        ir[ir_key] = [_component_record(row, row["key"], db_keys) for row in rows]
        component_by_name[source_key] = {_norm(row["name"]): row for row in ir[ir_key]}

    entity_specs = [
        (
            "compounds",
            "compound",
            "cmp",
            ["pathbank_compound_id", "pw_compound_id", "pathwhiz_id"],
            "pathbank_compound_id",
            True,
        ),
        (
            "proteins",
            "protein",
            "prot",
            ["pathbank_protein_id", "pw_protein_id", "pathwhiz_id"],
            "pathbank_protein_id",
            True,
        ),
        (
            "nucleic_acids",
            "nucleic_acid",
            "na",
            ["pathbank_nucleic_acid_id", "pw_nucleic_acid_id", "pathwhiz_id"],
            "pathbank_nucleic_acid_id",
            False,
        ),
        (
            "element_collections",
            "element_collection",
            "ec",
            ["pathbank_element_collection_id", "pw_element_collection_id", "pathwhiz_id"],
            "pathbank_element_collection_id",
            False,
        ),
        (
            "protein_complexes",
            "protein_complex",
            "pc",
            ["pathbank_complex_id", "pw_complex_id", "pathwhiz_id"],
            "pathbank_complex_id",
            True,
        ),
    ]

    entity_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    entity_by_key: Dict[str, Dict[str, Any]] = {}
    for source_key, entity_type, prefix, db_keys, db_field, strict_required in entity_specs:
        rows, _ = _dedupe_named_rows(
            _safe_list(entities.get(source_key)),
            key_prefix=prefix,
            report=report,
            pointer_prefix=f"/entities/{source_key}",
        )
        bucket = ENTITY_BUCKETS[entity_type]
        for row in rows:
            rec = _entity_record(row, row["key"], db_keys, db_field)
            if strict_db and strict_required and rec.get("pathwhiz_id") is None:
                issue = {
                    "entity_type": entity_type,
                    "entity_key": rec["key"],
                    "name": rec["name"],
                    "required_fields": list(db_keys),
                }
                report["unresolved"]["db_identities"].append(issue)
                _add_issue(
                    report,
                    "error" if strict_db else "warning",
                    "missing_db_identity",
                    f"{entity_type} '{rec['name']}' has no required DB-backed PWML identity.",
                    pointer=f"/entities/{source_key}",
                    **issue,
                )
            ir["entities"][bucket].append(rec)
            typed = dict(rec)
            typed["entity_type"] = entity_type
            entity_by_key[typed["key"]] = typed
            entity_by_name[_norm(rec["name"])].append(typed)

    proteins_by_key = {
        str(protein.get("key")): protein
        for protein in ir["entities"]["proteins"]
        if isinstance(protein, dict) and protein.get("key")
    }
    proteins_by_name = {
        _norm(protein.get("name")): protein
        for protein in ir["entities"]["proteins"]
        if isinstance(protein, dict) and _norm(protein.get("name"))
    }
    proteins_by_db_id: Dict[int, Dict[str, Any]] = {}
    for protein in ir["entities"]["proteins"]:
        if not isinstance(protein, dict):
            continue
        for db_key in ["pathbank_protein_id", "pw_protein_id", "pathwhiz_id"]:
            db_id = _to_int(protein.get(db_key))
            if db_id is not None:
                proteins_by_db_id[db_id] = protein

    for pc_idx, pc in enumerate(ir["entities"]["protein_complexes"]):
        raw_components = _safe_list(pc.get("components"))
        if not raw_components:
            _add_issue(
                report,
                "error",
                "protein_complex_missing_components",
                f"Protein complex '{pc.get('name')}' has no protein components.",
                pointer=f"/entities/protein_complexes/{pc_idx}/components",
                entity_key=pc.get("key"),
                entity_name=pc.get("name"),
            )
            pc["components"] = []
            continue

        structured_components: List[Dict[str, Any]] = []
        for comp_idx, component in enumerate(raw_components):
            pointer = f"/entities/protein_complexes/{pc_idx}/components/{comp_idx}"
            protein: Optional[Dict[str, Any]] = None
            if isinstance(component, dict):
                protein_key = _canonical(component.get("protein_key") or component.get("entity_key"))
                if protein_key:
                    protein = proteins_by_key.get(protein_key)
                if protein is None:
                    comp_db_id = _db_id(
                        component,
                        ["pathbank_protein_id", "pw_protein_id", "pathwhiz_id", "protein_id"],
                    )
                    if comp_db_id is not None:
                        protein = proteins_by_db_id.get(comp_db_id)
            comp_name = _component_protein_name(component)
            if protein is None and comp_name:
                protein = proteins_by_name.get(_norm(comp_name))

            stoich = _component_stoichiometry(component)
            if stoich is None:
                _add_issue(
                    report,
                    "error",
                    "component_missing_stoichiometry",
                    f"Component[{comp_idx}] in complex '{pc.get('name')}' is missing stoichiometry.",
                    pointer=pointer,
                    complex_key=pc.get("key"),
                    complex_name=pc.get("name"),
                    component_name=comp_name,
                )
                continue
            if protein is None:
                _add_issue(
                    report,
                    "error",
                    "component_protein_unresolved",
                    f"Component '{comp_name or comp_idx}' in complex '{pc.get('name')}' does not reference an existing protein.",
                    pointer=pointer,
                    complex_key=pc.get("key"),
                    complex_name=pc.get("name"),
                    component_name=comp_name,
                )
                continue

            structured_components.append(
                {
                    "protein_key": str(protein["key"]),
                    "stoichiometry": stoich,
                }
            )

        pc["components"] = structured_components
        if not structured_components:
            _add_issue(
                report,
                "error",
                "protein_complex_missing_components",
                f"Protein complex '{pc.get('name')}' has no resolved protein components.",
                pointer=f"/entities/protein_complexes/{pc_idx}/components",
                entity_key=pc.get("key"),
                entity_name=pc.get("name"),
            )

    enzyme_complex_by_protein_key: Dict[str, Dict[str, Any]] = {}
    for pc in ir["entities"]["protein_complexes"]:
        if not isinstance(pc, dict):
            continue
        components = _safe_list(pc.get("components"))
        if len(components) != 1 or not isinstance(components[0], dict):
            continue
        protein_key = str(components[0].get("protein_key") or "")
        if not protein_key or protein_key in enzyme_complex_by_protein_key:
            continue
        typed_pc = dict(pc)
        typed_pc["entity_type"] = "protein_complex"
        enzyme_complex_by_protein_key[protein_key] = typed_pc

    def resolve_component(source_key: str, value: Any, pointer: str) -> Optional[str]:
        name = _canonical(value)
        if not name:
            return None
        found = component_by_name.get(source_key, {}).get(_norm(name))
        if found:
            return str(found["key"])
        report["unresolved"]["biological_state_references"].append(
            {"component_type": source_key, "name": name, "pointer": pointer}
        )
        _add_issue(
            report,
            "error" if strict_db else "warning",
            "unresolved_biological_state_component",
            f"Biological state component '{name}' was not found in entities.{source_key}.",
            pointer=pointer,
            component_type=source_key,
            name=name,
        )
        return None

    raw_states = [row for row in _safe_list(payload.get("biological_states")) if isinstance(row, dict)]
    if not raw_states:
        report["generated_default_state"] = True
        context_exists = any(ir[key] for key in ["species", "subcellular_locations", "cell_types", "tissues"])
        raw_states = [{"name": "Default state"}]
        _add_issue(
            report,
            "warning",
            "generated_default_state",
            "No biological_states records were present; generated a default PWML biological state.",
        )
        if strict_db and not context_exists:
            _add_issue(
                report,
                "warning",
                "default_state_without_context",
                "Generated default biological state has no species, subcellular location, cell type, or tissue context.",
            )

    state_by_name: Dict[str, Dict[str, Any]] = {}
    for idx, raw in enumerate(raw_states):
        name = _canonical(raw.get("name")) or f"State {idx + 1}"
        state_key = f"bs_{idx + 1}"
        species_key = resolve_component("species", raw.get("species"), f"/biological_states/{idx}/species")
        subcellular_key = resolve_component(
            "subcellular_locations",
            raw.get("subcellular_location")
            or raw.get("subcellular-location")
            or raw.get("compartment")
            or raw.get("compartment_canonical"),
            f"/biological_states/{idx}/subcellular_location",
        )
        cell_type_key = resolve_component("cell_types", raw.get("cell_type"), f"/biological_states/{idx}/cell_type")
        tissue_key = resolve_component("tissues", raw.get("tissue"), f"/biological_states/{idx}/tissue")

        # If the state is named after one component, use that as deterministic context.
        if not species_key:
            species_key = (component_by_name.get("species") or {}).get(_norm(name), {}).get("key")
        if not subcellular_key:
            subcellular_key = (component_by_name.get("subcellular_locations") or {}).get(_norm(name), {}).get("key")
        if not cell_type_key:
            cell_type_key = (component_by_name.get("cell_types") or {}).get(_norm(name), {}).get("key")
        if not tissue_key:
            tissue_key = (component_by_name.get("tissues") or {}).get(_norm(name), {}).get("key")

        state = {
            "key": state_key,
            "name": name,
            "species_key": species_key,
            "subcellular_location_key": subcellular_key,
            "cell_type_key": cell_type_key,
            "tissue_key": tissue_key,
        }
        if not species_key:
            _add_issue(
                report,
                "error",
                "biological_state_missing_species",
                f"Biological state '{name}' has no resolved species reference.",
                pointer=f"/biological_states/{idx}",
                biological_state_key=state_key,
            )
        if not subcellular_key:
            _add_issue(
                report,
                "error",
                "biological_state_missing_subcellular_location",
                f"Biological state '{name}' has no resolved subcellular location reference.",
                pointer=f"/biological_states/{idx}",
                biological_state_key=state_key,
            )
        if not any([species_key, subcellular_key, cell_type_key, tissue_key]):
            _add_issue(
                report,
                "warning",
                "biological_state_without_components",
                f"Biological state '{name}' has no resolved species/location/cell/tissue references.",
                pointer=f"/biological_states/{idx}",
                biological_state_key=state_key,
            )
        ir["biological_states"].append(state)
        state_by_name[_norm(name)] = state

    default_bs_key = ir["biological_states"][0]["key"] if ir["biological_states"] else None

    def resolve_state(value: Any, pointer: str, *, required: bool = False) -> Optional[str]:
        name = _canonical(value)
        if not name:
            if required and default_bs_key:
                return default_bs_key
            return default_bs_key
        found = state_by_name.get(_norm(name))
        if found:
            return str(found["key"])
        report["unresolved"]["biological_state_references"].append(
            {"component_type": "biological_state", "name": name, "pointer": pointer}
        )
        _add_issue(
            report,
            "error" if strict_db or required else "warning",
            "unresolved_biological_state",
            f"Biological state '{name}' was not found.",
            pointer=pointer,
            biological_state=name,
        )
        return default_bs_key

    preferred_order = {
        "reaction_member": ["compound", "nucleic_acid", "element_collection", "protein_complex", "protein"],
        "enzyme": ["protein_complex", "protein"],
        "transporter": ["protein_complex", "protein"],
        "interaction": ["protein", "protein_complex", "compound", "nucleic_acid", "element_collection"],
    }

    def resolve_entity(name: Any, role: str, pointer: str, hint: str = "") -> Optional[Dict[str, Any]]:
        clean = _canonical(name)
        if not clean:
            return None
        candidates = entity_by_name.get(_norm(clean), [])
        if not candidates:
            report["unresolved"]["entity_references"].append({"name": clean, "role": role, "pointer": pointer})
            _add_issue(
                report,
                "error",
                "unresolved_entity_reference",
                f"Process member '{clean}' was not found in entity registries.",
                pointer=pointer,
                name=clean,
                role=role,
            )
            return None
        ordered = [hint] if hint in ENTITY_BUCKETS else []
        ordered.extend(preferred_order.get(role, []))
        for wanted in ordered:
            for candidate in candidates:
                if candidate.get("entity_type") == wanted:
                    return candidate
        if len(candidates) > 1:
            _add_issue(
                report,
                "warning",
                "ambiguous_entity_reference",
                f"Process member '{clean}' matched multiple entity types; using {candidates[0]['entity_type']}.",
                pointer=pointer,
                name=clean,
                entity_types=[c["entity_type"] for c in candidates],
            )
        return candidates[0]

    def ensure_enzyme_complex_entity(entity: Dict[str, Any], pointer: str) -> Dict[str, Any]:
        if entity.get("entity_type") == "protein_complex":
            return entity
        if entity.get("entity_type") != "protein":
            return entity
        protein_key = str(entity.get("key") or "")
        if not protein_key:
            return entity
        existing = enzyme_complex_by_protein_key.get(protein_key)
        if existing:
            return existing

        protein_name = _canonical(entity.get("name")) or protein_key
        base_name = f"{protein_name} complex"
        complex_name = base_name
        suffix = 2
        while any(candidate.get("entity_type") == "protein_complex" for candidate in entity_by_name.get(_norm(complex_name), [])):
            complex_name = f"{base_name} {suffix}"
            suffix += 1
        complex_record: Dict[str, Any] = {
            "key": f"pc_{len(ir['entities']['protein_complexes']) + 1}",
            "name": complex_name,
            "components": [{"protein_key": protein_key, "stoichiometry": 1}],
            "mapping_meta": {"resolution": {"status": "novel", "issue": "enzyme_wrapped_direct_protein"}},
        }
        if entity.get("species_id"):
            complex_record["species_id"] = entity.get("species_id")
        ir["entities"]["protein_complexes"].append(complex_record)
        typed_pc = dict(complex_record)
        typed_pc["entity_type"] = "protein_complex"
        entity_by_key[typed_pc["key"]] = typed_pc
        entity_by_name[_norm(complex_name)].append(typed_pc)
        enzyme_complex_by_protein_key[protein_key] = typed_pc
        _add_issue(
            report,
            "warning",
            "enzyme_protein_wrapped_as_complex",
            f"Reaction enzyme protein '{protein_name}' was represented as a novel single-component protein complex.",
            pointer=pointer,
            protein_key=protein_key,
            protein_complex_key=complex_record["key"],
            protein_complex_name=complex_name,
        )
        return typed_pc

    location_by_tuple: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def ensure_location(
        entity_type: str,
        entity_key: str,
        biological_state_key: Optional[str],
        *,
        x: int,
        y: int,
    ) -> Optional[Dict[str, Any]]:
        if not biological_state_key:
            return None
        entity = entity_by_key.get(entity_key)
        if not entity:
            return None
        cache_key = (entity_type, entity_key, biological_state_key)
        if cache_key in location_by_tuple:
            return location_by_tuple[cache_key]
        location_type = {
            "compound": "compound_location",
            "protein": "protein_location",
            "protein_complex": "protein_complex_location",
            "nucleic_acid": "nucleic_acid_location",
            "element_collection": "element_collection_location",
        }.get(entity_type)
        if not location_type:
            return None
        prefix = {
            "compound": "cl",
            "protein": "pl",
            "protein_complex": "pcl",
            "nucleic_acid": "nal",
            "element_collection": "ecl",
        }[entity_type]
        loc = {
            "key": keys.next(prefix),
            "type": location_type,
            "entity_type": entity_type,
            "entity_key": entity_key,
            "biological_state_key": biological_state_key,
            "x": int(x),
            "y": int(y),
            "zindex": 10,
            "visualization_template_id": 3 if entity_type == "compound" else 4,
            "hidden": False,
        }
        ir["locations"].append(loc)
        location_by_tuple[cache_key] = loc
        return loc

    def add_edge(x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
        edge = {
            "key": keys.next("edge"),
            "path": f"M{x1} {y1} L{x2} {y2}",
            "visualization_template_id": 5,
            "zindex": 18,
            "hidden": False,
        }
        ir["edges"].append(edge)
        return edge

    for idx, state in enumerate(ir["biological_states"]):
        ir["bound_visualizations"].append(
            {
                "key": keys.next("bv"),
                "biological_state_key": state["key"],
                "x": 0,
                "y": max(0, int(idx * (height / max(1, len(ir["biological_states"]))))),
                "width": int(width),
                "height": max(120, int(height / max(1, len(ir["biological_states"])))),
                "zindex": 1,
                "hidden": False,
            }
        )

    # Honor explicit element_locations as visual references without duplicating entities.
    element_locations = _safe_dict(payload.get("element_locations"))
    explicit_loc_idx = 0
    for bucket, rows in element_locations.items():
        entity_type, name_field = _entity_type_from_location_bucket(bucket)
        if not entity_type:
            continue
        for row_idx, raw in enumerate(_safe_list(rows)):
            if not isinstance(raw, dict):
                continue
            entity = resolve_entity(
                raw.get(name_field),
                "reaction_member",
                f"/element_locations/{bucket}/{row_idx}/{name_field}",
                hint=entity_type,
            )
            if not entity:
                continue
            bs_key = resolve_state(raw.get("biological_state"), f"/element_locations/{bucket}/{row_idx}/biological_state")
            explicit_loc_idx += 1
            ensure_location(
                entity_type,
                entity["key"],
                bs_key,
                x=40 + (explicit_loc_idx % 8) * 180,
                y=80 + (explicit_loc_idx // 8) * 100,
            )

    process_by_name: Dict[str, Dict[str, Any]] = {}
    process_by_key: Dict[str, Dict[str, Any]] = {}

    def process_xy(index: int) -> Tuple[int, int]:
        usable_w = max(800, width - 240)
        col_w = 360
        cols = max(1, usable_w // col_w)
        col = index % cols
        row = index // cols
        return 180 + col * col_w, 220 + row * 220

    reaction_index = 0
    for ridx, raw in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(raw, dict):
            continue
        reaction_index += 1
        rx_key = f"rxn_{reaction_index}"
        bs_key = resolve_state(raw.get("biological_state"), f"/processes/reactions/{ridx}/biological_state", required=True)
        rx_x, rx_y = process_xy(reaction_index - 1)
        reaction = {
            "key": rx_key,
            "type": "reaction",
            "name": _canonical(raw.get("name")) or f"Reaction {reaction_index}",
            "direction": _direction(raw.get("direction")),
            "biological_state_key": bs_key,
            "left": [],
            "right": [],
            "enzymes": [],
        }
        members_for_viz: List[Dict[str, Any]] = []

        for side_name, raw_key, x_offset in [("left", "inputs", -145), ("right", "outputs", 145)]:
            for midx, part in enumerate(_coerce_participants(raw.get(raw_key))):
                entity = resolve_entity(
                    part["name"],
                    "reaction_member",
                    f"/processes/reactions/{ridx}/{raw_key}/{midx}",
                )
                if not entity:
                    continue
                member_key = f"{rx_key}_{side_name}_{len(reaction[side_name]) + 1}"
                member = {
                    "key": member_key,
                    "entity_type": entity["entity_type"],
                    "entity_key": entity["key"],
                    "stoichiometry": int(part.get("stoichiometry") or 1),
                }
                reaction[side_name].append(member)
                loc = ensure_location(
                    entity["entity_type"],
                    entity["key"],
                    bs_key,
                    x=rx_x + x_offset,
                    y=rx_y + (len(reaction[side_name]) - 1) * 46,
                )
                if loc:
                    edge = add_edge(int(loc["x"]) + 60, int(loc["y"]) + 20, rx_x, rx_y)
                    members_for_viz.append(
                        {
                            "process_member_key": member_key,
                            "role": side_name,
                            "member_type": entity["entity_type"],
                            "location_key": loc["key"],
                            "edge_key": edge["key"],
                        }
                    )

        enzyme_sources: List[Any] = []
        enzyme_sources.extend(_safe_list(raw.get("enzymes")))
        for mod in _safe_list(raw.get("modifiers")):
            if not isinstance(mod, dict):
                continue
            role = _canonical(mod.get("role")).lower()
            if role in {"catalyst", "activator", "inhibitor", "enzyme"}:
                enzyme_sources.append(mod)
        for eidx, actor in enumerate(enzyme_sources):
            name, hint, role = _actor_name_and_hint(actor)
            if not name:
                if isinstance(actor, dict) and actor.get("evidence"):
                    _add_issue(
                        report,
                        "warning",
                        "enzyme_without_resolvable_actor",
                        "Reaction enzyme/modifier evidence did not include an explicit entity reference.",
                        pointer=f"/processes/reactions/{ridx}/enzymes/{eidx}",
                    )
                continue
            entity = resolve_entity(name, "enzyme", f"/processes/reactions/{ridx}/enzymes/{eidx}", hint=hint)
            if not entity:
                continue
            entity = ensure_enzyme_complex_entity(entity, f"/processes/reactions/{ridx}/enzymes/{eidx}")
            member_key = f"{rx_key}_enzyme_{len(reaction['enzymes']) + 1}"
            reaction["enzymes"].append(
                {"key": member_key, "entity_type": entity["entity_type"], "entity_key": entity["key"], "role": role or "catalyst"}
            )
            loc = ensure_location(entity["entity_type"], entity["key"], bs_key, x=rx_x, y=rx_y - 90)
            if loc:
                members_for_viz.append(
                    {
                        "process_member_key": member_key,
                        "role": "enzyme",
                        "member_type": entity["entity_type"],
                        "location_key": loc["key"],
                        "edge_key": None,
                    }
                )

        ir["processes"]["reactions"].append(reaction)
        process_by_key[rx_key] = reaction
        process_by_name[_norm(reaction["name"])] = reaction
        ir["process_visualizations"].append(
            {
                "key": keys.next("rv"),
                "type": "reaction_visualization",
                "process_key": rx_key,
                "biological_state_key": bs_key,
                "x": rx_x,
                "y": rx_y,
                "members": members_for_viz,
            }
        )

    transport_index = 0
    for tidx, raw in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(raw, dict):
            continue
        transport_index += 1
        tr_key = f"tr_{transport_index}"
        tr_x, tr_y = process_xy(reaction_index + transport_index - 1)
        transport = {
            "key": tr_key,
            "type": "transport",
            "name": _canonical(raw.get("name")) or f"Transport {transport_index}",
            "transport_elements": [],
            "transporters": [],
        }
        members_for_viz = []
        cargo_items = _safe_list(raw.get("transport_elements"))
        if not cargo_items:
            cargo_name = _canonical(raw.get("cargo") or raw.get("cargo_complex"))
            cargo_items = [{"element": cargo_name}] if cargo_name else []
        for cidx, cargo in enumerate(cargo_items):
            if isinstance(cargo, str):
                cargo = {"element": cargo}
            if not isinstance(cargo, dict):
                continue
            name = _canonical(cargo.get("element") or cargo.get("cargo") or cargo.get("name"))
            entity = resolve_entity(name, "reaction_member", f"/processes/transports/{tidx}/transport_elements/{cidx}")
            if not entity:
                continue
            left_bs = resolve_state(
                cargo.get("left_biological_state")
                or cargo.get("from_biological_state")
                or raw.get("from_biological_state"),
                f"/processes/transports/{tidx}/from_biological_state",
                required=True,
            )
            right_bs = resolve_state(
                cargo.get("right_biological_state")
                or cargo.get("to_biological_state")
                or raw.get("to_biological_state"),
                f"/processes/transports/{tidx}/to_biological_state",
                required=True,
            )
            member_key = f"{tr_key}_el_{len(transport['transport_elements']) + 1}"
            transport["transport_elements"].append(
                {
                    "key": member_key,
                    "entity_type": entity["entity_type"],
                    "entity_key": entity["key"],
                    "left_biological_state_key": left_bs,
                    "right_biological_state_key": right_bs,
                    "stoichiometry": _to_int(cargo.get("stoichiometry")) or 1,
                }
            )
            left_loc = ensure_location(entity["entity_type"], entity["key"], left_bs, x=tr_x - 145, y=tr_y)
            right_loc = ensure_location(entity["entity_type"], entity["key"], right_bs, x=tr_x + 145, y=tr_y)
            if left_loc:
                edge = add_edge(int(left_loc["x"]) + 60, int(left_loc["y"]) + 20, tr_x, tr_y)
                members_for_viz.append(
                    {
                        "process_member_key": member_key,
                        "role": "left",
                        "member_type": entity["entity_type"],
                        "location_key": left_loc["key"],
                        "edge_key": edge["key"],
                    }
                )
            if right_loc:
                edge = add_edge(tr_x, tr_y, int(right_loc["x"]) + 60, int(right_loc["y"]) + 20)
                members_for_viz.append(
                    {
                        "process_member_key": member_key,
                        "role": "right",
                        "member_type": entity["entity_type"],
                        "location_key": right_loc["key"],
                        "edge_key": edge["key"],
                    }
                )

        for aidx, actor in enumerate(_safe_list(raw.get("transporters"))):
            name, hint, _role = _actor_name_and_hint(actor)
            entity = resolve_entity(name, "transporter", f"/processes/transports/{tidx}/transporters/{aidx}", hint=hint)
            if not entity:
                continue
            member_key = f"{tr_key}_transporter_{len(transport['transporters']) + 1}"
            bs_key = resolve_state(
                actor.get("biological_state") if isinstance(actor, dict) else "",
                f"/processes/transports/{tidx}/transporters/{aidx}/biological_state",
                required=True,
            )
            transport["transporters"].append(
                {"key": member_key, "entity_type": entity["entity_type"], "entity_key": entity["key"], "biological_state_key": bs_key}
            )
            loc = ensure_location(entity["entity_type"], entity["key"], bs_key, x=tr_x, y=tr_y - 90)
            if loc:
                members_for_viz.append(
                    {
                        "process_member_key": member_key,
                        "role": "transporter",
                        "member_type": entity["entity_type"],
                        "location_key": loc["key"],
                        "edge_key": None,
                    }
                )

        ir["processes"]["transports"].append(transport)
        process_by_key[tr_key] = transport
        process_by_name[_norm(transport["name"])] = transport
        ir["process_visualizations"].append(
            {
                "key": keys.next("tv"),
                "type": "transport_visualization",
                "process_key": tr_key,
                "biological_state_key": default_bs_key,
                "x": tr_x,
                "y": tr_y,
                "members": members_for_viz,
            }
        )

    interaction_index = 0
    for iidx, raw in enumerate(_safe_list(processes.get("interactions"))):
        if not isinstance(raw, dict):
            continue
        left_name = _canonical(raw.get("left") or raw.get("entity_1") or raw.get("source"))
        right_name = _canonical(raw.get("right") or raw.get("entity_2") or raw.get("target"))
        left_entity = resolve_entity(left_name, "interaction", f"/processes/interactions/{iidx}/entity_1")
        right_entity = resolve_entity(right_name, "interaction", f"/processes/interactions/{iidx}/entity_2")
        interaction_index += 1
        int_key = f"int_{interaction_index}"
        bs_key = resolve_state(raw.get("biological_state"), f"/processes/interactions/{iidx}/biological_state", required=True)
        interaction = {
            "key": int_key,
            "type": "interaction",
            "name": _canonical(raw.get("name")) or f"Interaction {interaction_index}",
            "interaction_type": _canonical(raw.get("interaction_type") or raw.get("relationship") or raw.get("class") or "interaction"),
            "biological_state_key": bs_key,
            "left": (
                {"entity_type": left_entity["entity_type"], "entity_key": left_entity["key"]}
                if left_entity
                else None
            ),
            "right": (
                {"entity_type": right_entity["entity_type"], "entity_key": right_entity["key"]}
                if right_entity
                else None
            ),
        }
        ir["processes"]["interactions"].append(interaction)
        process_by_key[int_key] = interaction
        process_by_name[_norm(interaction["name"])] = interaction

    # Preserve RCT/sub-pathway records in typed buckets. Full visualization is left
    # explicit so the validator can reject incomplete records before serialization.
    for idx, raw in enumerate(_safe_list(processes.get("reaction_coupled_transports"))):
        if not isinstance(raw, dict):
            continue
        rct = {
            "key": f"rct_{idx + 1}",
            "type": "reaction_coupled_transport",
            "name": _canonical(raw.get("name")) or f"Reaction coupled transport {idx + 1}",
            "left": [],
            "right": [],
            "enzymes": [],
            "source": raw,
        }
        ir["processes"]["reaction_coupled_transports"].append(rct)
    for idx, raw in enumerate(_safe_list(processes.get("sub_pathways"))):
        if not isinstance(raw, dict):
            continue
        ir["processes"]["sub_pathways"].append(
            {
                "key": f"subp_{idx + 1}",
                "type": "sub_pathway",
                "name": _canonical(raw.get("name")) or f"Sub pathway {idx + 1}",
                "sub_pathway_type": _canonical(raw.get("sub_pathway_type") or raw.get("class")),
                "reference_pathway_id": raw.get("reference_pathway_id"),
                "alttext": _canonical(raw.get("alttext") or raw.get("description")),
            }
        )

    # Protein complex visualizations are explicit IR artifacts. Build one for
    # complexes that have member proteins already located in the same state.
    complex_locs = [loc for loc in ir["locations"] if loc.get("type") == "protein_complex_location"]
    for loc in complex_locs:
        ir["protein_complex_visualizations"].append(
            {
                "key": keys.next("pcv"),
                "entity_key": loc["entity_key"],
                "biological_state_key": loc["biological_state_key"],
                "location_key": loc["key"],
                "protein_member_location_keys": [],
                "compound_member_location_keys": [],
            }
        )

    validation = validate_pwml_ir(ir)
    report["ir_validation"] = validation
    report["counts"] = {
        "species": len(ir["species"]),
        "subcellular_locations": len(ir["subcellular_locations"]),
        "biological_states": len(ir["biological_states"]),
        "compounds": len(ir["entities"]["compounds"]),
        "proteins": len(ir["entities"]["proteins"]),
        "reactions": len(ir["processes"]["reactions"]),
        "transports": len(ir["processes"]["transports"]),
        "locations": len(ir["locations"]),
        "edges": len(ir["edges"]),
        "process_visualizations": len(ir["process_visualizations"]),
    }
    if validation.get("errors"):
        for err in validation["errors"]:
            _add_issue(
                report,
                "error",
                str(err.get("code", "ir_validation_error")),
                str(err.get("message", "PWML IR validation error.")),
                pointer=str(err.get("pointer", "")),
            )
    return ir, report


def validate_required_pwml_contract(payload_or_ir: Any) -> Dict[str, Any]:
    """
    Pre-export contract validator for the hydrated payload or built IR.

    Checks every required field defined by the PWML-ready contract without
    mutating the input or making any DB calls. Returns a structured report
    with ``errors``, ``warnings``, ``summary``, and ``ok``.
    """
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def err(code: str, message: str, pointer: str = "", **extra: Any) -> None:
        issue: Dict[str, Any] = {"code": code, "message": message}
        if pointer:
            issue["pointer"] = pointer
        issue.update(extra)
        errors.append(issue)

    def warn(code: str, message: str, pointer: str = "", **extra: Any) -> None:
        issue: Dict[str, Any] = {"code": code, "message": message}
        if pointer:
            issue["pointer"] = pointer
        issue.update(extra)
        warnings.append(issue)

    if not isinstance(payload_or_ir, dict):
        err("invalid_input", "Input must be a dict (payload or IR).")
        return {"ok": False, "errors": errors, "warnings": warnings, "summary": {}}

    is_ir = is_pwml_ir(payload_or_ir)

    # ── PATHWAY ──────────────────────────────────────────────────────────────
    if is_ir:
        pathway = _safe_dict(payload_or_ir.get("pathway"))
        pw_id = pathway.get("pw_id") or pathway.get("pathbank_pathway_id")
        name = pathway.get("name")
        subject = pathway.get("subject")
        width = pathway.get("width")
        height = pathway.get("height")
        base = "/pathway"
    else:
        meta = _safe_dict(payload_or_ir.get("metadata"))
        pw_id = meta.get("pw_id") or meta.get("pathbank_pathway_id") or meta.get("pathway_id")
        name = meta.get("pathway_name") or meta.get("name")
        subject = meta.get("pathway_subject") or meta.get("subject")
        width = meta.get("width")
        height = meta.get("height")
        base = "/metadata"

    if not pw_id:
        warn("pathway_missing_pw_id", "Pathway has no pw_id / pathbank_pathway_id.", f"{base}/pw_id")
    if not name:
        err("pathway_missing_name", "Pathway is missing a name.", f"{base}/name")
    if not subject:
        err("pathway_missing_subject", "Pathway is missing a subject.", f"{base}/subject")
    if is_ir:
        if not width:
            err("pathway_missing_width", "Pathway IR is missing width.", f"{base}/width")
        if not height:
            err("pathway_missing_height", "Pathway IR is missing height.", f"{base}/height")
    else:
        if not width:
            warn("pathway_missing_width", "Payload metadata has no explicit width (will default at IR build time).", f"{base}/width")
        if not height:
            warn("pathway_missing_height", "Payload metadata has no explicit height (will default at IR build time).", f"{base}/height")

    # ── ENTITY REGISTRIES ─────────────────────────────────────────────────────
    entities = _safe_dict(payload_or_ir.get("entities"))
    all_entity_names: set = set()
    all_entity_keys: set = set()
    protein_names: set = set()
    protein_keys: set = set()

    for bucket in ENTITY_BUCKETS.values():
        for e in _safe_list(entities.get(bucket)):
            if isinstance(e, dict):
                n = _canonical(e.get("name"))
                if n:
                    all_entity_names.add(_norm(n))
                    if bucket == "proteins":
                        protein_names.add(_norm(n))
                k = e.get("key")
                if k:
                    all_entity_keys.add(k)
                    if bucket == "proteins":
                        protein_keys.add(k)

    # ── PROTEINS ─────────────────────────────────────────────────────────────
    for idx, prot in enumerate(_safe_list(entities.get("proteins"))):
        if not isinstance(prot, dict):
            continue
        pname = _canonical(prot.get("name"))
        species = (
            prot.get("species")
            or prot.get("organism")
            or prot.get("taxonomy_id")
            or _safe_dict(prot.get("mapping_meta")).get("species")
        )
        if not species:
            err(
                "protein_missing_species",
                f"Protein '{pname}' is missing species/organism.",
                f"/entities/proteins/{idx}",
                entity_name=pname,
            )

    # ── PROTEIN COMPLEXES ────────────────────────────────────────────────────
    for idx, pc in enumerate(_safe_list(entities.get("protein_complexes"))):
        if not isinstance(pc, dict):
            continue
        pcname = _canonical(pc.get("name"))
        species = (
            pc.get("species")
            or pc.get("organism")
            or pc.get("taxonomy_id")
            or _safe_dict(pc.get("mapping_meta")).get("species")
        )
        if not species:
            err(
                "protein_complex_missing_species",
                f"Protein complex '{pcname}' is missing species.",
                f"/entities/protein_complexes/{idx}",
                entity_name=pcname,
            )
        components = _safe_list(pc.get("components"))
        if not components:
            err(
                "protein_complex_missing_components",
                f"Protein complex '{pcname}' has no protein components.",
                f"/entities/protein_complexes/{idx}/components",
                entity_name=pcname,
            )
        else:
            for cidx, comp in enumerate(components):
                pointer = f"/entities/protein_complexes/{idx}/components/{cidx}"
                protein_key = ""
                if isinstance(comp, dict):
                    protein_key = _canonical(comp.get("protein_key") or comp.get("entity_key"))
                comp_name = _component_protein_name(comp)
                stoich = _component_stoichiometry(comp)
                if stoich is None:
                    err(
                        "component_missing_stoichiometry",
                        f"Component[{cidx}] in complex '{pcname}' is missing stoichiometry.",
                        pointer,
                        complex_name=pcname,
                        component_name=comp_name,
                    )
                if protein_key:
                    if protein_key not in protein_keys:
                        err(
                            "component_protein_unresolved",
                            f"Component '{protein_key}' in complex '{pcname}' does not reference an existing protein.",
                            pointer,
                            complex_name=pcname,
                            protein_key=protein_key,
                        )
                elif comp_name:
                    if _norm(comp_name) not in protein_names:
                        err(
                            "component_protein_unresolved",
                            f"Component '{comp_name}' in complex '{pcname}' does not reference an existing protein.",
                            pointer,
                            complex_name=pcname,
                            component_name=comp_name,
                        )
                else:
                    err(
                        "component_protein_unresolved",
                        f"Component[{cidx}] in complex '{pcname}' does not reference an existing protein.",
                        pointer,
                        complex_name=pcname,
                    )

    # ── BIOLOGICAL STATES ────────────────────────────────────────────────────
    bio_states = _safe_list(payload_or_ir.get("biological_states"))
    if not bio_states:
        warn("no_biological_states", "No biological_states defined; a default will be generated.")
    for idx, state in enumerate(bio_states):
        if not isinstance(state, dict):
            continue
        sname = state.get("name", f"state[{idx}]")
        if is_ir:
            has_species = bool(state.get("species_key"))
            has_subcell = bool(state.get("subcellular_location_key"))
        else:
            has_species = bool(
                state.get("species") or state.get("organism") or state.get("taxonomy_id")
            )
            has_subcell = bool(
                state.get("subcellular_location")
                or state.get("subcellular-location")
                or state.get("compartment")
                or state.get("compartment_canonical")
            )
        if not has_species:
            err(
                "biological_state_missing_species",
                f"Biological state '{sname}' is missing species.",
                f"/biological_states/{idx}",
                state_name=sname,
            )
        if not has_subcell:
            err(
                "biological_state_missing_subcellular_location",
                f"Biological state '{sname}' has no subcellular location.",
                f"/biological_states/{idx}",
                state_name=sname,
            )

    # ── REACTIONS ─────────────────────────────────────────────────────────────
    processes = _safe_dict(payload_or_ir.get("processes"))
    for idx, rx in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(rx, dict):
            continue
        rname = rx.get("name", f"reaction[{idx}]")

        if is_ir:
            left = _safe_list(rx.get("left"))
            right = _safe_list(rx.get("right"))
            left_key, right_key = "left", "right"
        else:
            left = _safe_list(rx.get("inputs") or rx.get("left") or rx.get("substrates"))
            right = _safe_list(rx.get("outputs") or rx.get("right") or rx.get("products"))
            left_key = "inputs" if rx.get("inputs") else "left"
            right_key = "outputs" if rx.get("outputs") else "right"

        if not left:
            err(
                "reaction_missing_left_participants",
                f"Reaction '{rname}' has no left/input participants.",
                f"/processes/reactions/{idx}/{left_key}",
                reaction_name=rname,
            )
        if not right:
            err(
                "reaction_missing_right_participants",
                f"Reaction '{rname}' has no right/output participants.",
                f"/processes/reactions/{idx}/{right_key}",
                reaction_name=rname,
            )

        for side_label, side in [(left_key, left), (right_key, right)]:
            for midx, part in enumerate(side):
                if isinstance(part, str):
                    warn(
                        "reaction_participant_missing_stoichiometry",
                        f"Reaction '{rname}' {side_label}[{midx}] is a bare string with no explicit stoichiometry (defaults to 1).",
                        f"/processes/reactions/{idx}/{side_label}/{midx}",
                        reaction_name=rname,
                    )
                    continue
                if not isinstance(part, dict):
                    continue
                stoich = part.get("stoichiometry") or part.get("coefficient")
                if stoich is None:
                    warn(
                        "reaction_participant_missing_stoichiometry",
                        f"Reaction '{rname}' {side_label}[{midx}] has no explicit stoichiometry (defaults to 1).",
                        f"/processes/reactions/{idx}/{side_label}/{midx}",
                        reaction_name=rname,
                    )

        # Enzyme references against entity registry
        enzyme_raw = _safe_list(rx.get("enzymes") or rx.get("catalysts"))
        for eidx, actor in enumerate(enzyme_raw):
            if is_ir:
                # IR enzyme members use entity_key; cross-check against built keys
                if isinstance(actor, dict):
                    ekey = actor.get("entity_key")
                    if ekey and ekey not in all_entity_keys:
                        err(
                            "enzyme_reference_not_found",
                            f"Reaction '{rname}' enzyme entity_key '{ekey}' not found in entities.",
                            f"/processes/reactions/{idx}/enzymes/{eidx}",
                            reaction_name=rname,
                            entity_key=ekey,
                        )
            else:
                if isinstance(actor, str):
                    ename = _canonical(actor)
                elif isinstance(actor, dict):
                    ename = _canonical(
                        actor.get("name")
                        or actor.get("protein")
                        or actor.get("protein_complex")
                        or actor.get("entity")
                    )
                else:
                    continue
                if ename and _norm(ename) not in all_entity_names:
                    err(
                        "enzyme_reference_not_found",
                        f"Reaction '{rname}' enzyme '{ename}' not found in entity registries.",
                        f"/processes/reactions/{idx}/enzymes/{eidx}",
                        reaction_name=rname,
                        enzyme_name=ename,
                    )

    # ── LOCATION / VISUALIZATION REFERENCES (IR only) ────────────────────────
    if is_ir:
        bio_state_keys = {
            s.get("key")
            for s in _safe_list(payload_or_ir.get("biological_states"))
            if isinstance(s, dict)
        }
        process_keys: set = set()
        for bucket in PROCESS_BUCKETS.values():
            for p in _safe_list(_safe_dict(payload_or_ir.get("processes")).get(bucket)):
                if isinstance(p, dict) and p.get("key"):
                    process_keys.add(p["key"])
        location_keys = {
            loc.get("key")
            for loc in _safe_list(payload_or_ir.get("locations"))
            if isinstance(loc, dict)
        }

        for idx, loc in enumerate(_safe_list(payload_or_ir.get("locations"))):
            if not isinstance(loc, dict):
                continue
            if loc.get("entity_key") not in all_entity_keys:
                err(
                    "location_missing_entity",
                    f"Location '{loc.get('key', idx)}' references unknown entity_key '{loc.get('entity_key')}'.",
                    f"/locations/{idx}",
                )
            if loc.get("biological_state_key") not in bio_state_keys:
                err(
                    "location_missing_biological_state",
                    f"Location '{loc.get('key', idx)}' references unknown biological_state_key '{loc.get('biological_state_key')}'.",
                    f"/locations/{idx}",
                )

        for idx, viz in enumerate(_safe_list(payload_or_ir.get("process_visualizations"))):
            if not isinstance(viz, dict):
                continue
            if viz.get("process_key") not in process_keys:
                err(
                    "visualization_missing_process",
                    f"Process visualization '{viz.get('key', idx)}' references unknown process_key '{viz.get('process_key')}'.",
                    f"/process_visualizations/{idx}",
                )
            if viz.get("biological_state_key") and viz.get("biological_state_key") not in bio_state_keys:
                err(
                    "visualization_missing_biological_state",
                    f"Process visualization '{viz.get('key', idx)}' references unknown biological_state_key.",
                    f"/process_visualizations/{idx}",
                )
            for midx, member in enumerate(_safe_list(viz.get("members"))):
                if not isinstance(member, dict):
                    continue
                if member.get("location_key") and member.get("location_key") not in location_keys:
                    err(
                        "visualization_member_missing_location",
                        f"Visualization member references unknown location_key '{member.get('location_key')}'.",
                        f"/process_visualizations/{idx}/members/{midx}",
                    )

    error_codes = sorted({e["code"] for e in errors})
    warning_codes = sorted({w["code"] for w in warnings})
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "error_codes": error_codes,
            "warning_codes": warning_codes,
            "checked_as": "ir" if is_ir else "payload",
        },
    }


def validate_pwml_ir(ir: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def error(code: str, message: str, pointer: str = "", **extra: Any) -> None:
        issue = {"code": code, "message": message}
        if pointer:
            issue["pointer"] = pointer
        issue.update(extra)
        errors.append(issue)

    def warning(code: str, message: str, pointer: str = "", **extra: Any) -> None:
        issue = {"code": code, "message": message}
        if pointer:
            issue["pointer"] = pointer
        issue.update(extra)
        warnings.append(issue)

    if not is_pwml_ir(ir):
        error("invalid_ir_shape", "Input does not look like a PWML IR object.", "/")
        return {"ok": False, "errors": errors, "warnings": warnings, "counts": {}}

    entity_keys_by_type: Dict[str, set] = {entity_type: set() for entity_type in ENTITY_BUCKETS}
    all_entity_keys: set = set()
    for entity_type, bucket in ENTITY_BUCKETS.items():
        for idx, record in enumerate(_safe_list(_safe_dict(ir.get("entities")).get(bucket))):
            if not isinstance(record, dict):
                continue
            key = record.get("key")
            if not key:
                error("missing_entity_key", f"{bucket}[{idx}] is missing key.", f"/entities/{bucket}/{idx}")
                continue
            if key in all_entity_keys:
                error("duplicate_entity_key", f"Duplicate entity key '{key}'.", f"/entities/{bucket}/{idx}")
            entity_keys_by_type[entity_type].add(key)
            all_entity_keys.add(key)

    for idx, record in enumerate(_safe_list(_safe_dict(ir.get("entities")).get("protein_complexes"))):
        if not isinstance(record, dict):
            continue
        components = _safe_list(record.get("components"))
        if not components:
            error(
                "protein_complex_missing_components",
                f"Protein complex '{record.get('name') or idx}' has no protein components.",
                f"/entities/protein_complexes/{idx}/components",
            )
            continue
        for cidx, component in enumerate(components):
            pointer = f"/entities/protein_complexes/{idx}/components/{cidx}"
            if not isinstance(component, dict):
                error(
                    "protein_complex_component_not_structured",
                    "Protein complex component must be a structured record.",
                    pointer,
                )
                continue
            protein_key = component.get("protein_key")
            stoich = _to_int(component.get("stoichiometry"))
            if protein_key not in entity_keys_by_type["protein"]:
                error(
                    "component_protein_unresolved",
                    f"Protein complex component references unknown protein_key '{protein_key}'.",
                    pointer,
                    protein_key=protein_key,
                )
            if stoich is None or stoich < 1:
                error(
                    "component_missing_stoichiometry",
                    "Protein complex component is missing positive stoichiometry.",
                    pointer,
                    protein_key=protein_key,
                )

    biological_state_keys = {
        state.get("key")
        for state in _safe_list(ir.get("biological_states"))
        if isinstance(state, dict) and state.get("key")
    }
    for idx, state in enumerate(_safe_list(ir.get("biological_states"))):
        if not isinstance(state, dict):
            continue
        pointer = f"/biological_states/{idx}"
        if not state.get("key"):
            error("missing_biological_state_key", f"biological_states[{idx}] is missing key.", pointer)
        if not state.get("species_key"):
            error(
                "biological_state_missing_species",
                f"Biological state '{state.get('name') or idx}' is missing species_key.",
                pointer,
            )
        if not state.get("subcellular_location_key"):
            error(
                "biological_state_missing_subcellular_location",
                f"Biological state '{state.get('name') or idx}' is missing subcellular_location_key.",
                pointer,
            )
    location_keys = {
        loc.get("key")
        for loc in _safe_list(ir.get("locations"))
        if isinstance(loc, dict) and loc.get("key")
    }
    edge_keys = {
        edge.get("key")
        for edge in _safe_list(ir.get("edges"))
        if isinstance(edge, dict) and edge.get("key")
    }

    for idx, loc in enumerate(_safe_list(ir.get("locations"))):
        if not isinstance(loc, dict):
            continue
        ekey = loc.get("entity_key")
        bs_key = loc.get("biological_state_key")
        if ekey not in all_entity_keys:
            error("unresolved_location_entity", f"Location references unknown entity_key '{ekey}'.", f"/locations/{idx}")
        if bs_key not in biological_state_keys:
            error(
                "unresolved_location_biological_state",
                f"Location references unknown biological_state_key '{bs_key}'.",
                f"/locations/{idx}",
            )

    process_keys: set = set()
    process_member_keys: set = set()
    process_bucket_by_key: Dict[str, str] = {}
    process_members_by_key: Dict[str, Dict[str, Any]] = {}
    processes = _safe_dict(ir.get("processes"))

    def validate_member(member: Dict[str, Any], pointer: str) -> None:
        mkey = member.get("key")
        entity_type = member.get("entity_type")
        entity_key = member.get("entity_key")
        if not mkey:
            error("missing_process_member_key", "Process member is missing key.", pointer)
        elif mkey in process_member_keys:
            error("duplicate_process_member_key", f"Duplicate process member key '{mkey}'.", pointer)
        else:
            process_member_keys.add(mkey)
            process_members_by_key[mkey] = member
        if entity_type not in ENTITY_BUCKETS:
            error("invalid_member_entity_type", f"Invalid member entity_type '{entity_type}'.", pointer)
        elif entity_key not in entity_keys_by_type[entity_type]:
            error(
                "unresolved_member_entity",
                f"Process member references unknown {entity_type} entity_key '{entity_key}'.",
                pointer,
            )

    for idx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        key = reaction.get("key")
        if key:
            process_keys.add(key)
            process_bucket_by_key[key] = "reactions"
        if reaction.get("biological_state_key") not in biological_state_keys:
            error("unresolved_reaction_biological_state", "Reaction has unresolved biological_state_key.", f"/processes/reactions/{idx}")
        left = _safe_list(reaction.get("left"))
        right = _safe_list(reaction.get("right"))
        if not left:
            error("reaction_missing_left", "Reaction must have at least one left member.", f"/processes/reactions/{idx}/left")
        if not right:
            error("reaction_missing_right", "Reaction must have at least one right member.", f"/processes/reactions/{idx}/right")
        for midx, member in enumerate(left):
            if isinstance(member, dict):
                validate_member(member, f"/processes/reactions/{idx}/left/{midx}")
        for midx, member in enumerate(right):
            if isinstance(member, dict):
                validate_member(member, f"/processes/reactions/{idx}/right/{midx}")
        for midx, member in enumerate(_safe_list(reaction.get("enzymes"))):
            if isinstance(member, dict):
                validate_member(member, f"/processes/reactions/{idx}/enzymes/{midx}")
                if member.get("entity_type") != "protein_complex":
                    error(
                        "reaction_enzyme_must_be_protein_complex",
                        "Reaction enzyme must reference a protein_complex entity.",
                        f"/processes/reactions/{idx}/enzymes/{midx}",
                        entity_type=member.get("entity_type"),
                    )

    for idx, transport in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(transport, dict):
            continue
        key = transport.get("key")
        if key:
            process_keys.add(key)
            process_bucket_by_key[key] = "transports"
        elements = _safe_list(transport.get("transport_elements"))
        if not 1 <= len(elements) <= 3:
            error(
                "transport_element_count",
                "Transport must have 1 to 3 transport elements.",
                f"/processes/transports/{idx}/transport_elements",
                count=len(elements),
            )
        for midx, member in enumerate(elements):
            if not isinstance(member, dict):
                continue
            validate_member(member, f"/processes/transports/{idx}/transport_elements/{midx}")
            if member.get("left_biological_state_key") not in biological_state_keys:
                error("transport_missing_left_state", "Transport element has unresolved left biological state.", f"/processes/transports/{idx}/transport_elements/{midx}")
            if member.get("right_biological_state_key") not in biological_state_keys:
                error("transport_missing_right_state", "Transport element has unresolved right biological state.", f"/processes/transports/{idx}/transport_elements/{midx}")
        for midx, member in enumerate(_safe_list(transport.get("transporters"))):
            if isinstance(member, dict):
                validate_member(member, f"/processes/transports/{idx}/transporters/{midx}")

    for idx, rct in enumerate(_safe_list(processes.get("reaction_coupled_transports"))):
        if not isinstance(rct, dict):
            continue
        key = rct.get("key")
        if key:
            process_keys.add(key)
            process_bucket_by_key[key] = "reaction_coupled_transports"
        if not _safe_list(rct.get("left")):
            error("rct_missing_left", "Reaction-coupled transport must have a left member.", f"/processes/reaction_coupled_transports/{idx}/left")
        if not _safe_list(rct.get("right")):
            error("rct_missing_right", "Reaction-coupled transport must have a right member.", f"/processes/reaction_coupled_transports/{idx}/right")
        if not _safe_list(rct.get("enzymes")):
            error("rct_missing_enzyme", "Reaction-coupled transport must have at least one enzyme.", f"/processes/reaction_coupled_transports/{idx}/enzymes")

    for idx, interaction in enumerate(_safe_list(processes.get("interactions"))):
        if not isinstance(interaction, dict):
            continue
        key = interaction.get("key")
        if key:
            process_keys.add(key)
            process_bucket_by_key[key] = "interactions"
        if not interaction.get("interaction_type"):
            error("interaction_missing_type", "Interaction must have interaction_type.", f"/processes/interactions/{idx}")
        left = interaction.get("left")
        right = interaction.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            error("interaction_missing_side", "Interaction must have exactly one left and one right member.", f"/processes/interactions/{idx}")
        else:
            for side_name, side in [("left", left), ("right", right)]:
                entity_type = side.get("entity_type")
                entity_key = side.get("entity_key")
                if entity_type not in ENTITY_BUCKETS or entity_key not in entity_keys_by_type.get(entity_type, set()):
                    error(
                        "unresolved_interaction_entity",
                        f"Interaction {side_name} references unresolved entity.",
                        f"/processes/interactions/{idx}/{side_name}",
                    )

    for idx, subp in enumerate(_safe_list(processes.get("sub_pathways"))):
        if not isinstance(subp, dict):
            continue
        key = subp.get("key")
        if key:
            process_keys.add(key)
            process_bucket_by_key[key] = "sub_pathways"
        if not subp.get("sub_pathway_type"):
            error("sub_pathway_missing_type", "Sub-pathway has no sub_pathway_type.", f"/processes/sub_pathways/{idx}")
        if not subp.get("reference_pathway_id") and not subp.get("alttext"):
            error(
                "sub_pathway_missing_reference",
                "Sub-pathway needs either reference_pathway_id or alttext.",
                f"/processes/sub_pathways/{idx}",
            )

    expected_bucket_by_viz_type = {
        "reaction_visualization": "reactions",
        "reaction_coupled_transport_visualization": "reaction_coupled_transports",
        "transport_visualization": "transports",
        "interaction_visualization": "interactions",
        "sub_pathway_visualization": "sub_pathways",
    }
    for idx, viz in enumerate(_safe_list(ir.get("process_visualizations"))):
        if not isinstance(viz, dict):
            continue
        process_key = viz.get("process_key")
        viz_type = viz.get("type")
        if process_key not in process_keys:
            error("visualization_unknown_process", f"Visualization references unknown process_key '{process_key}'.", f"/process_visualizations/{idx}")
        elif expected_bucket_by_viz_type.get(viz_type) != process_bucket_by_key.get(process_key):
            warning(
                "visualization_type_process_mismatch",
                "Visualization type does not match process bucket.",
                f"/process_visualizations/{idx}",
                visualization_type=viz_type,
                process_bucket=process_bucket_by_key.get(process_key),
            )
        bs_key = viz.get("biological_state_key")
        if bs_key and bs_key not in biological_state_keys:
            error("visualization_unknown_biological_state", f"Visualization references unknown biological_state_key '{bs_key}'.", f"/process_visualizations/{idx}")
        for midx, member in enumerate(_safe_list(viz.get("members"))):
            if not isinstance(member, dict):
                continue
            pointer = f"/process_visualizations/{idx}/members/{midx}"
            member_key = member.get("process_member_key")
            if member_key not in process_member_keys:
                error("visualization_unknown_member", f"Visualization member references unknown process_member_key '{member_key}'.", pointer)
            loc_key = member.get("location_key")
            if loc_key not in location_keys:
                error("visualization_unknown_location", f"Visualization member references unknown location_key '{loc_key}'.", pointer)
            edge_key = member.get("edge_key")
            if edge_key and edge_key not in edge_keys:
                error("visualization_unknown_edge", f"Visualization member references unknown edge_key '{edge_key}'.", pointer)
            if member.get("role") in {"left", "right"} and not edge_key:
                error("visualization_member_missing_edge", "Left/right visualization members require an edge_key.", pointer)

    counts = {
        "entities": len(all_entity_keys),
        "biological_states": len(biological_state_keys),
        "processes": len(process_keys),
        "process_members": len(process_member_keys),
        "locations": len(location_keys),
        "edges": len(edge_keys),
        "process_visualizations": len(_safe_list(ir.get("process_visualizations"))),
    }
    return {"ok": not errors, "errors": errors, "warnings": warnings, "counts": counts}
