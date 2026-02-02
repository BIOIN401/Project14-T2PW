"""
Convert PWML-structured JSON (your extractor output) into a .pwml XML file.

Usage:
  python -m src.to_pwml --in extract.json --out out.pwml --name "My Pathway"
OR
  python src/to_pwml.py --in extract.json --out out.pwml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


def kebab(s: str) -> str:
    """snake_case -> kebab-case (PWML tags use kebab-case)."""
    return s.replace("_", "-").strip()


def singularize(plural: str) -> str:
    """
    Best-effort plural -> singular tag name for list items.
    We handle common PWML cases explicitly; fallback removes trailing 's'.
    """
    specials = {
        "species": "species",
        "cell_types": "cell-type",
        "tissues": "tissue",
        "subcellular_locations": "subcellular-location",
        "compounds": "compound",
        "proteins": "protein",
        "protein_complexes": "protein-complex",
        "element_collections": "element-collection",
        "nucleic_acids": "nucleic-acid",

        "biological_states": "biological-state",

        "compound_locations": "compound-location",
        "protein_locations": "protein-location",
        "element_collection_locations": "element-collection-location",
        "nucleic_acid_locations": "nucleic-acid-location",

        "reactions": "reaction",
        "transports": "transport",
        "interactions": "interaction",
        "reaction_coupled_transports": "reaction-coupled-transport",

        "bounds": "bound",
        "bound_visualizations": "bound-visualization",
        "protein_complex_visualizations": "protein-complex-visualization",
        "reaction_visualizations": "reaction-visualization",
        "reaction_coupled_transport_visualizations": "reaction-coupled-transport-visualization",
        "transport_visualizations": "transport-visualization",
        "interaction_visualizations": "interaction-visualization",
        "sub_pathway_visualizations": "sub-pathway-visualization",
        "edges": "edge",

        "vacuous_compound_visualizations": "vacuous-compound-visualization",
        "vacuous_protein_visualizations": "vacuous-protein-visualization",
        "vacuous_nucleic_acid_visualizations": "vacuous-nucleic-acid-visualization",
        "vacuous_element_collection_visualizations": "vacuous-element-collection-visualization",
    }

    if plural in specials:
        return specials[plural]

    # fallback: remove trailing s
    if plural.endswith("s") and len(plural) > 1:
        return plural[:-1]
    return plural


def needs_integer_type_attr(tag: str) -> bool:
    """
    In PWML exports, many IDs are represented like:
      <id type="integer">123</id>
      <compound-id type="integer">456</compound-id>
    We'll tag integers on likely ID fields.
    """
    t = tag.lower()
    return t == "id" or t.endswith("-id") or t in {"named-for-id"}


class IdFactory:
    """Simple incremental ID generator."""
    def __init__(self, start: int = 1) -> None:
        self._n = start

    def next(self) -> int:
        v = self._n
        self._n += 1
        return v


# ----------------------------
# Core conversion
# ----------------------------

def _add_text(node: ET.Element, value: Any) -> None:
    if value is None:
        node.text = ""
    elif isinstance(value, bool):
        node.text = "true" if value else "false"
    else:
        node.text = str(value)


def _dict_to_xml(parent: ET.Element, data: Dict[str, Any], ids: IdFactory) -> None:
    """
    Convert a dict into nested XML elements appended under parent.
    """
    for k, v in data.items():
        tag = kebab(k)

        if isinstance(v, dict):
            child = ET.SubElement(parent, tag)
            _dict_to_xml(child, v, ids)

        elif isinstance(v, list):
            container = ET.SubElement(parent, tag)
            item_tag = singularize(kebab(k).replace("-", "_")).replace("_", "-")  # safe-ish
            # Use singularize on original key (before kebab) for best mapping
            item_tag = singularize(k).replace("_", "-")

            for item in v:
                item_el = ET.SubElement(container, item_tag)

                # Auto-add an <id> for dict items if missing
                if isinstance(item, dict):
                    if "id" not in item:
                        id_el = ET.SubElement(item_el, "id")
                        id_el.set("type", "integer")
                        id_el.text = str(ids.next())
                    _dict_to_xml(item_el, item, ids)

                else:
                    # primitives: <item_tag>value</item_tag>
                    _add_text(item_el, item)

        else:
            child = ET.SubElement(parent, tag)
            # Add type="integer" if it looks like an ID field and the value is numeric
            if needs_integer_type_attr(tag) and isinstance(v, int):
                child.set("type", "integer")
            _add_text(child, v)


def pwml_from_extraction(
    extraction: Dict[str, Any],
    pathway_name: str = "Generated Pathway",
    pathway_description: str = "",
    named_for_id: int = 0,
) -> str:
    """
    Build a PWML XML string from extracted JSON dict.
    """
    ids = IdFactory(start=1)

    root = ET.Element("super-pathway-visualization")

    # Minimal header fields PathBank-like exports usually include
    nfid = ET.SubElement(root, "named-for-id")
    nfid.set("type", "integer")
    nfid.text = str(named_for_id)

    nft = ET.SubElement(root, "named-for-type")
    nft.text = "Pathway"

    cn = ET.SubElement(root, "cached-name")
    cn.text = pathway_name

    cd = ET.SubElement(root, "cached-description")
    cd.text = pathway_description

    # Flatten entities into top-level PWML sections
    entities = extraction.get("entities", {})
    if isinstance(entities, dict):
        for section_key, section_val in entities.items():
            # Each section becomes <compounds> ... etc.
            sec_tag = section_key  # e.g., compounds
            sec_el = ET.SubElement(root, kebab(sec_tag))
            if isinstance(section_val, list):
                item_tag = singularize(section_key).replace("_", "-")
                for item in section_val:
                    item_el = ET.SubElement(sec_el, item_tag)
                    if isinstance(item, dict):
                        if "id" not in item:
                            id_el = ET.SubElement(item_el, "id")
                            id_el.set("type", "integer")
                            id_el.text = str(ids.next())
                        _dict_to_xml(item_el, item, ids)
                    else:
                        _add_text(item_el, item)

    # Remaining top-level sections
    for top_key in ["biological_states", "element_locations", "processes", "visualizations", "vacuous"]:
        if top_key in extraction:
            block = extraction[top_key]
            block_el = ET.SubElement(root, kebab(top_key))
            if isinstance(block, dict):
                _dict_to_xml(block_el, block, ids)
            elif isinstance(block, list):
                item_tag = singularize(top_key).replace("_", "-")
                for item in block:
                    item_el = ET.SubElement(block_el, item_tag)
                    if isinstance(item, dict):
                        if "id" not in item:
                            id_el = ET.SubElement(item_el, "id")
                            id_el.set("type", "integer")
                            id_el.text = str(ids.next())
                        _dict_to_xml(item_el, item, ids)
                    else:
                        _add_text(item_el, item)
            else:
                _add_text(block_el, block)

    # Pretty-ish formatting
    _indent(root)

    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
    return xml


def pwml_from_extraction_structured(
    extraction: Dict[str, Any],
    pathway_name: str = "Generated Pathway",
    pathway_description: str = "",
    pathway_subject: str = "Metabolic",
    named_for_id: int = 0,
) -> str:
    """
    Build a more structured PWML XML string from extracted JSON dict.
    Focuses on entities, biological states, processes, and element states.
    Visualization/layout sections are omitted unless present in JSON.
    """
    ids = IdFactory(start=1)
    rel_ids = IdFactory(start=1000)

    root = ET.Element("super-pathway-visualization")

    nfid = ET.SubElement(root, "named-for-id")
    nfid.set("type", "integer")
    nfid.text = str(named_for_id)

    nft = ET.SubElement(root, "named-for-type")
    nft.text = "Pathway"

    cn = ET.SubElement(root, "cached-name")
    cn.text = pathway_name

    cd = ET.SubElement(root, "cached-description")
    cd.text = pathway_description

    cs = ET.SubElement(root, "cached-subject")
    cs.text = pathway_subject

    entities = extraction.get("entities", {}) if isinstance(extraction, dict) else {}
    entity_maps = _build_entity_maps(entities, ids)

    _emit_simple_entities(root, "cell-types", "cell-type", entity_maps["cell_types"])
    _emit_simple_entities(root, "species", "species", entity_maps["species"], extra_fields=("taxonomy_id",))
    _emit_simple_entities(root, "subcellular-locations", "subcellular-location", entity_maps["subcellular_locations"], extra_fields=("ontology_id",))
    _emit_simple_entities(root, "tissues", "tissue", entity_maps["tissues"])

    biological_states = extraction.get("biological_states", [])
    bs_map = _emit_biological_states(
        root,
        biological_states,
        entity_maps,
        ids,
    )

    element_states = _collect_element_states(extraction.get("element_locations", {}), bs_map)

    _emit_compounds(root, entity_maps["compounds"], element_states.get("compounds", []), ids)
    _emit_element_collections(root, entity_maps["element_collections"], element_states.get("element_collections", []), ids)
    _emit_nucleic_acids(root, entity_maps["nucleic_acids"], element_states.get("nucleic_acids", []), ids)
    _emit_proteins(root, entity_maps["proteins"], element_states.get("proteins", []), ids)
    _emit_protein_complexes(root, entity_maps["protein_complexes"], ids)

    processes = extraction.get("processes", {})
    _emit_reactions(root, processes.get("reactions", []), entity_maps, rel_ids)
    _emit_reaction_coupled_transports(root, processes.get("reaction_coupled_transports", []), entity_maps, rel_ids)
    _emit_transports(root, processes.get("transports", []), entity_maps, rel_ids)
    _emit_interactions(root, processes.get("interactions", []), rel_ids)

    _indent(root)
    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
    return xml


def _build_entity_maps(entities: Dict[str, Any], ids: IdFactory) -> Dict[str, Dict[str, Dict[str, Any]]]:
    def normalize_list(key: str) -> Dict[str, Dict[str, Any]]:
        items: Dict[str, Dict[str, Any]] = {}
        for item in entities.get(key, []) if isinstance(entities, dict) else []:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or "").strip()
            if not name:
                continue
            items[name] = {**item, "id": ids.next(), "name": name}
        return items

    return {
        "cell_types": normalize_list("cell_types"),
        "species": normalize_list("species"),
        "tissues": normalize_list("tissues"),
        "subcellular_locations": normalize_list("subcellular_locations"),
        "compounds": normalize_list("compounds"),
        "element_collections": normalize_list("element_collections"),
        "nucleic_acids": normalize_list("nucleic_acids"),
        "proteins": normalize_list("proteins"),
        "protein_complexes": normalize_list("protein_complexes"),
    }


def _emit_simple_entities(
    root: ET.Element,
    container_tag: str,
    item_tag: str,
    items: Dict[str, Dict[str, Any]],
    extra_fields: Tuple[str, ...] = (),
) -> None:
    if not items:
        return
    container = ET.SubElement(root, container_tag)
    for item in items.values():
        el = ET.SubElement(container, item_tag)
        id_el = ET.SubElement(el, "id")
        id_el.set("type", "integer")
        id_el.text = str(item["id"])
        name_el = ET.SubElement(el, "name")
        name_el.text = item["name"]
        for field in extra_fields:
            value = (item.get(field) or "").strip() if isinstance(item.get(field), str) else ""
            if value:
                sub = ET.SubElement(el, field.replace("_", "-"))
                sub.text = value


def _emit_biological_states(
    root: ET.Element,
    states: Any,
    entity_maps: Dict[str, Dict[str, Dict[str, Any]]],
    ids: IdFactory,
) -> Dict[str, int]:
    bs_map: Dict[str, int] = {}
    if not isinstance(states, list) or not states:
        return bs_map

    container = ET.SubElement(root, "biological-states")
    for item in states:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        bs_id = ids.next()
        if name:
            bs_map[name] = bs_id

        el = ET.SubElement(container, "biological-state")
        id_el = ET.SubElement(el, "id")
        id_el.set("type", "integer")
        id_el.text = str(bs_id)

        if name:
            name_el = ET.SubElement(el, "name")
            name_el.text = name

        _maybe_ref_entity(el, "species-id", item.get("species"), entity_maps["species"])
        _maybe_ref_entity(el, "cell-type-id", item.get("cell_type"), entity_maps["cell_types"])
        _maybe_ref_entity(el, "tissue-id", item.get("tissue"), entity_maps["tissues"])
        _maybe_ref_entity(el, "subcellular-location-id", item.get("subcellular_location"), entity_maps["subcellular_locations"])

    return bs_map


def _maybe_ref_entity(parent: ET.Element, tag: str, name: Any, entity_map: Dict[str, Dict[str, Any]]) -> None:
    if not isinstance(name, str):
        return
    key = name.strip()
    if not key or key not in entity_map:
        return
    el = ET.SubElement(parent, tag)
    el.set("type", "integer")
    el.text = str(entity_map[key]["id"])


def _collect_element_states(element_locations: Any, bs_map: Dict[str, int]) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {
        "compounds": [],
        "element_collections": [],
        "nucleic_acids": [],
        "proteins": [],
    }
    if not isinstance(element_locations, dict):
        return out

    for key, target in [
        ("compound_locations", "compounds"),
        ("element_collection_locations", "element_collections"),
        ("nucleic_acid_locations", "nucleic_acids"),
        ("protein_locations", "proteins"),
    ]:
        for item in element_locations.get(key, []) if isinstance(element_locations.get(key), list) else []:
            if not isinstance(item, dict):
                continue
            bs_name = (item.get("biological_state") or "").strip()
            if not bs_name or bs_name not in bs_map:
                continue
            entity_name = ""
            for field in ["compound", "element_collection", "nucleic_acid", "protein"]:
                if item.get(field):
                    entity_name = str(item.get(field)).strip()
                    break
            if not entity_name:
                continue
            out[target].append((entity_name, bs_map[bs_name]))

    return out


def _emit_element_states(parent: ET.Element, states: List[Tuple[str, int]], ids: IdFactory, entity_map: Dict[str, Dict[str, Any]]) -> None:
    if not states:
        return
    es_container = ET.SubElement(parent, "element-states")
    for name, bs_id in states:
        if name not in entity_map:
            continue
        es = ET.SubElement(es_container, "element-state")
        es_id = ET.SubElement(es, "id")
        es_id.set("type", "integer")
        es_id.text = str(ids.next())
        bs = ET.SubElement(es, "biological-state-id")
        bs.set("type", "integer")
        bs.text = str(bs_id)


def _emit_compounds(
    root: ET.Element,
    compounds: Dict[str, Dict[str, Any]],
    element_states: List[Tuple[str, int]],
    ids: IdFactory,
) -> None:
    if not compounds:
        return
    container = ET.SubElement(root, "compounds")
    for item in compounds.values():
        el = ET.SubElement(container, "compound")
        _emit_id_and_name(el, item)
        _emit_element_states(el, [s for s in element_states if s[0] == item["name"]], ids, compounds)


def _emit_element_collections(
    root: ET.Element,
    element_collections: Dict[str, Dict[str, Any]],
    element_states: List[Tuple[str, int]],
    ids: IdFactory,
) -> None:
    if not element_collections:
        return
    container = ET.SubElement(root, "element-collections")
    for item in element_collections.values():
        el = ET.SubElement(container, "element-collection")
        _emit_id_and_name(el, item)
        _emit_element_states(el, [s for s in element_states if s[0] == item["name"]], ids, element_collections)


def _emit_nucleic_acids(
    root: ET.Element,
    nucleic_acids: Dict[str, Dict[str, Any]],
    element_states: List[Tuple[str, int]],
    ids: IdFactory,
) -> None:
    if not nucleic_acids:
        return
    container = ET.SubElement(root, "nucleic-acids")
    for item in nucleic_acids.values():
        el = ET.SubElement(container, "nucleic-acid")
        _emit_id_and_name(el, item)
        _emit_element_states(el, [s for s in element_states if s[0] == item["name"]], ids, nucleic_acids)


def _emit_proteins(
    root: ET.Element,
    proteins: Dict[str, Dict[str, Any]],
    element_states: List[Tuple[str, int]],
    ids: IdFactory,
) -> None:
    if not proteins:
        return
    container = ET.SubElement(root, "proteins")
    for item in proteins.values():
        el = ET.SubElement(container, "protein")
        _emit_id_and_name(el, item)
        _emit_element_states(el, [s for s in element_states if s[0] == item["name"]], ids, proteins)


def _emit_protein_complexes(
    root: ET.Element,
    protein_complexes: Dict[str, Dict[str, Any]],
    ids: IdFactory,
) -> None:
    if not protein_complexes:
        return
    container = ET.SubElement(root, "protein-complexes")
    for item in protein_complexes.values():
        el = ET.SubElement(container, "protein-complex")
        _emit_id_and_name(el, item)


def _emit_id_and_name(parent: ET.Element, item: Dict[str, Any]) -> None:
    id_el = ET.SubElement(parent, "id")
    id_el.set("type", "integer")
    id_el.text = str(item["id"])
    name_el = ET.SubElement(parent, "name")
    name_el.text = item["name"]


def _emit_reactions(
    root: ET.Element,
    reactions: Any,
    entity_maps: Dict[str, Dict[str, Dict[str, Any]]],
    ids: IdFactory,
) -> None:
    if not isinstance(reactions, list) or not reactions:
        return
    container = ET.SubElement(root, "reactions")
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        inputs = [v.strip() for v in reaction.get("inputs", []) if isinstance(v, str) and v.strip()]
        outputs = [v.strip() for v in reaction.get("outputs", []) if isinstance(v, str) and v.strip()]
        if not inputs or not outputs:
            continue
        el = ET.SubElement(container, "reaction")
        rid = ET.SubElement(el, "id")
        rid.set("type", "integer")
        rid.text = str(ids.next())
        direction = ET.SubElement(el, "direction")
        direction.text = "Right"

        left = ET.SubElement(el, "reaction-left-elements")
        for name in inputs:
            _emit_reaction_element(left, name, entity_maps, ids, tag_name="reaction-left-element")

        right = ET.SubElement(el, "reaction-right-elements")
        for name in outputs:
            _emit_reaction_element(right, name, entity_maps, ids, tag_name="reaction-right-element")

        enzymes = reaction.get("enzymes", [])
        enzyme_ids = _resolve_enzymes(enzymes, entity_maps)
        if enzyme_ids:
            enz_container = ET.SubElement(el, "reaction-enzymes")
            for pc_id in enzyme_ids:
                enz = ET.SubElement(enz_container, "reaction-enzyme")
                enz_id = ET.SubElement(enz, "id")
                enz_id.set("type", "integer")
                enz_id.text = str(ids.next())
                pc = ET.SubElement(enz, "protein-complex-id")
                pc.set("type", "integer")
                pc.text = str(pc_id)


def _emit_reaction_element(
    parent: ET.Element,
    name: str,
    entity_maps: Dict[str, Dict[str, Dict[str, Any]]],
    ids: IdFactory,
    tag_name: str = "reaction-left-element",
) -> None:
    element_type = None
    element_id = None
    if name in entity_maps["compounds"]:
        element_type = "Compound"
        element_id = entity_maps["compounds"][name]["id"]
    elif name in entity_maps["element_collections"]:
        element_type = "ElementCollection"
        element_id = entity_maps["element_collections"][name]["id"]
    if not element_type or element_id is None:
        return
    rel = ET.SubElement(parent, tag_name)
    rel_id = ET.SubElement(rel, "id")
    rel_id.set("type", "integer")
    rel_id.text = str(ids.next())
    eid = ET.SubElement(rel, "element-id")
    eid.set("type", "integer")
    eid.text = str(element_id)
    et = ET.SubElement(rel, "element-type")
    et.text = element_type
    sto = ET.SubElement(rel, "stoichiometry")
    sto.set("type", "integer")
    sto.text = "1"


def _resolve_enzymes(
    enzymes: Any,
    entity_maps: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[int]:
    out: List[int] = []
    for item in enzymes if isinstance(enzymes, list) else []:
        if not isinstance(item, dict):
            continue
        name = (item.get("protein_complex") or "").strip()
        if not name:
            continue
        if name in entity_maps["protein_complexes"]:
            out.append(entity_maps["protein_complexes"][name]["id"])
    return out


def _emit_transports(
    root: ET.Element,
    transports: Any,
    entity_maps: Dict[str, Dict[str, Dict[str, Any]]],
    ids: IdFactory,
) -> None:
    if not isinstance(transports, list) or not transports:
        return
    container = ET.SubElement(root, "transports")
    for transport in transports:
        if not isinstance(transport, dict):
            continue
        cargo = (transport.get("cargo") or "").strip()
        if not cargo:
            continue
        el = ET.SubElement(container, "transport")
        tid = ET.SubElement(el, "id")
        tid.set("type", "integer")
        tid.text = str(ids.next())
        elems = ET.SubElement(el, "transport-elements")
        te = ET.SubElement(elems, "transport-element")
        te_id = ET.SubElement(te, "id")
        te_id.set("type", "integer")
        te_id.text = str(ids.next())
        if cargo in entity_maps["compounds"]:
            eid = ET.SubElement(te, "element-id")
            eid.set("type", "integer")
            eid.text = str(entity_maps["compounds"][cargo]["id"])
            et = ET.SubElement(te, "element-type")
            et.text = "Compound"


def _emit_reaction_coupled_transports(
    root: ET.Element,
    rcts: Any,
    entity_maps: Dict[str, Dict[str, Dict[str, Any]]],
    ids: IdFactory,
) -> None:
    if not isinstance(rcts, list) or not rcts:
        return
    container = ET.SubElement(root, "reaction-coupled-transports")
    for rct in rcts:
        if not isinstance(rct, dict):
            continue
        el = ET.SubElement(container, "reaction-coupled-transport")
        rid = ET.SubElement(el, "id")
        rid.set("type", "integer")
        rid.text = str(ids.next())


def _emit_interactions(
    root: ET.Element,
    interactions: Any,
    ids: IdFactory,
) -> None:
    if not isinstance(interactions, list) or not interactions:
        return
    container = ET.SubElement(root, "interactions")
    for inter in interactions:
        if not isinstance(inter, dict):
            continue
        e1 = (inter.get("entity_1") or "").strip()
        e2 = (inter.get("entity_2") or "").strip()
        if not e1 or not e2:
            continue
        el = ET.SubElement(container, "interaction")
        iid = ET.SubElement(el, "id")
        iid.set("type", "integer")
        iid.text = str(ids.next())
        n1 = ET.SubElement(el, "entity-1")
        n1.text = e1
        n2 = ET.SubElement(el, "entity-2")
        n2.text = e2


def _indent(elem: ET.Element, level: int = 0) -> None:
    """In-place pretty indentation for ElementTree."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="Input JSON file (extractor output)")
    p.add_argument("--out", dest="out_path", required=True, help="Output .pwml file path")
    p.add_argument("--name", dest="name", default="Generated Pathway", help="Pathway name")
    p.add_argument("--desc", dest="desc", default="", help="Pathway description")
    p.add_argument("--id", dest="pid", type=int, default=0, help="named-for-id (integer)")
    p.add_argument(
        "--profile",
        dest="profile",
        default="structured",
        choices=("minimal", "structured"),
        help="PWML output profile (minimal or structured)",
    )
    p.add_argument("--subject", dest="subject", default="Metabolic", help="Pathway subject")
    args = p.parse_args()

    inp = Path(args.in_path)
    outp = Path(args.out_path)

    data = json.loads(inp.read_text(encoding="utf-8"))

    if args.profile == "minimal":
        xml = pwml_from_extraction(
            data,
            pathway_name=args.name,
            pathway_description=args.desc,
            named_for_id=args.pid,
        )
    else:
        xml = pwml_from_extraction_structured(
            data,
            pathway_name=args.name,
            pathway_description=args.desc,
            pathway_subject=args.subject,
            named_for_id=args.pid,
        )

    outp.write_text(xml, encoding="utf-8")
    print(f"Wrote PWML to: {outp}")


if __name__ == "__main__":
    main()
