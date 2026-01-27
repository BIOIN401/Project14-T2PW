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
    args = p.parse_args()

    inp = Path(args.in_path)
    outp = Path(args.out_path)

    data = json.loads(inp.read_text(encoding="utf-8"))

    xml = pwml_from_extraction(
        data,
        pathway_name=args.name,
        pathway_description=args.desc,
        named_for_id=args.pid,
    )

    outp.write_text(xml, encoding="utf-8")
    print(f"Wrote PWML to: {outp}")


if __name__ == "__main__":
    main()
