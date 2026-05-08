from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple


def _normalize_name(value: str) -> str:
    lowered = value.strip().casefold()
    collapsed = re.sub(r"\s+", " ", lowered)
    return re.sub(r"[^a-z0-9 ]+", "", collapsed)


def _load_json(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _build_lookup(dictionary: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for section, values in dictionary.items():
        if not isinstance(values, dict):
            continue
        section_lookup: Dict[str, Dict[str, Any]] = {}
        for name, attrs in values.items():
            if not isinstance(name, str) or not isinstance(attrs, dict):
                continue
            section_lookup[_normalize_name(name)] = attrs
        lookup[section] = section_lookup
    return lookup


def apply_grounding(extracted: Dict[str, Any], dictionary: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    grounded = json.loads(json.dumps(extracted))
    lookup = _build_lookup(dictionary)

    entities = grounded.get("entities")
    if not isinstance(entities, dict):
        return grounded, {"sections": {}, "notes": ["No entities block found in input JSON."]}

    section_aliases = {
        "cell_types": "cell_types",
        "species": "species",
        "tissues": "tissues",
        "subcellular_locations": "subcellular_locations",
        "compounds": "compounds",
        "element_collections": "element_collections",
        "nucleic_acids": "nucleic_acids",
        "proteins": "proteins",
        "protein_complexes": "protein_complexes",
    }

    report: Dict[str, Any] = {"sections": {}, "notes": []}
    for entity_key, dict_key in section_aliases.items():
        items = entities.get(entity_key, [])
        if not isinstance(items, list):
            continue
        matches = 0
        misses = 0
        touched = 0
        section_lookup = lookup.get(dict_key, {})
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            attrs = section_lookup.get(_normalize_name(name))
            if not attrs:
                misses += 1
                continue
            matches += 1
            for field, value in attrs.items():
                if field not in item or item.get(field) in ("", None, []):
                    item[field] = value
                    touched += 1
        report["sections"][entity_key] = {
            "matched_names": matches,
            "unmatched_names": misses,
            "fields_added": touched,
        }

    report["notes"].append(
        "Grounding adds deterministic metadata only from the provided dictionary. No model inference is used."
    )
    return grounded, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic metadata grounding for extracted PWML JSON.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input extracted JSON")
    parser.add_argument("--dict", dest="dict_path", required=True, help="Grounding dictionary JSON")
    parser.add_argument("--out", dest="out_path", required=True, help="Output grounded JSON")
    parser.add_argument("--report", dest="report_path", default="", help="Optional grounding report path")
    args = parser.parse_args()

    extracted = _load_json(args.in_path)
    dictionary = _load_json(args.dict_path)
    grounded, report = apply_grounding(extracted, dictionary)

    out_path = Path(args.out_path)
    out_path.write_text(json.dumps(grounded, indent=2), encoding="utf-8")

    report_path = Path(args.report_path) if args.report_path else out_path.with_suffix(".grounding_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Grounded JSON written to: {out_path}")
    print(f"Grounding report: {report_path}")


if __name__ == "__main__":
    main()
