from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from json_to_sbml import build_sbml, sbml_species_id  # noqa: E402
from map_ids import route_entity_for_mapping  # noqa: E402
from process_normalizer import (  # noqa: E402
    dedupe_processes,
    normalize_composites,
    promote_catalysts,
    validate_no_composites,
    validate_registry_references,
)

try:
    import libsbml  # type: ignore  # noqa: F401
except Exception:  # noqa: BLE001
    libsbml = None


def _thyroid_payload() -> dict:
    return {
        "entities": {
            "compounds": [
                {"name": "thyroglobulin + iodotyrosine"},
                {"name": "thyroglobulin"},
                {"name": "iodide"},
                {"name": "hydrogen peroxide"},
                {"name": "2-aminoacrylic acid"},
                {"name": "liothyronine"},
                {"name": "thyroxine"},
            ],
            "proteins": [{"name": "thyroid peroxidase"}],
            "protein_complexes": [],
            "subcellular_locations": [{"name": "follicular lumen"}],
        },
        "biological_states": [{"name": "luminal", "subcellular_location": "follicular lumen"}],
        "element_locations": {
            "compound_locations": [
                {"compound": "thyroglobulin + iodotyrosine", "biological_state": "luminal"},
                {"compound": "iodide", "biological_state": "luminal"},
            ],
            "protein_locations": [{"protein": "thyroid peroxidase", "biological_state": "luminal"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "R0",
                    "inputs": ["thyroglobulin", "iodide", "hydrogen peroxide", "thyroid peroxidase"],
                    "outputs": ["thyroglobulin + iodotyrosine"],
                    "evidence": "observed",
                },
                {
                    "name": "R0",
                    "inputs": ["thyroglobulin", "iodide", "hydrogen peroxide", "thyroid peroxidase"],
                    "outputs": ["thyroglobulin + iodotyrosine"],
                    "inference": {"method": "inference", "confidence": 0.7},
                    "evidence": "short",
                },
                {
                    "name": "R1",
                    "inputs": ["thyroglobulin + iodotyrosine", "iodide", "hydrogen peroxide", "thyroid peroxidase"],
                    "outputs": ["thyroglobulin + 3,5-diiodo-L-tyrosine"],
                    "evidence": "observed",
                },
                {
                    "name": "R2",
                    "inputs": ["thyroglobulin + 3,5-diiodo-L-tyrosine", "hydrogen peroxide", "thyroid peroxidase"],
                    "outputs": [
                        "thyroglobulin+liothyronine",
                        "2-aminoacrylic acid",
                        "thyroglobulin",
                        "liothyronine",
                    ],
                },
                {
                    "name": "R3",
                    "inputs": ["thyroglobulin + liothyronine", "thyroid peroxidase"],
                    "outputs": ["thyroglobulin + thyroxine"],
                },
            ],
            "transports": [
                {
                    "cargo": "thyroglobulin + liothyronine",
                    "from_biological_state": "luminal",
                    "to_biological_state": "luminal",
                    "transporters": [{"protein": "thyroid peroxidase"}],
                },
                {
                    "cargo": "thyroglobulin+liothyronine",
                    "from_biological_state": "luminal",
                    "to_biological_state": "luminal",
                    "transporters": [{"protein": "thyroid peroxidase"}],
                    "inference": {"method": "inference", "confidence": 0.7},
                },
            ],
        },
    }


def _run_normalization(payload: dict) -> tuple[dict, dict]:
    data = json.loads(json.dumps(payload))
    report = {
        "summary": {
            "complexes_created": 0,
            "composites_rewritten": 0,
            "reactions_rewritten": 0,
            "entities_moved_out_of_compounds": 0,
            "entities_added_as_compounds": 0,
            "entities_added_as_proteins": 0,
            "catalysts_promoted_to_enzymes": 0,
            "scaffold_inputs_added": 0,
            "dedupe_removed_reactions": 0,
            "dedupe_removed_transports": 0,
            "dedupe_removed": 0,
        },
        "actions": [],
        "rewrite_map": {},
    }
    normalize_composites(data, report=report)
    promote_catalysts(data, report=report)
    dedupe_processes(data, report=report)
    validate_no_composites(data)
    validate_registry_references(data)
    return data, report


def test_thyroid_normalization_and_dedupe() -> None:
    normalized, report = _run_normalization(_thyroid_payload())

    complexes = {
        item["name"]
        for item in normalized["entities"]["protein_complexes"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    required = {
        "thyroglobulin:iodotyrosine",
        "thyroglobulin:3,5-diiodo-L-tyrosine",
        "thyroglobulin:liothyronine",
        "thyroglobulin:thyroxine",
    }
    assert required.issubset(complexes)
    assert len(complexes) >= 4

    for reaction in normalized["processes"]["reactions"]:
        for side in ["inputs", "outputs"]:
            assert all("+" not in token for token in reaction.get(side, []))
    for transport in normalized["processes"]["transports"]:
        cargo = str(transport.get("cargo_complex") or transport.get("cargo") or "")
        assert "+" not in cargo
    for compound in normalized["entities"]["compounds"]:
        assert "+" not in str(compound.get("name") or "")

    # Duplicate R0 and duplicate transport rows should be collapsed.
    assert len(normalized["processes"]["reactions"]) == 4
    assert len(normalized["processes"]["transports"]) == 1
    assert int(report.get("summary", {}).get("dedupe_removed", 0)) >= 2

    for reaction in normalized["processes"]["reactions"]:
        assert "thyroid peroxidase" not in reaction.get("inputs", [])
        enzymes = reaction.get("enzymes", [])
        names = {str(e.get("protein") or e.get("protein_complex") or e.get("name") or "") for e in enzymes if isinstance(e, dict)}
        assert "thyroid peroxidase" in names


def test_mapping_route_and_species_id_helpers() -> None:
    protein_like_names = {"thyroglobulin", "thyroid peroxidase"}
    assert route_entity_for_mapping("thyroglobulin", "compound", protein_like_names=protein_like_names)["route"] == "protein"
    assert route_entity_for_mapping("thyroglobulin:liothyronine", "compound", protein_like_names=protein_like_names)["route"] == "complex"
    assert route_entity_for_mapping("iodide", "compound", protein_like_names=protein_like_names)["route"] == "compound"

    p_id = sbml_species_id({"kind": "protein", "name": "thyroglobulin", "mapped_ids": {}}, "c_cell")
    m_id = sbml_species_id({"kind": "compound", "name": "iodide", "mapped_ids": {}}, "c_cell")
    assert p_id.startswith("p_")
    assert m_id.startswith("m_")
    assert p_id == sbml_species_id({"kind": "protein", "name": "thyroglobulin", "mapped_ids": {}}, "c_cell")


@pytest.mark.skipif(libsbml is None, reason="python-libsbml not installed")
def test_sbml_single_species_per_entity_and_compartment(tmp_path: Path) -> None:
    normalized, _ = _run_normalization(_thyroid_payload())
    in_path = tmp_path / "in.json"
    sbml_path = tmp_path / "model.sbml"
    report_json = tmp_path / "report.json"
    report_txt = tmp_path / "report.txt"
    in_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")

    build_sbml(in_path, sbml_path, report_json, report_txt, default_compartment_name="follicular lumen")

    import libsbml  # type: ignore

    doc = libsbml.readSBMLFromFile(str(sbml_path))
    model = doc.getModel()
    seen = set()
    for idx in range(model.getNumSpecies()):
        sp = model.getSpecies(idx)
        key = (sp.getName(), sp.getCompartment())
        assert key not in seen
        seen.add(key)
