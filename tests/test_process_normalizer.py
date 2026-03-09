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
from qa_graph import build_graph, connected_components  # noqa: E402
from process_normalizer import (  # noqa: E402
    attach_transporters_from_evidence,
    cleanup_disallowed_complexes,
    compute_normalization_stats,
    dedupe_processes,
    ensure_autostates,
    normalize_composites,
    normalize_process_actor_schema,
    promote_catalysts,
    rewrite_reactions_to_complex_states,
    run_strict_post_normalization_gates,
    validate_no_scaffold_modifiers,
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
            "proteins": [{"name": "thyroid peroxidase"}, {"name": "pendrin"}],
            "protein_complexes": [],
            "subcellular_locations": [{"name": "follicular lumen"}, {"name": "bloodstream"}],
        },
        "biological_states": [
            {"name": "luminal", "subcellular_location": "follicular lumen"},
            {"name": "blood", "subcellular_location": "bloodstream"},
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": "thyroglobulin + iodotyrosine", "biological_state": "luminal"},
                {"compound": "iodide", "biological_state": "luminal"},
            ],
            "protein_locations": [
                {"protein": "thyroid peroxidase", "biological_state": "luminal"},
                {
                    "protein": "pendrin",
                    "biological_state": "luminal",
                    "evidence": "iodide is transported through pendrin",
                },
            ],
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
                    "enzymes": [{"protein_complex": "thyroid_peroxidase_complex", "evidence": "legacy schema"}],
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
                    "evidence": "thyroglobulin + liothyronine is catalyzed by thyroid peroxidase to produce thyroglobulin + thyroxine",
                },
            ],
            "transports": [
                {
                    "cargo": "iodide",
                    "from_biological_state": "blood",
                    "to_biological_state": "luminal",
                    "evidence": "thyroglobulin is joined by iodide that has been transported from the blood using pendrin",
                    "transporters": [{"protein_complex": "pendrin_complex", "evidence": "legacy schema"}],
                },
                {
                    "cargo": "thyroglobulin + liothyronine",
                    "from_biological_state": "luminal",
                    "to_biological_state": "luminal",
                    "evidence": "thyroglobulin + liothyronine complex transport",
                },
                {
                    "cargo": "thyroglobulin+liothyronine",
                    "from_biological_state": "luminal",
                    "to_biological_state": "luminal",
                    "evidence": "thyroglobulin + liothyronine complex transport",
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
            "scaffold_split_reactions": 0,
            "entities_moved_out_of_compounds": 0,
            "entities_added_as_compounds": 0,
            "entities_added_as_proteins": 0,
            "catalysts_promoted_to_enzymes": 0,
            "scaffold_inputs_added": 0,
            "scaffold_in_modifiers_count": 0,
            "n_plus_tokens_remaining": 0,
            "complexes_list": [],
            "n_autostate_created": 0,
            "n_entities_assigned_to_autostate": 0,
            "transporters_attached": 0,
            "modifier_refs_canonicalized": 0,
            "modifier_refs_dropped": 0,
            "forbidden_complexes_removed": 0,
            "dedupe_removed_reactions": 0,
            "dedupe_removed_transports": 0,
            "dedupe_removed": 0,
            "dedupe_removed_total": 0,
            "no_op_removed_count": 0,
        },
        "actions": [],
        "rewrite_map": {},
    }
    normalize_composites(data, report=report)
    rewrite_reactions_to_complex_states(data, report=report)
    cleanup_disallowed_complexes(data, report=report)
    ensure_autostates(data, report=report)
    attach_transporters_from_evidence(data, report=report)
    promote_catalysts(data, report=report)
    normalize_process_actor_schema(data, report=report)
    dedupe_processes(data, report=report)
    run_strict_post_normalization_gates(
        data,
        report=report,
        forbidden_complexes=["thyroglobulin:2-aminoacrylic acid"],
        enforce_all_proteins_connected=True,
    )
    return data, report


def test_thyroid_normalization_and_dedupe() -> None:
    normalized, report = _run_normalization(_thyroid_payload())
    summary = report.get("summary", {})

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
    assert int(summary.get("complexes_created", 0)) >= 4
    assert "thyroglobulin:2-aminoacrylic acid" not in complexes
    forbidden = "thyroglobulin:2-aminoacrylic acid"
    assert forbidden not in json.dumps(normalized)
    for reaction in normalized["processes"]["reactions"]:
        assert forbidden not in reaction.get("inputs", [])
        assert forbidden not in reaction.get("outputs", [])
    for transport in normalized["processes"]["transports"]:
        assert str(transport.get("cargo") or "") != forbidden
        assert str(transport.get("cargo_complex") or "") != forbidden

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
    assert len(normalized["processes"]["transports"]) == 2
    assert int(summary.get("dedupe_removed", 0)) >= 2
    assert int(summary.get("dedupe_removed_total", 0)) >= 2
    assert int(summary.get("n_plus_tokens_remaining", 1)) == 0
    assert int(summary.get("reactions_rewritten", 0)) > 0
    assert int(summary.get("scaffold_in_modifiers_count", 0)) == 0
    assert int(summary.get("transporters_attached", 0)) > 0
    assert int(summary.get("modifier_refs_canonicalized", 0)) > 0
    assert int(summary.get("dedupe_removed_total", 0)) > 0 or int(summary.get("no_op_removed_count", 0)) > 0

    for reaction in normalized["processes"]["reactions"]:
        assert "thyroid peroxidase" not in reaction.get("inputs", [])
        enzymes = reaction.get("enzymes", [])
        names = {str(e.get("protein") or e.get("protein_complex") or e.get("name") or "") for e in enzymes if isinstance(e, dict)}
        assert "thyroid peroxidase" in names
        assert "thyroglobulin" not in names

    adj, _ = build_graph(normalized)
    comps = connected_components(adj)
    assert all(not n.startswith("cargo:") for n in adj.keys())
    assert all("+" not in n for n in adj.keys())
    assert len(comps) <= 1
    assert len(adj.get("protein:thyroid peroxidase", set())) > 0
    assert len(adj.get("protein:pendrin", set())) > 0
    protein_names = {
        str(item.get("name") or "").strip()
        for item in normalized.get("entities", {}).get("proteins", [])
        if isinstance(item, dict)
    }
    proteins_degree0 = sum(1 for pname in protein_names if len(adj.get(f"protein:{pname}", set())) == 0)
    assert proteins_degree0 == 0

    iodide_transports = [
        t
        for t in normalized["processes"]["transports"]
        if str(t.get("cargo") or t.get("cargo_complex") or "").strip().casefold() == "iodide"
    ]
    assert iodide_transports
    assert any(
        any(
            str(r.get("protein") or r.get("protein_complex") or r.get("name") or "").strip().casefold() == "pendrin"
            for r in t.get("transporters", [])
            if isinstance(r, dict)
        )
        for t in iodide_transports
    )
    for transport in normalized["processes"]["transports"]:
        for transporter in transport.get("transporters", []):
            if not isinstance(transporter, dict):
                continue
            assert "pendrin_complex" not in str(transporter.get("protein_complex") or "").casefold()

    for reaction in normalized["processes"]["reactions"]:
        for enzyme in reaction.get("enzymes", []):
            if not isinstance(enzyme, dict):
                continue
            value = str(enzyme.get("protein") or "").strip().casefold()
            if value == "thyroid peroxidase":
                assert "protein_complex" not in enzyme
            assert "thyroid_peroxidase_complex" not in str(enzyme.get("protein_complex") or "").casefold()


def test_generic_explicit_composite_still_materializes_complex() -> None:
    payload = {
        "entities": {
            "compounds": [{"name": "ligand x"}],
            "proteins": [{"name": "carrier protein"}],
            "protein_complexes": [],
            "subcellular_locations": [{"name": "cytosol"}],
        },
        "biological_states": [{"name": "cyto_state", "subcellular_location": "cytosol"}],
        "element_locations": {
            "compound_locations": [{"compound": "ligand x", "biological_state": "cyto_state"}],
            "protein_locations": [{"protein": "carrier protein", "biological_state": "cyto_state"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "carrier binding",
                    "inputs": ["carrier protein", "ligand x"],
                    "outputs": ["carrier protein + ligand x"],
                    "biological_state": "cyto_state",
                    "evidence": "carrier protein + ligand x forms in cytosol",
                }
            ],
            "transports": [],
        },
    }

    normalized, report = _run_normalization(payload)
    complexes = {
        item["name"]
        for item in normalized["entities"]["protein_complexes"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    assert "carrier protein:ligand x" in complexes
    assert "carrier protein:ligand x" in normalized["processes"]["reactions"][0]["outputs"]
    assert int(report.get("summary", {}).get("n_plus_tokens_remaining", 1)) == 0
    adj, _ = build_graph(normalized)
    assert len(connected_components(adj)) <= 1


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
    assert int(doc.checkConsistency()) >= 0
    severe_errors = [
        doc.getError(i)
        for i in range(doc.getNumErrors())
        if int(doc.getError(i).getSeverity()) >= 2
    ]
    assert not severe_errors
    model = doc.getModel()
    seen = set()
    for idx in range(model.getNumSpecies()):
        sp = model.getSpecies(idx)
        key = (sp.getName(), sp.getCompartment())
        assert key not in seen
        seen.add(key)

    pendrin_species = [
        model.getSpecies(i).getId()
        for i in range(model.getNumSpecies())
        if model.getSpecies(i).getName().strip().casefold() == "pendrin"
    ]
    assert pendrin_species
    pendrin_ids = set(pendrin_species)
    found_pendrin_transport_modifier = False
    for ridx in range(model.getNumReactions()):
        rxn = model.getReaction(ridx)
        reactant_names = {
            model.getSpecies(rxn.getReactant(r).getSpecies()).getName().strip().casefold()
            for r in range(rxn.getNumReactants())
        }
        if "iodide" not in reactant_names:
            continue
        modifier_species_ids = {rxn.getModifier(m).getSpecies() for m in range(rxn.getNumModifiers())}
        if modifier_species_ids & pendrin_ids:
            found_pendrin_transport_modifier = True
            break
    assert found_pendrin_transport_modifier
    assert model.getNumReactions() <= 7
