import ast
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "t2pw" / "app" / "streamlit_app.py"


def _load_stage8_validator() -> Any:
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_stage8_export_payload"
    )

    from t2pw.pipeline.process_normalizer import (
        GateValidationError,
        run_strict_post_normalization_gates,
    )
    from t2pw.pipeline.stage_contracts import validate_post_normalization

    namespace = {
        "Any": Any,
        "Dict": Dict,
        "Tuple": Tuple,
        "GateValidationError": GateValidationError,
        "run_strict_post_normalization_gates": run_strict_post_normalization_gates,
        "validate_post_normalization": validate_post_normalization,
        "_safe_dict": lambda value: value if isinstance(value, dict) else {},
    }
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, APP_PATH, "exec"), namespace)
    return namespace["_validate_stage8_export_payload"]


def _stage6_payload() -> dict[str, Any]:
    return {
        "metadata": {
            "name": "Caffeine demethylation",
            "pathway_name": "Caffeine demethylation",
            "subject": "Metabolic",
            "pathway_subject": "Metabolic",
        },
        "entities": {
            "species": [{"name": "Pseudomonas putida", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [
                {"name": "caffeine", "pathbank_compound_id": 101},
                {"name": "theobromine", "pathbank_compound_id": 102},
            ],
            "proteins": [
                {
                    "name": "NdmA",
                    "species": "Pseudomonas putida",
                    "mapped_ids": {"uniprot": "Q9I147"},
                }
            ],
            "protein_complexes": [
                {
                    "name": "NdmA complex",
                    "species": "Pseudomonas putida",
                    "generated": True,
                    "generation_reason": "single_protein_pathwhiz_wrapper",
                    "components": [{"name": "NdmA", "stoichiometry": 1}],
                }
            ],
        },
        "biological_states": [
            {
                "name": "Pseudomonas putida cytosol",
                "species": "Pseudomonas putida",
                "subcellular_location": "cytosol",
            }
        ],
        "processes": {
            "reactions": [
                {
                    "name": "caffeine demethylation",
                    "inputs": ["caffeine"],
                    "outputs": ["theobromine"],
                    "biological_state": "Pseudomonas putida cytosol",
                    "enzymes": [
                        {
                            "entity": "NdmA complex",
                            "entity_type": "protein_complex",
                            "role": "catalyst",
                        }
                    ],
                    "modifiers": [
                        {
                            "entity": "NdmA complex",
                            "entity_type": "protein_complex",
                            "role": "catalyst",
                        }
                    ],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


def test_stage8_validation_preserves_stage6_actor_collections_and_ir_succeeds() -> None:
    from t2pw.pipeline.stage_contracts import validate_pre_export
    from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir

    payload = _stage6_payload()
    before = deepcopy(payload)
    actors_before = deepcopy(payload["processes"]["reactions"][0]["enzymes"])

    gate_report, contract_report = _load_stage8_validator()(payload)
    pre_export_report = validate_pre_export(payload, strict_db=True)
    ir, ir_report = build_pwml_ir(payload, strict_db=True)

    assert gate_report["ok"] is True
    assert contract_report["ok"] is True
    assert pre_export_report["ok"] is True
    assert not ir_report["errors"]
    assert validate_pwml_ir(ir)["ok"] is True
    assert payload == before
    assert payload["processes"]["reactions"][0]["enzymes"] == actors_before
    assert ir["processes"]["reactions"][0]["enzymes"][0]["entity_type"] == "protein_complex"


def test_stage8_missing_state_fails_pre_export_without_creating_one() -> None:
    from t2pw.pipeline.stage_contracts import StageContractError, validate_pre_export

    payload = _stage6_payload()
    payload["biological_states"] = []
    before = deepcopy(payload)

    _gate_report, _contract_report = _load_stage8_validator()(payload)
    with pytest.raises(StageContractError) as exc_info:
        validate_pre_export(payload, strict_db=True)

    assert payload == before
    assert payload["biological_states"] == []
    assert exc_info.value.report["ok"] is False


def test_stage8_bare_protein_actor_fails_ir_without_wrapper_repair() -> None:
    from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir

    payload = _stage6_payload()
    reaction = payload["processes"]["reactions"][0]
    reaction["enzymes"] = [
        {"entity": "NdmA", "entity_type": "protein", "role": "catalyst"}
    ]
    reaction["modifiers"] = deepcopy(reaction["enzymes"])
    before = deepcopy(payload)

    _gate_report, _contract_report = _load_stage8_validator()(payload)
    ir, ir_report = build_pwml_ir(payload, strict_db=True)

    assert payload == before
    assert [row["name"] for row in payload["entities"]["protein_complexes"]] == [
        "NdmA complex"
    ]
    assert reaction["enzymes"][0]["entity"] == "NdmA"
    assert "reaction_enzyme_must_be_protein_complex" in {
        error["code"] for error in ir_report["errors"]
    }
    assert validate_pwml_ir(ir)["ok"] is False


def test_run_pwml_export_has_no_stage8_normalization_or_autostate_calls() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_pwml_export"
    )
    called_names = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "normalize_process_payload" not in called_names
    assert "ensure_autostates" not in called_names
    assert "map_payload" not in called_names
