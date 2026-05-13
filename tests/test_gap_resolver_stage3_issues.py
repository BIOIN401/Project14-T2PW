from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

llm_client = types.ModuleType("t2pw.llm.client")
llm_client.chat = lambda *args, **kwargs: "{}"
llm_client.chat_with_tools = lambda *args, **kwargs: "{}"
sys.modules.setdefault("t2pw.llm.client", llm_client)

from t2pw.curation.gap_resolver import _collect_stage3_issues  # noqa: E402


def _issues_by_type(payload: dict) -> dict[str, list[dict]]:
    issues = _collect_stage3_issues(payload, max_items=50)
    out: dict[str, list[dict]] = {}
    for issue in issues:
        out.setdefault(str(issue.get("entity_type")), []).append(issue)
    return out


def test_collect_stage3_issues_includes_species_states_complexes_and_locations() -> None:
    payload = {
        "entities": {
            "species": [{"name": "Homo sapiens", "taxonomy_id": "9606"}],
            "proteins": [{"name": "MPC complex"}],
            "protein_complexes": [
                {
                    "name": "MPC complex",
                    "components": [
                        {"name": "MPC1"},
                        {"name": "MPC2", "stoichiometry": 1, "mapping_status": "unmapped"},
                    ],
                },
                {"name": "empty complex", "species": "Homo sapiens", "species_id": 1, "components": []},
            ],
            "compounds": [{"name": "pyruvate", "mapping_meta": {"provider": "PathBankDB", "source": "db", "resolution": {"status": "unresolved"}}}],
        },
        "biological_states": [{"name": "cytosolic"}],
        "element_locations": {
            "compound_locations": [],
            "protein_locations": [{"protein": "MPC complex"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "pyruvate transport",
                    "enzymes": [{"protein": "MPC complex"}],
                }
            ]
        },
    }

    by_type = _issues_by_type(payload)

    assert by_type["species"][0]["species_missing_db_id"] is True
    assert by_type["biological_state"][0]["needs_species"] is True
    assert by_type["biological_state"][0]["needs_subcellular_location"] is True

    compound_issue = by_type["compound"][0]
    assert compound_issue["compound_missing_class_type"] is True
    assert compound_issue["compound_unmapped_after_db_pass"] is True
    assert compound_issue["visible_entity_missing_location"] is True

    complex_issues = {issue["name"]: issue for issue in by_type["protein_complex"]}
    assert complex_issues["MPC complex"]["needs_species"] is True
    assert complex_issues["MPC complex"]["component_missing_stoichiometry"] is True
    assert complex_issues["MPC complex"]["component_protein_unresolved"] is True
    assert complex_issues["empty complex"]["protein_complex_missing_components"] is True

    direct_enzyme_issue = by_type["reaction_modifier"][0]
    assert direct_enzyme_issue["enzyme_still_direct_protein_when_complex_required"] is True
