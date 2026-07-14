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

from t2pw.curation import gap_resolver  # noqa: E402
from t2pw.curation.gap_resolver import _collect_stage3_issues  # noqa: E402


def test_biological_state_without_species_fails_with_species_issue() -> None:
    payload = {
        "entities": {"compounds": [], "proteins": [], "protein_complexes": [], "species": []},
        "biological_states": [{"name": "cytosol", "subcellular_location": "cytosol"}],
        "element_locations": {},
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }

    issues = _collect_stage3_issues(payload, max_items=20)
    biological_state_issues = [issue for issue in issues if issue.get("entity_type") == "biological_state"]

    assert len(biological_state_issues) == 1
    assert biological_state_issues[0]["needs_species"] is True
    assert "biological_state_missing_species" in biological_state_issues[0]["reasons"]


def _ndmcde_payload() -> dict:
    organism = "Pseudomonas putida"
    state = "Pseudomonas cytosol"
    proteins = [
        {"name": name, "species": organism, "mapped_ids": {"uniprot": accession}}
        for name, accession in (
            ("NdmC", "Q88FY2"),
            ("NdmD", "Q88FY1"),
            ("NdmE", "Q88FY0"),
        )
    ]
    return {
        "entities": {
            "species": [
                {
                    "name": organism,
                    "taxonomy_id": "303",
                    "pathbank_species_id": 7,
                    "domain": "Bacteria",
                }
            ],
            "proteins": proteins,
            "protein_complexes": [
                {
                    "name": "NdmCDE",
                    "species": organism,
                    "components": [
                        "NdmC",
                        {"name": "NdmD", "stoichiometry": 2},
                        {"name": "NdmE", "coefficient": "3"},
                    ],
                }
            ],
            "compounds": [],
        },
        "biological_states": [
            {
                "name": state,
                "species": organism,
                "subcellular_location": "cytosol",
            }
        ],
        "element_locations": {
            "compound_locations": [],
            "protein_locations": [
                *[{"protein": row["name"], "biological_state": state} for row in proteins],
                {"protein_complex": "NdmCDE", "biological_state": state},
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "caffeine demethylation",
                    "inputs": ["caffeine"],
                    "outputs": ["theobromine"],
                    "enzymes": [
                        {
                            "entity": "NdmCDE",
                            "entity_type": "protein_complex",
                            "role": "catalyst",
                        }
                    ],
                }
            ]
        },
    }


def test_complex_issue_executes_and_resolves_members_without_default_stoichiometry() -> None:
    payload = _ndmcde_payload()
    original_reactions = payload["processes"]["reactions"]

    resolved, report = gap_resolver.resolve_gaps(
        payload,
        id_source="api",
        use_llm=False,
        enable_id_resolution=False,
    )

    complex_row = resolved["entities"]["protein_complexes"][0]
    components = complex_row["components"]
    assert [component["name"] for component in components] == ["NdmC", "NdmD", "NdmE"]
    assert [component["mapped_ids"]["uniprot"] for component in components] == [
        "Q88FY2",
        "Q88FY1",
        "Q88FY0",
    ]
    assert "stoichiometry" not in components[0]
    assert components[1]["stoichiometry"] == 2
    assert components[2]["coefficient"] == "3"
    assert resolved["processes"]["reactions"] == original_reactions

    execution = next(
        entry
        for entry in report["stage3"]["executions"]
        if entry["issue_key"] == "protein_complex:ndmcde"
    )
    assert execution["status"] == "ok"
    assert execution.get("reason") != "issue_not_found"
    assert execution["unresolved_issues"] == [
        {
            "component_index": 0,
            "component": "NdmC",
            "path": "/entities/protein_complexes/0/components/0",
            "code": "component_missing_stoichiometry",
            "resolution_owner": "audit",
        }
    ]
    complex_issue = next(
        issue
        for issue in report["stage3"]["issues"]
        if issue["issue_key"] == "protein_complex:ndmcde"
    )
    assert complex_issue["component_protein_unresolved"] is False
    assert complex_issue["component_missing_stoichiometry"] is True
    assert complex_issue["needs_id_mapping"] is False
    assert "protein_complex_missing_id_mapping" not in complex_issue["reasons"]
    assert report["summary"]["complex_components_resolved"] == 3


def test_bacterial_location_resolution_rejects_eukaryotic_only_organelles(monkeypatch) -> None:
    payload = _ndmcde_payload()
    payload["entities"]["compounds"] = [
        {
            "name": "theobromine",
            "class": "compound",
            "mapped_ids": {"hmdb": "HMDB0002825"},
        }
    ]

    monkeypatch.setattr(
        gap_resolver,
        "_db_location_candidates",
        lambda *args, **kwargs: [
            {
                "location": "endoplasmic reticulum membrane",
                "score": 100,
                "source": "pathbank_db",
                "evidence": "location_frequency=100",
            },
            {
                "location": "cytosol",
                "score": 2,
                "source": "pathbank_db",
                "evidence": "location_frequency=2",
            },
        ],
    )

    resolved, report = gap_resolver.resolve_gaps(
        payload,
        id_source="api",
        use_llm=False,
        enable_id_resolution=False,
    )

    compound_location = next(
        row
        for row in resolved["element_locations"]["compound_locations"]
        if row["compound"] == "theobromine"
    )
    state_by_name = {row["name"]: row for row in resolved["biological_states"]}
    chosen_state = state_by_name[compound_location["biological_state"]]
    assert chosen_state["subcellular_location"] == "cytosol"

    execution = next(
        entry
        for entry in report["stage3"]["executions"]
        if entry["issue_key"] == "compound:theobromine"
    )
    assert execution["organism_context"]["cell_type"] == "prokaryote"
    assert execution["location_candidates"][0]["location"] == "cytosol"
    assert execution["rejected_location_candidates"][0]["location"] == (
        "endoplasmic reticulum membrane"
    )


def test_unknown_taxonomy_prefers_neutral_location_over_db_frequency() -> None:
    ranked, rejected = gap_resolver._rank_compatible_location_candidates(
        [
            {
                "location": "mitochondrion",
                "score": 100,
                "source": "pathbank_db",
            },
            {
                "location": "cell",
                "score": 1,
                "source": "default",
            },
        ],
        organism_context={
            "organism": "Unclassified organism",
            "taxonomy_id": "",
            "lineage": "",
            "cell_type": "unknown",
        },
    )

    assert [candidate["location"] for candidate in ranked] == ["cell", "mitochondrion"]
    assert ranked[1]["organism_compatibility"] == "uncertain_eukaryotic_organelle"
    assert rejected == []
