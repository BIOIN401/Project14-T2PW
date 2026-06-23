from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **_: None)
            )

    class _FakeOpenAIError(Exception):
        pass

    fake_openai.OpenAI = _FakeOpenAI
    fake_openai.RateLimitError = _FakeOpenAIError
    fake_openai.APIError = _FakeOpenAIError
    fake_openai.APITimeoutError = _FakeOpenAIError
    fake_openai.AuthenticationError = _FakeOpenAIError
    fake_openai.BadRequestError = _FakeOpenAIError
    sys.modules["openai"] = fake_openai

from t2pw.pipeline.pipeline import build_and_save_draft_graph, filter_unresolvable_reactions  # noqa: E402


def test_filter_unresolvable_reactions_removes_ghosts_and_keeps_partial_matches() -> None:
    payload = {
        "entities": {
            "compounds": [
                {"name": "hexadecatrienoic acid (16:3)"},
                {"name": "OPC-8:0-CoA"},
                {"name": "OPC-6:0-CoA"},
            ],
            "proteins": [{"name": "12-oxophytodienoate reductase"}],
            "protein_complexes": [
                {
                    "name": "hexadecatrienoic acid (16:3)",
                    "components": ["hexadecatrienoic acid (16", "3)"],
                },
                {
                    "name": "valid enzyme complex",
                    "components": ["12-oxophytodienoate reductase"],
                },
            ],
            "nucleic_acids": [],
            "element_collections": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "good reaction",
                    "inputs": ["OPC-8:0-CoA", "ATP"],
                    "outputs": ["OPC-6:0-CoA", "ADP"],
                },
                {
                    "name": "bad reaction",
                    "inputs": [],
                    "outputs": ["some product"],
                },
                {
                    "name": "ghost reaction",
                    "inputs": ["nonexistent_fragment_A"],
                    "outputs": ["nonexistent_fragment_B"],
                },
            ]
        },
    }

    filtered, removed = filter_unresolvable_reactions(payload)

    reactions = filtered["processes"]["reactions"]
    reaction_names = [reaction["name"] for reaction in reactions]
    complex_names = [
        protein_complex["name"]
        for protein_complex in filtered["entities"]["protein_complexes"]
    ]

    assert reaction_names == ["good reaction"]
    assert removed == ["bad reaction", "ghost reaction"]
    assert "hexadecatrienoic acid (16:3)" not in complex_names
    assert complex_names == ["valid enzyme complex"]


def test_locked_reaction_with_missing_compound_is_repaired_not_dropped(tmp_path) -> None:
    payload = {
        "entities": {
            "compounds": [{"name": "A"}],
            "proteins": [],
            "protein_complexes": [],
            "nucleic_acids": [],
            "element_collections": [],
        },
        "processes": {
            "reactions": [
                {
                    "locked_reaction_id": "rxn_lock_009",
                    "name": "locked direct reaction",
                    "inputs": ["A"],
                    "outputs": ["compound X"],
                }
            ]
        },
    }

    artifact_path = tmp_path / "quarantined_locked_reactions.json"
    filtered, removed = filter_unresolvable_reactions(
        payload,
        quarantine_output_path=artifact_path,
    )

    assert removed == []
    assert [reaction["name"] for reaction in filtered["processes"]["reactions"]] == ["locked direct reaction"]
    assert {"name": "compound X"} in filtered["entities"]["compounds"]
    assert filtered["processes"]["reactions"][0]["preservation_status"] == "entity_repaired"
    assert filtered["quarantined_locked_reactions"] == []
    assert filtered["locked_reaction_filter_report"] == {
        "locked_reactions_found": 1,
        "exported_locked_reactions": 1,
        "quarantined_locked_reactions": 0,
    }
    assert artifact_path.read_text(encoding="utf-8")


def test_locked_reaction_with_unrepairable_empty_side_is_quarantined(tmp_path) -> None:
    payload = {
        "entities": {
            "compounds": [{"name": "A"}],
            "proteins": [],
            "protein_complexes": [],
            "nucleic_acids": [],
            "element_collections": [],
        },
        "processes": {
            "reactions": [
                {
                    "locked_reaction_id": "rxn_lock_010",
                    "name": "locked incomplete reaction",
                    "inputs": ["A"],
                    "outputs": [],
                }
            ]
        },
    }

    artifact_path = tmp_path / "quarantined_locked_reactions.json"
    filtered, removed = filter_unresolvable_reactions(
        payload,
        quarantine_output_path=artifact_path,
    )

    assert removed == []
    assert filtered["processes"]["reactions"] == []
    assert filtered["locked_reaction_filter_report"] == {
        "locked_reactions_found": 1,
        "exported_locked_reactions": 0,
        "quarantined_locked_reactions": 1,
    }
    assert filtered["quarantined_locked_reactions"] == [
        {
            "locked_reaction_id": "rxn_lock_010",
            "reaction_name": "locked incomplete reaction",
            "reason": "missing_outputs",
            "missing_entities": ["<missing outputs>"],
            "original_reaction": {
                "locked_reaction_id": "rxn_lock_010",
                "name": "locked incomplete reaction",
                "inputs": ["A"],
                "outputs": [],
            },
        }
    ]
    assert artifact_path.read_text(encoding="utf-8")


def test_locked_manifest_marks_matching_reaction_for_repair_without_filesystem() -> None:
    payload = {
        "entities": {
            "compounds": [{"name": "A"}],
            "proteins": [],
            "protein_complexes": [],
            "nucleic_acids": [],
            "element_collections": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "manifest locked reaction",
                    "inputs": ["A"],
                    "outputs": ["compound Y"],
                }
            ]
        },
    }
    locked_manifest = [
        {
            "locked_reaction_id": "rxn_lock_011",
            "source_reaction_id": "",
            "name": "manifest locked reaction",
            "inputs": ["A"],
            "outputs": ["compound Y"],
        }
    ]

    filtered, removed = filter_unresolvable_reactions(
        payload,
        locked_manifest=locked_manifest,
    )

    assert removed == []
    reaction = filtered["processes"]["reactions"][0]
    assert reaction["locked_reaction_id"] == "rxn_lock_011"
    assert reaction["preservation_status"] == "entity_repaired"
    assert {"name": "compound Y"} in filtered["entities"]["compounds"]


def test_build_and_save_draft_graph_reports_locked_export_and_quarantine_counts(tmp_path) -> None:
    payload = {
        "metadata": {"pathway_name": "Test pathway", "organism": "test organism"},
        "entities": {
            "compounds": [{"name": "A"}, {"name": "B"}],
            "proteins": [],
            "protein_complexes": [],
            "nucleic_acids": [],
            "element_collections": [],
        },
        "processes": {
            "reactions": [
                {
                    "locked_reaction_id": "rxn_lock_001",
                    "name": "kept reaction",
                    "inputs": ["A"],
                    "outputs": ["B"],
                }
            ]
        },
        "quarantined_locked_reactions": [
            {
                "locked_reaction_id": "rxn_lock_002",
                "reaction_name": "quarantined reaction",
                "reason": "missing_outputs",
                "missing_entities": ["<missing outputs>"],
                "original_reaction": {},
            }
        ],
        "locked_reaction_filter_report": {
            "locked_reactions_found": 2,
            "exported_locked_reactions": 1,
            "quarantined_locked_reactions": 1,
        },
    }

    _graph, _qa_report, summary = build_and_save_draft_graph(
        payload,
        output_path=tmp_path / "draft_graph.json",
    )

    assert "2 locked reactions found" in summary
    assert "1 exported" in summary
    assert "1 quarantined reaction due to unresolved compound reference" in summary
    artifact = tmp_path / "quarantined_locked_reactions.json"
    assert artifact.exists()
