from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation import interactive_curator as curator  # noqa: E402
from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402


def test_strip_payload_for_interactive_context_removes_bulky_fields_and_preserves_core_payload() -> None:
    payload = {
        "entities": {
            "compounds": [
                {
                    "name": "ATP",
                    "mapped_id": "HMDB0000538",
                    "selected_id": "ChEBI:15422",
                    "chosen_database_id": "HMDB0000538",
                    "candidate_ids": ["HMDB_BAD_1", "HMDB_BAD_2"],
                    "search_results": [{"id": "candidate"}],
                    "raw_text": "large source text",
                    "enrichment": {"drop": True},
                    "enrichment_cache": {"anything": "bulky"},
                }
            ],
            "proteins": [
                {
                    "name": "Hexokinase",
                    "mapped_ids": ["P52789"],
                    "chosen_ids": ["P52789"],
                    "match_results": [{"id": "candidate"}],
                    "cached_lookup": {"drop": True},
                }
            ],
        },
        "processes": [{"name": "glycolysis"}],
        "reactions": [{"id": "r1", "inputs": ["ATP"], "outputs": ["ADP"]}],
        "biological_states": [{"entity": "ATP", "state": "available"}],
        "element_locations": {"ATP": {"x": 10, "y": 20}},
        "mapping_candidates": [{"id": "HMDB_BAD_1"}],
        "raw_source_text": "paper text",
        "cache": {"drop": True},
    }

    stripped = curator.strip_payload_for_interactive_context(payload)

    assert stripped["processes"] == payload["processes"]
    assert stripped["reactions"] == payload["reactions"]
    assert stripped["biological_states"] == payload["biological_states"]
    assert stripped["element_locations"] == payload["element_locations"]

    compound = stripped["entities"]["compounds"][0]
    assert compound["name"] == "ATP"
    assert compound["mapped_id"] == "HMDB0000538"
    assert compound["selected_id"] == "ChEBI:15422"
    assert compound["chosen_database_id"] == "HMDB0000538"
    assert "candidate_ids" not in compound
    assert "search_results" not in compound
    assert "raw_text" not in compound
    assert "enrichment" not in compound
    assert "enrichment_cache" not in compound

    protein = stripped["entities"]["proteins"][0]
    assert protein["mapped_ids"] == ["P52789"]
    assert protein["chosen_ids"] == ["P52789"]
    assert "match_results" not in protein
    assert "cached_lookup" not in protein
    assert "mapping_candidates" not in stripped
    assert "raw_source_text" not in stripped
    assert "cache" not in stripped


def test_compact_qa_report_recursively_removes_empty_values_and_keeps_real_issues() -> None:
    compacted = curator.compact_qa_report(
        {
            "errors": [],
            "warnings": [
                {},
                None,
                {"path": "/reactions/0", "issues": ["missing output"], "notes": []},
            ],
            "nested": {
                "empty_dict": {},
                "empty_list": [],
                "none": None,
                "real": {"issues": ["orphan entity"]},
            },
        }
    )

    assert compacted == {
        "warnings": [{"path": "/reactions/0", "issues": ["missing output"]}],
        "nested": {"real": {"issues": ["orphan entity"]}},
    }


def test_compact_mapping_misses_detects_misses_in_dict_shaped_reports() -> None:
    report = {
        "summary": {"total": 4},
        "failed_mappings": [
            {"name": "Mystery metabolite", "status": "failed", "mapped_id": None},
            {"entity_name": "Novel enzyme", "reason": "novel protein"},
        ],
        "uncertain": {"entity": "Low confidence compound", "confidence": 0.42},
        "successful": [{"name": "ATP", "mapped_id": "HMDB0000538", "confidence": 0.99}],
    }

    misses = curator.compact_mapping_misses(report)

    names = {miss.get("name") or miss.get("entity_name") or miss.get("entity") for miss in misses}
    assert {"Mystery metabolite", "Novel enzyme", "Low confidence compound"} <= names
    assert "ATP" not in names


def test_compact_mapping_misses_handles_list_shaped_reports_and_none() -> None:
    misses = curator.compact_mapping_misses(
        [
            {"name": "Unmapped lipid", "mapped": False, "candidates": [{"id": "candidate"}]},
            {"entity_name": "Ambiguous protein", "status": "uncertain", "score": 0.5},
        ]
    )

    assert curator.compact_mapping_misses(None) == []
    assert misses == [
        {"name": "Unmapped lipid", "mapped": False},
        {"entity_name": "Ambiguous protein", "status": "uncertain", "score": 0.5},
    ]


def test_compact_history_keeps_last_two_rounds_without_full_patch_bodies() -> None:
    history = [
        {"round": 1, "user_request": "old", "patch": [{"op": "replace"}]},
        {
            "round": 2,
            "user_request": "rename ATP",
            "change_summary": "Renamed ATP.",
            "patch": [{"op": "replace"}, {"op": "add"}],
            "gate_errors": ["reaction missing output"],
            "norm_warnings": ["canonicalized alias"],
            "irrelevant": "drop me",
        },
        {
            "round": 3,
            "user_request": "remove orphan",
            "change_summary": "Removed orphan.",
            "patch_op_count": 1,
            "patch": [{"op": "remove", "path": "/entities/0"}],
            "patch_apply_errors": ["none"],
        },
    ]

    compacted = curator.compact_history(history)

    assert compacted == [
        {
            "round": 2,
            "user_request": "rename ATP",
            "change_summary": "Renamed ATP.",
            "patch_op_count": 2,
            "gate_errors": ["reaction missing output"],
            "norm_warnings": ["canonicalized alias"],
        },
        {
            "round": 3,
            "user_request": "remove orphan",
            "change_summary": "Removed orphan.",
            "patch_op_count": 1,
            "patch_apply_errors": ["none"],
        },
    ]
    assert all("patch" not in item for item in compacted)


def test_build_interactive_curator_messages_returns_system_and_multimodal_user_message() -> None:
    messages = curator.build_interactive_curator_messages(
        working_json={"entities": {"compounds": [{"name": "ATP"}]}},
        graph_png_bytes=b"fake-png",
        user_request="Add glucose input.",
        qa_report={"warnings": [{"issue": "missing glucose"}]},
        mapping_misses=[{"name": "Glucose", "mapped": False}],
        history=[{"round": 1, "user_request": "previous", "patch": [{"op": "add"}]}],
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["type"] == "text"
    assert "Add glucose input." in user_content[0]["text"]
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_validate_interactive_patch_response_accepts_valid_patch_actions() -> None:
    response = curator.validate_interactive_patch_response(
        {
            "patch": [
                {"action": "add", "json_pointer": "/entities/compounds/0", "new_value": {"name": "Glucose"}},
                {"action": "replace", "json_pointer": "/entities/compounds/0/name", "new_value": "D-Glucose"},
                {"action": "remove", "json_pointer": "/entities/compounds/1"},
            ],
            "rationale": "Apply requested changes.",
            "change_summary": "Updated compounds.",
            "needs_user_review": False,
        }
    )

    assert response["patch"][0]["action"] == "add"
    assert response["patch"][1]["action"] == "replace"
    assert response["patch"][2] == {
        "action": "remove",
        "json_pointer": "/entities/compounds/1",
        "new_value": None,
    }
    assert response["needs_user_review"] is False


@pytest.mark.parametrize(
    "patch_op, error",
    [
        ({"action": "move", "json_pointer": "/x", "new_value": 1}, "invalid action"),
        ({"action": "add", "json_pointer": "x", "new_value": 1}, "starting with '/'"),
        ({"action": "add", "json_pointer": "/x"}, "requires new_value"),
        ({"action": "replace", "json_pointer": "/x"}, "requires new_value"),
    ],
)
def test_validate_interactive_patch_response_rejects_invalid_patch_operations(
    patch_op: dict,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        curator.validate_interactive_patch_response(
            {
                "patch": [patch_op],
                "rationale": "",
                "change_summary": "",
                "needs_user_review": True,
            }
        )


@pytest.mark.parametrize(
    "response, error",
    [
        (
            {
                "patch": [],
                "entities": {"compounds": []},
                "rationale": "",
                "change_summary": "",
                "needs_user_review": True,
            },
            "only patch metadata",
        ),
        (
            {
                "patch": [
                    {
                        "action": "replace",
                        "json_pointer": "/entities/compounds/0/mapped_ids/hmdb",
                        "new_value": "HMDB0000122",
                    }
                ],
                "rationale": "",
                "change_summary": "",
                "needs_user_review": True,
            },
            "must not modify mapped IDs",
        ),
        (
            {
                "patch": [
                    {
                        "action": "add",
                        "json_pointer": "/entities/compounds/0",
                        "new_value": {"name": "Glucose", "mapped_ids": {"hmdb": "HMDB0000122"}},
                    }
                ],
                "rationale": "",
                "change_summary": "",
                "needs_user_review": True,
            },
            "must not invent mapped IDs",
        ),
        (
            {
                "patch": [
                    {
                        "action": "replace",
                        "json_pointer": "/",
                        "new_value": {
                            "entities": {"compounds": []},
                            "processes": {"reactions": []},
                        },
                    }
                ],
                "rationale": "",
                "change_summary": "",
                "needs_user_review": True,
            },
            "whole pathway JSON",
        ),
    ],
)
def test_validate_interactive_patch_response_rejects_full_payload_and_db_id_edits(
    response: dict,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        curator.validate_interactive_patch_response(response)


def test_apply_patch_and_rerender_writes_patch_list_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured_patch_payload = None

    def fake_run_apply(input_path: Path, patch_path: Path, output_path: Path, **_: object) -> dict:
        nonlocal captured_patch_payload
        captured_patch_payload = patch_path.read_text(encoding="utf-8")
        output_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
        return {"summary": {"accepted_count": 1, "rejected_count": 0}, "rejected": []}

    def fake_run_mapping(input_path: Path, output_path: Path, report_path: Path, **_: object) -> dict:
        payload = input_path.read_text(encoding="utf-8")
        output_path.write_text(payload, encoding="utf-8")
        report = {"summary": {"mapped": 1}}
        report_path.write_text('{"summary":{"mapped":1}}', encoding="utf-8")
        return report

    class FakeGraph:
        def to_dict(self) -> dict:
            return {"nodes": [], "edges": []}

    monkeypatch.setattr("t2pw.curation.apply_audit_patch.run_apply", fake_run_apply)
    monkeypatch.setattr("t2pw.mapping.map_ids.run_mapping", fake_run_mapping)
    monkeypatch.setattr("t2pw.pipeline.process_normalizer.apply_biochemical_aliases", lambda payload, report=None: None)
    monkeypatch.setattr("t2pw.pipeline.process_normalizer.canonicalize_same_as_aliases", lambda payload, report=None: None)
    monkeypatch.setattr("t2pw.pipeline.process_normalizer.normalize_process_actor_schema", lambda payload, report=None: None)
    monkeypatch.setattr("t2pw.pipeline.process_normalizer.run_strict_post_normalization_gates", lambda payload, report=None, enforce_all_proteins_connected=False: {})
    monkeypatch.setattr("t2pw.pipeline.draft_graph.build_draft_graph", lambda payload: FakeGraph())
    monkeypatch.setattr("t2pw.pipeline.draft_graph_render.render_draft_graph_to_png_bytes", lambda graph, dpi=100: b"png")
    monkeypatch.setattr("t2pw.pipeline.qa_graph.generate_qa_report", lambda graph, payload: {"summary": {"ok": True}})

    result = curator.apply_patch_and_rerender(
        working_json={"entities": {"compounds": [{"name": "ATP"}]}, "processes": {"reactions": []}},
        patch=[{"action": "replace", "json_pointer": "/entities/compounds/0/name", "new_value": "ADP"}],
        mapping_cache_path=tmp_path / "mapping_cache.json",
        work_dir=tmp_path,
    )

    assert result["graph_png_bytes"] == b"png"
    assert captured_patch_payload is not None
    captured = json.loads(captured_patch_payload)
    assert isinstance(captured, list)
    assert captured == [
        {
            "action": "replace",
            "json_pointer": "/entities/compounds/0/name",
            "new_value": "ADP",
            "op": "replace",
            "path": "/entities/compounds/0/name",
            "value": "ADP",
        }
    ]


def test_interactive_connectivity_patch_accepts_090_but_audit_default_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "entities": {
            "compounds": [{"name": "ATP"}, {"name": "ADP"}, {"name": "Glucose"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "ATP conversion",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "biological_state": "cytosol",
                }
            ]
        },
    }
    interactive_patch = [
        {
            "action": "add",
            "json_pointer": "/processes/reactions/0/inputs/-",
            "new_value": "Glucose",
            "confidence": curator.INTERACTIVE_CONNECTIVITY_CONFIDENCE_THRESHOLD,
            "reason": "User asked to add glucose as an input to this reaction.",
        }
    ]
    audit_patch = [
        {
            "op": "add",
            "path": "/processes/reactions/0/inputs/-",
            "value": "Glucose",
            "confidence": curator.INTERACTIVE_CONNECTIVITY_CONFIDENCE_THRESHOLD,
            "evidence": "User asked to add glucose as an input to this reaction.",
        }
    ]

    _, audit_report = apply_patch_with_policy(payload, audit_patch)
    assert audit_report["summary"] == {"accepted_count": 0, "rejected_count": 1, "total": 1}
    assert "confidence >= 0.98" in audit_report["rejected"][0]["reason"]

    def fake_run_mapping(input_path: Path, output_path: Path, report_path: Path, **_: object) -> dict:
        output_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
        report = {"summary": {"mapped": 0}}
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    class FakeGraph:
        def to_dict(self) -> dict:
            return {"nodes": [], "edges": []}

    monkeypatch.setattr("t2pw.mapping.map_ids.run_mapping", fake_run_mapping)
    monkeypatch.setattr("t2pw.pipeline.process_normalizer.apply_biochemical_aliases", lambda payload, report=None: None)
    monkeypatch.setattr("t2pw.pipeline.process_normalizer.canonicalize_same_as_aliases", lambda payload, report=None: None)
    monkeypatch.setattr("t2pw.pipeline.process_normalizer.normalize_process_actor_schema", lambda payload, report=None: None)
    monkeypatch.setattr("t2pw.pipeline.process_normalizer.run_strict_post_normalization_gates", lambda payload, report=None, enforce_all_proteins_connected=False: {})
    monkeypatch.setattr("t2pw.pipeline.draft_graph.build_draft_graph", lambda payload: FakeGraph())
    monkeypatch.setattr("t2pw.pipeline.draft_graph_render.render_draft_graph_to_png_bytes", lambda graph, dpi=100: b"png")
    monkeypatch.setattr("t2pw.pipeline.qa_graph.generate_qa_report", lambda graph, payload: {"summary": {"ok": True}})

    result = curator.apply_patch_and_rerender(
        working_json=payload,
        patch=interactive_patch,
        mapping_cache_path=tmp_path / "mapping_cache.json",
        work_dir=tmp_path,
    )

    assert result["patch_apply_errors"] == []
    assert result["updated_json"]["processes"]["reactions"][0]["inputs"] == ["ATP", "Glucose"]


def test_extract_entity_names_reads_supported_entity_groups_and_handles_missing_keys() -> None:
    payload = {
        "entities": {
            "compounds": [{"name": " ATP "}, {"id": "missing-name"}],
            "proteins": [{"name": "Hexokinase"}],
            "protein_complexes": [{"name": "Complex I"}],
            "nucleic_acids": [{"name": "mRNA A"}],
            "element_collections": [{"name": "Collection A"}],
            "unsupported": [{"name": "Ignore me"}],
        }
    }

    assert curator.extract_entity_names(payload) == {
        "ATP",
        "Hexokinase",
        "Complex I",
        "mRNA A",
        "Collection A",
    }
    assert curator.extract_entity_names({}) == set()
    assert curator.extract_entity_names({"entities": None}) == set()
