from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "t2pw" / "app" / "streamlit_app.py"

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The lifted orchestrator now takes an export-mode policy. These are injected
# for real (not stubbed) so this suite keeps proving that the default, strict
# path calls each contract exactly as it always did.
from t2pw.pipeline.export_mode import (  # noqa: E402
    DEFAULT_EXPORT_MODE,
    PATHWHIZ,
    coerce_mode,
    format_gaps,
    is_research,
    review_flags,
)
from t2pw.pipeline.stage_contracts import run_stage_contract  # noqa: E402


def _source_tree() -> ast.Module:
    return ast.parse(APP_PATH.read_text(encoding="utf-8"))


def _function_node(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _source_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_function(name: str, namespace: dict[str, Any]) -> Any:
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            _function_node(name),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, APP_PATH, "exec"), namespace)
    return namespace[name]


def _paper_payload() -> dict[str, Any]:
    return {
        "metadata": {
            "pathway_name": "Paper pathway",
            "organism": "Arabidopsis thaliana",
        },
        "entities": {
            "species": [{"name": "Arabidopsis thaliana"}],
            "compounds": [{"name": "substrate"}, {"name": "product"}],
            "proteins": [{"name": "paper enzyme"}],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "paper reaction",
                    "inputs": ["substrate"],
                    "outputs": ["product"],
                    "enzymes": [
                        {
                            "entity": "paper enzyme",
                            "entity_type": "protein",
                            "role": "catalyst",
                        }
                    ],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


def _invoke_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "build_legacy_sbml": False,
        "use_llm_audit": False,
        "use_sbml_overwatch": False,
        "default_compartment": "cell",
        "mapping_cache_path": str(tmp_path / "mapping-cache.json"),
        "id_source": "hybrid",
        "db_host": "db.example",
        "db_port": 3306,
        "db_user": "reader",
        "db_password": "secret",
        "db_schema": "pathbank",
        "audit_max_rounds": 1,
        "audit_timeout_seconds": 120,
        "audit_candidate_count": 1,
        "use_example_retrieval": False,
        "example_index_path": "missing-index.json",
        "example_top_k": 1,
        "use_gap_resolver": False,
        "use_llm_gap_resolver": False,
        "gap_resolver_max_items": 5,
    }


def _orchestration_harness(
    tmp_path: Path,
    *,
    stage2_result: Any = None,
    stage2_error: Exception | None = None,
    post_mapping_error: Exception | None = None,
    accepted_patch_count: int = 0,
    audit_error_count: int = 0,
    gap_change_rounds: set[int] | None = None,
    strict_gate_errors: list[dict[str, Any]] | None = None,
    gap_stage3_issues_by_round: dict[int, list[dict[str, Any]]] | None = None,
    normalized_quarantine: list[dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    calls: list[tuple[str, Any]] = []
    runtime_reports: list[str] = []
    db_config_ids: list[int] = []
    audit_quarantine_snapshots: list[dict[str, Any] | None] = []
    gap_call_count = 0
    gap_change_rounds = gap_change_rounds or set()
    strict_gate_errors = strict_gate_errors or []
    gap_stage3_issues_by_round = gap_stage3_issues_by_round or {}
    stage2_payload = deepcopy(_paper_payload())
    stage2_payload["stage_marker"] = "stage2"
    stage2_report = {
        "summary": {
            "proteins_total": 1,
            "proteins_mapped": 0,
            "proteins_ambiguous": 0,
            "compounds_total": 2,
            "compounds_mapped": 0,
            "compounds_ambiguous": 0,
            "protein_complexes_total": 0,
            "protein_complexes_mapped": 0,
            "protein_complexes_ambiguous": 0,
            "generated_wrappers_created": 0,
        },
        "entities": [{"name": "paper enzyme", "status": "unmapped"}],
    }
    if stage2_result is None:
        stage2_result = {"payload": stage2_payload, "report": stage2_report}

    def map_payload(payload: dict[str, Any], **kwargs: Any) -> Any:
        calls.append(("map", {"payload": payload, **kwargs}))
        db_config_ids.append(id(kwargs["db_config"]))
        if len(db_config_ids) == 1:
            if stage2_error is not None:
                raise stage2_error
            return stage2_result
        stage6_payload = deepcopy(payload)
        stage6_payload["stage_marker"] = "stage6"
        stage6_payload["entities"]["protein_complexes"] = [
            {
                "name": "paper enzyme wrapper",
                "generated": True,
                "generation_reason": "single_protein_pathwhiz_wrapper",
                "components": [{"name": "paper enzyme"}],
            }
        ]
        return {
            "payload": stage6_payload,
            "report": {"summary": {"generated_wrappers_created": 1}},
        }

    def validate_post_extraction(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(("validate_post_extraction", payload))
        return {"ok": True, "runtime_schema_report": {"boundary": "post_extraction"}}

    def validate_post_mapping(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(("validate_post_mapping", payload))
        if post_mapping_error is not None:
            raise post_mapping_error
        return {"ok": True, "runtime_schema_report": {"boundary": "post_mapping"}}

    def normalize_process_payload(
        payload: dict[str, Any],
        *,
        on_checkpoint: Any,
        mode: Any = DEFAULT_EXPORT_MODE,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # The default run must stay strict; research mode is opt-in only.
        assert mode == PATHWHIZ, f"expected strict normalization, got {mode!r}"
        calls.append(("normalize", payload))
        normalized = deepcopy(payload)
        normalized["stage_marker"] = "stage3"
        if normalized_quarantine is not None:
            normalized["quarantined_locked_reactions"] = deepcopy(normalized_quarantine)
        return normalized, {"summary": {}, "gate": {"ok": True, "errors": []}}

    def validate_post_normalization(
        payload: dict[str, Any], report: dict[str, Any]
    ) -> dict[str, Any]:
        calls.append(("validate_post_normalization", payload))
        return {"ok": True, "runtime_schema_report": {"boundary": "post_normalization"}}

    def validate_post_audit(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(("validate_post_audit", payload))
        return {"ok": True, "runtime_schema_report": {"boundary": "post_audit"}}

    def validate_post_remap(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(("validate_post_remap", payload))
        return {"ok": True, "runtime_schema_report": {"boundary": "post_remap"}}

    def validate_runtime_payload_contract(
        payload: dict[str, Any], *, boundary: str
    ) -> dict[str, Any]:
        runtime_reports.append(boundary)
        return {
            "ok": True,
            "boundary": boundary,
            "mode": "report",
            "errors": [],
            "warnings": [],
            "summary": {"error_count": 0, "warning_count": 0},
        }

    def run_audit(
        _input_path: Path,
        report_path: Path,
        patch_path: Path,
        **_kwargs: Any,
    ) -> None:
        calls.append(("audit", _input_path))
        canonical_quarantine_path = tmp_path / "tmp" / "quarantined_locked_reactions.json"
        audit_quarantine_snapshots.append(
            json.loads(canonical_quarantine_path.read_text(encoding="utf-8"))
            if canonical_quarantine_path.exists()
            else None
        )
        report_path.write_text(
            json.dumps(
                {
                    "summary": {
                        "error_count": audit_error_count,
                        "warning_count": 0,
                        "patch_count": accepted_patch_count,
                    },
                    "errors": [],
                    "llm": {},
                }
            ),
            encoding="utf-8",
        )
        patch_path.write_text("[]", encoding="utf-8")

    def run_apply(
        input_path: Path,
        _patch_path: Path,
        output_path: Path,
        *,
        apply_report_path: Path,
        **_kwargs: Any,
    ) -> None:
        applied_payload = json.loads(input_path.read_text(encoding="utf-8"))
        if accepted_patch_count > 0:
            applied_payload["accepted_audit_patch"] = output_path.stem
        output_path.write_text(json.dumps(applied_payload), encoding="utf-8")
        apply_report_path.write_text(
            json.dumps(
                {
                    "summary": {
                        "accepted_count": accepted_patch_count,
                        "rejected_count": 0,
                    }
                }
            ),
            encoding="utf-8",
        )

    def run_strict_post_normalization_gates(
        payload: dict[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        calls.append(("strict_gate", payload))
        return {
            "ok": not strict_gate_errors,
            "errors": deepcopy(strict_gate_errors),
            "summary": {"error_count": len(strict_gate_errors)},
        }

    class _Graph:
        def to_dict(self) -> dict[str, Any]:
            return {}

    def run_enrichment(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("offline enrichment in orchestration test")

    def run_gap_resolution(
        input_path: Path,
        output_path: Path,
        _report_path: Path,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal gap_call_count
        gap_call_count += 1
        calls.append(("gap_resolution", gap_call_count))
        resolved_payload = json.loads(input_path.read_text(encoding="utf-8"))
        changed = gap_call_count in gap_change_rounds
        stage3_issues = deepcopy(gap_stage3_issues_by_round.get(gap_call_count, []))
        if changed:
            resolved_payload["gap_resolution_round"] = gap_call_count
        output_path.write_text(json.dumps(resolved_payload), encoding="utf-8")
        return {
            "summary": {
                "mapped_ids_added": int(changed),
                "locations_added": 0,
                "location_states_filled": 0,
            },
            "db": {},
            "stage3": {
                "issues": stage3_issues,
                "executions": [
                    {
                        "issue_key": issue.get("issue_key", ""),
                        "status": "skipped",
                        "reason": "unresolved_after_attempt",
                    }
                    for issue in stage3_issues
                ],
            },
        }

    namespace: dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "PROJECT_ROOT": tmp_path,
        "Path": Path,
        "deepcopy": deepcopy,
        "dict": dict,
        "json": json,
        "hashlib": hashlib,
        "shutil": shutil,
        "time": time,
        "uuid4": lambda: SimpleNamespace(hex="orchestration-test"),
        "_safe_dict": lambda value: value if isinstance(value, dict) else {},
        "_safe_list": lambda value: value if isinstance(value, list) else [],
        "_read_json": lambda path: json.loads(path.read_text(encoding="utf-8")),
        "_audit_objective_score": lambda **values: tuple(values.values()),
        "_write_reaction_preservation_report": lambda *_args, **_kwargs: None,
        "_save_pipeline_outputs": lambda *_args, **_kwargs: "",
        "map_payload": map_payload,
        "validate_post_extraction": validate_post_extraction,
        "validate_post_mapping": validate_post_mapping,
        "validate_post_normalization": validate_post_normalization,
        "validate_post_audit": validate_post_audit,
        "validate_post_remap": validate_post_remap,
        "validate_runtime_payload_contract": validate_runtime_payload_contract,
        "normalize_process_payload": normalize_process_payload,
        "run_strict_post_normalization_gates": run_strict_post_normalization_gates,
        "run_audit": run_audit,
        "run_apply": run_apply,
        "run_gap_resolution": run_gap_resolution,
        "run_pathway_curator": lambda *_args, **_kwargs: {"summary": {}},
        "run_enrichment": run_enrichment,
        "load_motif_index": lambda *_args, **_kwargs: {},
        "payload_to_query_text": lambda *_args, **_kwargs: "",
        "build_retrieval_context": lambda *_args, **_kwargs: ("", {}),
        "build_draft_graph": lambda *_args, **_kwargs: _Graph(),
        "generate_qa_report": lambda *_args, **_kwargs: {},
        "generate_reaction_summary": lambda *_args, **_kwargs: "",
        "graph_summary": lambda *_args, **_kwargs: {},
        "compute_normalization_stats": lambda *_args, **_kwargs: {},
        "build_sbml": lambda *_args, **_kwargs: {},
        "run_sbml_overwatch": lambda *_args, **_kwargs: {},
        "build_render_artifacts": lambda *_args, **_kwargs: {},
        "strip_unmapped": lambda *_args, **_kwargs: (b"", {}),
        "GateValidationError": RuntimeError,
        "DEFAULT_EXPORT_MODE": DEFAULT_EXPORT_MODE,
        "PATHWHIZ": PATHWHIZ,
        "run_stage_contract": run_stage_contract,
        "coerce_mode": coerce_mode,
        "is_research": is_research,
        "review_flags": review_flags,
        "format_gaps": format_gaps,
    }
    function = _load_function("run_post_pipeline_sbml_artifacts", namespace)
    observations = {
        "calls": calls,
        "runtime_reports": runtime_reports,
        "db_config_ids": db_config_ids,
        "audit_quarantine_snapshots": audit_quarantine_snapshots,
        "stage2_payload": stage2_payload,
        "stage2_report": stage2_report,
    }
    return function, observations


def test_live_orchestration_maps_stage2_then_stage6_with_exact_policies(
    tmp_path: Path,
) -> None:
    function, observed = _orchestration_harness(tmp_path)
    original = _paper_payload()

    result = function(original, **_invoke_kwargs(tmp_path))

    call_names = [name for name, _ in observed["calls"]]
    assert call_names.index("validate_post_extraction") < call_names.index("map")
    assert call_names.index("validate_post_mapping") == call_names.index("map") + 1
    assert call_names.index("validate_post_mapping") < call_names.index("normalize")
    assert call_names.index("normalize") < call_names.index("validate_post_audit")
    assert call_names.index("validate_post_audit") < call_names.index("validate_post_remap")

    mapping_calls = [details for name, details in observed["calls"] if name == "map"]
    assert len(mapping_calls) == 2
    assert mapping_calls[0]["use_cache"] is True
    assert mapping_calls[0]["allow_complex_wrapper_creation"] is False
    assert mapping_calls[0]["allow_structural_cleanup"] is False
    assert mapping_calls[1]["use_cache"] is False
    assert mapping_calls[1]["allow_complex_wrapper_creation"] is True
    assert mapping_calls[1]["allow_structural_cleanup"] is True
    assert observed["db_config_ids"][0] == observed["db_config_ids"][1]

    normalization_input = next(
        details for name, details in observed["calls"] if name == "normalize"
    )
    assert normalization_input is observed["stage2_payload"]
    assert result["pre_mapping_input"] == original
    assert result["stage2_payload"] is observed["stage2_payload"]
    assert result["pre_normalization_input"] is result["stage2_payload"]
    assert result["stage2_mapping_report"] is observed["stage2_report"]
    assert result["stage2_mapping_report"] is not result["mapping_report"]
    assert result["stage2_runtime_schema_report"]["boundary"] == "post_mapping"
    assert observed["runtime_reports"] == ["post_enrichment", "pre_export"]


def test_post_normalization_refreshes_canonical_locked_quarantine_artifact(
    tmp_path: Path,
) -> None:
    stale_path = tmp_path / "tmp" / "quarantined_locked_reactions.json"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text(
        json.dumps({"quarantined_locked_reactions": [{"reason": "stale"}]}),
        encoding="utf-8",
    )
    normalized_quarantine = [
        {
            "locked_reaction_id": "rxn_lock_generic",
            "reason": "coarse_grained_same_entity_transformation",
            "original_reaction": {"inputs": ["A"], "outputs": ["A"]},
        }
    ]
    function, observed = _orchestration_harness(
        tmp_path,
        normalized_quarantine=normalized_quarantine,
    )

    result = function(_paper_payload(), **_invoke_kwargs(tmp_path))

    expected = {"quarantined_locked_reactions": normalized_quarantine}
    assert json.loads(stale_path.read_text(encoding="utf-8")) == expected
    assert result["locked_reaction_quarantine_artifact"] == expected
    assert result["locked_reaction_quarantine_path"] == str(stale_path)
    assert observed["audit_quarantine_snapshots"][0] == expected
    call_names = [name for name, _details in observed["calls"]]
    assert call_names.index("normalize") < call_names.index("audit")


@pytest.mark.parametrize(
    ("stage2_result", "stage2_error", "match"),
    [
        ({"payload": {}, "report": []}, None, "payload and report must be objects"),
        (None, RuntimeError("mapper unavailable"), "mapper unavailable"),
    ],
)
def test_stage2_failure_prevents_stage3_and_second_mapping(
    tmp_path: Path,
    stage2_result: Any,
    stage2_error: Exception | None,
    match: str,
) -> None:
    function, observed = _orchestration_harness(
        tmp_path,
        stage2_result=stage2_result,
        stage2_error=stage2_error,
    )

    with pytest.raises(Exception, match=match):
        function(_paper_payload(), **_invoke_kwargs(tmp_path))

    assert [name for name, _ in observed["calls"]] == [
        "validate_post_extraction",
        "map",
    ]


def test_unmapped_stage2_payload_reaches_audit_and_strict_gate_cadence_is_unchanged(
    tmp_path: Path,
) -> None:
    function, observed = _orchestration_harness(tmp_path, accepted_patch_count=1)

    result = function(_paper_payload(), **_invoke_kwargs(tmp_path))

    assert result["stage2_mapping_report"]["entities"][0]["status"] == "unmapped"
    assert [name for name, _ in observed["calls"]].count("normalize") == 1
    assert [name for name, _ in observed["calls"]].count("strict_gate") == 1
    assert len([name for name, _ in observed["calls"] if name == "map"]) == 2


def test_gap_only_progress_gets_fresh_gate_and_another_audit_round(
    tmp_path: Path,
) -> None:
    function, observed = _orchestration_harness(
        tmp_path,
        accepted_patch_count=0,
        audit_error_count=0,
        gap_change_rounds={1},
        strict_gate_errors=[
            {
                "path": "/processes/reactions/0/enzymes",
                "reason": "Reaction enzyme remains unresolved.",
            }
        ],
    )
    invoke_kwargs = _invoke_kwargs(tmp_path)
    invoke_kwargs.update(
        {
            "audit_max_rounds": 3,
            "use_gap_resolver": True,
        }
    )

    result = function(_paper_payload(), **invoke_kwargs)

    assert [name for name, _ in observed["calls"]].count("gap_resolution") == 2
    assert [name for name, _ in observed["calls"]].count("strict_gate") == 1
    assert len(result["audit_iterations"]) == 2
    first_round, second_round = result["audit_iterations"]
    assert first_round["accepted_patch_count"] == 0
    assert first_round["audit_payload_changed"] is False
    assert first_round["gap_payload_changed"] is True
    assert first_round["settled_payload_changed"] is True
    assert first_round["post_settlement_gate"]["ok"] is False
    assert first_round["current_stage3_gate_error_count"] == 1
    assert first_round["actionable_issues_remain"] is True
    assert second_round["gap_payload_changed"] is False
    assert second_round["settled_payload_changed"] is False
    assert result["audit_loop_summary"]["stop_reason"] == (
        "stalled_unchanged_payload_with_actionable_issues"
    )


def test_gap_only_progress_stops_when_fresh_stage3_gate_is_clean(
    tmp_path: Path,
) -> None:
    function, observed = _orchestration_harness(
        tmp_path,
        accepted_patch_count=0,
        audit_error_count=0,
        gap_change_rounds={1},
    )
    invoke_kwargs = _invoke_kwargs(tmp_path)
    invoke_kwargs.update(
        {
            "audit_max_rounds": 3,
            "use_gap_resolver": True,
        }
    )

    result = function(_paper_payload(), **invoke_kwargs)

    assert [name for name, _ in observed["calls"]].count("gap_resolution") == 1
    assert [name for name, _ in observed["calls"]].count("strict_gate") == 1
    assert len(result["audit_iterations"]) == 1
    assert result["audit_iterations"][0]["post_settlement_gate"]["ok"] is True
    assert result["audit_loop_summary"]["stop_reason"] == "clean_post_settlement_gate"


def test_unresolved_gap_complex_continues_past_clean_audit_and_gate(
    tmp_path: Path,
) -> None:
    unresolved_complex = {
        "issue_key": "protein_complex:ndmcde",
        "entity_type": "protein_complex",
        "name": "NdmCDE",
        "reasons": ["component_protein_unresolved"],
        "component_protein_unresolved": True,
    }
    function, observed = _orchestration_harness(
        tmp_path,
        accepted_patch_count=0,
        audit_error_count=0,
        gap_change_rounds={1},
        gap_stage3_issues_by_round={
            1: [unresolved_complex],
            2: [unresolved_complex],
        },
    )
    invoke_kwargs = _invoke_kwargs(tmp_path)
    invoke_kwargs.update(
        {
            "audit_max_rounds": 3,
            "use_gap_resolver": True,
        }
    )

    result = function(_paper_payload(), **invoke_kwargs)

    assert [name for name, _ in observed["calls"]].count("gap_resolution") == 2
    assert [name for name, _ in observed["calls"]].count("strict_gate") == 1
    assert len(result["audit_iterations"]) == 2
    first_round, second_round = result["audit_iterations"]
    assert first_round["error_count"] == 0
    assert first_round["post_settlement_gate"]["ok"] is True
    assert first_round["actionable_gap_issue_count"] == 1
    assert first_round["actionable_issues_remain"] is True
    assert first_round["settled_payload_changed"] is True
    assert result["gap_resolution_iterations"][0]["actionable_issue_count"] == 1
    assert second_round["actionable_gap_issue_count"] == 1
    assert second_round["settled_payload_changed"] is False
    assert result["audit_loop_summary"]["stop_reason"] == (
        "stalled_unchanged_payload_with_actionable_issues"
    )


def test_post_mapping_contract_failure_stops_before_normalization_and_stage6(
    tmp_path: Path,
) -> None:
    function, observed = _orchestration_harness(
        tmp_path,
        post_mapping_error=ValueError("post-mapping contract failed"),
    )

    with pytest.raises(ValueError, match="post-mapping contract failed"):
        function(_paper_payload(), **_invoke_kwargs(tmp_path))

    assert [name for name, _ in observed["calls"]] == [
        "validate_post_extraction",
        "map",
        "validate_post_mapping",
    ]


def test_stage_contract_failure_renderer_exposes_complete_structured_report() -> None:
    class _Failure(Exception):
        def __init__(self, report: dict[str, Any]) -> None:
            super().__init__("participant contract failed")
            self.stage = "post_extraction"
            self.report = report

    class _StreamlitRecorder:
        def __init__(self) -> None:
            self.errors: list[str] = []
            self.infos: list[str] = []
            self.writes: list[Any] = []
            self.json_reports: list[Any] = []
            self.downloads: list[dict[str, Any]] = []

        def error(self, value: str) -> None:
            self.errors.append(value)

        def info(self, value: str) -> None:
            self.infos.append(value)

        def write(self, value: Any) -> None:
            self.writes.append(value)

        def json(self, value: Any) -> None:
            self.json_reports.append(value)

        def download_button(self, label: str, **kwargs: Any) -> None:
            self.downloads.append({"label": label, **kwargs})

    report = {
        "ok": False,
        "stage": "post_extraction",
        "contract_type": "structural",
        "effect_on_failure": "abort",
        "summary": {"error_count": 2, "warning_count": 1},
        "errors": [
            {
                "code": "process_missing_participants",
                "pointer": "/processes/interactions/0",
                "message": "Process participants are missing.",
                "bucket": "interactions",
                "name": "enzyme association",
            },
            {
                "code": "entity_missing_name",
                "path": "/entities/proteins/1/name",
                "message": "Entity name is missing.",
            },
        ],
        "warnings": [
            {
                "code": "runtime_shape_warning",
                "pointer": "/metadata/source",
                "message": "Source metadata has an unexpected shape.",
            }
        ],
    }
    recorder = _StreamlitRecorder()
    namespace: dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Tuple": Tuple,
        "StageContractError": _Failure,
        "json": json,
        "st": recorder,
        "_safe_dict": lambda value: value if isinstance(value, dict) else {},
        "_safe_list": lambda value: value if isinstance(value, list) else [],
    }
    render_failure = _load_function("_render_stage_contract_failure", namespace)

    render_failure(
        _Failure(report),
        operation_label="Audit and DB mapping",
        download_key="contract-download",
    )

    assert recorder.errors == [
        "Audit and DB mapping stopped at the "
        "Stage 2A merged-payload → Stage 2B DB mapping input boundary."
    ]
    assert recorder.infos == [
        "Stage 2B DB mapping and Stage 4 audit did not run. "
        "The merged extraction/inference payload was rejected before the mapper was called."
    ]
    assert recorder.writes[0] == {
        "boundary": "Stage 2A merged-payload → Stage 2B DB mapping input boundary",
        "stage": "post_extraction",
        "effect_on_failure": "abort",
        "summary": {"error_count": 2, "warning_count": 1},
        "message": "participant contract failed",
    }
    displayed_issues = [
        value
        for value in recorder.writes
        if isinstance(value, dict) and "severity" in value
    ]
    assert displayed_issues == [
        {
            "severity": "error",
            "code": "process_missing_participants",
            "pointer": "/processes/interactions/0",
            "path": "",
            "message": "Process participants are missing.",
            "bucket": "interactions",
            "name": "enzyme association",
        },
        {
            "severity": "error",
            "code": "entity_missing_name",
            "pointer": "",
            "path": "/entities/proteins/1/name",
            "message": "Entity name is missing.",
        },
        {
            "severity": "warning",
            "code": "runtime_shape_warning",
            "pointer": "/metadata/source",
            "path": "",
            "message": "Source metadata has an unexpected shape.",
        },
    ]
    assert recorder.json_reports == [report]
    assert len(recorder.downloads) == 1
    assert recorder.downloads[0]["file_name"] == "stage_contract_error_report.json"
    assert recorder.downloads[0]["mime"] == "application/json"
    assert recorder.downloads[0]["key"] == "contract-download"
    assert json.loads(recorder.downloads[0]["data"]) == report

    later_report = {
        "stage": "post_mapping",
        "effect_on_failure": "abort",
        "summary": {"error_count": 1, "warning_count": 0},
        "errors": [],
        "warnings": [],
    }
    later_recorder = _StreamlitRecorder()
    namespace["st"] = later_recorder
    render_later_failure = _load_function("_render_stage_contract_failure", namespace)
    render_later_failure(
        _Failure(later_report),
        operation_label="Audit and DB mapping",
        download_key="later-contract-download",
    )
    assert "Stage 2B mapping output → Stage 3 normalization input boundary" in (
        later_recorder.errors[0]
    )
    assert later_recorder.infos == [
        "Stage 2B DB mapping ran, but Stage 3 normalization and Stage 4 audit did not run."
    ]


def test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference() -> None:
    tree = _source_tree()

    def _calls_named(name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]

    extraction_calls = _calls_named("run_stage_one_with_chunking")
    filter_calls = _calls_named("filter_out_of_scope_reactions")
    inference_calls = _calls_named("run_stage_two_with_feedback_loop")

    assert len(extraction_calls) == 1
    assert len(filter_calls) == 1
    assert len(inference_calls) == 1

    # The filter must run strictly after Stage 1 extraction returns and strictly
    # before its output is handed to Stage 2 inference, so out-of-scope reactions
    # never reach the inference pass.
    assert extraction_calls[0].lineno < filter_calls[0].lineno < inference_calls[0].lineno

    # Stage 2 inference must be called with the *filtered* Stage 1 payload, not
    # the raw extraction output.
    inference_call = inference_calls[0]
    stage_one_arg = inference_call.args[1]
    assert isinstance(stage_one_arg, ast.Name)
    assert stage_one_arg.id != "stage_one"

    # merge_additions (used to build final_payload when inference ran) must also
    # be seeded from the filtered payload.
    merge_calls = _calls_named("merge_additions")
    assert merge_calls
    merge_base_arg = merge_calls[0].args[0]
    assert isinstance(merge_base_arg, ast.Name)
    assert merge_base_arg.id != "stage_one"


def test_stage2_artifacts_are_visible_and_both_ui_paths_share_the_orchestrator() -> None:
    namespace: dict[str, Any] = {"Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "Tuple": Tuple}
    entries_function = _load_function("_json_artifact_entries", namespace)
    entries = entries_function(
        {
            "stage2_payload": {"stage": 2},
            "stage2_mapping_report": {"summary": {"mapped": 1}},
            "stage2_runtime_schema_report": {"boundary": "post_mapping"},
            "mapping_report": {"summary": {"stage": 6}},
        },
        None,
    )
    filenames = {filename for _label, filename, _value in entries}
    assert {
        "stage2.mapped.json",
        "stage2_mapping_report.json",
        "stage2_runtime_schema_report.json",
        "mapping_report.json",
    } <= filenames

    tree = _source_tree()
    live_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_post_pipeline_sbml_artifacts"
    ]
    assert len(live_calls) == 2
    assert {
        next(
            keyword.value.value
            for keyword in call.keywords
            if keyword.arg == "build_legacy_sbml" and isinstance(keyword.value, ast.Constant)
        )
        for call in live_calls
    } == {False, True}

    orchestrator_tries = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "run_post_pipeline_sbml_artifacts"
            for statement in node.body
            for candidate in ast.walk(statement)
        )
    ]
    assert len(orchestrator_tries) == 2
    for try_node in orchestrator_tries:
        assert isinstance(try_node.handlers[0].type, ast.Name)
        assert try_node.handlers[0].type.id == "StageContractError"
        assert any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "_render_stage_contract_failure"
            for statement in try_node.handlers[0].body
            for candidate in ast.walk(statement)
        )
        assert isinstance(try_node.handlers[1].type, ast.Name)
        assert try_node.handlers[1].type.id == "Exception"

    for button_label in ("Run audit and DB mapping", "Run legacy SBML export"):
        button_branch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Attribute)
            and node.test.func.attr == "button"
            and isinstance(node.test.args[0], ast.Constant)
            and node.test.args[0].value == button_label
        )
        assert any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and candidate.func.attr == "pop"
            and candidate.args
            and isinstance(candidate.args[0], ast.Constant)
            and candidate.args[0].value == "post_pipeline_artifacts"
            for candidate in ast.walk(button_branch)
        )

    source = APP_PATH.read_text(encoding="utf-8")
    assert "Mapped pathway still fails the pre-export Stage 3 revalidation." in source
    for filename in (
        "stage2.mapped.json",
        "stage2_mapping_report.json",
        "stage2_runtime_schema_report.json",
    ):
        assert f'Download {filename}' in source
