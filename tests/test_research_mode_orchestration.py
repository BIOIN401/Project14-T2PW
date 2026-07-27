"""The orchestrator wires research mode to the seams it is supposed to.

These are structural (AST) assertions against ``streamlit_app.py`` rather than
live runs, because the behaviours they guard are *call policies* -- which flag
is passed to which stage -- and a live run would need the whole LLM/DB stack.
The plan's "Idiom C": cheap insurance against a relaxation silently becoming a
deletion, or against research mode quietly re-enabling a strict-only policy.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.export_mode import (  # noqa: E402
    FORMAT_ISSUE_CODES,
    STRUCTURAL_GUARD_CODES,
    classify_issue,
)


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _map_payload_calls() -> list[ast.Call]:
    node = _function("run_post_pipeline_sbml_artifacts")
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "map_payload"
    ]


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def test_orchestrator_still_makes_exactly_two_mapping_calls() -> None:
    # Stage 2 (preview, never wraps) and Stage 6 (the real remap).
    assert len(_map_payload_calls()) == 2


def test_stage6_wrapper_creation_is_conditional_on_research_mode() -> None:
    stage6 = _map_payload_calls()[1]
    wrapper = _kwarg(stage6, "allow_complex_wrapper_creation")

    # `not research_mode` — the wrapper is dropped entirely in research mode.
    assert isinstance(wrapper, ast.UnaryOp), ast.dump(wrapper) if wrapper else "missing"
    assert isinstance(wrapper.op, ast.Not)
    assert isinstance(wrapper.operand, ast.Name)
    assert wrapper.operand.id == "research_mode"


def test_stage2_mapping_policy_is_unchanged() -> None:
    stage2 = _map_payload_calls()[0]
    wrapper = _kwarg(stage2, "allow_complex_wrapper_creation")

    # Stage 2 never wrapped, in either mode. It must stay a hard False.
    assert isinstance(wrapper, ast.Constant) and wrapper.value is False


def test_every_map_payload_call_threads_the_export_mode() -> None:
    for call in _map_payload_calls():
        assert _kwarg(call, "mode") is not None


def test_normalization_threads_the_export_mode() -> None:
    node = _function("run_post_pipeline_sbml_artifacts")
    calls = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "normalize_process_payload"
    ]
    assert calls, "normalize_process_payload is no longer called"
    for call in calls:
        assert _kwarg(call, "mode") is not None


def test_every_stage_contract_goes_through_the_mode_aware_wrapper() -> None:
    """A bare ``validate_*`` call would abort even in research mode."""

    node = _function("run_post_pipeline_sbml_artifacts")
    bare = [
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id.startswith("validate_post_")
    ]
    assert bare == [], f"these bypass the export-mode wrapper: {sorted(set(bare))}"


def test_orchestrator_defaults_to_strict() -> None:
    node = _function("run_post_pipeline_sbml_artifacts")
    names = [arg.arg for arg in node.args.kwonlyargs]
    assert "export_mode" in names

    default = node.args.kw_defaults[names.index("export_mode")]
    assert isinstance(default, ast.Name) and default.id == "DEFAULT_EXPORT_MODE"


# --------------------------------------------------------------------------
# The category sets themselves.
# --------------------------------------------------------------------------


def test_format_and_guard_sets_are_disjoint() -> None:
    assert FORMAT_ISSUE_CODES & STRUCTURAL_GUARD_CODES == frozenset()


@pytest.mark.parametrize(
    "code, expected",
    [
        ("generated_wrapper_component_missing_external_identity", "format"),
        ("mapping_resolution_field_invalid", "format"),
        ("invalid_payload", "guard"),
        ("process_not_object", "guard"),
        ("reaction_missing_participants", "review"),
        ("species_required", "review"),
        # The safe default: a code nobody enumerated must be reviewed, not skipped.
        ("some_check_added_next_year", "review"),
        ("", "review"),
    ],
)
def test_issue_classification(code, expected) -> None:
    assert classify_issue({"code": code}) == expected


def test_actor_schema_is_flagged_not_skipped() -> None:
    """One code covers both a bucket tag (format) and the catalyst identity.

    Splitting it would change strict-mode output, so research mode keeps the
    whole check as a review flag -- the "err toward keep-but-flag" rule.
    """

    assert classify_issue({"code": "actor_schema_not_canonical"}) == "review"
