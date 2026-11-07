"""PROPOSED rewrite (needs a PO ruling; NOT committed to tests/) of
tests/test_streamlit_stage2_orchestration.py::test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference
after R3d moves Stage 2 into service/extraction.py. Proof copy: ROOT is pinned to the tree under test
from its own location, since this file lives outside tests/.

Same rule as Q-R3c-1, one level further in:
  service/extraction.py run_stage_one: extraction before the filter; StageOneResult(stage_one_in_scope=) is the
      filter's output (unchanged);
  service/extraction.py run_stage_two: exactly one Stage-2 inference call; its args[1] is the Name bound from
      ``stage_one_result.stage_one_in_scope`` (review F1); the first merge_additions base is not ``stage_one``;
  app: exactly one run_stage_one( and one run_stage_two( call, in that order, and run_stage_two is handed the
      run_stage_one result itself (the Name it was assigned to).
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]  # proof copy only: it lives in docs/webserver/evidence/R3d/proposed/
APP_PATH = ROOT / "src" / "t2pw" / "app" / "streamlit_app.py"


def _source_tree() -> ast.Module:
    return ast.parse(APP_PATH.read_text(encoding="utf-8"))


def test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference() -> None:
    tree = _source_tree()

    def _calls_named(name: str, within: ast.AST = tree, attr: bool = False) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(within)
            if isinstance(node, ast.Call)
            and (getattr(node.func, "attr", None) if attr and isinstance(node.func, ast.Attribute)
                 else getattr(node.func, "id", None)) == name
        ]

    # Q-R3c-1 / R3d: Stage 1, the filter and Stage 2 live in service/extraction.py (run_stage_one and
    # run_stage_two), as call-time ``_pipeline.x(...)`` calls; the app calls the two steps in order.
    service_module = ast.parse((ROOT / "src" / "t2pw" / "service" / "extraction.py").read_text(encoding="utf-8"))
    service = next(node for node in service_module.body
                   if isinstance(node, ast.FunctionDef) and node.name == "run_stage_one")
    stage_two_fn = next(node for node in service_module.body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_stage_two")
    extraction_calls = _calls_named("run_stage_one_with_chunking", service, attr=True)
    filter_calls = _calls_named("filter_out_of_scope_reactions", service, attr=True)
    inference_calls = _calls_named("run_stage_two_with_feedback_loop", stage_two_fn, attr=True)
    stage_one_calls = _calls_named("run_stage_one")
    stage_two_calls = _calls_named("run_stage_two")

    assert len(extraction_calls) == 1
    assert len(filter_calls) == 1
    assert len(inference_calls) == 1
    assert len(stage_one_calls) == 1
    assert len(stage_two_calls) == 1

    # The filter must run strictly after Stage 1 extraction returns and strictly
    # before its output is handed to Stage 2 inference, so out-of-scope reactions
    # never reach the inference pass.
    assert extraction_calls[0].lineno < filter_calls[0].lineno
    assert stage_one_calls[0].lineno < stage_two_calls[0].lineno
    # ... and what run_stage_one returns as the in-scope payload is the filter's output.
    filtered = next(node.targets[0] for node in ast.walk(service)
                    if isinstance(node, ast.Assign) and node.value is filter_calls[0])
    returned = [kw.value for call in _calls_named("StageOneResult", service)
                for kw in call.keywords if kw.arg == "stage_one_in_scope"]
    assert len(returned) == 1 and isinstance(returned[0], ast.Name)
    assert returned[0].id == filtered.elts[0].id
    # ... and the app hands that result itself to run_stage_two (review F1).
    stage_one_result = next(node.targets[0] for node in ast.walk(tree)
                            if isinstance(node, ast.Assign) and node.value is stage_one_calls[0])
    assert isinstance(stage_two_calls[0].args[1], ast.Name)
    assert stage_two_calls[0].args[1].id == stage_one_result.id

    # Stage 2 inference must be called with the *filtered* Stage 1 payload, not
    # the raw extraction output: the name run_stage_two binds from the result's in-scope field.
    stage_one_arg = inference_calls[0].args[1]
    assert isinstance(stage_one_arg, ast.Name)
    assert stage_one_arg.id != "stage_one"
    in_scope_from = [node.value for node in ast.walk(stage_two_fn)
                     if isinstance(node, ast.Assign) and len(node.targets) == 1
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == stage_one_arg.id]
    assert len(in_scope_from) == 1 and isinstance(in_scope_from[0], ast.Attribute)
    assert in_scope_from[0].attr == "stage_one_in_scope"

    # merge_additions (used to build final_payload when inference ran) must also
    # be seeded from the filtered payload.
    merge_calls = _calls_named("merge_additions", stage_two_fn)
    assert merge_calls
    merge_base_arg = merge_calls[0].args[0]
    assert isinstance(merge_base_arg, ast.Name)
    assert merge_base_arg.id != "stage_one"
