"""FIX-S2SCOPE: with T2PW_STAGE2_RESPECT_SCOPE on, a reaction Stage 2 adds is not merged when it repeats a
reaction Stage 1 marked out of scope (the enterobactin server job's LDH assay, ACCURACY.md section 15).

Offline: Stage 2, the draft graph and the preservation report are patched; ``merge_additions`` is real
where the test says so.
"""

from __future__ import annotations

import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import t2pw.pipeline.pipeline as pipeline  # noqa: E402
import t2pw.service.postpipeline as postpipeline  # noqa: E402
from t2pw.batch import driver  # noqa: E402
from t2pw.pipeline.cofactor_policy import PathwayContext  # noqa: E402
from t2pw.service import extraction as ex  # noqa: E402

CTX = {"pathway_name": "enterobactin biosynthesis", "likely_organism": "Escherichia coli"}
PCTX = PathwayContext("enterobactin biosynthesis", "Escherichia coli")


def _rxn(name, inputs, outputs, **kw):
    return {"name": name, "inputs": list(inputs), "outputs": list(outputs), **kw}


ENTA = _rxn("EntA reaction", ["2,3-dihydro-2,3-dihydroxybenzoate", "NAD+"], ["2,3-dihydroxybenzoic acid", "NADH"],
            scope_membership="core")
LDH_S1 = _rxn("LDH reaction", ["pyruvate", "NADH"], ["lactate", "NAD"], scope_membership="out_of_scope")
LDH_S2 = _rxn("LDH reaction", ["pyruvate", "NADH", "H+"], ["lactate", "NAD+"],
              enzymes=[{"protein": "LDH", "provenance": "inferred", "confidence": 0.95}])
ENTB_S2 = _rxn("EntB reaction", ["isochorismate"], ["2,3-dihydro-2,3-dihydroxybenzoate", "pyruvate"])


COMPOUNDS = ["2,3-dihydro-2,3-dihydroxybenzoate", "2,3-dihydroxybenzoic acid", "isochorismate", "pyruvate",
             "lactate", "NAD+", "NAD", "NADH", "H+"]


def _stage_one(*rows):
    return {"entities": {"compounds": [{"name": n} for n in COMPOUNDS]},
            "processes": {"reactions": [deepcopy(r) for r in rows]}}


def _additions(*rows):
    return {"additions": {"processes": {"reactions": [deepcopy(r) for r in rows]}}}


def _reactions(envelope):
    return envelope["additions"]["processes"]["reactions"]


# ------------------------------------------------------------------ the rule
def test_no_out_of_scope_rows_drops_nothing():
    s2 = _additions(LDH_S2, ENTB_S2)
    out, report = ex.drop_stage_two_scope_repeats(s2, _stage_one(ENTA), PCTX)
    assert out is s2 and report["drops"] == [] and report["stage1_out_of_scope_rows"] == 0


def test_the_lactate_assay_repeat_is_dropped_and_recorded():
    s2 = _additions(LDH_S2, ENTB_S2)
    out, report = ex.drop_stage_two_scope_repeats(s2, _stage_one(ENTA, LDH_S1), PCTX)
    assert [r["name"] for r in _reactions(out)] == ["EntB reaction"]
    (drop,) = report["drops"]
    # merge's own cleaning splits on "+", so "NAD+" reaches the merge as "NAD"
    assert drop["stage2_row"]["outputs"] == ["lactate", "NAD"] and drop["stage1_row"] == LDH_S1
    assert drop["stage1_index"] == 1 and "extractor" in drop["out_of_scope_reason"]
    assert drop["match"] == {"substrates": ["pyruvate"], "products": ["lactate"]}
    assert len(_reactions(s2)) == 2  # the model's own output is not changed


def test_a_real_step_sharing_only_nadh_nad_with_the_assay_is_kept():
    s2 = _additions(_rxn("EntA reaction", ["2,3-dihydro-2,3-dihydroxybenzoate", "NAD+"],
                         ["2,3-dihydroxybenzoate", "NADH"]),
                    _rxn("pyruvate reduction", ["pyruvate", "NADH"], ["malate", "NAD+"]))
    out, report = ex.drop_stage_two_scope_repeats(s2, _stage_one(LDH_S1), PCTX)
    assert out is s2 and report["drops"] == [] and report["stage1_out_of_scope_rows"] == 1


def test_a_step_stage_1_also_holds_in_scope_is_kept():
    s2 = _additions(LDH_S2)
    in_scope_twin = dict(LDH_S1, scope_membership="core")
    out, report = ex.drop_stage_two_scope_repeats(s2, _stage_one(LDH_S1, in_scope_twin), PCTX)
    assert out is s2 and report["drops"] == []


def test_the_reverse_direction_is_not_a_repeat():
    s2 = _additions(_rxn("lactate oxidation", ["lactate", "NAD+"], ["pyruvate", "NADH"]))
    out, report = ex.drop_stage_two_scope_repeats(s2, _stage_one(LDH_S1), PCTX)
    assert out is s2 and report["drops"] == []


ENTF_AMP = _rxn("EntF reaction", ["2,3-dihydroxybenzoyl-AMP", "L-serine"], ["enterobactin"])


def test_a_row_the_verifier_omitted_or_folded_is_not_a_scope_verdict():
    # legs4g vt3-PMC12096016: the verifier omitted the EntF step (gold R5) as a "summary"; Stage 2 restored it.
    s1 = _stage_one(dict(ENTF_AMP, scope_membership="out_of_scope"), LDH_S1)
    verifier = {"reactions": {"rows": [{"index": 0, "action": "omitted", "verdict": "summary"}]}}
    s2 = _additions(ENTF_AMP, LDH_S2)
    out, report = ex.drop_stage_two_scope_repeats(s2, s1, PCTX, verifier)
    assert [r["name"] for r in _reactions(out)] == ["EntF reaction"]
    assert [d["stage1_index"] for d in report["drops"]] == [1] and report["verifier_removed_rows_ignored"] == [0]
    verifier["reactions"]["rows"][0]["action"] = "folded_into_3"
    out, _ = ex.drop_stage_two_scope_repeats(s2, s1, PCTX, verifier)
    assert [r["name"] for r in _reactions(out)] == ["EntF reaction"]
    # without the verifier's record the same Stage-1 row is the extractor's verdict and is respected
    out, _ = ex.drop_stage_two_scope_repeats(s2, s1, PCTX)
    assert _reactions(out) == []


def test_an_error_in_the_check_keeps_every_stage_two_row(monkeypatch):
    def boom(*a, **k):
        raise ValueError("broken key")

    monkeypatch.setattr(ex, "step_signature", boom)
    s2 = _additions(LDH_S2, ENTB_S2)
    out, report = ex.drop_stage_two_scope_repeats(s2, _stage_one(ENTA, LDH_S1), PCTX)
    assert out is s2 and report["drops"] == [] and "broken key" in report["error"]


def test_the_real_merge_of_the_filtered_additions_has_no_lactate_row():
    base = _stage_one(ENTA)
    out, _ = ex.drop_stage_two_scope_repeats(_additions(LDH_S2, ENTB_S2), _stage_one(ENTA, LDH_S1), PCTX)
    merged = pipeline.merge_additions(base, out, pathway_context=PCTX)
    names = [r["name"] for r in merged["processes"]["reactions"]]
    assert "LDH reaction" not in names and "EntB reaction" in names


# ------------------------------------------------------- run_stage_two wiring
@pytest.fixture()
def world(monkeypatch):
    w = types.SimpleNamespace(merged=[], additions=_additions(LDH_S2, ENTB_S2))

    def stage_two(text, payload, **kw):
        return deepcopy(w.additions), [], []

    def merge(base, additions, **kw):
        w.merged.append(additions)
        return deepcopy(base)

    monkeypatch.setattr(pipeline, "run_stage_two_with_feedback_loop", stage_two)
    monkeypatch.setattr(pipeline, "merge_additions", merge)
    monkeypatch.setattr(pipeline, "build_and_save_draft_graph",
                        lambda p: (types.SimpleNamespace(to_dict=lambda: {}), {}, ""))
    monkeypatch.setattr(postpipeline, "_write_reaction_preservation_report", lambda stage, p: {})
    return w


class _Quiet:
    def __getattr__(self, name):
        class _C:
            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False
        return lambda *a, **k: _C()


def _run(state):
    s1 = ex.StageOneResult(stage_one=_stage_one(ENTA, LDH_S1), chunk_details=[], stage_one_in_scope=_stage_one(ENTA),
                           out_of_scope_removed_reactions=["LDH reaction"], rag_result=None, user_task_context="")
    return ex.run_stage_two("paper", s1, pathway_context=CTX, run_inference=True, enable_chunking=False,
                            chunk_size=1, chunk_overlap=0, infer_rounds=1, infer_attempts=1, temperature=0.0,
                            infer_tokens=1, rag_incomplete_flag=False, export_mode="strict", state=state,
                            reporter=_Quiet())


def test_flag_off_merges_exactly_what_stage_two_returned(world, monkeypatch):
    monkeypatch.delenv(ex.STAGE2_SCOPE_FLAG, raising=False)
    monkeypatch.setattr(ex, "drop_stage_two_scope_repeats", lambda *a, **k: pytest.fail("ran with the flag off"))
    state = {}
    _run(state)
    assert world.merged == [world.additions] and ex.STAGE2_SCOPE_DROPS_KEY not in state


def test_flag_on_drops_the_repeat_before_the_merge_and_keeps_stage_two_raw(world, monkeypatch):
    monkeypatch.setenv(ex.STAGE2_SCOPE_FLAG, "1")
    state = {}
    _run(state)
    assert [r["name"] for r in _reactions(world.merged[0])] == ["EntB reaction"]
    assert len(state[ex.STAGE2_SCOPE_DROPS_KEY]["drops"]) == 1
    assert state["stage_two"] == world.additions  # the model's output, unfiltered


def test_the_driver_writes_the_drop_report_only_when_present():
    report = {"flag": ex.STAGE2_SCOPE_FLAG, "drops": []}
    out = {}
    driver._add_common_artifacts(types.SimpleNamespace(session_state={ex.STAGE2_SCOPE_DROPS_KEY: report}), {}, out)
    assert json.loads(out["stage2_scope_drops.json"]) == report
    out = {}
    driver._add_common_artifacts(types.SimpleNamespace(session_state={}), {}, out)
    assert "stage2_scope_drops.json" not in out
