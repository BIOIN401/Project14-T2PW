"""ORCH-772b -- a leg with no PWML names its recorded cause; the coverage headline leaves KI-008 out (D-114 AMENDMENT 2).

**G9 LABELS.**

* ``test_g9_*`` are **REGRESSION** proofs: each fails behaviourally at base ``00eb12d3`` (the evidence text there says
  only that no PWML file is present, labels it ``unclassified``, and the headline counts the KI-008 interactions) and
  passes at the tip. Measured on an export of the base tree with this file added (``ORCH-772b-RESULT.md`` § 5).
* ``test_new_*`` are **NEW ACCEPTANCE TESTS** for ``evidence/orch772b_no_pwml_cause.py``, which does not exist at base.

The leg used is ORCH-767 enterobactin (no PWML: the final pre-export gate refused the invented wrapper, KI-012), whose
inputs are committed gzipped under ``tests/fixtures/orch768/`` and are byte-identical to the corpus manifest.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from types import SimpleNamespace

from t2pw.app.reaction_evidence import build_report, known_issue_id, render_text

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "docs" / "pwml_recovery_sprint" / "evidence"
FIXTURE = ROOT / "tests" / "fixtures" / "orch768"
LEG_ID = "ORCH-767/enterobactin-strict/2026-10-23_2224"
CAUSE = ("Generated protein complex 'enterobactin synthase complex' component 'enterobactin synthase' is missing "
         "positive stoichiometry.")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "") + "_772b", EV / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_leg():
    return next(leg for leg in json.loads((FIXTURE / "legs.json").read_text(encoding="utf-8"))["legs"] if leg["id"] == LEG_ID)


@pytest.fixture()
def leg_dir(tmp_path: Path) -> Path:
    """The leg's own files, laid out as the batch driver writes a leg (``<out>/<stamp>/papers/<paper>/strict``)."""

    leg = tmp_path / "legout" / "2026-10-23_2224" / "papers" / "PMC12096016" / "strict"
    leg.mkdir(parents=True)
    for name, meta in _fixture_leg()["inputs"].items():
        if meta["path"]:
            (leg / name).write_bytes(gzip.decompress((FIXTURE / meta["path"]).read_bytes()))
    return leg


# G9 regression proofs ------------------------------------------------------------------------------------------
def test_g9_the_review_pack_writes_the_legs_recorded_cause_and_its_known_issue(leg_dir, tmp_path):
    review = tmp_path / "review"
    done = subprocess.run(
        [sys.executable, str(EV / "orch756_review_pack.py"), "--task", "T", "--leg-out", str(tmp_path / "legout"),
         "--cond", "enterobactin", "--mode", "strict", "--started", "0", "--finished", "1", "--exit", "1",
         "--review", str(review)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    text = (review / "enterobactin" / "strict" / "reaction_evidence.txt").read_text(encoding="utf-8")
    assert f"\n  No PWML written: {CAUSE}\n" in text
    assert f"\nNo PWML written: {CAUSE} (known issue: KI-012). Nothing is drawn;" in text
    assert "neither pathway.pwml nor" not in text


def test_g9_the_morning_summary_names_the_known_issue_of_a_leg_with_no_pwml(leg_dir):
    summary = _load("orch770_morning_summary.py")
    assert summary.coverage(leg_dir) == "0/21 (0%); full count 0/21 (0%); no PWML: KI-012"


def test_g9_the_night_runner_measures_coverage_with_the_pinned_worktrees_code():
    night = (EV / "orch770_night.sh").read_text(encoding="utf-8")
    call = night[night.index('"$EV/orch770_morning_summary.py"'):]
    assert '--src "$WT/src"' in call[: call.index("morning_summary.log")]


def test_g9_the_corpus_replay_records_its_own_cause_for_each_process_lost_with_the_leg(leg_dir):
    """Round 1 (merge rule 4): after ORCH-773 the fixture leg writes a PWML at this tree, so the replay path is driven
    with the stop the leg recorded at 00eb12d3 (its final-gate error) and the leg's own canonical payload."""

    H = _load("orch768_replay_corpus.py")
    out = {"failure_stage": "final_pre_export_stage3_gates",
           "final_gate": {"errors": [["/entities/protein_complexes/0/components/0", CAUSE]]}}
    assert H.replay_no_pwml_cause(out) == CAUSE
    import t2pw.app.reaction_evidence as evidence

    payload = json.loads((leg_dir / "final_mapped.json").read_text(encoding="utf-8"))
    cov = H.pwml_coverage(SimpleNamespace(evidence=evidence), {}, payload, None, None,
                          no_pwml_cause=H.replay_no_pwml_cause(out))
    lost = [row for row in cov["omitted_by_reason"] if row[2] == "no_pwml"]
    assert [row[:3] for row in lost] == [[17, "KI-012", "no_pwml"]]
    assert lost[0][3] == f"no PWML written: {CAUSE}"[:120]
    assert (cov["drawn"], cov["total"]) == (0, 17)  # no quarantine set passed here: the payload's 17 processes


def test_g9_the_headline_leaves_out_the_interactions_pathwhiz_cannot_carry():
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}]},
               "processes": {"reactions": [{"name": "R1", "inputs": ["A"], "outputs": ["B"]}],
                             "interactions": [{"name": "A competes with B", "entity_1": "A", "entity_2": "B",
                                               "relationship": "competes with"}]}}
    ir = {"processes": {"reactions": [{"name": "R1"}], "transports": [], "interactions": [
        {"name": "A competes with B", "export_status": "withheld", "interaction_type": "competes with",
         "withheld": {"reason": "untyped_interaction", "raw_interaction_type": "competes with", "left": "A", "right": "B"}}]}}
    lines = render_text(build_report({"final_mapped.json": payload, "pwml_ir.json": ir, "pathway.pwml": True})).splitlines()
    assert lines[2].startswith("PWML coverage: 1 of 1 processes (100.0%) -- target 75% (D-114 AMENDMENT 2")
    assert lines[3] == "  Full count, the 1 KI-008 interactions included: 1 of 2 (50.0%)"


# new acceptance tests: the leg-directory reader ----------------------------------------------------------------
def _write(leg: Path, **files) -> Path:
    leg.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        name = name.replace("__", ".")
        (leg / name).write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return leg


RESULT = ("RESULT: FAIL\n\nWHAT WENT WRONG\n----\nfailure_kind : contract\nmessage      : {message}\n"
          "issue codes  : {codes}\n")


def test_new_the_reader_takes_the_first_recorded_stop_in_pipeline_order(tmp_path):
    cause = _load("orch772b_no_pwml_cause.py").recorded_no_pwml_cause
    refused = _write(tmp_path / "a", quarantine_report__json={"ok": False, "refusal_reasons": ["unexportable_entity:1"]},
                     final_stage3_gate_report__json={"ok": False, "errors": [{"reason": "never read"}]})
    assert cause(refused) == "unexportable_entity:1"
    gate = _write(tmp_path / "b", quarantine_report__json={"ok": True},
                  final_stage3_gate_report__json={"ok": False, "errors": [
                      {"path": "/x", "reason": "Unknown protein/modifier reference: PIF1"},
                      {"path": "/y", "reason": "Unknown protein/modifier reference: GBP"},
                      {"path": "/z", "reason": "Unknown protein/modifier reference: GBP"}]},
                  RESULT__txt=RESULT.format(message="m", codes="gate.unknown_protein_modifier_reference"))
    assert cause(gate) == "Unknown protein/modifier reference: PIF1; Unknown protein/modifier reference: GBP"
    export = _write(tmp_path / "c", quarantine_report__json={"ok": True}, final_stage3_gate_report__json={"ok": True},
                    RESULT__txt=RESULT.format(message="PWML export failed: PWML required-field gate failed.",
                                              codes="reaction_enzyme_must_be_protein_complex"))
    assert cause(export) == "reaction_enzyme_must_be_protein_complex"
    crash = _write(tmp_path / "d", RESULT__txt=RESULT.format(message="the app never rendered the button", codes="(none reported)"))
    assert cause(crash) == "the app never rendered the button"
    assert cause(_write(tmp_path / "e")) == ""  # nothing recorded: the module then says no PWML file is present


def test_new_the_reader_on_the_committed_leg_returns_the_final_gates_own_message(leg_dir):
    assert _load("orch772b_no_pwml_cause.py").recorded_no_pwml_cause(leg_dir) == CAUSE


# review round 1 (r0 B-1, N-1): G9 regression proofs, each fails behaviourally at 30e4c675 ------------------------
def test_g9_r1_an_ir_validation_failure_is_read_from_its_report_not_labelled_a_time_limit(tmp_path):
    """The shape of runs_verify/2026-08-25_1216/papers/PMC12856317/strict: no issue codes, IR validation ok=false."""

    cause = _load("orch772b_no_pwml_cause.py").recorded_no_pwml_cause
    leg = _write(tmp_path / "leg", quarantine_report__json={"ok": True}, final_stage3_gate_report__json={"ok": True},
                 pwml_ir_validation__json={"ok": False, "errors": [
                     {"code": "protein_complex_missing_components", "message": "Protein complex 'LONP1' has no protein components.",
                      "pointer": "/entities/protein_complexes/1/components"}]},
                 RESULT__txt=RESULT.format(message="PWML export failed: PWML IR validation failed.", codes="(none reported)"))
    assert cause(leg) == "protein_complex_missing_components"
    assert known_issue_id(cause(leg)) == "KI-015"
    # The bare messages are no longer read as KI-028's time-limit message, which only the driver's own wording names.
    assert known_issue_id("PWML export failed: PWML IR validation failed.") == "unclassified"
    assert known_issue_id("PWML export failed: PWML required-field gate failed.") == "unclassified"
    assert known_issue_id("PWML export did not finish inside the time budget") == "KI-028"


def test_g9_r1_a_cause_listing_several_exact_codes_is_labelled_by_the_first_that_maps():
    assert known_issue_id("species_missing_taxonomy, species_missing_classification") == "KI-019"
    assert known_issue_id("pathway_missing_pw_id; species_missing_classification") == "KI-019"
    assert known_issue_id("no_such_code, another_one") == "unclassified"
    # A whole match still wins: a message with commas is labelled whole, never by a piece.
    assert known_issue_id("LLM returned an empty completion for stage1 (model m, finish_reason=length, tokens 0)") == "KI-054"
