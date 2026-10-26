"""ORCH-768 (D-112 § 1) -- NEW CAPABILITY, labelled acceptance test (G9: a new harness, no base failure claimed).

``docs/pwml_recovery_sprint/evidence/orch768_replay_corpus.py`` replays a saved leg offline, from its last
checkpoint that needs no LLM call, through the production tail (``freeze_canonical_payload``, the batch
driver's gate verdict, ``initialize_refinement_review_state``, ``run_pwml_export``). This test pins it on three
ORCH-767 legs whose inputs are committed gzipped under ``tests/fixtures/orch768/`` and are byte-identical to the
corpus manifest's sha256:

1. **fidelity** -- the legs ran on ``d6849c9e``, whose ``src`` tree is ``main``'s (``e3d5b65f``), so at this tree
   the replay must reproduce each recorded verdict: sour taste writes ``pathway.review_required.pwml``
   byte-identical to the leg's; enterobactin fails the final pre-export gate on the "enterobactin synthase
   complex" wrapper component (since ORCH-773 the wrapper is omitted and the leg writes a PWML); WNT PCP
   crashes opening refinement review (no reaction nodes to draw; since ORCH-769c it writes a PWML);
2. **determinism** -- two replays give identical records and identical output files;
3. **offline** -- every socket connect and DNS lookup is refused while the harness runs; the probe proves the
   block and the legs attempt none.

A lane that changes one of these verdicts ON PURPOSE (ORCH-769) moves the pin in ``EXPECTED`` with its documented
delta (merge rule 4). It never relaxes the fidelity comparison itself.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "orch768"
HARNESS = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "orch768_replay_corpus.py"
MANIFEST = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "orch768_corpus_manifest.json"

SOUR = "ORCH-767/sour_taste_v2-strict/2026-10-24_0042"
ENTEROBACTIN = "ORCH-767/enterobactin-strict/2026-10-23_2224"
WNT_PCP = "ORCH-767/wnt_pcp-strict/2026-10-24_0131"
EXPECTED = {
    SOUR: {"pwml_written": True, "failure_stage": None, "pwml_matches_leg": True},
    # ORCH-773 (merge rule 4): the inferred wrapper is omitted before the freeze (D-113 s1, D-114 s2).
    ENTEROBACTIN: {"pwml_written": True, "failure_stage": None, "canonical_matches_leg": False},
    # ORCH-769c (merge rule 4): the review step no longer crashes on an interactions-only graph.
    WNT_PCP: {"pwml_written": True, "failure_stage": None},
}


def _load_harness():
    spec = importlib.util.spec_from_file_location("orch768_replay_corpus", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    H = _load_harness()
    legs = json.loads((FIXTURE / "legs.json").read_text(encoding="utf-8"))["legs"]
    inputs = tmp_path_factory.mktemp("orch768_inputs")
    for leg in legs:
        for meta in leg["inputs"].values():
            if meta["path"]:
                raw = meta["path"][: -len(".gz")]
                (inputs / raw).parent.mkdir(parents=True, exist_ok=True)
                (inputs / raw).write_bytes(gzip.decompress((FIXTURE / meta["path"]).read_bytes()))
                meta["path"] = raw
    try:
        T = H.load_tree(None)
        probe = H.probe_offline()
        runs = []
        for attempt in ("a", "b"):
            out = tmp_path_factory.mktemp(f"orch768_replay_{attempt}")
            runs.append({leg["id"]: H.replay_leg(T, leg, out / leg["id"].replace("/", "_"), primary=inputs)
                         for leg in legs})
    finally:
        H.Offline.uninstall()
    return {"H": H, "T": T, "legs": {leg["id"]: leg for leg in legs}, "inputs": inputs, "probe": probe, "runs": runs}


def test_fixture_is_the_corpus_input_byte_for_byte(corpus):
    manifest = {leg["id"]: leg for leg in json.loads(MANIFEST.read_text(encoding="utf-8"))["legs"]}
    assert set(corpus["legs"]) == set(EXPECTED)
    for leg_id, leg in corpus["legs"].items():
        for name, meta in leg["inputs"].items():
            assert manifest[leg_id]["inputs"][name]["sha256"] == meta["sha256"], (leg_id, name)
            if meta["path"]:
                digest = hashlib.sha256((corpus["inputs"] / meta["path"]).read_bytes()).hexdigest()
                assert digest == meta["sha256"], (leg_id, name)


def test_measured_tree_is_this_checkout(corpus):
    assert corpus["T"].t2pw == (ROOT / "src" / "t2pw").resolve()


def test_network_is_refused_and_counted_and_the_legs_attempt_none(corpus):
    probe = corpus["probe"]
    assert probe["ok"], probe
    assert probe["counted"] == 3 and set(probe["attempts"].values()) == {"refused"}
    for run in corpus["runs"]:
        for leg_id, record in run.items():
            assert record["network_refused"] == [], leg_id


@pytest.mark.parametrize("leg_id", sorted(EXPECTED))
def test_replay_reproduces_the_recorded_verdict(corpus, leg_id):
    H = corpus["H"]
    record = corpus["runs"][0][leg_id]
    assert record["checkpoint"] == "canonical"
    assert record["canonical_matches_leg"] is EXPECTED[leg_id].get("canonical_matches_leg", True)
    # ORCH-769c moved WNT PCP on purpose: its one fidelity difference is the outcome.
    # ORCH-773 moved enterobactin on purpose: the outcome and the final gate's verdict.
    expected_diffs = {
        WNT_PCP: ["outcome recorded=refinement_review_crash replay=pwml"],
        ENTEROBACTIN: [
            "outcome recorded=final_pre_export_stage3_gates replay=pwml",
            "final_gate_ok recorded=False replay=True",
            f"final_gate_errors recorded={record['recorded_reports']['final_gate_errors']} replay=[]",
        ],
    }.get(leg_id, [])
    assert H.fidelity(corpus["legs"][leg_id], record) == expected_diffs
    for key, value in EXPECTED[leg_id].items():
        assert record.get(key) == value, (leg_id, key, record.get(key))


def test_enterobactin_wrapper_component_recorded_failure_is_omitted_not_refused(corpus):
    """The leg's own final gate failed on the wrapper component; the replay omits the wrapper (ORCH-773)."""

    record = corpus["runs"][0][ENTEROBACTIN]
    (path, reason), = record["recorded_reports"]["final_gate_errors"]
    assert "enterobactin synthase complex" in reason.lower()
    assert record["final_gate"] == {"ok": True, "codes": [], "errors": []}
    assert record["quarantine"]["release"] == "review_required"


def test_the_fidelity_check_bites(corpus):
    """Non-vacuity: a recorded verdict the replay did not produce is reported, on each channel."""

    H = corpus["H"]
    sour = json.loads(json.dumps(corpus["legs"][SOUR]))
    sour["recorded"]["pwml"] = []
    sour["recorded"]["message"] = "post-pipeline validation failed: 1 blocking issue(s) at final_pre_export_stage3_gates"
    sour["recorded"]["stage"] = "post_pipeline"
    assert any(d.startswith("outcome ") for d in H.fidelity(sour, corpus["runs"][0][SOUR]))
    # ORCH-773: enterobactin's final gate now differs from its record by design, so the bite is shown on sour.
    sour_run = json.loads(json.dumps(corpus["runs"][0][SOUR]))
    assert not any(d.startswith("final_gate_errors ") for d in H.fidelity(corpus["legs"][SOUR], sour_run))
    sour_run["final_gate"]["errors"] = [["/entities/proteins/0", "injected"]]
    assert any(d.startswith("final_gate_errors ") for d in H.fidelity(corpus["legs"][SOUR], sour_run))
    pcp = json.loads(json.dumps(corpus["runs"][0][WNT_PCP]))
    pcp["quarantine"]["release"] = "release_ready"
    assert any(d.startswith("release ") for d in H.fidelity(corpus["legs"][WNT_PCP], pcp))


def test_two_replays_are_identical(corpus):
    first, second = corpus["runs"]
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(second, sort_keys=True, default=str)
    assert first[SOUR]["files"]["pathway.review_required.pwml"]
