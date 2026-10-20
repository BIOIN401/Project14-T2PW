"""NEW ACCEPTANCE TESTS for Stage-1 N-draw ensembling (ORCH-750 (dispatched as ORCH-750; renumbered because another session had already claimed 748 in the ledger for the pathway-mining lane)).

G9 labelling: ensembling is a NEW capability (``t2pw.pipeline.stage_one_ensemble`` and the
``draws`` parameter of ``run_extraction_pipeline``), so these are explicitly new acceptance
tests and no base-commit failure is claimed for them. The one pre-existing behaviour the
change must PRESERVE -- N=1 is a single, unchanged ``_run_json_stage`` call -- is pinned
here (`test_default_is_one_unchanged_call`) and proved against the committed base by
``docs/pwml_recovery_sprint/evidence/orch750_n1_base_identity_proof.py``.

No model is called anywhere here: ``_run_json_stage`` is replaced by a recorder that hands
back canned payloads, one per draw.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import t2pw.pipeline.pipeline as pipeline_mod  # noqa: E402
from t2pw.paths import PROMPTS_DIR  # noqa: E402
from t2pw.pipeline import extraction_diagnostics as diag  # noqa: E402
from t2pw.pipeline.pipeline import PipelineFailure, _build_extraction_prompt, run_extraction_pipeline  # noqa: E402
from t2pw.pipeline.stage_one_ensemble import (  # noqa: E402
    BOUNDARY_STAGE1_ENSEMBLE,
    ENV_VAR,
    REPORT_NAME,
    extraction_draws,
    merge_draws,
    row_key,
)

TEXT = "Glucose is phosphorylated by hexokinase to glucose 6-phosphate using ATP."
CONTEXT = {"name": "glycolysis", "organism": "Homo sapiens"}

# Three draws of the same paper. Each misses something the others found (PAPER_FINDINGS
# § 5.3), and draw 2 restates draw 1's first reaction under a different name, a different
# participant order and the evaluator-equivalent spellings NAD+/NAD, pyrophosphate/diphosphate.
DRAW_1: Dict[str, Any] = {
    "pathway_context": {"name": "glycolysis", "organism": ""},
    "entities": {
        "compounds": [{"name": "glucose", "class": "compound"}, {"name": "NAD+", "class": "compound"}],
        "proteins": [{"name": "hexokinase", "class": "protein"}],
    },
    "processes": {
        "reactions": [
            {"name": "glucose phosphorylation", "inputs": ["glucose", "ATP"],
             "outputs": ["glucose 6-phosphate", "ADP"], "evidence": "E1 verbatim"},
            {"name": "GAPDH step", "inputs": ["G3P", "NAD+", "Pi"], "outputs": ["1,3-BPG", "NADH"],
             "evidence": "E2 verbatim"},
        ],
        "interactions": [
            {"name": "i1", "entity_1": "ATP", "relationship": "inhibition", "entity_2": "hexokinase",
             "evidence": "I1 verbatim"},
        ],
    },
}
DRAW_2: Dict[str, Any] = {
    "pathway_context": {"name": "glycolysis", "organism": "Homo sapiens"},
    "entities": {
        "compounds": [{"name": "NAD", "class": "compound"}, {"name": "pyruvate", "class": "compound"}],
        "proteins": [{"name": "Hexokinase", "class": "protein"}],
    },
    "processes": {
        "reactions": [
            {"name": "hexokinase reaction", "inputs": ["ATP", "glucose"],
             "outputs": ["ADP", "glucose 6-phosphate"], "evidence": "E1 RESTATED - must not survive"},
            {"name": "pyruvate kinase", "inputs": ["PEP", "ADP"], "outputs": ["pyruvate", "ATP"],
             "evidence": "E3 verbatim"},
        ],
        "interactions": [
            {"name": "same claim", "entity_1": "atp", "relationship": "Inhibition", "entity_2": "Hexokinase",
             "evidence": "I1 RESTATED - must not survive"},
            {"name": "i2", "entity_1": "citrate", "relationship": "inhibition", "entity_2": "PFK-1",
             "evidence": "I2 verbatim"},
        ],
        "transports": [
            {"name": "t1", "cargo": "glucose", "from_biological_state": "extracellular",
             "to_biological_state": "cytosol", "evidence": "T1 verbatim"},
        ],
    },
}
DRAW_3: Dict[str, Any] = {
    "entities": {
        "compounds": [{"name": "isopentenyl pyrophosphate", "class": "compound"}],
    },
    "processes": {
        "reactions": [
            {"name": "IPP isomerase", "inputs": ["isopentenyl pyrophosphate"],
             "outputs": ["dimethylallyl diphosphate"], "evidence": "E4 verbatim"},
            {"name": "IPP isomerase again", "inputs": ["isopentenyl diphosphate"],
             "outputs": ["dimethylallyl pyrophosphate"], "evidence": "E4 RESTATED - must not survive"},
        ],
    },
}


class Recorder:
    """Stands in for ``_run_json_stage``: records every call, answers from a script."""

    def __init__(self, script: List[Any]) -> None:
        self.script = list(script)
        self.calls: List[Dict[str, Any]] = []
        self.returned: List[Any] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.script[len(self.calls) - 1]
        if isinstance(item, BaseException):
            raise item
        result = (deepcopy(item), [{"attempt": 1, "raw": json.dumps(item)}])
        self.returned.append(result)
        return result


def _reactions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return payload["processes"]["reactions"]


def _key_paths(value: Any, path: str = "") -> Set[str]:
    """Every dict key path in a payload, list items collapsed to their parent path."""

    out: Set[str] = set()
    if isinstance(value, dict):
        for k, v in value.items():
            out.add(f"{path}/{k}")
            out |= _key_paths(v, f"{path}/{k}")
    elif isinstance(value, list):
        for item in value:
            out |= _key_paths(item, path)
    return out


@pytest.fixture
def no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    diag.deactivate()


# --------------------------------------------------------------------------- N = 1


def test_default_is_one_unchanged_call(no_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Env unset: exactly one ``_run_json_stage`` call, the historical keyword set, the
    historical values, and the stage's return value handed back untouched (same objects)."""

    rec = Recorder([DRAW_1])
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)

    result = run_extraction_pipeline(
        TEXT, pathway_context=CONTEXT, pathway_scope="glycolysis", user_task_context="ctx",
        max_attempts=2, temperature=0.0, max_tokens=12000,
    )

    assert len(rec.calls) == 1
    assert result is rec.returned[0]
    call = rec.calls[0]
    assert set(call) == {
        "stage_name", "model_env_var", "system_prompt", "build_user_prompt",
        "max_attempts", "temperature", "max_tokens", "retry_on_empty_payload",
    }
    assert call["stage_name"] == "extraction"
    assert call["model_env_var"] == "OPENROUTER_EXTRACTION_MODEL"
    assert call["system_prompt"] == (PROMPTS_DIR / "pwml_system.txt").read_text(encoding="utf-8")
    assert (call["max_attempts"], call["temperature"], call["max_tokens"]) == (2, 0.0, 12000)
    assert call["retry_on_empty_payload"] is True
    for prev, err in ((None, None), ("{}", "previous error")):
        assert call["build_user_prompt"](prev, err) == _build_extraction_prompt(
            TEXT, prev, err, pathway_context=CONTEXT, pathway_scope="glycolysis", user_task_context="ctx")
    # Nothing of the ensemble is recorded at N=1.
    assert not [b for b in diag.current().boundaries if b["boundary"] == BOUNDARY_STAGE1_ENSEMBLE]


def test_env_value_one_is_the_same_single_call(no_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "1")
    rec = Recorder([DRAW_1])
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)
    result = run_extraction_pipeline(TEXT)
    assert len(rec.calls) == 1 and result is rec.returned[0]


def test_explicit_parameter_overrides_env(no_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "3")
    rec = Recorder([DRAW_1, DRAW_2, DRAW_3])
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)
    result = run_extraction_pipeline(TEXT, draws=1)
    assert len(rec.calls) == 1 and result is rec.returned[0]

    monkeypatch.delenv(ENV_VAR)
    rec = Recorder([DRAW_1, DRAW_2])
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)
    run_extraction_pipeline(TEXT, draws=2)
    assert len(rec.calls) == 2


@pytest.mark.parametrize("bad", ["0", "-2", "three", "1.5"])
def test_invalid_draw_count_raises_before_any_call(no_env: None, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv(ENV_VAR, bad)
    rec = Recorder([DRAW_1])
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)
    with pytest.raises(ValueError, match=ENV_VAR):
        run_extraction_pipeline(TEXT)
    assert rec.calls == []


def test_blank_env_means_default(no_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "  ")
    assert extraction_draws() == 1
    assert extraction_draws(4) == 4


# --------------------------------------------------------------------------- N = 3


def test_three_draws_union_with_dedup_first_occurrence_wins(no_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "3")
    rec = Recorder([DRAW_1, DRAW_2, DRAW_3])
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)

    merged, attempts = run_extraction_pipeline(TEXT)

    assert len(rec.calls) == 3
    # Reactions: 6 rows drawn, 4 distinct under the evaluator's input/output key sets.
    names = [r["name"] for r in _reactions(merged)]
    assert names == ["glucose phosphorylation", "GAPDH step", "pyruvate kinase", "IPP isomerase"]
    evidence = [r["evidence"] for r in _reactions(merged)]
    assert evidence == ["E1 verbatim", "E2 verbatim", "E3 verbatim", "E4 verbatim"]
    assert not any("RESTATED" in e for e in evidence)
    # Interactions: (entity_1, relationship, entity_2) normalised; the restatement collapses.
    inter = merged["processes"]["interactions"]
    assert [i["evidence"] for i in inter] == ["I1 verbatim", "I2 verbatim"]
    # Transports came only from draw 2 and are kept.
    assert [t["evidence"] for t in merged["processes"]["transports"]] == ["T1 verbatim"]
    # Entity registries: NAD+/NAD and hexokinase/Hexokinase collapse, first spelling kept.
    assert [c["name"] for c in merged["entities"]["compounds"]] == [
        "glucose", "NAD+", "pyruvate", "isopentenyl pyrophosphate"]
    assert [p["name"] for p in merged["entities"]["proteins"]] == ["hexokinase"]
    # First non-empty scalar wins: draw 1 left organism blank, draw 2 filled it.
    assert merged["pathway_context"] == {"name": "glycolysis", "organism": "Homo sapiens"}
    # Every attempt log survives, tagged with the draw that produced it.
    assert [a["draw"] for a in attempts] == [1, 2, 3]
    assert all("raw" in a for a in attempts)


def test_merge_adds_no_key_the_draws_did_not_carry() -> None:
    merged, _report = merge_draws([DRAW_1, DRAW_2, DRAW_3])
    assert _key_paths(merged) <= _key_paths(DRAW_1) | _key_paths(DRAW_2) | _key_paths(DRAW_3)
    # and every row object is a draw's own row, verbatim
    for row in _reactions(merged):
        assert any(row in _reactions(d) for d in (DRAW_1, DRAW_2, DRAW_3))


def test_single_payload_merge_is_identity() -> None:
    merged, report = merge_draws([DRAW_2])
    assert merged == DRAW_2
    assert report["draws_requested"] == report["draws_succeeded"] == 1


def test_row_keys_match_the_evaluator() -> None:
    k = lambda row: row_key("/processes/reactions", row)  # noqa: E731
    assert k({"inputs": ["NAD+", "G3P"], "outputs": ["NADH"]}) == k({"inputs": ["g3p", "nad"], "outputs": ["nadh"]})
    assert k({"inputs": ["isopentenyl pyrophosphate"], "outputs": ["x"]}) == k(
        {"inputs": ["isopentenyl diphosphate"], "outputs": ["x"]})
    assert k({"inputs": ["2,3-dihydroxybenzoic acid"], "outputs": ["x"]}) == k(
        {"inputs": ["2,3-dihydroxybenzoate"], "outputs": ["x"]})
    assert k({"inputs": ["MK-7"], "outputs": ["x"]}) != k({"inputs": ["MK-8"], "outputs": ["x"]})
    # Participant dicts read like the production participant reader.
    assert k({"inputs": [{"entity": "ATP"}], "outputs": [{"compound": "ADP"}]}) == k(
        {"inputs": ["atp"], "outputs": ["adp"]})
    # Two unkeyable rows never collapse into one.
    assert k({"name": "a", "inputs": [], "outputs": []}) != k({"name": "b", "inputs": [], "outputs": []})
    assert row_key("/entities/compounds", {"name": "NAD+"}) == row_key("/entities/compounds", {"name": "nad"})
    assert row_key("/element_locations/compound_locations",
                   {"compound": "ATP", "biological_state": "cytosol"}) == row_key(
        "/element_locations/compound_locations", {"compound": "atp", "biological_state": "Cytosol"})


# --------------------------------------------------------------------------- failures


def test_a_failed_draw_is_survived_and_reported(no_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_VAR, "3")
    failure = PipelineFailure("extraction", "draw 2 produced nothing", [{"attempt": 1, "raw": "", "error": "empty"}])
    rec = Recorder([DRAW_1, failure, DRAW_3])
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)
    recorder = diag.activate(run_id="orch750", artifact_dir=tmp_path)

    merged, attempts = run_extraction_pipeline(TEXT)

    assert len(rec.calls) == 3
    assert [r["evidence"] for r in _reactions(merged)] == ["E1 verbatim", "E2 verbatim", "E4 verbatim"]
    assert [a["draw"] for a in attempts] == [1, 2, 3]  # the failed draw's attempt log is kept too

    [record] = [b for b in recorder.boundaries if b["boundary"] == BOUNDARY_STAGE1_ENSEMBLE]
    summary = record["ensemble"]
    assert summary["draws_requested"] == 3
    assert summary["draws_succeeded"] == 2
    assert summary["draws_failed"] == 1
    assert summary["failed_draws"][0]["draw"] == 2
    assert summary["failed_draws"][0]["stage"] == "extraction"
    # processes + entities rows: draw 1 = 2 reactions + 1 interaction + 3 entities = 6;
    # draw 3 = 2 reactions + 1 entity = 3; of which 1 reaction collapses onto itself.
    assert summary["rows_per_draw"] == [6, 0, 3]
    assert summary["first_seen_in_draw"] == [6, 0, 2]
    assert summary["merged_rows"] == 8
    assert summary["unique_to_one_draw"] == 8
    assert summary["in_every_successful_draw"] == 0

    side = tmp_path / REPORT_NAME
    assert summary["report_path"] == str(side)
    [report] = json.loads(side.read_text(encoding="utf-8"))
    rows = report["buckets"]["/processes/reactions"]["rows"]
    assert [(r["name"], r["draws"]) for r in rows] == [
        ("glucose phosphorylation", [1]), ("GAPDH step", [1]), ("IPP isomerase", [3])]
    assert report["failed_draws"][0]["error"] == "draw 2 produced nothing"
    # The boundary report on disk carries the bounded summary too.
    on_disk = json.loads((tmp_path / diag.BOUNDARY_REPORT_NAME).read_text(encoding="utf-8"))
    assert any(b.get("boundary") == BOUNDARY_STAGE1_ENSEMBLE for b in on_disk["boundaries"])


def test_every_draw_failing_raises_the_last_failure(no_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "3")
    failures = [PipelineFailure("extraction", f"draw {i} failed", []) for i in (1, 2, 3)]
    rec = Recorder(failures)
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)
    with pytest.raises(PipelineFailure) as excinfo:
        run_extraction_pipeline(TEXT)
    assert excinfo.value is failures[-1]
    assert len(rec.calls) == 3


def test_an_unrelated_exception_propagates_immediately(no_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "3")
    rec = Recorder([RuntimeError("leg deadline"), DRAW_2, DRAW_3])
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)
    with pytest.raises(RuntimeError, match="leg deadline"):
        run_extraction_pipeline(TEXT)
    assert len(rec.calls) == 1


def test_ambiguous_review_still_rejected_before_any_draw(no_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "3")
    rec = Recorder([DRAW_1, DRAW_2, DRAW_3])
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)
    with pytest.raises(PipelineFailure) as excinfo:
        run_extraction_pipeline(TEXT, pathway_context={
            "document_type": "multi_example_review", "selected_example": "", "candidate_examples": []})
    assert excinfo.value.stage == "ambiguous_review_scope"
    assert rec.calls == []


def test_side_file_accumulates_one_entry_per_ensemble(no_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The chunked path calls run_extraction_pipeline once per chunk; each gets its own entry."""

    monkeypatch.setenv(ENV_VAR, "2")
    rec = Recorder([DRAW_1, DRAW_2, DRAW_2, DRAW_3])
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)
    diag.activate(run_id="orch750", artifact_dir=tmp_path)
    run_extraction_pipeline(TEXT)
    run_extraction_pipeline(TEXT)
    entries = json.loads((tmp_path / REPORT_NAME).read_text(encoding="utf-8"))
    assert len(entries) == 2
    assert [e["draws_succeeded"] for e in entries] == [2, 2]


# --------------------------------------------------------------------------- ORCH-752: cofactor-blind key
# G9: a CORRECTION of ORCH-750's delivered behaviour, so these fail on the pre-change module
# (certified run ``evidence/g11/ORCH-752/01-focused-pre.json``) and pass at the tip. The
# module is untracked at base ``c9265fb8``, so the failure is proved on the file as delivered.


def _rk(row: Dict[str, Any]) -> Any:
    return row_key("/processes/reactions", row)


def test_cofactors_do_not_define_a_reaction() -> None:
    """ORCH-751 § 2.2: ``A + NADPH -> B + NADP+`` and ``A -> B`` are one reaction."""

    assert _rk({"inputs": ["A", "NADPH"], "outputs": ["B", "NADP+"]}) == _rk({"inputs": ["A"], "outputs": ["B"]})
    assert _rk({"inputs": ["Proto IX", "Mg2+", "ATP"], "outputs": ["Mg-Proto IX", "ADP", "Pi"]}) == _rk(
        {"inputs": ["Proto IX"], "outputs": ["Mg-Proto IX"]})
    # A participant the payload itself marks as a cofactor is one, whatever its name.
    assert _rk({"inputs": ["A", {"entity": "ferredoxin", "role": "cofactor"}], "outputs": ["B"]}) == _rk(
        {"inputs": ["A"], "outputs": ["B"]})


def test_a_defining_participant_always_separates() -> None:
    assert _rk({"inputs": ["A"], "outputs": ["B"]}) != _rk({"inputs": ["A", "C"], "outputs": ["B"]})
    assert _rk({"inputs": ["A", "NADPH"], "outputs": ["B"]}) != _rk({"inputs": ["A", "NADPH"], "outputs": ["B2"]})
    # Currency IS the chemistry when nothing else is on a side: that side keeps every name.
    assert _rk({"inputs": ["ADP", "Pi"], "outputs": ["ATP"]}) != _rk({"inputs": ["GDP", "Pi"], "outputs": ["GTP"]})
    assert _rk({"inputs": ["ADP", "Pi"], "outputs": ["ATP"]}) == _rk({"inputs": ["Pi", "ADP"], "outputs": ["ATP"]})


def test_collapsed_duplicate_donates_its_cofactors_to_the_survivor() -> None:
    # ORCH-754a finding 1: a cofactor is donated only when the SURVIVOR's own quote names it,
    # so this survivor's quote now names all three (see the withheld case below).
    first = {"name": "r", "inputs": ["A"], "outputs": ["B"], "evidence": "A is reduced by NADPH to B, giving NADP+ and H2O"}
    later = {"name": "r again", "inputs": ["NADPH", "A"], "outputs": ["NADP+", "B", "H2O"], "evidence": "E later"}
    merged, report = merge_draws([{"processes": {"reactions": [first]}}, {"processes": {"reactions": [later]}}])
    [row] = merged["processes"]["reactions"]
    assert (row["name"], row["evidence"]) == ("r", first["evidence"])
    assert row["inputs"] == ["A", "NADPH"] and row["outputs"] == ["B", "NADP+", "H2O"]
    [rec] = report["buckets"]["/processes/reactions"]["rows"]
    assert rec["draws"] == [1, 2] and rec["cofactors_added"] == ["NADPH", "NADP+", "H2O"]
    assert later["inputs"] == ["NADPH", "A"]  # the draw's own row is never touched
    # Nothing is added twice, and a survivor that already names the cofactor gains nothing.
    merged2, report2 = merge_draws([{"processes": {"reactions": [later]}}, {"processes": {"reactions": [first]}},
                                    {"processes": {"reactions": [later]}}])
    [row2] = merged2["processes"]["reactions"]
    assert row2["inputs"] == ["NADPH", "A"] and row2["evidence"] == "E later"  # nothing to donate
    assert "cofactors_added" not in report2["buckets"]["/processes/reactions"]["rows"][0]


def test_interaction_verb_forms_are_one_relationship() -> None:
    k = lambda rel: row_key("/processes/interactions", {"entity_1": "HY5", "relationship": rel, "entity_2": "CHLH"})  # noqa: E731
    assert k("activates") == k("Activation") == k("activation")
    assert k("inhibits") == k("inhibition") != k("activates")
    # A word carrying no sign stays as written: "binds" is not "interacts_with".
    assert k("binds") != k("interacts_with")


# --------------------------------------------------------------------------- ORCH-754a: the paper is the dictionary
# G9: a CORRECTION of the delivered pooling behaviour (ORCH-751 § 2.2, ORCH-752 § 2): the
# chlorophyll draws pool the same chemistry under the paper's abbreviation, its long form
# and a misspelling as separate rows. This test drives the PUBLIC entry point with the three
# real draws and the real paper text, so it fails behaviourally on the base module and
# passes at the tip (base pre-runs: ``evidence/g11/ORCH-754a/01``, ``06``, ``46`` and the one
# for this expectation, named in ORCH-754a-RESULT.md). D-104 rules 1 and 2, under finding 1's
# compatibility rule as the product owner ruled it on 2026-10-20: draw 2's undefined
# ``plastidic state`` (it wrote the bucket as ``biogical_states``) counts as no state, so
# every step pools into ONE row -- except two, where the enzymes conflict: ``PPO`` vs
# ``Protogen IX oxidase (PPO)`` (Schwartz-Hearst cannot derive the pair) and ``POR`` vs
# ``Pchlide oxidoreductase (POR)`` (the paper gives POR two long forms). The base module
# pools every step into 2 or 3 rows.

EVIDENCE = ROOT / "docs" / "pwml_recovery_sprint" / "evidence"
CHLOROPHYLL_DRAWS = [EVIDENCE / "orch752_chlorophyll_payloads" / f"control__run{i}__payload.json" for i in (1, 2, 3)]
CHLOROPHYLL_PAPER = EVIDENCE / "orch754a_inputs" / "papers" / "PMC5130987.txt"

# Each step is one reaction of the paper written under every spelling the three draws used
# (abbreviation, ``long (ABBR)``, misspelling) plus the paper's long form.
_GSA = {"GSA", "Glu 1-semialdehyde (GSA)", "Glu 1-semialdehyde"}
_ALA = {"ALA", "5-aminolevulinic acid (ALA)", "5-aminolevulinic acid"}
_PBG = {"PBG", "porphobilinogen (PBG)", "porphobilogen (PBG)", "porphobilinogen"}
_HMB = {"HMB", "hydroxymethybilane (HMB)", "hydroxymethylbilane (HMB)", "hydroxymethylbilane"}
_UROGEN = {"Urogen III", "uroporphyrinogen III"}
_COPROGEN = {"Coprogen III", "coproporphyrinogen III"}
_PROTOGEN = {"Protogen IX", "protoporphyrinogen IX"}
_PROTO = {"Proto IX", "protoporphyrin IX"}
_MGPROTO_ME = {"Mg-Proto ME", "Mg-Proto IX monomethyl ester (Mg-Proto ME)", "Mg-Proto IX monomethyl ester"}
_DV_PCHLIDE = {"DV-Pchlide", "3,8-divinyl-protochlorophyllide (DV-Pchlide)", "3,8-divinyl-protochlorophyllide"}
_DV_CHLIDE = {"DV-Chlide", "3,8-divinyl-chlorophyllide (DV-Chlide)", "3,8-divinyl-chlorophyllide"}
CHLOROPHYLL_STEPS = [
    ({"Glu-tRNAGlu"}, _GSA), (_GSA, _ALA), (_ALA, _PBG), (_PBG, _HMB), (_HMB, _UROGEN),
    (_UROGEN, _COPROGEN), (_COPROGEN, _PROTOGEN), (_PROTOGEN, _PROTO), ({"Mg-Proto IX"}, _MGPROTO_ME),
    (_MGPROTO_ME, _DV_PCHLIDE), (_DV_PCHLIDE, _DV_CHLIDE),
]


def _keys(names: Any) -> Set[str]:
    from t2pw.bench.eval_v2.matcher import strict_key

    return {strict_key(n) for n in names}


def test_chlorophyll_pools_each_paper_step_once(no_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    draws = [json.loads(p.read_text(encoding="utf-8")) for p in CHLOROPHYLL_DRAWS]
    paper = CHLOROPHYLL_PAPER.read_text(encoding="utf-8")
    rec = Recorder(draws)
    monkeypatch.setattr(pipeline_mod, "_run_json_stage", rec)

    merged, _attempts = run_extraction_pipeline(paper, draws=3)

    assert len(rec.calls) == 3
    pooled = [(_keys(r.get("inputs") or []), _keys(r.get("outputs") or [])) for r in _reactions(merged)]
    counts = {}
    for ins, outs in CHLOROPHYLL_STEPS:
        want_in, want_out = _keys(ins), _keys(outs)
        counts[f"{sorted(ins)[0]} -> {sorted(outs)[0]}"] = sum(
            1 for got_in, got_out in pooled if got_in & want_in and got_out & want_out)
    split = {"Protogen IX -> Proto IX", "3,8-divinyl-protochlorophyllide -> 3,8-divinyl-chlorophyllide"}
    assert counts == {step: 2 if step in split else 1 for step in counts}, counts


# --------------------------------------------------------------------------- ORCH-754a finding 1: chimeric reactions
# G9: CORRECTIONS of ORCH-750/752's delivered behaviour, so each fails behaviourally on the
# base module (base pre-runs ``evidence/g11/ORCH-754a/06``, ``46``, ``69`` and the round-3 one
# named in ORCH-754a-RESULT.md) and passes at the tip. The base key
# ignores cofactors, enzymes and biological_state and the merge unions cofactors, so an
# NAD-dependent and an NADP-dependent IDH row became ONE chimeric row
# ``isocitrate + NAD+ + NADP+ -> 2-oxoglutarate + NADH + CO2 + NADPH`` carrying only IDH3 and
# draw 1's evidence (reviewer probe ``C:/t/bt-orch754r0/probe/probe_chimera.py``). Rule: two
# rows merge only when they are compatible -- one omitting what the other names -- never
# when they conflict.

IDH_NAD = {"name": "IDH (NAD)", "inputs": ["isocitrate", "NAD+"], "outputs": ["2-oxoglutarate", "NADH", "CO2"],
           "enzymes": [{"entity": "IDH3"}], "evidence": "q1"}
IDH_NADP = {"name": "IDH (NADP)", "inputs": ["isocitrate", "NADP+"], "outputs": ["2-oxoglutarate", "NADPH", "CO2"],
            "enzymes": [{"entity": "IDH1"}], "evidence": "q2"}


def _shape(row: Dict[str, Any]) -> Any:
    return (row["name"], row["inputs"], row["outputs"], row.get("enzymes"), row["evidence"])


@pytest.mark.parametrize("draws", [
    [{"processes": {"reactions": [IDH_NAD]}}, {"processes": {"reactions": [IDH_NADP]}}],   # two draws
    [{"processes": {"reactions": [IDH_NAD, IDH_NADP]}}],                                   # one draw
], ids=["across_draws", "within_one_draw"])
def test_the_reviewers_idh_chimera_stays_two_reactions(draws: List[Dict[str, Any]]) -> None:
    merged, _report = merge_draws(deepcopy(draws))
    assert [_shape(r) for r in _reactions(merged)] == [_shape(IDH_NAD), _shape(IDH_NADP)]


def test_two_rows_each_naming_a_cofactor_the_other_lacks_stay_apart() -> None:
    """Review B3: cofactor sets must NEST per side. A chemical-family table let MDH (NAD+)
    meet MQO (FAD) and ATP-PFK meet PPi-PFP (reviewer probes p1, p3); both stay apart."""

    def pair(x: Dict[str, Any], y: Dict[str, Any]) -> List[str]:
        merged, _ = merge_draws([{"processes": {"reactions": [x]}}, {"processes": {"reactions": [y]}}])
        return [r["name"] for r in _reactions(merged)]

    mdh = {"name": "MDH", "inputs": ["malate", "NAD+"], "outputs": ["oxaloacetate", "NADH"],
           "enzymes": [{"entity": "MDH"}], "evidence": "MDH oxidises malate with NAD+"}
    mqo = {"name": "MQO", "inputs": ["malate", "FAD"], "outputs": ["oxaloacetate", "FADH2"],
           "evidence": "malate:quinone oxidoreductase uses FAD"}
    pfk = {"name": "PFK", "inputs": ["fructose 6-phosphate", "ATP"], "outputs": ["fructose 1,6-bisphosphate", "ADP"],
           "enzymes": [{"entity": "PFK"}], "evidence": "PFK uses ATP"}
    pfp = {"name": "PFP", "inputs": ["fructose 6-phosphate", "PPi"], "outputs": ["fructose 1,6-bisphosphate", "Pi"],
           "evidence": "PPi-dependent PFP"}
    assert pair(mdh, mqo) == ["MDH", "MQO"]
    assert pair(pfk, pfp) == ["PFK", "PFP"]
    a = {"name": "a", "inputs": ["X", "ATP"], "outputs": ["Y"], "evidence": "X and ATP give Y"}
    b = {"name": "b", "inputs": ["X", "ADP"], "outputs": ["Y"], "evidence": "X and ADP give Y"}
    c = {"name": "c", "inputs": ["X", "Mg2+"], "outputs": ["Y"], "evidence": "X with Mg2+ gives Y"}
    d = {"name": "d", "inputs": ["X", "Mn2+"], "outputs": ["Y"], "evidence": "X with Mn2+ gives Y"}
    merged, _ = merge_draws([{"processes": {"reactions": [a]}}, {"processes": {"reactions": [b]}}])
    assert [r["name"] for r in _reactions(merged)] == ["a", "b"]
    merged, _ = merge_draws([{"processes": {"reactions": [c]}}, {"processes": {"reactions": [d]}}])
    assert [r["name"] for r in _reactions(merged)] == ["c", "d"]


def test_different_enzymes_or_defined_states_separate_the_same_chemistry() -> None:
    states = [{"name": "cyto", "subcellular_location": "cytosol"}, {"name": "mito", "subcellular_location": "mitochondrion"}]
    one = {"name": "one", "inputs": ["X"], "outputs": ["Y"], "enzymes": [{"entity": "E1"}], "evidence": "q"}
    two = {"name": "two", "inputs": ["X"], "outputs": ["Y"], "enzymes": [{"entity": "E2"}], "evidence": "q"}
    merged, _ = merge_draws([{"processes": {"reactions": [one]}}, {"processes": {"reactions": [two]}}])
    assert [r["name"] for r in _reactions(merged)] == ["one", "two"]
    cyto = {"name": "cyto row", "inputs": ["X"], "outputs": ["Y"], "biological_state": "cyto", "evidence": "q"}
    mito = {"name": "mito row", "inputs": ["X"], "outputs": ["Y"], "biological_state": "mito", "evidence": "q"}
    merged, _ = merge_draws([{"biological_states": states, "processes": {"reactions": [cyto]}},
                             {"biological_states": states, "processes": {"reactions": [mito]}}])
    assert [r["name"] for r in _reactions(merged)] == ["cyto row", "mito row"]


def test_a_cofactor_the_survivors_own_quote_does_not_support_is_not_donated() -> None:
    first = {"name": "r", "inputs": ["A"], "outputs": ["B"], "evidence": "A becomes B"}
    later = {"name": "r again", "inputs": ["A", "NADPH"], "outputs": ["B"], "evidence": "NADPH reduces A to B"}
    merged, report = merge_draws([{"processes": {"reactions": [first]}}, {"processes": {"reactions": [later]}}])
    [row] = _reactions(merged)
    assert row["inputs"] == ["A"] and row["evidence"] == "A becomes B"


def test_the_draw_count_variable_is_the_one_conftest_pins() -> None:
    from conftest import DRAWS_ENV_VAR

    assert DRAWS_ENV_VAR == ENV_VAR
