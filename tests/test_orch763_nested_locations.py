"""ORCH-763 (D-109 § 1): nested locations are merge-compatible; an unknown pair still conflicts.

G9 labelling, test by test:

* **G9 CORRECTION PROOFS** -- fail behaviourally at base ``5d1ea64c``, pass at the tip. The leg
  tests run on the ORCH-759 menaquinone leg's own draws (``tests/fixtures/orch763/``, rebuilt by
  ``evidence/orch763/build_fixture.py``; re-pooled at base they reproduce the leg's side file row
  for row, ``evidence/orch763/`` re-pool report):
  ``test_g9_leg_menh_to_meni_duplicates_merge_keeping_cytosol``, ``test_g9_leg_pool_16_to_11``,
  and ``test_g9_a_row_without_location_does_not_join_two_non_nested_locations`` (at base a row
  that states no location carried two locations that do not nest into one pooled row).
* **NEW ACCEPTANCE (D-109 § 1)** -- the new rule itself, labelled new; they fail at base by
  construction and claim nothing more: ``test_new_*``.
* **PRESERVATION** -- pass at base and at the tip: ``test_preserved_*``.

String literals, not new module symbols, so the file collects at base and its failures there
are behaviour, not a missing name. No model, network or database is called.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench.eval_v2 import reactions as RX  # noqa: E402
from t2pw.bench.eval_v2.gold import load_fulltext, load_supplement  # noqa: E402
from t2pw.pipeline import extraction_diagnostics as diag  # noqa: E402
from t2pw.pipeline import stage_one_ensemble as E  # noqa: E402
from t2pw.pipeline.stage_one_ensemble import (  # noqa: E402
    BOUNDARY_STAGE1_ENSEMBLE,
    ENV_VAR,
    IDENTITY_REPORT_KEY,
    merge_draws,
    merge_on_mapped_identity,
    render_report_text,
)

FIX = ROOT / "tests" / "fixtures" / "orch763"
PAPERS = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "orch754a_inputs" / "papers"
CHLOROPHYLL_751 = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "orch752_chlorophyll_payloads"
FLAG = "nested_location_merged"
MEN_STEPS = ("MenH reaction", "MenC reaction", "MenE reaction", "MenB reaction", "MenI reaction")


def _fixture(name: str) -> Dict[str, Any]:
    return json.loads((FIX / f"{name}_orch759_draws.json").read_text(encoding="utf-8"))


def _paper(pid: str) -> str:
    return (PAPERS / f"{pid}.txt").read_text(encoding="utf-8")


def _rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return report["buckets"]["/processes/reactions"]["rows"]


def _reactions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return payload["processes"]["reactions"]


STATES = [{"name": "cell state", "species": "Escherichia coli"},
          {"name": "intracellular state", "species": "Escherichia coli", "subcellular_location": "intracellular"},
          {"name": "cytosolic state", "species": "Escherichia coli", "subcellular_location": "cytoplasm",
           "compartment_canonical": "cytosol"},
          {"name": "matrix state", "species": "Escherichia coli", "compartment_canonical": "mitochondrial_matrix"},
          {"name": "membrane state", "species": "Escherichia coli", "subcellular_location": "membrane"},
          {"name": "periplasm state", "species": "Escherichia coli", "subcellular_location": "periplasm"},
          {"name": "nuclear state", "species": "Escherichia coli", "subcellular_location": "nucleus"},
          {"name": "subtilis intracellular", "species": "Bacillus subtilis", "subcellular_location": "intracellular"}]


def _row(name: str, state: str, **extra: Any) -> Dict[str, Any]:
    return {"name": name, "inputs": ["X1"], "outputs": ["X2"], "biological_state": state,
            "evidence": f"quote of {name}", **extra}


def _draw(*rows: Dict[str, Any]) -> Dict[str, Any]:
    return {"biological_states": deepcopy(STATES), "processes": {"reactions": list(rows)}}


# ------------------------------------------------------------------ G9: the leg's own draws


def _men_steps(draw: Dict[str, Any]) -> Dict[str, Any]:
    """Only the MenH..MenI rows of a draw: verbatim leg rows (the side file kept each apart)."""

    return {"biological_states": deepcopy(draw["biological_states"]),
            "processes": {"reactions": [r for r in draw["processes"]["reactions"] if r["name"] in MEN_STEPS]}}


def test_g9_leg_menh_to_meni_duplicates_merge_keeping_cytosol() -> None:
    """At base the 5 MenH..MenI rows of draw 3 stay apart from draw 2's on 'cytosol' vs
    'intracellular' (the ORCH-759 side file: 5 extra rows). At the tip each pair merges, the
    merge is flagged, and the survivor carries draw 2's state, whose location is cytosol."""

    draws = _fixture("menaquinone")["draws"]
    merged, report = merge_draws([_men_steps(draws[1]), _men_steps(draws[2])], paper_text=_paper("PMC12421875"))
    assert [r["name"] for r in _reactions(merged)] == list(MEN_STEPS)  # base: 10 rows
    states = {s["name"]: s for s in merged["biological_states"]}
    for row, rec in zip(_reactions(merged), _rows(report)):
        assert row["biological_state"] == "intracellular state"
        assert states[row["biological_state"]]["compartment_canonical"] == "cytosol"
        [merge] = rec["merges"]
        assert merge["draw"] == 2 and merge["location_flag"] == {"reason": FLAG, "locations": "'cytosol' within 'intracellular'"}
        assert "kept_apart_from" not in rec
    text = render_report_text(report)
    assert text.count(f"FLAG (review): {FLAG} -- 'cytosol' within 'intracellular'") == 5
    assert "5 merge(s) flagged for review" in text


def test_g9_leg_pool_16_to_11() -> None:
    """The whole leg: 16 pooled rows at base (orch759_gold_accuracy.txt, Stage 1), 11 at the tip.
    The other 5 surplus rows were the MenH..MenI pairs; MenA and MenG stay split on DMK/MK (the
    paper dictionary has no such pair, ORCH-763 (b)). Gold 9/9 is kept."""

    fixture = _fixture("menaquinone")
    assert [len(d["processes"]["reactions"]) for d in fixture["draws"]] == [4, 9, 9]  # the leg's rows_per_draw
    merged, report = merge_draws(fixture["draws"], paper_text=_paper("PMC12421875"))
    assert len(_reactions(merged)) == 11
    case = load_supplement()["PMC12421875"]
    t1 = RX.score_leg(case, merged, [], load_fulltext(case, PAPERS), expected_species=case.species)["table1_paper_extraction"]
    assert (len(t1["gold_reactions_matched"]), t1["gold_reactions"], t1["precision"]) == (9, 9, 1.0)
    keys = [r["key"] for r in _rows(report)]
    assert len(keys) - len(set(keys)) == 0  # no reaction key is split any more


def test_g9_a_row_without_location_does_not_join_two_non_nested_locations() -> None:
    """At base the record kept the first row's state as its claim: a row stating no location
    let a cytosol row AND a mitochondrial-matrix row join, one pooled row across two locations
    that do not nest. At the tip the record claims its narrowest location, so the matrix row
    is kept apart."""

    merged, report = merge_draws([_draw(_row("plain", "cell state")), _draw(_row("cyto", "cytosolic state")),
                                  _draw(_row("matrix", "matrix state"))])
    assert [r["name"] for r in _reactions(merged)] == ["plain", "matrix"]  # base: ["plain"]
    assert _rows(report)[1]["kept_apart_from"][0]["reason"] == "biological_state location 'cytosol' vs 'mitochondrial matrix'"


# ------------------------------------------------------------------ new acceptance (D-109 § 1)


def test_new_the_survivor_takes_the_more_specific_location_when_the_broader_row_came_first() -> None:
    merged, report = merge_draws([_draw(_row("broad", "intracellular state")), _draw(_row("narrow", "cytosolic state"))])
    [row] = _reactions(merged)
    assert (row["name"], row["evidence"], row["biological_state"]) == ("broad", "quote of broad", "cytosolic state")
    [rec] = _rows(report)
    assert rec["state_taken"] == ["cytosolic state"]
    assert rec["merges"][0]["location_flag"] == {"reason": FLAG, "locations": "'cytosol' within 'intracellular'"}
    assert "Biological state taken from a merged row: 'cytosolic state'" in render_report_text(report)


def test_new_a_record_never_joins_two_locations_through_their_common_container() -> None:
    """intracellular holds both cytosol and nucleus; once cytosol has joined, nucleus may not."""

    merged, report = merge_draws([_draw(_row("broad", "intracellular state")), _draw(_row("cyto", "cytosolic state")),
                                  _draw(_row("nuc", "nuclear state"))])
    assert [r["name"] for r in _reactions(merged)] == ["broad", "nuc"]
    assert _rows(report)[1]["kept_apart_from"][0]["reason"].startswith("biological_state location 'cytosol' vs ")


@pytest.fixture
def pooled_run(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv(ENV_VAR, "3")
    diag.activate(run_id="orch763-rule3")
    diag.current().record_boundary(stage="extraction", boundary=BOUNDARY_STAGE1_ENSEMBLE, attempts=3,
                                   ensemble={"draws_succeeded": 3})
    yield
    diag.deactivate()


def _mapped(*rows: Dict[str, Any]) -> Dict[str, Any]:
    ids = {"mapped_ids": {"chebi": "CHEBI:1"}, "mapping_meta": {"confidence": 1.0, "identity_status": "verified"}}
    return {"entities": {"compounds": [{"name": "X2", **ids}, {"name": "X2 alias", **deepcopy(ids)}]},
            "biological_states": deepcopy(STATES), "processes": {"reactions": deepcopy(list(rows))}}


@pytest.mark.usefixtures("pooled_run")
def test_new_rule3_merges_nested_locations_and_flags_the_merge() -> None:
    payload = _mapped(_row("cyto", "cytosolic state"), {**_row("broad", "intracellular state"), "outputs": ["X2 alias"]})
    record = merge_on_mapped_identity(payload)[IDENTITY_REPORT_KEY]
    [row] = _reactions(payload)
    assert (row["name"], row["biological_state"]) == ("cyto", "cytosolic state")
    [merge] = record["merges"]
    assert merge["location_flag"] == {"reason": FLAG, "locations": "'cytosol' within 'intracellular'"}
    assert FLAG not in json.dumps(payload)  # the payload a model reads carries no mark


@pytest.mark.usefixtures("pooled_run")
def test_preserved_rule3_keeps_non_nested_locations_apart() -> None:
    payload = _mapped(_row("cyto", "cytosolic state"), {**_row("matrix", "matrix state"), "outputs": ["X2 alias"]})
    record = merge_on_mapped_identity(payload)[IDENTITY_REPORT_KEY]
    assert [r["name"] for r in _reactions(payload)] == ["cyto", "matrix"]
    assert record["refused"][0]["reason"] == "biological_state location 'cytosol' vs 'mitochondrial matrix'"


def test_new_the_containment_table_is_a_tree_of_folded_names() -> None:
    parents = E._LOCATION_PARENT
    for child in parents:
        assert E._location(child) == child and E._location(parents[child]) == parents[child]
        seen, at = set(), child
        while at in parents:  # no cycles: every chain ends
            assert at not in seen
            seen.add(at)
            at = parents[at]
    assert E._nests("intracellular", "cytosol") and E._nests("plastid", "thylakoid membrane")
    assert not E._nests("cytosol", "membrane") and not E._nests("cytosol", "nucleus")
    assert not E._nests("cytosol", "periplasm")  # unknown: fail closed


# ------------------------------------------------------------------ preservation


@pytest.mark.parametrize("state, reason", [
    ("periplasm state", "biological_state location 'cytosol' vs 'periplasm'"),           # unknown pair
    ("matrix state", "biological_state location 'cytosol' vs 'mitochondrial matrix'"),   # do not nest
    ("membrane state", "biological_state location 'cytosol' vs 'membrane'"),             # D-109's own example
], ids=["unknown_pair", "cytosol_vs_matrix", "cytosol_vs_membrane"])
def test_preserved_locations_that_do_not_nest_stay_apart(state: str, reason: str) -> None:
    merged, report = merge_draws([_draw(_row("cyto", "cytosolic state")), _draw(_row("other", state))])
    assert [r["name"] for r in _reactions(merged)] == ["cyto", "other"]
    assert _rows(report)[1]["kept_apart_from"][0]["reason"] == reason


@pytest.mark.parametrize("extra_a, extra_b, state_b", [
    ({"inputs": ["X1", "NAD+"]}, {"inputs": ["X1", "NADP+"]}, "intracellular state"),
    ({"enzymes": [{"entity": "E1"}]}, {"enzymes": [{"entity": "E2"}]}, "intracellular state"),
    ({}, {}, "subtilis intracellular"),
], ids=["cofactor", "enzyme", "species"])
def test_preserved_a_chimeric_pair_stays_apart_even_with_nested_locations(extra_a: Any, extra_b: Any, state_b: str) -> None:
    a = {**_row("a", "cytosolic state"), **extra_a, "evidence": "E1 and E2 with NAD+ and NADP+"}
    b = {**_row("b", state_b), **extra_b, "evidence": "E1 and E2 with NAD+ and NADP+"}
    merged, _report = merge_draws([_draw(a), _draw(b)])
    assert [r["name"] for r in _reactions(merged)] == ["a", "b"]


def test_preserved_equal_locations_under_two_names_merge_unflagged() -> None:
    states = [{"name": "s1", "subcellular_location": "cytosol"}, {"name": "s2", "compartment_canonical": "cytosol"}]
    merged, report = merge_draws([{"biological_states": states, "processes": {"reactions": [_row("a", "s1")]}},
                                  {"biological_states": states, "processes": {"reactions": [_row("b", "s2")]}}])
    assert [r["name"] for r in _reactions(merged)] == ["a"]
    assert "location_flag" not in _rows(report)[0]["merges"][0]


def test_preserved_the_pooled_payload_carries_no_merge_mark() -> None:
    """D-104 amendment 2: the pooled payload is the Stage-2 prompt; flags live in the side file."""

    merged, _report = merge_draws([_draw(_row("broad", "intracellular state")), _draw(_row("narrow", "cytosolic state"))])
    text = json.dumps(merged)
    assert FLAG not in text and "location_flag" not in text


def test_preserved_enterobactin_leg_keeps_5_of_5() -> None:
    merged, _report = merge_draws(_fixture("enterobactin")["draws"], paper_text=_paper("PMC12096016"))
    case = load_supplement()["PMC12096016"]
    t1 = RX.score_leg(case, merged, [], load_fulltext(case, PAPERS), expected_species=case.species)["table1_paper_extraction"]
    assert (len(t1["gold_reactions_matched"]), t1["gold_reactions"], len(_reactions(merged))) == (5, 5, 6)


def _pairwise_conflicts(draws: List[Dict[str, Any]], text: str) -> List[str]:
    """Every pair of draw rows inside one pooled row, re-compared on their OWN profiles."""

    from t2pw.pipeline.name_keys import strict_key
    from t2pw.pipeline.paper_lexicon import PaperLexicon

    _merged, report = merge_draws(draws, paper_text=text)
    lex = PaperLexicon(text)
    found = []
    for rec in _rows(report):
        members = [(rec["draws"][0], rec["name"], None)] + [(m["draw"], m["name"], m.get("evidence"))
                                                           for m in rec.get("merges") or []]
        profiles = []
        for d, name, evidence in members:
            draw = draws[d - 1]
            row = next(r for r in draw["processes"]["reactions"]
                       if r.get("name") == name and (evidence is None or r.get("evidence") == evidence))
            profiles.append(E._profile(row, E._states_by_key(draw), lambda s: strict_key(lex.canonical(s)[0])))
        found += [f"{rec['name']}: {E._conflict(p, q)}" for i, p in enumerate(profiles) for q in profiles[i + 1:]
                  if E._conflict(p, q)]
    return found


@pytest.mark.parametrize("source", ["orch759", "orch751"])
def test_preserved_chlorophyll_repools_without_a_chimera(source: str) -> None:
    if source == "orch759":
        draws = _fixture("chlorophyll")["draws"]
    else:
        draws = [json.loads((CHLOROPHYLL_751 / f"control__run{i}__payload.json").read_text(encoding="utf-8"))
                 for i in (1, 2, 3)]
    assert _pairwise_conflicts(draws, _paper("PMC5130987")) == []
