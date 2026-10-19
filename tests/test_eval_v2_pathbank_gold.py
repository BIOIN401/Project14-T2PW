"""NEW ACCEPTANCE TESTS (ORCH-747) -- PathBank structural extractor and its draft case.

A genuinely NEW capability: nothing in the repository read pathway- or reaction-level
PathBank tables before this module (``pwml/db_resolver.py`` and
``mapping/build_pathwhiz_id_db.py`` map compound/protein ids only). So these are labelled
new acceptance tests under merge gate G9 and carry no base-SHA failure.

Every test here is DATABASE-FREE: it runs against the committed extract of PW701720.
The point of the suite is the safety property -- a PathBank record can never be mistaken
for paper evidence -- plus the shape of the draft the paper-gold drafter is handed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from t2pw.bench.eval_v2.gold import GoldV2Error, load_supplement
from t2pw.bench.eval_v2.pathbank_gold import (REVERSIBLE_DIRECTIONS, SCHEMA_DRAFT,
                                              SCHEMA_EXTRACT, _synonyms, draft_case)

ROOT = Path(__file__).resolve().parents[1]
EXTRACT = ROOT / "docs/evaluation_v2/pathbank_gold/PW701720__extract.json"


@pytest.fixture(scope="module")
def extract() -> dict:
    if not EXTRACT.is_file():
        pytest.skip(f"{EXTRACT} not committed yet")
    return json.loads(EXTRACT.read_text(encoding="utf-8"))


def test_committed_extract_is_the_autoregulation_pathway(extract):
    """The accession resolves through the visualization table, not ``pathways.id``.

    PW701720 as a ``pathways.id`` is a cardiolipin pathway; via
    ``super_pathway_visualizations.pw_id`` it is pathway 703792, the AraC autoregulation
    pathway. This pins which of the two the extractor read.
    """
    assert extract["schema"] == SCHEMA_EXTRACT
    assert extract["accession"] == "PW701720"
    assert extract["pathway_id"] == 703792
    assert extract["name"] == "Ara operon Inactivation (Autoregulation of AraC)"
    assert extract["species"]["name"] == "Escherichia coli"
    assert extract["species"]["taxonomy_id"] == "562"
    assert extract["unhandled_process_types"] == []


def test_extract_records_mutability_and_a_content_hash(extract):
    """Nine of the ten eval pathways have ``status`` NULL and can still be edited, so an
    extract that cannot be compared later is not evidence."""
    assert len(extract["content_sha256"]) == 64
    assert extract["curator"]["updated_at"]
    assert set(extract["mutability"]) >= {"private_pathway", "locked", "status_is_null"}


def test_every_curated_process_is_carried_with_its_accession(extract):
    """7 interactions + 1 reaction, each with its PW accession and participants."""
    assert extract["counts"] == {"Interaction": 7, "Reaction": 1}
    for p in extract["processes"]:
        assert p["accession"], p
        assert p["participants"], p
        for part in p["participants"]:
            assert part["element_type"]
            assert part["resolved"] is True, part
            assert part["side"] in {"left", "right", "moved"}


def test_draft_case_shape_and_the_non_metabolic_finding(extract):
    """PW701720's only 'reaction' is araC gene -> AraC protein: a NucleicAcid input and a
    ProteinComplex output, i.e. expression, not a metabolite conversion. The whole
    mechanism lives in the interactions, which the eval-v2 reaction matcher does not
    score. This test pins that finding so the scoring decision cannot be lost."""
    d = draft_case(extract)
    assert d["schema"] == SCHEMA_DRAFT
    case = d["case"]
    assert len(case["reactions"]) == 1
    assert len(case["interactions"]) == 7
    assert case["transports"] == []

    r = case["reactions"][0]
    assert [t["entity_kind"] for t in r["inputs"]] == ["NucleicAcid"]
    assert [t["entity_kind"] for t in r["outputs"]] == ["ProteinComplex"]
    assert not any(t.get("entity_kind") == "Compound" for t in r["inputs"] + r["outputs"])

    effects = sorted(i["effect"] for i in case["interactions"])
    assert effects.count("activation") == 6 and effects.count("inhibition") == 1
    for i in case["interactions"]:
        assert i["source"] and i["target"] and i["pathbank_id"]


def test_the_draft_carries_no_evidence_and_says_so(extract):
    d = draft_case(extract)
    assert d["case"]["supported_reactions_complete"] is False
    assert d["case_id"] is None
    assert "not from any paper" in d["not_gold"]
    for row in d["case"]["reactions"]:
        assert row["quote"] == ""
        assert row["supporting_quotes"] == []
        assert "verbatim quote" in row["evidence_required"]
    for row in d["case"]["interactions"] + d["case"]["transports"]:
        assert row["quote"] == ""


def test_the_draft_can_never_load_as_gold(extract, tmp_path):
    """THE safety property. Two ways someone could try to use the draft as gold:

    * hand the file straight to the loader -- it holds no ``cases`` key, so no case is
      ever produced;
    * wrap its case under ``cases`` -- the loader refuses it, because a reaction without
      a quote is an assertion, not evidence.
    """
    d = draft_case(extract)
    straight = tmp_path / "draft.json"
    straight.write_text(json.dumps(d), encoding="utf-8")
    assert load_supplement(straight) == {}

    wrapped = tmp_path / "wrapped.json"
    case = dict(d["case"])
    case["source"] = {"pmcid": "PMC319880", "source_url": "u", "fetch_date": "2026-10-13",
                      "xml_sha256": "x", "text_sha256": "y"}
    wrapped.write_text(json.dumps({"cases": {"PMC319880": case}}), encoding="utf-8")
    with pytest.raises(GoldV2Error) as err:
        load_supplement(wrapped)
    assert "quote" in str(err.value)


def test_pathbank_identifiers_survive_onto_draft_terms(extract):
    """The compound/protein ids are what the mapped-ID metric is later judged against, so
    losing them in the draft would silently weaken it."""
    d = draft_case(extract)
    enz = [i for i in d["case"]["interactions"]
           if any(t.get("entity_kind") == "ProteinComplex" for t in i["source"])]
    assert enz, "this pathway has protein-complex sources"
    terms = [t for i in d["case"]["interactions"] for t in i["source"] + i["target"]]
    assert any(t.get("pathbank_identifiers") for t in terms)


@pytest.mark.parametrize("raw,expected", [
    ("", []),
    (None, []),
    ("ATP", ["ATP"]),
    ("a; b;c", ["a", "b", "c"]),
    ("a\nb; a", ["a", "b"]),
])
def test_synonyms_split_on_both_separators_and_dedupe(raw, expected):
    assert _synonyms(raw) == expected


def test_reversible_directions_do_not_include_the_forward_arrow():
    """``reactions.direction`` is ``'>'`` for the curated forward step; treating that as
    reversible would let a backwards row score as a match."""
    assert ">" not in REVERSIBLE_DIRECTIONS
    assert "<>" in REVERSIBLE_DIRECTIONS


# ---------------------------------------------------------------------------
# ORCH-747, NEW CAPABILITY: a second gold supplement can be scored without
# editing the pinned one. Labelled new acceptance tests; no base-SHA failure is
# claimed, because before this commit the parameter did not exist.
# ---------------------------------------------------------------------------

def test_evaluate_and_run_live_accept_a_supplement_path():
    import inspect

    from t2pw.bench.eval_v2 import report as RP

    for fn in (RP.evaluate, RP.run_live):
        p = inspect.signature(fn).parameters.get("supplement_path")
        assert p is not None, f"{fn.__name__} takes no supplement_path"
        assert p.default is None, "the default must keep every existing number identical"


def test_scoring_a_foreign_supplement_scores_none_of_the_pinned_cases(tmp_path):
    """The plumbing is real, not just a parameter: pointed at a supplement that labels a
    paper the committed corpus does not contain, the evaluator scores no leg at all."""
    from t2pw.bench.eval_v2 import report as RP

    empty = tmp_path / "other_gold.json"
    empty.write_text(json.dumps({"schema": "t2pw.eval_v2.gold_supplement/1", "cases": {}}),
                     encoding="utf-8")
    res = RP.evaluate(ROOT, None, empty)
    assert res["gold_supplement"]["complete_cases"] == []
    assert res["reaction_metrics"]["legs"] == []
