"""NEW ACCEPTANCE TESTS (ORCH-742c, evaluation v2) -- reaction metrics, mapped-ID
correctness, RAG failure split, retrieval rank metrics, un-gated RAG influence audit,
reproducibility-metadata completeness, and the production import boundary.

NEW capabilities, so per G9 there is no base-SHA failure to show: the evaluator does
not exist at the base. Every fixture is synthetic and in-file except the two
committed-artifact characterisations, which read git-tracked files only.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from t2pw.bench.eval_v2 import reactions as R
from t2pw.bench.eval_v2.gold import CaseV2, Reaction, Term
from t2pw.bench.eval_v2.rag import archived_failure_split, rank_metrics, valid_source_reference
from t2pw.bench.eval_v2.repro import completeness

ROOT = Path(__file__).resolve().parents[1]


def _rx(rid, ins, outs, enzyme=None, reversible=False, kind="in_scope", cof_in=()):
    return Reaction(
        id=rid, label=rid,
        inputs=tuple(Term(n) for n in ins) + tuple(Term(n, role="cofactor") for n in cof_in),
        outputs=tuple(Term(n) for n in outs),
        enzyme=Term(enzyme) if enzyme else None, reversible=reversible,
        quote=f"{' '.join(ins)} is converted to {' '.join(outs)}", kind=kind)


CASE = CaseV2(
    paper_id="PMC0", complete=True, species="Escherichia coli", scope="", source={},
    reactions=(
        _rx("R1", ["chorismate"], ["isochorismate"], "EntC", reversible=True),
        _rx("R2", ["isochorismate"], ["2,3-diDHB", "pyruvate"], "EntB"),
        _rx("R3", ["2,3-DHB"], ["2,3-DHB-AMP"], "EntE", cof_in=["ATP"]),
    ),
    off_pathway=(_rx("O1", ["pyruvate", "NADH"], ["lactate"], "LDH", kind="assay"),),
    summaries=(_rx("S1", ["chorismate"], ["2,3-DHB"], kind="summary"),),
)
PAPER = "chorismate is converted to isochorismate by EntC. isochorismate is converted to 2,3-diDHB pyruvate."


def _row(ins, outs, enz=(), state="cyto", evidence=""):
    return {"inputs": list(ins), "outputs": list(outs),
            "enzymes": [{"entity": e} for e in enz], "biological_state": state, "evidence": evidence}


PAYLOAD = {
    "biological_states": [{"name": "cyto", "species": "Escherichia coli K-12"}],
    "processes": {"reactions": [
        _row(["isochorismate"], ["chorismate"], ["EntC"], evidence="chorismate is converted to isochorismate by EntC"),
        _row(["isochorismate"], ["2,3-diDHB", "pyruvate"], ["EntA"]),
        _row(["isochorismate"], ["2,3-diDHB", "pyruvate"], ["EntA"]),          # duplicate
        _row(["2,3-DHB"], ["2,3-DHB-AMP"], []),                                  # ATP missing, enzyme missing
        _row(["pyruvate", "NADH"], ["lactate"], ["LDH"]),                       # off-pathway
        _row(["chorismate"], ["2,3-DHB"], []),                                   # summary (lumped)
        _row(["isochorismate"], ["2,3-diDHB"], ["EntB"]),                       # partial
        _row(["glycine"], ["heme"], []),                                         # unsupported
        _row(["menaquinone-7"], ["DHNA-CoA"], [], state="none"),                 # unsupported, species unknown
        _row(["X"], ["Y"], []),                                                  # external by lineage
    ]},
}
LINEAGE = [{"row_index": i, "support_class": "target_paper_supported", "source_resolutions": []}
           for i in range(10)]
LINEAGE[9] = {"row_index": 9, "support_class": "external_rag_supported", "source_resolutions": ["external"]}


@pytest.fixture(scope="module")
def scored():
    return R.score_leg(CASE, PAYLOAD, LINEAGE, PAPER)


def test_row_classes_follow_the_documented_precedence(scored):
    classes = [r["class"] for r in scored["rows"]]
    assert classes == [R.TP, R.TP, R.TP, R.TP, R.OFF, R.SUMMARY, R.PARTIAL, R.UNSUP, R.UNSUP, R.EXT]
    assert scored["duplicate_rows_collapsed"] == 1
    assert scored["rows"][2]["duplicate_of"] == 1


def test_two_tables_keep_separate_denominators(scored):
    t1, t2 = scored["table1_paper_extraction"], scored["table2_final_support"]
    # Table 1 excludes the externally attributed row; the duplicate is collapsed.
    assert t1["rows_claimed_as_paper_extraction"] == 8
    assert t1["true_positive_rows"] == 3
    assert t1["precision"] == round(3 / 8, 4)
    assert t1["recall"] == 1.0
    assert t1["f1"] == round(2 * (3 / 8) / (3 / 8 + 1), 4)
    assert t2["retained_unique"] == 9
    assert t2["unsupported"] == 2 and t2["unsupported_rate"] == round(2 / 9, 4)
    assert t2["external_rag_supported"] == 1 and t2["paper_stated_off_pathway"] == 1
    assert t2["paper_stated_summary"] == 1 and t2["partial_match"] == 1


def test_enzyme_is_scored_apart_from_the_reaction(scored):
    ea = scored["entity_accuracy"]
    assert ea["enzyme_verdicts"] == {"correct": 1, "wrong": 1, "missing": 1}
    assert ea["enzyme_accuracy"] == round(1 / 3, 4)
    assert scored["rows"][1]["class"] == R.TP  # the wrong enzyme did not fail the chemistry


def test_minor_cofactor_gap_is_not_a_missing_reaction(scored):
    assert scored["minor_cofactors_missing_on_matched_reactions"] == [
        {"gold_id": "R3", "row_index": 3, "missing": ["ATP"]}]
    assert "R3" in scored["table1_paper_extraction"]["gold_reactions_matched"]


def test_species_is_binomial_and_unknown_is_not_wrong(scored):
    assert scored["entity_accuracy"]["species_verdicts"] == {"correct": 8, "unknown": 1}
    assert R.species_key("Lactococcus lactis subsp. lactis KF147") == R.species_key("Lactococcus lactis")


def test_evidence_support_is_verbatim_quote_presence(scored):
    ev = scored["evidence"]
    assert scored["rows"][0]["evidence"] == "verbatim_in_paper"
    assert ev["verdicts"]["missing"] == 8
    assert R.evidence_verdict({"evidence": "x" * 2500}, "abc") == "dump_only"
    assert R.evidence_verdict({"evidence": "a sentence that is not in the paper at all"}, "zzz") == "not_in_paper"


def test_substrate_and_product_accuracy_use_the_aligned_row(scored):
    ea = scored["entity_accuracy"]
    assert (ea["substrate_hits"], ea["substrate_total"]) == (3, 3)
    assert (ea["product_hits"], ea["product_total"]) == (4, 4)


def test_old_lenient_matcher_is_reported_beside_v2():
    case = CaseV2(paper_id="P", complete=True, species="", scope="", source={},
                  reactions=(_rx("R7", ["DHNA-CoA"], ["DHNA"]),), off_pathway=())
    res = R.score_leg(case, {"processes": {"reactions": [_row(["DHNA"], ["DHNA"])]}}, [], None)
    old = res["old_lenient_matcher_same_rows"]
    assert old["gold_reactions_matched"] == ["R7"]          # congener credit by containment
    assert res["table1_paper_extraction"]["gold_reactions_matched"] == []
    assert old["rows_where_old_and_v2_disagree"][0]["v2_class"] == R.UNSUP


def test_id_correctness_three_way_plus_not_evaluable():
    ref = {"independent": {("kegg", "C00021"): {"S-Adenosyl-L-homocysteine", "SAH"},
                           ("uniprot", "P0AEJ2"): {"Isochorismate synthase EntC", "entC"}}}
    payload = {"entities": {
        "compounds": [
            {"name": "SAM", "mapped_ids": {"kegg": "C00021"}},                 # claimed SAM, ID is SAH
            {"name": "mystery", "mapped_ids": {}},                              # unresolved
            {"name": "orphan", "mapped_ids": {"kegg": "C99999"}},               # no reference record
        ],
        "proteins": [{"name": "EntC", "mapped_ids": {"uniprot": "P0AEJ2"}}],  # correct via gene name
    }}
    res = R.id_correctness(payload, ref)
    assert res["counts"] == {"wrong_identity": 1, "unresolved_or_novel": 1,
                             "not_evaluable_no_reference": 1, "correct_identity": 1}
    assert res["wrong"][0]["name"] == "SAM"


def test_id_claim_side_never_reads_synonyms():
    """A synonyms list written from the record under test would make the check circular."""
    ref = {"independent": {("kegg", "C00021"): {"S-Adenosyl-L-homocysteine"}}}
    row = {"name": "SAM", "synonyms": ["S-Adenosyl-L-homocysteine"], "mapped_ids": {"kegg": "C00021"}}
    assert R.id_correctness({"entities": {"compounds": [row]}}, ref)["counts"] == {"wrong_identity": 1}


# Regression fixtures for review finding B1 (ORCH-742c repair 1). NEW ACCEPTANCE TESTS:
# the evaluator is new on this branch (G9: no base-SHA failure applies).
SAM_ON_AKG = {"entities": {"compounds": [
    {"name": "S-adenosyl-L-methionine", "mapped_ids": {"kegg": "C00019", "pathbank_compound_id": "134"}}]}}


def test_one_correct_id_does_not_mask_a_wrong_pathwhiz_id():
    """Each identifier is judged against its own record, never a union across them."""
    ref = {"independent": {("kegg", "C00019"): {"S-Adenosyl-L-methionine"},
                           ("pathbank_compound_id", "134"): {"Oxoglutaric acid"}}}
    res = R.id_correctness(SAM_ON_AKG, ref)
    assert res["counts"] == {"wrong_identity": 1}
    w = res["wrong"][0]
    assert w["contradicting_ids"] == ["pathbank_compound_id:134"] and w["independent"] is True
    assert {p["id"]: p["verdict"] for p in w["per_id"]} == {
        "kegg:C00019": "correct", "pathbank_compound_id:134": "wrong"}


def test_claimed_sam_on_pathwhiz_134_is_wrong_against_the_committed_tables():
    """The reviewer's fixture on the committed data: PathWhiz compound 134 is
    Oxoglutaric acid in data/pathwhiz_id_db.json, read under the payload's own key."""
    ref = R.build_reference_index(ROOT)
    assert "Oxoglutaric acid" in ref["independent"][("pathbank_compound_id", "134")]
    res = R.id_correctness(SAM_ON_AKG, ref)
    assert res["counts"] == {"wrong_identity": 1}
    assert res["wrong"][0]["contradicting_ids"] == ["pathbank_compound_id:134"]
    assert res["wrong"][0]["independent"] is True
    # The right molecule on the same identifiers stays correct, through the KEGG record
    # that PathWhiz record 134 declares for itself.
    akg = {"entities": {"compounds": [{"name": "alpha-ketoglutarate",
                                       "mapped_ids": {"kegg": "C00026", "pathbank_compound_id": "134"}}]}}
    assert R.id_correctness(akg, ref)["counts"] == {"correct_identity": 1}


def test_local_pathwhiz_ids_that_collide_with_pathbank_are_not_an_authority():
    """PathWhiz-table records with no external identifier are local sample-.pwml ids:
    compound 14 is 'ATP' there, PathBank compound 14 is Iodotyrosine; protein 13 is
    'diglyceride lipase' there, PathBank protein 13 is ACAT2. Using them would convict
    correct mappings."""
    ref = R.build_reference_index(ROOT)
    assert ("pathbank_compound_id", "14") not in ref["independent"]
    assert ("pathbank_protein_id", "13") not in ref["independent"]
    acat2 = {"entities": {"proteins": [{"name": "ACAT2", "mapped_ids": {"pathbank_protein_id": "13"}}]}}
    assert R.id_correctness(acat2, ref)["counts"] != {"wrong_identity": 1}


def test_a_verdict_resting_only_on_the_mapper_cache_is_marked_non_independent():
    ref = {"mapper_cache": {("pathbank_compound_id", "9"): {"Glycine"}}}
    ok = {"entities": {"compounds": [{"name": "glycine", "mapped_ids": {"pathbank_compound_id": "9"}}]}}
    bad = {"entities": {"compounds": [{"name": "serine", "mapped_ids": {"pathbank_compound_id": "9"}}]}}
    assert R.id_correctness(ok, ref)["counts_resting_on_mapper_cache_only"] == {"correct_identity": 1}
    res = R.id_correctness(bad, ref)
    assert res["counts"] == {"wrong_identity": 1} and res["wrong"][0]["independent"] is False


def test_heptanoyl_hepp_is_a_name_conflict_not_a_true_positive():
    """NEW ACCEPTANCE TEST, regression fixture for review finding B2 (ORCH-742c repair 1),
    on the committed PMC12657337 supplement case."""
    from t2pw.bench.eval_v2.gold import load_supplement

    case = load_supplement()["PMC12657337"]
    bad = _row(["FPP"], ["heptanoyl pyrophosphate (HepPP)"], ["HepPPS"])
    good = _row(["FPP"], ["heptaprenyl diphosphate (HepPP)"], ["HepPPS"])
    res = R.score_leg(case, {"processes": {"reactions": [bad]}}, [], None)
    assert res["rows"][0]["class"] == R.CONFLICT and res["rows"][0]["gold_ids"] == ["R1"]
    assert res["table1_paper_extraction"]["true_positive_rows"] == 0
    assert res["table2_final_support"]["name_conflict"] == 1
    assert res["table2_final_support"]["unsupported_rate_upper_bound"] == 1.0
    assert res["name_conflicts_for_hand_review"][0]["conflicting_names"] == ["heptanoyl pyrophosphate (HepPP)"]
    ctrl = R.score_leg(case, {"processes": {"reactions": [good]}}, [], None)
    assert ctrl["rows"][0]["class"] == R.TP and ctrl["rows"][0]["gold_ids"] == ["R1"]
    assert ctrl["name_conflicts_for_hand_review"] == []


def test_leakage_audit_flags_only_ungated_rag_rows():
    stage1 = {"processes": {"reactions": [_row(["a"], ["b"])]}}
    stage2 = {"processes": {"reactions": [
        _row(["a"], ["b"]),                                      # in Stage 1
        _row(["c"], ["d"]),                                      # matches a REJECTED candidate
        _row(["e"], ["f"]),                                      # matches an ACCEPTED candidate
        _row(["g"], ["h"]),                                      # no RAG correspondence at all
    ]}}
    admission = {"rejected": [{"inputs": ["c"], "outputs": ["d"], "evidence": {"span": "c goes to d", "source_id": "PMC999"}}],
                 "accepted": [{"inputs": ["e"], "outputs": ["f"], "evidence": {"span": "e goes to f"}}]}
    res = R.leakage_audit(stage1, stage2, admission, "a becomes b")
    assert res["rows_absent_from_stage1"] == 3
    got = res["ungated_rag_influenced_candidates"]
    assert [c["inputs"] for c in got] == [["c"]]
    assert got[0]["matched_admission_verdicts"] == ["rejected"]
    assert "admission_match" in got[0]["criteria"] and "participant_only_in_rag" in got[0]["criteria"]
    assert got[0]["external_evidence"] is True  # PMC999 is neither the seed nor the target


def test_leakage_audit_separates_seed_paper_retrieval_from_external():
    stage1 = {"processes": {"reactions": []}}
    stage2 = {"processes": {"reactions": [_row(["c"], ["d"])]}}
    admission = {"rejected": [{"inputs": ["c"], "outputs": ["d"],
                               "evidence": {"span": "c goes to d", "source_id": "seed_paper"}}]}
    got = R.leakage_audit(stage1, stage2, admission, "c goes to d", target_paper="PMC1")
    assert got["ungated_rag_influenced_candidates"][0]["external_evidence"] is False


def test_archived_failure_split_a_b_c_ok_and_d():
    case = CaseV2(paper_id="P", complete=True, species="", scope="", source={}, off_pathway=(),
                  reactions=(_rx("R1", ["a"], ["b"]), _rx("R2", ["c"], ["d"]), _rx("R3", ["e"], ["f"]),
                             _rx("R4", ["g"], ["h"]), _rx("R5", ["i"], ["j"])))
    stage1 = {"processes": {"reactions": [_row(["i"], ["j"])]}}
    ev = lambda span: {"span": span, "source_id": "PMC123", "source_uri": ""}  # noqa: E731
    admission = {
        "accepted": [{"inputs": ["e"], "outputs": ["f"], "evidence": ev("e is converted to f")},
                     {"inputs": ["x"], "outputs": ["y"], "evidence": ev("nothing relevant here")}],
        "rejected": [{"inputs": ["c"], "outputs": ["d"], "evidence": ev("c to d")},
                     {"inputs": ["q"], "outputs": ["r"], "evidence": ev("we saw g and h together")}],
    }
    res = archived_failure_split(case, admission, stage1)
    assert res["per_gold_reaction"] == {"R1": "A_not_retrieved", "R2": "C_proposed_but_rejected",
                                        "R3": "ok_admitted", "R4": "B_retrieved_not_used",
                                        "R5": "already_in_stage1"}
    assert res["D_accepted_unsupported"] == 1          # x -> y accepted on a span naming neither
    assert res["accepted_external_biology"] == 1
    assert res["chunk_provenance"]["valid_source_reference=True"] == 4


def test_valid_source_reference():
    assert valid_source_reference("PMC12312563", "")
    assert valid_source_reference("seed_paper", "")
    assert valid_source_reference("", "https://doi.org/10.1/x")
    assert not valid_source_reference("chunk-7", "")


def test_rank_metrics():
    m = rank_metrics([[False, True, False], [True], [False] * 6])
    assert m["recall_at_1"] == round(1 / 3, 4) and m["recall_at_3"] == round(2 / 3, 4)
    assert m["mrr"] == round((0.5 + 1.0 + 0.0) / 3, 4)
    assert m["precision_at_5"] == round((1 / 5 + 1 / 5 + 0) / 3, 4)


def test_repro_completeness_on_a_synthetic_trace(tmp_path):
    leg = tmp_path / "runs_x/papers/PMC1/strict"
    leg.mkdir(parents=True)
    rows = [{"kind": "model_attempt", "stage": "s1", "attempt": 1, "model": "m", "finish_reason": "stop",
             "request_hash": "sha256:a", "response_hash": ""},
            {"kind": "model_attempt", "stage": "s1", "attempt": 2, "model": "m", "finish_reason": "length",
             "request_hash": "sha256:a", "response_hash": ""}]
    (leg / "LEG_TRACE.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    res = completeness(tmp_path, ["runs_x/papers/PMC1/strict/LEG_TRACE.jsonl"])
    by = {f["field"]: f for f in res["fields"]}
    assert by["requested_model"]["status"] == "recorded"
    assert by["response_hash"]["status"] == "not_recorded"
    assert by["seed"]["status"] == "not_recorded"
    assert res["calls_with_retries"] == 1


def test_committed_traces_never_carry_a_response_hash():
    """Characterisation of committed artifacts (a finding, not a fix): every committed
    model_attempt row has an EMPTY response_hash."""
    paths = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    res = completeness(ROOT, paths)
    if not res["model_attempt_records"]:
        pytest.skip("no committed LEG_TRACE.jsonl in this checkout")
    by = {f["field"]: f for f in res["fields"]}
    assert by["response_hash"]["model_attempts_with_field"] == 0
    assert by["request_hash"]["status"] == "recorded"


def test_no_production_module_imports_evaluation_v2():
    offenders = []
    for py in (ROOT / "src" / "t2pw").rglob("*.py"):
        if "eval_v2" in py.parts:
            continue
        if re.search(r"eval_v2", py.read_text(encoding="utf-8", errors="replace")):
            offenders.append(str(py))
    assert offenders == []
