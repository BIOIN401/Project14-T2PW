"""Benchmark scorer over scoped views (synthetic payloads; the one DB reader is injected)."""

import json
from pathlib import Path

import pytest

from t2pw.bench.eval_v2 import reactions as R
from t2pw.bench.eval_v2.gold import load_supplement
from t2pw.vision.bench import (TYPES, build_full_gold, gold_from_pathbank, gold_from_supplement, run_benchmark,
                               score_view)
from t2pw.vision.payload import build_big_payload, merge_extension
from t2pw.vision.view import DEFAULT, FULL, PAPER_ONLY

ROOT = Path(__file__).resolve().parents[1]
DB = {"kind": "database", "source": "pathbank:PW1", "quote": None, "stage": "extend_db", "note": ""}
OP = {"kind": "other_paper", "source": "PMC9", "quote": "q", "stage": "extend_lit", "note": ""}


def comp(name, chebi, **kw):
    return {"name": name, "mapped_ids": {"chebi": f"CHEBI:{chebi}"}, **kw}


def rx(name, ins, outs, enz=(), **kw):
    return {"name": name, "inputs": ins, "outputs": outs, "enzymes": [{"entity": e} for e in enz],
            "biological_state": "cyt", **kw}


def base():
    """A -> B -> C stated by the paper; B -> Z only by a retrieved passage (other_paper)."""
    return {
        "entities": {
            "species": [{"name": "Escherichia coli"}],
            "subcellular_locations": [{"name": "cytoplasm", "evidence": "in the cytoplasm"}],
            "compounds": [comp(n, i, evidence=f"{n} is made") for n, i in (("A", 1), ("B", 2), ("C", 3))]
                         + [comp("Z", 26)],
            "proteins": [{"name": "EnzA", "uniprot": "P1", "evidence": "EnzA"}, {"name": "EnzB", "evidence": "EnzB"}],
            "protein_complexes": [],
        },
        "biological_states": [{"name": "cyt", "species": "Escherichia coli", "subcellular_location": "cytoplasm"}],
        "element_locations": {"compound_locations": []},
        "processes": {
            "reactions": [
                rx("r1", ["A"], ["B"], ["EnzA"], evidence="A to B"),
                rx("r2", ["B"], ["C"], ["EnzB"], evidence="B to C"),
                rx("rag", ["B"], ["Z"], [], evidence="passage", rag_provenance={"source_id": "PMC7", "source_type": "paper"}),
            ],
            "transports": [], "interactions": [],
        },
    }


def with_extension(b=None):
    """+ database rows C -> D (hop 1) and D -> E (hop 2); only C -> D is in the curated pathway."""
    big = build_big_payload(b or base())
    rows = {"entities.compounds": [comp("D", 4), comp("E", 5)], "entities.proteins": [{"name": "EnzC", "uniprot": "P3"}],
            "processes.reactions": [rx("r3", ["C"], ["D"], ["EnzC"]), rx("r4", ["D"], ["E"], [])]}
    prov = {"entities.compounds": [[DB], [DB]], "entities.proteins": [[DB]], "processes.reactions": [[DB], [OP]]}
    return merge_extension(big, rows, prov)


def tm(name, chebi=None, **kw):
    return {"name": name, "ids": [["chebi", str(chebi)]] if chebi else [], **kw}


def rgold(rid, ins, outs, enz=None, **kw):
    return {"id": rid, "label": rid, "inputs": [{"aliases": [], "role": "defining", **t} for t in ins],
            "outputs": [{"aliases": [], "role": "defining", **t} for t in outs],
            "enzyme": enz and {"aliases": [], "ids": [], "role": "defining", **enz}, "reversible": False, **kw}


def paper_gold(**kw):
    return {"id": "P", "kind": "paper", "complete": {"reactions": True}, "off_scope_reactions": [],
            "reactions": [rgold("R1", [tm("A")], [tm("B")], tm("EnzA")), rgold("R2", [tm("B")], [tm("C")], tm("EnzB")),
                          rgold("R3", [tm("C")], [tm("D")], tm("EnzC"), in_paper=False)],
            "enzymes": [], "transports": [], "interactions": [], "compartments": [], **kw}


def test_paper_only_ignores_gold_outside_the_paper_and_the_view_drops_non_paper_rows():
    big = with_extension()
    s = score_view(big, PAPER_ONLY, paper_gold())["types"]["reactions"]
    # R3 is flagged outside the paper: not in the denominator; the rag row, r3 and r4 are not in the view
    assert (s["gold"], s["recalled"], s["predicted"], s["correct"]) == (2, 2, 2, 2)
    assert s["recall"] == 1.0 and s["precision"] == 1.0
    f = score_view(big, FULL, paper_gold())["types"]["reactions"]
    assert (f["gold"], f["recalled"], f["predicted"], f["correct"]) == (3, 3, 5, 3)  # r3 now counts, rag/r4 are wrong
    assert f["recovered_by_extension"] == 1 and f["recalled_base_only"] == 2


def test_paper_precision_needs_a_complete_flag():
    g = paper_gold(complete={"reactions": False})
    s = score_view(with_extension(), PAPER_ONLY, g)["types"]["reactions"]
    assert s["recall"] == 1.0 and s["precision"] is None


def test_default_equals_base_when_there_is_no_extension():
    big = build_big_payload(base())
    a, b = score_view(big, DEFAULT, paper_gold()), score_view(big, FULL, paper_gold())
    assert a["types"] == b["types"]
    assert a["extension"]["all"] == {"rows": 0, "correct": 0, "precision": None}


def extract():
    def part(side, name, chebi=None, **x):
        return {"side": side, "name": name, "element_type": "Compound", "identifiers": {"chebi_id": f"CHEBI:{chebi}"} if chebi else {}, **x}

    def reaction(acc, ins, outs, enz):
        return {"process_type": "Reaction", "accession": acc, "direction": ">", "biological_state": {"location": "Cytosol"},
                "participants": [part("left", n, i) for n, i in ins] + [part("right", n, i) for n, i in outs],
                "catalysts": [{"name": f"{enz} complex", "components": [{"name": enz, "gene_name": enz.lower(), "uniprot": {"EnzA": "P1", "EnzB": "P2", "EnzC": "P3"}[enz]}]}]
                if enz else []}

    return {"accession": "PW1", "pathway_id": 1, "name": "toy", "species": {"name": "Escherichia coli"}, "processes": [
        reaction("PW_R1", [("Compound A", 1)], [("Compound B", 2)], "EnzA"),
        reaction("PW_R2", [("Compound B", 2)], [("Compound C", 3)], "EnzB"),
        reaction("PW_R3", [("Compound C", 3)], [("Compound D", 4)], "EnzC"),
        {"process_type": "Transport", "accession": "PW_T1", "biological_state": {"location": "Cytosol"},
         "participants": [part("moved", "Compound C", 3, from_state={"location": "Cytosol"}, to_state={"location": "Extracellular Space"})],
         "catalysts": []},
        {"process_type": "Interaction", "accession": "PW_I1", "interaction_type": "Inhibition",
         "biological_state": {"location": "Cytosol"},
         "participants": [part("left", "Compound D", 4), part("right", "EnzA", None, element_type="Protein")], "catalysts": []}]}


def test_full_gold_comes_from_the_injected_extractor_and_covers_every_type():
    seen = []
    g = build_full_gold("PW1", lambda acc: seen.append(acc) or extract())
    assert seen == ["PW1"] and g["kind"] == "full" and all(g["complete"].values())
    assert [len(g[t]) for t in TYPES] == [3, 3, 1, 1, 2]  # 3 reactions, EnzA/B/C, one transport, one interaction, Cytosol + Extracellular
    assert g["interactions"][0]["relationship"] == "inhibition"
    assert g == gold_from_pathbank(extract(), "PW1")


def test_full_view_recall_counts_curated_elements_recovered_through_extension_rows():
    gold = build_full_gold("PW1", lambda _: extract())
    big = with_extension()
    full = score_view(big, FULL, gold)
    r = full["types"]["reactions"]
    # names differ ("Compound A" vs "A"): the match is by the grounded chebi ids
    assert (r["gold"], r["recalled"], r["recovered_by_extension"], r["recalled_base_only"]) == (3, 3, 1, 2)
    e = full["extension"]["reactions"]  # r3 curated, r4 not
    assert (e["rows"], e["correct"], e["precision"]) == (2, 1, 0.5)
    # extension enzymes: EnzC (curated, found by its UniProt id and name); r4 names none
    assert full["extension"]["enzymes"] == {"rows": 1, "correct": 1, "precision": 1.0}
    assert full["extension"]["all"] == {"rows": 3, "correct": 2, "precision": round(2 / 3, 4)}
    base_only = score_view(build_big_payload(base()), FULL, gold)["types"]["reactions"]
    assert base_only["recalled"] == 2 and base_only["recovered_by_extension"] == 0
    default = score_view(big, DEFAULT, gold)["types"]["reactions"]  # hop <= 1: r3 yes, r4 (hop 2) no
    assert (default["recalled"], default["predicted"]) == (3, 4)


def test_full_view_scores_enzymes_transports_interactions_compartments():
    b = base()
    b["processes"]["transports"] = [{"name": "export", "cargo": "C", "from_biological_state": "cyt", "to_biological_state": "out"}]
    b["processes"]["interactions"] = [{"name": "i", "entity_1": "EnzA", "entity_2": "D", "relationship": "inhibits",
                                       "biological_state": "cyt", "evidence": "x"}]
    b["entities"]["compounds"].append(comp("D", 4))
    b["biological_states"].append({"name": "out", "subcellular_location": "extracellular space"})
    b["entities"]["subcellular_locations"].append({"name": "extracellular space"})
    t = score_view(build_big_payload(b), FULL, build_full_gold("PW1", lambda _: extract()))["types"]
    assert (t["transports"]["recalled"], t["interactions"]["recalled"], t["compartments"]["recalled"]) == (1, 1, 2)
    assert t["enzymes"]["recalled"] == 2 and t["enzymes"]["gold"] == 3  # EnzA, EnzB found (names/aliases); EnzC not in base
    assert t["compartments"]["precision"] == 1.0


def test_wrong_relationship_or_wrong_compartment_does_not_match():
    b = base()
    b["processes"]["transports"] = [{"name": "t", "cargo": "C", "from_biological_state": "cyt", "to_biological_state": "cyt"}]
    b["processes"]["interactions"] = [{"name": "i", "entity_1": "EnzA", "entity_2": "D", "relationship": "activates",
                                       "biological_state": "cyt", "evidence": "x"}]
    b["entities"]["compounds"].append(comp("D", 4))
    t = score_view(build_big_payload(b), FULL, gold_from_pathbank(extract(), "PW1"))["types"]
    assert t["transports"]["recalled"] == 0 and t["interactions"]["recalled"] == 0


def test_provenance_breakdown_partitions_the_predicted_elements():
    gold = gold_from_pathbank(extract(), "PW1")
    s = score_view(with_extension(), FULL, gold)
    for t in TYPES:
        prim = s["provenance"][t]["primary"]
        assert sum(v["rows"] for v in prim.values()) == s["types"][t]["predicted"]
        assert sum(v["correct"] for v in prim.values()) == s["types"][t]["correct"]
        assert all(v["correct"] <= v["rows"] for v in prim.values())
        assert sum(v["rows"] for v in s["provenance"][t]["any"].values()) >= s["types"][t]["predicted"]
    rxp = s["provenance"]["reactions"]["primary"]
    # r1, r2 paper (correct); rag row other_paper (wrong); r3 database (correct); r4 other_paper record only (wrong)
    assert rxp["paper"] == {"rows": 2, "correct": 2} and rxp["database"] == {"rows": 1, "correct": 1}
    assert rxp["other_paper"] == {"rows": 2, "correct": 0}


def test_run_benchmark_keeps_kinds_apart_and_reports_cases_without_runs():
    paper, full = paper_gold(), gold_from_pathbank(extract(), "PW1")
    other = {**paper, "id": "Q"}
    rep = run_benchmark([paper, full, other], {"P": {"run1": with_extension(), "run2": with_extension()},
                                               "PW1": {"run1": with_extension()}}, FULL)
    assert rep["cases_without_runs"] == ["Q"] and len(rep["legs"]) == 3
    assert set(rep["aggregate"]) == {"paper", "full"} and rep["aggregate"]["paper"]["legs"] == 2
    assert rep["aggregate"]["paper"]["types"]["reactions"]["recall_num_den"] == [6, 6]
    assert rep["aggregate"]["full"]["types"]["reactions"]["recall_num_den"] == [3, 3]


def test_dedupe_matches_eval_v2_duplicate_collapse():
    b = base()
    b["processes"]["reactions"].append(rx("r1 again", ["A"], ["B"], [], evidence="A to B"))
    s = score_view(build_big_payload(b), PAPER_ONLY, paper_gold())["types"]["reactions"]
    assert s["predicted"] == 2


def test_default_view_numbers_equal_eval_v2_score_leg_on_a_real_gold_case():
    """With no extension DEFAULT == base, so reaction recall/precision must equal the existing scorer's."""
    cases = load_supplement()
    case = cases["PMC12421875"]
    raw = json.loads((ROOT / "src/t2pw/bench/eval_v2/gold_v2_supported_reactions.json").read_text(encoding="utf-8"))["cases"]["PMC12421875"]
    gold = gold_from_supplement(raw, "PMC12421875")
    assert len(gold["reactions"]) == len(case.reactions) == 9 and gold["complete"]["reactions"] is True
    # a payload: the first 4 gold reactions by their defining names, plus one invented row
    rows = [{"name": g.id, "inputs": [t.name for t in g.defining("inputs")], "outputs": [t.name for t in g.defining("outputs")],
             "enzymes": [], "evidence": g.quote} for g in case.reactions[:4]]
    rows.append({"name": "bogus", "inputs": ["not a thing"], "outputs": ["nor this"], "enzymes": [], "evidence": "e"})
    p = {"entities": {"compounds": [], "proteins": []}, "processes": {"reactions": rows, "transports": [], "interactions": []}}
    big = build_big_payload(p)
    mine = score_view(big, DEFAULT, gold)["types"]["reactions"]
    leg = R.score_leg(case, p, [], None)["table1_paper_extraction"]
    assert (mine["recalled"], mine["gold"]) == (len(leg["gold_reactions_matched"]), leg["gold_reactions"])
    assert (mine["correct"], mine["predicted"]) == (leg["true_positive_rows"], leg["rows_claimed_as_paper_extraction"])
    assert mine["gold"] == 9 and mine["recalled"] >= 4 and mine["predicted"] <= 5
    assert mine["recall"] == leg["recall"] and mine["precision"] == leg["precision"]


def test_gold_builders_over_the_committed_gold_files():
    sup = json.loads((ROOT / "src/t2pw/bench/eval_v2/gold_v2_supported_reactions.json").read_text(encoding="utf-8"))["cases"]
    for pid, c in load_supplement().items():
        g = gold_from_supplement(sup[pid], pid)
        assert len(g["reactions"]) == len(c.reactions) and len(g["off_scope_reactions"]) == len(c.off_pathway) + len(c.summaries)
        assert g["complete"]["reactions"] is c.complete
        assert g["enzymes"] and all(e["name"] for e in g["enzymes"])
    ex = ROOT / "docs/evaluation_v2/pathbank_gold/PW701720__extract.json"
    g = gold_from_pathbank(json.loads(ex.read_text(encoding="utf-8")), "PW701720")
    assert (len(g["reactions"]), len(g["interactions"]), len(g["transports"])) == (1, 7, 0) and not g["skipped_processes"]


def test_benchmark_design_file_lists_existing_gold_and_runs_nothing_unlisted():
    d = json.loads((ROOT / "docs/vision/BENCHMARK.json").read_text(encoding="utf-8"))
    assert {"sets", "metrics", "rules"} <= set(d)
    paper = d["sets"]["paper"]
    sup = json.loads((ROOT / paper["gold_file"]).read_text(encoding="utf-8"))["cases"]
    assert all(c["case_id"] in sup for c in paper["cases"])
    assert [c["case_id"] for c in paper["cases"] if c["precision"]] == [k for k, v in sup.items() if v["supported_reactions_complete"]]
    full = d["sets"]["full"]["cases"]
    assert 3 <= len(full) <= 6 and all(c["pw_id"].startswith("PW") for c in full)
    assert d["metrics"]["types"] == list(TYPES)
