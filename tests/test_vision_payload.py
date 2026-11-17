"""Big payload: build, keys, provenance mapping, save/load, merge, hops (synthetic payloads only)."""

import copy
import json

from t2pw.vision.payload import (big_from_run, build_big_payload, element_keys, load_big_payload,
                                 merge_extension, save_big_payload)


def lin(origin, stage="paper_extraction", sources=(), reason="r"):
    return {"stage": stage, "origin": origin, "support": "direct", "paper_explicit": "explicit",
            "reason": reason, "review_required": False, "uncertainty": "", "sources": list(sources)}


def payload():
    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "provenance_lineage": [lin("paper_stated")]}],
            "subcellular_locations": [{"name": "cytoplasm"}],
            "compounds": [
                {"name": "A", "synonyms": ["alpha-A"], "mapped_ids": {"chebi": "CHEBI:1"},
                 "provenance_lineage": [lin("paper_stated"), lin("database_grounded", "identifier_mapping",
                                                                 [{"source_id": "CHEBI:1", "source_type": "chebi"}])]},
                {"name": "B", "provenance_lineage": [lin("inferred", "gap_resolution")]},
                {"name": "C", "provenance_lineage": [lin("unresolved", "identifier_mapping", reason="novel")]},
                {"name": "D", "evidence": "D is made"},
                {"name": "E"},
            ],
            "proteins": [{"name": "EnzA", "uniprot": "P1", "provenance_lineage": [lin("audit_modified", "audit_repair")]}],
            "protein_complexes": [],
        },
        "biological_states": [{"name": "E. coli cytoplasm", "species": "Escherichia coli",
                               "subcellular_location": "cytoplasm"}],
        "element_locations": {"compound_locations": [{"compound": "A", "biological_state": "E. coli cytoplasm"}]},
        "processes": {
            "reactions": [
                {"name": "r1", "inputs": ["A"], "outputs": ["B"], "enzymes": [{"entity": "EnzA"}],
                 "evidence": "A to B", "biological_state": "E. coli cytoplasm",
                 "provenance_lineage": [lin("paper_stated")]},
                {"name": "r1 again", "inputs": ["A"], "outputs": ["B"], "enzymes": [],
                 "evidence": "A to B again", "biological_state": "E. coli cytoplasm"},
                {"name": "rag", "inputs": ["B"], "outputs": ["C"], "enzymes": [], "evidence": "passage",
                 "rag_provenance": {"source_id": "seed_paper", "source_type": "paper"},
                 "provenance_lineage": [lin("rag_literature", "rag_admission", [
                     {"source_id": "seed_paper", "source_type": "paper"},
                     {"source_id": "PMC9", "source_type": "paper", "uri": "https://doi.org/10.1/x"}])]},
                {"name": "other", "inputs": ["C"], "outputs": ["D"], "enzymes": [], "evidence": "x",
                 "rag_provenance": {"source_id": "PMC7", "source_type": "paper", "source_title": "T"}},
            ],
            "transports": [],
            "interactions": [],
        },
        "reaction_order": ["r1"],
    }


def by_name(big, section, name):
    rows = big["base"]
    for part in section.split("."):
        rows = rows[part]
    keys = [k for k in element_keys(big) if k.startswith(section + ":")]
    for row, key in zip(rows, keys):
        if row.get("name") == name:
            return key
    raise KeyError(name)


def kinds(big, section, name):
    return [r["kind"] for r in big["provenance"][by_name(big, section, name)]]


def test_base_is_same_content_and_input_untouched():
    p = payload()
    before = copy.deepcopy(p)
    big = build_big_payload(p)
    assert p == before
    assert big["base"] == p and big["base"] is not p
    assert json.dumps(big["base"]) == json.dumps(p)
    assert big["schema"] == "t2pw.big_payload/1" and big["extension"] == {}
    assert set(big["hops"].values()) == {0}
    big["base"]["entities"]["compounds"].append({"name": "Z"})
    assert p == before


def test_keys_are_unique_deterministic_and_suffix_duplicates():
    big = build_big_payload(payload())
    keys = element_keys(big)
    assert len(keys) == len(set(keys)) == len(big["provenance"])
    assert "entities.compounds:a" in keys
    rx = [k for k in keys if k.startswith("processes.reactions:")]
    assert rx[0] == "processes.reactions:a>b@ecolicytoplasm" and rx[1] == rx[0] + "#2"
    assert element_keys(build_big_payload(payload())) == keys


def test_keys_stable_across_save_load(tmp_path):
    big = build_big_payload(payload(), paper_meta={"pmcid": "PMC1"})
    save_big_payload(big, tmp_path / "big.json")
    again = load_big_payload(tmp_path / "big.json")
    assert again == big
    assert element_keys(again) == element_keys(big)
    assert sorted(again["provenance"]) == sorted(element_keys(again))


def test_big_from_run(tmp_path):
    (tmp_path / "final_mapped.json").write_text(json.dumps(payload()), encoding="utf-8")
    assert big_from_run(tmp_path)["base"] == payload()


def test_provenance_mapping_per_origin():
    big = build_big_payload(payload(), paper_meta={"pmcid": "PMC1"})
    assert kinds(big, "entities.compounds", "A") == ["paper", "database"]
    db = big["provenance"][by_name(big, "entities.compounds", "A")][1]
    assert db["source"] == "CHEBI:1" and big["sources"]["CHEBI:1"]["db"] == "chebi"
    assert kinds(big, "entities.compounds", "B") == ["inferred"]
    c = big["provenance"][by_name(big, "entities.compounds", "C")]
    assert [r["kind"] for r in c] == ["unclassified"] and c[0]["note"].startswith("unresolved")
    d = big["provenance"][by_name(big, "entities.compounds", "D")]
    assert d == [{"kind": "paper", "source": "paper", "quote": "D is made", "stage": None,
                  "note": "no lineage; evidence quote"}]
    assert kinds(big, "entities.compounds", "E") == ["unclassified"]
    assert kinds(big, "entities.proteins", "EnzA") == ["curation"]
    r1 = big["provenance"][by_name(big, "processes.reactions", "r1")][0]
    assert r1["kind"] == "paper" and r1["quote"] == "A to B" and r1["source"] == "paper"


def test_rag_seed_tag_is_never_paper():
    big = build_big_payload(payload())
    rag = big["provenance"][by_name(big, "processes.reactions", "rag")]
    assert [r["kind"] for r in rag] == ["other_paper"]  # seed-tagged rag_provenance adds nothing
    assert rag[0]["source"] == "seed_paper" and "PMC9" in rag[0]["note"] and rag[0]["quote"] == "passage"
    assert big["sources"]["PMC9"]["doi"] == "10.1/x" and big["sources"]["PMC9"]["pmcid"] == "PMC9"
    other = big["provenance"][by_name(big, "processes.reactions", "other")]
    assert [r["kind"] for r in other] == ["other_paper"] and other[0]["source"] == "PMC7"
    assert big["sources"]["PMC7"]["title"] == "T"


def test_merge_duplicate_adds_provenance_not_row():
    big = build_big_payload(payload())
    rec = {"kind": "database", "source": "pathbank:PW1", "quote": None, "stage": "extend_db", "note": ""}
    rows = {"entities.compounds": [{"name": "alpha A", "mapped_ids": {"chebi": "1"}}],
            "processes.reactions": [{"name": "same", "inputs": ["alpha-A"], "outputs": ["B"]}]}
    merged = merge_extension(big, rows, {"entities.compounds": [[rec]], "processes.reactions": [[rec]]},
                             {"pathbank:PW1": {"kind": "database"}})
    assert merged["extension"] == {}
    assert kinds(merged, "entities.compounds", "A")[-1] == "database"
    assert merged["provenance"][by_name(merged, "processes.reactions", "r1")][-1] == rec
    assert "pathbank:PW1" in merged["sources"]
    assert big["extension"] == {} and kinds(big, "entities.compounds", "A") == ["paper", "database"]


def test_merge_reversed_reaction():
    big = build_big_payload(payload())
    rec = {"kind": "database", "source": None, "quote": None, "stage": "extend_db", "note": ""}
    merged = merge_extension(big, {"processes.reactions": [{"name": "rev", "inputs": ["B"], "outputs": ["A"]}]},
                             {"processes.reactions": [[rec]]})
    assert merged["extension"] == {}
    last = merged["provenance"][by_name(merged, "processes.reactions", "r1")][-1]
    assert last["note"] == "orientation reversed"


def test_merge_by_synonym_renames_reference():
    big = build_big_payload(payload())
    merged = merge_extension(big, {"processes.reactions": [{"name": "n", "inputs": ["alpha-A"], "outputs": ["Q"]}],
                                   "entities.compounds": [{"name": "Q"}]})
    rx = merged["extension"]["processes"]["reactions"][0]
    assert rx["inputs"] == ["A"]  # synonym of base A rewritten to the base name


def test_hops_on_a_chain():
    big = build_big_payload(payload())
    rows = {"entities.compounds": [{"name": "F"}, {"name": "G"}, {"name": "lonely"}],
            "processes.reactions": [{"name": "far", "inputs": ["F"], "outputs": ["G"]},
                                    {"name": "near", "inputs": ["D"], "outputs": ["F"]}],
            "biological_states": [{"name": "periplasm state"}],
            "element_locations.compound_locations": [{"compound": "G", "biological_state": "periplasm state"}]}
    merged = merge_extension(big, rows)
    hops = merged["hops"]
    ext_keys = [k for k in element_keys(merged) if k not in big["hops"]]
    # extension order: entities, processes, element_locations, then states
    named = dict(zip(["F", "G", "lonely", "far", "near", "G@periplasm", "periplasm state"], ext_keys))
    assert named["near"].startswith("processes.reactions:d>f")
    assert hops[named["near"]] == 1
    assert hops[named["far"]] == 2
    assert hops[named["F"]] == 1 and hops[named["G"]] == 2
    assert hops[named["lonely"]] is None
    assert hops[named["G@periplasm"]] == 2 and hops[named["periplasm state"]] == 2
    assert all(hops[k] == 0 for k in big["hops"])
