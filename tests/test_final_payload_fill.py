"""Final payload phase 1 (D-W12): the flag-guarded database id fill. No network: the fill layer's FakeFetcher answers."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from clean_model_fill_fakes import FakeFetcher, chebi, ols, uniprot, uniprot_search  # noqa: E402
from helpers_prefreeze import prefrozen  # noqa: E402
from t2pw.clean_model.fill.cache import Cache  # noqa: E402
from t2pw.clean_model.fill.http import FetchError  # noqa: E402
from t2pw.pipeline.stage_contracts import validate_pre_export, validate_runtime_payload_contract  # noqa: E402
from t2pw.pwml.compound_resolution import NO_DB_RESOLVER  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature, repair_tree  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402
from t2pw.service.final_payload import (  # noqa: E402
    FINAL_PAYLOAD_FILL_FIELD, apply_final_id_fill, fill_meta, final_payload_enabled, maybe_apply_final_id_fill,
)

TODAY = "2026-11-10"
ID_KEYS = ("chebi_id", "kegg_id", "hmdb_id", "pubchem_cid", "cas", "drugbank_id")


def _payload():
    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "taxonomy_id": "562"}],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [
                {"name": "chorismate"},
                {"name": "ATP", "chebi_id": "CHEBI:15422"},
                {"name": "ADP", "chebi_id": "CHEBI:16761", "kegg_id": "C99999"},
                {"name": "zinc"},
                {"name": "mystery"},
                {"name": "complex of Fe and S"},
                {"name": "has mapped", "mapped_ids": {"chebi": "CHEBI:1"}},
                {"name": "GTP"},
                {"name": "GTP phosphate"},
            ],
            "proteins": [
                {"name": "EntC", "gene_name": "entC", "taxonomy_id": "562"},
                {"name": "Unknown", "gene_name": "unk", "taxonomy_id": "562", "pathbank_protein_id": 9659},
                {"name": "Kinase family", "gene_name": "kin", "taxonomy_id": "562"},
            ],
            "protein_complexes": [{"name": "EntC dimer", "components": ["EntC", "EntC"]}],
        },
        "biological_states": [],
        "processes": {"reactions": []},
    }


def _routes():
    return [
        ("/compound/15422/", None, chebi()),
        ("/compound/16761/", None, chebi(16761, "ADP", kegg="C00008", hmdb="HMDB0001341")),
        ("/compound/29748/", None, chebi(29748, "chorismate(2-)", "C10H8O6", "WTFXTQVDAKGDEY-HTQZYQBOSA-L", "C00251", "HMDB0000975")),
        ("/compound/30185/", None, chebi(30185, "GTP", kegg="C00044", hmdb="HMDB0001273")),
        ("ols4", {"q": "chorismate"}, ols((29748, "chorismate(2-)", ["chorismate"]))),
        ("ols4", {"q": "zinc"}, ols((30185, "zinc(0)", ["zinc"]), (27363, "zinc atom", ["zinc"]))),
        ("ols4", {"q": "complex of Fe and S"}, ols((33000, "complex of Fe and S", []))),
        ("ols4", {"q": "has mapped"}, ols((77, "has mapped", []))),
        ("ols4", {"q": "GTP"}, ols((30185, "GTP", []))),
        ("ols4", {"q": "GTP phosphate"}, ols((30185, "GTP phosphate", []))),
        ("/search", {"query": "gene_exact:entC AND taxonomy_id:562"}, uniprot_search(uniprot("P0AEJ2"))),
        ("/search", {"query": "gene_exact:kin AND taxonomy_id:562"}, uniprot_search(uniprot("Q1", gene="kin"))),
        ("/search", {"query": "gene_exact:unk AND taxonomy_id:562"}, uniprot_search(uniprot("Q2", gene="unk"))),
    ]


def _fill(payload, routes=None, **kw):
    f = FakeFetcher(routes if routes is not None else _routes())
    return apply_final_id_fill(payload, {"organism": "Escherichia coli"}, fetcher=f, cache=Cache(None), today=TODAY, **kw), f


def _rows(p, kind="compounds"):
    return {r["name"]: r for r in p["entities"][kind]}


def test_flag_parsing(monkeypatch):
    for v, want in (("1", True), ("TRUE", True), ("Yes", True), ("0", False), ("", False), ("off", False)):
        monkeypatch.setenv("T2PW_FINAL_PAYLOAD", v)
        assert final_payload_enabled() is want
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD")
    assert final_payload_enabled() is False


def test_flag_off_wiring_helper_leaves_payload_untouched(monkeypatch):
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    p = _payload()
    before = copy.deepcopy(p)
    called = []
    monkeypatch.setattr("t2pw.service.final_payload.apply_final_id_fill", lambda *a, **k: called.append(1))
    assert maybe_apply_final_id_fill(p, {"likely_organism": "Escherichia coli"}, {}) == {}
    assert p == before and not called


def test_flag_on_wiring_helper_runs_and_never_raises(monkeypatch):
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    monkeypatch.setattr("t2pw.service.final_payload.apply_final_id_fill", lambda *a, **k: 1 / 0)
    p = _payload()
    rep = maybe_apply_final_id_fill(p, {"likely_organism": "Escherichia coli"}, {"pmid": "123"})
    assert rep["written"] == 0 and "ZeroDivisionError" in rep["error"]


def test_postpipeline_seam_is_after_prefreeze_and_before_validation():
    src = (ROOT / "src/t2pw/service/postpipeline.py").read_text(encoding="utf-8")
    a = src.index("final_payload_fill_report = maybe_apply_final_id_fill(")
    assert src.rindex("run_prefreeze_resolution(", 0, a) < a
    assert a < src.index('boundary="post_enrichment"', a)


def test_fill_meta_sources():
    assert fill_meta({"likely_organism": "E. coli", "pathway_name": "x"}, {"pmid": 5, "pmcid": ""}) == \
        {"organism": "E. coli", "pmid": "5"}
    assert fill_meta(None, None) == {}


def test_ids_filled_with_provenance_and_existing_untouched():
    p = _payload()
    before = copy.deepcopy(p)
    rep, _f = _fill(p)
    assert rep["error"] is None
    c = _rows(p)
    assert c["chorismate"]["chebi_id"] == "CHEBI:29748" and c["chorismate"]["kegg_id"] == "C00251"
    assert c["ATP"]["chebi_id"] == "CHEBI:15422" and c["ATP"]["kegg_id"] == "C00002" and c["ATP"]["hmdb_id"] == "HMDB0000538"
    # never overwrite: ADP keeps its (disagreeing) kegg id, and every pre-existing value survives
    assert c["ADP"]["kegg_id"] == "C99999"
    for r0 in before["entities"]["compounds"]:
        for k, v in r0.items():
            assert _rows(p)[r0["name"]][k] == v
    fp = p[FINAL_PAYLOAD_FILL_FIELD]
    assert fp["written"] == len(fp["entries"]) == rep["written"] > 0
    e = next(x for x in fp["entries"] if x["entity"] == "chorismate" and x["field"] == "chebi_id")
    assert e == {"kind": "compound", "entity": "chorismate", "field": "chebi_id", "accession": "CHEBI:29748",
                 "source": "ChEBI", "match_type": "exact_synonym+charged_form", "retrieved": TODAY}
    # nothing besides id slots of compounds/species and the one provenance key changed
    p2 = copy.deepcopy(p)
    del p2[FINAL_PAYLOAD_FILL_FIELD]
    for r in p2["entities"]["compounds"]:
        for k in ID_KEYS:
            r.pop(k, None)
    b2 = copy.deepcopy(before)
    for r in b2["entities"]["compounds"]:
        for k in ID_KEYS:
            r.pop(k, None)
    assert p2 == b2


def test_ambiguous_name_gives_no_id_only_a_suggestion():
    p = _payload()
    rep, _f = _fill(p)
    assert not any(k in _rows(p)["zinc"] for k in ID_KEYS)
    assert any(s["entity"] == "zinc" and s["field"] == "chebi_id" and len(s["candidates"]) == 2 for s in rep["suggestions"])
    assert not any(k in _rows(p)["mystery"] for k in ID_KEYS)


def test_complex_family_unknown_and_proteins_get_no_id():
    p = _payload()
    rep, f = _fill(p)
    assert not any(k in _rows(p)["complex of Fe and S"] for k in ID_KEYS)
    assert rep["refused"].get("not_a_single_molecule") == 1
    for r in p["entities"]["proteins"] + p["entities"]["protein_complexes"]:
        assert "uniprot" not in r and "uniprot_id" not in r and "mapped_ids" not in r
    # the candidate for EntC is a report suggestion, searched within the pathway species only
    assert any(s["entity"] == "EntC" and s["field"] == "uniprot" for s in rep["suggestions"])
    assert not any(s["entity"] in ("Unknown", "Kinase family") for s in rep["suggestions"])
    assert all("taxonomy_id:562" in prm.get("query", "") for url, prm in f.calls if "uniprot" in url and "/search" in url)


def test_protein_without_known_species_is_not_searched():
    p = _payload()
    for r in p["entities"]["proteins"]:
        r.pop("taxonomy_id")
    p["entities"]["species"][0].pop("taxonomy_id")
    rep, f = _fill(p)
    assert not [1 for url, _ in f.calls if "uniprot" in url and "/search" in url]
    assert not rep["suggestions"] or all(s["field"] != "uniprot" for s in rep["suggestions"])


def test_id_used_by_other_row_is_refused_and_mapped_ids_respected():
    p = _payload()
    _rows(p)["ATP"]["kegg_id"] = "C00008"  # ADP's cross-reference id is already used by ATP
    rep, _f = _fill(p)
    c = _rows(p)
    assert "kegg_id" in c["ADP"] and c["ADP"]["kegg_id"] == "C99999"  # untouched
    assert "chebi_id" not in c["has mapped"]  # mapped_ids already carries a chebi id
    # GTP and GTP phosphate resolve to the same ChEBI id: fill's own collision guard lets only the first take it
    assert [n for n in ("GTP", "GTP phosphate") if "chebi_id" in c[n]] == ["GTP"]
    # a second entity cannot take an id another row holds (kegg C00008 on ATP vs ADP's own xref): refused when empty
    _rows(p)["ADP"].pop("kegg_id")
    p2 = copy.deepcopy(p)
    del p2[FINAL_PAYLOAD_FILL_FIELD]
    rep2, _ = _fill(p2)
    assert "kegg_id" not in _rows(p2)["ADP"] and rep2["refused"].get("id_used_by_other_row", 0) >= 1


def test_db_row_id_counts_as_filled():
    p = _payload()
    _rows(p)["chorismate"]["db_row"] = {"name": "chorismate", "kegg_id": "C00251", "chebi_id": ""}
    _fill(p)
    assert "kegg_id" not in _rows(p)["chorismate"]


@pytest.mark.parametrize("exc", [FetchError("boom"), RuntimeError("boom")])
def test_failure_leaves_payload_deep_equal(exc, monkeypatch):
    p = _payload()
    before = copy.deepcopy(p)
    rep, _f = _fill(p, routes=[("ols4", None, exc), ("/compound/", None, exc), ("/search", None, exc)])
    assert p == before and rep["written"] == 0 and FINAL_PAYLOAD_FILL_FIELD not in p
    if isinstance(exc, RuntimeError):
        assert rep["error"] is None or "boom" in rep["error"]


def test_error_after_partial_planning_restores_payload(monkeypatch):
    p = _payload()
    before = copy.deepcopy(p)

    def boom(*a, **k):
        raise RuntimeError("late failure")

    monkeypatch.setattr("t2pw.service.final_payload._plan", boom)
    rep, _f = _fill(p)
    assert p == before and rep["error"] == "RuntimeError: late failure" and rep["written"] == 0


def test_species_taxonomy_filled_only_when_empty():
    p = _payload()
    p["entities"]["species"].append({"name": "Pseudomonas putida"})
    routes = _routes() + [("esearch.fcgi", {"term": "Pseudomonas putida[Scientific Name]"},
                           {"esearchresult": {"count": "1", "idlist": ["303"]}})]
    rep, _f = _fill(p, routes=routes)
    sp = _rows(p, "species")
    assert sp["Escherichia coli"]["taxonomy_id"] == "562" and sp["Pseudomonas putida"]["taxonomy_id"] == "303"
    assert any(e["kind"] == "species" and e["source"] == "NCBI Taxonomy" for e in p[FINAL_PAYLOAD_FILL_FIELD]["entries"])


# ---- the filled payload still goes through the real validators / IR / writer --------------------------------------------

def _pipeline_payload():
    return {
        "metadata": {"name": "x", "pathway_name": "x", "subject": "Metabolic", "pathway_subject": "Metabolic"},
        "entities": {
            "species": [{"name": "Pseudomonas putida", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [{"name": "caffeine", "pathbank_compound_id": 101}, {"name": "theobromine", "pathbank_compound_id": 102}],
            "proteins": [{"name": "NdmA", "species": "Pseudomonas putida", "mapped_ids": {"uniprot": "Q9I147"}}],
            "protein_complexes": [{"name": "NdmA complex", "species": "Pseudomonas putida", "generated": True,
                                   "generation_reason": "single_protein_pathwhiz_wrapper",
                                   "components": [{"name": "NdmA", "stoichiometry": 1}]}],
        },
        "biological_states": [{"name": "Pseudomonas putida cytosol", "species": "Pseudomonas putida",
                               "subcellular_location": "cytosol"}],
        "processes": {"reactions": [{"name": "caffeine demethylation", "inputs": ["caffeine"], "outputs": ["theobromine"],
                                     "biological_state": "Pseudomonas putida cytosol",
                                     "enzymes": [{"entity": "NdmA complex", "entity_type": "protein_complex", "role": "catalyst"}],
                                     "modifiers": [{"entity": "NdmA complex", "entity_type": "protein_complex", "role": "catalyst"}]}],
                      "transports": [], "interactions": []},
    }


def _pwml(payload):
    ir, rep = build_pwml_ir(copy.deepcopy(payload), strict_db=False)
    assert rep["errors"] == [] and validate_pwml_ir(ir)["ok"]
    ref = ROOT / "reference" / "PW000001.pwml"
    sig = discover_structure_signature(ref)
    args = SimpleNamespace(name="g", description="", subject="Metabolic", pw_id="PW000000", height=1400, width=3200,
                           background_color="#FFFFFF", ref=str(ref))
    root = repair_tree(etree.ElementTree(DeterministicPwmlBuilder(extraction=ir, signature=sig, args=args).build().root), sig).getroot()
    return ir, root


def test_filled_payload_builds_ir_and_pwml_with_only_the_ids_changed():
    p = prefrozen(_pipeline_payload(), db_resolver=NO_DB_RESOLVER)
    ir0, root0 = _pwml(p)
    f = FakeFetcher([("ols4", {"q": "theobromine"}, ols((28946, "theobromine", []))),
                     ("/compound/28946/", None, chebi(28946, "theobromine", "C7H8N4O2", "YAPQBXQYLJRXSA-UHFFFAOYSA-N", "C07480", "HMDB0002825", "83-67-0"))])
    rep = apply_final_id_fill(p, {"organism": "Pseudomonas putida"}, fetcher=f, cache=Cache(None), today=TODAY)
    assert rep["error"] is None and rep["written"] >= 2
    for boundary in ("post_enrichment", "pre_export"):  # same verdict as before the fill (this fixture has no mapping_meta)
        assert validate_runtime_payload_contract(p, boundary=boundary)["errors"] == \
            validate_runtime_payload_contract(prefrozen(_pipeline_payload(), db_resolver=NO_DB_RESOLVER), boundary=boundary)["errors"]
    assert validate_pre_export(p, strict_db=False)["summary"]["error_count"] == 0
    ir1, root1 = _pwml(p)
    theo = next(c for c in root1.iter("compound") if c.findtext("name") == "theobromine")
    assert theo.findtext("chebi-id") == "28946" and theo.findtext("kegg-id") == "C07480" and theo.findtext("cas") == "83-67-0"
    idtags = {"chebi-id", "kegg-id", "hmdb-id", "pubchem-cid", "cas", "drugbank-id"}
    for root in (root0, root1):
        for el in root.iter():
            if el.tag in idtags:
                el.text = None; el.attrib.pop("nil", None)
    assert etree.tostring(root0) == etree.tostring(root1)
    assert ir0["entities"]["compounds"] != ir1["entities"]["compounds"]


def test_freeze_quarantine_and_hash_tolerate_the_provenance_key(tmp_path):
    from t2pw.service.postpipeline import freeze_canonical_payload

    def run(payload):
        return freeze_canonical_payload(payload, strict_db=False, outputs_dir=tmp_path, pathway_context={},
                                        export_mode="pathwhiz", research_mode=False, tmp=tmp_path, state={})

    p0 = prefrozen(_pipeline_payload(), db_resolver=NO_DB_RESOLVER)
    p1 = copy.deepcopy(p0)
    f = FakeFetcher([("ols4", {"q": "theobromine"}, ols((28946, "theobromine", []))),
                     ("/compound/28946/", None, chebi(28946, "theobromine", "C7H8N4O2", "YAPQBXQYLJRXSA-UHFFFAOYSA-N", "C07480", "HMDB0002825", "83-67-0"))])
    apply_final_id_fill(p1, {"organism": "Pseudomonas putida"}, fetcher=f, cache=Cache(None), today=TODAY)
    r0, r1 = run(p0), run(p1)
    assert r1["quarantine_ok"] == r0["quarantine_ok"]
    if r1["quarantine_ok"]:
        assert FINAL_PAYLOAD_FILL_FIELD in r1["payload"]


def _abbrev_fill(names_routes, names):
    p = {"entities": {"species": [], "subcellular_locations": [], "compounds": [{"name": n} for n in names], "proteins": [],
                      "protein_complexes": []}, "biological_states": [], "processes": {"reactions": []}}
    f = FakeFetcher(names_routes)
    return p, apply_final_id_fill(p, {"organism": "Escherichia coli"}, fetcher=f, cache=Cache(None), today=TODAY)


def test_short_abbreviation_synonym_is_refused_with_its_cross_references():
    p, rep = _abbrev_fill([("ols4", {"q": "MK"}, ols((74706, "Met-Lys", ["MK"]))),
                           ("/compound/74706/", None, chebi(74706, "Met-Lys", "C11H23N3O3S", "X-Y-Z", "C99999", "HMDB0011745", "1-2-3"))], ["MK"])
    mk = _rows(p)["MK"]
    assert rep["error"] is None and rep["written"] == 0
    assert all(k not in mk for k in ID_KEYS) and "final_payload_fill" not in p
    assert rep["refused"].get("short_abbreviation_synonym", 0) >= 2  # chebi_id and the fields derived from that entry
    assert {d["field"] for d in rep["refused_detail"] if d["reason"] == "short_abbreviation_synonym"} >= {"chebi_id", "kegg_id", "hmdb_id"}


def test_exact_label_short_name_still_accepted():
    p, rep = _abbrev_fill([("ols4", {"q": "ATP"}, ols((15422, "ATP", []))), ("/compound/15422/", None, chebi())], ["ATP"])
    assert _rows(p)["ATP"]["chebi_id"] == "CHEBI:15422" and _rows(p)["ATP"]["kegg_id"] == "C00002"
    assert "short_abbreviation_synonym" not in rep["refused"]


def test_synonym_with_lowercase_or_long_name_still_accepted():
    p, rep = _abbrev_fill([("ols4", {"q": "MgCl2"}, ols((6636, "magnesium dichloride", ["MgCl2"]))),
                           ("/compound/6636/", None, chebi(6636, "magnesium dichloride", "Cl2Mg", "TWRXJAOTZQYOKJ-UHFFFAOYSA-L", "C00305", "HMDB0000547")),
                           ("ols4", {"q": "Cys-Asp-Gly"}, ols((140231, "Cys-Asp-Gly peptide", ["Cys-Asp-Gly"]))),
                           ("/compound/140231/", None, chebi(140231, "Cys-Asp-Gly peptide", "C9H15N3O6S", "A-B-C", "C88888", "HMDB0000001")),
                           ("ols4", {"q": "SHCHCX"}, ols((39564, "SHCHC-like", ["SHCHCX"]))),
                           ("/compound/39564/", None, chebi(39564, "SHCHC-like", "C11H11O5", "D-E-F", "C16", "HMDB0002"))],
                          ["MgCl2", "Cys-Asp-Gly", "SHCHCX"])
    for n in ("MgCl2", "Cys-Asp-Gly", "SHCHCX"):
        assert _rows(p)[n]["chebi_id"].startswith("CHEBI:"), n
    assert "short_abbreviation_synonym" not in rep["refused"]


def test_short_abbreviation_rule_shape():
    from t2pw.service.final_payload import _short_abbreviation as sa
    assert [sa(n) for n in ("MK", "PMEA", "CDG", "DMAPP", "SHCHC", "M K")] == [True] * 6
    assert [sa(n) for n in ("H2O", "H+", "O2", "NAD+", "Ca2+", "MgCl2", "SHCHCX", "A", "Cys-Asp-Gly", "")] == [False] * 10


def test_h2o_hplus_o2_synonym_fills_are_kept():
    routes, names = [], ["H2O", "H+", "O2"]
    for q, cid, lab, kegg in (("H2O", 15377, "water", "C00001"), ("H+", 15378, "proton", "C00080"), ("O2", 15379, "dioxygen", "C00007")):
        routes += [("ols4", {"q": q}, ols((cid, lab, [q]))), (f"/compound/{cid}/", None, chebi(cid, lab, kegg=kegg, hmdb="HMDB0001", cas="1-1-1"))]
    p, rep = _abbrev_fill(routes, names)
    for n, cid in zip(names, (15377, 15378, 15379)):
        assert _rows(p)[n]["chebi_id"] == f"CHEBI:{cid}", n
    assert "short_abbreviation_synonym" not in rep["refused"]
