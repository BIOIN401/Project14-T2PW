"""Final payload P5 (D-W21 b/e): singular forms, Cellosaurus cell lines, numbered cell types from the paper's own phrase, and
organism-aware inner / outer membrane. No network: fakes only."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "tests"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from clean_model_fill_fakes import FakeFetcher  # noqa: E402
from test_pwml_writer import build_pwml_ir  # noqa: E402
from t2pw.clean_model.fill.cache import Cache  # noqa: E402
from t2pw.clean_model.fill.http import FetchError  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature  # noqa: E402
from t2pw.pwml.vocabulary_map import map_vocabulary_name, membrane_row, organism_kind, organism_kinds, singular_form  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402
from t2pw.service.final_payload import _paper_kw, apply_final_id_fill  # noqa: E402
from t2pw.service.final_payload_ontology import (CellosaurusPageFull, cellosaurus_exact, fill_ontology_ids, location_organisms,  # noqa: E402
                                                 numbered_phrases)

TODAY = "2026-11-13"


def ols(*docs):
    return {"response": {"docs": [{"obo_id": i, "label": lab, "synonym": syn} for i, lab, syn in docs]}}


def cello(*lines):
    """Cellosaurus search answer: lines are (accession, [(type, name), ...], [taxon ids]) ."""
    return {"Cellosaurus": {"cell-line-list": [
        {"accession-list": [{"type": "primary", "value": acc}], "name-list": [{"type": t, "value": v} for t, v in names],
         "species-list": [{"accession": tx, "database": "NCBI_TaxID", "label": {"9606": "Homo sapiens (Human)", "10090": "Mus musculus (Mouse)"}[tx]}
                          for tx in taxa]} for acc, names, taxa in lines]}}


MG63 = cello(("CVCL_0426", [("identifier", "MG-63"), ("synonym", "MG63")], ["9606"]),
             ("CVCL_C6UW", [("identifier", "MG-63/CISR8")], ["9606"]))
HUMAN = [{"name": "Homo sapiens", "taxonomy_id": "9606", "classification": "Eukaryote"}]


# ---- B: singular form ------------------------------------------------------------------------------------------------------
def test_singular_form_cuts_only_a_plural_s() -> None:
    assert singular_form("synaptic vesicles") == "synaptic vesicle" and singular_form("Type III cells") == "Type III cell"
    for kept in ("nucleus", "Golgi apparatus", "stroma", "thylakoid lumen", "cytosis", "mass", "ER"):
        assert singular_form(kept) is None, kept


def test_plural_of_a_pathwhiz_row_maps_to_the_row_and_unknown_plurals_stay_unmapped() -> None:
    assert map_vocabulary_name("cell-type", "granulosa cells", None) == ("Granulosa Cell", None, "mapped")
    assert map_vocabulary_name("subcellular-location", "peroxisomes", None) == ("Peroxisome", None, "mapped")
    assert map_vocabulary_name("subcellular-location", "plasma_membrane", None) == ("Cell Membrane", None, "mapped")
    assert map_vocabulary_name("cell-type", "Granulosa cells", "CL:9999999")[2] != "mapped"  # a disagreeing id blocks it
    for kind, name in [("cell-type", "Type III cells"), ("cell-type", "erythroid cells"), ("subcellular-location", "synaptic vesicles")]:
        assert map_vocabulary_name(kind, name, None)[2] == "unmapped", name


def test_ols_singular_is_a_second_exact_lookup_and_the_matched_form_is_recorded() -> None:
    p = {"entities": {"subcellular_locations": [{"name": "synaptic vesicles"}, {"name": "synaptic cleft"}]}, "biological_states": []}
    f = FakeFetcher([("ols4", {"q": "synaptic vesicle"}, ols(("GO:0008021", "synaptic vesicle", []))),
                     ("ols4", {"q": "synaptic cleft"}, ols(("GO:0043083", "synaptic cleft", [])))])
    rep = fill_ontology_ids(p, f, Cache(None), TODAY)
    rows = {r["name"]: r["ontology_id"] for r in p["entities"]["subcellular_locations"]}
    assert rows == {"synaptic vesicles": "GO:0008021", "synaptic cleft": "GO:0043083"}
    e = next(x for x in rep["entries"] if x["entity"] == "synaptic vesicles")
    assert e["matched_form"] == "synaptic vesicle" and e["match_type"] == "exact_label" and rep["extra"]["singular_form_resolved"] == 1
    assert "matched_form" not in next(x for x in rep["entries"] if x["entity"] == "synaptic cleft")
    assert [c[1]["q"] for c in f.calls] == ["synaptic cleft", "synaptic vesicles", "synaptic vesicle"]  # plural-looking names last; exact name first


# ---- B: Cellosaurus ----------------------------------------------------------------------------------------------------------
def _cl_payload(names=("MG-63",), species=HUMAN, states=()):
    return {"entities": {"cell_types": [{"name": n} for n in names], "species": copy.deepcopy(species)},
            "biological_states": [dict(s) for s in states]}


def test_cellosaurus_exact_drops_derivatives_and_wrong_names() -> None:
    f = FakeFetcher([("cellosaurus", {"q": 'idsy:"MG-63"'}, MG63)])
    assert [x["id"] for x in cellosaurus_exact(f, "MG-63")] == ["CVCL_0426"]
    assert cellosaurus_exact(f, "mg63") == [] and f.calls[0][1]["format"] == "json"  # a different name matches nothing
    only_derivative = cello(("CVCL_C6UW", [("identifier", "MG-63/CISR8")], ["9606"]))
    assert cellosaurus_exact(FakeFetcher([("cellosaurus", None, only_derivative)]), "MG-63") == []
    full = {"Cellosaurus": {"cell-line-list": [{"accession-list": [{"value": f"CVCL_{i}"}], "name-list": []} for i in range(100)]}}
    try:
        cellosaurus_exact(FakeFetcher([("cellosaurus", None, full)]), "MG-63")
        raise AssertionError("a full page cannot prove absence")
    except CellosaurusPageFull:
        pass
    p = _cl_payload()  # refused as a name, not counted as a service failure (the lookups after it still run)
    rep = fill_ontology_ids(p, FakeFetcher([("cellosaurus", None, full)]), Cache(None), TODAY)
    assert rep["extra"]["cellosaurus_refused"] == {"result_page_full": 1} and rep["counts"]["fetch_error"] == 0


def test_cell_line_gets_cellosaurus_id_not_ontology_id() -> None:
    p = _cl_payload()
    f = FakeFetcher([("cellosaurus", {"q": 'idsy:"MG-63"'}, MG63)])
    rep = fill_ontology_ids(p, f, Cache(None), TODAY)
    row = p["entities"]["cell_types"][0]
    assert row == {"name": "MG-63", "cellosaurus_id": "CVCL_0426"} and "ontology_id" not in row
    e = rep["entries"][0]
    assert e == {"kind": "cell_type", "entity": "MG-63", "field": "cellosaurus_id", "accession": "CVCL_0426", "source": "Cellosaurus",
                 "label": "MG-63", "match_type": "exact_label", "species": ["Homo sapiens (Human)"], "pathwhiz_can_use": False,
                 "retrieved": TODAY}
    assert rep["counts"]["resolved"] == 0 and rep["counts"]["no_exact_match"] == 1  # no ontology id was written
    assert rep["refused"] == [{"kind": "cell_type", "name": "MG-63", "reason": "no_exact_match", "cellosaurus": "CVCL_0426"}]
    assert rep["extra"]["cellosaurus_found"] == 1


def test_cell_line_refusals_species_ambiguity_and_cl_first() -> None:
    mouse = cello(("CVCL_X", [("identifier", "MG-63")], ["10090"]))
    p = _cl_payload()
    fill_ontology_ids(p, FakeFetcher([("cellosaurus", None, mouse)]), Cache(None), TODAY)
    assert "cellosaurus_id" not in p["entities"]["cell_types"][0]  # a mouse line in a human paper
    p = _cl_payload(species=[])
    rep = fill_ontology_ids(p, FakeFetcher([("cellosaurus", None, MG63)]), Cache(None), TODAY)
    assert "cellosaurus_id" not in p["entities"]["cell_types"][0] and rep["extra"]["cellosaurus_refused"] == {"no_species_context": 1}
    two = cello(("CVCL_A", [("identifier", "MG-63")], ["9606"]), ("CVCL_B", [("identifier", "x"), ("synonym", "mg-63")], ["9606"]))
    p = _cl_payload()
    rep = fill_ontology_ids(p, FakeFetcher([("cellosaurus", None, two)]), Cache(None), TODAY)
    assert "cellosaurus_id" not in p["entities"]["cell_types"][0] and rep["extra"]["cellosaurus_refused"] == {"ambiguous": 1}
    p = _cl_payload(["neuron x"])  # a CL term is found first: Cellosaurus is never asked
    f = FakeFetcher([("ols4", {"q": "neuron x"}, ols(("CL:0000999", "neuron x", []))), ("cellosaurus", None, MG63)])
    fill_ontology_ids(p, f, Cache(None), TODAY)
    assert p["entities"]["cell_types"][0]["ontology_id"] == "CL:0000999" and not [c for c in f.calls if "cellosaurus" in c[0]]
    # synonym match, taxon id taken from the payload species by name when the species row has no id
    syn = cello(("CVCL_0548", [("identifier", "SaOS-2"), ("synonym", "Saos 2")], ["9606"]))
    p = _cl_payload(["Saos 2"], species=[{"name": "Homo sapiens"}])
    rep = fill_ontology_ids(p, FakeFetcher([("cellosaurus", None, syn)]), Cache(None), TODAY)
    assert p["entities"]["cell_types"][0]["cellosaurus_id"] == "CVCL_0548" and rep["entries"][0]["match_type"] == "exact_synonym"


def test_cell_line_only_in_a_state_keeps_a_row_and_a_fetch_error_is_counted() -> None:
    p = {"entities": {"species": copy.deepcopy(HUMAN)}, "biological_states": [{"name": "s", "species": "Homo sapiens", "cell_type": "MG-63"}]}
    fill_ontology_ids(p, FakeFetcher([("cellosaurus", None, MG63)]), Cache(None), TODAY)
    assert p["entities"]["cell_types"] == [{"name": "MG-63", "cellosaurus_id": "CVCL_0426"}]
    p = _cl_payload()
    rep = fill_ontology_ids(p, FakeFetcher([("cellosaurus", None, FetchError("down"))]), Cache(None), TODAY)
    assert rep["counts"]["fetch_error"] == 1 and "cellosaurus_id" not in p["entities"]["cell_types"][0]


# ---- B: numbered cell types --------------------------------------------------------------------------------------------------
TASTE = ("Taste cells are of three types. Type II taste cells detect sweet and bitter. Type II taste cells release ATP. "
         "Type II cells and Type III cells differ; type II and type III cells also differ. Type III cells have synapses.")


def test_numbered_phrases_need_inserted_words_and_skip_fillers() -> None:
    got = numbered_phrases("Type II cell", [("paper", TASTE)])
    assert [(g["phrase"], g["count"]) for g in got] == [("type ii taste cell", 2)] and "Type II taste cells release" in got[0]["quote"]
    assert [g["phrase"] for g in numbered_phrases("Type III cells", [("paper", "Type III taste cells. type iii presynaptic cells.")])] == [
        "type iii presynaptic cell", "type iii taste cell"]
    assert numbered_phrases("erythroid cells", [("paper", "erythroid progenitor cells")]) == []  # not a numbered name: never narrowed


def _num_payload(*names, states=()):
    return {"entities": {"cell_types": [{"name": n} for n in names], "species": [{"name": "Mus musculus", "taxonomy_id": "10090"}]},
            "biological_states": [dict(s) for s in states]}


def test_numbered_name_resolves_only_when_the_paper_names_exactly_one_cl_term() -> None:
    f = FakeFetcher([("ols4", {"q": "type ii taste cell"}, ols(("CL:0002286", "type II taste cell", [])))])
    p = _num_payload("Type II cell")
    rep = fill_ontology_ids(p, f, Cache(None), TODAY, paper_text=TASTE, paper_source="01_source_text.txt")
    assert p["entities"]["cell_types"][0]["ontology_id"] == "CL:0002286"
    e = rep["entries"][0]
    assert (e["via"], e["context_phrase"], e["context_source"], e["context_count"], e["match_type"]) == (
        "paper_context", "type ii taste cell", "01_source_text.txt", 2, "exact_label") and "type ii taste cells" in e["context_quote"].lower()
    assert rep["extra"]["numbered_context_resolved"] == 1
    # two different CL terms named by the paper (alveolar and taste): ambiguous, left unresolved
    f = FakeFetcher([("ols4", {"q": "type ii taste cell"}, ols(("CL:0002286", "type II taste cell", []))),
                     ("ols4", {"q": "type ii alveolar cell"}, ols(("CL:0002063", "type II alveolar cell", [])))])
    p = _num_payload("Type II cell")
    rep = fill_ontology_ids(p, f, Cache(None), TODAY, paper_text=TASTE + " The type II alveolar cells secrete surfactant.")
    assert "ontology_id" not in p["entities"]["cell_types"][0] and rep["extra"]["numbered_context_refused"] == {"ambiguous_context": 1}
    # no paper text and nothing in the payload: unresolved, and no lookup was spent on it besides the name itself
    f = FakeFetcher([])
    p = _num_payload("Type II cell")
    rep = fill_ontology_ids(p, f, Cache(None), TODAY)
    assert "ontology_id" not in p["entities"]["cell_types"][0] and rep["extra"]["numbered_context_refused"] == {}
    assert {c[1]["q"] for c in f.calls if "ols4" in c[0]} == {"Type II cell"}


def test_numbered_name_uses_the_reviewed_synonym_and_never_doubles_a_row_the_payload_already_has() -> None:
    # "type III taste cell" is not a CL label; the reviewed vocabulary synonym names CL:0002285 (PathWhiz row 'type III taste bud cell')
    p = _num_payload("Type III cells")
    rep = fill_ontology_ids(p, FakeFetcher([]), Cache(None), TODAY, paper_text="Type III taste cells carry synapses. Type III taste cells sense sour.")
    e = rep["entries"][0]
    assert p["entities"]["cell_types"][0]["ontology_id"] == "CL:0002285" and e["match_type"] == "reviewed_synonym"
    assert (e["context_phrase"], e["context_source"], e["context_count"]) == ("type iii taste cell", "paper text", 2)
    # the id maps to the PathWhiz row by id in the writer, with no id written
    assert map_vocabulary_name("cell-type", "Type III cells", "CL:0002285") == ("type III taste bud cell", None, "mapped_by_id")
    # the payload also carries "Type III taste cell", which the writer already maps to that row: a second row must not reach it
    p = _num_payload("Type III cells", "Type III taste cell", "Type III cell")
    rep = fill_ontology_ids(p, FakeFetcher([]), Cache(None), TODAY)
    assert not any(r.get("ontology_id") for r in p["entities"]["cell_types"])
    assert [(r["name"], r["reason"], r["other_row"]) for r in rep["refused"]] == [
        ("Type III cell", "id_used_by_other_row", "Type III taste cell"), ("Type III cells", "id_used_by_other_row", "Type III taste cell")]
    assert rep["extra"]["id_used_by_other_row"] == 2 and rep["extra"]["numbered_context_resolved"] == 0


def test_two_names_for_one_ontology_term_resolve_only_one_row_the_exact_name() -> None:
    # synaptic vesicles (plural, listed first) and synaptic vesicle: the exact name keeps GO:0008021, the plural is refused (one id, one row)
    p = {"entities": {"subcellular_locations": [{"name": "synaptic vesicles"}, {"name": "synaptic vesicle"}]}, "biological_states": []}
    f = FakeFetcher([("ols4", {"q": "synaptic vesicle"}, ols(("GO:0008021", "synaptic vesicle", [])))])
    rep = fill_ontology_ids(p, f, Cache(None), TODAY)
    rows = {r["name"]: r.get("ontology_id") for r in p["entities"]["subcellular_locations"]}
    assert rows == {"synaptic vesicles": None, "synaptic vesicle": "GO:0008021"}
    assert rep["refused"] == [{"kind": "location", "name": "synaptic vesicles", "reason": "id_used_by_other_row",
                               "candidates": ["GO:0008021"], "other_row": "synaptic vesicle"}]
    assert rep["counts"]["resolved"] == 1 and rep["extra"]["singular_form_resolved"] == 0
    # an id the payload already carries on another row blocks too (existing ids are never touched, never doubled)
    p = {"entities": {"subcellular_locations": [{"name": "synaptic vesicle", "ontology_id": "GO:0008021"}, {"name": "synaptic vesicles"}]},
         "biological_states": []}
    rep = fill_ontology_ids(p, f, Cache(None), TODAY)
    assert p["entities"]["subcellular_locations"][1].get("ontology_id") is None and rep["refused"][0]["reason"] == "id_used_by_other_row"


# ---- E: organism-aware inner / outer membrane ---------------------------------------------------------------------------------
def test_organism_kind_tables_are_exact_and_contradictions_are_unknown() -> None:
    assert organism_kind("Escherichia coli", "Prokaryote") == "bacterium" and organism_kind("E. coli") == "bacterium"
    assert organism_kind("Homo sapiens", "Eukaryote") == "eukaryote" and organism_kind("Homo sapiens") is None
    assert organism_kind("Methanosarcina mazei", "Prokaryote") == "archaeon"
    assert organism_kind("Clostridium difficile", "Eukaryote") is None and organism_kind("Escherichia coli", "Eukaryote") is None
    # F8 (review 1): "Prokaryote" with an unlisted genus is undetermined, no longer "bacterium" (the old assertion here said "bacterium")
    assert organism_kind("Pseudomonas aeruginosa") == "bacterium" and organism_kind("Unknownus bacter", "Prokaryote") is None
    assert organism_kinds([("Escherichia coli", "Prokaryote"), ("Homo sapiens", None)]) == {"bacterium", "unknown"}


def test_membrane_names_map_to_their_own_bacterial_row_only_for_bacteria() -> None:
    bac, euk = frozenset({"bacterium"}), frozenset({"eukaryote"})
    for name, row in [("inner membrane", "Inner Membrane"), ("Inner Membrane", "Inner Membrane"), ("inner membrane (IM)", "Inner Membrane"),
                      ("outer membrane", "Outer Membrane"), ("Outer membrane (OM)", "Outer Membrane")]:
        assert map_vocabulary_name("subcellular-location", name, None, bac) == (row, None, "mapped"), name
    assert len({map_vocabulary_name("subcellular-location", n, None, bac)[0] for n in ("inner membrane (IM)", "outer membrane (OM)")}) == 2
    for orgs in (euk, frozenset(), frozenset({"unknown"}), frozenset({"bacterium", "eukaryote"}), frozenset({"archaeon"})):
        for name in ("inner membrane", "Outer Membrane", "inner membrane (IM)", "outer membrane (OM)"):
            assert map_vocabulary_name("subcellular-location", name, "GO:0019867", orgs) == (name, None, "unmapped"), (name, orgs)
    # (IM) on an outer membrane, leaflets and the mitochondrial membranes are never the bacterial rows
    assert membrane_row("outer membrane (IM)") is None and membrane_row("cytoplasmic leaflet of the IM") is None
    assert map_vocabulary_name("subcellular-location", "periplasmic leaflet of the IM", None, bac)[2] == "unmapped"
    assert map_vocabulary_name("subcellular-location", "inner membrane", "GO:0005743", euk) == ("Mitochondrial Inner Membrane", None, "mapped_by_id")
    assert map_vocabulary_name("subcellular-location", "mitochondrial inner membrane", None, euk)[2] == "exact"  # its own PathWhiz row
    # no context given: the phase 3 behaviour
    assert map_vocabulary_name("subcellular-location", "Inner Membrane", None) == ("Inner Membrane", None, "exact")
    assert map_vocabulary_name("subcellular-location", "inner membrane (IM)", None)[2] == "unmapped"


def _mem_payload(species, classification, names):
    return {"entities": {"species": [{"name": species, "classification": classification}],
                         "subcellular_locations": [{"name": n} for n in names]},
            "biological_states": [{"name": f"s{i}", "species": species, "subcellular_location": n} for i, n in enumerate(names)]}


def test_fill_skips_bacterial_membrane_names_and_looks_up_the_rest() -> None:
    names = ["inner membrane (IM)", "outer membrane (OM)", "cytoplasmic leaflet of the IM"]
    f = FakeFetcher([])
    p = _mem_payload("Escherichia coli", "Prokaryote", names)
    rep = fill_ontology_ids(p, f, Cache(None), TODAY)
    assert sorted(c[1]["q"] for c in f.calls if "ols4" in c[0] and "/api/search" in c[0]) == [
        "cytoplasmic leaflet of the IM"]  # the two membranes are resolved by the vocabulary map: not looked up; the leaflet is not
    assert [r["name"] for r in rep["refused"]] == ["cytoplasmic leaflet of the IM"]
    assert [(m["name"], m["organisms"], m["row"]) for m in rep["extra"]["membrane_rows_by_organism"]][:2] == [
        ("inner membrane (IM)", ["bacterium"], "Inner Membrane"), ("outer membrane (OM)", ["bacterium"], "Outer Membrane")]
    # a human paper: the same name is NOT the bacterial row, so the generic GO term is looked up (and the writer then drops the name)
    f = FakeFetcher([("ols4", {"q": "outer membrane"}, ols(("GO:0019867", "outer membrane", [])))])
    p = _mem_payload("Homo sapiens", "Eukaryote", ["outer membrane"])
    fill_ontology_ids(p, f, Cache(None), TODAY)
    assert p["entities"]["subcellular_locations"][0]["ontology_id"] == "GO:0019867"
    assert location_organisms(p, "outer membrane", ("subcellular_location",)) == {"eukaryote"}
    assert location_organisms({"entities": {"species": HUMAN}, "biological_states": []}, "x", ("subcellular_location",)) == {"eukaryote"}


def _build(payload):
    ir, report = build_pwml_ir(payload, strict_db=True)
    assert report["errors"] == []
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000", height=1400, width=3200,
                           background_color="#FFFFFF", ref=str(ROOT / "reference" / "PW000001.pwml"))
    b = DeterministicPwmlBuilder(extraction=ir, signature=discover_structure_signature(ROOT / "reference" / "PW000001.pwml"), args=args)
    return b, b.build().root


def _loc_names(root):
    return [n.findtext("name") for n in root.findall(".//subcellular-locations/subcellular-location")]


def _writer_payload(species_row, names):
    return {"entities": {"species": [species_row], "subcellular_locations": [{"name": n} for n in names],
                         "tissues": [{"name": f"t{i}", "pathwhiz_id": 10 + i} for i in range(len(names))],
                         "compounds": [{"name": "A", "pathbank_compound_id": 103}]},
            "biological_states": [{"name": f"s{i}", "species": species_row["name"], "tissue": f"t{i}",
                                   "subcellular_location": n} for i, n in enumerate(names)],
            "processes": {"reactions": [], "transports": [], "interactions": []}}


def test_writer_keeps_inner_and_outer_separate_for_bacteria_and_drops_them_for_a_human(monkeypatch) -> None:
    ecoli = {"name": "Escherichia coli", "pathwhiz_id": 1, "classification": "Prokaryote"}
    names = ["inner membrane (IM)", "outer membrane (OM)", "cytosol"]
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    b, root = _build(_writer_payload(ecoli, names))
    assert _loc_names(root) == names and not b.vocabulary_outcomes  # flag off: untouched
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    b, root = _build(_writer_payload(ecoli, names))
    assert _loc_names(root) == ["Inner Membrane", "Outer Membrane", "cytosol"]
    assert {(o["name"], o["written"], tuple(o["organisms"])) for o in b.vocabulary_outcomes if o["kind"] == "subcellular-location"} == {
        ("inner membrane (IM)", "Inner Membrane", ("bacterium",)), ("outer membrane (OM)", "Outer Membrane", ("bacterium",))}
    # plain forms of a bacterium that PathWhiz rows already name: still the same rows
    b, root = _build(_writer_payload(ecoli, ["Inner Membrane", "outer membrane", "cytosol"]))
    assert _loc_names(root) == ["Inner Membrane", "Outer Membrane", "cytosol"]
    # a human paper: the bacterial row's name must not be written (it would bind at import); the unresolved name is dropped
    human = {"name": "Homo sapiens", "pathwhiz_id": 1, "classification": "Eukaryote"}
    b, root = _build(_writer_payload(human, ["Inner Membrane", "outer membrane", "cytosol"]))
    assert _loc_names(root) == ["cytosol"] and b.layout_debug_counts["locations_dropped_no_id"] == 2
    # no species information at all: undetermined, so unresolved too
    unknown = {"name": "Zzz zzz", "pathwhiz_id": 1}
    b, root = _build(_writer_payload(unknown, ["Inner Membrane", "cytosol"]))
    assert "Inner Membrane" not in _loc_names(root)


def test_apply_final_id_fill_passes_the_paper_text_and_reports_the_p5_counters() -> None:
    f = FakeFetcher([("ols4", {"q": "type ii taste cell"}, ols(("CL:0002286", "type II taste cell", []))),
                     ("cellosaurus", {"q": 'idsy:"MG-63"'}, MG63)])
    p = _num_payload("Type II cell", "MG-63")
    p["entities"]["species"].append({"name": "Homo sapiens", "taxonomy_id": "9606"})
    rep = apply_final_id_fill(p, {}, fetcher=f, cache=Cache(None), today=TODAY, paper_text=TASTE, paper_source="01_source_text.txt")
    assert rep["error"] is None
    rows = {r["name"]: r for r in p["entities"]["cell_types"]}
    assert rows["Type II cell"]["ontology_id"] == "CL:0002286" and rows["MG-63"] == {"name": "MG-63", "cellosaurus_id": "CVCL_0426"}
    assert rep["ontology"]["extra"]["numbered_context_resolved"] == 1 and rep["ontology"]["extra"]["cellosaurus_found"] == 1
    assert rep["by_field"] == {"cell_type.ontology_id": 1, "cell_type.cellosaurus_id": 1}
    prov = p["final_payload_fill"]
    assert prov["written"] == 2 and {e["source"] for e in prov["entries"]} == {"OLS4", "Cellosaurus"}
    # without the paper text the numbered name stays unresolved (nothing else names a full phrase)
    q = _num_payload("Type II cell")
    rep = apply_final_id_fill(q, {}, fetcher=FakeFetcher([]), cache=Cache(None), today=TODAY)
    assert "ontology_id" not in q["entities"]["cell_types"][0] and rep["ontology"]["extra"]["numbered_context_resolved"] == 0


def test_paper_kw_reads_the_run_state_only() -> None:
    assert _paper_kw({"final_payload_paper_text": "t", "final_payload_paper_text_source": "01_source_text.txt"}) == {
        "paper_text": "t", "paper_source": "01_source_text.txt"}
    assert _paper_kw({"final_payload_paper_text": "t"}) == {"paper_text": "t", "paper_source": "paper text"}
    assert _paper_kw(None) == {} and _paper_kw({}) == {} and _paper_kw({"final_payload_paper_text": ""}) == {}
