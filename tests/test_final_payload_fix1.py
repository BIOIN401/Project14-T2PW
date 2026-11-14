"""Final payload P5 fix round 1: review 1 findings F1-F9. Fakes only, no network."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "tests"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from clean_model_fill_fakes import FakeFetcher  # noqa: E402
from test_final_payload_nucleic_lookup import E, Fake, G, P, TODAY, row, run  # noqa: E402
from test_final_payload_ontology_p5 import _build, _loc_names, _writer_payload, ols  # noqa: E402
from t2pw.clean_model.fill.cache import Cache  # noqa: E402
from t2pw.clean_model.fill.http import FetchError  # noqa: E402
from t2pw.pwml.vocabulary_map import map_vocabulary_name, organism_kind, organism_kinds  # noqa: E402
from t2pw.service import final_payload as fp  # noqa: E402
from t2pw.service.final_payload_ontology import MAX_PHRASES, fill_ontology_ids, numbered_context  # noqa: E402

HUMAN = {"name": "Homo sapiens", "pathwhiz_id": 1, "classification": "Eukaryote"}


# ---- F1: a non-bacterial membrane is decided from the WRITTEN id -------------------------------------------------------------
def _human_membrane_payload(second_location_tissue):
    p = _writer_payload(HUMAN, ["outer membrane", "cytosol"])
    p["entities"]["subcellular_locations"][0]["ontology_id"] = "GO:0019867"  # the fill found the generic GO term for a human outer membrane
    p["entities"]["subcellular_locations"][1]["pathwhiz_id"] = 3
    p["biological_states"][1]["tissue"] = second_location_tissue
    return p


def test_f1_human_outer_membrane_with_a_go_id_is_dropped_not_written_by_name(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    b, root = _build(_human_membrane_payload("t1"))  # the two states differ by tissue: dropping the location binds nothing
    assert _loc_names(root) == ["cytosol"] and b.layout_debug_counts["locations_dropped_no_id"] == 1
    assert "nonbacterial_membranes_kept_by_name" not in b.layout_debug_counts
    assert map_vocabulary_name("subcellular-location", "outer membrane", "GO:0019867", frozenset({"eukaryote"})) == ("outer membrane", None, "unmapped")


def test_f1_kept_nonbacterial_membrane_is_renamed_so_it_cannot_bind_to_the_bacterial_row(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    b, root = _build(_human_membrane_payload("t0"))  # same tissue: without the location the first state is a subset of the second, so it is kept
    names = _loc_names(root)
    assert "outer membrane" not in [n.casefold() for n in names] and "Outer Membrane" not in names
    assert names == ["outer membrane (non-bacterial, no ontology id)", "cytosol"]
    assert b.layout_debug_counts["nonbacterial_membranes_kept_by_name"] == 1 and b.layout_debug_counts["components_kept_wildcard_guard"] == 1


def test_f1_flag_off_is_untouched(monkeypatch) -> None:
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    b, root = _build(_human_membrane_payload("t0"))
    assert _loc_names(root) == ["outer membrane", "cytosol"] and not b.layout_debug_counts.get("nonbacterial_membranes_kept_by_name")


# ---- F2: a conflicting row is not sent to the lookup -------------------------------------------------------------------------
def test_f2_conflicting_row_stays_unknown_and_is_not_looked_up() -> None:
    fake = Fake([G(1, "LHCB1.1", "protein-coding", "", 3702)])
    p = P({"name": "LHCB1.1"}, taxid="3702")
    rep = run(p, fake, text="The LHCB1.1 gene mRNA rose.")
    r = row(p)
    assert r["nucleotide_type_unknown_reason"] == "conflicting" and r["nucleotide_type_unknown_lookup"] == "not_attempted"
    assert "nucleotide_type" not in r and fake.calls == [] and rep["lookup_resolved"] == 0 and rep["unknown_by_reason"] == {"conflicting": 1}


# ---- F3: protein-coding -> DNA only without protein wording around the row ---------------------------------------------------
_CRD1 = [G(7, "CRD1", "protein-coding", "AT3G56940, CHL27", 3702, "Arabidopsis thaliana")]


def test_f3_protein_context_keeps_the_row_unknown() -> None:
    p = P({"name": "CHL27"}, taxid="3702")
    rep = run(p, Fake(_CRD1), text="The CHL27 subunit of the cyclase binds iron.")
    r = row(p)
    assert r["nucleotide_type_status"] == "unknown" and r["nucleotide_type_unknown_lookup"] == "protein_context" and "nucleotide_type" not in r
    assert rep["lookup_codes"] == {"protein_context": 1} and rep["lookup_resolved"] == 0


def test_f3_no_protein_word_in_the_sentences_of_the_row_resolves_dna_and_other_sentences_do_not_count() -> None:
    p = P({"name": "CHL27"}, taxid="3702")
    run(p, Fake(_CRD1), text="CHL27 was measured in leaves. The cyclase subunit binds iron.")
    assert row(p)["nucleotide_type"] == "DNA" and row(p)["nucleotide_type_source"]["match_type"] == "exact_alias"
    p = P({"name": "Foo1", "synonyms": ["CHL27"]}, taxid="3702")  # a synonym of the row counts as naming it
    rep = run(p, Fake([G(8, "Foo1", "protein-coding", "", 3702)]), text="The CHL27 protein is abundant.")
    assert row(p)["nucleotide_type_unknown_lookup"] == "protein_context" and rep["lookup_resolved"] == 0
    p = P({"name": "RyhB"}, taxid="562")  # an RNA gene type is not affected by the rule
    run(p, Fake([G(1, "ryhB", "ncRNA")]), text="The RyhB protein-binding region.")
    assert row(p)["nucleotide_type"] == "RNA"


# ---- F4: a refusal by one source blocks the other source's answer ------------------------------------------------------------
def test_f4_ncbi_refusal_blocks_an_rnacentral_hit() -> None:
    two_genes = [G(1, "ryhB", "ncRNA"), G(2, "ryhB", "ncRNA")]
    p = P({"name": "RyhB"})
    rep = run(p, Fake(two_genes, rnac={("ryhb", "562"): [E("URS1", "562", ["ryhB"])]}))
    assert row(p)["nucleotide_type_unknown_lookup"] == "refused_by_a_source" and "nucleotide_type" not in row(p) and rep["lookup_resolved"] == 0
    p = P({"name": "RyhB"})  # NCBI simply finds nothing: RNAcentral alone still resolves (not_found is not a refusal)
    run(p, Fake([], rnac={("ryhb", "562"): [E("URS1", "562", ["ryhB"])]}))
    assert row(p)["nucleotide_type"] == "RNA"
    p = P({"name": "RyhB"})  # NCBI pseudogene: not decisive, a refusal
    run(p, Fake([G(1, "ryhB", "pseudo")], rnac={("ryhb", "562"): [E("URS1", "562", ["ryhB"])]}))
    assert row(p)["nucleotide_type_unknown_lookup"] == "refused_by_a_source"


# ---- F5: more phrases than the cap are refused, not cut ----------------------------------------------------------------------
def test_f5_too_many_phrases_is_refused_and_nothing_is_looked_up() -> None:
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta"]
    text = " ".join(f"Type II {w} cells." for w in words)
    f = FakeFetcher([])
    assert len(words) > MAX_PHRASES and numbered_context(f, "Type II cell", [("paper", text)]) == (None, "too_many_phrases") and f.calls == []
    few = " ".join(f"Type II {w} cells." for w in words[:MAX_PHRASES])
    assert numbered_context(FakeFetcher([]), "Type II cell", [("paper", few)])[1] == "no_exact_phrase"  # at the cap the phrases are looked up
    p = {"entities": {"cell_types": [{"name": "Type II cell"}], "species": []}, "biological_states": []}
    rep = fill_ontology_ids(p, FakeFetcher([]), Cache(None), TODAY, paper_text=text)
    assert rep["extra"]["numbered_context_refused"] == {"too_many_phrases": 1}


# ---- F6: a row with its own exact CL label claims the id before a context-resolved numbered name ---------------------------
_TASTE = "Type II foo cells detect sweet. Type II foo cells release ATP. Type III cells have synapses."


def test_f6_exact_row_wins_over_the_context_resolved_numbered_name_whatever_the_list_order() -> None:
    for names in (["Type II cell", "Type II foo cell"], ["Type II foo cell", "Type II cell"]):  # (not a PathWhiz row, so it needs the lookup)
        p = {"entities": {"cell_types": [{"name": n} for n in names], "species": []}, "biological_states": []}
        term = ols(("CL:0009999", "type II foo cell", []))
        f = FakeFetcher([("ols4", {"q": "Type II foo cell"}, term), ("ols4", {"q": "type ii foo cell"}, term)])  # the row name; the paper phrase
        rep = fill_ontology_ids(p, f, Cache(None), TODAY, paper_text=_TASTE)
        got = {r["name"]: r.get("ontology_id") for r in p["entities"]["cell_types"]}
        assert got == {"Type II foo cell": "CL:0009999", "Type II cell": None}, names
        assert rep["extra"]["id_used_by_other_row"] == 1


# ---- F7: a Cellosaurus failure does not stop the OLS lookups -----------------------------------------------------------------
def test_f7_cellosaurus_fetch_error_does_not_stop_ols() -> None:
    p = {"entities": {"cell_types": [{"name": "MG-63"}, {"name": "neuron x"}], "species": [{"name": "Homo sapiens", "taxonomy_id": "9606"}]},
         "biological_states": []}
    f = FakeFetcher([("cellosaurus", None, FetchError("HTTP 400")), ("ols4", {"q": "neuron x"}, ols(("CL:0000999", "neuron x", [])))])
    rep = fill_ontology_ids(p, f, Cache(None), TODAY)
    assert p["entities"]["cell_types"][1]["ontology_id"] == "CL:0000999" and "cellosaurus_id" not in p["entities"]["cell_types"][0]
    assert rep["extra"]["cellosaurus_refused"] == {"fetch_error": 1} and rep["counts"]["skipped_after_fetch_error"] == 0
    assert rep["counts"]["resolved"] == 1


def test_f7_an_ols_failure_still_stops_the_remaining_lookups() -> None:
    p = {"entities": {"cell_types": [{"name": "aaa x"}, {"name": "bbb x"}], "species": []}, "biological_states": []}
    rep = fill_ontology_ids(p, FakeFetcher([("ols4", None, FetchError("down"))]), Cache(None), TODAY)
    assert rep["counts"]["fetch_error"] == 1 and rep["counts"]["skipped_after_fetch_error"] == 1


# ---- F8: Prokaryote with an unlisted genus is undetermined -------------------------------------------------------------------
def test_f8_unlisted_prokaryote_genus_is_undetermined_and_listed_ones_still_map() -> None:
    assert organism_kind("Unknownus bacter", "Prokaryote") is None and organism_kind("Unknownus bacter") is None
    assert organism_kind("Escherichia coli", "Prokaryote") == "bacterium" and organism_kind("Pseudomonas putida", "Prokaryote") == "bacterium"
    assert organism_kind("E. coli", "Prokaryote") == "bacterium" and organism_kind("Methanosarcina mazei", "Prokaryote") == "archaeon"
    assert organism_kind("Pseudomonas aeruginosa", None) == "bacterium"
    kinds = organism_kinds([("Unknownus bacter", "Prokaryote")])
    assert kinds == {"unknown"} and map_vocabulary_name("subcellular-location", "outer membrane", None, kinds) == ("outer membrane", None, "unmapped")


# ---- F9: one fetcher (one budget) for the three network steps ----------------------------------------------------------------
def test_f9_the_three_steps_share_one_fetcher(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    built, seen = [], {}
    sentinel = object()
    monkeypatch.setattr(fp, "_live_fetcher", lambda: built.append(1) or sentinel)
    monkeypatch.setattr(fp, "apply_final_id_fill", lambda payload, meta, **kw: seen.__setitem__("fill", kw.get("fetcher")) or {})
    monkeypatch.setattr(fp, "maybe_apply_paper_nucleic_types", lambda payload, state=None, **kw: seen.__setitem__("nucleic", kw.get("fetcher")) or {})
    monkeypatch.setattr(fp, "maybe_apply_final_pathway_meta", lambda payload, ctx=None, state=None, **kw: seen.__setitem__("meta", kw.get("fetcher")) or {})
    fp.maybe_apply_final_id_fill({}, {}, {})
    assert built == [1] and seen == {"fill": sentinel, "nucleic": sentinel, "meta": sentinel}
    mine = object()  # a caller-given fetcher is used as it is, and no live one is built
    fp.maybe_apply_final_id_fill({}, {}, {}, fetcher=mine)
    assert built == [1] and seen == {"fill": mine, "nucleic": mine, "meta": mine}
