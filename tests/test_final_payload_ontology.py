"""Final payload P4 (D-W20a): OLS4 ontology ids for location / cell-type / tissue names. No network: fakes only."""

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
from t2pw.pwml.vocabulary_map import map_vocabulary_name  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402
from t2pw.service.final_payload import FINAL_PAYLOAD_FILL_FIELD, apply_final_id_fill  # noqa: E402
from t2pw.service.final_payload_ontology import fill_ontology_ids, ols_exact  # noqa: E402

TODAY = "2026-11-12"


def ols(*docs):
    return {"response": {"docs": [{"obo_id": i, "label": lab, "synonym": syn} for i, lab, syn in docs]}}


def term(onto, num, exact=(), related=()):
    """Route for the OLS4 v2 term endpoint of one term, shaped like the live answer (value objects for annotated synonyms)."""
    from t2pw.service.final_payload_ontology import EXACT_SYNONYM_KEY
    key = f"classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252F{onto.upper()}_{num}"
    body = {EXACT_SYNONYM_KEY: [{"type": ["reification"], "value": x, "axioms": []} if i else x for i, x in enumerate(exact)],
            "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym": list(related)}
    return (key, None, body)


def _payload():
    return {
        "entities": {
            "subcellular_locations": [{"name": "thylakoid membrane"}, {"name": "saliva"}, {"name": "stroma"},
                                      {"name": "cytosol"}, {"name": "ER"}, {"name": "nucleus", "ontology_id": "GO:0005634"},
                                      {"name": "weird place"}],
            "cell_types": [{"name": "MG-63"}],
            "tissues": [{"name": "root"}],
        },
        "biological_states": [{"name": "s", "subcellular_location": "state only place", "tissue": "root"}],
    }


def _fake():
    return FakeFetcher([
        ("ols4", {"q": "thylakoid membrane", "ontology": "go"}, ols(("GO:0042651", "thylakoid membrane", []),
                                                                      ("GO:0009579", "thylakoid", ["thylakoid membrane x"]))),
        ("ols4", {"q": "saliva"}, ols(("GO:0000001", "saliva", []), ("GO:0000002", "other", ["saliva"]))),  # two distinct: ambiguous
        ("ols4", {"q": "stroma"}, ols(("GO:0009570", "chloroplast stroma", ["Stroma"]))),  # exact synonym, case folded
        ("ols4", {"q": "state only place"}, ols(("GO:0000077", "state only place", []))),
        ("ols4", {"q": "weird place"}, ols(("GO:0000099", "weirder place", ["close but no"]))),  # fuzzy hit: not exact
        ("ols4", {"q": "MG-63", "ontology": "cl"}, ols(("CL:0000999", "mg-63", []), ("UBERON:1", "mg-63", []))),
        ("ols4", {"q": "root", "ontology": "bto"}, ols(("BTO:0001188", "root", []))),
        term("go", "0009570", exact=["Stroma"]),
        term("go", "0000002", exact=["saliva"]),
    ])


def test_ols_exact_label_synonym_and_prefix_filter() -> None:
    f = _fake()
    assert ols_exact(f, "thylakoid membrane", "go", "GO:", None) == [
        {"id": "GO:0042651", "label": "thylakoid membrane", "match_type": "exact_label"}]
    assert ols_exact(f, "stroma", "go", "GO:", None)[0]["match_type"] == "exact_synonym"
    assert ols_exact(f, "weird place", "go", "GO:", None) == []  # no exact label or synonym: the fuzzy hit is never taken
    assert [h["id"] for h in ols_exact(f, "MG-63", "cl", "CL:", None)] == ["CL:0000999"]  # UBERON hit filtered
    assert f.calls[0][1]["exact"] == "true" and "allChildrenOf" not in f.calls[0][1]
    ols_exact(f, "x", "go", "GO:", "http://purl.obolibrary.org/obo/GO_0005575")
    assert f.calls[-1][1]["allChildrenOf"].endswith("GO_0005575")


def test_fill_writes_only_unresolved_names_and_refuses_ambiguous() -> None:
    p = _payload()
    rep = fill_ontology_ids(p, _fake(), Cache(None), TODAY)
    rows = {r["name"]: r.get("ontology_id") for r in p["entities"]["subcellular_locations"]}
    assert rows == {"thylakoid membrane": "GO:0042651", "saliva": None, "stroma": "GO:0009570", "cytosol": None, "ER": None,
                    "nucleus": "GO:0005634", "weird place": None, "state only place": "GO:0000077"}  # state-only name got a row
    assert p["entities"]["cell_types"][0]["ontology_id"] == "CL:0000999"
    assert p["entities"]["tissues"][0]["ontology_id"] == "BTO:0001188"
    assert rep["counts"] == {"looked_up": 7, "resolved": 5, "ambiguous": 1, "no_exact_match": 1, "fetch_error": 0,
                             "skipped_after_fetch_error": 0}  # M2/minor: one new counter
    assert {r["name"]: r["reason"] for r in rep["refused"]} == {"saliva": "ambiguous", "weird place": "no_exact_match"}
    e = next(x for x in rep["entries"] if x["entity"] == "stroma")
    assert e == {"kind": "location", "entity": "stroma", "field": "ontology_id", "accession": "GO:0009570", "source": "OLS4",
                 "label": "chloroplast stroma", "match_type": "exact_synonym", "retrieved": TODAY}


def test_fetch_error_is_counted_not_raised_and_unresolved_state_only_names_leave_no_row() -> None:
    p = {"entities": {"tissues": []}, "biological_states": [{"tissue": "bark"}]}
    rep = fill_ontology_ids(p, FakeFetcher([("ols4", None, FetchError("down"))]), Cache(None), TODAY)
    assert rep["counts"]["fetch_error"] == 1 and p["entities"]["tissues"] == []


def test_apply_final_id_fill_records_provenance_and_cache_serves_second_run() -> None:
    p = _payload()
    cache = Cache(None)
    rep = apply_final_id_fill(p, {}, fetcher=_fake(), cache=cache, today=TODAY)
    assert rep["error"] is None and rep["ontology"]["counts"]["resolved"] == 5
    prov = p[FINAL_PAYLOAD_FILL_FIELD]
    assert prov["written"] == 5 and prov["by_field"]["location.ontology_id"] == 3 and prov["by_field"]["tissue.ontology_id"] == 1
    assert {e["accession"] for e in prov["entries"]} >= {"GO:0042651", "CL:0000999", "BTO:0001188"}
    f2 = FakeFetcher([])
    q = _payload()
    rep2 = apply_final_id_fill(q, {}, fetcher=f2, cache=cache, today="2026-11-13")
    assert [c for c in f2.calls if c[1].get("ontology") in ("go", "cl", "bto")] == []  # all from the cache
    assert q["entities"]["subcellular_locations"][0]["ontology_id"] == "GO:0042651"
    assert q[FINAL_PAYLOAD_FILL_FIELD]["entries"][0]["retrieved"] == TODAY  # the cached entry keeps its own retrieval date
    assert rep2["ontology"]["cache_hits"] > 0


def test_id_that_pathwhiz_already_holds_maps_to_its_row_without_the_id() -> None:
    assert map_vocabulary_name("subcellular-location", "plasma membrane x", "GO:0005886") == ("Cell Membrane", None, "mapped_by_id")
    assert map_vocabulary_name("tissue", "root", "BTO:0001188")[2] == "new_row"


_TISSUE = [{"name": "root", "pathwhiz_id": 5}]
_STATES = [{"name": "s", "species": "Homo sapiens", "tissue": "root", "subcellular_location": "thylakoid membrane"},
           {"name": "t", "species": "Homo sapiens", "tissue": "root", "subcellular_location": "cytosol"}]


def _build(loc_rows, states=None, tissues=None):
    payload = {"entities": {"species": [{"name": "Homo sapiens", "pathwhiz_id": 1}], "subcellular_locations": loc_rows,
                            "tissues": tissues or [],
                            "compounds": [{"name": "A", "pathbank_compound_id": 103}]},
               "biological_states": states or _STATES,
               "processes": {"reactions": [], "transports": [], "interactions": []}}
    ir, report = build_pwml_ir(payload, strict_db=True)
    assert report["errors"] == []
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000", height=1400,
                           width=3200, background_color="#FFFFFF", ref=str(ROOT / "reference" / "PW000001.pwml"))
    b = DeterministicPwmlBuilder(extraction=ir, signature=discover_structure_signature(ROOT / "reference" / "PW000001.pwml"), args=args)
    return b, b.build().root


def _loc_names(root):
    return [n.findtext("name") for n in root.findall(".//subcellular-locations/subcellular-location")]


def test_writer_drops_idless_location_and_keeps_state_flag_on_only(monkeypatch) -> None:
    # M1 fix: the thylakoid row now carries no database id (a row with pathwhiz_id is never dropped, see the next test)
    rows = [{"name": "thylakoid membrane"}, {"name": "cytosol", "pathwhiz_id": 3}]
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    b, root = _build(copy.deepcopy(rows), tissues=_TISSUE)
    assert _loc_names(root) == ["thylakoid membrane", "cytosol"] and "locations_dropped_no_id" not in b.layout_debug_counts
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    b, root = _build(copy.deepcopy(rows), tissues=_TISSUE)
    assert _loc_names(root) == ["cytosol"] and b.layout_debug_counts["locations_dropped_no_id"] == 1
    states = root.findall(".//biological-states/biological-state")
    assert len(states) == 2 and sum(1 for s in states if (s.findtext("subcellular-location-id") or "").strip()) == 1
    # the same location with an id found online is kept, with its id
    b, root = _build([{"name": "thylakoid membrane", "ontology_id": "GO:0042651"}, rows[1]], tissues=_TISSUE)
    got = {n.findtext("name"): n.findtext("ontology-id") or None for n in root.findall(".//subcellular-locations/subcellular-location")}
    assert got == {"thylakoid membrane": "GO:0042651", "cytosol": None} and "locations_dropped_no_id" not in b.layout_debug_counts


def _state_locs(root):
    return {st.findtext("name"): (st.findtext("subcellular-location-id") or "").strip() for st in root.findall(".//biological-states/biological-state")}


def test_row_with_database_id_is_never_dropped(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    b, root = _build([{"name": "thylakoid membrane", "pathwhiz_id": 4}, {"name": "cytosol", "pathwhiz_id": 3}])
    assert _loc_names(root) == ["thylakoid membrane", "cytosol"] and "locations_dropped_no_id" not in b.layout_debug_counts
    assert all(_state_locs(root).values())


def test_idless_location_kept_when_dropping_would_make_two_states_identical(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    states = [{"name": "s", "species": "Homo sapiens", "tissue": "root", "subcellular_location": "thylakoid membrane"},
              {"name": "t", "species": "Homo sapiens", "tissue": "root", "subcellular_location": "stroma x"},
              {"name": "u", "species": "Homo sapiens", "tissue": "root", "subcellular_location": "cytosol"}]
    rows = [{"name": "thylakoid membrane"}, {"name": "stroma x"}, {"name": "cytosol", "pathwhiz_id": 3}]
    b, root = _build(copy.deepcopy(rows), states, tissues=_TISSUE)
    # s and t differ only by an id-less location: dropping both would make them identical (PathWhiz uniqueness), so both stay
    assert _loc_names(root) == ["thylakoid membrane", "stroma x", "cytosol"]
    assert b.layout_debug_counts["locations_kept_state_collision"] == 2 and "locations_dropped_no_id" not in b.layout_debug_counts
    ids = _state_locs(root)
    assert len(set(ids.values())) == 3 and all(ids.values())
    # no collision: s differs from u by a kept location; the unrelated id-less location is dropped
    b, root = _build(copy.deepcopy(rows), [states[0], states[2]], tissues=_TISSUE)
    assert _loc_names(root) == ["cytosol"] and b.layout_debug_counts["locations_dropped_no_id"] == 2
    assert "locations_kept_state_collision" not in b.layout_debug_counts

def test_idless_location_kept_for_species_only_state_dropped_with_tissue(monkeypatch) -> None:
    # d2 idx 10: species alone is a wildcard in the PathWhiz importer, so a state whose only other component is an id-less
    # location keeps that location by name; with a tissue the state is still distinguishable and the location is dropped
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    rows = [{"name": "saliva"}, {"name": "cytosol", "pathwhiz_id": 3}]
    only = [{"name": "s", "species": "Homo sapiens", "subcellular_location": "saliva"},
            {"name": "t", "species": "Homo sapiens", "subcellular_location": "cytosol"}]
    b, root = _build(copy.deepcopy(rows), only)
    assert _loc_names(root) == ["saliva", "cytosol"] and all(_state_locs(root).values())
    assert b.layout_debug_counts["locations_kept_state_collision"] == 1 and "locations_dropped_no_id" not in b.layout_debug_counts
    withtissue = [dict(only[0], tissue="root"), only[1]]
    b, root = _build(copy.deepcopy(rows), withtissue, tissues=_TISSUE)
    assert _loc_names(root) == ["cytosol"] and b.layout_debug_counts["locations_dropped_no_id"] == 1


def test_synonym_hit_needs_the_name_in_the_terms_exact_synonyms(monkeypatch) -> None:
    # nerve fiber is only a RELATED synonym of GO:0043005 (live OLS4, 2026-11-12): the search index lists it, the term says related
    f = FakeFetcher([("ols4", {"q": "nerve fiber"}, ols(("GO:0043005", "neuron projection", ["nerve fiber", "neurite"]))),
                     term("go", "0043005", exact=["neuron process"], related=["nerve fiber"])])
    assert ols_exact(f, "nerve fiber", "go", "GO:", None) == []
    f = FakeFetcher([("ols4", {"q": "neurite"}, ols(("GO:0043005", "neuron projection", ["neurite"]))),
                     term("go", "0043005", exact=["neurite"])])
    assert ols_exact(f, "neurite", "go", "GO:", None) == [{"id": "GO:0043005", "label": "neuron projection", "match_type": "exact_synonym"}]
    f = FakeFetcher([("ols4", {"q": "neurite"}, ols(("GO:0043005", "neuron projection", ["neurite"])))])  # term unknown (404): unverified
    assert ols_exact(f, "neurite", "go", "GO:", None) == []
    f = FakeFetcher([("ols4", {"q": "neurite"}, ols(("GO:0043005", "neuron projection", ["neurite"]))),
                     ("GO_0043005", None, FetchError("down"))])  # term fetch fails: the error reaches the caller, which refuses
    try:
        ols_exact(f, "neurite", "go", "GO:", None)
        raise AssertionError("expected FetchError")
    except FetchError:
        pass
    # label matches need no term fetch
    f = FakeFetcher([("ols4", {"q": "root"}, ols(("BTO:0001188", "root", [])))])
    assert ols_exact(f, "root", "bto", "BTO:", None)[0]["match_type"] == "exact_label" and len(f.calls) == 1


def test_term_fetch_error_refuses_and_skips_the_remaining_lookups() -> None:
    p = {"entities": {"tissues": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}, "biological_states": []}
    f = FakeFetcher([("ols4", {"q": "a"}, ols(("BTO:0000001", "x", ["a"]))), ("BTO_0000001", None, FetchError("down"))])
    rep = fill_ontology_ids(p, f, Cache(None), TODAY)
    assert rep["counts"] == {"looked_up": 1, "resolved": 0, "ambiguous": 0, "no_exact_match": 0, "fetch_error": 1,
                             "skipped_after_fetch_error": 2}
    assert [r["reason"] for r in rep["refused"]][1:] == ["skipped_after_fetch_error"] * 2
    assert not any(r.get("ontology_id") for r in p["entities"]["tissues"]) and len(f.calls) == 2


def test_written_states_stay_distinct_whenever_the_input_states_were(monkeypatch) -> None:
    """The proof for 'keeping a location only adds distinctions': over several state sets, flag on writes distinct location ids
    for exactly the states that were distinct by location name before, whatever mix of id-less and id-bearing rows."""
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    rows = [{"name": "a place"}, {"name": "b place"}, {"name": "c place"}, {"name": "cytosol", "pathwhiz_id": 903},  # 903: apart from the fallback ids 1, 2, 3
            {"name": "d place", "ontology_id": "GO:0000044"}]
    names = [r["name"] for r in rows]
    import itertools
    for k in range(1, 6):
        for combo in itertools.combinations(names, k):
            states = [{"name": f"s{i}", "species": "Homo sapiens", "subcellular_location": n} for i, n in enumerate(combo)]
            _, root = _build(copy.deepcopy(rows), states)
            written = [v for v in _state_locs(root).values()]
            if len(combo) == 1:
                continue  # one state: nothing to collide with (the species alone still identifies it)
            # every state still differs from every other: unique location ids, except dropped ones which may be at most one
            assert len([w for w in written if w]) == len(set(w for w in written if w)), combo
            assert len([w for w in written if not w]) <= 1, combo
