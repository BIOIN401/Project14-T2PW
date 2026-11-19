"""Lane loc/assign: every element and transport where the paper says; where it does not, an inferred place that is always
flagged and never overrides a stated one. L7 compares transport ends by place and never fills an export's destination
from neighbours; leaflet spellings merge (inner and outer membrane never do); Stage-0 hint evidence is flagged; bacteria
get the cytoplasm as a flagged organism default; the LLM pass (faked here) accepts only a closed-vocabulary place with a
verbatim quote that names it."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2pw.service import final_payload_locations as FL  # noqa: E402
from t2pw.service.final_payload_compartments import estimate_compartments  # noqa: E402
from t2pw.service.final_payload_merge import location_key  # noqa: E402

ECOLI = (("Escherichia coli", "Prokaryote", "562"),)
HUMAN = (("Homo sapiens", "Eukaryote", "9606"),)


def _rx(name, ins, outs, state=None, **kw):
    row = {"name": name, "inputs": list(ins), "outputs": list(outs), "enzymes": [], "evidence": f"paper says {name}"}
    if state is not None:
        row["biological_state"] = state
    row.update(kw)
    return row


def _st(name, loc, species="Escherichia coli", **kw):
    return {"name": name, "species": species, "subcellular_location": loc, **kw}


def _tr(name, cargo, f=None, to=None, **kw):
    row = {"name": name, "cargo": cargo, "evidence": f"paper says {name}"}
    if f is not None:
        row["from_biological_state"] = f
    if to is not None:
        row["to_biological_state"] = to
    row.update(kw)
    return row


def _payload(states, reactions=(), transports=(), species=ECOLI, elements=()):
    return {"biological_states": states,
            "entities": {"species": [{"name": n, "classification": c, "taxonomy_id": t} for n, c, t in species],
                         "subcellular_locations": [{"name": s["subcellular_location"]} for s in states]},
            "processes": {"reactions": list(reactions), "transports": list(transports), "interactions": []},
            "element_locations": {"compound_locations": list(elements)}}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr("t2pw.clean_model.fill.cache.Cache._path", lambda self, key: None)  # memory only
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD_BUDGET_S", raising=False)
    monkeypatch.delenv(FL.LLM_ENV, raising=False)
    monkeypatch.setenv("T2PW_OFFLINE_CURATOR", "1")


def _state_of(p, ref):
    return next(s for s in p["biological_states"] if s["name"] == ref)


# ---- consistency fixes -------------------------------------------------------------------------------------------------
def test_leaflet_spellings_share_one_key_and_inner_outer_never_merge():
    same = ["cytoplasmic leaflet of inner membrane", "cytoplasmic leaflet of the IM", "inner membrane cytoplasmic leaflet",
            "the cytoplasmic leaflet of the inner membrane state"]
    assert len({location_key(s, True) for s in same}) == 1
    assert location_key("periplasmic leaflet of the IM", True) != location_key("cytoplasmic leaflet of the IM", True)
    assert location_key("outer leaflet of the inner membrane", True) != location_key("inner leaflet of the outer membrane", True)
    assert location_key("outer leaflet of the OM", True) == location_key("outer membrane outer leaflet", True)
    assert location_key("inner membrane", True) != location_key("outer membrane", True)


def test_leaflet_variants_merge_in_the_location_step():
    p = _payload([_st("cytoplasmic state", "cytoplasm"),
                  _st("cytoplasmic leaflet of IM state", "cytoplasmic leaflet of the inner membrane"),
                  _st("E. coli inner membrane cytoplasmic leaflet", "inner membrane cytoplasmic leaflet"),
                  _st("outer leaflet of OM state", "outer leaflet of the outer membrane")],
                 [_rx("LpxH", ["UDP-DAGn"], ["lipid X"], "cytoplasmic leaflet of IM state"),
                  _rx("LpxB", ["lipid X"], ["DS-1-P"], "E. coli inner membrane cytoplasmic leaflet")])
    FL.assign_locations(p)
    names = [s["name"] for s in p["biological_states"]]
    assert names == ["cytoplasmic state", "cytoplasmic leaflet of IM state", "outer leaflet of OM state"]
    assert {r["biological_state"] for r in p["processes"]["reactions"]} == {"cytoplasmic leaflet of IM state"}


def test_l7_compares_transport_ends_by_place_and_never_fills_an_export_destination():
    # menaquinone: "DHNA export" cytoplasm -> ?; the cargo's other reactions sit in "intracellular" (= the cytoplasm)
    states = [_st("L. lactis intracellular state", "intracellular", "Lactococcus lactis"),
              _st("cytoplasmic state", "cytoplasm", "Lactococcus lactis")]
    rx = [_rx("MenB", ["OSB-CoA"], ["DHNA-CoA"], "L. lactis intracellular state"),
          _rx("thioesterase", ["DHNA-CoA"], ["DHNA"], "L. lactis intracellular state")]
    sp = (("Lactococcus lactis", "Prokaryote", "1358"),)
    p = _payload(copy.deepcopy(states), rx, [_tr("DHNA export", "DHNA", "cytoplasmic state")], sp)
    estimate_compartments(p)
    assert "to_biological_state" not in p["processes"]["transports"][0]
    p = _payload(copy.deepcopy(states), copy.deepcopy(rx), [_tr("DHNA transfer", "DHNA", "cytoplasmic state")], sp)
    estimate_compartments(p)  # not an export: still no fill, the neighbours are the same place as the stated side
    assert "to_biological_state" not in p["processes"]["transports"][0]


def test_an_old_neighbour_fill_of_an_export_is_replaced_and_the_synonym_states_merge():
    sp = (("Lactococcus lactis", "Prokaryote", "1358"),)
    t = _tr("DHNA export", "DHNA", "cytoplasmic state", "L. lactis intracellular state", compartment_source="neighbour_estimate",
            compartment_estimate=[{"step": "L7", "field": "to_biological_state", "from_state": None,
                                   "to_state": "L. lactis intracellular state", "source": "neighbour_estimate", "reason": "r"}])
    p = _payload([_st("L. lactis intracellular state", "intracellular", "Lactococcus lactis"),
                  _st("cytoplasmic state", "cytoplasm", "Lactococcus lactis"),
                  _st("L. lactis extracellular state", "extracellular", "Lactococcus lactis")],
                 [_rx("thioesterase", ["DHNA-CoA"], ["DHNA"], "L. lactis intracellular state")], [t], sp)
    rep = FL.assign_locations(p)
    t = p["processes"]["transports"][0]
    assert t["to_biological_state"] == "L. lactis extracellular state"
    assert t["inferred_sides"] == {"to_biological_state": "organism_default"}
    assert t["from_biological_state"] == "L. lactis intracellular state"  # merged: same place, now no transport end keeps them apart
    assert [s["name"] for s in p["biological_states"]] == ["L. lactis intracellular state", "L. lactis extracellular state"]
    assert rep["counts"]["organism_default"] == 1 and not rep["same_place_left"]


def test_stage0_hint_evidence_is_flagged_not_moved():
    p = _payload([_st("E. coli cytoplasm", "cytoplasm", evidence="Expected compartments: cytoplasm")],
                 [_rx("EntC", ["chorismate"], ["isochorismate"], "E. coli cytoplasm")])
    FL.assign_locations(p)
    s = p["biological_states"][0]
    assert s["evidence_is_stage0_hint"] and s["review_required"] and s["subcellular_location"] == "cytoplasm"
    assert p[FL.FIELD]["hint_states"][0]["state"] == "E. coli cytoplasm"


# ---- bacterial organism default ----------------------------------------------------------------------------------------
def test_bacterial_unplaced_reactions_and_elements_go_to_the_cytoplasm_flagged():
    p = _payload([_st("E. coli periplasmic state", "periplasm"), _st("__auto_state__", "cell")],
                 [_rx("EntC", ["chorismate"], ["isochorismate"]),
                  _rx("EntB", ["isochorismate"], ["DHB"], "__auto_state__"),
                  _rx("FepB", ["Fe-Ent"], ["Fe-Ent"], "E. coli periplasmic state")],
                 elements=[{"compound": "chorismate", "biological_state": "__auto_state__"},
                           {"compound": "Fe-Ent", "biological_state": "E. coli periplasmic state"}])
    stated = copy.deepcopy(p["processes"]["reactions"][2])
    FL.assign_locations(p)
    rx = p["processes"]["reactions"]
    new = rx[0]["biological_state"]
    assert new == rx[1]["biological_state"] == "Escherichia coli cytoplasm"
    assert _state_of(p, new)["subcellular_location"] == "cytoplasm"
    for r in rx[:2]:
        assert r["inferred"] == "organism_default" and r["compartment_source"] == "organism_default" and r["review_required"]
        assert r["compartment_estimate"][-1]["reason"]
    assert rx[2] == stated  # a stated place never moves
    el = p["element_locations"]["compound_locations"]
    assert el[0]["biological_state"] == new and el[0]["inferred"] == "organism_default"
    assert "inferred" not in el[1]


def test_an_existing_plain_cytoplasm_state_is_used():
    p = _payload([_st("E. coli state", "cell")], [_rx("EntC", ["chorismate"], ["isochorismate"])])
    FL.assign_locations(p)
    assert p["processes"]["reactions"][0]["biological_state"] == "E. coli state"
    assert len(p["biological_states"]) == 1


def test_eukaryotes_get_no_default():
    p = _payload([_st("cytosolic state", "cytosol", "Homo sapiens")], [_rx("X", ["a"], ["b"])], species=HUMAN)
    FL.assign_locations(p)
    assert "biological_state" not in p["processes"]["reactions"][0]
    assert FL.FIELD not in p


def test_bacterial_auto_auto_transport_gets_a_direction_from_its_name_and_a_stated_side_never_moves():
    p = _payload([_st("__auto_state__", "cell"), _st("cytoplasmic state", "cytoplasm")], [],
                 [_tr("TubI-mediated TBN export", "TBN", "__auto_state__", "__auto_state__"),
                  _tr("ferric-enterobactin import", "Fe-Ent", "__auto_state__", "cytoplasmic state"),
                  _tr("enterobactin secretion", "Ent", "cytoplasmic state", "cytoplasmic state")])
    rep = FL.assign_locations(p)
    t0, t1, t2 = p["processes"]["transports"]
    assert (_state_of(p, t0["from_biological_state"])["subcellular_location"],
            _state_of(p, t0["to_biological_state"])["subcellular_location"]) == ("cytoplasm", "extracellular space")
    assert t0["inferred_sides"] == {"from_biological_state": "organism_default", "to_biological_state": "organism_default"}
    assert _state_of(p, t1["from_biological_state"])["subcellular_location"] == "extracellular space"
    assert t1["to_biological_state"] == "cytoplasmic state" and list(t1["inferred_sides"]) == ["from_biological_state"]
    # both ends stated and the same: never changed, flagged
    assert (t2["from_biological_state"], t2["to_biological_state"]) == ("cytoplasmic state", "cytoplasmic state")
    assert "inferred_sides" not in t2 and any("transport_ends_same_place" in r for r in t2["review_reasons"])
    assert [x["transport"] for x in rep["same_place_left"]] == ["enterobactin secretion"]


def test_runs_once_and_fails_open(monkeypatch):
    p = _payload([], [_rx("EntC", ["chorismate"], ["isochorismate"])])
    FL.assign_locations(p)
    once = copy.deepcopy(p)
    assert FL.assign_locations(p)["skipped"] == "already_assigned" and p == once
    q = _payload([], [_rx("EntC", ["chorismate"], ["isochorismate"])])
    before = copy.deepcopy(q)
    monkeypatch.setattr(FL, "_bacterial_default", lambda *a: 1 / 0)
    assert FL.assign_locations(q)["error"].startswith("ZeroDivisionError") and q == before


# ---- LLM pass ----------------------------------------------------------------------------------------------------------
PAPER = ("Enterobactin is assembled by EntF in the cytoplasm of the cell. The siderophore is then secreted into the "
         "extracellular medium by EntS. The pathway of TBN uptake is not described. Unlike fungi, the bacterium has no "
         "nucleus; EntF works alone.")


def _fake(reply, calls):
    def chat(messages, **kw):
        calls.append((messages, kw))

        class R:
            text = json.dumps(reply)
        return R()
    return chat


def test_llm_pass_off_by_default_never_calls():
    calls = []
    p = _payload([_st("cytosolic state", "cytosol", "Homo sapiens")], [_rx("X", ["a"], ["b"])], species=HUMAN)
    rep = FL.assign_locations(p, {"final_payload_paper_text": PAPER}, chat_fn=_fake({"items": []}, calls))
    assert calls == [] and rep["llm"]["skipped"] == "off"


def test_llm_pass_accepts_only_a_place_naming_quote_from_the_vocabulary(monkeypatch):
    monkeypatch.setenv(FL.LLM_ENV, "1")
    calls = []
    reply = {"items": [
        {"id": 0, "state": None, "new_location": "cytoplasm", "quote": "Enterobactin is assembled by EntF in the cytoplasm"},
        {"id": 1, "state": None, "new_location": "cytoplasm", "quote": "The pathway of TBN uptake is not described"},
        {"id": 2, "state": None, "new_location": "nucleus", "quote": "the bacterium has no nucleus; EntF works alone"},
        {"id": 3, "to": {"state": None, "new_location": "extracellular space",
                         "quote": "secreted into the extracellular medium by EntS"}}]}
    p = _payload([_st("E. coli periplasmic state", "periplasm"), _st("cytoplasmic state", "cytoplasm")],
                 [_rx("EntF step", ["DHB-AMP"], ["enterobactin"], enzymes=["EntF"]),
                  _rx("TBN step", ["TBN"], ["TBN-P"]),
                  _rx("EntF other", ["serine"], ["Ser-EntF"], enzymes=["EntF"]),
                  _rx("stated", ["x"], ["y"], "E. coli periplasmic state")],
                 [_tr("enterobactin secretion", "enterobactin", "cytoplasmic state", transporters=["EntS"])])
    rep = FL.assign_locations(p, {"final_payload_paper_text": PAPER}, chat_fn=_fake(reply, calls))
    assert len(calls) == 1 and calls[0][1]["model_override"] == FL.LLM_MODEL
    user = json.loads(calls[0][0][1]["content"])
    assert user["new_location_vocabulary"] == FL.VOCABULARY["bacterium"]
    assert [it["name"] for it in user["items"]] == ["EntF step", "TBN step", "EntF other", "enterobactin secretion"]
    rx = p["processes"]["reactions"]
    assert rx[0]["biological_state"] == "cytoplasmic state" and rx[0]["inferred"] == "llm_quote"
    assert "cytoplasm" in rx[0]["compartment_estimate"][-1]["quote"]
    # no place in the quote / not in the vocabulary: rejected, then the bacterial default (flagged as such)
    assert rx[1]["inferred"] == rx[2]["inferred"] == "organism_default"
    assert rx[3]["biological_state"] == "E. coli periplasmic state" and "inferred" not in rx[3]
    t = p["processes"]["transports"][0]
    assert _state_of(p, t["to_biological_state"])["subcellular_location"] == "extracellular space"
    assert t["inferred_sides"] == {"to_biological_state": "llm_quote"} and t["compartment_source"] == "llm_estimate"
    reasons = {r["reaction"]: r["reason"] for r in rep["llm"]["rejected"] if "reaction" in r}
    assert "TBN step" in reasons and "vocabulary" in reasons["EntF other"]


def test_llm_quote_may_name_the_item_by_a_synonym_or_a_currency_cargo(monkeypatch):
    monkeypatch.setenv(FL.LLM_ENV, "1")
    paper = "The intracellular and extracellular DHNA accumulation rate was measured. Protons pass through OTOP1 channels."
    reply = {"items": [{"id": 0, "state": None, "new_location": None, "quote": None},  # reactions come first
                       {"id": 1, "to": {"state": "L. lactis extracellular state", "quote": "extracellular DHNA accumulation rate"}}]}
    sp = (("Lactococcus lactis", "Prokaryote", "1358"),)
    p = _payload([_st("cytoplasmic state", "cytoplasm", "Lactococcus lactis"),
                  _st("L. lactis extracellular state", "extracellular", "Lactococcus lactis")],
                 [_rx("unknown step", ["a"], ["b"])],
                 [_tr("DHNA export", "1,4-dihydroxy-2-naphthoic acid", "cytoplasmic state")], sp)
    p["entities"]["compounds"] = [{"name": "1,4-dihydroxy-2-naphthoic acid", "synonyms": ["{'-': 'DHNA'}"]}]
    rep = FL.assign_locations(p, {"final_payload_paper_text": paper}, chat_fn=_fake(reply, []))
    t = p["processes"]["transports"][0]
    assert t["to_biological_state"] == "L. lactis extracellular state" and t["inferred_sides"] == {"to_biological_state": "llm_quote"}
    assert rep["llm"]["rejected"] == []  # a null answer is no answer, not a rejection


def test_every_vocabulary_term_is_a_pathwhiz_location():
    from t2pw.pwml.vocabulary_map import map_vocabulary_name

    for kind, terms in FL.VOCABULARY.items():
        for term in terms:
            assert map_vocabulary_name("subcellular-location", term, None)[2] in ("exact", "mapped"), (kind, term)


# ---- text record -------------------------------------------------------------------------------------------------------
def test_text_record_lists_every_inferred_location():
    from t2pw.app.reaction_evidence import _inferred_locations

    p = _payload([_st("E. coli cytoplasm", "cytoplasm", evidence="Expected compartments: cytoplasm")],
                 [_rx("EntC", ["chorismate"], ["isochorismate"])],
                 [_tr("enterobactin secretion", "Ent", "E. coli cytoplasm", "E. coli cytoplasm")])
    FL.assign_locations(p)
    lines = _inferred_locations(p[FL.FIELD])
    assert any(l.startswith("reaction EntC: state -> 'E. coli cytoplasm' [inferred: organism_default:") for l in lines)
    assert any(l.startswith("state E. coli cytoplasm: location 'cytoplasm' [evidence is only the Stage-0 hint") for l in lines)
    assert any(l.startswith("transport enterobactin secretion: both ends are one place") for l in lines)
