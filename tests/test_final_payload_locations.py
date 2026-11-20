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


STREP = (("Streptomyces tubercidicus", "Prokaryote", "1960"),)


def test_gram_positive_auto_auto_transport_gets_a_direction_from_its_name_and_a_stated_side_never_moves():
    sp = "Streptomyces tubercidicus"
    p = _payload([_st("__auto_state__", "cell", sp), _st("cytoplasmic state", "cytoplasm", sp)], [],
                 [_tr("TubI-mediated TBN export", "TBN", "__auto_state__", "__auto_state__"),
                  _tr("ferric-enterobactin import", "Fe-Ent", "__auto_state__", "cytoplasmic state"),
                  _tr("enterobactin secretion", "Ent", "cytoplasmic state", "cytoplasmic state")], STREP)
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


@pytest.mark.parametrize("states", [
    [_st("__auto_state__", "cell")],                                              # E. coli: Gram-negative
    [_st("__auto_state__", "cell", "Bacillus subtilis"), _st("periplasm state", "periplasm", "Bacillus subtilis")],
    [_st("__auto_state__", "cell", "Corynebacterium glutamicum")],                # mycomembrane: not on the list (N4)
    [_st("__auto_state__", "cell", "Mycobacterium tuberculosis")],
])
def test_no_name_default_for_gram_negative_or_a_paper_with_periplasm(states):
    sp = states[0]["species"]
    p = _payload(copy.deepcopy(states), [], [_tr("LPS export", "LPS", "__auto_state__", "__auto_state__")],
                 ((sp, "Prokaryote", "1"),))
    rep = FL.assign_locations(p)
    t = p["processes"]["transports"][0]
    assert (t["from_biological_state"], t["to_biological_state"]) == ("__auto_state__", "__auto_state__")
    assert "inferred_sides" not in t and any("transport_ends_same_place" in r for r in t["review_reasons"])
    assert rep["transport_default_skipped"]


def test_two_differently_named_unplaced_ends_are_flagged_as_possibly_one_place():
    sp = "Homo sapiens"
    p = _payload([_st("__auto_state__", "cell", sp), _st("__auto_state__ (other side)", "cell", sp)], [],
                 [_tr("OTOP1-mediated H+ permeation", "H+", "__auto_state__", "__auto_state__ (other side)")], HUMAN)
    rep = FL.assign_locations(p)
    assert [x["transport"] for x in rep["same_place_left"]] == ["OTOP1-mediated H+ permeation"]


@pytest.mark.parametrize("species,states", [
    (ECOLI, [_st("E. coli state", "cell"), _st("Arabidopsis leaf", "cell", "Arabidopsis thaliana")]),  # stray state
    (ECOLI + (("Homo sapiens", "Eukaryote", "9606"),), [_st("E. coli state", "cell")]),               # host species
    (ECOLI + (("Bacillus subtilis", "Prokaryote", "1423"),), [_st("E. coli state", "cell")]),           # two bacteria
])
def test_a_mixed_organism_paper_gets_the_main_organisms_default_naming_it(species, states):
    # PO ruling 1: "just use main organism" (the most frequent species among the states here)
    p = _payload(copy.deepcopy(states), [_rx("EntC", ["chorismate"], ["isochorismate"])],
                 [_tr("enterobactin secretion", "Ent", "E. coli state")], species,
                 elements=[{"compound": "chorismate"}])
    rep = FL.assign_locations(p)
    r, el, t = p["processes"]["reactions"][0], p["element_locations"]["compound_locations"][0], p["processes"]["transports"][0]
    other = next(n for n, *_ in species[1:]) if len(species) > 1 else "Arabidopsis thaliana"
    for row in (r, el):
        assert row["biological_state"] == "E. coli state" and row["inferred"] == "organism_default"
        e = row["compartment_estimate"][-1]
        assert e["organism"] == "Escherichia coli"
        assert "main organism assumed: Escherichia coli" in e["reason"] and other in e["reason"]
    assert "to_biological_state" not in t  # E. coli is Gram-negative: no destination from the name
    assert rep["main_organism"].startswith("main organism assumed: Escherichia coli")


def test_the_organism_hint_picks_the_main_organism():
    states = [_st("cytosolic state", "cytosol", "Homo sapiens"), _st("nuclear state", "nucleus", "Homo sapiens")]
    sp = HUMAN + (("Escherichia coli", "Prokaryote", "562"),)
    p = _payload(copy.deepcopy(states), [_rx("EntC", ["chorismate"], ["isochorismate"])], species=sp)
    FL.assign_locations(p, organism_hint="Escherichia coli")
    r = p["processes"]["reactions"][0]
    assert r["inferred"] == "organism_default" and r["compartment_estimate"][-1]["organism"] == "Escherichia coli"


def test_a_eukaryote_main_organism_gets_no_default_even_beside_a_bacterium():
    states = [_st("cytosolic state", "cytosol", "Homo sapiens"), _st("nuclear state", "nucleus", "Homo sapiens"),
              _st("E. coli state", "cell")]
    p = _payload(copy.deepcopy(states), [_rx("X", ["a"], ["b"])], species=HUMAN + ECOLI,
                 elements=[{"compound": "a"}])
    rep = FL.assign_locations(p)
    assert "biological_state" not in p["processes"]["reactions"][0]
    assert "biological_state" not in p["element_locations"]["compound_locations"][0]
    assert not rep["counts"].get("organism_default") and "main_organism" not in rep


def test_a_eukaryote_export_keeps_its_neighbour_fill():
    # citrate leaves the mitochondrion; its only other reaction is in the cytosol: L7 fills the destination, and the
    # location step leaves that flagged estimate alone
    sp = "Homo sapiens"
    p = _payload([_st("mitochondrial matrix state", "mitochondrial matrix", sp), _st("cytosolic state", "cytosol", sp)],
                 [_rx("citrate synthase", ["oxaloacetate", "acetyl-CoA"], ["citrate"], "mitochondrial matrix state"),
                  _rx("ATP citrate lyase", ["citrate"], ["oxaloacetate", "acetyl-CoA"], "cytosolic state")],
                 [_tr("citrate export", "citrate", "mitochondrial matrix state")], HUMAN)
    estimate_compartments(p)
    t = p["processes"]["transports"][0]
    assert t["to_biological_state"] == "cytosolic state" and t["compartment_source"] == "neighbour_estimate"
    before = copy.deepcopy(t)
    FL.assign_locations(p)
    assert p["processes"]["transports"][0] == before

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


def test_a_chat_call_that_raises_leaves_the_rules_only_result(monkeypatch):
    def build():
        return _payload([_st("E. coli periplasmic state", "periplasm"), _st("cytoplasmic state", "cytoplasm")],
                        [_rx("EntF step", ["DHB-AMP"], ["enterobactin"], enzymes=["EntF"])],
                        [_tr("enterobactin secretion", "enterobactin", "cytoplasmic state")])

    off = build()
    FL.assign_locations(off, {"final_payload_paper_text": PAPER})
    monkeypatch.setenv(FL.LLM_ENV, "1")

    def boom(messages, **kw):
        raise RuntimeError("provider down at C:/secret/path/.x")

    on = build()
    rep = FL.assign_locations(on, {"final_payload_paper_text": PAPER}, chat_fn=boom)
    assert rep["error"] is None and rep["llm"]["calls"] == 1 and rep["llm"]["error"].startswith("RuntimeError")
    assert "secret" not in rep["llm"]["error"]  # paths redacted
    strip = lambda p: {k: v for k, v in p.items() if k != FL.FIELD}  # noqa: E731
    assert strip(on) == strip(off)  # only the rules-only placements; nothing from the failed call


def test_the_location_pass_has_its_own_prompt_and_l7_keeps_its_request(monkeypatch):
    from t2pw.service import final_payload_compartments as C

    assert FL.PROMPT_FILE != C.PROMPT_FILE and FL.PROMPT_ID != C.PROMPT_ID
    seen = []
    monkeypatch.setattr(C, "_system", lambda prompt_file=C.PROMPT_FILE: seen.append(prompt_file) or prompt_file)
    ctx = C._Ctx(_payload([_st("cytosolic state", "cytosol", "Homo sapiens")], species=HUMAN))
    system, user = C.build_request([{"id": 0}], ctx, "text")
    assert system == "compartment_estimate_system.txt" and "new_location_vocabulary" not in user
    assert list(json.loads(user)) == ["organism", "candidate_states", "items", "paper_text"]  # L7's request bytes


# ---- PO ruling 2: contradictory stated locations ------------------------------------------------------------------------
def _conflict_payload():
    return _payload([_st("E. coli cytoplasmic state", "cytoplasm"), _st("E. coli extracellular state", "extracellular")],
                    [_rx("EntF step", ["DHB-AMP"], ["enterobactin"], "E. coli cytoplasmic state", enzymes=["EntF"])],
                    [_tr("enterobactin secretion", "enterobactin", "E. coli cytoplasmic state", "E. coli cytoplasmic state",
                         transporters=["EntS"])])


def _conflict_reply(quote):
    return {"items": [{"id": 0, "from": None,
                       "to": {"state": "E. coli extracellular state", "new_location": None, "quote": quote}}]}


def test_rules_only_never_touches_a_stated_conflict():
    p = _conflict_payload()
    rep = FL.assign_locations(p, {"final_payload_paper_text": PAPER})
    t = p["processes"]["transports"][0]
    assert t["to_biological_state"] == "E. coli cytoplasmic state" and "inferred_sides" not in t
    assert rep["llm_conflict"]["skipped"] == "off" and rep["same_place_left"]


def test_llm_corrects_a_stated_same_place_transport_with_a_verified_quote(monkeypatch):
    from t2pw.app.reaction_evidence import _inferred_locations

    monkeypatch.setenv(FL.LLM_ENV, "1")
    calls = []
    p = _conflict_payload()
    rep = FL.assign_locations(p, {"final_payload_paper_text": PAPER},
                              chat_fn=_fake(_conflict_reply("secreted into the extracellular medium by EntS"), calls))
    assert len(calls) == 1 and json.loads(calls[0][0][1]["content"])["items"][0]["kind"] == "transport_conflict"
    assert calls[0][0][0]["content"].startswith("You resolve contradictory locations")  # its own prompt
    t = p["processes"]["transports"][0]
    assert t["from_biological_state"] == "E. coli cytoplasmic state"  # the right end is untouched
    assert t["to_biological_state"] == "E. coli extracellular state"
    assert t["inferred_sides"] == {"to_biological_state": "llm_corrected"} and t["compartment_source"] == "llm_estimate"
    e = t["compartment_estimate"][-1]
    assert e["stated_was"] == "E. coli cytoplasmic state" and e["stated_location_was"] == "cytoplasm"
    assert e["quote"] == "secreted into the extracellular medium by EntS"
    assert rep["counts"]["llm_corrected"] == 1 and not rep["same_place_left"]
    assert any("[inferred: llm_corrected:" in l and "was stated as 'E. coli cytoplasmic state'" in l
               for l in _inferred_locations(p[FL.FIELD]))


@pytest.mark.parametrize("quote", ["secreted into the periplasm by EntS",          # not in the paper
                                   "Enterobactin is assembled by EntF in the cytoplasm"])  # names no new place
def test_a_correction_whose_quote_fails_the_check_changes_nothing(monkeypatch, quote):
    monkeypatch.setenv(FL.LLM_ENV, "1")
    p = _conflict_payload()
    before = copy.deepcopy(p["processes"]["transports"][0])
    rep = FL.assign_locations(p, {"final_payload_paper_text": PAPER}, chat_fn=_fake(_conflict_reply(quote), []))
    t = p["processes"]["transports"][0]
    assert (t["from_biological_state"], t["to_biological_state"]) == (before["from_biological_state"], before["to_biological_state"])
    assert "inferred_sides" not in t and rep["llm_conflict"]["rejected"] and rep["same_place_left"]


def _side(quote="secreted into the extracellular medium by EntS"):
    return {"state": "E. coli extracellular state", "new_location": None, "quote": quote}


@pytest.mark.parametrize("reply,why", [
    ({"items": [{"id": 0, "from": _side(), "to": _side()}]}, "exactly one"),          # N1: both stated ends returned
    ({"items": [{"id": 0, "from": _side(), "to": None}]}, "only the to side"),        # N2: a reversed secretion
])
def test_a_conflict_answer_that_would_overwrite_both_ends_or_reverse_the_direction_is_rejected(monkeypatch, reply, why):
    monkeypatch.setenv(FL.LLM_ENV, "1")
    p = _conflict_payload()
    rep = FL.assign_locations(p, {"final_payload_paper_text": PAPER}, chat_fn=_fake(reply, []))
    t = p["processes"]["transports"][0]
    assert (t["from_biological_state"], t["to_biological_state"]) == ("E. coli cytoplasmic state",) * 2
    assert "inferred_sides" not in t and any(why in r["reason"] for r in rep["llm_conflict"]["rejected"])
    assert rep["same_place_left"]


def test_an_import_conflict_may_correct_only_its_from_side(monkeypatch):
    monkeypatch.setenv(FL.LLM_ENV, "1")
    p = _conflict_payload()
    p["processes"]["transports"][0]["name"] = "enterobactin uptake"
    FL.assign_locations(p, {"final_payload_paper_text": PAPER},
                        chat_fn=_fake({"items": [{"id": 0, "from": _side(), "to": None}]}, []))
    t = p["processes"]["transports"][0]
    assert t["from_biological_state"] == "E. coli extracellular state" and t["inferred_sides"] == {"from_biological_state": "llm_corrected"}


def test_no_paper_text_means_no_conflict_call(monkeypatch):
    # N3: a stated location is corrected only against the paper, never against the extraction's own evidence strings
    monkeypatch.setenv(FL.LLM_ENV, "1")
    calls = []
    p = _conflict_payload()
    p["processes"]["transports"][0]["evidence"] = "secreted into the extracellular medium by EntS"
    rep = FL.assign_locations(p, {}, chat_fn=_fake(_conflict_reply("secreted into the extracellular medium by EntS"), calls))
    assert calls == [] and rep["llm_conflict"]["skipped"] == "no_paper_text"
    assert p["processes"]["transports"][0]["to_biological_state"] == "E. coli cytoplasmic state"


def test_a_conflict_call_that_raises_changes_nothing(monkeypatch):
    monkeypatch.setenv(FL.LLM_ENV, "1")

    def boom(messages, **kw):
        raise RuntimeError("provider down")

    off, on = _conflict_payload(), _conflict_payload()
    FL.assign_locations(off, {"final_payload_paper_text": PAPER})
    rep = FL.assign_locations(on, {"final_payload_paper_text": PAPER}, chat_fn=boom)
    assert rep["error"] is None and rep["llm_conflict"]["error"].startswith("RuntimeError")
    strip = lambda p: {k: v for k, v in p.items() if k != FL.FIELD}  # noqa: E731
    assert strip(on) == strip(off) and rep["same_place_left"]


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
