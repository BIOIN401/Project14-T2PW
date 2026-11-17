"""L7 (D-W24 2 and 7a): a reaction or transport side with no stated compartment is estimated, not put in the first state.

Order: neighbours sharing a non-currency participant (deterministic), then one LLM call (faked here; cached; skipped under
the offline switch), then the organism's general cell state flagged ``compartment_not_stated``. Every placement is marked
(``compartment_source`` + ``compartment_estimate`` reason) and shows in the text record and our SBML. Transports: per
missing side; never both sides the same state, never an invented direction. Flag off leaves the payload untouched."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2pw.clean_model.fill.cache import Cache  # noqa: E402
from t2pw.service.final_payload_compartments import FIELD, estimate_compartments  # noqa: E402

PAPER = ("Carnitine biosynthesis starts in the mitochondria. TML is hydroxylated by TMLD in the mitochondrial matrix. "
         "The aldehyde is then oxidised in the cytosol by TMABADH. Some steps take place in peroxisomes of liver cells.")


def _rx(name, ins, outs, state=None, **kw):
    row = {"name": name, "inputs": list(ins), "outputs": list(outs), "enzymes": [], "evidence": f"paper says {name}"}
    if state is not None:
        row["biological_state"] = state
    row.update(kw)
    return row


def _st(name, loc, species="Homo sapiens", **kw):
    return {"name": name, "species": species, "subcellular_location": loc, **kw}


def _payload(states, reactions, transports=(), species=(("Homo sapiens", "Eukaryote", "9606"),)):
    return {"biological_states": states,
            "entities": {"species": [{"name": n, "classification": c, "taxonomy_id": t} for n, c, t in species],
                         "subcellular_locations": [{"name": s["subcellular_location"]} for s in states]},
            "processes": {"reactions": list(reactions), "transports": list(transports), "interactions": []},
            "element_locations": {"compound_locations": []}}


def _by_name(p):
    return {r["name"]: r for r in p["processes"]["reactions"]}


@pytest.fixture(autouse=True)
def _offline_cache(monkeypatch):
    monkeypatch.setattr("t2pw.clean_model.fill.cache.Cache._path", lambda self, key: None)  # memory only
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD_BUDGET_S", raising=False)
    monkeypatch.setenv("T2PW_OFFLINE_CURATOR", "1")  # the real client is never reached; a test passes its own chat_fn


class _Chat:
    def __init__(self, reply):
        self.reply, self.calls, self.user = reply, 0, None

    def __call__(self, messages, **kw):
        self.calls += 1
        self.user = json.loads(messages[1]["content"])
        assert kw["temperature"] == 0.0 and kw["response_json"] is True
        return json.dumps(self.reply)


# ---- step 1: neighbours ----------------------------------------------------------------------------------------------
def test_l7_a_chain_with_no_state_takes_its_neighbours_state_not_the_first_state() -> None:
    p = _payload([_st("cytosolic state", "cytosol"), _st("ER state", "endoplasmic reticulum")],
                 [_rx("A", ["lanosterol"], ["FF-MAS"], "ER state"), _rx("B", ["FF-MAS", "NADPH"], ["T-MAS", "NADP+"]),
                  _rx("C", ["T-MAS"], ["zymosterol"]), _rx("D", ["NADPH"], ["NADP+"], "cytosolic state")])
    rep = estimate_compartments(p, {})
    rx = _by_name(p)
    assert rx["B"]["biological_state"] == "ER state" and rx["C"]["biological_state"] == "ER state"
    assert rx["B"]["compartment_source"] == "neighbour_estimate"
    reason = rx["B"]["compartment_estimate"][-1]["reason"]
    assert "'A'" in reason and "FF-MAS" in reason and "NADPH" not in reason  # currency never links
    assert "through 1 other" in rx["C"]["compartment_estimate"][-1]["reason"]
    assert "review_required" not in rx["B"] and rep["counts"]["neighbour_estimate"] == 2 and FIELD in p


def test_l7_neighbours_in_two_states_are_not_a_neighbour_estimate() -> None:
    p = _payload([_st("mitochondrial state", "mitochondria"), _st("cytosolic state", "cytosol")],
                 [_rx("A", ["TML"], ["HTML"], "mitochondrial state"), _rx("B", ["HTML"], ["TMABA"]),
                  _rx("C", ["TMABA"], ["GBB"], "cytosolic state")])
    estimate_compartments(p, {})  # no paper text, no evidence-only call: offline curator in the test env
    b = _by_name(p)["B"]
    assert b["compartment_source"] == "unresolved" and b["review_required"] is True
    assert b["review_reasons"][-1].startswith("compartment_not_stated:")
    assert b["biological_state"] == "Homo sapiens cell"  # the general cell state, not row 0 "mitochondrial state"
    assert "2 different states" in b["compartment_estimate"][-1]["reason"]
    assert {"name": "cell"} in p["entities"]["subcellular_locations"]


def test_l7_a_transported_compound_links_no_reactions() -> None:
    """idx 84: enterobactin is made inside and bound to iron outside; the biosynthesis is not extracellular."""
    p = _payload([_st("E. coli cytosolic state", "cytosol", "Escherichia coli"), _st("E. coli extracellular", "extracellular", "Escherichia coli")],
                 [_rx("EntF", ["DHB"], ["enterobactin"]), _rx("bind", ["enterobactin", "Fe3+"], ["ferri-enterobactin"], "E. coli extracellular")],
                 [{"name": "secretion", "cargo": "enterobactin", "from_biological_state": "E. coli cytosolic state",
                   "to_biological_state": "E. coli extracellular"}],
                 species=(("Escherichia coli", "Prokaryote", "562"),))
    estimate_compartments(p, {})
    ent = _by_name(p)["EntF"]
    assert ent["biological_state"] == "E. coli cytosolic state" and ent["compartment_source"] == "unresolved"


def test_l7_a_state_that_names_no_compartment_does_not_decide_a_neighbour_estimate() -> None:
    p = _payload([_st("mitochondrial state", "mitochondria"), _st("human carnitine biosynthesis", "cell")],
                 [_rx("A", ["TML"], ["HTML"], "human carnitine biosynthesis"), _rx("B", ["TML"], ["HTML"])])
    estimate_compartments(p, {})
    b = _by_name(p)["B"]
    assert b["compartment_source"] == "unresolved" and b["biological_state"] == "Homo sapiens cell"


# ---- step 2: the LLM (faked) -----------------------------------------------------------------------------------------
def _ambiguous():
    return _payload([_st("cytosolic state", "cytosol"), _st("mitochondrial state", "mitochondria")],
                    [_rx("TMLD reaction", ["TML"], ["HTML"])])


def test_l7_the_model_places_a_reaction_in_a_candidate_state_with_a_verbatim_quote_and_the_reply_is_cached() -> None:
    chat = _Chat({"items": [{"id": 0, "state": "mitochondrial state", "quote": "hydroxylated by TMLD in the mitochondrial matrix"}]})
    cache = Cache(None)
    p = _ambiguous()
    rep = estimate_compartments(p, {"final_payload_paper_text": PAPER}, cache=cache, chat_fn=chat)
    r = _by_name(p)["TMLD reaction"]
    assert r["biological_state"] == "mitochondrial state" and r["compartment_source"] == "llm_estimate"
    assert "mitochondrial matrix" in r["compartment_estimate"][-1]["quote"] and "review_required" not in r
    assert chat.calls == 1 and rep["llm"]["calls"] == 1
    assert {c["name"] for c in chat.user["candidate_states"]} == {"cytosolic state", "mitochondrial state"}
    assert chat.user["organism"] == "Homo sapiens" and chat.user["paper_text"] == PAPER
    q = _ambiguous()  # same inputs: the cached reply, no second call
    rep2 = estimate_compartments(q, {"final_payload_paper_text": PAPER}, cache=cache, chat_fn=chat)
    assert chat.calls == 1 and rep2["llm"]["cache_hits"] == 1 and q["processes"] == p["processes"]


@pytest.mark.parametrize("answer,why", [
    ({"id": 0, "state": "mitochondrial state", "quote": "TMLD works in the mitochondria"}, "quote not found"),
    ({"id": 0, "state": "nucleus state", "quote": "hydroxylated by TMLD in the mitochondrial matrix"}, "not a candidate"),
    ({"id": 0, "state": None, "new_location": "lysosome", "quote": "hydroxylated by TMLD in the mitochondrial matrix"}, "not in the quote"),
])
def test_l7_an_unsupported_model_answer_is_rejected_and_the_row_flagged(answer, why) -> None:
    p = _ambiguous()
    rep = estimate_compartments(p, {"final_payload_paper_text": PAPER}, chat_fn=_Chat({"items": [answer]}))
    r = _by_name(p)["TMLD reaction"]
    assert r["compartment_source"] == "unresolved" and r["review_required"] is True
    assert why in rep["llm"]["rejected"][0]["reason"]


def test_l7_the_model_may_name_a_new_pathwhiz_location_found_in_its_quote() -> None:
    p = _ambiguous()
    estimate_compartments(p, {"final_payload_paper_text": PAPER},
                          chat_fn=_Chat({"items": [{"id": 0, "state": None, "new_location": "peroxisome",
                                                    "quote": "Some steps take place in peroxisomes of liver cells"}]}))
    r = _by_name(p)["TMLD reaction"]
    assert r["biological_state"] == "Homo sapiens peroxisome" and r["compartment_source"] == "llm_estimate"
    new = p["biological_states"][-1]
    assert new["species"] == "Homo sapiens" and new["subcellular_location"] == "peroxisome"
    assert {"name": "peroxisome"} in p["entities"]["subcellular_locations"]


def test_l7_the_offline_switch_skips_the_model_and_keeps_the_row_flagged(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD_BUDGET_S", "0")
    chat = _Chat({"items": []})
    p = _ambiguous()
    rep = estimate_compartments(p, {"final_payload_paper_text": PAPER}, chat_fn=chat)
    assert chat.calls == 0 and rep["llm"]["skipped"] == "offline_switch"
    r = _by_name(p)["TMLD reaction"]
    assert r["compartment_source"] == "unresolved" and "offline_switch" in r["compartment_estimate"][-1]["reason"]


def test_l7_a_failing_model_call_leaves_the_rows_flagged() -> None:
    def boom(*a, **k):
        raise RuntimeError("provider down")

    p = _ambiguous()
    rep = estimate_compartments(p, {"final_payload_paper_text": PAPER}, chat_fn=boom)
    assert "provider down" in rep["llm"]["error"] and _by_name(p)["TMLD reaction"]["compartment_source"] == "unresolved"


# ---- step 3: the general cell state ----------------------------------------------------------------------------------
def test_l7_a_whole_pathway_without_states_goes_to_the_organisms_cell_not_the_extracellular_box() -> None:
    """idx 138: every reaction unstated; row 0 was 'extracellular'. The pipeline's own placeholder becomes the named cell
    state, and every reference to it follows."""
    sp = (("Streptomyces tubercidicus", "Prokaryote", "47759"),)
    p = _payload([_st("Streptomyces tubercidicus extracellular", "extracellular", "Streptomyces tubercidicus"),
                  _st("__auto_state__", "cell", "Streptomyces tubercidicus")],
                 [_rx("TubA", ["ATP", "PRPP"], ["X"]), _rx("TubB", ["X"], ["Y"])], species=sp)
    p["element_locations"]["compound_locations"] = [{"compound": "X", "biological_state": "__auto_state__"}]
    rep = estimate_compartments(p, {})
    assert {r["biological_state"] for r in p["processes"]["reactions"]} == {"Streptomyces tubercidicus cell"}
    assert [s["name"] for s in p["biological_states"]] == ["Streptomyces tubercidicus extracellular", "Streptomyces tubercidicus cell"]
    assert p["element_locations"]["compound_locations"][0]["biological_state"] == "Streptomyces tubercidicus cell"
    assert rep["general_state"]["how"] == "renamed_placeholder" and rep["counts"]["unresolved"] == 2
    assert all(r["review_required"] for r in p["processes"]["reactions"])


def test_l7_an_existing_plain_state_of_the_organism_is_reused_but_not_a_named_place() -> None:
    p = _payload([_st("afferent nerve fiber state", "cell", "Mus musculus"), _st("Type III cell state", "cell", "Mus musculus",
                                                                                  cell_type="Type III cell")],
                 [_rx("R", ["a"], ["b"])], species=(("Mus musculus", "Eukaryote", "10090"),))
    rep = estimate_compartments(p, {})
    assert rep["general_state"] == {"name": "Mus musculus cell", "how": "created", "location": "cell"}
    q = _payload([_st("Homo sapiens state", "cell"), _st("ER state", "endoplasmic reticulum")], [_rx("R", ["a"], ["b"])])
    rep = estimate_compartments(q, {})
    assert rep["general_state"] == {"name": "Homo sapiens state", "how": "existing_state"}


def test_l7_the_pathway_organism_wins_over_a_heterologous_host() -> None:
    """idx 135: the default species picked Arabidopsis (expression host); the pathway is Saposhnikovia's."""
    sp = (("Saposhnikovia divaricata", "Eukaryote", "1"), ("Arabidopsis thaliana", "Eukaryote", "3702"))
    p = _payload([], [_rx("SdPCS", ["malonyl-CoA"], ["noreugenin"])], species=sp)
    estimate_compartments(p, {}, organism_hint="Saposhnikovia divaricata")
    assert p["biological_states"] == [{"name": "Saposhnikovia divaricata cell", "species": "Saposhnikovia divaricata",
                                       "subcellular_location": "cell"}]


# ---- transports ------------------------------------------------------------------------------------------------------
def _tr(name, cargo, frm=None, to=None):
    return {"name": name, "cargo": cargo, "from_biological_state": frm, "to_biological_state": to}


def test_l7_a_missing_transport_side_takes_the_cargos_other_state() -> None:
    p = _payload([_st("mitochondrial state", "mitochondria"), _st("cytosolic state", "cytosol")],
                 [_rx("A", ["TML"], ["HTML"], "mitochondrial state")],
                 [_tr("HTML export", "HTML", None, "cytosolic state")])
    estimate_compartments(p, {})
    t = p["processes"]["transports"][0]
    assert t["from_biological_state"] == "mitochondrial state" and t["compartment_source"] == "neighbour_estimate"
    assert t["compartment_estimate"][-1]["field"] == "from_biological_state"


def test_l7_both_sides_missing_are_never_given_a_direction_offline() -> None:
    p = _payload([_st("E. coli cytoplasm", "cytoplasm", "Escherichia coli"), _st("__auto_state__", "cell", "Escherichia coli")],
                 [_rx("A", ["x"], ["TBN"], "E. coli cytoplasm")],
                 [_tr("TubI export", "TBN", "__auto_state__", "__auto_state__")], species=(("Escherichia coli", "Prokaryote", "562"),))
    estimate_compartments(p, {})
    t = p["processes"]["transports"][0]
    assert t["from_biological_state"] == t["to_biological_state"] == "__auto_state__"
    assert t["review_required"] is True and t["compartment_source"] == "unresolved"
    assert "no direction is invented" in t["review_reasons"][-1]


def test_l7_a_side_nobody_estimated_is_not_put_in_the_general_cell_state() -> None:
    """idx 3: 'OTOP1 H+ permeation' had no from side; the general cell state would make it cell -> Type III cell."""
    p = _payload([_st("apical side state", "cell"), _st("Type III cell state", "cell", cell_type="Type III cell")],
                 [_rx("R", ["a"], ["b"])], [_tr("OTOP1-mediated H+ permeation", "H+", None, "Type III cell state")])
    estimate_compartments(p, {})
    t = p["processes"]["transports"][0]
    assert t["from_biological_state"] is None and t["review_required"] is True
    assert "not placed in the general cell state" in t["review_reasons"][-1]
    assert _by_name(p)["R"]["biological_state"] == "Homo sapiens cell"  # a reaction still is


def test_l7_a_side_that_would_equal_the_other_side_is_not_written() -> None:
    p = _payload([_st("E. coli cytoplasm", "cytoplasm", "Escherichia coli")], [],
                 [_tr("enterobactin secretion", "enterobactin", "E. coli cytoplasm", None)], species=(("Escherichia coli", "Prokaryote", "562"),))
    estimate_compartments(p, {})
    t = p["processes"]["transports"][0]
    assert t["to_biological_state"] is None and t["review_required"] is True


def test_l7_the_model_gives_both_sides_of_a_transport_only_when_they_differ() -> None:
    text = "TubI exports TBN out of the cell into the medium. TBN is made in the cytoplasm of the producer."
    sp = (("Streptomyces tubercidicus", "Prokaryote", "47759"),)
    states = [_st("S. t. cytoplasm", "cytoplasm", "Streptomyces tubercidicus"), _st("S. t. extracellular", "extracellular", "Streptomyces tubercidicus")]
    good = {"items": [{"id": 0, "from": {"state": "S. t. cytoplasm", "quote": "TBN is made in the cytoplasm"},
                       "to": {"state": "S. t. extracellular", "quote": "TubI exports TBN out of the cell"}}]}
    p = _payload(copy.deepcopy(states), [], [_tr("TubI export", "TBN")], species=sp)
    estimate_compartments(p, {"final_payload_paper_text": text}, chat_fn=_Chat(good))
    t = p["processes"]["transports"][0]
    assert (t["from_biological_state"], t["to_biological_state"]) == ("S. t. cytoplasm", "S. t. extracellular")
    assert t["compartment_source"] == "llm_estimate"
    same = {"items": [{"id": 0, "from": {"state": "S. t. cytoplasm", "quote": "TBN is made in the cytoplasm"},
                       "to": {"state": "S. t. cytoplasm", "quote": "TBN is made in the cytoplasm"}}]}
    q = _payload(copy.deepcopy(states), [], [_tr("TubI export", "TBN")], species=sp)
    rep = estimate_compartments(q, {"final_payload_paper_text": text}, chat_fn=_Chat(same))
    u = q["processes"]["transports"][0]
    assert u["from_biological_state"] is None and u["to_biological_state"] is None and u["review_required"] is True
    assert rep["llm"]["rejected"][-1]["reason"] == "both sides would be the same state"


# ---- records: text record, SBML, IR ----------------------------------------------------------------------------------
def _placed():
    p = _payload([_st("cytosolic state", "cytosol"), _st("ER state", "endoplasmic reticulum")],
                 [_rx("A", ["lanosterol"], ["FF-MAS"], "ER state"), _rx("B", ["FF-MAS"], ["T-MAS"])],
                 [_tr("T-MAS export", "T-MAS", "ER state", None)])
    estimate_compartments(p, {})
    return p


def test_l7_the_text_record_prints_how_the_compartment_was_placed() -> None:
    from t2pw.app.reaction_evidence import FINAL_MAPPED_FILENAME, build_report, render_text

    txt = render_text(build_report({FINAL_MAPPED_FILENAME: _placed()}))
    assert "Compartment (not stated in the paper record): ER state [neighbour_estimate: no compartment stated;" in txt
    assert "TRANSPORT SIDES NOT STATED IN THE PAPER RECORD (1)" in txt and "from=ER state; to=" in txt and "[unresolved:" in txt
    assert txt.count("Compartment (not stated") == 1  # the stated reaction prints no such line


def test_l7_our_sbml_annotates_the_estimate() -> None:
    from t2pw.clean_model import build_from_payload
    from t2pw.clean_model.export.sbml import sbml_text

    xml = sbml_text(build_from_payload(_placed()))
    assert '<t2pw:compartmentSource source="neighbour_estimate">' in xml
    assert '<t2pw:compartmentSource source="unresolved">' in xml
    assert xml.count("t2pw:compartmentSource source=") == 2


def test_l7_the_ir_puts_the_reaction_in_the_estimated_state_not_row_0() -> None:
    from t2pw.pwml.ir import build_pwml_ir

    p = _placed()
    ir, _ = build_pwml_ir(p, strict_db=False)
    states = {s["key"]: s["name"] for s in ir["biological_states"]}
    assert {r["name"]: states[r["biological_state_key"]] for r in ir["processes"]["reactions"]} == {"A": "ER state", "B": "ER state"}


# ---- the seam, flag off, runs once, fails open -----------------------------------------------------------------------
class _NoNet:
    def get_json(self, url, params=None):
        return None

    get_text = get_json


def test_l7_the_final_payload_seam_runs_the_estimate_after_the_merge(monkeypatch) -> None:
    import t2pw.service.final_payload as fp

    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    p = _payload([_st("cytosolic state", "cytosol"), _st("ER state", "endoplasmic reticulum")],
                 [_rx("A", ["lanosterol"], ["FF-MAS"], "ER state"), _rx("B", ["FF-MAS"], ["T-MAS"])])
    rep = fp.maybe_apply_final_id_fill(p, {"likely_organism": "Homo sapiens"}, {}, fetcher=_NoNet())
    assert rep["compartments"]["counts"]["neighbour_estimate"] == 1
    assert _by_name(p)["B"]["biological_state"] == "ER state" and FIELD in p


def test_l7_flag_off_leaves_the_payload_untouched(monkeypatch) -> None:
    import t2pw.service.final_payload as fp

    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "0")
    p = _ambiguous()
    before = copy.deepcopy(p)
    assert fp.maybe_apply_final_id_fill(p, None, {}, fetcher=_NoNet()) == {}
    assert p == before


def test_l7_runs_once_and_fails_open(monkeypatch) -> None:
    p = _ambiguous()
    estimate_compartments(p, {})
    after = copy.deepcopy(p)
    assert estimate_compartments(p, {})["skipped"] == "already_estimated" and p == after
    import t2pw.service.final_payload_compartments as fpc

    q = _ambiguous()
    before = copy.deepcopy(q)
    monkeypatch.setattr(fpc, "_neighbours", lambda *a, **k: 1 / 0)
    rep = estimate_compartments(q, {})
    assert rep["error"].startswith("ZeroDivisionError") and q == before


def test_l7_stated_reactions_and_payloads_with_nothing_to_do_are_untouched() -> None:
    p = _payload([_st("ER state", "endoplasmic reticulum")], [_rx("A", ["a"], ["b"], "ER state")])
    before = copy.deepcopy(p)
    rep = estimate_compartments(p, {})
    assert p == before and rep["reactions"] == [] and FIELD not in p
