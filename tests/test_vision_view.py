"""Scoped views: the no-regression invariant, paper-only closure, extension hops, re-scoping."""

import copy
import json

from t2pw.vision.payload import build_big_payload, merge_extension, norm
from t2pw.vision.view import (DEFAULT, FULL, PAPER_ONLY, ScopeSpec, less_detailed, more_detailed,
                              unresolved_refs, view, view_stats)

from test_vision_payload import payload


def new_dangling(base, v):
    """References unresolved in the view whose name resolved in the base."""
    before = {(kind, norm(n)) for _, kind, n in unresolved_refs(base)}
    return [x for x in unresolved_refs(v) if (x[1], norm(x[2])) not in before]


def test_empty_extension_default_and_full_equal_base():
    p = payload()
    big = build_big_payload(p)
    for scope in (DEFAULT, FULL, ScopeSpec(False, 0), ScopeSpec(False, 5)):
        v = view(big, scope)
        assert v == p
        assert json.dumps(v) == json.dumps(p)
        assert v is not big["base"]


def test_view_never_mutates_big():
    big = build_big_payload(payload())
    before = copy.deepcopy(big)
    view(big, PAPER_ONLY)["entities"]["compounds"].clear()
    view(big, DEFAULT)["entities"]["compounds"].clear()
    assert big == before


def test_paper_only_keeps_paper_rows_and_is_closed():
    p = payload()
    big = build_big_payload(p)
    v = view(big, PAPER_ONLY)
    assert [r["name"] for r in v["processes"]["reactions"]] == ["r1", "r1 again"]  # rag and other dropped
    # A, B referenced by kept reactions; D has its own paper record; C and E dropped
    assert [r["name"] for r in v["entities"]["compounds"]] == ["A", "B", "D"]
    assert [r["name"] for r in v["entities"]["proteins"]] == ["EnzA"]  # enzyme of a kept reaction
    assert v["entities"]["species"] == p["entities"]["species"]
    assert v["biological_states"] == p["biological_states"]
    assert v["element_locations"] == p["element_locations"]
    assert v["reaction_order"] == p["reaction_order"]
    assert list(v) == list(p)
    assert new_dangling(p, v) == []


def test_paper_only_prunes_locations_and_states_that_lose_their_element():
    p = payload()
    p["biological_states"].append({"name": "periplasm", "species": "Escherichia coli"})
    p["biological_states"].append({"name": "unused state"})
    p["element_locations"]["compound_locations"].append({"compound": "E", "biological_state": "periplasm"})
    v = view(build_big_payload(p), PAPER_ONLY)
    assert [r["compound"] for r in v["element_locations"]["compound_locations"]] == ["A"]
    assert [r["name"] for r in v["biological_states"]] == ["E. coli cytoplasm", "unused state"]
    assert new_dangling(p, v) == []


def test_paper_only_keeps_complex_components():
    p = payload()
    p["entities"]["proteins"].append({"name": "SubB"})
    p["entities"]["protein_complexes"].append({"name": "Cplx", "components": [{"name": "SubB"}]})
    p["processes"]["reactions"][0]["enzymes"].append({"entity": "Cplx"})
    v = view(build_big_payload(p), PAPER_ONLY)
    assert [r["name"] for r in v["entities"]["proteins"]] == ["EnzA", "SubB"]
    assert new_dangling(p, v) == []


def extended():
    big = build_big_payload(payload())
    rows = {"entities.compounds": [{"name": "F", "chebi_id": "10"}, {"name": "G", "chebi_id": "11"},
                                   {"name": "lonely"}],
            "processes.reactions": [{"name": "near", "inputs": ["A"], "outputs": ["F"]},
                                    {"name": "far", "inputs": ["F"], "outputs": ["G"],
                                     "biological_state": "periplasm state"}],
            "biological_states": [{"name": "periplasm state"}]}
    return merge_extension(big, rows)


def test_default_and_full_with_extension():
    big = extended()
    base = big["base"]
    d = view(big, DEFAULT)
    assert [r["name"] for r in d["processes"]["reactions"]][-1] == "near"
    assert [r["name"] for r in d["entities"]["compounds"]][-1] == "F"
    assert "periplasm state" not in [r["name"] for r in d["biological_states"]]
    assert d["entities"]["species"] == base["entities"]["species"]
    two = view(big, more_detailed(DEFAULT))
    assert [r["name"] for r in two["processes"]["reactions"]][-2:] == ["near", "far"]
    assert [r["name"] for r in two["biological_states"]][-1] == "periplasm state"
    f = view(big, FULL)
    assert [r["name"] for r in f["entities"]["compounds"]][-3:] == ["F", "G", "lonely"]
    for v in (d, two, f):
        assert new_dangling(base, v) == []
    assert view(big, PAPER_ONLY) == view(build_big_payload(payload()), PAPER_ONLY)
    stats = view_stats(big, DEFAULT)
    assert stats["processes.reactions"] == {"total": 6, "kept": 5, "dropped": 1}
    assert stats["entities.compounds"] == {"total": 8, "kept": 6, "dropped": 2}


def test_more_and_less_detailed():
    assert more_detailed(PAPER_ONLY) == DEFAULT
    assert more_detailed(DEFAULT) == ScopeSpec(False, 2)
    assert more_detailed(FULL) == FULL
    assert less_detailed(ScopeSpec(False, 2)) == DEFAULT
    assert less_detailed(DEFAULT) == PAPER_ONLY
    assert less_detailed(PAPER_ONLY) == PAPER_ONLY
    assert less_detailed(FULL) == DEFAULT


# ---- review round 1 (each fails on 626fe8e9) ----

def test_f6_paper_only_prunes_reaction_order():
    p = payload()
    p["reaction_order"] = ["r1", "rag", "other", "not a reaction"]
    v = view(build_big_payload(p), PAPER_ONLY)
    assert v["reaction_order"] == ["r1", "not a reaction"]
    assert view(build_big_payload(p), DEFAULT)["reaction_order"] == p["reaction_order"]


def test_f7_paper_only_keeps_state_with_own_paper_record():
    p = payload()
    p["biological_states"].append({"name": "stated state", "evidence": "the paper says so"})
    p["processes"]["reactions"][2]["biological_state"] = "stated state"  # used only by the dropped rag row
    p["biological_states"].append({"name": "lost state"})
    p["processes"]["reactions"][3]["biological_state"] = "lost state"
    v = view(build_big_payload(p), PAPER_ONLY)
    assert [r["name"] for r in v["biological_states"]] == ["E. coli cytoplasm", "stated state"]
    assert new_dangling(p, v) == []
