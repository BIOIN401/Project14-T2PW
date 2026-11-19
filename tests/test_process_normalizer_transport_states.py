"""ensure_autostates: a transport side with no row-level state takes the state the row itself
declares in elements_with_states; otherwise the placeholder (unchanged behaviour)."""
import copy

from t2pw.pipeline.process_normalizer import ensure_autostates

AUTO = "__auto_state__"
AUTO_OTHER = "__auto_state__ (other side)"


def _payload(transport):
    return {
        "entities": {"compounds": [{"name": "H+"}]},
        "biological_states": [{"name": "Cytosol state"}, {"name": "Outside state"}],
        "processes": {"reactions": [], "transports": [transport]},
    }


def _run(transport):
    out = ensure_autostates(_payload(transport))
    return out["processes"]["transports"][0]


def _ews(side, state):
    return {"element": "H+", "side": side, "biological_state": state}


def test_declared_side_state_used_instead_of_placeholder():
    t = _run({"name": "t", "cargo": "H+", "elements_with_states": [_ews("left", ""), _ews("right", "cytosol state")]})
    assert t["from_biological_state"] == AUTO
    assert t["to_biological_state"] == "Cytosol state"  # canonical name of the existing state


def test_both_sides_declared():
    t = _run({"name": "t", "cargo": "H+", "elements_with_states": [_ews("left", "Outside state"), _ews("right", "Cytosol state")]})
    assert (t["from_biological_state"], t["to_biological_state"]) == ("Outside state", "Cytosol state")


def test_nothing_declared_keeps_placeholders_on_two_different_states():
    # lane loc/assign (D-W21 c): one placeholder on both ends was a transport from a place into itself (__auto__ -> __auto__)
    out = ensure_autostates(_payload({"name": "t", "cargo": "H+"}))
    t = out["processes"]["transports"][0]
    assert (t["from_biological_state"], t["to_biological_state"]) == (AUTO, AUTO_OTHER)
    other = [s for s in out["biological_states"] if s["name"] == AUTO_OTHER]
    auto = [s for s in out["biological_states"] if s["name"] == AUTO]
    assert len(other) == 1 and other[0]["subcellular_location"] == auto[0]["subcellular_location"] == "cell"


def test_declared_state_not_in_payload_keeps_placeholder():
    t = _run({"name": "t", "cargo": "H+", "elements_with_states": [_ews("right", "Nowhere state")]})
    assert t["to_biological_state"] == AUTO_OTHER  # from is the placeholder too


def test_conflicting_declarations_keep_placeholder():
    t = _run({"name": "t", "cargo": "H+", "elements_with_states": [_ews("right", "Cytosol state"), _ews("right", "Outside state")]})
    assert t["to_biological_state"] == AUTO_OTHER


def test_a_stated_from_side_keeps_the_one_placeholder():
    t = _run({"name": "t", "cargo": "H+", "from_biological_state": "Outside state"})
    assert (t["from_biological_state"], t["to_biological_state"]) == ("Outside state", AUTO)


def test_row_level_state_wins_over_declaration():
    t = _run({"name": "t", "cargo": "H+", "from_biological_state": "Outside state", "to_biological_state": "Outside state",
              "elements_with_states": [_ews("right", "Cytosol state")]})
    assert (t["from_biological_state"], t["to_biological_state"]) == ("Outside state", "Outside state")


def test_idempotent():
    p = _payload({"name": "t", "cargo": "H+", "elements_with_states": [_ews("right", "Cytosol state")]})
    once = ensure_autostates(p)
    assert ensure_autostates(copy.deepcopy(once)) == once


def test_declared_state_equal_to_the_other_sides_state_keeps_placeholder():
    # right is explicit "Cytosol state"; left declares the same state: using it would create a new left == right
    t = _run({"name": "t", "cargo": "H+", "to_biological_state": "cytosol STATE",
              "elements_with_states": [_ews("left", "Cytosol state")]})
    assert (t["from_biological_state"], t["to_biological_state"]) == (AUTO, "cytosol STATE")
    # both sides declare the same state and neither is explicit: the second cannot also take it
    t = _run({"name": "t", "cargo": "H+", "elements_with_states": [_ews("left", "Cytosol state"), _ews("right", "Cytosol state")]})
    assert (t["from_biological_state"], t["to_biological_state"]) == (AUTO, "Cytosol state")


def test_declarations_that_differ_only_in_case_agree():
    t = _run({"name": "t", "cargo": "H+", "elements_with_states": [_ews("right", "Cytosol state"), _ews("right", "cytosol  STATE")]})
    assert (t["from_biological_state"], t["to_biological_state"]) == (AUTO, "Cytosol state")
