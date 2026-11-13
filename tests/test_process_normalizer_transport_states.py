"""ensure_autostates: a transport side with no row-level state takes the state the row itself
declares in elements_with_states; otherwise the placeholder (unchanged behaviour)."""
import copy

from t2pw.pipeline.process_normalizer import ensure_autostates

AUTO = "__auto_state__"


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


def test_nothing_declared_keeps_placeholder():
    t = _run({"name": "t", "cargo": "H+"})
    assert (t["from_biological_state"], t["to_biological_state"]) == (AUTO, AUTO)


def test_declared_state_not_in_payload_keeps_placeholder():
    t = _run({"name": "t", "cargo": "H+", "elements_with_states": [_ews("right", "Nowhere state")]})
    assert t["to_biological_state"] == AUTO


def test_conflicting_declarations_keep_placeholder():
    t = _run({"name": "t", "cargo": "H+", "elements_with_states": [_ews("right", "Cytosol state"), _ews("right", "Outside state")]})
    assert t["to_biological_state"] == AUTO


def test_row_level_state_wins_over_declaration():
    t = _run({"name": "t", "cargo": "H+", "from_biological_state": "Outside state", "to_biological_state": "Outside state",
              "elements_with_states": [_ews("right", "Cytosol state")]})
    assert (t["from_biological_state"], t["to_biological_state"]) == ("Outside state", "Outside state")


def test_idempotent():
    p = _payload({"name": "t", "cargo": "H+", "elements_with_states": [_ews("right", "Cytosol state")]})
    once = ensure_autostates(p)
    assert ensure_autostates(copy.deepcopy(once)) == once
