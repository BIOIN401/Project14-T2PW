"""Final payload writer fixes (bound-visualizations empty, spontaneous nil, transport compound locations).

Every fix is gated by T2PW_FINAL_PAYLOAD; with the flag off the writer output is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from test_pwml_writer import build_pwml_ir  # noqa: E402  (Stage-6 + pre-freeze wrapper)
from t2pw.pwml.validate import discover_structure_signature  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402


def _payload(reaction_spontaneous: object = None, reaction_uses_glucose: bool = False) -> dict:
    reactions = []
    if reaction_uses_glucose:
        reactions.append({"name": "Glucose to G6P", "inputs": ["Glucose"], "outputs": ["G6P"],
                          "biological_state": "cytosol"})
    elif reaction_spontaneous is not None:
        reactions.append({"name": "A to B", "inputs": ["A"], "outputs": ["B"],
                          "biological_state": "cytosol", "spontaneous": reaction_spontaneous})
    if reaction_uses_glucose and reaction_spontaneous is not None:
        reactions[0]["spontaneous"] = reaction_spontaneous
    return {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "extracellular", "pathwhiz_id": 2}, {"name": "cytosol", "pathwhiz_id": 3}],
            "compounds": [{"name": "Glucose", "pathbank_compound_id": 101}, {"name": "G6P", "pathbank_compound_id": 102},
                          {"name": "A", "pathbank_compound_id": 103}, {"name": "B", "pathbank_compound_id": 104}],
            "proteins": [{"name": "GLUT1", "species": "Homo sapiens", "uniprot": "P11166", "pathbank_protein_id": 301}],
        },
        "biological_states": [
            {"name": n, "species": "Homo sapiens", "subcellular_location": n, "compartment_canonical": n}
            for n in ("extracellular", "cytosol")
        ],
        "processes": {
            "reactions": reactions,
            "transports": [{"name": "Glucose import", "cargo": "Glucose", "from_biological_state": "extracellular",
                            "to_biological_state": "cytosol", "transporters": [{"protein": "GLUT1"}]}],
            "interactions": [],
        },
    }


def _build(payload: dict, with_bound: bool = False, spontaneous: object = None):
    ir, report = build_pwml_ir(payload, strict_db=True)
    assert report["errors"] == []
    if spontaneous is not None:  # the IR hard-codes False today (ir.py), so set what a future IR would carry
        ir["processes"]["reactions"][0]["spontaneous"] = spontaneous
    if with_bound:
        ir["bound_visualizations"] = [{"key": "bv_1", "biological_state_key": ir["biological_states"][0]["key"],
                                       "x": 0, "y": 0}]
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000",
                           height=1400, width=3200, background_color="#FFFFFF", ref=str(ROOT / "reference" / "PW000001.pwml"))
    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args)
    return builder, builder.build().root


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)


def test_bound_visualizations_written_when_flag_off(flag_off) -> None:
    builder, root = _build(_payload(), with_bound=True)
    assert len(builder.section_items["bound-visualizations"]) == 1
    assert len(root.findall(".//bound-visualizations/bound-visualization")) == 1


def test_bound_visualizations_empty_when_flag_on(flag_on) -> None:
    builder, root = _build(_payload(), with_bound=True)
    assert builder.section_items["bound-visualizations"] == []
    assert root.findall(".//bound-visualizations/bound-visualization") == []


def _spont(root):
    return [r.find("spontaneous") for r in root.findall(".//reactions/reaction")]


@pytest.mark.parametrize("value", [None, False, True])
def test_spontaneous_flag_off_is_plain_bool(flag_off, value) -> None:
    _, root = _build(_payload(reaction_spontaneous=False), spontaneous=value)
    nodes = _spont(root)
    assert len(nodes) == 1 and nodes[0].get("nil") is None
    assert nodes[0].text == ("true" if value else "false")


def test_spontaneous_flag_on_true_or_nil(flag_on) -> None:
    _, root = _build(_payload(reaction_spontaneous=False), spontaneous=True)
    node = _spont(root)[0]
    assert node.get("nil") is None and node.text == "true"
    for not_stated in (False, None):  # the IR default is False today: written as unknown, not as "enzyme-catalysed"
        _, root = _build(_payload(reaction_spontaneous=False), spontaneous=not_stated)
        node = _spont(root)[0]
        assert node.get("nil") == "true" and node.get("type") == "boolean"


def _transport_loc_refs(root):
    return [int(n.text) for n in root.findall(".//transport-visualizations//compound-location-id")]


def _loc_ids(root):
    return [int(n.text) for n in root.findall(".//compound-locations/compound-location/id")]


def test_transport_location_kept_when_compound_also_in_reaction_flag_on(flag_on) -> None:
    _, root = _build(_payload(reaction_uses_glucose=True))
    refs, ids = _transport_loc_refs(root), _loc_ids(root)
    assert refs
    assert set(refs) <= set(ids)
    assert len(ids) == len(set(ids))


def test_transport_location_dangles_when_flag_off(flag_off) -> None:
    _, root = _build(_payload(reaction_uses_glucose=True))
    refs, ids = _transport_loc_refs(root), _loc_ids(root)
    assert refs and not set(refs) <= set(ids)  # the documented defect, unchanged with the flag off


def test_transport_only_compound_unchanged_both_modes(monkeypatch) -> None:
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    _, off = _build(_payload())
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    _, on = _build(_payload())
    for root in (off, on):
        assert set(_transport_loc_refs(root)) <= set(_loc_ids(root))
