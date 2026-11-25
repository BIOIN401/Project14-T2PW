"""FIX-TRNA: T2PW_AUDIT_NA_DECLARED (default off) makes the deterministic audit count a nucleic acid
as a declared reference, so a reaction input that names one does not get a duplicate compound row."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation.audit_json_llm import audit_payload  # noqa: E402

FLAG = "T2PW_AUDIT_NA_DECLARED"


def _payload() -> dict:
    return {
        "entities": {
            "compounds": [{"name": "Glu"}],
            "nucleic_acids": [{"name": "tRNA Glu"}],
            "proteins": [{"name": "GluRS"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "Glu to Glu-tRNA Glu",
                    "inputs": ["Glu", "tRNA Glu", "Mystery X"],
                    "outputs": ["Glu-tRNA Glu"],
                    "enzymes": ["GluRS"],
                }
            ]
        },
    }


def _added_compounds(monkeypatch, value) -> list:
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)
    ops = audit_payload(_payload(), use_llm=False)["patch"]
    return [o["value"]["name"] for o in ops if o["op"] == "add" and o["path"] == "/entities/compounds/-"]


def test_flag_off_reproduces_the_duplicate(monkeypatch) -> None:
    added = _added_compounds(monkeypatch, None)
    assert "tRNA Glu" in added
    assert "Mystery X" in added


def test_flag_on_adds_no_duplicate_for_a_declared_nucleic_acid(monkeypatch) -> None:
    for v in ("1", "true", "YES"):
        assert "tRNA Glu" not in _added_compounds(monkeypatch, v)


def test_flag_on_still_adds_an_undeclared_reference(monkeypatch) -> None:
    added = _added_compounds(monkeypatch, "1")
    assert "Mystery X" in added
    assert "Glu-tRNA Glu" in added  # an undeclared output is still added as before


@pytest.mark.parametrize("v", ["", "0", "off"])
def test_other_values_are_off(monkeypatch, v) -> None:
    assert "tRNA Glu" in _added_compounds(monkeypatch, v)


def _transport_payload() -> dict:
    return {
        "entities": {
            "nucleic_acids": [{"name": "tRNA Glu"}],
            "subcellular_locations": [{"name": "cytosol"}, {"name": "nucleus"}],
        },
        "processes": {
            "reactions": [],
            "transports": [
                {"name": "tRNA Glu export", "cargo": "tRNA Glu", "from_location": "nucleus", "to_location": "cytosol"}
            ],
        },
    }


def _transport_run(monkeypatch, on: bool):
    if on:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)
    res = audit_payload(_transport_payload(), use_llm=False)
    added = [o["value"]["name"] for o in res["patch"] if o["op"] == "add" and o["path"] == "/entities/compounds/-"]
    cargo_errors = [e for e in res["report"]["errors"] if "cargo" in e.get("reason", "")]
    return added, cargo_errors


def test_transport_cargo_flag_off_adds_duplicate_and_reports_it_missing(monkeypatch) -> None:
    added, cargo_errors = _transport_run(monkeypatch, False)
    assert added == ["tRNA Glu"]
    assert cargo_errors


def test_transport_cargo_flag_on_counts_a_nucleic_acid_as_declared(monkeypatch) -> None:
    added, cargo_errors = _transport_run(monkeypatch, True)
    assert added == []
    assert not cargo_errors


def test_flag_on_removes_the_missing_entity_error_and_warning_for_the_reaction_input(monkeypatch) -> None:
    def run(on):
        if on:
            monkeypatch.setenv(FLAG, "1")
        else:
            monkeypatch.delenv(FLAG, raising=False)
        r = audit_payload(_payload(), use_llm=False)["report"]
        return (
            [e["evidence"] for e in r["errors"] if "does not exist" in e["reason"]],
            [w["evidence"] for w in r["warnings"] if w["reason"].startswith("Added missing compound")],
        )

    off_err, off_warn = run(False)
    on_err, on_warn = run(True)
    assert "tRNA Glu" in off_err and "tRNA Glu" in off_warn
    assert "tRNA Glu" not in on_err and "tRNA Glu" not in on_warn
