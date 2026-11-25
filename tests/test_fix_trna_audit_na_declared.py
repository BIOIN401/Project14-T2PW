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


def test_flag_off_output_is_unchanged_by_the_change(monkeypatch) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    a = audit_payload(_payload(), use_llm=False)
    monkeypatch.setenv(FLAG, "0")
    b = audit_payload(_payload(), use_llm=False)
    assert a == b
