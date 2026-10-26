"""ORCH-774s -- the morning summary lists each paper's omissions with KI ids, counts the reactions PathWhiz would drop on
import, and shows a "would import" coverage.

**G9 LABEL: NEW ACCEPTANCE TESTS FOR A NEW CAPABILITY.** ``no_enzyme_reactions``, ``omitted_lines`` and the "would import"
figure do not exist at base ``1f223061``; no base failure is claimed. The cell of a leg with no PWML is unchanged (its pin
is ``tests/test_orch772b_no_pwml_cause.py``).
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "docs" / "pwml_recovery_sprint" / "evidence"
FIXTURE = ROOT / "tests" / "fixtures" / "orch768"


def _summary():
    spec = importlib.util.spec_from_file_location("orch770_morning_summary_774s", EV / "orch770_morning_summary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reaction(enzyme: bool, spontaneous: str = "false") -> str:
    enzymes = ("<reaction-enzymes><reaction-enzyme><id type=\"integer\">9</id>"
               "<protein-complex-id type=\"integer\">40000</protein-complex-id></reaction-enzyme></reaction-enzymes>"
               if enzyme else "<reaction-enzymes/>")
    return f"<reaction><id type=\"integer\">1</id><spontaneous type=\"boolean\">{spontaneous}</spontaneous>{enzymes}</reaction>"


def _leg(tmp_path: Path, reactions_xml: str) -> Path:
    """Two drawn reactions in payload, IR and PWML; the PWML says which carry an enzyme."""

    leg = tmp_path / "leg"
    leg.mkdir()
    payload = {"processes": {"reactions": [{"name": "R1", "inputs": ["A"], "outputs": ["B"]},
                                           {"name": "R2", "inputs": ["B"], "outputs": ["C"]}]}}
    ir = {"processes": {"reactions": [{"name": "R1"}, {"name": "R2"}], "transports": [], "interactions": []}}
    (leg / "final_mapped.json").write_text(json.dumps(payload), encoding="utf-8")
    (leg / "pwml_ir.json").write_text(json.dumps(ir), encoding="utf-8")
    (leg / "pathway.review_required.pwml").write_text(
        f"<?xml version=\"1.0\"?><pathwhiz><pathway><reactions>{reactions_xml}</reactions></pathway></pathwhiz>", encoding="utf-8")
    return leg


def test_new_a_reaction_with_no_enzyme_and_not_spontaneous_is_counted_from_the_xml(tmp_path):
    summary = _summary()
    leg = _leg(tmp_path, _reaction(True) + _reaction(False) + _reaction(False, spontaneous="true"))
    assert summary.no_enzyme_reactions(leg) == 1  # the spontaneous one imports: PathWhiz accepts it without an enzyme
    assert summary.no_enzyme_reactions(tmp_path) is None  # no PWML written


def test_new_the_cell_shows_what_would_survive_import(tmp_path):
    summary = _summary()
    leg = _leg(tmp_path, _reaction(True) + _reaction(False))
    assert summary.coverage(leg) == "2/2 (100%); full count 2/2 (100%); would import 1/2 (50%)"


def test_new_each_paper_lists_its_omissions_with_their_known_issues(tmp_path):
    """The ORCH-767 enterobactin leg as it ran (no PWML): 17 processes lost with the leg (KI-012) and 4 removed before
    the freeze (KI-001), every one on its own line."""

    leg_meta = next(l for l in json.loads((FIXTURE / "legs.json").read_text(encoding="utf-8"))["legs"]
                    if l["id"] == "ORCH-767/enterobactin-strict/2026-10-23_2224")
    leg = tmp_path / "enterobactin"
    leg.mkdir()
    for name, meta in leg_meta["inputs"].items():
        if meta["path"]:
            (leg / name).write_bytes(gzip.decompress((FIXTURE / meta["path"]).read_bytes()))
    lines = _summary().omitted_lines("enterobactin", leg)
    assert lines[0] == "### enterobactin: 21 processes, 0 actors left out"
    assert lines[2] == "By KI: KI-012 17, KI-001 4"
    rows = [line for line in lines if line.startswith("- ")]
    assert len(rows) == 21 and all(line.rsplit(": ", 1)[1] in ("KI-012", "KI-001") for line in rows)
    assert _summary().no_enzyme_reactions(leg) is None
