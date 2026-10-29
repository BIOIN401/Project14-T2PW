"""ORCH-792 (ORCH-789 C3, C6, D4): no deterministic Stage-6 step drops a protein row unrecorded.

Pinned by the offline removal tracer ``docs/pwml_recovery_sprint/evidence/orch792_trace.py``
(output ``orch792_trace.out.txt``): every vanished row of ORCH-789 (chl GBP, CHL27, LPOR,
ferrochelatase, Chl synthase, EXECUTER; wnt_ca PKC; wnt_pcp Frizzled, Wnt, Smurf, DIA1,
Profilin, MRLC, G-actin) leaves in ``_rewrite_reaction_protein_enzymes_to_complexes``, the
KI-007 drop ("no UniProt or DrugBank identifier; other enzymes remain"): the row was deleted
from ``entities.proteins`` and skipped in every other actor list, with no record.

**G9 -- CORRECTION of observable behaviour** (``test_g9_*``): each fails on base ``f3a80e09``.
**PRESERVATION** (``test_preservation_*``): passes at base and tip.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pytest

from t2pw.mapping import map_ids
from t2pw.pipeline.process_normalizer import GateValidationError, run_strict_post_normalization_gates
from test_orch759e_rewrap_after_phase2 import ORGANISM, _actor, _mapped, _payload, _protein, _run


def _names(payload: Dict[str, Any], bucket: str = "proteins") -> List[str]:
    return [r["name"] for r in payload["entities"].get(bucket) or []]


def _records(payload: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    return [r for r in payload.get("quarantined_proteins") or [] if r.get("name") == name]


def test_g9_an_enzyme_dropped_beside_a_resolved_one_keeps_its_row_and_is_recorded(tmp_path: Path) -> None:
    """The chlorophyll GBP shape: a no-id enzyme beside a resolved one, and named by an
    interaction. Base deleted the row, so the interaction was later omitted as
    ``undeclared_entity_in_entity_1`` (KI-015) instead of for the protein's real reason."""

    payload = _payload(["HEMA1", "GBP"], ["HEMA1", "GBP"], interactions=[
        {"name": "GBP binds HEMA1", "entity_1": "GBP", "entity_2": "HEMA1", "relationship": "binds",
         "evidence": "GBP binds HEMA1"}])
    out = _run(payload, tmp_path, main={"HEMA1": _mapped("HEMA1", "P42804")}, phase2={})["payload"]
    assert "GBP" in _names(out)
    enzymes = [a["entity"] for a in out["processes"]["reactions"][0]["enzymes"]]
    assert "GBP" not in enzymes and enzymes  # D-115 s3: the reaction keeps the enzyme that resolved
    [record] = _records(out, "GBP")
    assert record["reason"] == "reaction_enzyme_protein_dropped_no_external_id"
    assert record["action"] == "dropped_from_reaction_actor_list" and record["row_kept"] is True


def test_g9_the_same_enzyme_alone_in_another_reaction_keeps_its_place(tmp_path: Path) -> None:
    """Base skipped the dropped protein in EVERY later actor list, so a reaction where it is the
    only enzyme lost its enzyme silently (menaquinone ``DHNA prenylation`` shipped with none,
    ORCH-789 D2). Now the lone enzyme stays and takes the D-070 Unknown-backed complex (D-115)."""

    payload = _payload(["HEMA1", "CPO"], ["HEMA1", "CPO"])
    second = deepcopy(payload["processes"]["reactions"][0])
    second.update(name="CPO alone", enzymes=[_actor("CPO")])
    payload["processes"]["reactions"].append(second)
    out = _run(payload, tmp_path, main={"HEMA1": _mapped("HEMA1", "P42804")}, phase2={})["payload"]
    alone = out["processes"]["reactions"][1]
    assert [a["entity"] for a in alone["enzymes"]] == ["CPO"]


def test_g9_a_complex_component_a_process_names_is_not_pruned(tmp_path: Path) -> None:
    """wnt_ca TCF (ORCH-789 D4): ``_prune_invalid_complex_components`` removed an unresolved
    component as "no longer referenced" while two interactions named it."""

    payload = _payload(["HEMA1", "TCF", "NLK"], ["HEMA1"], interactions=[
        {"name": "NLK phosphorylates TCF", "entity_1": "NLK", "entity_2": "TCF", "relationship": "inhibits",
         "evidence": "NLK phosphorylates TCF"}])
    payload["entities"]["protein_complexes"] = [
        {"name": "HEMA1/TCF complex", "species": ORGANISM, "components": [{"name": "HEMA1"}, {"name": "TCF"}]}]
    out = _run(payload, tmp_path, main={"HEMA1": _mapped("HEMA1", "P42804"), "NLK": _mapped("NLK", "O43318")},
               phase2={})["payload"]
    assert "TCF" in _names(out)
    assert not [r for r in _records(out, "TCF") if r["reason"] == "unresolved_complex_component_no_longer_referenced"]


def _gate_messages(payload: Dict[str, Any]) -> List[str]:
    # A location row with a state is an edge (qa_graph.build_graph); the gate's "isolated"
    # case is a location row without one, which is what ORAI2's was.
    for row in payload["element_locations"]["protein_locations"]:
        if row["protein"] == "ORAI2":
            row.pop("biological_state", None)
    try:
        details = run_strict_post_normalization_gates(deepcopy(payload), enforce_all_proteins_connected=True)
    except GateValidationError as exc:
        details = exc.details
    return [str(e.get("reason") or e.get("message") or e) for e in (details or {}).get("errors") or []]


def test_g9_a_located_family_complex_member_is_not_isolated() -> None:
    """wnt_ca ORAI2 (ORCH-789 C6): a member of ``Orai family proteins`` with its own location
    row was "isolated in connectivity graph", which omitted the complex and both STIM rows."""

    payload = _payload(["STIM1", "ORAI2"], [], interactions=[
        {"name": "STIM1 activates Orai", "entity_1": "STIM1", "entity_2": "Orai family proteins",
         "relationship": "activates", "evidence": "STIM1 activates Orai"}])
    for row in payload["entities"]["proteins"]:
        row["mapped_ids"] = {"uniprot": {"STIM1": "Q13586", "ORAI2": "Q96SN7"}[row["name"]]}
    payload["entities"]["protein_complexes"] = [
        {"name": "Orai family proteins", "species": ORGANISM, "components": [{"name": "ORAI2"}]}]
    assert not [m for m in _gate_messages(payload) if "isolated in connectivity graph: ORAI2" in m]


def test_preservation_a_located_protein_attached_to_nothing_is_still_isolated() -> None:
    payload = _payload(["STIM1", "ORAI2"], [], interactions=[])
    for row in payload["entities"]["proteins"]:
        row["mapped_ids"] = {"uniprot": {"STIM1": "Q13586", "ORAI2": "Q96SN7"}[row["name"]]}
    assert [m for m in _gate_messages(payload) if "isolated in connectivity graph: ORAI2" in m]
