"""ORCH-769b: quarantine judges an enzyme/modifier/transporter by the final gate's own predicate.

**G9: A REGRESSION FIX, NOT NEW WORK.** Every name used here exists at the base SHA
``e3d5b65f`` with the same signature, so this module imports cleanly there and every
``test_g9_*`` test fails there on VALUES: the reaction is admitted, and the strict final
gate then raises "Unknown protein/modifier reference". The ``test_preservation_*`` tests
pass at base and at tip by design; they fail only if the fix reaches further than the gate.

THE DEFECT. ``_actor_verdict`` accepted an actor when its name resolved to ANY declared
entity (compounds, proteins, protein_complexes, nucleic_acids, element_collections) and
that entity was representable. The final gate (``process_normalizer.
run_strict_post_normalization_gates``) accepts a dict actor of a reaction's ``enzymes`` /
``modifiers`` or a transport's ``transporters`` only when its normalized name is a name or
synonym of a row in ``proteins`` or ``protein_complexes``. On the ORCH-767 chlorophyll-strict
leg (PMC5130987, 2026-10-23_2328) the audit had deleted the protein rows PIF1, GBP, FAR1,
CPO1 and PPO1 while same-named gene rows survived, so the five reactions naming them were
admitted and the final gate raised 9 x ``unknown_protein_modifier_reference``: no PWML for
a ``review_required`` pathway (merge rule 7).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from t2pw.pipeline.process_normalizer import (
    GateValidationError,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.strict_quarantine import (
    ACCEPTED_STATES,
    QUARANTINED_BROKEN_REFERENCE,
    QUARANTINED_DISCONNECTED,
    quarantine_and_close,
)

from test_strict_quarantine_contract_alignment import _base

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "orch769b"

#: sha256 of the leg's ``final_mapped.json`` (uncompressed). See the fixture README.
LEG_FINAL_MAPPED_SHA256 = "5805a848ac81e4eafcd7f12d401f63fb63aaf84d3168f75c7174ed9fd3484f8c"

#: (index, name, reason, participant) of every process the tip quarantines on the leg.
LEG_QUARANTINED = [
    (17, "GluTR (HEMA1) reaction", "non_protein_actor_in_modifiers", "PIF1"),
    (18, "ALAD (HEMB1) reaction", "non_protein_actor_in_modifiers", "FAR1"),
    (21, "Glu-tRNAGlu reduction to GSA", "non_protein_actor_in_enzymes", "GBP"),
    (22, "Coprogen III to Protogen IX", "non_protein_actor_in_enzymes", "CPO1"),
    (23, "Protogen IX to Proto IX", "non_protein_actor_in_enzymes", "PPO1"),
]

_UNKNOWN_ACTOR = ("Unknown protein/modifier reference", "Unknown transporter reference")


def _leg() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = gzip.decompress((FIXTURES / "chlorophyll_PMC5130987_767_final_mapped.json.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == LEG_FINAL_MAPPED_SHA256
    context = json.loads(
        (FIXTURES / "chlorophyll_PMC5130987_767_requested_context.json").read_text(encoding="utf-8")
    )
    return json.loads(raw.decode("utf-8")), context


def _gate_errors(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``(path, reason)`` of every final-gate error, run as ``freeze_canonical_payload`` runs it."""

    try:
        run_strict_post_normalization_gates(deepcopy(payload), enforce_all_proteins_connected=True)
    except GateValidationError as exc:
        return [(str(row.get("path")), str(row.get("reason"))) for row in exc.details.get("errors") or []]
    return []


def _unknown_actor_errors(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    """The final gate's unknown-actor errors on ``payload`` (its other checks are not this lane's)."""

    return [row for row in _gate_errors(payload) if row[1].startswith(_UNKNOWN_ACTOR)]


def _record(result, name: str) -> Dict[str, Any]:
    return next(row for row in result.admissions if row["name"] == name)


# ── G9 on the leg's own input ───────────────────────────────────────────────


def test_g9_leg_quarantines_exactly_the_five_reactions_whose_actor_is_no_protein() -> None:
    payload, context = _leg()
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    report = result.quarantine_report

    # The live run's decision input was sha256:6c5f9c8b2410d8af; ORCH-773 bumped QUARANTINE_POLICY_VERSION
    # (2026-07-31.1 -> 2026-10-25.1), which moves it and nothing else (merge rule 4); ORCH-780's bump
    # (-> 2026-10-26.1) moves it again, sha256:0bb36bd9e1f252d9 -> below.
    assert report["decision_input_hash"] == "sha256:a5cf19b0eebf9c75"
    quarantined = [
        (row["index"], row["name"], row["reason"], row["essential_participant"])
        for row in result.admissions
        if row["state"] not in ACCEPTED_STATES
    ]
    # Base: [] -- every one of the 32 processes admitted.
    assert quarantined == LEG_QUARANTINED
    assert all(
        row["state"] == QUARANTINED_BROKEN_REFERENCE
        for row in result.admissions
        if row["state"] not in ACCEPTED_STATES
    )
    assert report["counts"] == {
        "core_accepted": 17,
        "auxiliary_accepted": 10,
        "quarantined_unmapped_entity": 0,
        "quarantined_out_of_scope": 0,
        "quarantined_weak_evidence": 0,
        "quarantined_broken_reference": 5,
        "quarantined_disconnected": 0,
    }
    assert result.ok and result.refusal_reasons == []
    assert report["release"]["status"] == "review_required"
    # The cost, pinned so it cannot move silently: GLK1 was reached only through r17.
    assert report["release"]["missing_anchors"] == ["POR (PORB)", "GLK1"]
    processes = result.payload["processes"]
    assert (len(processes["reactions"]), len(processes["interactions"])) == (22, 5)


def test_g9_leg_final_gate_passes_on_the_quarantined_payload() -> None:
    """Base: the gate raises 9 errors (PIF1, GBP x3, FAR1, CPO1 x2, PPO1 x2)."""

    payload, context = _leg()
    assert len(_unknown_actor_errors(payload)) == 9  # the leg's own final gate report
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    assert result.ok
    # freeze_canonical_payload runs exactly this once quarantine is ok.
    assert _gate_errors(result.payload) == []


# ── G9 synthetic: each adds one actor to a pathway that passes both gates ───


def test_g9_a_modifier_declared_only_as_a_gene_is_quarantined() -> None:
    payload = _base()
    payload["entities"]["nucleic_acids"].append({"name": "PIF1", "species": "Homo sapiens"})
    payload["processes"]["reactions"][1]["modifiers"] = [{"entity": "PIF1", "role": "activator"}]
    assert _unknown_actor_errors(payload)  # the gate rejects the input

    result = quarantine_and_close(payload, strict_db=True)

    record = _record(result, "glutathione synthesis")
    assert (record["state"], record["reason"], record["essential_participant"]) == (
        QUARANTINED_BROKEN_REFERENCE,
        "non_protein_actor_in_modifiers",
        "PIF1",
    )
    assert _record(result, "gamma-glutamylcysteine synthesis")["state"] in ACCEPTED_STATES
    assert _unknown_actor_errors(result.payload) == []


def test_g9_a_compound_named_as_a_modifier_is_quarantined() -> None:
    """The gate's registry holds no compounds, so a compound modifier fails it too."""

    payload = _base()
    payload["processes"]["reactions"][1]["modifiers"] = [{"entity": "L-cysteine", "role": "inhibitor"}]
    assert _unknown_actor_errors(payload)

    result = quarantine_and_close(payload, strict_db=True)

    record = _record(result, "glutathione synthesis")
    assert (record["state"], record["reason"], record["essential_participant"]) == (
        QUARANTINED_BROKEN_REFERENCE,
        "non_protein_actor_in_modifiers",
        "L-cysteine",
    )
    assert _unknown_actor_errors(result.payload) == []


def test_g9_a_transporter_declared_only_as_a_gene_is_quarantined() -> None:
    payload = _base()
    payload["entities"]["nucleic_acids"].append({"name": "PIF1", "species": "Homo sapiens"})
    payload["processes"]["transports"] = [
        {
            "name": "glutathione efflux",
            "cargo": "glutathione",
            "from_biological_state": "cytosol",
            "to_biological_state": "cytosol",
            "transporters": [{"entity": "PIF1"}],
        }
    ]
    assert _unknown_actor_errors(payload)

    result = quarantine_and_close(payload, strict_db=True)

    record = _record(result, "glutathione efflux")
    assert (record["state"], record["reason"], record["essential_participant"]) == (
        QUARANTINED_BROKEN_REFERENCE,
        "non_protein_actor_in_transporters",
        "PIF1",
    )
    assert _unknown_actor_errors(result.payload) == []


def test_g9_closure_revalidation_applies_the_same_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The leg's mechanism, at the closure stage: the protein row goes, a same-named gene stays.

    ``GSS`` is a synonym of the protein ``glutathione synthetase`` and the name of a gene row,
    so admission resolves it to the protein (proteins register before nucleic_acids) and
    admits. A removal behind closure's back then deletes the protein, as the audit did on the
    leg. Base: revalidation resolves ``GSS`` to the gene and keeps the reaction.
    """

    from t2pw.pipeline import strict_quarantine as sq

    payload = _base()
    payload["entities"]["proteins"][1]["synonyms"] = ["GSS"]
    payload["entities"]["nucleic_acids"].append({"name": "GSS", "species": "Homo sapiens"})
    payload["processes"]["reactions"][1]["enzymes"] = []
    payload["processes"]["reactions"][1]["modifiers"] = [{"entity": "GSS", "role": "catalyst"}]

    real_prune = sq._prune_entities
    calls = {"n": 0}

    def _prune_and_also_drop_the_protein(working, keep_norms):
        removed = real_prune(working, keep_norms)
        calls["n"] += 1
        if calls["n"] == 1:
            proteins = working["entities"]["proteins"]
            working["entities"]["proteins"] = [
                row for row in proteins if row.get("name") != "glutathione synthetase"
            ]
            removed.append({"bucket": "proteins", "name": "glutathione synthetase", "reason": "simulated"})
        return removed

    monkeypatch.setattr(sq, "_prune_entities", _prune_and_also_drop_the_protein)
    result = sq.quarantine_and_close(payload, strict_db=True)

    record = _record(result, "glutathione synthesis")
    assert record["state"] == QUARANTINED_DISCONNECTED
    assert record["reason"] == "stranded_by_closure:non_protein_actor_in_modifiers"
    assert record["essential_participant"] == "GSS"
    assert _unknown_actor_errors(result.payload) == []


# ── Preservation: what the gate accepts, quarantine still admits ────────────


def test_preservation_a_modifier_matching_a_protein_synonym_is_admitted() -> None:
    payload = _base()
    payload["entities"]["proteins"][1]["synonyms"] = ["GSS"]
    payload["processes"]["reactions"][1]["modifiers"] = [{"entity": "GSS", "role": "activator"}]

    result = quarantine_and_close(payload, strict_db=True)

    assert _record(result, "glutathione synthesis")["state"] in ACCEPTED_STATES
    assert _unknown_actor_errors(result.payload) == []


def test_preservation_a_complex_named_modifier_is_admitted() -> None:
    payload = _base()
    payload["processes"]["reactions"][1]["modifiers"] = [
        {"entity": "glutamate-cysteine ligase complex", "role": "activator"}
    ]

    result = quarantine_and_close(payload, strict_db=True)

    assert _record(result, "glutathione synthesis")["state"] in ACCEPTED_STATES
    assert _unknown_actor_errors(result.payload) == []


def test_preservation_the_predicate_is_the_gate_registry_not_the_resolved_bucket() -> None:
    """``GSH synthetase`` is a compound synonym AND a protein synonym.

    The registry resolves it to the compound (compounds register first), but the gate
    accepts it because it is in the protein registry. A rule keyed on the resolved bucket
    would quarantine a reaction the gate passes; the mirrored predicate does not.
    """

    payload = _base()
    payload["entities"]["compounds"][4]["synonyms"] = ["GSH synthetase"]
    payload["entities"]["proteins"][1]["synonyms"] = ["GSH synthetase"]
    payload["processes"]["reactions"][1]["modifiers"] = [{"entity": "GSH synthetase", "role": "activator"}]

    result = quarantine_and_close(payload, strict_db=True)

    assert _record(result, "glutathione synthesis")["state"] in ACCEPTED_STATES
    assert _unknown_actor_errors(result.payload) == []


def test_preservation_a_string_actor_is_outside_the_rule_as_it_is_outside_the_gate() -> None:
    """The gate reads dict actors only (``if not isinstance(actor, dict): continue``).

    A bare-string actor naming a gene is admitted at base and at tip; the rule does not
    reach wider than the gate. (Stage-8 revalidation's ``actor_schema_not_canonical`` is
    what refuses a non-canonical actor shape; unchanged here.)
    """

    payload = _base()
    payload["entities"]["nucleic_acids"].append({"name": "PIF1", "species": "Homo sapiens"})
    payload["processes"]["reactions"][1]["modifiers"] = ["PIF1"]

    result = quarantine_and_close(payload, strict_db=True)

    assert _record(result, "glutathione synthesis")["state"] in ACCEPTED_STATES
    assert _unknown_actor_errors(result.payload) == []
