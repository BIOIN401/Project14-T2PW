"""ORCH-773 (D-114 section 2): a per-row export failure omits its rows before the freeze; the pathway exports.

**G9: A REGRESSION FIX OF OBSERVABLE BEHAVIOUR.** On each fixture leg below the base (``3ef3fa9c``) writes
no PWML for a pathway that is ``review_required`` because ONE row fails ONE per-row requirement: the strict
quarantine refuses (``unexportable_entity``), or the final pre-export gate, the Stage-8 contract, the
required-field gate or the IR refuses. Every name imported here exists at base with the same signature, so
the module imports there and every ``test_g9_*`` fails there on VALUES (``_refusals`` is not empty, nothing
is recorded as omitted). ``test_preservation_*`` pass at base and at tip: they fail only if the omission
reaches further than the gates. ``test_new_release_cap`` is a NEW acceptance test of a new function (it
fails at base on its import, which is inside the test) and claims no base failure as G9.

Round 1 (D-114 AMENDMENT 1; orchestrator reading of D-114 section 2): a transport whose transporter fails keeps
the transport and loses only that transporter (``test_g9_stage8_non_canonical_transporter_*``, and the two
Wnt/Ca2+ transports); and any omitted process or dropped actor in the frozen record caps the release at
``review_required``, including a process quarantined at admission
(``test_g9_a_process_quarantined_at_admission_caps_a_release_ready_pathway``, on a small payload).

Each fixture is the exact ``{"payload", "pathway_context"}`` the ORCH-768 replay corpus hands
``freeze_canonical_payload`` on that leg (``tests/fixtures/orch773/README.md``). ``_refusals`` runs what the
freeze and ``run_pwml_export`` run after the quarantine, in their order, with the export's own pre-freeze
seam before the IR: an empty list means a PWML is written.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from t2pw.pipeline.process_normalizer import GateValidationError, run_strict_post_normalization_gates
from t2pw.pipeline.stage_contracts import StageContractError, validate_post_normalization, validate_pre_export
from t2pw.pipeline.strict_quarantine import (
    QUARANTINED_BROKEN_REFERENCE,
    QUARANTINED_UNMAPPED_ENTITY,
    quarantine_and_close,
)
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir
from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution
from t2pw.pwml.writer import blocking_pwml_ir_errors

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "orch773"
ORCH768 = HERE / "fixtures" / "orch768"

#: sha256 of each uncompressed fixture (``evidence/orch773/make_fixtures.py``, G11 ``ORCH-773/12``).
SHA256 = {
    "ORCH-755_enterobactin-strict_2026-10-20_2201": "211f2e1f518af5ff1356f4fbd84cf10ecdf9efe31a32726d217f79ef9b2e853c",
    "ORCH-767_wnt_ca-strict_2026-10-24_0314": "bec4807ba92c62194544a67a3ef950af6fc8f06f2a45fd42547844c6a1013305",
    "ORCH-759_enterobactin-strict_2026-10-22_1037": "6ef32a705a4dd37237140761be17a403f231b621b2e4ae2f9c0438e1906d83bd",
    "runs_verify_2026-08-21_2057_PMC12452463": "f3d8a4bd99a15faefcae791dafb7b968ce025194288913bc60311cc226d84f83",
    "ORCH-755_sour_taste-strict_2026-10-21_0139": "2dc5bebcf72ebb83841a9251bf8c90c519b83cc0394608880253875ba3e54dc9",
    "runs_verify_2026-08-21_2057_PMC12096016": "2af4a3f70971ed7dfb82c9841dd17a6efd4674e9298621dffe521d49f8187bcc",
    "runs_verify_2026-08-25_1216_PMC12856317": "a373924987db5847646c2fac1159040b126738aeff1066711f1f1feaf4ad7042",
    "runs_verify_2026-08-21_2239_PMC12180156": "46dbca4805be35169af4dbf119e165ed4e57b6187d8026f0b5b8425f696cc2a0",
}


def _leg(slug: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = gzip.decompress((FIXTURES / f"{slug}.freeze_input.json.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == SHA256[slug]
    doc = json.loads(raw.decode("utf-8"))
    return doc["payload"], doc["pathway_context"]


def _orch768_leg(slug: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """(final_mapped, pathway_context, recorded quarantine report) of an ORCH-768 fixture leg."""

    def load(name: str) -> Dict[str, Any]:
        return json.loads(gzip.decompress((ORCH768 / slug / f"{name}.gz").read_bytes()).decode("utf-8"))

    report = load("quarantine_report.json")
    return load("final_mapped.json"), (report.get("coverage") or {}).get("requested_context"), report


def _quarantine(payload: Dict[str, Any], context: Any, mode: str = "pathwhiz"):
    return quarantine_and_close(payload, strict_db=True, pathway_context=context, mode=mode)


def _refusals(result) -> List[str]:
    """What stops a PWML after this quarantine, in the freeze's and the export's order. ``[]`` = written."""

    if not result.ok:
        return [f"quarantine:{reason}" for reason in result.refusal_reasons]
    payload = result.payload
    try:
        final = {"ok": True, **run_strict_post_normalization_gates(deepcopy(payload), enforce_all_proteins_connected=True)}
    except GateValidationError as exc:
        return [f"final:{row.get('reason')}" for row in exc.details.get("errors") or []]
    stage8 = validate_post_normalization(deepcopy(payload), final)
    if not stage8.get("ok", False):
        return [f"stage8:{row.get('code')}" for row in stage8.get("errors") or []]
    gate_payload = deepcopy(payload)
    gate_payload.setdefault("metadata", {}).update(
        {"pathway_name": "Generated Pathway", "name": "Generated Pathway", "pathway_subject": "Metabolic",
         "subject": "Metabolic", "description": "", "width": 3200, "height": 1400}
    )
    try:
        required = validate_pre_export(gate_payload, strict_db=True)
    except StageContractError as exc:
        required = exc.report or {}
    contract = required.get("pwml_contract_report") or {}
    if not contract.get("ok", False):
        return [f"required:{row.get('code')}" for row in contract.get("errors") or []]
    export = deepcopy(payload)
    run_prefreeze_resolution(export, strict_db=True)
    ir, report = build_pwml_ir(export, pathway_name="Generated Pathway", pathway_subject="Metabolic", strict_db=True,
                               width=3200, height=1400)
    errors = [*blocking_pwml_ir_errors(report), *(validate_pwml_ir(ir).get("errors") or [])]
    return [f"ir:{row.get('code')}" for row in errors]


def _omitted(result) -> List[Tuple[str, str, str, str, str, str]]:
    """(name, state, reason, cause, slot, via) of every process the per-row pass omitted."""

    return [
        (row["name"], row["state"], row["reason"], row["detail"]["omission"]["cause"],
         row["detail"]["omission"]["slot"], row["detail"]["omission"]["name"])
        for row in result.quarantine_report["quarantined"]
        if "omission" in (row.get("detail") or {})
    ]


def _removed(result) -> List[Tuple[str, str, str]]:
    """(bucket, name, reason) of every entity removed and every transporter dropped by the per-row pass."""

    return [
        (row["bucket"], row["name"], row["reason"])
        for row in result.removed_entity_report["removed_entities"]
        if row.get("detail")
    ]


def _dropped_transporters(result) -> List[Tuple[str, str, str]]:
    """(transport name, transporter name, reason) of every transporter dropped from a kept transport."""

    return [
        (entry["process_name"], row["name"], row["reason"])
        for row in result.removed_entity_report["removed_entities"]
        if row["bucket"] == "transporters"
        for entry in row["detail"]["actor_removed_from"]
    ]


def _review_required(result, actors: int = 0) -> None:
    """``review_required``, never strict success; and the cap reason exactly when something was omitted."""

    report = result.quarantine_report
    release = report["release"]
    assert release["status"] == "review_required"
    assert release["strict_acceptance_eligible"] is False
    processes = sum(1 for row in report["admissions"] if row["state"] not in ("core_accepted", "auxiliary_accepted"))
    expected = [f"omitted_from_pwml:{processes}_processes+{actors}_actors"] if processes or actors else []
    assert [r for r in release["reasons"] if r.startswith("omitted_from_pwml:")] == expected
    assert [r for r in report["review_reasons"] if r.startswith("omitted_from_pwml:")] == expected


def _surviving(result, bucket: str) -> List[Dict[str, Any]]:
    return result.payload["processes"].get(bucket) or []


# ── G9: the strict quarantine's own refusal ────────────────────────────────


def test_g9_unexportable_component_takes_its_complex_and_processes_not_the_pathway() -> None:
    """ORCH-755 enterobactin: EntB (no UniProt id, KI-017) is a component of the declared complex
    'isochorismatase'. Base: ``unexportable_entity:1``, no PWML."""

    payload, context = _leg("ORCH-755_enterobactin-strict_2026-10-20_2201")
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    # Round 1 (N1): the declared complex left because its component was omitted, which the reason now says.
    reason = "protein_complex_component_omitted:EntB:protein_missing_external_identity"
    assert _omitted(result) == [
        ("EntB isochorismatase reaction", QUARANTINED_UNMAPPED_ENTITY, reason, "actor", "enzymes", "isochorismatase"),
        ("2,3-DHB competitive inhibition of EntB", QUARANTINED_UNMAPPED_ENTITY, reason, "process", "entity_2", "isochorismatase"),
    ]
    assert _removed(result) == [
        ("protein_complexes", "isochorismatase", reason),
        ("proteins", "EntB", "protein_missing_external_identity"),
    ]
    # A reaction whose enzyme fails is still omitted whole (D-114 AMENDMENT 1 s1).
    _review_required(result)


def test_g9_unexportable_row_kept_only_by_a_synonym_is_removed_alone() -> None:
    """ORCH-767 Wnt/Ca2+: 'phosphorylated CAMKII' (no identity, KI-018) survives only because it carries the
    synonym 'calcineurin B', which every process resolves to the real 'calcineurin B' row. Base: refusal.
    The next round then meets the Stage-8 contract: two transports with a non-canonical transporter, which
    keep the transport and lose only that transporter (D-114 AMENDMENT 1 s2)."""

    payload, context = _leg("ORCH-767_wnt_ca-strict_2026-10-24_0314")
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    insp3r = "inositol-1,4,5-trisphosphate receptor (InsP3R)"
    assert _removed(result) == [
        ("proteins", "phosphorylated CAMKII", "protein_missing_external_identity"),
        ("transporters", insp3r, "actor_schema_not_canonical"),
        ("transporters", insp3r, "actor_schema_not_canonical"),
    ]
    assert _dropped_transporters(result) == [
        ("IP3-induced Ca2+ release", insp3r, "actor_schema_not_canonical"),
        ("SERCA pumps Ca2+ into ER", insp3r, "actor_schema_not_canonical"),
    ]
    assert _omitted(result) == []
    transports = {row["name"]: row for row in _surviving(result, "transports")}
    for name in ("IP3-induced Ca2+ release", "SERCA pumps Ca2+ into ER"):
        assert transports[name]["transporters"] == []
    assert "calcineurin B component of calcineurin complex" in {row["name"] for row in _surviving(result, "interactions")}
    _review_required(result, actors=2)


# ── G9: the final pre-export gate (freeze) ─────────────────────────────────


def test_g9_final_gate_degree_zero_protein_is_removed_and_every_reaction_stays() -> None:
    """ORCH-759 enterobactin: the stray row 'enterobactin synthase' (KI-013) is attached to nothing by the
    final gate's own test. Base: "Protein has degree 0 after normalization", no PWML."""

    payload, context = _leg("ORCH-759_enterobactin-strict_2026-10-22_1037")
    reactions_in = len(payload["processes"]["reactions"])
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    assert _removed(result) == [
        ("proteins", "enterobactin synthase", "Protein has degree 0 after normalization: enterobactin synthase")
    ]
    assert _omitted(result) == []
    assert len(_surviving(result, "reactions")) == reactions_in
    _review_required(result)


def test_g9_d113_inferred_wrapper_leaves_its_reaction_and_the_paper_catalyst_stays() -> None:
    """ORCH-767 enterobactin (ORCH-769a, D-113 s1): Stage 2 inferred 'enterobactin synthase complex', whose
    component has no stoichiometry; the reaction keeps the paper's 'EntF complex'. Base: final gate, no PWML."""

    payload, context, _recorded = _orch768_leg("ORCH-767_enterobactin-strict_2026-10-23_2224")
    reactions_in = len(payload["processes"]["reactions"])
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    message = ("Generated protein complex 'enterobactin synthase complex' component 'enterobactin synthase' "
               "is missing positive stoichiometry.")
    assert ("protein_complexes", "enterobactin synthase complex", message) in _removed(result)
    wrapper = next(row for row in result.removed_entity_report["removed_entities"]
                   if row["name"] == "enterobactin synthase complex")
    (removal,) = wrapper["detail"]["actor_removed_from"]
    assert removal["process_name"] == "EntF enterobactin synthase reaction"
    assert removal["slots"] == ["enzymes", "modifiers"]
    assert "EntF complex" in removal["catalysts_kept"]
    assert _omitted(result) == []
    reaction = next(row for row in _surviving(result, "reactions") if row["name"] == "EntF enterobactin synthase reaction")
    enzymes = [actor.get("entity") for actor in reaction["enzymes"]]
    assert "EntF complex" in enzymes and "enterobactin synthase complex" not in enzymes
    assert len(_surviving(result, "reactions")) == reactions_in
    _review_required(result, actors=1)


def test_g9_final_gate_wrapper_with_no_other_catalyst_omits_its_reaction() -> None:
    """runs_verify 2026-08-21_2057 PMC12452463: a generated 'enterobactin synthase complex' (EntD/EntE/EntF,
    no stoichiometry) is its reaction's only catalyst, so D-113 s1 does not apply: the process leaves."""

    payload, context = _leg("runs_verify_2026-08-21_2057_PMC12452463")
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    message = ("Generated protein complex 'enterobactin synthase complex' component 'EntD' "
               "is missing positive stoichiometry.")
    assert _omitted(result) == [
        ("enterobactin synthase complex reaction", QUARANTINED_UNMAPPED_ENTITY, message, "actor", "enzymes",
         "enterobactin synthase complex"),
    ]
    assert _removed(result) == [("protein_complexes", "enterobactin synthase complex", message)]
    _review_required(result)


# ── G9: the export's gates ─────────────────────────────────────────────────


def test_g9_stage8_non_canonical_transporter_is_dropped_and_its_transport_stays() -> None:
    """ORCH-755 sour taste (KI-023): three transports name OTOP1 in Stage-1 shape. Base: Stage-8 stop.
    D-114 AMENDMENT 1 s2: each transport stays and loses only that transporter, recorded with the gate's code."""

    payload, context = _leg("ORCH-755_sour_taste-strict_2026-10-21_0139")
    names = ("OTOP1-mediated H+ transport", "H+ permeation through OTOP1", "OTOP1-mediated H+ permeation")
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    assert _omitted(result) == []
    assert _dropped_transporters(result) == [(name, "OTOP1", "actor_schema_not_canonical") for name in names]
    dropped = [row for row in result.removed_entity_report["removed_entities"] if row["bucket"] == "transporters"]
    assert [row["detail"]["actor"] for row in dropped] == [
        transport["transporters"][0] for transport in payload["processes"]["transports"] if transport["name"] in names
    ]
    transports = {row["name"]: row for row in _surviving(result, "transports")}
    assert all(transports[name]["transporters"] == [] for name in names)
    _review_required(result, actors=3)


def test_g9_required_gate_bare_protein_enzyme_omits_its_reaction() -> None:
    """runs_verify 2026-08-21_2057 PMC12096016 (KI-006): 'lactate dehydrogenase' is a bare protein enzyme.
    Base: "PWML required-field gate failed", no PWML."""

    payload, context = _leg("runs_verify_2026-08-21_2057_PMC12096016")
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    assert _omitted(result) == [
        ("LDH coupled reaction", QUARANTINED_BROKEN_REFERENCE, "reaction_enzyme_must_be_protein_complex", "actor",
         "enzymes", "lactate dehydrogenase"),
    ]
    _review_required(result)


def test_g9_ir_complex_without_components_omits_its_interaction() -> None:
    """runs_verify 2026-08-25_1216 PMC12856317: the IR resolves none of 'LONP1''s components.
    Base: "PWML IR validation failed", no PWML."""

    payload, context = _leg("runs_verify_2026-08-25_1216_PMC12856317")
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    assert _omitted(result) == [
        ("LONP1 degrades ALAS1", QUARANTINED_UNMAPPED_ENTITY, "protein_complex_missing_components", "process",
         "entity_1", "LONP1"),
    ]
    assert _removed(result) == [("protein_complexes", "LONP1", "protein_complex_missing_components")]
    _review_required(result)


# ── G9: the release cap (orchestrator reading of D-114 s2) ─────────────────


def _cap_toy(*, quarantined_reaction: bool) -> Dict[str, Any]:
    """The contract-alignment pathway (release_ready on its own), optionally with one reaction admission
    quarantines: its input names nothing declared."""

    from test_strict_quarantine_contract_alignment import _base

    payload = _base()
    if quarantined_reaction:
        payload["processes"]["reactions"].append(
            {"name": "glutathione export", "inputs": ["glutathione", "an undeclared carrier"],
             "outputs": ["glycine"], "enzymes": [], "biological_state": "cytosol"}
        )
    return payload


def test_g9_a_process_quarantined_at_admission_caps_a_release_ready_pathway() -> None:
    """Base: a pathway that omits a process at admission is still ``release_ready`` (a bare ``pathway.pwml``).
    Tip: any omitted process in the frozen record caps it at ``review_required``."""

    control = _quarantine(_cap_toy(quarantined_reaction=False), None)
    assert control.quarantine_report["release"]["status"] == "release_ready"
    assert not any(r.startswith("omitted_from_pwml") for r in control.quarantine_report["release"]["reasons"])

    result = _quarantine(_cap_toy(quarantined_reaction=True), None)
    assert [row["state"] for row in result.admissions] == ["core_accepted", "core_accepted", QUARANTINED_UNMAPPED_ENTITY]
    assert result.ok
    _review_required(result)


def test_preservation_an_entity_only_removal_does_not_cap_the_release() -> None:
    """An unreferenced species row is removed (D-111) and nothing else: the release stays ``release_ready``."""

    payload = _cap_toy(quarantined_reaction=False)
    payload["entities"]["species"].append({"name": "Mus musculus", "taxonomy_id": "10090", "classification": "Eukaryote"})
    result = _quarantine(payload, None)
    assert [row["name"] for row in result.removed_entity_report["removed_entities"] if row["bucket"] == "species"] == [
        "Mus musculus"
    ]
    assert result.quarantine_report["release"]["status"] == "release_ready"


# ── Preservation ───────────────────────────────────────────────────────────


def test_preservation_a_whole_pathway_refusal_still_refuses_and_nothing_is_omitted() -> None:
    """runs_verify 2026-08-21_2239 PMC12180156: no reaction has defensible support (KI-022), a whole-pathway
    requirement. It names no row, so nothing is omitted and the refusal stands."""

    payload, context = _leg("runs_verify_2026-08-21_2239_PMC12180156")
    result = _quarantine(payload, context)
    assert _omitted(result) == [] and _removed(result) == []
    assert _refusals(result) == ["required:no_defensible_reaction_support"]


def test_preservation_omissions_that_buy_no_pwml_are_rolled_back() -> None:
    """The bare-enzyme leg with its pathway species stripped of every identity: the required-field gate
    names the LDH reaction AND the species, a whole-pathway requirement. Omitting LDH would still leave no
    PWML, so nothing is omitted and both refusals stand, exactly as at base."""

    payload, context = _leg("runs_verify_2026-08-21_2057_PMC12096016")
    for key in ("pathbank_species_id", "pw_species_id", "pathwhiz_id", "species_id", "taxonomy_id", "taxonomy-id"):
        payload["entities"]["species"][0].pop(key, None)
    result = _quarantine(payload, context)
    assert _omitted(result) == [] and _removed(result) == []
    refusals = _refusals(result)
    assert "required:reaction_enzyme_must_be_protein_complex" in refusals
    assert "required:species_missing_taxonomy" in refusals


def test_preservation_a_passing_leg_is_untouched() -> None:
    """ORCH-767 sour taste writes a PWML at base: the same graph, nothing omitted, the same release."""

    payload, context, recorded = _orch768_leg("ORCH-767_sour_taste_v2-strict_2026-10-24_0042")
    result = _quarantine(payload, context)
    assert _omitted(result) == [] and _removed(result) == []
    assert result.quarantine_report["resulting_payload_hash"] == recorded["resulting_payload_hash"]
    assert result.quarantine_report["release"]["status"] == recorded["release"]["status"]
    assert _refusals(result) == []


def test_preservation_research_mode_omits_nothing() -> None:
    payload, context = _leg("ORCH-755_sour_taste-strict_2026-10-21_0139")
    result = _quarantine(payload, context, mode="research")
    assert result.payload == payload
    assert _omitted(result) == [] and _removed(result) == []


def test_preservation_an_omission_that_would_leave_nothing_is_rolled_back() -> None:
    """Every reaction of this toy fails Stage 8 (no ``entity_type``) and the required-field gate (bare protein
    enzymes). Omitting them all would leave an empty graph, so nothing is omitted: the export's own gate
    refuses, exactly as at base."""

    from test_strict_quarantine import _glutathione_payload

    result = _quarantine(_glutathione_payload(), None)
    assert _omitted(result) == [] and _removed(result) == []
    assert [row["state"] for row in result.admissions] == ["core_accepted", "core_accepted"]
    assert result.ok
    assert _refusals(result)[0].startswith("stage8:")


# ── NEW acceptance test (no base failure claimed) ──────────────────────────


def test_new_release_cap() -> None:
    """``_cap_release_for_omissions`` is monotone: ``release_ready`` -> ``review_required`` only."""

    from t2pw.pipeline.strict_quarantine import OMITTED_FROM_PWML_REASON, _cap_release_for_omissions

    reason = f"{OMITTED_FROM_PWML_REASON}:2_processes+0_actors"
    ready = {"status": "release_ready", "strict_acceptance_eligible": True, "reasons": []}
    capped = _cap_release_for_omissions(ready, reason)
    assert capped == {"status": "review_required", "strict_acceptance_eligible": False, "reasons": [reason]}
    assert ready["status"] == "release_ready"  # not mutated
    assert _cap_release_for_omissions(ready, "") is ready
    blocked = {"status": "diagnostic_only", "strict_acceptance_eligible": False, "reasons": ["x"]}
    assert _cap_release_for_omissions(blocked, "r")["status"] == "diagnostic_only"
