"""ORCH-773 (D-114 section 2): a per-row export failure omits its rows before the freeze; the pathway exports.

**G9: A REGRESSION FIX OF OBSERVABLE BEHAVIOUR.** On each fixture leg below the base (``3ef3fa9c``) writes
no PWML for a pathway that is ``review_required`` because ONE row fails ONE per-row requirement: the strict
quarantine refuses (``unexportable_entity``), or the final pre-export gate, the Stage-8 contract, the
required-field gate or the IR refuses. Every name imported here exists at base with the same signature, so
the module imports there and every ``test_g9_*`` fails there on VALUES (``_refusals`` is not empty, nothing
is recorded as omitted). ``test_preservation_*`` pass at base and at tip: they fail only if the omission
reaches further than the gates. ``test_new_release_cap`` is a NEW acceptance test of a new function (it
fails at base on its import, which is inside the test) and claims no base failure as G9.

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
    """(bucket, name, reason) of every entity the per-row pass removed."""

    return [
        (row["bucket"], row["name"], row["reason"])
        for row in result.removed_entity_report["removed_entities"]
        if row.get("detail")
    ]


def _review_required_with_the_omission(result) -> None:
    release = result.quarantine_report["release"]
    assert release["status"] == "review_required"
    assert release["strict_acceptance_eligible"] is False
    assert any(r.startswith("per_row_omissions_removed_from_pwml:") for r in release["reasons"])
    assert any(r.startswith("per_row_omissions_removed_from_pwml:") for r in result.quarantine_report["review_reasons"])


def _surviving(result, bucket: str) -> List[Dict[str, Any]]:
    return result.payload["processes"].get(bucket) or []


# ── G9: the strict quarantine's own refusal ────────────────────────────────


def test_g9_unexportable_component_takes_its_complex_and_processes_not_the_pathway() -> None:
    """ORCH-755 enterobactin: EntB (no UniProt id, KI-017) is a component of the declared complex
    'isochorismatase'. Base: ``unexportable_entity:1``, no PWML."""

    payload, context = _leg("ORCH-755_enterobactin-strict_2026-10-20_2201")
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    reason = "protein_complex_component_unrepresentable:EntB:protein_missing_external_identity"
    assert _omitted(result) == [
        ("EntB isochorismatase reaction", QUARANTINED_UNMAPPED_ENTITY, reason, "actor", "enzymes", "isochorismatase"),
        ("2,3-DHB competitive inhibition of EntB", QUARANTINED_UNMAPPED_ENTITY, reason, "process", "entity_2", "isochorismatase"),
    ]
    assert _removed(result) == [
        ("protein_complexes", "isochorismatase", reason),
        ("proteins", "EntB", "protein_missing_external_identity"),
    ]
    _review_required_with_the_omission(result)


def test_g9_unexportable_row_kept_only_by_a_synonym_is_removed_alone() -> None:
    """ORCH-767 Wnt/Ca2+: 'phosphorylated CAMKII' (no identity, KI-018) survives only because it carries the
    synonym 'calcineurin B', which every process resolves to the real 'calcineurin B' row. Base: refusal.
    The next round then meets the Stage-8 contract: two transports with a non-canonical transporter."""

    payload, context = _leg("ORCH-767_wnt_ca-strict_2026-10-24_0314")
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    assert _removed(result) == [("proteins", "phosphorylated CAMKII", "protein_missing_external_identity")]
    insp3r = "inositol-1,4,5-trisphosphate receptor (InsP3R)"
    assert _omitted(result) == [
        ("IP3-induced Ca2+ release", QUARANTINED_BROKEN_REFERENCE, "actor_schema_not_canonical", "actor", "transporters", insp3r),
        ("SERCA pumps Ca2+ into ER", QUARANTINED_BROKEN_REFERENCE, "actor_schema_not_canonical", "actor", "transporters", insp3r),
    ]
    assert "calcineurin B component of calcineurin complex" in {row["name"] for row in _surviving(result, "interactions")}
    _review_required_with_the_omission(result)


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
    _review_required_with_the_omission(result)


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
    _review_required_with_the_omission(result)


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
    _review_required_with_the_omission(result)


# ── G9: the export's gates ─────────────────────────────────────────────────


def test_g9_stage8_non_canonical_transporter_omits_its_transport() -> None:
    """ORCH-755 sour taste (KI-023): three transports name OTOP1 in Stage-1 shape. Base: Stage-8 stop."""

    payload, context = _leg("ORCH-755_sour_taste-strict_2026-10-21_0139")
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    assert _omitted(result) == [
        (name, QUARANTINED_BROKEN_REFERENCE, "actor_schema_not_canonical", "actor", "transporters", "OTOP1")
        for name in ("OTOP1-mediated H+ transport", "H+ permeation through OTOP1", "OTOP1-mediated H+ permeation")
    ]
    _review_required_with_the_omission(result)


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
    _review_required_with_the_omission(result)


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
    _review_required_with_the_omission(result)


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
    assert not any(r.startswith("per_row_omissions") for r in result.quarantine_report["review_reasons"])
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

    from t2pw.pipeline.strict_quarantine import PER_ROW_OMISSION_REASON, _cap_release_for_omissions

    ready = {"status": "release_ready", "strict_acceptance_eligible": True, "reasons": []}
    capped = _cap_release_for_omissions(ready, f"{PER_ROW_OMISSION_REASON}:2")
    assert capped == {"status": "review_required", "strict_acceptance_eligible": False,
                      "reasons": [f"{PER_ROW_OMISSION_REASON}:2"]}
    assert ready["status"] == "release_ready"  # not mutated
    assert _cap_release_for_omissions(ready, "") is ready
    blocked = {"status": "diagnostic_only", "strict_acceptance_eligible": False, "reasons": ["x"]}
    assert _cap_release_for_omissions(blocked, "r")["status"] == "diagnostic_only"
