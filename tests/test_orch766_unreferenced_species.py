"""ORCH-766 / D-111 section 1: a species row nothing references is removed before the freeze.

**G9: A CORRECTION OF OBSERVABLE BEHAVIOUR.** Every name used here exists at the base
SHA ``d6849c9e`` with the same signature, so this module imports cleanly there. Every
``test_g9_*`` test fails there on VALUES: the unreferenced row is still in the
quarantined payload and the required-field gate still refuses on it. The
``test_preservation_*`` tests are labelled preservation: they pass at base and at tip
by design and fail only if the removal reaches a row it must not.

THE DEFECT. The 3-draw menaquinone leg (ORCH-759p, PMC12421875, 2026-10-22_2154)
reached ``release_ready`` with 19 reactions, then the required-field gate
(``ir.py`` SPECIES block) refused the export on ``/entities/species/2``, "L. lactis
subsp. lactis KF147": a pooled row that nothing in the pathway names and that NCBI
cannot resolve (``evidence/orch759p_species_diag/``). Closure already removes every
other kind of unreferenced row, but ``species`` was outside ``_GRAPH_ENTITY_BUCKETS``,
so this one row blocked a pathway nothing was wrong with (PRODUCT_CONTRACT: an
irrelevant entity must not be a terminal blocker). D-111 section 1 rules that such a
row is removed and recorded like the degree-zero removals; a referenced row never is,
and the gate is unchanged for every row that remains.

THE BYTE-IDENTITY PINS (``PINS``) are each payload's full quarantine output --
payload, ``removed_entity_report``, closure report and quarantine report -- hashed
at the base SHA by ``evidence/orch766/pins_at_base.py`` (log under
``evidence/orch766/``). At tip they must not move: a payload with no unreferenced
species row comes out of the quarantine byte for byte as it did before.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from t2pw.pipeline.strict_quarantine import admitted_payload_hash, quarantine_and_close
from t2pw.pwml.ir import validate_required_pwml_contract

from test_strict_quarantine_contract_alignment import _base

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "orch766"

#: sha256 of the leg's ``final_mapped.json`` (uncompressed); see the fixture README.
LEG_FINAL_MAPPED_SHA256 = "34a2b0e15a750df396c3178e3f663dce0e2be3938fec49cbd7e1bf2b9a953650"
#: The leg's own ``resulting_payload_hash``: ``final_mapped.json`` IS the quarantine output.
LEG_RESULTING_PAYLOAD_HASH = "sha256:47d5937e59d3da3d"
ABBREVIATED = "L. lactis subsp. lactis KF147"
RESOLVED = "Lactococcus lactis subsp. lactis KF147"
EXTRA = "Mus musculus"


def _leg():
    raw = gzip.decompress((FIXTURES / "menaquinone_PMC12421875_759p_final_mapped.json.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == LEG_FINAL_MAPPED_SHA256
    context = json.loads(
        (FIXTURES / "menaquinone_PMC12421875_759p_requested_context.json").read_text(encoding="utf-8")
    )
    return json.loads(raw.decode("utf-8")), context


def _gate(payload: Dict[str, Any], pathway_name: str = "Generated Pathway") -> Dict[str, Any]:
    """The required-field gate as ``run_pwml_export`` calls it (metadata defaults set)."""

    gate_payload = deepcopy(payload)
    metadata = gate_payload.setdefault("metadata", {})
    for key, value in (
        ("pathway_name", pathway_name),
        ("name", pathway_name),
        ("pathway_subject", "Metabolic"),
        ("subject", "Metabolic"),
    ):
        metadata.setdefault(key, value)
    return validate_required_pwml_contract(gate_payload, strict_db=True)


def _species_errors(report: Dict[str, Any]) -> List[tuple]:
    return [
        (e["code"], e.get("pointer"), e.get("species_name"))
        for e in report["errors"]
        if str(e.get("code", "")).startswith("species_")
    ]


def _names(payload: Dict[str, Any]) -> List[str]:
    return [row.get("name") for row in payload["entities"]["species"]]


def _digest(result) -> str:
    blob = json.dumps(
        {
            "payload": result.payload,
            "removed": result.removed_entity_report,
            "closure": result.closure_report,
            "quarantine": result.quarantine_report,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _species_removals(result) -> List[Dict[str, Any]]:
    return [
        row for row in result.removed_entity_report["removed_entities"] if row["bucket"] == "species"
    ]


# ── The leg ─────────────────────────────────────────────────────────────────


def test_g9_menaquinone_leg_drops_the_unreferenced_species_row_and_exports() -> None:
    payload, context = _leg()
    assert admitted_payload_hash(payload) == LEG_RESULTING_PAYLOAD_HASH
    # Base, on the leg's own payload: the gate refuses on the unreferenced row.
    assert _species_errors(_gate(payload, context["pathway_name"])) == [
        ("species_missing_taxonomy", "/entities/species/2", ABBREVIATED),
        ("species_missing_classification", "/entities/species/2", ABBREVIATED),
    ]

    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")

    # Exactly that row leaves; nothing else in the payload moves.
    expected = deepcopy(payload)
    del expected["entities"]["species"][2]
    assert result.payload == expected
    # Recorded like a degree-zero removal: removed_entity_report and the closure trace.
    record = {
        "bucket": "species",
        "name": ABBREVIATED,
        "reason": "unreferenced_after_quarantine",
        "had_external_identity": None,
    }
    assert _species_removals(result) == [{**record, "iteration": 1}]
    assert result.removed_entity_report["counts"]["entities"] == 1
    assert result.closure_report["iterations"][0]["removed_entities"] == [record]
    assert result.closure_report["converged"] is True
    # The verdict is the leg's own, and the gate now passes.
    assert result.ok and result.refusal_reasons == []
    assert result.quarantine_report["release"]["status"] == "release_ready"
    gate = _gate(result.payload, context["pathway_name"])
    assert _species_errors(gate) == []
    assert gate["ok"], gate["errors"]


def test_g9_tip_output_is_a_fixed_point() -> None:
    payload, context = _leg()
    first = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    second = quarantine_and_close(first.payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    assert ABBREVIATED not in _names(first.payload)
    assert second.payload == first.payload
    assert _species_removals(second) == []


def test_preservation_menaquinone_resolved_row_684738_is_kept() -> None:
    payload, context = _leg()
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    rows = {row["name"]: row for row in result.payload["entities"]["species"]}
    assert rows[RESOLVED]["taxonomy_id"] == "684738"
    assert rows[RESOLVED]["classification"] == "Prokaryote"
    assert {"Lactococcus lactis", RESOLVED, "Arabidopsis thaliana"} <= set(rows)
    men = [p for p in result.payload["entities"]["proteins"] if str(p.get("name", "")).startswith("Men")]
    assert len(men) == 7 and {p["species"] for p in men} == {RESOLVED}


def test_preservation_menaquinone_row_once_referenced_is_kept_and_still_fails_the_gate() -> None:
    """No gate is relaxed: the same row, given one reference, stays and is refused."""

    payload, context = _leg()
    payload["biological_states"][0]["species"] = ABBREVIATED
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    assert ABBREVIATED in _names(result.payload)
    assert _species_removals(result) == []
    assert ("species_missing_taxonomy", "/entities/species/2", ABBREVIATED) in _species_errors(
        _gate(result.payload, context["pathway_name"])
    )


def test_preservation_research_mode_applies_nothing() -> None:
    payload, context = _leg()
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="research")
    assert result.payload == payload
    assert result.removed_entity_report["removed_entities"] == []


# ── Synthetic payloads: every reference kind ────────────────────────────────


def _with_extra(row: Dict[str, Any] | None = None, *, first: bool = False) -> Dict[str, Any]:
    payload = _base()
    extra = row or {"name": EXTRA}
    if first:
        payload["entities"]["species"].insert(0, extra)
    else:
        payload["entities"]["species"].append(extra)
    return payload


def _rename_state(payload: Dict[str, Any], old: str, new: str) -> None:
    for state in payload["biological_states"]:
        if state["name"] == old:
            state["name"] = new
    for rows in payload["element_locations"].values():
        for row in rows:
            if row.get("biological_state") == old:
                row["biological_state"] = new
    for rows in payload["processes"].values():
        for row in rows:
            if row.get("biological_state") == old:
                row["biological_state"] = new


def _state_species(p):
    p["biological_states"][0]["species"] = EXTRA


def _state_organism(p):
    p["biological_states"][0].pop("species")
    p["biological_states"][0]["organism"] = EXTRA


def _state_name(p):
    # The IR names a state's species after the state when it carries none (ir.py,
    # "If the state is named after one component").
    p["biological_states"][0].pop("species")
    _rename_state(p, "cytosol", EXTRA)


def _protein(field):
    def wire(p):
        protein = p["entities"]["proteins"][0]
        protein.pop("species")
        if field == "species_ref.name":
            protein["species_ref"] = {"name": EXTRA}
        elif field == "mapping_meta.species":
            protein["mapping_meta"] = {"species": EXTRA}
        else:
            protein[field] = EXTRA

    return wire


def _complex(field):
    def wire(p):
        row = p["entities"]["protein_complexes"][0]
        row.pop("species")
        row[field] = EXTRA

    return wire


def _compound_species(p):
    p["entities"]["compounds"][0]["species"] = EXTRA


def _location_row(p):
    p["element_locations"]["compound_locations"][0]["species"] = EXTRA


def _metadata_organism(p):
    p["metadata"]["organism"] = EXTRA


def _process_field(p):
    p["processes"]["reactions"][0]["species"] = EXTRA


def _alias(p):
    p["entities"]["species"][1]["aliases"] = ["house mouse"]
    p["biological_states"][0]["species"] = "house mouse"


def _synonym(p):
    p["entities"]["species"][1]["synonyms"] = ["laboratory mouse"]
    p["entities"]["proteins"][0]["species"] = "laboratory mouse"


def _raw_name(p):
    p["entities"]["species"][1]["raw_name"] = "M. musculus"
    p["entities"]["proteins"][0]["species"] = "M. musculus"


def _spelling(p):
    # ir.resolve_component matches a state's species on ir._norm.
    p["biological_states"][0]["species"] = "MUS  musculus."


#: Name references: the referenced row carries no taxonomy id, so it must be KEPT
#: and must still FAIL ``species_missing_taxonomy``.
NAME_KINDS: Dict[str, Callable[[Dict[str, Any]], None]] = {
    "state_species": _state_species,
    "state_organism": _state_organism,
    "state_name": _state_name,
    "protein_species": _protein("species"),
    "protein_organism": _protein("organism"),
    "protein_species_ref_name": _protein("species_ref.name"),
    "protein_mapping_meta_species": _protein("mapping_meta.species"),
    "complex_species": _complex("species"),
    "complex_organism": _complex("organism"),
    "other_entity_species": _compound_species,
    "location_row_species": _location_row,
    "pathway_metadata_organism": _metadata_organism,
    "surviving_process_field": _process_field,
    "alias": _alias,
    "synonym": _synonym,
    "raw_name": _raw_name,
    "case_and_punctuation_variant": _spelling,
}


def _id_payload(kind: str) -> Dict[str, Any]:
    payload = _with_extra({"name": EXTRA, "taxonomy_id": "10090", "classification": "Eukaryote", "pathbank_species_id": 77})
    protein = payload["entities"]["proteins"][0]
    if kind == "state_taxonomy_id":
        payload["biological_states"][0].pop("species")
        payload["biological_states"][0]["taxonomy_id"] = "10090"
        return payload
    protein.pop("species")
    if kind == "protein_taxonomy_id":
        protein["taxonomy_id"] = "10090"
    elif kind == "protein_species_id":
        protein["species_id"] = 77
    elif kind == "protein_species_ref_ids":
        protein["species_ref"] = {"taxonomy_id": 10090, "pathbank_species_id": 77}
    elif kind == "species_id_as_float_text":
        # ir._to_int reads the row's "77.0" as PathBank id 77, the writer's key.
        payload["entities"]["species"][1]["pathbank_species_id"] = "77.0"
        protein["species_id"] = 77
    elif kind in ("species_id_non_integral_float", "species_id_non_integral_text"):
        # Repair round 1 (B1): ir._to_int TRUNCATES (int(float(text))), so a row's
        # 77.5 or "77.9" is IR pathwhiz_id 77, and the writer resolves species_id 77
        # to it.
        value = 77.5 if kind.endswith("_float") else "77.9"
        payload["entities"]["species"][1]["pathbank_species_id"] = value
        protein["species_id"] = 77
    return payload


ID_KINDS = (
    "protein_taxonomy_id",
    "protein_species_id",
    "protein_species_ref_ids",
    "state_taxonomy_id",
    "species_id_as_float_text",
    "species_id_non_integral_float",
    "species_id_non_integral_text",
)


def _kind_payload(kind: str) -> Dict[str, Any]:
    if kind in ID_KINDS:
        return _id_payload(kind)
    if kind == "create_default_name":
        payload = _with_extra({"name": "Narcissus sp. aff. pseudonarcissus"})
        payload["biological_states"][0]["species"] = "Narcissus aff. pseudonarcissus MK-2014"
        return payload
    if kind == "first_row":
        # The writer's pathway species, every complex's species and every
        # unresolved protein's species are the FIRST species row (writer.py).
        return _with_extra(first=True)
    if kind == "requested_organism":
        return _with_extra()
    payload = _with_extra()
    NAME_KINDS[kind](payload)
    return payload


ALL_KINDS = (*NAME_KINDS, *ID_KINDS, "create_default_name", "first_row", "requested_organism")

#: ``_digest`` of each payload's quarantine output (see module doc). Moved by ORCH-773 (merge rule 4): the
#: only bytes that differ from the ORCH-766 base-SHA pins are five version fields
#: (QUARANTINE_POLICY_VERSION 2026-07-31.1 -> 2026-10-25.1, so decision_inputs, decision_input_hash and
#: decision_id; removed_entity_report schema_version 1 -> 2). ``evidence/orch773/pins_delta.py`` strips those
#: five and gets identical digests at base and tip (G11 ORCH-773/24, /25); the old pins are in ORCH-773-RESULT.
#: Moved again by ORCH-780 (QUARANTINE_POLICY_VERSION -> 2026-10-26.1, the same three version-derived fields):
#: the normalized digests are identical at base 4a15ef42 and tip for all 28 (G11 ORCH-780/14, /15); old pins
#: in ORCH-780-RESULT.md.
# ORCH-797a moved all 28 pins deliberately: only the quarantine report's KEY TEXT moved (core terms in the D-120
# reference key); payload, removed, closure and the report minus key fields are byte-identical to base
# (evidence/orch797a_pinprobe.py, g11/ORCH-797a/14-, 15-). Old -> new values: ORCH-797a-RESULT.md § 5.
PINS: Dict[str, str] = {
    "base": "7ee1e24baf7e35b7ee4461627aafa40525830e5d5defa3bb54dcfdac1efd9591",
    "state_species": "42e7c8ca79c87aa740d3a91e54b6671039808ac8f0dd49cd41ea0bd387565648",
    "state_organism": "6c322b23d01268afc5d4776a6c2f4a91ae72b99c3743e38c2f0d71a54e31f6c6",
    "state_name": "9b8c582965ceacac4a7e139f17559577af019efcb3a89255a8a077cf3626b5e6",
    "protein_species": "97d31e50d9bfd4cb12fedd0843efa2227c8b2a8bb7f642e8f09436e7c0ddb41e",
    "protein_organism": "48eb53ef98941c6cbd7adc30e881c01fb92b25d329678e7f4cad5c684c8ad764",
    "protein_species_ref_name": "60070b71652ad13b7bd9d9f6fcaabbcbc2d531453a9d170be1fc722df2018018",
    "protein_mapping_meta_species": "c248f8aa5e035caecc005977c3b7b8474cb3c64ed4cb5610e28ad978a8b08e1d",
    "complex_species": "68af94d31ba9d31de1af886e92906aca12178235fd25de783c5650a319626c17",
    "complex_organism": "a07648c205cb8ecbde3f7c6b37311ac5361cc6551ef663f128990e2da65bb86f",
    "other_entity_species": "59848cca3ccaecaa47a2c72f45d691006e6059a4ce019d60120ccb307ceb479e",
    "location_row_species": "ab6b5819e73ea295d799d55ca685b800711d18912d8221a53759cab89e8abb61",
    "pathway_metadata_organism": "180ebca37067f3e6688874a5a2606e270d41ce859166753948f64d292b9dbfbc",
    "surviving_process_field": "158d8a41f6d0086d3b85fe358fda58f4306ed96b5c469e0b318d44e62620d24a",
    "alias": "afe2e047be567544fafc692bfa1e287bf45d018b105f836453fe97d012ffda73",
    "synonym": "c61100abc184f807987cd622a667e918cef39926afaff7ac41094c53dd865a9e",
    "raw_name": "50362983d670a2d6d6d4f770a7dd6fe0e965435a7a7d7304600e20eea89676aa",
    "case_and_punctuation_variant": "62c8ec9ca9cf2984c5dd55ed0ecf67b18d338f73a3bc29d882862fb6a3f4036c",
    "protein_taxonomy_id": "b41f47767ab7dfe916bdf471e6738e73e7006a11305d0a0c389cf7e17dc7dcf9",
    "protein_species_id": "88cd97ddc10ce5d634a0fb898b7c122ba345a81a6d4c0a2042ac7fd66c8887c2",
    "protein_species_ref_ids": "6a8b7b1aeccb1eb1867f5db27aa540b1f1e721f2bded06f95bbc38db6009a06a",
    "state_taxonomy_id": "8245a6733bfb71ed28511ae8c04bc7eaf360e331547914765052934b465aa516",
    "species_id_as_float_text": "cbaff37cecfa22be80df36ebd1794b115e2a61abd98fc4df373baa6a0ebd04c2",
    "species_id_non_integral_float": "8e036922090ab99803872689512d5a668b4dc10c3f9c08b55b9f81ac6ea33af9",
    "species_id_non_integral_text": "66653bde626002dd67dee7557a1e817abc542794826f0c9c4fe63b30655c651a",
    "create_default_name": "0fccf968036b093b7bb925f983c721d412cc2e4cc977fafa82f8a8720ac30e27",
    "first_row": "0b721eb01f56141a51e650337a4ac3045b790c5854385525bdc69eb45643b306",
    "requested_organism": "4f25a124db191559f97391801223a3437aeab2f1f5d90c9d66bbee2716519caf",
}


def _context(kind: str):
    return {"pathway_name": "Glutathione biosynthesis", "likely_organism": EXTRA} if kind == "requested_organism" else None


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_preservation_referenced_species_row_is_kept_byte_identical(kind: str) -> None:
    payload = _kind_payload(kind)
    result = quarantine_and_close(payload, strict_db=True, pathway_context=_context(kind), mode="pathwhiz")
    assert _species_removals(result) == []
    assert _names(result.payload) == _names(payload)
    assert _digest(result) == PINS[kind]


@pytest.mark.parametrize("kind", [k for k in ALL_KINDS if k not in ID_KINDS])
def test_preservation_referenced_row_without_taxonomy_still_fails_the_gate(kind: str) -> None:
    payload = _kind_payload(kind)
    result = quarantine_and_close(payload, strict_db=True, pathway_context=_context(kind), mode="pathwhiz")
    index = _names(result.payload).index(payload["entities"]["species"][0 if kind == "first_row" else 1]["name"])
    codes = {(code, pointer) for code, pointer, _name in _species_errors(_gate(result.payload))}
    assert ("species_missing_taxonomy", f"/entities/species/{index}") in codes


def test_preservation_payload_without_extra_species_is_byte_identical() -> None:
    result = quarantine_and_close(_base(), strict_db=True, mode="pathwhiz")
    assert _digest(result) == PINS["base"]


# ── Synthetic payloads: what is removed ─────────────────────────────────────


def test_g9_unreferenced_species_row_is_removed_and_the_gate_passes() -> None:
    payload = _with_extra()
    assert _species_errors(_gate(payload))  # base refuses on it
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    assert _names(result.payload) == ["Homo sapiens"]
    assert [(r["name"], r["reason"]) for r in _species_removals(result)] == [
        (EXTRA, "unreferenced_after_quarantine")
    ]
    assert _species_errors(_gate(result.payload)) == []
    assert result.ok


def test_g9_species_kept_only_by_a_removed_protein_goes_with_it() -> None:
    payload = _with_extra()
    payload["entities"]["proteins"].append(
        {"name": "orphan kinase", "species": EXTRA, "mapped_ids": {"uniprot": "Q00001"}}
    )
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    removed = [(r["bucket"], r["name"]) for r in result.removed_entity_report["removed_entities"]]
    assert ("proteins", "orphan kinase") in removed and ("species", EXTRA) in removed
    assert _names(result.payload) == ["Homo sapiens"]


def test_g9_an_id_collision_outside_a_species_field_does_not_keep_a_row() -> None:
    payload = _with_extra({"name": EXTRA, "pathbank_species_id": 4})
    payload["entities"]["compounds"][0]["pubchem_cid"] = "4"
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    assert _names(result.payload) == ["Homo sapiens"]


# ── Repair round 1: a recorded behaviour change ─────────────────────────────


def test_delta_r1_restored_autostate_takes_a_species_the_pathway_references() -> None:
    """DELTA, labelled: fails at base by design, passes at tip (reviewer note 1, round 0).

    When closure sweeps every state, the F-192 restore (``restore_autostates_if_required``,
    after the loop) gives the placeholder ``__auto_state__`` the species
    ``_select_default_species_name`` scores highest among the rows PRESENT. At base an
    unreferenced row with species-resolution confidence 0.99 wins it; at tip that row
    has already been removed by the species closure, so the placeholder takes the row
    the proteins reference. Recorded, not changed (ORCH-766-RESULT.md, repair round 1).
    """

    payload = _with_extra(
        {"name": EXTRA, "mapping_meta": {"species_resolution": {"confidence": 0.99}}}
    )
    # Nothing references the one state, so closure sweeps it and the restore fires.
    for reaction in payload["processes"]["reactions"]:
        reaction.pop("biological_state")
    payload["element_locations"]["compound_locations"] = []
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    states = {row["name"]: row for row in result.payload["biological_states"]}
    assert "cytosol" not in states
    assert states["__auto_state__"]["species"] == "Homo sapiens"  # base: "Mus musculus"
    assert _names(result.payload) == ["Homo sapiens"]
    assert [(r["name"], r["reason"]) for r in _species_removals(result)] == [
        (EXTRA, "unreferenced_after_quarantine")
    ]
