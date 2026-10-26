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
PINS: Dict[str, str] = {
    "base": "32f3a34e079148f70808ce6d7fb33df6e6480d15aecb6d08c41fd66a4c7eed1b",
    "state_species": "59c95f8350b5cd6a4cb423fb887192cfbe487391b695ea6ab4dc8a50d9647613",
    "state_organism": "7bcc8be85e7ba7684777147309fad447db2de5a5aa8919a78d4dc3325e6f1f70",
    "state_name": "6ed999a521ec69e13009e2be64b1efb2185d7e7269489006ecfb30831999e054",
    "protein_species": "0c49a4c54b1606539d26678508774ff1b3c043d64ad78848f7a30ef440ed358a",
    "protein_organism": "4369288701c8c61ef50ded62ea3baa3de1b8edb9bb5efeffea591ad28c241f4e",
    "protein_species_ref_name": "403562c2b097bbfc8bcb3282661bcc1501e47fdf98eac89f20554511a86694f9",
    "protein_mapping_meta_species": "b242c789b8a250ffa13e779e0a94329bbccfa943cacfc1fb37338eb5042f6902",
    "complex_species": "06bca968108862db2a2f0b22bdb48db2851f4434433d8ecf274089050e11c452",
    "complex_organism": "78ca9bcc0d6bc07178fa5d297ae492d945aab381635bdaf0c32cb08e43241549",
    "other_entity_species": "45fdda53937884e6cd04b5362ff81400072746827610999728a721a60076f6b5",
    "location_row_species": "ddae74b65ee9a429ec323f9940fa45595f87530759f9aab84bdcaa9f4f33908b",
    "pathway_metadata_organism": "4396f5fb751358b4c76ee7c0e0f48e4e744bf1a3597a66dd746e0f7cf7279c68",
    "surviving_process_field": "d38f6f140d6a3ee49f9053aba6b222aee1f2f12c83d8b1346163681367828460",
    "alias": "266c611e6bf5151c3788b08774757016d73231d21d04a10d7f402415a117fa60",
    "synonym": "85b1f3ec0b42ef805589495f4f8ffdda8771aede07ce6c0ecec7ddeb0042c419",
    "raw_name": "e09541ee1e977c5f365398258a0b4e3d88ca3d13c77d0cbe02b4c9cbe1ec549f",
    "case_and_punctuation_variant": "04e9132d1a1bad5c9f71dff1baaf0d6338f8a865c5e3b1091347f2978ca0bd44",
    "protein_taxonomy_id": "65df837b8b64a3d34011ccefa770fe8132a0be162279307b638bd1802f8cbe7e",
    "protein_species_id": "90d7763ccb739ebd43d73198a4acda632a4aada3c06e6d890faa8657abe89b4f",
    "protein_species_ref_ids": "0022b4494c4a6f4922ea9e5f24a32beec50ca5160860bd022c199c4f1a17f431",
    "state_taxonomy_id": "64d40b43afb29bfa813b558facd0f8bd45c32d4c0f77575cab4a726af2e46c19",
    "species_id_as_float_text": "70eaa1f6f37eae6b7a820323c2ec8f1cf7ff76bfa3e6ee95aa50050d9c5420b0",
    "species_id_non_integral_float": "22a120807c87d803434cb56554332574ad6ace8181e445ee0c29c74f219a30c3",
    "species_id_non_integral_text": "c3ca99976a23bef32ed1d147e348b3884e88ade72c4a7bcb5f1942608acb261f",
    "create_default_name": "0ca5c0c3e75f63455ac3aa7e23b6b12c3407bc504ecbd9fcb374e270a2170cc8",
    "first_row": "22a5965ac1c5a503231a477963ff6eec1c6b198e199c77a580167f5675194efa",
    "requested_organism": "f7dc73d4b23db3e81893881cd65a1861942c6ed5bb1710c081b2d843922101e3",
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
