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
# D-W24 3 / L5 moved all 28 pins: QUARANTINE_POLICY_VERSION -> 2026-11-15.1 only. With policy_version,
# decision_inputs, decision_input_hash and decision_id stripped the digests are identical at base 111dea4e and
# tip for all 28 (corr-assess/lanes/L5scripts/pins_delta.py; old -> new in lanes/L5.json).
# L5 review F7 (D-W24 3): all 28 re-pinned for quarantine_report schema 6 -> 7 (+ enzyme_wrappers). With the
# report normalised back (schema_version 6, enzyme_wrappers removed) every digest equals its previous pin
# (corr-assess/lanes/L5cprobe/probe_766.json).
# L8 (D-W24 7b) moved all pins: QUARANTINE_POLICY_VERSION -> 2026-11-16.1 only. With policy_version,
# decision_inputs, decision_input_hash and decision_id stripped the digests are identical at base 5c40e9f0 and
# tip (corr-assess/lanes/L8probe/pins_basewt.json, pins_corr-L8.json; old -> new in lanes/L8.json).
PINS: Dict[str, str] = {
    "base": "bd0dd8bc2f34261e4ff83baa3f0cadba6ea8fd8251300d1a98cb7f068a78fbfe",
    "state_species": "8f689d1b8c882a2a8a98bf81b080be5a09fc4b43976949ed9a24e23faccf2530",
    "state_organism": "2ae719576f39316b4d9a16bd7b62a61167777b6110442cca4345c37686a14c12",
    "state_name": "36507b5f00fbb070111821e3d1431fc96fd9f4fc067802e714027b2efc3027ef",
    "protein_species": "77b8139b094d0df88269465e4623ebdb00517cd802f8db1b2391b12e5db5f85b",
    "protein_organism": "f0f89092fb78a68e49765bba892a46f096ef7b14c5917568ace84a1eef5f41ca",
    "protein_species_ref_name": "f2aa3bbb2c64ccfa4037063909842e4b44350d137b150006d9665f2f5b936333",
    "protein_mapping_meta_species": "e4ed950607c083de5f5ff57f6c1a6034ef141c82a2f7b821d0e76fb563d597c4",
    "complex_species": "76387fbcac7813fec176efb1e42e2ab8d008246b19d10bde23368d3ce317d3e0",
    "complex_organism": "269907cbe9d3c24d05497dc2d511dd2a563686ac20fa338145ac9768de60991e",
    "other_entity_species": "96e57fb796f65188a9fc5cb80efdce14a606f2f6af975e6fb0aeccc565cea896",
    "location_row_species": "54d45dc3984d4b58c81744875d83bd81035d2f6efcdbf020dcdb297f80f50c9a",
    "pathway_metadata_organism": "68e0f60492bfec604701dde4a5725f5d386d81bede2bbc47c96f609c25b9a25c",
    "surviving_process_field": "0b90e3ebd5bd4c35d95d0cb270121c4f751b167aaf15a9f50c77094e52349700",
    "alias": "561fa8765e66dd1e94ff806b0fb0fec3768c1f98b6b12371e533761798a5471a",
    "synonym": "01fc073d8f2a9a40540bce3a7d6536531310e7cd7a70ee516b5b8b3f879b5428",
    "raw_name": "85b07da59fbd031af67f3a689cc954312979670a60d88a29e0f31cdf5bca00bf",
    "case_and_punctuation_variant": "4ded60a48cddfdbd2910cb74d4085fa62669a971d8e1d2ae0dd372ce25032c7e",
    "protein_taxonomy_id": "20d76dbee278a6640b024ef478cfd3027918a0aa713783212687add28224f3e3",
    "protein_species_id": "383b83295f82de9e2ee86d5d03098fd56128ac5a18887d5608359b7dbe81c46a",
    "protein_species_ref_ids": "2a2f2f9122d528ec3ffd2af5d1171fa2e29864182479fded4db2fb67a44362d7",
    "state_taxonomy_id": "9b8c69237fb93023d841418144345dae8276adc1421515cf96060d8b4290977d",
    "species_id_as_float_text": "3e2af945cb203629b39506e5d82858eefd1d18103bde159ac66f7ec859f0cc2f",
    "species_id_non_integral_float": "8b78541cf4b425edff4b5c9d5ea31bf920dc6863eccd3639b577be91b584051a",
    "species_id_non_integral_text": "36988696d7eecc4af4bb1ea1ab02b943ced00ebb40481de0ac2adec9e0999289",
    "create_default_name": "94aced50ac219c607d4ffc91eb16bf3b99bbf50bfd95b9b25361a6d9331eaf27",
    "first_row": "632931b1227f666ec16f3ee747680aa7fe899f8faf10ddeac91e9cecd10f7afb",
    "requested_organism": "f447ecfd7072cfcaaf51cbf5ab89b35ee2a3a16680dd1884cb2db1e3681f7903",
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
