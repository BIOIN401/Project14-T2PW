"""D-001: one hash meant two things; these are the two things (C-013).

Every case asserts BOTH directions where both are meaningful. "lineage did not
move the graph hash" alone is also true of a hash that ignores lineage entirely,
and "lineage moved the payload hash" alone is also true of a single hash that was
merely renamed. Only the pair proves the split.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline import canonical_hash as ch  # noqa: E402
from t2pw.pipeline import gate_reports as gr  # noqa: E402

_STRICT = ROOT / "runs_verify" / "2026-08-04_1647" / "papers" / "PMC12856317" / "strict"
_BINDING = "cccf95c83f4cc054a2c312c5a4ba27f77762f747eccb00e7f8678d856324c52e"


def _payload() -> dict:
    """Every dimension the graph hash must cover, plus the record it must ignore."""
    return {
        "entities": {
            "species": [{"name": "Homo sapiens", "taxonomy_id": "9606"}],
            "proteins": [{"name": "ALAS2", "class": "protein", "uniprot": "P22557",
                          "mapped_ids": {"uniprot": "P22557"}, "confidence": 1.0,
                          "provenance": "extracted", "lineage": [{"stage": "map"}]}],
            "protein_complexes": [{"name": "ALAS2 complex",
                                   "components": [{"name": "ALAS2", "stoichiometry": 1}]}],
            "compounds": [{"name": "glycine", "chebi_id": "CHEBI:15428"}],
        },
        "biological_states": [{"name": "matrix state", "species": "Homo sapiens",
                               "subcellular_location": "mitochondrial matrix"}],
        "element_locations": {"protein_locations": [{"protein": "ALAS2", "evidence": "text",
                                                     "biological_state": "matrix state"}]},
        "processes": {"reactions": [{
            "name": "ALA synthesis", "inputs": ["glycine", "succinyl-CoA"],
            "outputs": ["aminolevulinic acid"], "reversible": False, "modifiers": [],
            "enzymes": [{"entity": "ALAS2 complex", "role": "catalyst"}],
            "biological_state": "matrix state", "evidence": "text",
            "lineage": [{"stage": "stage1"}, {"stage": "audit"}],
        }]},
    }


def _edited(edit) -> tuple:
    base = _payload()
    moved = deepcopy(base)
    edit(moved)
    return base, moved


def _artifacts(report: dict, canonical: dict) -> dict:
    return {gr.ARTIFACT_SET_VERSION_KEY: gr.ARTIFACT_SET_VERSION,
            gr.FINAL_GATE_REPORT_KEY: report, gr.CANONICAL_PAYLOAD_KEY: canonical}


def _final(payload: dict, **kw) -> dict:
    return gr.stamp_report({"stage": "final", "ok": True, "errors": []},
                           phase=gr.PHASE_FINAL_PRE_EXPORT, payload=payload, **kw)


#: The evidence record moving. None of it is biology.
_LINEAGE = {
    "added_at_top_level": lambda p: p.update(lineage=[{"stage": "rag"}]),
    "added_to_an_entity": lambda p: p["entities"]["compounds"][0].update(lineage=[{"s": 1}]),
    "edited_on_a_reaction": lambda p: p["processes"]["reactions"][0]["lineage"][0].update(stage="rag"),
    "reordered_on_a_reaction": lambda p: p["processes"]["reactions"][0]["lineage"].reverse(),
}
#: Fields the allowlist does not name, at every depth a future field could arrive.
_UNNAMED = {
    "a_new_top_level_section": lambda p: p.update(weather={"sky": "blue"}),
    "a_new_entity_field": lambda p: p["entities"]["proteins"][0].update(invented=7),
    "a_new_entity_bucket": lambda p: p["entities"].update(invented=[{"name": "X"}]),
    "a_new_mapped_id": lambda p: p["entities"]["proteins"][0]["mapped_ids"].update(invented="X"),
    "a_new_reaction_field": lambda p: p["processes"]["reactions"][0].update(invented=7),
}
#: PRODUCT_CONTRACT section 5, one dimension per case.
_BIOLOGY = {
    "an_accession": lambda p: p["entities"]["proteins"][0]["mapped_ids"].update(uniprot="P00001"),
    "a_compartment": lambda p: p["biological_states"][0].update(subcellular_location="cytosol"),
    "a_complex_component": lambda p: p["entities"]["protein_complexes"][0]["components"].append(
        {"name": "ALAS1"}),
    "direction": lambda p: p["processes"]["reactions"][0].update(reversible=True),
    "stoichiometry": lambda p: p["entities"]["protein_complexes"][0]["components"][0].update(
        stoichiometry=2),
}


@pytest.mark.parametrize("case", sorted(_LINEAGE))
def test_lineage_moves_the_payload_hash_and_never_the_graph_hash(case: str) -> None:
    base, moved = _edited(_LINEAGE[case])
    assert ch.canonical_graph_sha256(moved) == ch.canonical_graph_sha256(base)
    assert ch.canonical_payload_sha256(moved) != ch.canonical_payload_sha256(base)


@pytest.mark.parametrize("case", sorted(_UNNAMED))
def test_a_field_the_allowlist_does_not_name_stays_out_of_the_graph_hash(case: str) -> None:
    """The honesty test. A denylist projection admits every one of these."""
    base, moved = _edited(_UNNAMED[case])
    assert ch.canonical_graph_sha256(moved) == ch.canonical_graph_sha256(base)
    assert ch.canonical_payload_sha256(moved) != ch.canonical_payload_sha256(base)


@pytest.mark.parametrize("case", sorted(_BIOLOGY))
def test_every_biological_dimension_moves_the_graph_hash(case: str) -> None:
    base, moved = _edited(_BIOLOGY[case])
    assert ch.canonical_graph_sha256(moved) != ch.canonical_graph_sha256(base)
    assert ch.canonical_payload_sha256(moved) != ch.canonical_payload_sha256(base)


def test_a_payload_already_carrying_hash_keys_hashes_identically() -> None:
    """Self-exclusion, both projections: a hash is never an input to itself."""
    base, moved = _edited(lambda p: p.update(dict.fromkeys(ch.HASH_AND_STAMP_FIELDS, "x")))
    assert ch.canonical_graph_sha256(moved) == ch.canonical_graph_sha256(base)
    assert ch.canonical_payload_sha256(moved) == ch.canonical_payload_sha256(base)


def test_schema_one_artifacts_still_verify_unchanged() -> None:
    """Nothing opts into schema 2 by accident, and schema 1 keeps its one reason."""
    payload = _payload()
    report = _final(payload)
    assert ch.HASH_SCHEMA_VERSION_KEY not in report
    assert gr.gate_verdict(_artifacts(report, payload)).failed is False
    _, moved = _edited(_BIOLOGY["direction"])
    assert gr.gate_verdict(_artifacts(report, moved)).reason == gr.REASON_PAYLOAD_MISMATCH


def test_the_committed_pmc12856317_binding_still_reproduces() -> None:
    payload = json.loads((_STRICT / "final_mapped.json").read_text(encoding="utf-8"))
    report = json.loads((_STRICT / "final_stage3_gate_report.json").read_text(encoding="utf-8"))
    assert gr.payload_sha256(payload) == report[gr.PAYLOAD_SHA256_KEY] == _BINDING
    assert ch.canonical_payload_sha256(payload) == _BINDING


def test_stamping_schema_two_writes_all_three_keys_and_mutates_no_caller_input() -> None:
    """The stamp is a copy. The caller's report is a slice other code still reads,
    and the payload is the object about to be serialized -- stamping either in
    place would make the artifact disagree with the hash taken of it."""
    report, payload = {"stage": "final", "ok": True, "errors": []}, _payload()
    before, before_payload = deepcopy(report), deepcopy(payload)
    stamped = gr.stamp_report(report, phase=gr.PHASE_FINAL_PRE_EXPORT, payload=payload,
                              hash_schema=ch.HASH_SCHEMA_VERSION)
    assert (report, payload) == (before, before_payload)
    assert {gr.PAYLOAD_SHA256_KEY, ch.CANONICAL_GRAPH_SHA256_KEY,
            ch.CANONICAL_PAYLOAD_SHA256_KEY, ch.HASH_SCHEMA_VERSION_KEY} <= set(stamped)


def test_the_two_schema_two_mismatches_are_distinct_and_neither_reaches_the_other() -> None:
    payload = _payload()
    report = _final(payload, hash_schema=ch.HASH_SCHEMA_VERSION)
    assert gr.gate_verdict(_artifacts(report, payload)).failed is False
    _, biology = _edited(_BIOLOGY["a_compartment"])
    _, evidence = _edited(_LINEAGE["added_at_top_level"])
    assert gr.gate_verdict(_artifacts(report, biology)).reason == gr.REASON_CANONICAL_GRAPH_MISMATCH
    assert gr.gate_verdict(_artifacts(report, evidence)).reason == (
        gr.REASON_CANONICAL_PAYLOAD_MISMATCH_GRAPH_INTACT)
