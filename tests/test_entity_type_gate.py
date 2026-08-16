"""Regression tests for the Stage-3 entity-TYPE gate.

THE DEFECT THESE PIN. Run 2026-07-28_0919, paper PMC13278307 ("An Overview of
Mobile Colistin Resistance (mcr) Genes in Gram-Negative Bacilli"), strict leg:
PASS in 2098s, 14 reactions, 0 gate errors, a 200,266-byte PWML -- and six of
its 35 entity rows are DNA filed as molecules:

    entities.compounds : 'pmrHFIJKLM operon'
    entities.proteins  : 'arnBCADTEF operon', 'pmrCAB operon', 'mcr genes',
                         'arnBCADTEF', 'pmrHFIJKLM'

The bucket picks the database (map_ids._DIRECTLY_MAPPED_ENTITY_BUCKETS), so the
operon in ``compounds`` was queried against PathBank compound tables and the five
in ``proteins`` against UniProt, which answered 'pmrHFIJKLM' with P03023
"Lactose operon repressor" and 'mcr genes' with P08235, the human
mineralocorticoid receptor. No validator in the tree asked what KIND of thing a
name denotes -- every one of them checks structure -- so the leg was certified
clean. These tests are the check that was missing.

Every name below is a literal string from that run's
strict/merged_payload.json or strict/final_mapped.json; the near-miss cases in
the truth table are the names that MUST NOT fire, chosen because each is a real
shape the rules could plausibly have swallowed.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# C-051 / D-015 (LOCKED): the exporter no longer resolves compound identity
# after the canonical freeze, so a raw extraction payload is taken through the
# pre-freeze stage that now does that work. helpers_prefreeze asserts the stage
# actually ruled on every compound row.
from helpers_prefreeze import prefrozen_when_compounded  # noqa: E402
from t2pw.pipeline.process_normalizer import (  # noqa: E402
    # NOTE: GateValidationError is deliberately NOT imported. Every gate this file
    # exercises goes through run_strict_post_normalization_gates, which RETURNS a
    # report of errors rather than raising -- the raise lives in the separate
    # hard-gate path at process_normalizer.py:4490. Importing the exception here
    # was dead weight (ruff F401 on 2026-07-28) and, worse, implied a raising
    # contract these tests do not assert.
    _new_report,
    enforce_entity_type_consistency,
    normalize_process_payload,
    nucleic_acid_name_verdict,
    run_strict_post_normalization_gates,
    validate_registry_references,
)
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir  # noqa: E402


# ---------------------------------------------------------------------------
# 1. The name-shape truth table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        # -- fires: explicit nucleic-acid noun in terminal position -----------
        ("pmrHFIJKLM operon", "gene_or_operon_noun"),
        ("arnBCADTEF operon", "gene_or_operon_noun"),
        ("pmrCAB operon", "gene_or_operon_noun"),
        ("mcr genes", "gene_or_operon_noun"),
        ("lac operon", "gene_or_operon_noun"),
        ("his gene cluster", "gene_or_operon_noun"),
        # The paper's own title shape: the noun is still terminal after ')'.
        ("colistin resistance (mcr) genes", "gene_or_operon_noun"),
        # -- fires: bacterial gene-cluster shorthand (advisory) ---------------
        ("arnBCADTEF", "gene_cluster_symbol"),
        ("pmrHFIJKLM", "gene_cluster_symbol"),
        ("trpEDCBA", "gene_cluster_symbol"),
        # -- must NOT fire ----------------------------------------------------
        # P03023 is the exact UniProt entry run 2026-07-28_0919 shipped as the
        # identity of pmrHFIJKLM. It is a protein, and a rule that swallowed it
        # would relocate real proteins into entities.nucleic_acids.
        ("Lactose operon repressor", ""),
        ("operon regulatory protein", ""),
        # map_ids._wrapper_complex_name produced this from 'mcr genes' in the
        # same run. It is a (wrong) protein complex, not DNA.
        ("mcr genes complex", ""),
        ("arnBCADTEF operon complex", ""),
        # Biochemical prefixes that look like three-letter gene roots.
        ("proBDNF", ""),
        ("proNGF", ""),
        # Short lowercase-prefixed metabolites.
        ("cAMP", ""),
        ("dATP", ""),
        ("mRNA", ""),
        # Ordinary members of this corpus.
        ("lipid A", ""),
        ("MCR-1", ""),
        ("PhoP", ""),
        ("phosphorylated PhoP", ""),
        ("acetyl-CoA", ""),
        ("18:1 CoA", ""),
        ("glucose-6-phosphate", ""),
        ("", ""),
    ],
)
def test_nucleic_acid_name_verdict_truth_table(name: str, expected: str) -> None:
    assert nucleic_acid_name_verdict(name) == expected


# ---------------------------------------------------------------------------
# 2. The real payload shape
# ---------------------------------------------------------------------------

def _pmc13278307_payload() -> dict:
    """A reduced copy of PMC13278307's strict merged_payload.json.

    Names, buckets, reaction wiring and the ``compound_locations`` row for the
    operon are all as they really were in run 2026-07-28_0919. Trimmed to the
    rows the type gate reasons about so the assertions below are readable.
    """

    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [
                {"name": "lipid A", "class": "compound", "provenance": "extracted"},
                {"name": "modified lipid A", "class": "compound", "provenance": "extracted"},
                # The operon, filed as a compound because it appeared as a
                # reaction input rather than in an enzymes[] slot.
                {
                    "name": "pmrHFIJKLM operon",
                    "rag_confidence": 0.97,
                    "mapped_ids": {"chebi": "CHEBI:99999"},
                    "chebi_id": "CHEBI:99999",
                },
            ],
            "proteins": [
                {"name": "MCR-1", "species": "Escherichia coli", "uniprot": "A0A0R6L508"},
                # Same DNA, other bucket, because it appeared in enzymes[].
                # P03023 is "Lactose operon repressor" and P08235 is the human
                # mineralocorticoid receptor: the two accessions run
                # 2026-07-28_0919 really did attach to these two names.
                {"name": "pmrHFIJKLM", "species": "Escherichia coli", "uniprot": "P03023"},
                {"name": "mcr genes", "species": "Escherichia coli", "uniprot": "P08235"},
                {"name": "arnBCADTEF operon", "species": "Escherichia coli", "uniprot": "P30843"},
            ],
            "protein_complexes": [],
        },
        "biological_states": [
            {"name": "cyto_state", "species": "Escherichia coli", "subcellular_location": "cytosol"}
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": "lipid A", "biological_state": "cyto_state"},
                {"compound": "pmrHFIJKLM operon", "biological_state": "cyto_state"},
            ],
            "protein_locations": [
                {"protein": "MCR-1", "biological_state": "cyto_state"},
                {"protein": "pmrHFIJKLM", "biological_state": "cyto_state"},
                {"protein": "mcr genes", "biological_state": "cyto_state"},
                {"protein": "arnBCADTEF operon", "biological_state": "cyto_state"},
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "modification of lipid A",
                    "inputs": ["pmrHFIJKLM operon"],
                    "outputs": ["modified lipid A"],
                    "biological_state": "cyto_state",
                    "enzymes": [
                        {"entity": "pmrHFIJKLM", "entity_type": "protein"},
                        {"entity": "mcr genes", "entity_type": "protein"},
                        {"entity": "arnBCADTEF operon", "entity_type": "protein"},
                    ],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


def _names(payload: dict, bucket: str) -> list:
    rows = (payload.get("entities") or {}).get(bucket) or []
    return [str(row.get("name") or "") for row in rows if isinstance(row, dict)]


def _error_codes(validation: dict) -> list:
    return sorted((e["code"], e["pointer"]) for e in validation["errors"])


def _normalized_without_gate(payload: dict) -> dict:
    """Normalize a payload with the type gate stubbed out, for before/after diffs."""

    from t2pw.pipeline import process_normalizer as module

    original = module.enforce_entity_type_consistency
    module.enforce_entity_type_consistency = lambda data, **_kwargs: data
    try:
        normalized, _ = module.normalize_process_payload(deepcopy(payload))
    finally:
        module.enforce_entity_type_consistency = original
    return normalized


def test_unreferenced_operon_leaves_compounds_for_nucleic_acids() -> None:
    """'pmrHFIJKLM operon' is DNA and nothing catalyses with it: relocate it."""

    payload = _pmc13278307_payload()
    report = _new_report()

    enforce_entity_type_consistency(payload, report=report)

    assert "pmrHFIJKLM operon" not in _names(payload, "compounds")
    assert _names(payload, "nucleic_acids") == ["pmrHFIJKLM operon"]
    assert report["summary"]["nucleic_acid_named_entities_relocated"] == 1

    relocated = payload["entities"]["nucleic_acids"][0]
    assert relocated["class"] == "nucleic_acid"
    assert relocated["entity_type_gate"] == {
        "declared_bucket": "compounds",
        "name_denotes": "nucleic_acid",
        "rule": "gene_or_operon_noun",
        "action": "relocated_to_nucleic_acids",
        "stripped_identifier_fields": ["chebi_id", "mapped_ids.chebi"],
    }
    # The row keeps its provenance; only compound-database identity is dropped.
    assert relocated["rag_confidence"] == 0.97


def test_relocation_drops_identifiers_the_compound_mapper_wrote() -> None:
    """A ChEBI id on a DNA operon came from querying the wrong database.

    This is the identifier-falsification class 92e1192 fixed at the resolver,
    applied at the relocation: whatever the compound mapper attached before the
    type was known must not ride along into entities.nucleic_acids.
    """

    payload = _pmc13278307_payload()
    enforce_entity_type_consistency(payload, report=_new_report())

    relocated = payload["entities"]["nucleic_acids"][0]
    assert "chebi_id" not in relocated
    assert relocated["mapped_ids"] == {}


def test_relocation_moves_the_element_location_row_too() -> None:
    """A relocated entity whose location row stays behind is worse than no fix.

    pwml/ir.py builds locations from the location list, so a nucleic acid whose
    only row lives under ``compound_locations`` would be exported as a compound
    location pointing at a row that is no longer in entities.compounds.
    ensure_autostates had already written that row for 'pmrHFIJKLM operon' in run
    2026-07-28_0919, so this is not hypothetical.
    """

    payload = _pmc13278307_payload()
    report = _new_report()

    enforce_entity_type_consistency(payload, report=report)

    locations = payload["element_locations"]
    assert [row["compound"] for row in locations["compound_locations"]] == ["lipid A"]
    assert locations["nucleic_acid_locations"] == [
        {"biological_state": "cyto_state", "nucleic_acid": "pmrHFIJKLM operon"}
    ]
    action = next(a for a in report["actions"] if a["type"] == "entity_relocated_to_nucleic_acids")
    assert action["element_locations_moved"] == 1


def test_catalyst_named_operons_are_flagged_but_never_moved() -> None:
    """Moving a name out of the actor registry would MANUFACTURE a gate error.

    'mcr genes' and 'arnBCADTEF operon' are enzymes of the reaction, and
    run_strict_post_normalization_gates checks every actor against
    protein_registry_norms -- which is compounds-free. Relocating them would turn
    a correction into "Unknown protein/modifier reference: mcr genes". It is also
    the defensible reading: an operon named as a catalyst is a claim about its
    protein products.
    """

    payload = _pmc13278307_payload()
    report = _new_report()

    enforce_entity_type_consistency(payload, report=report)

    assert "mcr genes" in _names(payload, "proteins")
    assert "arnBCADTEF operon" in _names(payload, "proteins")
    assert "pmrHFIJKLM" in _names(payload, "proteins")
    assert report["summary"]["entity_type_mismatches_flagged"] == 3

    flagged = {
        str(row["name"]): row["entity_type_gate"]
        for row in payload["entities"]["proteins"]
        if "entity_type_gate" in row
    }
    assert set(flagged) == {"mcr genes", "arnBCADTEF operon", "pmrHFIJKLM"}
    assert flagged["mcr genes"]["action"] == "kept_in_place"
    assert flagged["mcr genes"]["rule"] == "gene_or_operon_noun"
    assert flagged["mcr genes"]["pinned_by_actor_reference"] is True
    assert "enzyme/modifier/transporter" in flagged["mcr genes"]["reason"]
    # The stamp records the rule and the pin separately, so a reviewer can tell
    # "its name says DNA and only the actor registry keeps it here" from
    # "the rule that fired was only a shape heuristic".
    assert flagged["pmrHFIJKLM"]["rule"] == "gene_cluster_symbol"
    assert flagged["pmrHFIJKLM"]["pinned_by_actor_reference"] is True
    # MCR-1 is a real protein and must be left completely alone.
    mcr1 = next(row for row in payload["entities"]["proteins"] if row["name"] == "MCR-1")
    assert "entity_type_gate" not in mcr1


def test_gene_cluster_shape_alone_never_relocates() -> None:
    """The shape heuristic flags; only the explicit noun is allowed to move.

    'arnBCADTEF' matches by shape, not by an unambiguous nucleic-acid noun, so
    even with nothing referencing it the row stays where it is and only reports.
    """

    payload = _pmc13278307_payload()
    payload["processes"]["reactions"][0]["inputs"] = ["lipid A"]
    payload["processes"]["reactions"][0]["enzymes"] = []
    payload["entities"]["compounds"] = [
        {"name": "lipid A"},
        {"name": "modified lipid A"},
    ]
    payload["entities"]["proteins"] = [
        {"name": "arnBCADTEF", "species": "Escherichia coli", "uniprot": "P30843"}
    ]
    report = _new_report()

    enforce_entity_type_consistency(payload, report=report)

    assert _names(payload, "proteins") == ["arnBCADTEF"]
    assert "nucleic_acids" not in payload["entities"]
    assert report["summary"]["nucleic_acid_named_entities_relocated"] == 0
    assert report["summary"]["entity_type_mismatches_flagged"] == 1
    stamp = payload["entities"]["proteins"][0]["entity_type_gate"]
    assert stamp["pinned_by_actor_reference"] is False
    assert "advisory" in stamp["reason"]


def test_pass_is_idempotent() -> None:
    """normalize_process_payload runs a second time inside pwml/writer.py:2650.

    A non-idempotent pass would double every counter and every action entry of
    the report the reviewer reads.
    """

    payload = _pmc13278307_payload()
    first = _new_report()
    enforce_entity_type_consistency(payload, report=first)
    second = _new_report()
    enforce_entity_type_consistency(payload, report=second)

    assert second["summary"]["nucleic_acid_named_entities_relocated"] == 0
    assert second["summary"]["entity_type_mismatches_flagged"] == 0
    assert [a for a in second["actions"] if a["type"].startswith("entity_")] == []
    assert _names(payload, "nucleic_acids") == ["pmrHFIJKLM operon"]


def test_pass_never_raises_on_malformed_payloads() -> None:
    """The gate must never be the reason a run dies. It only ever annotates."""

    for payload in (
        {},
        {"entities": None, "processes": None},
        {"entities": {"compounds": "not-a-list"}, "processes": {"reactions": [None, 7]}},
        {"entities": {"compounds": [None, {"name": ""}, {"name": "lac operon"}]}},
    ):
        enforce_entity_type_consistency(payload, report=_new_report())


# ---------------------------------------------------------------------------
# 3. The registry gap the relocation would otherwise have exposed
# ---------------------------------------------------------------------------

def test_registry_accepts_a_nucleic_acid_reaction_participant() -> None:
    """``nucleic_acids`` belongs in the registry and never was.

    pwml/ir.py:1394 has always listed ``nucleic_acid`` as a legal
    ``reaction_member``, but validate_registry_references only knew about
    compounds/proteins/protein_complexes. The gap was invisible while nothing
    used the bucket; relocating 'pmrHFIJKLM operon' -- a reaction input -- would
    have reported "/processes/reactions/0/inputs/0 unknown entity".
    """

    payload = _pmc13278307_payload()
    enforce_entity_type_consistency(payload, report=_new_report())

    validate_registry_references(payload)  # must not raise


def test_registry_still_rejects_a_genuinely_unknown_participant() -> None:
    payload = _pmc13278307_payload()
    payload["processes"]["reactions"][0]["inputs"].append("entity that exists nowhere")

    with pytest.raises(ValueError, match="unknown entity"):
        validate_registry_references(payload)


def test_strict_gates_report_no_new_errors_after_relocation() -> None:
    """The correction must not cost a single gate error, in either direction."""

    baseline = _pmc13278307_payload()
    before = run_strict_post_normalization_gates(baseline, report=_new_report())
    assert before["errors"] == []

    payload = _pmc13278307_payload()
    report = _new_report()
    enforce_entity_type_consistency(payload, report=report)
    after = run_strict_post_normalization_gates(payload, report=report)
    assert after["errors"] == []


# ---------------------------------------------------------------------------
# 4. Wiring and export
# ---------------------------------------------------------------------------

def test_normalize_process_payload_runs_the_gate() -> None:
    checkpoints: list = []
    payload = _pmc13278307_payload()

    normalized, report = normalize_process_payload(
        payload,
        on_checkpoint=lambda name, _payload, _report: checkpoints.append(name),
    )

    assert "enforce_entity_type_consistency" in checkpoints
    # It must run before the gate that would otherwise certify the wrong type.
    assert checkpoints.index("enforce_entity_type_consistency") < checkpoints.index(
        "run_strict_post_normalization_gates"
    )
    assert report["gate"]["ok"] is True
    assert report["summary"]["nucleic_acid_named_entities_relocated"] == 1
    assert _names(normalized, "nucleic_acids") == ["pmrHFIJKLM operon"]
    assert "pmrHFIJKLM operon" not in _names(normalized, "compounds")


def test_relocated_operon_still_exports_through_the_pwml_ir() -> None:
    """End-to-end proof that the new bucket is a place the exporter can reach.

    A relocation that made the row unexportable would be a worse defect than the
    one it fixes, so this asserts the operon comes out the far end as a
    nucleic_acid reaction member with a nucleic_acid location, and that the IR
    raises no unresolved-reference error for it.
    """

    payload = _pmc13278307_payload()
    normalized, _ = normalize_process_payload(deepcopy(payload))

    ir, ir_report = build_pwml_ir(prefrozen_when_compounded(normalized), strict_db=False)

    na_rows = ir["entities"]["nucleic_acids"]
    assert [row["name"] for row in na_rows] == ["pmrHFIJKLM operon"]
    na_key = na_rows[0]["key"]

    reaction = ir["processes"]["reactions"][0]
    left = [m for m in reaction["left"] if m.get("entity_key") == na_key]
    assert left, reaction["left"]
    assert left[0]["entity_type"] == "nucleic_acid"

    assert any(
        loc["entity_type"] == "nucleic_acid" and loc["entity_key"] == na_key
        for loc in ir["locations"]
    )
    unresolved = ir_report["unresolved"]["entity_references"]
    assert [ref for ref in unresolved if "operon" in str(ref.get("name"))] == []

    # And it costs nothing: the IR validation errors are byte-identical to the
    # ones this fixture already produced with the gate disabled. (They are all
    # "reaction_enzyme_must_be_protein_complex", an artefact of the fixture
    # naming bare proteins as enzymes -- map_ids wraps those into single-protein
    # complexes in the real pipeline, a step this reduced payload skips.)
    baseline_ir, _ = build_pwml_ir(prefrozen_when_compounded(_normalized_without_gate(payload)), strict_db=False)
    assert _error_codes(validate_pwml_ir(ir)) == _error_codes(validate_pwml_ir(baseline_ir))
