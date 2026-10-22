"""ORCH-759c: a complex-component name keeps a PROTEIN row alive, not a same-named gene.

**G9: A REGRESSION FIX, NOT NEW WORK.** Every name used here exists at the base SHA
``5d1ea64c`` with the same signature, so this module imports cleanly there and the
two ``test_g9_*`` tests fail there on VALUES (``ok`` False,
``refusal_reasons == ["entity_type_overlap:..."]``). The ``test_preservation_*``
tests are labelled preservation: they pass at base and at tip by design and fail
only if the fix reaches further than it should.

THE DEFECT. Closure kept a row in ANY bucket whose name met
``referenced | _complex_component_norms(...)``. The chlorophyll-strict leg
(PMC5130987, 2026-10-22_1231) declared the gene rows ``HEMA1``, ``CAO``, ``GUN4`` and
``PORB`` beside same-named proteins; mapping wrapped each protein as a
``single_protein_pathwhiz_wrapper`` complex (``HEMA1 complex`` -> ``HEMA1``).
Admission correctly quarantined every process naming the bare symbol
(``ambiguous_entity_type:nucleic_acids+proteins``), so nothing surviving referenced
the genes -- but the component name ``hema1`` kept the GENE row too.
``_entity_type_overlaps`` then found four overlaps and the run refused, suppressing a
correct ``review_required`` PWML (merge rule 7; ``product_contract_violation``).

A component names a protein: ``_component_name``, ``ir.py`` (``proteins_by_name``)
and ``writer.py`` (``_protein_complex_members``) resolve it against proteins only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from t2pw.pipeline.process_normalizer import run_strict_post_normalization_gates
from t2pw.pipeline.strict_quarantine import quarantine_and_close

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "orch759c"

#: sha256 of the leg's ``final_mapped.json`` (uncompressed), so the fixture is
#: provably that file byte for byte. See ``tests/fixtures/orch759c/README.md``.
LEG_FINAL_MAPPED_SHA256 = "20c61fd792a7a7e085fa0e2da409ef3d9f4eec400300ff265673337a86431526"

#: Base-code facts on that input (``evidence/orch759c/01-base-facts.log``), pinned so
#: the tip can be shown to differ from base by exactly the four gene rows.
BASE_ADMISSIONS_SHA256 = "ba2aaf41f34a16fe6c81631a9b16bf269de3d5b85191a9699b857c5944847ff4"
BASE_REMOVED_SHA256 = "458182341f2975feeee3dd3de78b7e4c2bc61d395981397ed1d2eb915c0011da"
BASE_REMOVED_COUNT = 37
BASE_COUNTS = {
    "core_accepted": 19,
    "auxiliary_accepted": 36,
    "quarantined_unmapped_entity": 14,
    "quarantined_out_of_scope": 0,
    "quarantined_weak_evidence": 0,
    "quarantined_broken_reference": 28,
    "quarantined_disconnected": 0,
}
ORPHAN_GENES = ["CAO", "GUN4", "HEMA1", "PORB"]


def _digest(lines) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def _leg():
    raw = gzip.decompress((FIXTURES / "chlorophyll_PMC5130987_final_mapped.json.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == LEG_FINAL_MAPPED_SHA256
    context = json.loads(
        (FIXTURES / "chlorophyll_PMC5130987_requested_context.json").read_text(encoding="utf-8")
    )
    return json.loads(raw.decode("utf-8")), context


def test_g9_chlorophyll_leg_releases_review_required_once_orphan_genes_close() -> None:
    payload, context = _leg()
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    report = result.quarantine_report

    # The live decision, replayed: same decision input, same admissions, same counts.
    assert report["decision_input_hash"] == "sha256:d2ef87f2ae5da361"
    assert report["counts"] == BASE_COUNTS
    assert _digest(f"{row['pointer']}|{row['state']}" for row in result.admissions) == BASE_ADMISSIONS_SHA256

    # Base: ok False, ["entity_type_overlap:4"], diagnostic_only.
    assert report["strict_invariants"]["entity_type_overlaps"] == []
    assert result.refusal_reasons == []
    assert result.ok
    assert report["release"]["status"] == "review_required"

    # Exactly the four orphan gene rows go, recorded; every base removal is unchanged.
    removed = [
        f"{row['bucket']}|{row['name']}|{row['reason']}"
        for row in result.removed_entity_report["removed_entities"]
    ]
    orphans = [f"nucleic_acids|{name}|unreferenced_after_quarantine" for name in ORPHAN_GENES]
    assert sorted(line for line in removed if line in orphans) == orphans
    rest = [line for line in removed if line not in orphans]
    assert (len(rest), _digest(rest)) == (BASE_REMOVED_COUNT, BASE_REMOVED_SHA256)

    entities = result.payload["entities"]
    assert set(ORPHAN_GENES) <= {row["name"] for row in entities["proteins"]}
    assert not set(ORPHAN_GENES) & {row["name"] for row in entities["nucleic_acids"]}
    assert len(entities["nucleic_acids"]) == 9
    processes = result.payload["processes"]
    assert (len(processes["reactions"]), len(processes["interactions"])) == (30, 25)

    # The strict final gate freeze_canonical_payload runs once quarantine is ok.
    details = run_strict_post_normalization_gates(
        deepcopy(result.payload), enforce_all_proteins_connected=True
    )
    assert details["connectivity"]["n_components"] == 1


# ── Synthetic, minimal: the same mechanism in eight rows ────────────────────
#
# Deliberately synthetic names on real symbols' shape; ``strict_db=False`` so no
# accession is needed or minted (identity is not the seam under test).


def _payload(*, gene_synonym: str = "") -> dict:
    """An enzyme wrapped as a single-protein complex, plus a same-named gene.

    The interaction names the gene by its bare symbol, so admission quarantines it
    as ambiguous and nothing surviving references the gene -- unless
    ``gene_synonym`` is given, in which case the interaction names the gene by that
    unambiguous synonym and survives.
    """

    gene: dict = {"name": "HEMA1"}
    if gene_synonym:
        gene["synonyms"] = [gene_synonym]
    return {
        "entities": {
            "species": [],
            "subcellular_locations": [],
            "compounds": [{"name": "A"}, {"name": "B"}],
            "proteins": [{"name": "HEMA1"}, {"name": "HY5"}],
            "protein_complexes": [
                {
                    "name": "HEMA1 complex",
                    "class": "protein_complex",
                    "components": [{"name": "HEMA1"}],
                    "generated": True,
                    "generation_reason": "single_protein_pathwhiz_wrapper",
                }
            ],
            "nucleic_acids": [gene],
        },
        "biological_states": [],
        "element_locations": {},
        "processes": {
            "reactions": [
                {"name": "A to B", "inputs": ["A"], "outputs": ["B"], "enzymes": [{"entity": "HEMA1 complex"}]}
            ],
            "transports": [],
            "interactions": [{"name": "HY5 activates HEMA1", "left": "HY5", "right": gene_synonym or "HEMA1"}],
        },
    }


def _removed(result) -> list:
    return sorted(
        (row["bucket"], row["name"], row["reason"])
        for row in result.removed_entity_report["removed_entities"]
    )


def _names(result, bucket: str) -> list:
    return sorted(row["name"] for row in result.payload["entities"][bucket])


def test_g9_component_name_no_longer_keeps_an_unreferenced_same_named_gene() -> None:
    result = quarantine_and_close(_payload(), strict_db=False)

    assert result.states()["HY5 activates HEMA1"] == "quarantined_broken_reference"
    assert result.states()["A to B"] in {"core_accepted", "auxiliary_accepted"}
    # Base: [{"name": "HEMA1", "buckets": "proteins+nucleic_acids"}] and a refusal.
    assert result.quarantine_report["strict_invariants"]["entity_type_overlaps"] == []
    assert result.refusal_reasons == []
    assert result.ok
    assert ("nucleic_acids", "HEMA1", "unreferenced_after_quarantine") in _removed(result)
    assert _names(result, "nucleic_acids") == []
    assert _names(result, "proteins") == ["HEMA1"]


def test_preservation_a_referenced_gene_is_kept_and_its_overlap_still_refuses() -> None:
    """PRESERVATION (passes at base and tip). No gate is relaxed (D-106)."""

    result = quarantine_and_close(_payload(gene_synonym="HEMA1 gene"), strict_db=False)

    assert result.states()["HY5 activates HEMA1"] in {"core_accepted", "auxiliary_accepted"}
    assert _names(result, "nucleic_acids") == ["HEMA1"]
    assert _names(result, "proteins") == ["HEMA1", "HY5"]
    assert result.quarantine_report["strict_invariants"]["entity_type_overlaps"] == [
        {"name": "HEMA1", "buckets": "proteins+nucleic_acids"}
    ]
    assert result.refusal_reasons == ["entity_type_overlap:1"]
    assert not result.ok


def _no_overlap_payload() -> dict:
    """No name shared across buckets: a two-subunit complex whose subunits nothing
    else references, a gene an interaction references, and one orphan per bucket."""

    return {
        "entities": {
            "species": [],
            "subcellular_locations": [],
            "compounds": [{"name": "A"}, {"name": "B"}, {"name": "orphan compound"}],
            "proteins": [{"name": "P1"}, {"name": "P2"}, {"name": "TF1"}, {"name": "orphan protein"}],
            "protein_complexes": [
                {"name": "X complex", "class": "protein_complex", "components": ["P1", {"protein": "P2"}]},
                {"name": "orphan complex", "class": "protein_complex", "components": ["P1"]},
            ],
            "nucleic_acids": [{"name": "geneY"}, {"name": "orphan gene"}],
        },
        "biological_states": [],
        "element_locations": {},
        "processes": {
            "reactions": [
                {"name": "A to B", "inputs": ["A"], "outputs": ["B"], "enzymes": [{"entity": "X complex"}]}
            ],
            "transports": [],
            "interactions": [{"name": "TF1 activates geneY", "left": "TF1", "right": "geneY"}],
        },
    }


def test_preservation_a_protein_named_only_by_a_component_is_kept() -> None:
    """PRESERVATION (passes at base and tip). Requirement 3: subunits of a surviving
    complex stay even at degree zero; both component row shapes count."""

    result = quarantine_and_close(_no_overlap_payload(), strict_db=False)

    assert {"P1", "P2"} <= set(_names(result, "proteins"))
    assert result.ok


def test_preservation_nothing_changes_when_no_names_overlap() -> None:
    """PRESERVATION (passes at base and tip): exactly the orphans go, nothing else."""

    result = quarantine_and_close(_no_overlap_payload(), strict_db=False)

    assert _removed(result) == [
        ("compounds", "orphan compound", "unreferenced_after_quarantine"),
        ("nucleic_acids", "orphan gene", "unreferenced_after_quarantine"),
        ("protein_complexes", "orphan complex", "degree_zero_after_quarantine"),
        ("proteins", "orphan protein", "degree_zero_after_quarantine"),
    ]
    assert _names(result, "compounds") == ["A", "B"]
    assert _names(result, "proteins") == ["P1", "P2", "TF1"]
    assert _names(result, "protein_complexes") == ["X complex"]
    assert _names(result, "nucleic_acids") == ["geneY"]
    assert result.quarantine_report["strict_invariants"]["entity_type_overlaps"] == []
    assert result.ok
