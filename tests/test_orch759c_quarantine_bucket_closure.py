"""ORCH-759c: a complex-component name keeps a PROTEIN row alive, not a same-named gene.

**G9: A REGRESSION FIX, NOT NEW WORK.** Every name used here exists at the base SHA
``5d1ea64c`` with the same signature, so this module imports cleanly there and every
``test_g9_*`` test fails there on VALUES (``ok`` False,
``refusal_reasons == ["entity_type_overlap:..."]``). ``test_g9_r1_leg_*`` (correction
round 1, N1) also fails at the round-0 tip ``7b30e564``. The ``test_preservation_*``
tests are labelled preservation: they pass at base and at tip by design and fail
only if the fix reaches further than it should. The ``test_delta_n2_*`` tests pin a
DELIBERATE behaviour change (round 1, N2): they fail at base, where no PWML was
written either, and pass at tip.

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

from t2pw.pipeline.process_normalizer import (
    _entity_name_norms,
    _normalize,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.strict_quarantine import quarantine_and_close

from test_strict_quarantine_contract_alignment import _base

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
#: ORCH-782 (KI-011) moved the leg pins below deliberately: a gene clashing with a
#: protein is now "<SYMBOL> gene", 18 gene-regulation rows are admitted, and only the
#: CAO and PORB genes are still orphans (ORCH-782-RESULT.md, baseline move).
#: ORCH-794 (D-119 § 3) moved them again: tRNAGlu, declared as compound and nucleic acid, keeps the nucleic
#: acid its name says, so "Glu-tRNAGlu reduction to GSA" is admitted (core 26 -> 27, broken-reference 10 -> 9);
#: the compound row and its location go as same_name_declared_as_another_kind:nucleic_acids (ORCH-794-RESULT.md).
ORCH782_COUNTS = {
    **BASE_COUNTS,
    "core_accepted": 27,
    "auxiliary_accepted": 47,
    "quarantined_broken_reference": 9,
}
ORCH782_ADMISSIONS_SHA256 = "53355ece7103ffd90afd3042a7b7589d4b719667bf2f28c30c476154c0e17387"
ORCH782_ORPHAN_GENES = ["CAO gene", "PORB gene"]
ORCH782_REMOVED = (23, "54084dfdd62116c5042768b532f515ce452503502488c4cd38e95c6319e9e45f")
ORCH782_REMOVED_LOCATIONS = (41, "eda2b5d1b050853b87968f60311cc838fa74d7b23d5d64e1ad571efd4cd9f907")
#: ``removed_locations`` at base, as ``bucket|name|reason|biological_state`` lines
#: (``evidence/orch759c/26-r1-probe-base.log``; identical at the round-0 tip, ``24``).
BASE_REMOVED_LOCATIONS_SHA256 = "66891e59028a1dc0fd2c0919d30e6f77571d612c7e2420de71329cb685bde7ff"
BASE_REMOVED_LOCATIONS_COUNT = 53
#: element_locations bucket -> (its name field, the entity buckets its rows locate).
#: Written out, not imported, so this module reads only names that exist at base.
LOCATES = {
    "compound_locations": ("compound", ("compounds",)),
    "protein_locations": ("protein", ("proteins", "protein_complexes")),
    "nucleic_acid_locations": ("nucleic_acid", ("nucleic_acids",)),
    "element_collection_locations": ("element_collection", ("element_collections",)),
}


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

    # The live decision, replayed: same admissions, same counts. Its decision input was
    # sha256:d2ef87f2ae5da361; ORCH-773's QUARANTINE_POLICY_VERSION bump moves it and nothing else, and
    # ORCH-780's (-> 2026-10-26.1) again: sha256:8b2be66a5cacc344 -> below.
    assert report["decision_input_hash"] == "sha256:6db09994c3616bb6"
    assert report["counts"] == ORCH782_COUNTS
    assert _digest(f"{row['pointer']}|{row['state']}" for row in result.admissions) == ORCH782_ADMISSIONS_SHA256

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
    orphans = [f"nucleic_acids|{name}|unreferenced_after_quarantine" for name in ORCH782_ORPHAN_GENES]
    assert sorted(line for line in removed if line in orphans) == orphans
    rest = [line for line in removed if line not in orphans]
    assert (len(rest), _digest(rest)) == ORCH782_REMOVED

    entities = result.payload["entities"]
    assert set(ORPHAN_GENES) <= {row["name"] for row in entities["proteins"]}
    assert not set(ORPHAN_GENES) & {row["name"] for row in entities["nucleic_acids"]}
    assert len(entities["nucleic_acids"]) == 21  # ORCH-794: + tRNAGlu
    processes = result.payload["processes"]
    assert (len(processes["reactions"]), len(processes["interactions"])) == (31, 43)  # ORCH-794: + Glu-tRNAGlu reduction

    # The strict final gate freeze_canonical_payload runs once quarantine is ok.
    details = run_strict_post_normalization_gates(
        deepcopy(result.payload), enforce_all_proteins_connected=True
    )
    # ORCH-782: a gene is its own element, so a gene-only regulation can be an island.
    assert details["connectivity"]["n_components"] == 2


def test_g9_r1_leg_no_location_row_moves_to_an_entity_of_another_kind() -> None:
    """Correction round 1, N1. Red at the round-0 tip ``7b30e564`` and at base.

    Round 0 pruned the four gene rows but kept their four ``nucleus state`` rows:
    ``_prune_locations`` checked one name list shared by every bucket, and the
    same-named PROTEIN kept each name in it. ``ir.py``'s ``resolve_entity`` then
    placed proteins CAO, HEMA1, GUN4 and PORB in the nucleus with no issue raised,
    which is an invented location. A location row now goes with its own-kind
    entity when closure removes it.

    The leg's only remaining cross-kind rows are two that base had too: complex
    locations filed under ``element_collection_locations``, where no element
    collection was ever declared (``evidence/orch759c/25-r1-ec-locations.log``). This
    lane leaves them exactly as base did; RESULT, correction round 1, says why.
    """

    payload, context = _leg()
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")

    removed = [
        f"{row['bucket']}|{row['name']}|{row['reason']}|{row['biological_state']}"
        for row in result.removed_entity_report["removed_locations"]
    ]
    new = [
        f"element_locations/nucleic_acid_locations|{name}|location_entity_removed|nucleus state"
        for name in ORCH782_ORPHAN_GENES
    ]
    assert sorted(line for line in removed if line in new) == new
    rest = [line for line in removed if line not in new]
    assert (len(rest), _digest(rest)) == ORCH782_REMOVED_LOCATIONS

    entities = result.payload["entities"]
    cross_kind = []
    for bucket, (field, owners) in LOCATES.items():
        own = {norm for owner in owners for norm in _entity_name_norms(entities.get(owner) or [])}
        for row in (result.payload.get("element_locations") or {}).get(bucket) or []:
            name = row.get(field) or row.get("entity")
            if _normalize(name) not in own:
                cross_kind.append((bucket, name, row.get("biological_state")))
    assert sorted(cross_kind) == [
        ("element_collection_locations", "Mg-Proto ME cyclase complex", "thylakoid membrane state"),
        ("element_collection_locations", "Mg-chelatase complex", "thylakoid membrane state"),
    ]
    assert not {name for _bucket, name, _state in cross_kind} & set(ORPHAN_GENES)

    assert result.ok
    assert result.quarantine_report["release"]["status"] == "review_required"
    details = run_strict_post_normalization_gates(
        deepcopy(result.payload), enforce_all_proteins_connected=True
    )
    assert details["connectivity"]["n_components"] == 2  # ORCH-782, see above


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
    assert ("nucleic_acids", "HEMA1 gene", "unreferenced_after_quarantine") in _removed(result)  # ORCH-782 name
    assert _names(result, "nucleic_acids") == []
    assert _names(result, "proteins") == ["HEMA1"]


def test_delta_orch782_a_referenced_gene_is_renamed_and_no_overlap_remains() -> None:
    """Was ``test_preservation_a_referenced_gene_is_kept_and_its_overlap_still_refuses``.

    DELIBERATE DELTA (ORCH-782; fails at 5e1edb3f, passes at tip): the gene is now named "HEMA1 gene", so this
    vehicle no longer has an overlap and exports. The overlap gate itself is unchanged
    and still pinned by ``test_strict_quarantine.py::test_an_entity_type_overlap_that_
    survives_closure_blocks_the_export`` and ``test_strict_quarantine_release_seam.py``.
    """

    result = quarantine_and_close(_payload(gene_synonym="HEMA1 gene"), strict_db=False)

    assert result.states()["HY5 activates HEMA1"] in {"core_accepted", "auxiliary_accepted"}
    assert _names(result, "nucleic_acids") == ["HEMA1 gene"]
    assert _names(result, "proteins") == ["HEMA1", "HY5"]
    assert result.quarantine_report["strict_invariants"]["entity_type_overlaps"] == []
    assert result.refusal_reasons == []
    assert result.ok


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


# ── Correction round 1: the other buckets (N4) and the revalidation delta (N2) ──
#
# All on ``test_strict_quarantine_contract_alignment._base`` (strict_db=True, real
# accessions): two reactions, each catalysed by a generated single-protein wrapper.

GS = "glutathione synthetase"


def _gs_wrapper(payload: dict) -> dict:
    return next(row for row in payload["entities"]["protein_complexes"] if row["name"] == f"{GS} complex")


def _record(result, name: str) -> tuple:
    row = next(row for row in result.admissions if row["name"] == name)
    return row["state"], row.get("reason") or ""


def _assert_both_reactions_survive_and_only(result, removed: list) -> None:
    assert result.quarantine_report["strict_invariants"]["entity_type_overlaps"] == []
    assert result.refusal_reasons == []
    assert result.ok
    assert _record(result, "gamma-glutamylcysteine synthesis")[0] == "core_accepted"
    assert _record(result, "glutathione synthesis")[0] == "core_accepted"
    assert _removed(result) == removed


def test_g9_r1_n4_unreferenced_compound_named_like_a_wrapped_protein_is_pruned() -> None:
    """N4 ``c3ok``. Base: refused, ``entity_type_overlap:1`` (compounds+proteins)."""

    payload = _base()
    payload["entities"]["compounds"].append({"name": GS})
    result = quarantine_and_close(payload, strict_db=True)

    # ORCH-794 (D-119 § 3): the same row goes one step earlier, as the kind GS's UniProt id does not back.
    _assert_both_reactions_survive_and_only(result, [("compounds", GS, "same_name_declared_as_another_kind:proteins")])


def test_g9_r1_n4_unreferenced_element_collection_named_like_a_wrapped_protein_is_pruned() -> None:
    """N4 ``c7``. Base: refused, ``entity_type_overlap:1`` (proteins+element_collections)."""

    payload = _base()
    payload["entities"]["element_collections"] = [{"name": GS}]
    result = quarantine_and_close(payload, strict_db=True)

    _assert_both_reactions_survive_and_only(
        result, [("element_collections", GS, "same_name_declared_as_another_kind:proteins")]  # ORCH-794, as above
    )


def _assert_glutathione_synthesis_stranded(result, reason: str) -> None:
    assert _record(result, "glutathione synthesis") == ("quarantined_disconnected", f"stranded_by_closure:{reason}")
    assert _record(result, "gamma-glutamylcysteine synthesis")[0] == "core_accepted"
    assert result.refusal_reasons == []
    assert result.ok


def test_delta_n2_compound_shadowing_a_wrapped_protein_without_accession() -> None:
    """DELIBERATE DELTA (round 1, N2, shape 1); it only quarantines more.

    Admission resolves the wrapper's component ``glutathione synthetase`` to the
    COMPOUND (compounds register first) and accepts the reaction. Base kept both rows
    and refused: ``entity_type_overlap:1`` + ``unexportable_entity:1``, no PWML. Tip
    prunes the unreferenced compound, revalidation then resolves the component to the
    protein, which has no accession, and the reaction is quarantined.
    """

    payload = _base()
    payload["entities"]["compounds"].append({"name": GS})
    next(row for row in payload["entities"]["proteins"] if row["name"] == GS).pop("mapped_ids")
    result = quarantine_and_close(payload, strict_db=True)

    # ORCH-794 (D-119 § 3): the compound goes before admission (GS is the wrapper's component, a protein-only
    # role), so admission refuses the reaction for the same component; closure stranded it before.
    assert _record(result, "glutathione synthesis") == (
        "quarantined_unmapped_entity", f"protein_complex_component_unrepresentable:{GS}:protein_missing_external_identity")
    assert _record(result, "gamma-glutamylcysteine synthesis")[0] == "core_accepted"
    assert result.refusal_reasons == [] and result.ok


def test_delta_n2_generated_wrapper_whose_component_names_a_complex() -> None:
    """DELIBERATE DELTA (round 1, N2, shape 2); it only quarantines more.

    Base kept the named complex alive by the component name and accepted the reaction.
    The exporter's contract then failed with ``component_protein_unresolved``
    (``evidence/orch759c/27-r1-probe-base2.log``), so no PWML was written. Tip prunes the
    complex, because a component names a protein, and the wrapper's component no
    longer resolves.
    """

    payload = _base()
    _gs_wrapper(payload)["components"] = [{"name": "GS holoenzyme", "stoichiometry": 1}]
    payload["entities"]["protein_complexes"].append(
        {
            "name": "GS holoenzyme",
            "class": "protein_complex",
            "species": "Homo sapiens",
            "components": [{"name": GS, "stoichiometry": 1}],
        }
    )
    result = quarantine_and_close(payload, strict_db=True)

    _assert_glutathione_synthesis_stranded(result, "protein_complex_component_unresolved:GS holoenzyme")


def test_delta_n2_wrapper_whose_component_names_only_a_gene() -> None:
    """DELIBERATE DELTA (round 1, N2, shape 3); it only quarantines more.

    Base kept the gene alive by the component name and accepted the reaction. The
    exporter's contract then failed with ``component_protein_unresolved`` (``27``), so
    no PWML was written. Tip prunes the gene and the component no longer resolves.
    """

    payload = _base()
    _gs_wrapper(payload)["components"] = [{"name": "GSS", "stoichiometry": 1}]
    payload["entities"]["nucleic_acids"] = [{"name": "GSS"}]
    result = quarantine_and_close(payload, strict_db=True)

    _assert_glutathione_synthesis_stranded(result, "protein_complex_component_unresolved:GSS")
