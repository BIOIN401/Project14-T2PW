"""The one authoritative protein export policy, end to end.

Three outcomes are allowed for a protein, and exactly one applies to any row:

A. **Verified real mapping** -- the identifier passed all six rungs of
   :func:`t2pw.mapping.map_ids.verify_real_protein_identity` (entity type,
   species/taxon, name/alias/gene, identifier resolution, minimum score,
   margin over rivals).
B. **Unknown-backed functional complex** -- a reaction enzyme, catalytic
   modifier or transporter with a usable functional name and direct role
   evidence but no verified identity. The functional name is preserved on a
   generated ``protein_complex`` whose only component is PathBank's Unknown
   protein 9659, and the row says ``identity_status="placeholder"``.
C. **Unsupported or unused** -- removed/quarantined before the strict gates
   (unused), or the process claim itself quarantined (referenced but with no
   evidence for the role it claims).

What must never happen: an identifier chosen by list order, a placeholder
reported as a real UniProt mapping, a compound promoted to a protein, or a
reaction actor pointing at the literal name ``Unknown``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from t2pw.mapping.map_ids import map_payload, verify_real_protein_identity
from t2pw.pipeline.entity_identity import (
    PATHBANK_UNKNOWN_PROTEIN_ID,
    compound_moiety_verdict,
    identity_status,
    is_pathbank_unknown_protein,
    placeholder_claims_real_identity,
)
from t2pw.pipeline.process_normalizer import (
    GateValidationError,
    normalize_process_payload,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.stage_contracts import validate_post_normalization, validate_post_remap
from t2pw.pwml.ir import build_pwml_ir, validate_required_pwml_contract
from t2pw.pwml.validate import discover_structure_signature
from t2pw.pwml.writer import DeterministicPwmlBuilder


ROOT = Path(__file__).resolve().parents[1]
ORGANISM = "Camellia sinensis"


# ── payload builders ────────────────────────────────────────────────────────


def _payload(
    *,
    proteins: Optional[List[Dict[str, Any]]] = None,
    compounds: Optional[List[Dict[str, Any]]] = None,
    reactions: Optional[List[Dict[str, Any]]] = None,
    transports: Optional[List[Dict[str, Any]]] = None,
    protein_complexes: Optional[List[Dict[str, Any]]] = None,
    protein_locations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "pathway_name": "Caffeine degradation",
            "pathway_subject": "Metabolic",
            "organism": ORGANISM,
            "width": 3200,
            "height": 1400,
        },
        "entities": {
            "species": [{"name": ORGANISM, "taxonomy_id": "4442", "pathbank_species_id": 99}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": compounds if compounds is not None else [
                {"name": "caffeine"},
                {"name": "theobromine"},
            ],
            "proteins": proteins if proteins is not None else [],
            "protein_complexes": protein_complexes if protein_complexes is not None else [],
        },
        "biological_states": [
            {"name": "leaf cytosol", "species": ORGANISM, "subcellular_location": "cytosol"}
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": "caffeine", "biological_state": "leaf cytosol"},
                {"compound": "theobromine", "biological_state": "leaf cytosol"},
            ],
            "protein_locations": protein_locations if protein_locations is not None else [],
        },
        "processes": {
            "reactions": reactions if reactions is not None else [],
            "transports": transports if transports is not None else [],
            "interactions": [],
        },
    }


def _enzyme_reaction(actor: Dict[str, Any], *, name: str = "caffeine demethylation") -> Dict[str, Any]:
    return {
        "name": name,
        "inputs": ["caffeine"],
        "outputs": ["theobromine"],
        "biological_state": "leaf cytosol",
        "enzymes": [actor],
    }


def _protein(name: str, **extra: Any) -> Dict[str, Any]:
    row = {"name": name, "species": ORGANISM, "organism": ORGANISM}
    row.update(extra)
    return row


# ── resolver stubs ──────────────────────────────────────────────────────────


def _unmapped_result(reason: str = "no_match") -> Dict[str, Any]:
    return {
        "status": "unmapped",
        "reason": reason,
        "provider": "UniProt",
        "source": "api",
        "confidence": 0.0,
        "candidates": [],
        "resolution": {"status": "unresolved", "issue": reason, "order_step": "api_uniprot"},
    }


def _strong_match_result(accession: str = "Q9TEST") -> Dict[str, Any]:
    """An unambiguous hit: exact name, right organism, top score, no rival."""

    return {
        "status": "mapped",
        "provider": "UniProt",
        "source": "api",
        "confidence": 0.99,
        "chosen_rule": "reviewed_exact",
        "resolved_name": "N-methyl nucleosidase",
        "mapped_ids": {"uniprot": accession},
        "candidates": [
            {
                "accession": accession,
                "protein_name": "N-methyl nucleosidase",
                "organism": ORGANISM,
                "score": 0.99,
            }
        ],
        "resolution": {"status": "matched", "order_step": "api_uniprot"},
    }


def _ambiguous_result() -> Dict[str, Any]:
    """Two same-named orthologues, one of them in the wrong organism.

    The list order is what the removed ``ambiguous_first_candidate`` rule read:
    the *first* candidate here is the Bacillus one, so order alone would ship an
    organism the pathway never mentions.
    """

    return {
        "status": "unmapped",
        "reason": "ambiguous",
        "provider": "UniProt",
        "source": "api",
        "confidence": 0.0,
        "candidates": [
            {
                "accession": "O34362",
                "uniprot": "O34362",
                "protein_name": "N-methyl nucleosidase",
                "organism": "Bacillus subtilis",
                "score": 0.9,
            },
            {
                "accession": "P17109",
                "uniprot": "P17109",
                "protein_name": "N-methyl nucleosidase",
                "organism": "Escherichia coli",
                "score": 0.9,
            },
        ],
        "resolution": {"status": "ambiguous", "issue": "api_ambiguous", "order_step": "api_uniprot"},
    }


def _map(
    payload: Dict[str, Any],
    tmp_path: Path,
    *,
    protein_result: Any,
    allow_complex_wrapper_creation: bool = True,
) -> Dict[str, Any]:
    """Run Stage 6 offline: no network, no DB, resolver answers are supplied."""

    if callable(protein_result):
        protein_patch = {"side_effect": protein_result}
    else:
        protein_patch = {"return_value": protein_result}
    with (
        patch(
            "t2pw.mapping.map_ids.hydrate_species_references",
            return_value={"hydrated": 0, "matched": 0, "novel": 0},
        ),
        patch("t2pw.mapping.map_ids._map_protein_with_strategy", **protein_patch),
        patch("t2pw.mapping.map_ids.map_protein_uniprot", return_value=_unmapped_result()),
        patch("t2pw.mapping.map_ids._map_compound_with_strategy", return_value=_unmapped_result()),
        patch("t2pw.mapping.map_ids._map_complex_with_strategy", return_value=_unmapped_result()),
    ):
        return map_payload(
            payload,
            cache_path=tmp_path / "mapping-cache.json",
            id_source="api",
            use_cache=False,
            allow_complex_wrapper_creation=allow_complex_wrapper_creation,
        )


def _names(rows: Any) -> List[str]:
    return [str(row.get("name")) for row in rows if isinstance(row, dict)]


def _policy(result: Dict[str, Any]) -> Dict[str, Any]:
    return result["report"]["protein_export_policy"]["summary"]


# ── 1. a strong real match stays a real mapping ─────────────────────────────


def test_strong_real_match_remains_a_real_mapping(tmp_path: Path) -> None:
    payload = _payload(
        proteins=[_protein("N-methyl nucleosidase")],
        reactions=[
            _enzyme_reaction(
                {
                    "entity": "N-methyl nucleosidase",
                    "entity_type": "protein",
                    "role": "catalyst",
                }
            )
        ],
    )

    result = _map(payload, tmp_path, protein_result=_strong_match_result())
    mapped = result["payload"]

    protein = next(
        row for row in mapped["entities"]["proteins"] if row["name"] == "N-methyl nucleosidase"
    )
    assert protein["mapped_ids"]["uniprot"] == "Q9TEST"
    assert protein["mapping_meta"]["identity_status"] == "verified"
    assert identity_status(protein) == "verified"
    assert protein["mapping_meta"]["identity_verdict"]["verified"] is True
    assert protein["mapping_meta"]["identity_verdict"]["checks"] == {
        "identifier_resolution": "ok",
        "candidate_evidence": "ok",
        "entity_type": "ok",
        "species": "ok",
        "name": "keep",
        "score": "ok",
        "margin": "no_competing_candidate",
    }
    assert protein["mapping_meta"]["identity_verdict"]["species_compatibility"] == "ok"
    assert _policy(result)["verified_real_proteins"] == 1
    assert _policy(result)["unknown_backed_functional_complexes"] == 0
    assert _policy(result)["bare_unresolved_proteins_remaining"] == 0
    assert all(row["name"] != "Unknown" for row in mapped["entities"]["proteins"])


def test_every_rung_of_the_ladder_can_reject_on_its_own() -> None:
    """Each check is load-bearing: flip one input, the verdict flips with it."""

    base = dict(
        candidates=[
            {
                "accession": "Q9TEST",
                "protein_name": "N-methyl nucleosidase",
                "organism": ORGANISM,
                "score": 0.99,
            }
        ],
        mapped_ids={"uniprot": "Q9TEST"},
        organism=ORGANISM,
        source="api",
    )
    assert verify_real_protein_identity("N-methyl nucleosidase", **base)["verified"] is True

    # identifier resolution: the sentinel is not a real accession
    sentinel = dict(base, mapped_ids={"uniprot": "Unknown"})
    assert verify_real_protein_identity("N-methyl nucleosidase", **sentinel)["reason"] == (
        "no_real_identifier"
    )

    # entity type: the entity name belongs to the compound rules
    coa = dict(
        base,
        candidates=[{"accession": "Q9TEST", "protein_name": "coenzyme A", "organism": ORGANISM, "score": 0.99}],
    )
    assert verify_real_protein_identity("coenzyme A", **coa)["reason"] == "entity_type_incompatible"

    # species: same name, different organism
    wrong_species = dict(
        base,
        candidates=[
            {
                "accession": "Q9TEST",
                "protein_name": "N-methyl nucleosidase",
                "organism": "Homo sapiens",
                "score": 0.99,
            }
        ],
    )
    assert verify_real_protein_identity("N-methyl nucleosidase", **wrong_species)["reason"] == (
        "species_mismatch"
    )

    # name: nothing in common with what the accession names
    wrong_name = dict(
        base,
        candidates=[
            {"accession": "Q9TEST", "protein_name": "Mineralocorticoid receptor", "organism": ORGANISM, "score": 0.99}
        ],
    )
    assert verify_real_protein_identity("N-methyl nucleosidase", **wrong_name)["reason"] == (
        "implausible_name_match"
    )

    # score: a scored candidate below the floor
    weak = dict(
        base,
        candidates=[
            {"accession": "Q9TEST", "protein_name": "N-methyl nucleosidase", "organism": ORGANISM, "score": 0.2}
        ],
    )
    assert verify_real_protein_identity("N-methyl nucleosidase", **weak)["reason"] == (
        "score_below_minimum"
    )

    # margin: an equally good rival makes the winner a coin flip
    tie = dict(
        base,
        candidates=[
            {"accession": "Q9TEST", "protein_name": "N-methyl nucleosidase", "organism": ORGANISM, "score": 0.9},
            {"accession": "Q9OTHER", "protein_name": "N-methyl nucleosidase", "organism": ORGANISM, "score": 0.9},
        ],
    )
    verdict = verify_real_protein_identity("N-methyl nucleosidase", **tie)
    assert verdict["reason"] == "ambiguous_insufficient_margin"
    assert verdict["competing_accessions"] == ["q9other"]


# ── 1b. the ladder fails closed: missing evidence is not a pass ────────────


def test_an_accession_no_candidate_describes_is_not_verified() -> None:
    """The regression the fail-closed rule exists for.

    P08235 is the human mineralocorticoid receptor and MenD is a bacterial
    synthase; nothing in this call says so, because nothing in this call says
    anything. Under the old rule the empty candidate list made every rung
    unjudgeable and every unjudgeable rung a pass.
    """

    verdict = verify_real_protein_identity(
        "MenD",
        candidates=[],
        mapped_ids={"uniprot": "P08235"},
        organism="Escherichia coli",
        result={},
    )

    assert verdict["verified"] is False
    assert verdict["reason"] == "identity_evidence_missing"
    assert verdict["checks"]["candidate_evidence"] == (
        "no_candidate_describes_the_shipped_identifier"
    )


def test_a_candidate_with_no_organism_is_not_verified() -> None:
    """Species unknown is missing evidence, not a pass."""

    verdict = verify_real_protein_identity(
        "MenD",
        candidates=[{"accession": "P17109", "protein_name": "MenD", "score": 0.99}],
        mapped_ids={"uniprot": "P17109"},
        organism="Escherichia coli",
        result={},
    )

    assert verdict["verified"] is False
    assert verdict["reason"] == "identity_evidence_missing"
    assert verdict["checks"]["species"] == "unknown"


def test_a_candidate_with_no_name_evidence_is_not_verified() -> None:
    """A name-gate ``skip`` means nothing to judge on, which is not a pass."""

    verdict = verify_real_protein_identity(
        "MenD",
        candidates=[{"accession": "P17109", "organism": "Escherichia coli", "score": 0.99}],
        mapped_ids={"uniprot": "P17109"},
        organism="Escherichia coli",
        result={},
    )

    assert verdict["verified"] is False
    assert verdict["reason"] == "identity_evidence_missing"
    assert verdict["checks"]["name"].startswith("skip:")


def test_an_unscored_non_pathbank_candidate_is_not_verified() -> None:
    verdict = verify_real_protein_identity(
        "MenD",
        candidates=[
            {"accession": "P17109", "protein_name": "MenD", "organism": "Escherichia coli"}
        ],
        mapped_ids={"uniprot": "P17109"},
        organism="Escherichia coli",
        result={},
    )

    assert verdict["verified"] is False
    assert verdict["checks"]["score"] == "unscored_non_pathbank_candidate"


def test_an_unscored_pathbank_row_may_skip_only_the_score() -> None:
    """PathBank supplies no score. It supplies a gene symbol, and that is judged."""

    verdict = verify_real_protein_identity(
        "MenD",
        candidates=[
            {
                "pathbank_protein_id": 4471,
                "uniprot": "P17109",
                "protein_name": "2-succinyl-5-enolpyruvyl-6-hydroxy-3-cyclohexene-1-carboxylate synthase",
                "gene_name": "menD",
                "organism": "Escherichia coli",
            }
        ],
        mapped_ids={"uniprot": "P17109"},
        organism="Escherichia coli",
        result={},
    )

    assert verdict["verified"] is True
    assert verdict["checks"]["score"] == "skipped_unscored_pathbank_row"
    assert verdict["name_gate"]["reason"] == "exact_symbol_identity"


def test_a_wrong_pathbank_name_in_the_right_species_is_not_verified() -> None:
    """Curated provenance is not identity evidence. The name still has to hold."""

    verdict = verify_real_protein_identity(
        "MenD",
        candidates=[
            {
                "pathbank_protein_id": 123,
                "uniprot": "P03023",
                "protein_name": "Lactose operon repressor",
                "organism": "Escherichia coli",
                "score": 0.01,
            }
        ],
        mapped_ids={"uniprot": "P03023"},
        organism="Escherichia coli",
        result={},
    )

    assert verdict["verified"] is False
    assert verdict["reason"] == "implausible_name_match"


def test_an_unverifiable_but_legitimate_actor_still_ships_as_a_placeholder(
    tmp_path: Path,
) -> None:
    """Fail-closed must not mean fail-deleted: the enzyme keeps its name."""

    evidence_free_match = {
        "status": "mapped",
        "provider": "PathBankDB",
        "source": "db",
        "confidence": 1.0,
        "chosen_rule": "direct_id_match:uniprot_id",
        "mapped_ids": {"uniprot": "P03023", "pathbank_protein_id": "123"},
        "pathbank_protein_id": 123,
        "candidates": [
            {
                "pathbank_protein_id": 123,
                "uniprot": "P03023",
                "protein_name": "Lactose operon repressor",
                "organism": "Camellia sinensis",
                "score": 0.01,
            }
        ],
        "resolution": {"status": "matched", "order_step": "uniprot"},
    }
    payload = _payload(
        proteins=[_protein("MenD")],
        reactions=[
            _enzyme_reaction({"entity": "MenD", "entity_type": "protein", "role": "catalyst"})
        ],
    )

    result = _map(payload, tmp_path, protein_result=evidence_free_match)
    mapped = result["payload"]

    assert "P03023" not in str(mapped["entities"])
    assert _names(mapped["entities"]["protein_complexes"]) == ["MenD"]
    assert mapped["entities"]["protein_complexes"][0]["mapping_meta"]["identity_status"] == (
        "placeholder"
    )
    assert _policy(result)["verified_real_proteins"] == 0
    assert _policy(result)["unknown_backed_functional_complexes"] == 1


# ── 1c. species matching ───────────────────────────────────────────────────


def _species_verdict(declared: str, requested: str, **extra: Any) -> Dict[str, Any]:
    candidate = {
        "accession": "P17109",
        "protein_name": "MenD",
        "organism": declared,
        "score": 0.99,
    }
    candidate.update(extra)
    return verify_real_protein_identity(
        "MenD",
        candidates=[candidate],
        mapped_ids={"uniprot": "P17109"},
        organism=requested,
        result={},
    )


def test_exact_taxonomy_id_is_an_exact_species_match() -> None:
    verdict = _species_verdict(
        "", "Escherichia coli", taxonomy_id="562", requested_taxonomy_id="562"
    )

    assert verdict["verified"] is True
    assert verdict["checks"]["species"] == "ok"


def test_mismatched_taxonomy_id_is_a_species_mismatch() -> None:
    verdict = _species_verdict(
        "Escherichia coli", "Escherichia coli", taxonomy_id="562", requested_taxonomy_id="1613"
    )

    assert verdict["verified"] is False
    assert verdict["reason"] == "species_mismatch"


def test_strain_qualified_name_is_the_same_species() -> None:
    verdict = _species_verdict("Escherichia coli K-12", "Escherichia coli")

    assert verdict["verified"] is True
    assert verdict["checks"]["species"] == "ok"
    assert verdict["species_compatibility"] == "ok"


@pytest.mark.parametrize(
    "declared,requested",
    [
        ("Escherichia fergusonii", "Escherichia coli"),
        ("Bacillus cereus", "Bacillus subtilis"),
        ("Escherichia colicinogenes", "Escherichia coli"),
    ],
)
def test_same_genus_different_species_is_a_mismatch(declared: str, requested: str) -> None:
    """The rule that once accepted E. coli for E. fergusonii. Two organisms."""

    verdict = _species_verdict(declared, requested)

    assert verdict["verified"] is False
    assert verdict["reason"] == "species_mismatch"


def test_genus_level_request_accepts_a_species_in_it_and_says_so() -> None:
    """Allowed only because the request was explicitly genus-level."""

    verdict = _species_verdict("Escherichia coli", "Escherichia")

    assert verdict["verified"] is True
    assert verdict["checks"]["species"] == "genus_level"
    assert verdict["species_compatibility"] == "genus_level"


def test_genus_level_request_still_rejects_another_genus() -> None:
    verdict = _species_verdict("Bacillus subtilis", "Escherichia")

    assert verdict["verified"] is False
    assert verdict["reason"] == "species_mismatch"


# ── 1d. confidence belongs to one accession only ───────────────────────────


def test_shipped_accession_does_not_inherit_another_candidates_confidence() -> None:
    """The second-pass shape: the row carries P08235, the resolver chose P24200.

    ``_merge_mapped_ids`` keeps the identifier that arrived first, so result-level
    confidence describes the accession the resolver picked *this* pass -- not the
    one being shipped. Reading it let a weak identity borrow a strong one's score.
    """

    candidates = [
        # Already on the row from an earlier pass. Weak, and in the wrong organism.
        {
            "accession": "P08235",
            "uniprot": "P08235",
            "protein_name": "MenD-like protein",
            "organism": "Escherichia coli",
            "score": 0.05,
        },
        # What the resolver actually chose this pass, with a strong confidence.
        {
            "accession": "P24200",
            "uniprot": "P24200",
            "protein_name": "MenD",
            "organism": "Escherichia coli",
            "score": 0.99,
        },
    ]
    result = {"mapped_ids": {"uniprot": "P24200"}, "confidence": 0.99}

    shipped = verify_real_protein_identity(
        "MenD",
        candidates=candidates,
        mapped_ids={"uniprot": "P08235"},
        organism="Escherichia coli",
        result=result,
    )

    assert shipped["verified"] is False
    assert shipped["score"] == 0.05
    assert shipped["reason"] == "score_below_minimum"

    # ...and the accession the confidence does belong to still verifies on it.
    chosen = verify_real_protein_identity(
        "MenD",
        candidates=candidates,
        mapped_ids={"uniprot": "P24200"},
        organism="Escherichia coli",
        result=result,
    )
    assert chosen["verified"] is True


def test_result_confidence_is_used_when_it_names_the_shipped_candidate() -> None:
    """The narrow case where borrowing is not borrowing: it is the same row."""

    verdict = verify_real_protein_identity(
        "MenD",
        candidates=[
            {"accession": "P24200", "uniprot": "P24200", "protein_name": "MenD", "organism": "Escherichia coli"}
        ],
        mapped_ids={"uniprot": "P24200"},
        organism="Escherichia coli",
        result={"mapped_ids": {"uniprot": "P24200"}, "confidence": 0.97},
    )

    assert verdict["verified"] is True
    assert verdict["score"] == 0.97


def test_two_pass_mapping_does_not_launder_the_old_accession(tmp_path: Path) -> None:
    """End to end: the weak shipped id is stripped, not blessed by the new one."""

    payload = _payload(
        proteins=[
            _protein("MenD", mapped_ids={"uniprot": "P08235"}),
        ],
        reactions=[
            _enzyme_reaction({"entity": "MenD", "entity_type": "protein", "role": "catalyst"})
        ],
    )
    second_pass = {
        "status": "mapped",
        "provider": "UniProt",
        "source": "api",
        "confidence": 0.99,
        "chosen_rule": "reviewed_exact",
        "mapped_ids": {"uniprot": "P24200"},
        "candidates": [
            {
                "accession": "P08235",
                "uniprot": "P08235",
                "protein_name": "MenD-like protein",
                "organism": ORGANISM,
                "score": 0.05,
            },
            {
                "accession": "P24200",
                "uniprot": "P24200",
                "protein_name": "MenD",
                "organism": ORGANISM,
                "score": 0.99,
            },
        ],
        "resolution": {"status": "matched", "order_step": "api_uniprot"},
    }

    result = _map(payload, tmp_path, protein_result=second_pass)
    mapped = result["payload"]

    assert "P08235" not in str(mapped["entities"])
    assert _policy(result)["verified_real_proteins"] == 0
    assert _names(mapped["entities"]["protein_complexes"]) == ["MenD"]


# ── 1e. role support is described honestly ─────────────────────────────────


def test_role_assertion_without_direct_evidence_says_so(tmp_path: Path) -> None:
    """Collection membership authorizes the placeholder; it is not called evidence."""

    payload = _payload(
        proteins=[_protein("HepPPS")],
        reactions=[
            _enzyme_reaction({"entity": "HepPPS", "entity_type": "protein", "role": "catalyst"})
        ],
    )

    result = _map(payload, tmp_path, protein_result=_unmapped_result())
    meta = result["payload"]["entities"]["protein_complexes"][0]["mapping_meta"]

    assert meta["role_evidence"] == {
        "declared_role": "catalyst",
        "allowed_role": "enzyme",
        "role_basis": "canonical_actor_membership",
        "reason": "canonical_actor_membership",
        "ok": True,
        "direct_evidence_present": False,
    }
    # Nothing was invented to fill the gap.
    assert "evidence_digest" not in meta
    assert "source_refs" not in meta


def test_a_large_evidence_blob_is_excerpted_not_copied(tmp_path: Path) -> None:
    """`rag/conform.py` flattens retrieved evidence; one payload's was 139,576 chars."""

    blob = "HepPPS catalyses the prenyl transfer. " * 400
    payload = _payload(
        proteins=[_protein("HepPPS")],
        reactions=[
            _enzyme_reaction(
                {
                    "entity": "HepPPS",
                    "entity_type": "protein",
                    "role": "catalyst",
                    "evidence": blob,
                    "source_refs": ["seed_paper"],
                }
            )
        ],
    )

    result = _map(payload, tmp_path, protein_result=_unmapped_result())
    meta = result["payload"]["entities"]["protein_complexes"][0]["mapping_meta"]
    digest = meta["evidence_digest"]

    assert digest["chars"] == len(blob.strip())
    assert digest["truncated"] is True
    assert len(digest["excerpt"]) == 200
    assert len(json.dumps(meta)) < 2000
    assert meta["role_evidence"]["direct_evidence_present"] is True
    assert meta["source_refs"] == ["seed_paper"]


def test_sub_threshold_best_effort_match_is_not_a_verified_identity(tmp_path: Path) -> None:
    """A fuzzy hit whose only evidence is a 0.42 score is not an identity."""

    weak = {
        "status": "mapped",
        "provider": "UniProt",
        "source": "api",
        "confidence": 0.42,
        "chosen_rule": "best_effort_fallback",
        "best_effort": True,
        "mapped_ids": {"uniprot": "Q9WEAK"},
        "candidates": [{"accession": "Q9WEAK", "score": 0.42, "organism": ORGANISM}],
        "resolution": {"status": "matched", "issue": "best_effort", "order_step": "api_uniprot"},
    }
    payload = _payload(
        proteins=[_protein("N-methyl nucleosidase")],
        reactions=[
            _enzyme_reaction(
                {"entity": "N-methyl nucleosidase", "entity_type": "protein", "role": "catalyst"}
            )
        ],
    )

    result = _map(payload, tmp_path, protein_result=weak)
    mapped = result["payload"]

    assert _policy(result)["verified_real_proteins"] == 0
    # ...and because it is a legitimate enzyme, it still ships -- as a placeholder.
    complex_row = mapped["entities"]["protein_complexes"][0]
    assert complex_row["name"] == "N-methyl nucleosidase"
    assert complex_row["mapping_meta"]["identity_status"] == "placeholder"


# ── 2. a wrong ambiguous candidate becomes an Unknown-backed complex ────────


def test_wrong_ambiguous_candidate_becomes_an_unknown_backed_complex(tmp_path: Path) -> None:
    payload = _payload(
        proteins=[_protein("N-methyl nucleosidase")],
        reactions=[
            _enzyme_reaction(
                {
                    "entity": "N-methyl nucleosidase",
                    "entity_type": "protein",
                    "role": "catalyst",
                    "evidence": "N-methyl nucleosidase catalyses caffeine demethylation",
                }
            )
        ],
    )

    result = _map(payload, tmp_path, protein_result=_ambiguous_result())
    mapped = result["payload"]

    # Neither wrong-organism accession shipped, and none was picked by order.
    assert "O34362" not in str(mapped["entities"])
    assert "P17109" not in str(mapped["entities"])
    assert _policy(result)["ambiguous_real_candidates_rejected"] == 1
    assert _policy(result)["verified_real_proteins"] == 0

    complex_row = mapped["entities"]["protein_complexes"][0]
    assert complex_row["name"] == "N-methyl nucleosidase"
    assert complex_row["components"][0]["pathbank_protein_id"] == PATHBANK_UNKNOWN_PROTEIN_ID
    assert complex_row["mapping_meta"]["identity_status"] == "placeholder"
    assert complex_row["mapping_meta"]["cross_species_placeholder"] is True
    assert complex_row["mapping_meta"]["target_organism"] == ORGANISM
    assert complex_row["mapping_meta"]["fallback_used"] is True
    assert _policy(result)["unknown_backed_functional_complexes"] == 1


def test_ambiguous_candidates_are_never_resolved_by_list_order(tmp_path: Path) -> None:
    """Reversing the candidate list must not change the outcome."""

    payload = _payload(
        proteins=[_protein("N-methyl nucleosidase")],
        reactions=[
            _enzyme_reaction(
                {"entity": "N-methyl nucleosidase", "entity_type": "protein", "role": "catalyst"}
            )
        ],
    )
    forward = _ambiguous_result()
    reverse = _ambiguous_result()
    reverse["candidates"] = list(reversed(reverse["candidates"]))

    first = _map(deepcopy(payload), tmp_path, protein_result=forward)
    second = _map(deepcopy(payload), tmp_path, protein_result=reverse)

    assert _names(first["payload"]["entities"]["proteins"]) == _names(
        second["payload"]["entities"]["proteins"]
    )
    assert _names(first["payload"]["entities"]["protein_complexes"]) == _names(
        second["payload"]["entities"]["protein_complexes"]
    )
    for result in (first, second):
        protein = result["payload"]["entities"]["proteins"][0]
        assert is_pathbank_unknown_protein(protein)
        assert "ambiguous_first_candidate" not in str(result["payload"])


def test_a_single_verifiable_candidate_still_resolves_the_ambiguity(tmp_path: Path) -> None:
    """Ambiguity resolved on evidence, not deferred: one survivor is a mapping."""

    result_stub = _ambiguous_result()
    # Only the second candidate is in the pathway's organism now.
    result_stub["candidates"][1]["organism"] = ORGANISM
    payload = _payload(
        proteins=[_protein("N-methyl nucleosidase")],
        reactions=[
            _enzyme_reaction(
                {"entity": "N-methyl nucleosidase", "entity_type": "protein", "role": "catalyst"}
            )
        ],
    )

    result = _map(payload, tmp_path, protein_result=result_stub)
    protein = result["payload"]["entities"]["proteins"][0]

    assert protein["mapped_ids"]["uniprot"] == "P17109"
    assert protein["mapping_meta"]["chosen_rule"] == "ambiguous_verified_single_candidate"
    assert protein["mapping_meta"]["identity_status"] == "verified"
    assert protein["mapping_meta"]["ambiguity"]["verified_count"] == 1
    assert _policy(result)["ambiguous_real_candidates_rejected"] == 0


# ── 3. a no-match, evidence-backed enzyme becomes an Unknown-backed complex ──


def test_no_match_evidence_backed_enzyme_becomes_an_unknown_backed_complex(tmp_path: Path) -> None:
    payload = _payload(
        proteins=[
            _protein(
                "HepPPS",
                evidence="heptaprenyl diphosphate synthase (HepPPS) supplies the prenyl donor",
                source_refs=["seed_paper"],
            )
        ],
        reactions=[
            _enzyme_reaction(
                {
                    "entity": "HepPPS",
                    "entity_type": "protein",
                    "role": "catalyst",
                    "evidence": "HepPPS catalyses the prenyl transfer",
                }
            )
        ],
        protein_locations=[{"protein": "HepPPS", "biological_state": "leaf cytosol"}],
    )

    result = _map(payload, tmp_path, protein_result=_unmapped_result())
    mapped = result["payload"]

    assert _names(mapped["entities"]["proteins"]) == ["Unknown"]
    sentinel = mapped["entities"]["proteins"][0]
    assert is_pathbank_unknown_protein(sentinel)
    assert sentinel["mapping_meta"]["identity_status"] == "placeholder"
    assert sentinel["mapping_meta"]["placeholder_target_organisms"] == [ORGANISM]

    complex_row = mapped["entities"]["protein_complexes"][0]
    assert complex_row["name"] == "HepPPS"
    assert complex_row["mapping_meta"]["functional_enzyme_name"] == "HepPPS"
    assert complex_row["mapping_meta"]["identity_status"] == "placeholder"
    assert complex_row["mapping_meta"]["real_mapping_failure_reason"] == "no_match"
    # Bounded: the excerpt and hash cite the claim, the raw text is not copied.
    digest = complex_row["mapping_meta"]["evidence_digest"]
    assert digest["excerpt"] == "HepPPS catalyses the prenyl transfer"
    assert digest["truncated"] is False
    assert digest["chars"] == len("HepPPS catalyses the prenyl transfer")
    assert complex_row["mapping_meta"]["source_refs"] == ["seed_paper"]
    role_evidence = complex_row["mapping_meta"]["role_evidence"]
    assert role_evidence["ok"] is True
    assert role_evidence["role_basis"] == "canonical_actor_membership"
    assert role_evidence["direct_evidence_present"] is True
    # The superseded unresolved protein row and its location are both gone.
    assert mapped["element_locations"]["protein_locations"] == []
    assert _policy(result)["bare_unresolved_proteins_remaining"] == 0


def test_a_placeholder_is_never_reported_as_a_real_uniprot_mapping(tmp_path: Path) -> None:
    payload = _payload(
        proteins=[_protein("HepPPS")],
        reactions=[
            _enzyme_reaction({"entity": "HepPPS", "entity_type": "protein", "role": "catalyst"})
        ],
    )

    result = _map(payload, tmp_path, protein_result=_unmapped_result())

    assert _policy(result)["verified_real_proteins"] == 0
    assert _policy(result)["unknown_backed_functional_complexes"] == 1
    assert result["report"]["summary"]["proteins_mapped"] == 0
    complex_row = result["payload"]["entities"]["protein_complexes"][0]
    assert placeholder_claims_real_identity(complex_row) == ""
    assert identity_status(complex_row) == "placeholder"


# ── 4. a gene/operon with supported catalytic activity keeps its name ───────


def test_operon_with_supported_catalytic_activity_uses_a_named_functional_complex(
    tmp_path: Path,
) -> None:
    """'pmrHFIJKLM operon' has no accession of its own and must keep its name.

    Run 2026-07-28_0919 shipped this entity as UniProt P03023, the lactose
    operon repressor, because 'operon' was issued as a search term. The name is
    the biology here; the identifier never was.
    """

    payload = _payload(
        proteins=[
            _protein(
                "pmrHFIJKLM operon",
                evidence="the pmrHFIJKLM operon encodes the L-Ara4N transferase activity",
            )
        ],
        reactions=[
            _enzyme_reaction(
                {
                    "entity": "pmrHFIJKLM operon",
                    "entity_type": "protein",
                    "role": "catalyst",
                    "evidence": "the pmrHFIJKLM operon catalyses lipid A modification",
                }
            )
        ],
    )

    result = _map(payload, tmp_path, protein_result=_unmapped_result("novel_protein"))
    mapped = result["payload"]

    complex_row = mapped["entities"]["protein_complexes"][0]
    assert complex_row["name"] == "pmrHFIJKLM operon"
    assert complex_row["components"] == [
        {
            "name": "Unknown",
            "stoichiometry": 1,
            "pathbank_protein_id": PATHBANK_UNKNOWN_PROTEIN_ID,
            "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": PATHBANK_UNKNOWN_PROTEIN_ID},
        }
    ]
    actor = mapped["processes"]["reactions"][0]["enzymes"][0]
    assert actor["entity"] == "pmrHFIJKLM operon"
    assert actor["entity_type"] == "protein_complex"


# ── 5. CoA and succinyl-CoA never become protein complexes ─────────────────


@pytest.mark.parametrize(
    "name",
    [
        "CoA",
        "coenzyme A",
        "coenzyme A (CoA)",
        "succinyl-CoA",
        "succinyl coenzyme A",
        "succinyl-coenzyme A (ScoA)",
        "acetyl-CoA",
        "CoA-SH",
        "beta-hydroxyacyl-ACP",
        "malonyl-ACP",
        "trans-2-acyl-ACP",
    ],
)
def test_carrier_bound_metabolite_names_are_never_protein_like(name: str) -> None:
    assert compound_moiety_verdict(name) != ""


@pytest.mark.parametrize(
    "name",
    [
        "acyl carrier protein",
        "beta-hydroxyacyl-ACP dehydratase",
        "succinyl-CoA synthetase",
        "acetyl-CoA carboxylase",
        "CoA ligase",
        "ACP",
        "holo-ACP",
    ],
)
def test_the_enzymes_that_act_on_them_are_still_protein_like(name: str) -> None:
    """The guard must separate the substrate from its catalyst, not blanket both."""

    assert compound_moiety_verdict(name) == ""


def test_cofactors_are_not_promoted_to_proteins_or_complexes() -> None:
    payload = _payload(
        compounds=[
            {"name": "coenzyme A (CoA)"},
            {"name": "succinyl-coenzyme A (ScoA)"},
            {"name": "glycine"},
            {"name": "5-aminolevulinate"},
        ],
        proteins=[_protein("ALAS")],
        reactions=[
            {
                "name": "aminolevulinate synthesis",
                "inputs": ["succinyl-coenzyme A (ScoA)", "glycine"],
                "outputs": ["5-aminolevulinate", "coenzyme A (CoA)"],
                "biological_state": "leaf cytosol",
                "enzymes": [{"entity": "ALAS", "entity_type": "protein", "role": "catalyst"}],
            }
        ],
    )

    normalized, report = normalize_process_payload(payload)

    protein_names = set(_names(normalized["entities"]["proteins"]))
    complex_names = set(_names(normalized["entities"]["protein_complexes"]))
    for cofactor in ("coenzyme A (CoA)", "succinyl-coenzyme A (ScoA)"):
        assert cofactor not in protein_names
        assert cofactor not in complex_names
        assert cofactor in set(_names(normalized["entities"]["compounds"]))
    # Both cofactors would have been promoted by PROTEIN_LIKE_RE ("coenzyme"
    # contains "enzyme"); the block is counted, not silently applied.
    assert report["summary"]["false_protein_promotions_blocked"] == 1
    assert {
        action["name"]
        for action in report["actions"]
        if action.get("type") == "false_protein_promotion_blocked"
    } == {"succinyl-coenzyme A (ScoA)"}
    assert normalized["protein_export_policy"]["summary"]["false_protein_promotions_blocked"] == 1


def test_a_cofactor_actor_never_receives_the_unknown_fallback(tmp_path: Path) -> None:
    """Even parked in the enzymes list, a compound gets no functional complex."""

    payload = _payload(
        compounds=[{"name": "caffeine"}, {"name": "theobromine"}, {"name": "coenzyme A (CoA)"}],
        proteins=[_protein("coenzyme A (CoA)")],
        reactions=[
            _enzyme_reaction(
                {"entity": "coenzyme A (CoA)", "entity_type": "protein", "role": "catalyst"}
            )
        ],
    )

    result = _map(payload, tmp_path, protein_result=_unmapped_result())
    mapped = result["payload"]

    assert mapped["entities"]["protein_complexes"] == []
    assert all(row["name"] != "Unknown" for row in mapped["entities"]["proteins"])
    claim = mapped["quarantined_process_claims"][0]
    assert claim["actor"] == "coenzyme A (CoA)"
    assert claim["reason"].startswith("actor_names_a_compound:")
    assert result["report"]["summary"]["unsupported_process_claims_quarantined"] == 1


# ── 6. an unsupported actor does not receive the fallback ──────────────────


def test_unsupported_actor_does_not_receive_the_fallback(tmp_path: Path) -> None:
    """An inhibitor parked in ``enzymes`` is not a catalysis claim."""

    payload = _payload(
        proteins=[_protein("Regulator X")],
        reactions=[
            _enzyme_reaction(
                {"entity": "Regulator X", "entity_type": "protein", "role": "inhibitor"}
            )
        ],
    )

    result = _map(payload, tmp_path, protein_result=_unmapped_result())
    mapped = result["payload"]

    assert mapped["entities"]["protein_complexes"] == []
    assert mapped["processes"]["reactions"][0]["enzymes"] == []
    assert result["report"]["summary"]["reaction_enzyme_unknown_fallbacks"] == 0
    assert result["report"]["summary"]["unknown_fallbacks_skipped_no_role_evidence"] == 1

    claim = mapped["quarantined_process_claims"][0]
    assert claim["actor"] == "Regulator X"
    assert claim["role"] == "inhibitor"
    assert claim["reason"] == "declared_role_is_not_a_enzyme_claim"
    assert claim["action"] == "quarantined_from_process_actors"
    # The claim is quarantined, not deleted: the original row is preserved.
    assert claim["original_actor"]["entity"] == "Regulator X"


def test_a_protein_referenced_only_outside_a_catalytic_role_is_left_alone(tmp_path: Path) -> None:
    """The pre-existing rule, re-pinned: an interaction participant is not an enzyme."""

    payload = _payload(
        proteins=[_protein("Binder Y")],
        reactions=[],
    )
    payload["processes"]["interactions"] = [
        {"name": "observation", "entity_1": "Binder Y", "entity_2": "caffeine", "relationship": "binds"}
    ]

    result = _map(payload, tmp_path, protein_result=_unmapped_result())

    assert _names(result["payload"]["entities"]["proteins"]) == ["Binder Y"]
    assert result["payload"]["entities"]["protein_complexes"] == []
    assert result["report"]["summary"]["reaction_enzyme_unknown_fallbacks"] == 0


# ── 7. unused proteins leave strict export, mapped or not ─────────────────


def test_unused_unmapped_and_mapped_proteins_are_removed_from_strict_export() -> None:
    payload = _payload(
        proteins=[
            _protein("FabZ", mapped_ids={"uniprot": "P0A6Q6"}),
            _protein("FabA", mapped_ids={"uniprot": "P0A6Q3"}),  # mapped, unused
            _protein("Ghost enzyme"),  # unmapped, unused
        ],
        compounds=[{"name": "caffeine"}, {"name": "theobromine"}],
        reactions=[
            _enzyme_reaction({"entity": "FabZ", "entity_type": "protein", "role": "catalyst"})
        ],
    )

    normalized, report = normalize_process_payload(payload)

    assert "FabA" not in set(_names(normalized["entities"]["proteins"]))
    assert "Ghost enzyme" not in set(_names(normalized["entities"]["proteins"]))
    quarantined = {row["name"]: row for row in normalized["quarantined_proteins"]}
    assert {"FabA", "Ghost enzyme"} <= set(quarantined)
    assert quarantined["FabA"]["had_external_identity"] is True
    assert quarantined["Ghost enzyme"]["had_external_identity"] is False
    assert report["summary"]["unused_proteins_quarantined"] >= 2
    # The gate the old prune left them to now has nothing to reject.
    assert report["gate"]["ok"] is True


def test_research_mode_retains_and_flags_the_same_unused_proteins() -> None:
    payload = _payload(
        proteins=[
            _protein("FabZ", mapped_ids={"uniprot": "P0A6Q6"}),
            _protein("FabA", mapped_ids={"uniprot": "P0A6Q3"}),
            _protein("Ghost enzyme"),
        ],
        reactions=[
            _enzyme_reaction({"entity": "FabZ", "entity_type": "protein", "role": "catalyst"})
        ],
    )

    normalized, report = normalize_process_payload(payload, mode="research")

    assert {"FabA", "Ghost enzyme"} <= set(_names(normalized["entities"]["proteins"]))
    assert "quarantined_proteins" not in normalized
    assert report["summary"]["unused_proteins_flagged_for_review"] >= 2


def test_the_sentinel_component_is_not_treated_as_an_unused_protein(tmp_path: Path) -> None:
    """A component of a surviving functional complex is not a degree-zero orphan."""

    payload = _payload(
        proteins=[_protein("HepPPS")],
        reactions=[
            _enzyme_reaction({"entity": "HepPPS", "entity_type": "protein", "role": "catalyst"})
        ],
    )
    mapped = _map(payload, tmp_path, protein_result=_unmapped_result())["payload"]

    normalized, report = normalize_process_payload(mapped)

    assert _names(normalized["entities"]["proteins"]) == ["Unknown"]
    assert report["summary"].get("unused_proteins_quarantined", 0) == 0
    assert report["gate"]["ok"] is True


# ── 8. many unresolved enzymes, one sentinel, distinct visible names ───────


def test_multiple_unresolved_enzymes_reuse_one_sentinel_with_distinct_names(
    tmp_path: Path,
) -> None:
    payload = _payload(
        proteins=[_protein("HepPPS"), _protein("MenD"), _protein("ABCG-116")],
        reactions=[
            _enzyme_reaction(
                {"entity": "HepPPS", "entity_type": "protein", "role": "catalyst"},
                name="prenyl transfer",
            ),
            _enzyme_reaction(
                {"entity": "MenD", "entity_type": "protein", "role": "catalyst"},
                name="menaquinone step",
            ),
        ],
        transports=[
            {
                "name": "caffeine export",
                "cargo": "caffeine",
                "from_biological_state": "leaf cytosol",
                "to_biological_state": "leaf cytosol",
                "transporters": [
                    {"entity": "ABCG-116", "entity_type": "protein", "role": "transporter"}
                ],
            }
        ],
    )

    result = _map(payload, tmp_path, protein_result=_unmapped_result())
    mapped = result["payload"]

    # One shared sentinel...
    assert _names(mapped["entities"]["proteins"]) == ["Unknown"]
    assert _names(mapped["entities"]["proteins"]).count("Unknown") == 1
    # ...three distinct visible functional complexes.
    assert sorted(_names(mapped["entities"]["protein_complexes"])) == [
        "ABCG-116",
        "HepPPS",
        "MenD",
    ]
    for complex_row in mapped["entities"]["protein_complexes"]:
        assert complex_row["components"][0]["pathbank_protein_id"] == PATHBANK_UNKNOWN_PROTEIN_ID
    assert result["report"]["summary"]["reaction_enzyme_unknown_fallbacks"] == 2
    assert result["report"]["summary"]["transporter_unknown_fallbacks"] == 1
    assert _policy(result)["unknown_backed_functional_complexes"] == 3


# ── 9. no actor ever points at the literal name "Unknown" ─────────────────


def test_reaction_and_transport_actors_never_point_to_literal_unknown(tmp_path: Path) -> None:
    payload = _payload(
        proteins=[_protein("HepPPS"), _protein("ABCG-116")],
        reactions=[
            _enzyme_reaction(
                {"entity": "HepPPS", "entity_type": "protein", "role": "catalyst"}
            )
        ],
        transports=[
            {
                "name": "caffeine export",
                "cargo": "caffeine",
                "from_biological_state": "leaf cytosol",
                "to_biological_state": "leaf cytosol",
                "transporters": [
                    {"entity": "ABCG-116", "entity_type": "protein", "role": "transporter"}
                ],
            }
        ],
    )
    payload["processes"]["reactions"][0]["modifiers"] = [
        {"entity": "HepPPS", "entity_type": "protein", "role": "catalyst"}
    ]

    mapped = _map(payload, tmp_path, protein_result=_unmapped_result())["payload"]
    normalized, _ = normalize_process_payload(mapped)

    actors: List[Dict[str, Any]] = []
    for reaction in normalized["processes"]["reactions"]:
        actors.extend(reaction.get("enzymes", []))
        actors.extend(reaction.get("modifiers", []))
    for transport in normalized["processes"]["transports"]:
        actors.extend(transport.get("transporters", []))

    assert actors, "the fixture must actually produce actors"
    for actor in actors:
        for key in ("entity", "name", "protein", "protein_complex"):
            assert str(actor.get(key) or "") != "Unknown"
    assert {str(a.get("entity")) for a in actors} == {"HepPPS", "ABCG-116"}


# ── 10. the PWML visualization shows the functional name ───────────────────


def _build_pwml_root(normalized: Dict[str, Any]) -> Any:
    ir, ir_report = build_pwml_ir(normalized, strict_db=False)
    assert ir_report["errors"] == []
    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    builder = DeterministicPwmlBuilder(
        extraction=ir,
        signature=signature,
        args=SimpleNamespace(
            name="Caffeine degradation",
            description="",
            subject="Metabolic",
            pw_id="PW000000",
            height=1400,
            width=3200,
            background_color="#FFFFFF",
            ref=str(ROOT / "reference" / "PW000001.pwml"),
        ),
    )
    return builder.build().root


def test_pwml_visualization_displays_the_functional_complex_name(tmp_path: Path) -> None:
    payload = _payload(
        proteins=[_protein("N-methyl nucleosidase")],
        reactions=[
            _enzyme_reaction(
                {"entity": "N-methyl nucleosidase", "entity_type": "protein", "role": "catalyst"}
            )
        ],
    )
    mapped = _map(payload, tmp_path, protein_result=_unmapped_result())["payload"]
    normalized, _ = normalize_process_payload(mapped)
    root = _build_pwml_root(normalized)

    complex_el = root.find(".//protein-complexes/protein-complex")
    assert complex_el is not None
    # The biology-bearing name is what the reader sees...
    assert complex_el.findtext("name") == "N-methyl nucleosidase"
    complex_id = complex_el.findtext("id")

    # ...backed by the sentinel, which appears only as a member.
    member = complex_el.find(".//protein-complex-protein")
    assert member is not None
    assert member.findtext("protein-id") == str(PATHBANK_UNKNOWN_PROTEIN_ID)
    protein_el = root.find(".//proteins/protein")
    assert protein_el is not None
    assert protein_el.findtext("name") == "Unknown"

    # ...and it is actually drawn: a visualization points at the complex.
    viz = [
        element
        for element in root.iter("protein-complex-visualization")
        if element.findtext("protein-complex-id") == complex_id
    ]
    assert viz, "the functional complex must have a visualization to be visible"
    assert all(
        element.findtext("name") != "Unknown"
        for element in root.iter("protein-complex")
    )


# ── 11. strict gates: accept the right shape, reject the poser ─────────────


def _gate_errors(payload: Dict[str, Any], **kwargs: Any) -> List[Dict[str, Any]]:
    try:
        result = run_strict_post_normalization_gates(payload, **kwargs)
    except GateValidationError as exc:
        return list(exc.details.get("errors") or [])
    return list((result or {}).get("errors") or [])


def test_strict_gates_accept_a_correctly_formed_unknown_backed_complex(tmp_path: Path) -> None:
    payload = _payload(
        proteins=[_protein("N-methyl nucleosidase")],
        reactions=[
            _enzyme_reaction(
                {"entity": "N-methyl nucleosidase", "entity_type": "protein", "role": "catalyst"}
            )
        ],
    )
    mapped = _map(payload, tmp_path, protein_result=_unmapped_result())["payload"]
    validate_post_remap(mapped)

    normalized, normalization_report = normalize_process_payload(mapped)

    assert normalization_report["gate"]["ok"] is True
    assert validate_post_normalization(normalized, normalization_report["gate"])["ok"] is True
    assert validate_required_pwml_contract(normalized, strict_db=False)["ok"] is True


def test_strict_gates_reject_a_bare_unresolved_protein() -> None:
    payload = _payload(
        proteins=[_protein("HepPPS")],
        reactions=[
            _enzyme_reaction({"entity": "HepPPS", "entity_type": "protein", "role": "catalyst"})
        ],
    )

    reasons = {error["reason"] for error in _gate_errors(payload)}

    assert "Protein 'HepPPS' is missing a UniProt or DrugBank identifier." in reasons


def test_strict_gates_reject_a_placeholder_posing_as_a_real_uniprot_match() -> None:
    """The one shape every other check waves through: an identifier is present."""

    poser = _protein(
        "HepPPS",
        pathbank_protein_id=PATHBANK_UNKNOWN_PROTEIN_ID,
        mapped_ids={"uniprot": "P0A722"},
        identity_status="placeholder",
        mapping_meta={
            "chosen_rule": "pathbank_unknown_protein_fallback",
            "identity_status": "placeholder",
            "fallback_used": True,
            "cross_species_placeholder": True,
        },
    )
    payload = _payload(
        proteins=[poser],
        reactions=[
            _enzyme_reaction({"entity": "HepPPS", "entity_type": "protein", "role": "catalyst"})
        ],
    )

    errors = _gate_errors(payload)
    match = next(
        error for error in errors if "presented as a real mapping" in error["reason"]
    )

    assert match["path"] == "/entities/proteins/0"
    assert match["detail"]["rule"] == "placeholder_ships_real_uniprot_accession"
    assert placeholder_claims_real_identity(poser) == "placeholder_ships_real_uniprot_accession"


def test_strict_gates_reject_a_malformed_unknown_backed_complex() -> None:
    """A component merely *named* Unknown does not inherit the sentinel's pass."""

    payload = _payload(
        proteins=[_protein("Unknown", mapped_ids={"uniprot": "Q9FAKE"})],
        protein_complexes=[
            {
                "name": "HepPPS",
                "species": ORGANISM,
                "generated": True,
                "generation_reason": "single_protein_pathwhiz_wrapper",
                "components": [{"name": "Unknown", "stoichiometry": 1}],
            }
        ],
        reactions=[
            _enzyme_reaction(
                {"entity": "HepPPS", "entity_type": "protein_complex", "role": "catalyst"}
            )
        ],
    )

    reasons = [error["reason"] for error in _gate_errors(payload)]

    assert any("malformed Unknown placeholder" in reason for reason in reasons)


# ── 12. normalization must not undo the wrapper ────────────────────────────


def test_later_normalization_keeps_the_wrapper_and_drops_the_original(tmp_path: Path) -> None:
    payload = _payload(
        proteins=[_protein("N-methyl nucleosidase")],
        reactions=[
            _enzyme_reaction(
                {"entity": "N-methyl nucleosidase", "entity_type": "protein", "role": "catalyst"}
            )
        ],
        protein_locations=[
            {"protein": "N-methyl nucleosidase", "biological_state": "leaf cytosol"}
        ],
    )
    mapped = _map(payload, tmp_path, protein_result=_unmapped_result())["payload"]

    normalized, report = normalize_process_payload(mapped)

    # The wrapper survives, with its placeholder metadata intact...
    complex_row = next(
        row
        for row in normalized["entities"]["protein_complexes"]
        if row["name"] == "N-methyl nucleosidase"
    )
    assert complex_row["generated"] is True
    assert complex_row["mapping_meta"]["chosen_rule"] == "pathbank_unknown_protein_fallback"
    assert complex_row["components"][0]["pathbank_protein_id"] == PATHBANK_UNKNOWN_PROTEIN_ID
    # ...the superseded protein is not resurrected...
    assert _names(normalized["entities"]["proteins"]) == ["Unknown"]
    # ...and the actor still points at the functional complex, not the sentinel.
    actor = normalized["processes"]["reactions"][0]["enzymes"][0]
    assert actor["entity"] == "N-methyl nucleosidase"
    assert actor["entity_type"] == "protein_complex"
    assert report["gate"]["ok"] is True


def test_re_mapping_a_wrapped_payload_is_idempotent(tmp_path: Path) -> None:
    """Stage 6 runs twice per pipeline; the second pass must be a no-op here."""

    payload = _payload(
        proteins=[_protein("N-methyl nucleosidase")],
        reactions=[
            _enzyme_reaction(
                {"entity": "N-methyl nucleosidase", "entity_type": "protein", "role": "catalyst"}
            )
        ],
    )
    first = _map(payload, tmp_path, protein_result=_unmapped_result())["payload"]
    second_result = _map(deepcopy(first), tmp_path, protein_result=_unmapped_result())
    second = second_result["payload"]

    assert _names(second["entities"]["proteins"]) == ["Unknown"]
    assert _names(second["entities"]["protein_complexes"]) == ["N-methyl nucleosidase"]
    assert second_result["report"]["summary"]["reaction_enzyme_unknown_fallbacks"] == 0
    assert _policy(second_result)["unknown_backed_functional_complexes"] == 1
