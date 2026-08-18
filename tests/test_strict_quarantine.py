"""Pre-export quarantine and graph closure.

Strict export used to be all-or-nothing: one peripheral reaction naming an
entity nobody declared failed ``validate_required_pwml_contract`` for the whole
pathway, and the good reactions beside it were never written. These tests pin
the alternative -- quarantine the bad process as a unit, close the graph, and
export what is left -- and, just as importantly, pin the three ways that must
NOT be allowed to look like success:

* an essential participant quietly deleted so the reaction "passes";
* an Unknown-backed functional complex quarantined for lacking a real UniProt ID,
  which would delete exactly the biology the PathBank fallback exists to keep;
* an empty or off-topic graph passing because everything invalid was removed.
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

from t2pw.pipeline.process_normalizer import (  # noqa: E402
    GateValidationError,
    run_strict_post_normalization_gates,
    validate_registry_references,
)
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    AUXILIARY_ACCEPTED,
    CLOSURE_REPORT_FILENAME,
    CORE_ACCEPTED,
    COVERAGE_REPORT_FILENAME,
    QUARANTINE_REPORT_FILENAME,
    QUARANTINED_BROKEN_REFERENCE,
    QUARANTINED_DISCONNECTED,
    QUARANTINED_OUT_OF_SCOPE,
    QUARANTINED_UNMAPPED_ENTITY,
    QUARANTINED_WEAK_EVIDENCE,
    REMOVED_ENTITY_REPORT_FILENAME,
    StrictQuarantineInvariantError,
    _degree_zero_exports,
    _surviving_processes,
    evaluate_core_coverage,
    quarantine_and_close,
    write_quarantine_artifacts,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _glutathione_payload() -> dict:
    """Two-step glutathione biosynthesis: small, valid, and fully declared.

    Every test below starts from this and breaks exactly one thing, so a state
    change in the result is attributable to that one edit.
    """

    return {
        "metadata": {
            "pathway_name": "Glutathione biosynthesis",
            "pathway_subject": "Metabolic",
            "organism": "Homo sapiens",
            "key_compounds": ["L-glutamate", "glutathione"],
            "key_proteins": ["glutamate-cysteine ligase"],
        },
        "entities": {
            "species": [{"name": "Homo sapiens", "taxonomy_id": "9606"}],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [
                {"name": "L-glutamate"},
                {"name": "L-cysteine"},
                {"name": "gamma-glutamylcysteine"},
                {"name": "glycine"},
                {"name": "glutathione"},
            ],
            "proteins": [
                {
                    "name": "glutamate-cysteine ligase",
                    "species": "Homo sapiens",
                    "mapped_ids": {"uniprot": "P48506"},
                },
                {
                    "name": "glutathione synthetase",
                    "species": "Homo sapiens",
                    "mapped_ids": {"uniprot": "P48637"},
                },
            ],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "biological_states": [
            {
                "name": "cytosol",
                "species": "Homo sapiens",
                "subcellular_location": "cytosol",
            }
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": name, "biological_state": "cytosol"}
                for name in (
                    "L-glutamate",
                    "L-cysteine",
                    "gamma-glutamylcysteine",
                    "glycine",
                    "glutathione",
                )
            ],
            "protein_locations": [
                {"protein": "glutamate-cysteine ligase", "biological_state": "cytosol"},
                {"protein": "glutathione synthetase", "biological_state": "cytosol"},
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "gamma-glutamylcysteine synthesis",
                    "inputs": ["L-glutamate", "L-cysteine"],
                    "outputs": ["gamma-glutamylcysteine"],
                    "enzymes": [{"entity": "glutamate-cysteine ligase"}],
                    "biological_state": "cytosol",
                },
                {
                    "name": "glutathione synthesis",
                    "inputs": ["gamma-glutamylcysteine", "glycine"],
                    "outputs": ["glutathione"],
                    "enzymes": [{"entity": "glutathione synthetase"}],
                    "biological_state": "cytosol",
                },
            ],
            "transports": [],
            "interactions": [],
        },
    }


def _add_reaction(payload: dict, reaction: dict) -> dict:
    payload["processes"].setdefault("reactions", []).append(reaction)
    return payload


def _states(result) -> dict:
    return result.states()


# ── 1. A bad peripheral reaction leaves; the core survives ──────────────────


def test_one_bad_peripheral_reaction_is_quarantined_while_the_valid_core_survives() -> None:
    payload = _add_reaction(
        _glutathione_payload(),
        {
            # The shape that fails the required-field gate for the whole file:
            # a participant no entity bucket declares.
            "name": "peripheral oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol",
        },
    )

    result = quarantine_and_close(payload, strict_db=True)

    assert _states(result) == {
        "gamma-glutamylcysteine synthesis": CORE_ACCEPTED,
        "glutathione synthesis": CORE_ACCEPTED,
        "peripheral oxidation": QUARANTINED_UNMAPPED_ENTITY,
    }
    assert result.ok is True
    surviving = [row["name"] for row in result.payload["processes"]["reactions"]]
    assert surviving == ["gamma-glutamylcysteine synthesis", "glutathione synthesis"]

    # The point of the exercise: what is left now passes the strict gates it
    # previously failed, without any gate having been relaxed.
    run_strict_post_normalization_gates(result.payload, enforce_all_proteins_connected=True)
    validate_registry_references(result.payload)


def test_the_quarantine_record_names_the_participant_and_keeps_the_original_row() -> None:
    payload = _add_reaction(
        _glutathione_payload(),
        {
            "name": "peripheral oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol",
        },
    )

    result = quarantine_and_close(payload, strict_db=True)
    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "peripheral oxidation"
    )

    assert record["state"] == QUARANTINED_UNMAPPED_ENTITY
    assert record["essential_participant"] == "glutathione disulfide"
    assert record["pointer"] == "/processes/reactions/2"
    # Retained verbatim, not digested: a reviewer restoring this reaction needs
    # the participants back, not a summary of them.
    assert record["original_process"]["outputs"] == ["glutathione disulfide"]


# ── 2. An unrepresentable essential participant takes the whole process ─────


def test_unrepresentable_essential_participant_quarantines_the_whole_reaction() -> None:
    """The reaction leaves as a unit; the participant is never edited out.

    Deleting just the offending participant would leave a reaction with one
    input where the paper described two -- a different reaction, exported as if
    it were the one that was read.
    """

    payload = _glutathione_payload()
    payload["entities"]["compounds"].append({"name": "glutathione disulfide"})
    _add_reaction(
        payload,
        {
            "name": "glutathione oxidation",
            "inputs": ["glutathione", "hydrogen peroxide"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol",
        },
    )

    result = quarantine_and_close(payload, strict_db=True)

    assert _states(result)["glutathione oxidation"] == QUARANTINED_UNMAPPED_ENTITY
    assert "glutathione oxidation" not in [
        row["name"] for row in result.payload["processes"]["reactions"]
    ]
    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "glutathione oxidation"
    )
    assert record["essential_participant"] == "hydrogen peroxide"
    assert record["original_process"]["inputs"] == ["glutathione", "hydrogen peroxide"]


def test_a_protein_without_external_identity_quarantines_the_reaction_not_the_enzyme_slot() -> None:
    payload = _glutathione_payload()
    payload["entities"]["proteins"].append(
        {"name": "unmapped oxidase", "species": "Homo sapiens"}
    )
    payload["entities"]["compounds"].append({"name": "glutathione disulfide"})
    _add_reaction(
        payload,
        {
            "name": "glutathione oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "enzymes": [{"entity": "unmapped oxidase"}],
            "biological_state": "cytosol",
        },
    )

    result = quarantine_and_close(payload, strict_db=True)
    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "glutathione oxidation"
    )

    assert record["state"] == QUARANTINED_UNMAPPED_ENTITY
    assert record["reason"] == "protein_missing_external_identity"
    # The enzyme row is still in the retained original, untouched.
    assert record["original_process"]["enzymes"] == [{"entity": "unmapped oxidase"}]

    # strict_db=False is the DB-less run, where the identity rule does not apply
    # and the same reaction must be admitted.
    relaxed = quarantine_and_close(payload, strict_db=False)
    assert _states(relaxed)["glutathione oxidation"] in {CORE_ACCEPTED, AUXILIARY_ACCEPTED}


def test_a_reaction_with_an_empty_side_is_a_broken_reference_not_an_unmapped_entity() -> None:
    payload = _add_reaction(
        _glutathione_payload(),
        {"name": "hollow reaction", "inputs": [], "outputs": ["glutathione"], "biological_state": "cytosol"},
    )

    result = quarantine_and_close(payload, strict_db=True)
    record = next(row for row in result.quarantine_report["quarantined"] if row["name"] == "hollow reaction")

    assert record["state"] == QUARANTINED_BROKEN_REFERENCE
    assert record["reason"] == "missing_inputs"


# ── 3. Closure reaches a stable state ───────────────────────────────────────


def _cascading_payload() -> dict:
    """A quarantined reaction whose complex, and that complex's subunits, follow.

    Round 1 drops the complex and its now-unexempt components; round 2 finds
    nothing. Real cascade, no invented process kind.
    """

    payload = _glutathione_payload()
    payload["entities"]["proteins"].append(
        {"name": "GPX1", "species": "Homo sapiens", "mapped_ids": {"uniprot": "P07203"}}
    )
    payload["entities"]["protein_complexes"].append(
        {
            "name": "glutathione peroxidase complex",
            "class": "protein_complex",
            "generated": True,
            "generation_reason": "single_protein_pathwhiz_wrapper",
            "species": "Homo sapiens",
            "components": [{"name": "GPX1", "stoichiometry": 1}],
        }
    )
    _add_reaction(
        payload,
        {
            "name": "doomed reaction",
            "inputs": ["undeclared substrate"],
            "outputs": ["glutathione"],
            "enzymes": [{"entity": "glutathione peroxidase complex"}],
            "biological_state": "cytosol",
        },
    )
    return payload


def test_graph_closure_reaches_a_stable_state() -> None:
    result = quarantine_and_close(_cascading_payload(), strict_db=True)
    closure = result.closure_report

    assert closure["converged"] is True
    assert closure["stable_after"] == closure["iteration_count"]
    # The last round is the proof of stability: it changed nothing.
    assert closure["iterations"][-1]["changed"] is False
    assert _states(result)["doomed reaction"] == QUARANTINED_UNMAPPED_ENTITY

    removed = {row["name"] for row in result.removed_entity_report["removed_entities"]}
    assert {"glutathione peroxidase complex", "GPX1"} <= removed


def test_closure_is_a_fixpoint_rerunning_it_changes_nothing() -> None:
    first = quarantine_and_close(_cascading_payload(), strict_db=True)
    second = quarantine_and_close(first.payload, strict_db=True)

    assert second.payload == first.payload
    assert second.quarantine_report["counts"][QUARANTINED_DISCONNECTED] == 0
    assert second.removed_entity_report["counts"] == {
        "entities": 0,
        "locations": 0,
        "biological_states": 0,
    }


def test_closure_records_every_iteration_it_ran() -> None:
    result = quarantine_and_close(_cascading_payload(), strict_db=True)
    iterations = result.closure_report["iterations"]

    assert [row["iteration"] for row in iterations] == list(range(1, len(iterations) + 1))
    assert all(row["changed"] for row in iterations[:-1])
    assert iterations[-1]["changed"] is False


def test_the_closure_backstop_catches_a_removal_that_was_not_reference_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``quarantined_disconnected`` is a guarantee, not decoration.

    Entity pruning is driven by what the survivors reference, so a surviving
    process can never be stranded and this state stays at zero on real payloads.
    That is a property of the *current* pruning rules, not of the design -- so the
    re-validation pass exists to make a future non-reference-driven removal leave
    with a recorded reason instead of putting a dangling reference in front of the
    strict gate. Here that removal is simulated directly.
    """

    from t2pw.pipeline import strict_quarantine as sq

    real_prune = sq._prune_entities
    calls = {"n": 0}

    def _prune_and_also_drop_glycine(payload, keep_norms):
        removed = real_prune(payload, keep_norms)
        calls["n"] += 1
        if calls["n"] == 1:
            compounds = payload["entities"]["compounds"]
            kept = [row for row in compounds if row.get("name") != "glycine"]
            if len(kept) != len(compounds):
                payload["entities"]["compounds"] = kept
                removed.append(
                    {"bucket": "compounds", "name": "glycine", "reason": "simulated_rule"}
                )
        return removed

    monkeypatch.setattr(sq, "_prune_entities", _prune_and_also_drop_glycine)
    result = sq.quarantine_and_close(_glutathione_payload(), strict_db=True)

    # "glutathione synthesis" consumes glycine and was admitted; the backstop is
    # the only thing between that removal and an unresolvable reaction input.
    assert _states(result)["glutathione synthesis"] == QUARANTINED_DISCONNECTED
    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "glutathione synthesis"
    )
    assert record["reason"].startswith("stranded_by_closure:")
    assert record["essential_participant"] == "glycine"
    assert record["iteration"] >= 1


# ── 4. No dangling locations, states, or complex components ─────────────────


def test_closure_leaves_no_dangling_locations_or_biological_states() -> None:
    payload = _glutathione_payload()
    payload["entities"]["compounds"].append({"name": "peroxisomal ghost"})
    payload["biological_states"].append(
        {"name": "peroxisome", "species": "Homo sapiens", "subcellular_location": "peroxisome"}
    )
    payload["element_locations"]["compound_locations"].append(
        {"compound": "peroxisomal ghost", "biological_state": "peroxisome"}
    )

    result = quarantine_and_close(payload, strict_db=True)

    declared = {row["name"] for row in result.payload["entities"]["compounds"]}
    assert "peroxisomal ghost" not in declared
    located = {
        row["compound"] for row in result.payload["element_locations"]["compound_locations"]
    }
    assert located <= declared
    # The state existed only to hold that location, and goes with it.
    assert [row["name"] for row in result.payload["biological_states"]] == ["cytosol"]

    removed = result.removed_entity_report
    assert {row["name"] for row in removed["removed_entities"]} == {"peroxisomal ghost"}
    assert {row["name"] for row in removed["removed_locations"]} == {"peroxisomal ghost"}
    assert {row["name"] for row in removed["removed_biological_states"]} == {"peroxisome"}


def test_components_of_a_surviving_complex_are_preserved_even_at_degree_zero() -> None:
    payload = _glutathione_payload()
    payload["entities"]["proteins"].append(
        {
            "name": "GCLC",
            "species": "Homo sapiens",
            "mapped_ids": {"uniprot": "P48506"},
        }
    )
    payload["entities"]["proteins"].append(
        {
            "name": "GCLM",
            "species": "Homo sapiens",
            "mapped_ids": {"uniprot": "P48507"},
        }
    )
    payload["entities"]["protein_complexes"].append(
        {
            "name": "glutamate-cysteine ligase complex",
            "class": "protein_complex",
            "species": "Homo sapiens",
            "components": [{"name": "GCLC", "stoichiometry": 1}, {"name": "GCLM", "stoichiometry": 1}],
        }
    )
    payload["processes"]["reactions"][0]["enzymes"] = [
        {"entity": "glutamate-cysteine ligase complex"}
    ]

    result = quarantine_and_close(payload, strict_db=True)
    proteins = {row["name"] for row in result.payload["entities"]["proteins"]}

    # Neither component appears in any reaction; the complex carries the edge.
    assert {"GCLC", "GCLM"} <= proteins
    assert [row["name"] for row in result.payload["entities"]["protein_complexes"]] == [
        "glutamate-cysteine ligase complex"
    ]
    assert result.quarantine_report["strict_invariants"]["degree_zero_exports"] == []


def test_components_go_when_the_complex_that_preserved_them_goes() -> None:
    payload = _glutathione_payload()
    payload["entities"]["proteins"].append(
        {"name": "orphan subunit", "species": "Homo sapiens", "mapped_ids": {"uniprot": "Q00000"}}
    )
    payload["entities"]["protein_complexes"].append(
        {
            "name": "orphan complex",
            "class": "protein_complex",
            "species": "Homo sapiens",
            "components": [{"name": "orphan subunit", "stoichiometry": 1}],
        }
    )

    result = quarantine_and_close(payload, strict_db=True)
    removed = {row["name"] for row in result.removed_entity_report["removed_entities"]}

    assert removed == {"orphan complex", "orphan subunit"}
    assert result.payload["entities"]["protein_complexes"] == []


# ── 5. Unknown-backed functional complexes are valid ────────────────────────


def _unknown_backed_payload() -> dict:
    """The Stage 6 structural fallback, in the shape map_ids actually emits.

    The complex keeps the biology-bearing enzyme name; the component is the
    PathBank ``Unknown`` sentinel (protein 9659, UniProt "Unknown"). Quarantining
    this for "no real UniProt ID" would delete a genuinely novel enzyme from a
    2026 paper -- the exact case the fallback was built for.
    """

    payload = _glutathione_payload()
    payload["entities"]["proteins"].append(
        {
            "name": "Unknown",
            "species": "Homo sapiens",
            "organism": "Homo sapiens",
            "pathbank_protein_id": 9659,
            "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659},
            "mapping_meta": {
                "chosen_rule": "pathbank_unknown_protein_fallback",
                "fallback_used": True,
                "pathbank_protein_id": 9659,
                "resolution": {"status": "fallback", "issue": "pathbank_unknown_sentinel"},
            },
        }
    )
    payload["entities"]["protein_complexes"].append(
        {
            "name": "glutathione hydrolase",
            "class": "protein_complex",
            "generated": True,
            "species": "Homo sapiens",
            "components": [{"name": "Unknown", "stoichiometry": 1}],
            "mapping_meta": {"chosen_rule": "pathbank_unknown_protein_fallback"},
        }
    )
    payload["entities"]["compounds"].append({"name": "cysteinylglycine"})
    _add_reaction(
        payload,
        {
            "name": "glutathione hydrolysis",
            "inputs": ["glutathione"],
            "outputs": ["cysteinylglycine"],
            "enzymes": [{"entity": "glutathione hydrolase"}],
            "biological_state": "cytosol",
        },
    )
    payload["element_locations"]["compound_locations"].append(
        {"compound": "cysteinylglycine", "biological_state": "cytosol"}
    )
    return payload


def test_unknown_backed_functional_complex_survives_quarantine() -> None:
    result = quarantine_and_close(_unknown_backed_payload(), strict_db=True)

    assert _states(result)["glutathione hydrolysis"] in {CORE_ACCEPTED, AUXILIARY_ACCEPTED}
    assert [row["name"] for row in result.payload["entities"]["protein_complexes"]] == [
        "glutathione hydrolase"
    ]
    assert "Unknown" in {row["name"] for row in result.payload["entities"]["proteins"]}
    assert result.quarantine_report["counts"][QUARANTINED_UNMAPPED_ENTITY] == 0
    assert result.ok is True


def test_a_placeholder_shipping_a_real_accession_is_still_quarantined() -> None:
    """Requirement 4 protects correctly-formed placeholders, not forged ones."""

    payload = _unknown_backed_payload()
    unknown = next(row for row in payload["entities"]["proteins"] if row["name"] == "Unknown")
    unknown["mapped_ids"]["uniprot"] = "P19440"

    result = quarantine_and_close(payload, strict_db=True)
    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "glutathione hydrolysis"
    )

    assert record["state"] == QUARANTINED_UNMAPPED_ENTITY
    assert "malformed_placeholder" in record["reason"]


def test_a_complex_whose_component_is_undeclared_is_a_broken_reference() -> None:
    payload = _unknown_backed_payload()
    payload["entities"]["proteins"] = [
        row for row in payload["entities"]["proteins"] if row["name"] != "Unknown"
    ]

    result = quarantine_and_close(payload, strict_db=True)
    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "glutathione hydrolysis"
    )

    assert record["state"] == QUARANTINED_UNMAPPED_ENTITY
    assert record["reason"].startswith("protein_complex_component_unresolved")


# ── 6. Out-of-scope and weak-evidence admission states ──────────────────────


def test_only_an_explicit_out_of_scope_verdict_quarantines() -> None:
    """Mirrors pipeline.filter_out_of_scope_reactions: absent means unclassified.

    Every RAG-imported and Stage-2-added reaction arrives with no label at all,
    so treating absence as a verdict would delete evidenced biology on the
    strength of a field nobody filled in.
    """

    payload = _glutathione_payload()
    payload["processes"]["reactions"][0]["scope_membership"] = "OUT_OF_SCOPE"
    payload["processes"]["reactions"][1].pop("scope_membership", None)

    result = quarantine_and_close(payload, strict_db=True)
    states = _states(result)

    assert states["gamma-glutamylcysteine synthesis"] == QUARANTINED_OUT_OF_SCOPE
    assert states["glutathione synthesis"] == CORE_ACCEPTED


def test_weak_evidence_is_off_until_a_floor_is_passed_and_never_fires_on_absence() -> None:
    payload = _glutathione_payload()
    payload["processes"]["reactions"][0]["confidence"] = 0.2
    # Reaction 1 declares no confidence at all, which is the common case.

    assert quarantine_and_close(payload, strict_db=True).quarantine_report["counts"][
        QUARANTINED_WEAK_EVIDENCE
    ] == 0

    floored = quarantine_and_close(payload, strict_db=True, confidence_floor=0.5)
    states = _states(floored)
    assert states["gamma-glutamylcysteine synthesis"] == QUARANTINED_WEAK_EVIDENCE
    assert states["glutathione synthesis"] == CORE_ACCEPTED


# ── 7. Minimum viable requested-pathway core ────────────────────────────────


def test_empty_graph_cannot_pass() -> None:
    """Removing everything invalid is not a way to succeed."""

    payload = _glutathione_payload()
    for reaction in payload["processes"]["reactions"]:
        reaction["scope_membership"] = "out_of_scope"

    result = quarantine_and_close(payload, strict_db=True)

    assert result.payload["processes"]["reactions"] == []
    assert result.ok is False
    assert "no_surviving_process" in result.coverage["reasons"]
    assert result.coverage["minimum_core_satisfied"] is False


def test_a_payload_with_no_processes_at_all_cannot_pass() -> None:
    payload = _glutathione_payload()
    payload["processes"] = {"reactions": [], "transports": [], "interactions": []}

    result = quarantine_and_close(payload, strict_db=True)

    assert result.ok is False
    assert "no_surviving_process" in result.coverage["reasons"]


def test_unrelated_surviving_reactions_cannot_satisfy_minimum_core_coverage() -> None:
    """Survivors have to be the pathway that was asked for.

    Here every requested-core term is quarantined away and what is left is a
    complete, gate-clean, entirely unrelated reaction. It must not carry the run.
    """

    payload = _glutathione_payload()
    payload["entities"]["compounds"].extend(
        [{"name": "citrate"}, {"name": "isocitrate"}]
    )
    payload["element_locations"]["compound_locations"].extend(
        [
            {"compound": "citrate", "biological_state": "cytosol"},
            {"compound": "isocitrate", "biological_state": "cytosol"},
        ]
    )
    _add_reaction(
        payload,
        {
            "name": "citrate isomerisation",
            "inputs": ["citrate"],
            "outputs": ["isocitrate"],
            "biological_state": "cytosol",
        },
    )
    for reaction in payload["processes"]["reactions"][:2]:
        reaction["scope_membership"] = "out_of_scope"

    result = quarantine_and_close(payload, strict_db=True)

    assert _states(result)["citrate isomerisation"] == AUXILIARY_ACCEPTED
    assert result.payload["processes"]["reactions"] != []
    # MOVED BY C-041a (D-002, LOCKED). At the base SHA this was ``ok is False``
    # and the run ended with no PWML. The claim the test makes is unchanged and
    # is now asserted directly rather than through a boolean: an unrelated
    # survivor still cannot produce SUCCESS. It produces review_required PWML,
    # which is explicitly not release_ready and never enters the strict
    # denominator. Deleting the fragment instead is what PRODUCT_CONTRACT 1
    # forbids.
    assert result.ok is True
    release = result.quarantine_report["release"]
    assert release["status"] == "review_required"
    assert release["strict_acceptance_eligible"] is False
    assert any(
        reason.startswith("minimum_core:")
        for reason in result.quarantine_report["review_reasons"]
    )
    assert result.coverage["coverage_ratio"] == 0.0
    assert result.coverage["core_accepted_processes"] == 0
    assert any(
        reason.startswith("core_process_count_below_minimum")
        for reason in result.coverage["reasons"]
    )


def test_partial_core_coverage_below_the_floor_fails() -> None:
    payload = _glutathione_payload()
    payload["metadata"]["key_compounds"] = [
        "L-glutamate",
        "ophthalmate",
        "hypotaurine",
        "taurine",
    ]
    payload["metadata"]["key_proteins"] = []

    result = quarantine_and_close(payload, strict_db=True, min_core_coverage=0.5)

    assert result.coverage["matched_terms"] == ["L-glutamate"]
    assert result.coverage["coverage_ratio"] == pytest.approx(0.25)
    # MOVED BY C-041a (D-002, LOCKED): the threshold blocks release-ready status,
    # not PWML production, so a shortfall is review_required rather than a
    # refusal. Base SHA: ``ok is False``. The threshold value is unchanged and
    # the reason is still raised and still recorded.
    assert result.ok is True
    assert result.quarantine_report["release"]["status"] == "review_required"
    assert result.quarantine_report["release"]["strict_acceptance_eligible"] is False
    assert any(
        reason.startswith("requested_core_coverage_below_minimum")
        for reason in result.coverage["reasons"]
    )


def test_an_undeclared_core_falls_back_to_the_emptiness_rule_alone() -> None:
    """No declared core means relevance is unjudgeable, not that it is zero."""

    payload = _glutathione_payload()
    payload["metadata"].pop("key_compounds")
    payload["metadata"].pop("key_proteins")

    result = quarantine_and_close(payload, strict_db=True)

    assert result.coverage["requested_core_declared"] is False
    assert result.coverage["core_accepted_processes"] == 2
    assert result.ok is True


def test_an_explicit_requested_core_argument_overrides_the_payload() -> None:
    result = quarantine_and_close(
        _glutathione_payload(), strict_db=True, requested_core=["urea cycle", "ornithine"]
    )

    assert result.coverage["requested_core_terms"] == ["urea cycle", "ornithine"]
    assert result.coverage["unmatched_terms"] == ["urea cycle", "ornithine"]
    # MOVED BY C-041a (D-002, LOCKED). Base SHA: ``ok is False``. The explicit
    # argument still outranks the payload -- that is what this test is about --
    # and covering none of it is now review_required, never strict success.
    assert result.ok is True
    assert result.quarantine_report["release"]["status"] == "review_required"
    assert result.quarantine_report["release"]["strict_acceptance_eligible"] is False


def test_evaluate_core_coverage_counts_only_core_accepted_processes() -> None:
    admissions = [
        {"state": CORE_ACCEPTED, "core_terms": ["glutathione"]},
        {"state": AUXILIARY_ACCEPTED, "core_terms": ["citrate", "isocitrate"]},
    ]
    coverage = evaluate_core_coverage(
        {}, admissions, requested_core=["glutathione", "citrate"]
    )

    assert coverage["matched_terms"] == ["glutathione"]
    assert coverage["unmatched_terms"] == ["citrate"]


# ── 8. Strict invariants are reported, not repaired ─────────────────────────


def test_an_ambiguous_entity_name_quarantines_the_processes_that_reference_it() -> None:
    """One name, two entity types, and no answer to "which one did you mean".

    Resolving it would mean picking whichever bucket happened to be registered
    first -- an arbitrary answer to a question the payload does not answer. The
    reference is a broken reference, so the process holding it leaves.
    """

    payload = _glutathione_payload()
    payload["entities"]["protein_complexes"].append(
        {
            "name": "glutathione synthetase",
            "class": "protein_complex",
            "species": "Homo sapiens",
            "components": [{"name": "glutathione synthetase", "stoichiometry": 1}],
        }
    )

    result = quarantine_and_close(payload, strict_db=True)
    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "glutathione synthesis"
    )

    assert record["state"] == QUARANTINED_BROKEN_REFERENCE
    assert record["reason"] == "ambiguous_entity_type:protein_complexes+proteins"
    # Both declarations then fall out as unreferenced, so the closed graph is
    # unambiguous rather than merely un-exported.
    assert result.quarantine_report["strict_invariants"]["entity_type_overlaps"] == []
    run_strict_post_normalization_gates(result.payload, enforce_all_proteins_connected=True)


def test_an_entity_type_overlap_that_survives_closure_blocks_the_export() -> None:
    """The surviving case: kept alive as a complex component, never referenced.

    Closure cannot pick which declaration is real, and guessing would be the
    silent edit this module exists to avoid -- so it reports and refuses.
    """

    payload = _glutathione_payload()
    payload["entities"]["compounds"].append({"name": "GCLC"})
    payload["entities"]["proteins"].append(
        {"name": "GCLC", "species": "Homo sapiens", "mapped_ids": {"uniprot": "P48506"}}
    )
    payload["entities"]["protein_complexes"].append(
        {
            "name": "glutamate-cysteine ligase complex",
            "class": "protein_complex",
            "species": "Homo sapiens",
            "components": [{"name": "GCLC", "stoichiometry": 1}],
        }
    )
    payload["processes"]["reactions"][0]["enzymes"] = [
        {"entity": "glutamate-cysteine ligase complex"}
    ]

    result = quarantine_and_close(payload, strict_db=True)
    invariants = result.quarantine_report["strict_invariants"]

    assert invariants["entity_type_overlaps"] == [
        {"name": "GCLC", "buckets": "compounds+proteins"}
    ]
    assert invariants["ok"] is False
    assert result.ok is False


def test_the_input_payload_is_never_mutated() -> None:
    payload = _add_reaction(
        _glutathione_payload(),
        {"name": "peripheral", "inputs": ["nothing declared"], "outputs": ["glutathione"]},
    )
    before = deepcopy(payload)

    quarantine_and_close(payload, strict_db=True)

    assert payload == before


# ── 9. Artifacts ────────────────────────────────────────────────────────────


def test_all_four_artifacts_are_written_even_when_nothing_was_quarantined(tmp_path: Path) -> None:
    result = quarantine_and_close(_glutathione_payload(), strict_db=True)
    written = write_quarantine_artifacts(result, tmp_path)

    assert set(written) == {
        QUARANTINE_REPORT_FILENAME,
        REMOVED_ENTITY_REPORT_FILENAME,
        CLOSURE_REPORT_FILENAME,
        COVERAGE_REPORT_FILENAME,
    }
    for filename in written:
        document = json.loads((tmp_path / filename).read_text(encoding="utf-8"))
        # Every artifact is versioned. The quarantine report moved to 2 when it
        # grew the payload hash that binds a decision to one payload version, to
        # 3 when it grew the decision-input hash that binds it to one set of
        # rules as well, and to 4 (C-041a) when a coverage shortfall stopped
        # being a refusal and the report grew ``review_reasons`` and the
        # ``release`` classification beside the ``ok`` it can no longer be read
        # off. 3 -> 4 is additive: every schema-3 key kept its name and meaning.
        # MERGE RULE 4 BASELINE MOVE (C-056b): 4 -> 5, ``release`` grew
        # ``semantic_failed_checks``. Additive in the same sense; no schema-4 key
        # changed name, type or meaning and ``decision_id`` does not move.
        assert document["schema_version"] >= 1

    quarantine = json.loads((tmp_path / QUARANTINE_REPORT_FILENAME).read_text(encoding="utf-8"))
    # MERGE RULE 4 BASELINE MOVE (C-056c): 5 -> 6, ``release`` grew
    # ``semantic_check_evaluability`` (F-053 / D-054 section 8). Additive in the
    # same sense again; no schema-5 key changed name, type or meaning and
    # ``decision_id`` does not move -- ``canonical_decision_inputs`` (:246-278)
    # contains neither this key nor ``schema_version``.
    #
    # DISCLOSED: D-054 section 8 named only the pin at
    # ``test_strict_quarantine_release_seam.py:699``. There are TWO pins of this
    # one fact, and this is the second; C-056b moved this same line for its own
    # 4 -> 5 bump (``8eee549``). The record was incomplete, not wrong.
    assert quarantine["schema_version"] == 6
    assert quarantine["counts"][CORE_ACCEPTED] == 2
    assert quarantine["quarantined"] == []


def test_the_quarantine_report_counts_every_state(tmp_path: Path) -> None:
    payload = _cascading_payload()
    payload["processes"]["reactions"][0]["scope_membership"] = "out_of_scope"
    result = quarantine_and_close(payload, strict_db=True, confidence_floor=None)
    counts = result.quarantine_report["counts"]

    assert set(counts) == {
        CORE_ACCEPTED,
        AUXILIARY_ACCEPTED,
        QUARANTINED_UNMAPPED_ENTITY,
        QUARANTINED_OUT_OF_SCOPE,
        QUARANTINED_WEAK_EVIDENCE,
        QUARANTINED_BROKEN_REFERENCE,
        QUARANTINED_DISCONNECTED,
    }
    assert sum(counts.values()) == len(result.quarantine_report["admissions"])


# ── 10. Admission indices address the pre-prune lists ───────────────────────


def _admission(index: int) -> dict:
    return {"state": CORE_ACCEPTED, "bucket": "reactions", "index": index}


def _stale_index_payload() -> dict:
    """The committed reproduction fixture, loaded rather than copied: a reviewer
    re-runs ``evidence/repro_stale_index_synthetic.py`` and a copy could drift."""

    evidence = ROOT / "docs" / "pwml_recovery_sprint" / "evidence"
    sys.path.insert(0, str(evidence))
    try:
        import repro_stale_index_synthetic as fixture
    finally:
        sys.path.remove(str(evidence))
    return fixture.payload()


def test_a_compacted_process_list_does_not_manufacture_degree_zero_entities() -> None:
    """Five reactions: 0 and 1 quarantined, 2-4 admitted with EntC, EntD, EntF.

    ``_drop_quarantined_processes`` compacts the bucket to three rows while the
    admission records still say 2, 3 and 4. Index 2 then aliased onto the third
    survivor, 3 and 4 fell out of range and were skipped, so only EntF read as
    referenced: EntC and EntD were reported degree-zero and a complete, correctly
    declared pathway refused export over nothing.
    """

    result = quarantine_and_close(_stale_index_payload(), strict_db=True)
    proteins = [row["name"] for row in result.payload["entities"]["proteins"]]

    assert result.quarantine_report["strict_invariants"]["degree_zero_exports"] == []
    assert result.ok is True
    assert result.refusal_reasons == []
    assert proteins == ["EntC", "EntD", "EntF"]


def test_an_admission_index_that_does_not_resolve_raises_rather_than_being_skipped() -> None:
    """Skipping one deletes a surviving process's edges without saying so."""

    payload = _glutathione_payload()
    admissions = [_admission(7)]

    with pytest.raises(StrictQuarantineInvariantError) as excinfo:
        _degree_zero_exports(payload, admissions, process_snapshot=payload["processes"])

    # The message has to name the row, not just say an index did not resolve.
    assert "/processes/reactions/7" in str(excinfo.value)
    # The default is unchanged: the audited-safe callers must keep skipping,
    # because _revalidate_surviving_processes records the vanished row itself as
    # process_row_vanished_during_closure and raising there would lose that.
    assert _surviving_processes(payload, admissions) == []


def test_a_protein_no_admitted_process_reaches_is_still_reported_degree_zero() -> None:
    """The fix removes a FALSE degree-zero; the check itself still fires.

    Only reaction 0 is admitted, so reaction 1's enzyme is unreferenced, and
    ``SOD1`` is declared but reached by nothing. Both must still be reported.
    """

    payload = _glutathione_payload()
    payload["entities"]["proteins"].append(
        {"name": "SOD1", "species": "Homo sapiens", "mapped_ids": {"uniprot": "P00441"}}
    )

    found = _degree_zero_exports(
        payload, [_admission(0)], process_snapshot=payload["processes"]
    )

    assert [row["name"] for row in found] == ["glutathione synthetase", "SOD1"]
